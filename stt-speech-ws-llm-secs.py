#pip install azure-cognitiveservices-speech websockets aiohttp
#
# Variant of stt-speech-ws-llm.py that adds EXACT FIXED-TIME SEGMENTS for the LLM
# calls by slicing the audio CLIENT-SIDE (not relying on the service segmentation).
#
# Behavior:
#   * Without --interval (default): works exactly like stt-speech-ws-llm.py, i.e.
#     SEMANTIC segmentation is enabled and each final phrase returned by Azure
#     Speech is analyzed by the LLM immediately (single continuous turn).
#   * With --interval [SECONDS]: the audio is sliced into EXACT SECONDS-long
#     segments on the client. Each segment is sent as its OWN recognition turn
#     (a fresh X-RequestId) over the SAME WebSocket connection -- there are NO
#     reconnections, so no extra connection latency. After a segment is fully
#     transcribed, its text is sent to the LLM. Because we control exactly which
#     audio bytes go into each segment, the time boundaries are precise to the
#     audio sample, independent of the service's own silence/semantic segmentation.
#
# Trade-off: cutting raw audio at exact time boundaries may split a word that
# straddles a boundary (part in one segment, part in the next). This is inherent
# to exact time slicing; phrase/word-aligned cutting cannot be exact to the second.
#
# Usage examples:
#   python stt-speech-ws-llm-secs.py                  # immediate, per-phrase (as now)
#   python stt-speech-ws-llm-secs.py -i               # exact 10 s segments (default)
#   python stt-speech-ws-llm-secs.py --interval 5     # exact 5 s segments
#   python stt-speech-ws-llm-secs.py audio.wav -i 8   # custom audio + 8 s segments
import argparse
import asyncio
import websockets
import wave
import json
import uuid

from common_functions import (
    SPEECH_REGION, LANGUAGE, CHUNK_MS, TARGET_SAMPLE_RATE,
    SHOW_TIME, INTENTS, WS_URL, DEFAULT_AUDIO_FILE,
    print_info, print_partial,
    load_audio_16k_mono, create_aoai_clients, warmup_aoai, analyze_phrase,
    get_auth_token, create_speech_config_message,
    create_speech_context_message, create_audio_message,
    ResultsCsvWriter,
)

# Default time-window length (seconds) used when --interval is passed without a value.
DEFAULT_INTERVAL_SECONDS = 10.0


def parse_args():
    """Parses the optional audio file and the optional segment interval.

    --interval / -i is optional:
      * Omitted              -> None  -> immediate per-phrase analysis (legacy mode).
      * Given without value  -> DEFAULT_INTERVAL_SECONDS (10 s).
      * Given with a value   -> that many seconds.
    """
    parser = argparse.ArgumentParser(
        description="Real-time speech transcription with LLM analysis and optional "
                    "exact fixed-time audio segments for the LLM calls."
    )
    parser.add_argument(
        "audio_file",
        nargs="?",
        default=DEFAULT_AUDIO_FILE,
        help="Path to the audio file to process (default: %(default)s).",
    )
    parser.add_argument(
        "--interval", "-i",
        type=float,
        nargs="?",
        const=DEFAULT_INTERVAL_SECONDS,
        default=None,
        help="Slice the audio into exact N-second segments (client-side) and call "
             "the LLM once per segment. If the flag is given without a number, "
             "%(const)s s is used. If the flag is omitted, each phrase is analyzed "
             "immediately with semantic segmentation (legacy behavior).",
    )
    parser.add_argument(
        "--csv", "-o",
        dest="csv_path",
        default=None,
        help="Path to a CSV file where the per-segment results are written "
             "(columns: segment, transcription, intent, sentiment, summary, time). "
             "If omitted, no CSV is written.",
    )
    args = parser.parse_args()
    if args.interval is not None and args.interval <= 0:
        parser.error("--interval must be greater than 0 seconds.")
    return args.audio_file, args.interval, args.csv_path


AUDIO_FILE, INTERVAL_SECONDS, CSV_PATH = parse_args()
csv_writer = ResultsCsvWriter(CSV_PATH) if CSV_PATH else None

# Azure OpenAI configuration (Entra ID authentication)
print("[STARTUP] Initializing services...")
print("[STARTUP] Initializing Azure OpenAI client...")
aoai_clients = create_aoai_clients()
warmup_aoai(aoai_clients)
print("[STARTUP] Azure OpenAI client ready.")

# Check audio information
with wave.open(AUDIO_FILE, "rb") as wf:
    print(f"Audio information:")
    print(f"\tAudio file: {AUDIO_FILE}")
    print(f"\tChannels: {wf.getnchannels()}")
    print(f"\tSample rate: {wf.getframerate()} Hz")
    print(f"\tBits per sample: {wf.getsampwidth() * 8}")
    print(f"\tRequired format: 16000 Hz, mono, 16-bit PCM")
    print()

# Pre-load and convert the audio once so it can be streamed in chunks.
AUDIO_PCM_16K = load_audio_16k_mono(AUDIO_FILE)

# Prefined Intents
print(f"Possible intents to detect: {', '.join(INTENTS)}")
if INTERVAL_SECONDS is not None:
    print(f"[MODE] Exact {INTERVAL_SECONDS:g} s audio segments (client-side slicing, "
          f"one recognition turn per segment, single connection)")
else:
    print(f"[MODE] Immediate per-phrase analysis (semantic segmentation enabled)")
print()


async def run_analysis(latest_phrase, full_conversation):
    """Runs the (blocking) LLM call off the event loop and prints the result.

    Shared by both the immediate and the windowed modes."""
    analysis = await asyncio.to_thread(
        analyze_phrase, aoai_clients, latest_phrase, full_conversation
    )
    if analysis and "error" not in analysis:
        print(f"[INTENT] {analysis.get('intent', 'unknown')}")
        print(f"[SENTIMENT] {analysis.get('sentiment', 'unknown')}")
        print(f"[SUMMARY] {analysis.get('summary', '')}")
        if SHOW_TIME:
            client_s = analysis.get('elapsed_s', 0)
            server_s = analysis.get('server_elapsed_s')
            if server_s is not None:
                print(f"[TIME] Text model call: {server_s:.3f} s (server) | {client_s:.3f} s (client)")
            else:
                print(f"[TIME] Text model call: {client_s:.3f} s")
        if csv_writer:
            csv_writer.write_row(latest_phrase, analysis)
    elif analysis and "error" in analysis:
        print(f"[ERROR] LLM analysis failed: {analysis['error']}")


def _parse_message(message):
    """Splits a Speech WebSocket text frame into (path, parsed_json_or_None).
    Returns (None, None) for non-text frames or frames without a body."""
    if not isinstance(message, str):
        return None, None
    parts = message.split('\r\n\r\n', 1)
    if len(parts) <= 1:
        return None, None
    headers_text, body = parts[0], parts[1]
    path = None
    for line in headers_text.split('\r\n'):
        if line.startswith('Path:'):
            path = line.split(':', 1)[1]
            break
    return path, body


async def receive_messages(ws, state):
    """Continuous receiver used in IMMEDIATE mode (legacy behavior): each final
    phrase is analyzed by the LLM as soon as it arrives. Ends on turn.end."""
    try:
        async for message in ws:
            if not isinstance(message, str):
                print(f"[DEBUG] Binary message received: {len(message)} bytes")
                continue
            path, body = _parse_message(message)
            if path is None:
                continue

            if path == 'speech.phrase':
                result = json.loads(body)
                recognition_status = result.get('RecognitionStatus')
                if recognition_status == 'Success':
                    display_text = result.get('DisplayText', '')
                    if display_text.strip():
                        print("-"*50)
                        print(f"[TRANSCRIPTION] {display_text}")
                        state["conversation_phrases"].append(display_text)
                        full_conversation = " ".join(state["conversation_phrases"])
                        await run_analysis(display_text, full_conversation)
                elif recognition_status == 'NoMatch':
                    print_info("[INFO] No speech detected in this segment")
                else:
                    print_info(f"[INFO] Status: {recognition_status}")
            elif path == 'speech.hypothesis':
                result = json.loads(body)
                print_partial(f"[PARTIAL] {result.get('Text', '')}")
            elif path == 'turn.end':
                print_info("[INFO] Turn ended")
                return
    except websockets.ConnectionClosed:
        print_info("[INFO] Connection closed by the server")
    except Exception as e:
        print(f"[ERROR] Error receiving messages: {e}")


async def receive_segment(ws):
    """Receiver used in SEGMENT mode: reads the messages of the CURRENT recognition
    turn and returns the list of final phrase texts once turn.end is received. The
    audio fed to this turn is an exact client-side slice, so its time boundaries are
    precise regardless of the service's internal segmentation."""
    phrases = []
    try:
        async for message in ws:
            if not isinstance(message, str):
                print(f"[DEBUG] Binary message received: {len(message)} bytes")
                continue
            path, body = _parse_message(message)
            if path is None:
                continue

            if path == 'speech.phrase':
                result = json.loads(body)
                if result.get('RecognitionStatus') == 'Success':
                    display_text = result.get('DisplayText', '')
                    if display_text.strip():
                        phrases.append(display_text)
                elif result.get('RecognitionStatus') == 'NoMatch':
                    print_info("[INFO] No speech detected in this segment")
            elif path == 'speech.hypothesis':
                result = json.loads(body)
                print_partial(f"[PARTIAL] {result.get('Text', '')}")
            elif path == 'turn.end':
                # End of this segment's turn: return everything collected.
                return phrases
    except websockets.ConnectionClosed:
        print_info("[INFO] Connection closed by the server")
    return phrases


async def stream_audio():
    print("[STARTUP] Getting authentication token...")
    token = await get_auth_token()
    request_id = str(uuid.uuid4()).replace('-', '')

    print_info(f"[INFO] Connecting to Azure Speech Service...")
    print_info(f"[INFO] Request ID: {request_id}")

    # Entra ID authentication via Authorization header
    headers = {
        "Authorization": f"Bearer {token}"
    }

    # Shared state: the running conversation used as LLM context.
    state = {
        "conversation_phrases": [],  # all phrases/segments so far (LLM context)
    }

    bytes_per_sample = 2
    bytes_per_chunk = int(TARGET_SAMPLE_RATE * (CHUNK_MS / 1000.0)) * bytes_per_sample

    print("[STARTUP] Connecting to Azure Speech Service...")
    async with websockets.connect(WS_URL, additional_headers=headers, max_size=10**7) as ws:
        print_info("[INFO] Connected to the service")

        # 1. Send the configuration message (connection-level, sent once for the
        #    whole connection regardless of how many turns we run on it).
        print("[STARTUP] Sending configuration...")
        await ws.send(create_speech_config_message(request_id))
        print_info("[INFO] Configuration sent")

        if INTERVAL_SECONDS is None:
            # ----------------------- IMMEDIATE MODE (legacy) -----------------------
            # Semantic segmentation + one continuous recognition turn.
            await ws.send(create_speech_context_message(request_id))
            print_info("[INFO] Speech context (Semantic segmentation) sent")

            receive_task = asyncio.create_task(receive_messages(ws, state))

            print("[STARTUP] Starting streaming audio in Real-Time\n")
            print_info(f"[INFO] Sending audio from {AUDIO_FILE}")
            chunk_count = 0
            for offset in range(0, len(AUDIO_PCM_16K), bytes_per_chunk):
                data = AUDIO_PCM_16K[offset:offset + bytes_per_chunk]
                if not data:
                    break
                await ws.send(create_audio_message(request_id, data))
                chunk_count += 1
                if chunk_count % 10 == 0:
                    print_info(f"[INFO] Sent {chunk_count} chunks...")
                await asyncio.sleep(CHUNK_MS / 1000.0)  # simulate real time
            print_info(f"[INFO] Full audio sent ({chunk_count} chunks)")

            await ws.send(create_audio_message(request_id, b''))  # end of stream
            print_info("[INFO] End-of-stream message sent")
            await receive_task
        else:
            # ----------------- SEGMENT MODE (exact X-second slices) ----------------
            # Slice the audio into exact INTERVAL_SECONDS segments and send each one
            # as its OWN recognition turn (fresh X-RequestId) over the SAME WebSocket
            # connection -- no reconnections. After each segment is fully transcribed
            # (turn.end), its text is analyzed by the LLM before the next segment.
            print("[STARTUP] Starting streaming audio in Real-Time\n")
            print_info(f"[INFO] Sending audio from {AUDIO_FILE}")
            bytes_per_segment = int(TARGET_SAMPLE_RATE * INTERVAL_SECONDS) * bytes_per_sample
            total_bytes = len(AUDIO_PCM_16K)
            seconds_per_byte = 1.0 / (TARGET_SAMPLE_RATE * bytes_per_sample)

            seg_count = 0
            for seg_start in range(0, total_bytes, bytes_per_segment):
                seg_end = min(seg_start + bytes_per_segment, total_bytes)
                segment_request_id = str(uuid.uuid4()).replace('-', '')

                # Receive this turn's phrases concurrently while streaming its audio.
                recv_task = asyncio.create_task(receive_segment(ws))

                # Stream this segment's audio chunks, paced for real time.
                for offset in range(seg_start, seg_end, bytes_per_chunk):
                    data = AUDIO_PCM_16K[offset:min(offset + bytes_per_chunk, seg_end)]
                    if not data:
                        break
                    await ws.send(create_audio_message(segment_request_id, data))
                    await asyncio.sleep(CHUNK_MS / 1000.0)  # simulate real time

                # End this turn so the service finalizes exactly this segment.
                await ws.send(create_audio_message(segment_request_id, b''))
                phrases = await recv_task

                start_s = seg_start * seconds_per_byte
                end_s = seg_end * seconds_per_byte
                segment_text = " ".join(phrases).strip()
                print("-" * 50)
                print(f"[SEGMENT {start_s:g}-{end_s:g}s] {segment_text or '(no speech)'}")

                if segment_text:
                    state["conversation_phrases"].append(segment_text)
                    full_conversation = " ".join(state["conversation_phrases"])
                    await run_analysis(segment_text, full_conversation)
                seg_count += 1

            print_info(f"[INFO] Full audio sent ({seg_count} segment(s))")

        print_info("[INFO] Process completed")


if __name__ == "__main__":
    try:
        asyncio.run(stream_audio())
    finally:
        if csv_writer:
            csv_writer.close()
            print(f"[INFO] Results written to {CSV_PATH}")
