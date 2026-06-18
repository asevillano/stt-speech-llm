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
import json
import time
import uuid

from common_functions import (
    SPEECH_REGION, LANGUAGE, CHUNK_MS, TARGET_SAMPLE_RATE,
    SHOW_TIME, INTENTS, WS_URL, DEFAULT_AUDIO_FILE,
    MAX_CONNECTION_AUDIO_SECONDS,
    print_info, print_partial,
    load_audio_16k_mono, create_aoai_clients, warmup_aoai, analyze_phrase,
    get_auth_token, create_speech_config_message,
    create_speech_context_message, create_audio_message,
    print_audio_info, iter_audio_connection_slices,
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
print_audio_info(AUDIO_FILE)

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


async def receive_messages(ws, state, analysis_queue):
    """Continuous receiver used in IMMEDIATE mode (legacy behavior): each final
    phrase is handed off to a background worker for LLM analysis. Ends on turn.end.

    The (multi-second) LLM analysis is NOT run here: phrases are enqueued so this
    loop keeps draining the socket. Otherwise the blocking LLM call would stall the
    receive loop, the websockets keepalive ping would time out, and the connection
    would be closed with 1011 (internal error) keepalive ping timeout."""
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
                        state["conversation_phrases"].append(display_text)
                        full_conversation = " ".join(state["conversation_phrases"])
                        await analysis_queue.put((display_text, full_conversation))
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


async def analysis_worker(analysis_queue):
    """Background consumer for IMMEDIATE mode: runs the LLM analysis serially, one
    phrase at a time, in arrival order, so the console output and CSV rows stay
    ordered even though the receive loop never waits for the (multi-second) LLM
    call. A None item is the sentinel that stops the worker once drained."""
    while True:
        item = await analysis_queue.get()
        try:
            if item is None:
                return
            latest_phrase, full_conversation = item
            print("-" * 50)
            print(f"[TRANSCRIPTION] {latest_phrase}")
            await run_analysis(latest_phrase, full_conversation)
        finally:
            analysis_queue.task_done()


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


async def stream_immediate_connection(audio_bytes, state, analysis_queue, connection_index, total_connections):
    """IMMEDIATE mode: streams one slice of audio over its own WebSocket connection
    using semantic segmentation. 'state' is shared across reconnections so the LLM
    keeps the full conversation context."""
    token = await get_auth_token()
    request_id = str(uuid.uuid4()).replace('-', '')

    if total_connections > 1:
        print_info(f"[INFO] Connection {connection_index}/{total_connections}")
    print_info(f"[INFO] Connecting to Azure Speech Service...")
    print_info(f"[INFO] Request ID: {request_id}")

    headers = {"Authorization": f"Bearer {token}"}
    bytes_per_sample = 2
    bytes_per_chunk = int(TARGET_SAMPLE_RATE * (CHUNK_MS / 1000.0)) * bytes_per_sample

    print("[STARTUP] Connecting to Azure Speech Service...")
    async with websockets.connect(WS_URL, additional_headers=headers, max_size=10**7) as ws:
        print_info("[INFO] Connected to the service")

        print("[STARTUP] Sending configuration and speech context...")
        await ws.send(create_speech_config_message(request_id))
        await ws.send(create_speech_context_message(request_id))
        print_info("[INFO] Speech context (Semantic segmentation) sent")

        receive_task = asyncio.create_task(receive_messages(ws, state, analysis_queue))

        print_info(f"[INFO] Sending audio from {AUDIO_FILE}")
        chunk_count = 0
        for offset in range(0, len(audio_bytes), bytes_per_chunk):
            data = audio_bytes[offset:offset + bytes_per_chunk]
            if not data:
                break
            await ws.send(create_audio_message(request_id, data))
            chunk_count += 1
            if chunk_count % 10 == 0:
                print_info(f"[INFO] Sent {chunk_count} chunks...")
            await asyncio.sleep(CHUNK_MS / 1000.0)  # simulate real time
        print_info(f"[INFO] Audio slice sent ({chunk_count} chunks)")

        await ws.send(create_audio_message(request_id, b''))  # end of stream
        print_info("[INFO] End-of-stream message sent")
        await receive_task


async def stream_segments(state):
    """SEGMENT mode: slices the audio into exact INTERVAL_SECONDS turns. To respect
    the ~635s WebSocket limit, whole segments are grouped into back-to-back
    connections (the cut always falls on a segment boundary, so no segment is split
    across connections). 'state' is shared so the LLM context spans all segments.

    Unlike the continuous modes, here the LLM analysis runs synchronously BETWEEN
    segments, so the connection's wall-clock grows by audio time PLUS analysis
    time. We therefore track the real elapsed wall-clock per connection and open a
    fresh connection before it would exceed the budget, and we also retry a segment
    over a new connection if the current one drops (duration limit or 1012 service
    restart)."""
    bytes_per_sample = 2
    bytes_per_chunk = int(TARGET_SAMPLE_RATE * (CHUNK_MS / 1000.0)) * bytes_per_sample
    bytes_per_segment = int(TARGET_SAMPLE_RATE * INTERVAL_SECONDS) * bytes_per_sample
    total_bytes = len(AUDIO_PCM_16K)
    seconds_per_byte = 1.0 / (TARGET_SAMPLE_RATE * bytes_per_sample)

    seg_starts = list(range(0, total_bytes, bytes_per_segment))
    print_info(f"[INFO] Sending audio from {AUDIO_FILE}")

    # Current connection and the wall-clock time (monotonic) at which it was opened.
    conn = {"ws": None, "start": 0.0, "index": 0}

    async def open_connection():
        """Opens a fresh Speech WebSocket connection and sends the connection-level
        configuration. Records the wall-clock start so we can respect the budget."""
        token = await get_auth_token()
        conn_request_id = str(uuid.uuid4()).replace('-', '')
        headers = {"Authorization": f"Bearer {token}"}
        conn["index"] += 1
        print(f"[STARTUP] Connecting to Azure Speech Service... (connection {conn['index']})")
        conn["ws"] = await websockets.connect(WS_URL, additional_headers=headers, max_size=10**7)
        await conn["ws"].send(create_speech_config_message(conn_request_id))
        print_info("[INFO] Configuration sent")
        conn["start"] = time.monotonic()

    async def close_connection():
        if conn["ws"] is not None:
            try:
                await conn["ws"].close()
            except Exception:
                pass
            conn["ws"] = None

    seg_count = 0
    try:
        for seg_start in seg_starts:
            seg_end = min(seg_start + bytes_per_segment, total_bytes)
            start_s = seg_start * seconds_per_byte
            end_s = seg_end * seconds_per_byte

            phrases = None
            attempts = 0
            while phrases is None:
                attempts += 1
                # (Re)connect if there is no connection yet, or if streaming this
                # segment would push the connection's wall-clock past the budget.
                if (conn["ws"] is None or
                        (time.monotonic() - conn["start"] + INTERVAL_SECONDS) > MAX_CONNECTION_AUDIO_SECONDS):
                    await close_connection()
                    await open_connection()

                segment_request_id = str(uuid.uuid4()).replace('-', '')
                recv_task = asyncio.create_task(receive_segment(conn["ws"]))
                try:
                    # Stream this segment's audio chunks, paced for real time.
                    for offset in range(seg_start, seg_end, bytes_per_chunk):
                        data = AUDIO_PCM_16K[offset:min(offset + bytes_per_chunk, seg_end)]
                        if not data:
                            break
                        await conn["ws"].send(create_audio_message(segment_request_id, data))
                        await asyncio.sleep(CHUNK_MS / 1000.0)  # simulate real time

                    # End this turn so the service finalizes exactly this segment.
                    await conn["ws"].send(create_audio_message(segment_request_id, b''))
                    phrases = await recv_task
                except websockets.ConnectionClosed:
                    # Connection dropped mid-segment (duration limit or 1012 service
                    # restart). Reconnect and retry this same segment from scratch.
                    recv_task.cancel()
                    await close_connection()
                    if attempts >= 5:
                        print(f"[ERROR] Giving up on segment {start_s:g}-{end_s:g}s "
                              f"after {attempts} attempts")
                        phrases = []
                    else:
                        print_info(f"[INFO] Connection closed during segment "
                                   f"{start_s:g}-{end_s:g}s; reconnecting and retrying...")

            segment_text = " ".join(phrases).strip()
            print("-" * 50)
            print(f"[SEGMENT {start_s:g}-{end_s:g}s] {segment_text or '(no speech)'}")

            if segment_text:
                state["conversation_phrases"].append(segment_text)
                full_conversation = " ".join(state["conversation_phrases"])
                await run_analysis(segment_text, full_conversation)
            seg_count += 1
    finally:
        await close_connection()

    print_info(f"[INFO] Full audio sent ({seg_count} segment(s))")


async def stream_audio():
    # Shared state: the running conversation used as LLM context, preserved across
    # reconnections.
    state = {
        "conversation_phrases": [],  # all phrases/segments so far (LLM context)
    }

    print("[STARTUP] Starting streaming audio in Real-Time\n")
    if INTERVAL_SECONDS is None:
        # ----------------------- IMMEDIATE MODE (legacy) -----------------------
        # Semantic segmentation; long audio is split into back-to-back connections.
        # A single analysis queue + worker runs the LLM serially in arrival order,
        # so the receive loop never blocks on the LLM and the websockets keepalive
        # never times out (no 1011) during the multi-second analysis.
        analysis_queue = asyncio.Queue()
        worker_task = asyncio.create_task(analysis_worker(analysis_queue))
        slices = list(iter_audio_connection_slices(AUDIO_PCM_16K))
        total_connections = len(slices)
        if total_connections > 1:
            total_seconds = len(AUDIO_PCM_16K) / (TARGET_SAMPLE_RATE * 2)
            print_info(
                f"[INFO] Audio is {total_seconds:.0f}s long; it will be streamed over "
                f"{total_connections} connections to stay within the ~635s WebSocket limit."
            )
        for connection_index, _total, audio_slice in slices:
            await stream_immediate_connection(
                audio_slice, state, analysis_queue, connection_index, total_connections
            )
        # All audio streamed: drain the backlog, then stop the worker.
        await analysis_queue.join()
        await analysis_queue.put(None)
        await worker_task
    else:
        # ----------------- SEGMENT MODE (exact X-second slices) ----------------
        await stream_segments(state)

    print_info("[INFO] Process completed")


if __name__ == "__main__":
    try:
        asyncio.run(stream_audio())
    finally:
        if csv_writer:
            csv_writer.close()
            print(f"[INFO] Results written to {CSV_PATH}")
