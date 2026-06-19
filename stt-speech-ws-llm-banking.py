#pip install azure-cognitiveservices-speech websockets aiohttp
#
# Banking call-center variant of stt-speech-ws-llm.py.
#
# What is different from stt-speech-ws-llm.py:
#   * The intent, sentiment and summary prompts are the banking call-center prompts
#     loaded from the JSON payloads in ./banking_prompts/ (intent.json,
#     sentiment.json, summary.json). The system
#     message(s) of each payload are used as-is; only the user message is replaced
#     with the live conversation transcript.
#   * For every final phrase, THREE calls (intent, sentiment, summary) are ALWAYS
#     made IN PARALLEL, even when the configured endpoints/models are identical.
#   * Each task has its own output contract (intent -> JSON with an "Intent" field,
#     sentiment -> a single word, summary -> a structured multi-line text), so the
#     responses are parsed per task instead of expecting one combined JSON object.
import asyncio
import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import websockets

from common_functions import (
    SPEECH_REGION, LANGUAGE, CHUNK_MS, TARGET_SAMPLE_RATE,
    SHOW_TIME, SHOW_DEBUG, WS_URL,
    SERVERLESS_MODELS,
    AOAI_INTENT_ENDPOINT, AOAI_INTENT_MODEL,
    AOAI_SUMMARY_ENDPOINT, AOAI_SUMMARY_MODEL,
    print_info, print_partial, parse_audio_file_arg,
    load_audio_16k_mono, create_aoai_clients, warmup_aoai,
    get_auth_token, create_speech_config_message,
    create_speech_context_message, create_audio_message,
    print_audio_info, iter_audio_connection_slices,
    ResultsCsvWriter,
    _extract_json, _server_seconds_from_headers,
)

# Optional command-line arguments: audio file to process and an optional CSV file
# (--csv/-o) where the per-segment results are written.
AUDIO_FILE, CSV_PATH = parse_audio_file_arg(
    "Real-time speech transcription with banking intent/sentiment/summary analysis.",
    with_csv=True,
)
csv_writer = ResultsCsvWriter(CSV_PATH) if CSV_PATH else None

# --------------------------------------------------------------------------- #
# Banking prompts (loaded from the JSON payloads next to this script)
# --------------------------------------------------------------------------- #
BANKING_PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "banking_prompts")

def _load_system_prompt(filename):
    """Loads a payload JSON and returns its system prompt: the concatenation of all
    its 'system' messages (the intent payload has two: the instructions and the
    INTENT_LIBRARY). The payload's 'user' message is ignored; the live transcript
    is supplied at call time instead."""
    path = os.path.join(BANKING_PROMPTS_DIR, filename)
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    system_parts = [
        m.get("content", "")
        for m in payload.get("messages", [])
        if m.get("role") == "system"
    ]
    return "\n\n".join(p for p in system_parts if p).strip()


INTENT_SYSTEM_PROMPT = _load_system_prompt("intent.json")
SENTIMENT_SYSTEM_PROMPT = _load_system_prompt("sentiment.json")
SUMMARY_SYSTEM_PROMPT = _load_system_prompt("summary.json")


def _build_user_prompt(full_conversation):
    """Builds the user message in the same shape used by the banking payload samples:
    the conversation transcript wrapped in an <transcript> block."""
    return (
        "INPUT\n"
        "<transcript>\n"
        f"''' {full_conversation} '''\n"
        "</transcript>"
    )


# --------------------------------------------------------------------------- #
# Model call (raw text) and three-way parallel analysis
# --------------------------------------------------------------------------- #
def _call_model_raw(client, model, system_prompt, user_prompt):
    """Calls the model once and returns (raw_text, elapsed_s, server_s).

    Unlike common_functions._call_model, the raw text is NOT parsed as JSON here,
    because the banking sentiment and summary tasks return plain text rather than JSON.
    Uses the Azure AI Inference chat completions API when SERVERLESS_MODELS is True,
    otherwise the Azure OpenAI Responses API."""
    if SHOW_DEBUG:
        print(f"[DEBUG] System prompt:\n{system_prompt}\nUser prompt:\n{user_prompt}\n")
        print('-' * 50)
    start = time.perf_counter()
    server_s = None
    if SERVERLESS_MODELS:
        from azure.ai.inference.models import SystemMessage, UserMessage
        response = client.complete(
            model=model,
            messages=[
                SystemMessage(content=system_prompt),
                UserMessage(content=user_prompt),
            ],
            temperature=0.0,
            max_tokens=512,
        )
        raw = response.choices[0].message.content.strip()
    else:
        try:
            raw_response = client.responses.with_raw_response.create(
                model=model,
                instructions=system_prompt,
                input=user_prompt,
                temperature=0.0,
                max_output_tokens=512,
            )
            server_s = _server_seconds_from_headers(raw_response.headers)
            response = raw_response.parse()
        except AttributeError:
            response = client.responses.create(
                model=model,
                instructions=system_prompt,
                input=user_prompt,
                temperature=0.0,
                max_output_tokens=512,
            )
        raw = response.output_text.strip()
    elapsed_s = time.perf_counter() - start
    return raw, elapsed_s, server_s


def _parse_intent(raw):
    """Extracts the Intent string from the intent task's JSON response. Falls back
    to the raw text if it is not valid JSON."""
    try:
        data = _extract_json(raw)
        return (data.get("Intent") or "Unknown_Unknown_Unknown").strip()
    except Exception:
        return raw.strip()


def _parse_sentiment(raw):
    """The sentiment task returns a single word; keep the first non-empty token."""
    text = raw.strip().strip('"').strip()
    return text.split()[0] if text else "Neutral"


def analyze_phrase_banking(clients, full_conversation):
    """Runs the banking intent, sentiment and summary prompts ALWAYS as three parallel
    calls (even when the configured endpoints/models are identical).

    Returns a dict with 'intent', 'sentiment', 'summary' and timing keys, or
    {'error': ...} on failure."""
    user_prompt = _build_user_prompt(full_conversation)
    # Route the three tasks to the configured clients/models. Intent and sentiment
    # use the intent endpoint/model; summary uses the summary endpoint/model. When
    # the configuration points them all to the same place, three independent calls
    # are still issued in parallel.
    tasks = [
        ("intent", clients["intent"], AOAI_INTENT_MODEL, INTENT_SYSTEM_PROMPT, AOAI_INTENT_ENDPOINT),
        ("sentiment", clients["intent"], AOAI_INTENT_MODEL, SENTIMENT_SYSTEM_PROMPT, AOAI_INTENT_ENDPOINT),
        ("summary", clients["summary"], AOAI_SUMMARY_MODEL, SUMMARY_SYSTEM_PROMPT, AOAI_SUMMARY_ENDPOINT),
    ]
    if SHOW_DEBUG:
        print(f"[DEBUG] Calling intent '{AOAI_INTENT_MODEL}' @ {AOAI_INTENT_ENDPOINT}, "
              f"sentiment '{AOAI_INTENT_MODEL}' @ {AOAI_INTENT_ENDPOINT} and "
              f"summary '{AOAI_SUMMARY_MODEL}' @ {AOAI_SUMMARY_ENDPOINT} in parallel")
    try:
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                name: executor.submit(_call_model_raw, client, model, system_prompt, user_prompt)
                for name, client, model, system_prompt, _ in tasks
            }
            intent_raw, intent_elapsed, intent_server = futures["intent"].result()
            sentiment_raw, sentiment_elapsed, sentiment_server = futures["sentiment"].result()
            summary_raw, summary_elapsed, summary_server = futures["summary"].result()
        elapsed_s = time.perf_counter() - start

        # Server-side time is the slowest of the three concurrent calls when all
        # services report it; otherwise None.
        servers = [intent_server, sentiment_server, summary_server]
        server_s = max(servers) if all(s is not None for s in servers) else None

        return {
            "intent": _parse_intent(intent_raw),
            "sentiment": _parse_sentiment(sentiment_raw),
            "summary": summary_raw.strip(),
            "elapsed_s": elapsed_s,
            "server_elapsed_s": server_s,
            "intent_elapsed_s": intent_elapsed,
            "sentiment_elapsed_s": sentiment_elapsed,
            "summary_elapsed_s": summary_elapsed,
            "intent_server_elapsed_s": intent_server,
            "sentiment_server_elapsed_s": sentiment_server,
            "summary_server_elapsed_s": summary_server,
        }
    except Exception as ex:
        return {"error": str(ex)}


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

print("[MODE] Banking prompts | always three parallel calls (intent + sentiment + summary)")
print()


async def receive_messages(ws, conversation_phrases, analysis_queue):
    """Task to receive messages from the server.

    'conversation_phrases' is owned by the caller and shared across reconnections,
    so the cumulative transcript (and therefore the LLM context) survives when a
    long audio is streamed over several back-to-back WebSocket connections.

    The (multi-second) banking analysis is NOT run here: each final phrase is handed
    off to a background worker via 'analysis_queue' so this loop keeps draining the
    socket. Otherwise the blocking LLM calls would stall the receive loop, the
    websockets keepalive ping would time out, and the connection would be closed
    with 1011 (internal error) keepalive ping timeout."""
    try:
        async for message in ws:
            if isinstance(message, str):
                # Parse the text message
                lines = message.split('\r\n\r\n', 1)
                if len(lines) > 1:
                    headers_text = lines[0]
                    body = lines[1]

                    # Extract the Path from the header
                    path = None
                    for line in headers_text.split('\r\n'):
                        if line.startswith('Path:'):
                            path = line.split(':', 1)[1]
                            break

                    if path == 'speech.phrase':
                        result = json.loads(body)
                        recognition_status = result.get('RecognitionStatus')

                        if recognition_status == 'Success':
                            display_text = result.get('DisplayText', '')

                            # Accumulate the phrase and hand the analysis off to the
                            # background worker, keeping this receive loop free to
                            # drain the socket (prevents the 1011 keepalive timeout).
                            if display_text.strip():
                                conversation_phrases.append(display_text)
                                full_conversation = " ".join(conversation_phrases)
                                await analysis_queue.put((display_text, full_conversation))
                        elif recognition_status == 'NoMatch':
                            print_info(f"[INFO] No speech detected in this segment")
                        else:
                            print_info(f"[INFO] Status: {recognition_status}")

                    elif path == 'speech.hypothesis':
                        result = json.loads(body)
                        text = result.get('Text', '')
                        print_partial(f"[PARTIAL] {text}")

                    elif path == 'turn.end':
                        print_info("[INFO] Turn ended")
                        return

            else:
                # Binary message (not expected in response)
                print(f"[DEBUG] Binary message received: {len(message)} bytes")

    except websockets.ConnectionClosed:
        print_info("[INFO] Connection closed by the server")
    except Exception as e:
        print(f"[ERROR] Error receiving messages: {e}")


async def analysis_worker(analysis_queue):
    """Background consumer that runs the banking analysis serially, one phrase at a
    time, in arrival order. It lives for the whole stream (across all connections)
    so the LLM context, console output and CSV rows stay ordered even though the
    receive loop never waits for the (multi-second) LLM calls.

    A None item is the sentinel that tells the worker to stop once the queue has
    been fully drained."""
    while True:
        item = await analysis_queue.get()
        try:
            if item is None:
                return
            display_text, full_conversation = item
            print("-" * 50)
            print(f"[TRANSCRIPTION] {display_text}")
            # Run the (blocking) LLM calls without blocking the event loop.
            analysis = await asyncio.to_thread(
                analyze_phrase_banking, aoai_clients, full_conversation
            )
            if analysis and "error" not in analysis:
                print(f"[INTENT] {analysis.get('intent', 'unknown')}")
                print(f"[SENTIMENT] {analysis.get('sentiment', 'unknown')}")
                print(f"[SUMMARY]\n{analysis.get('summary', '')}")
                if SHOW_TIME:
                    client_s = analysis.get('elapsed_s', 0)
                    server_s = analysis.get('server_elapsed_s')
                    if server_s is not None:
                        print(f"[TIME] Text model call: {server_s:.3f} s (server) | {client_s:.3f} s (client)")
                    else:
                        print(f"[TIME] Text model call: {client_s:.3f} s")
                if csv_writer:
                    csv_writer.write_row(display_text, analysis)
            elif analysis and "error" in analysis:
                print(f"[ERROR] LLM analysis failed: {analysis['error']}")
        finally:
            analysis_queue.task_done()


async def stream_audio_connection(audio_bytes, conversation_phrases, analysis_queue, connection_index, total_connections):
    """Streams one slice of audio over a single WebSocket connection and runs the
    banking analysis on every final phrase.

    Azure Speech closes a WebSocket after ~635 s (10 min 35 s) of total duration,
    so long audio is split across several connections (see stream_audio). Each
    connection gets a fresh request_id and re-sends the config + speech.context
    messages; the cumulative transcript is preserved by the shared
    'conversation_phrases' list."""
    token = await get_auth_token()
    request_id = str(uuid.uuid4()).replace('-', '')

    if total_connections > 1:
        print_info(f"[INFO] Connection {connection_index}/{total_connections} | Request ID: {request_id}")
    else:
        print_info(f"[INFO] Connecting to Azure Speech Service...")
        print_info(f"[INFO] Request ID: {request_id}")

    # Entra ID authentication via Authorization header
    headers = {
        "Authorization": f"Bearer {token}"
    }

    async with websockets.connect(WS_URL, additional_headers=headers, max_size=10**7) as ws:
        print_info("[INFO] Connected to the service")

        # 1. Send the configuration message
        config_message = create_speech_config_message(request_id)
        await ws.send(config_message)
        print_info("[INFO] Configuration sent")

        # 1b. Send the speech.context message to enable Semantic segmentation
        context_message = create_speech_context_message(request_id)
        await ws.send(context_message)
        print_info("[INFO] Speech context (Semantic segmentation) sent")

        # Start the task to receive messages
        receive_task = asyncio.create_task(receive_messages(ws, conversation_phrases, analysis_queue))

        # 2. Send audio in chunks (already converted to 16 kHz mono 16-bit PCM)
        # Bytes per chunk: 16-bit (2 bytes) * TARGET_SAMPLE_RATE * CHUNK_MS
        bytes_per_sample = 2
        bytes_per_chunk = int(TARGET_SAMPLE_RATE * (CHUNK_MS / 1000.0)) * bytes_per_sample

        print_info(f"[INFO] Chunks of {bytes_per_chunk} bytes ({CHUNK_MS}ms)")

        chunk_count = 0
        for offset in range(0, len(audio_bytes), bytes_per_chunk):
            data = audio_bytes[offset:offset + bytes_per_chunk]
            if not data:
                break

            # Create audio message with binary header
            audio_message = create_audio_message(request_id, data)
            await ws.send(audio_message)

            chunk_count += 1
            if chunk_count % 10 == 0:
                print_info(f"[INFO] Sent {chunk_count} chunks...")

            # Simulate real-time streaming
            await asyncio.sleep(CHUNK_MS / 1000.0)

        print_info(f"[INFO] Audio slice sent ({chunk_count} chunks)")

        # 3. Send an empty audio message to signal the end of this turn
        final_message = create_audio_message(request_id, b'')
        await ws.send(final_message)
        print_info("[INFO] End-of-stream message sent")

        # 4. Wait to receive all results for this connection
        await receive_task


async def stream_audio():
    print("[STARTUP] Getting authentication token...")
    print("[STARTUP] Connecting to Azure Speech Service...")
    print("[STARTUP] Sending configuration and speech context...")

    # Conversation transcript, shared across all connections so the LLM context is
    # preserved when a long audio is split over several WebSocket connections.
    conversation_phrases = []

    # Single analysis queue + worker for the whole stream. The receive loop only
    # enqueues phrases (never blocks on the LLM), and this worker runs the banking
    # analysis serially in arrival order, so the console output and CSV rows stay
    # ordered and the websockets keepalive never times out during the LLM calls.
    analysis_queue = asyncio.Queue()
    worker_task = asyncio.create_task(analysis_worker(analysis_queue))

    # Azure Speech closes a connection after ~635 s of total duration, so the audio
    # is split into slices of at most MAX_CONNECTION_AUDIO_SECONDS each, every slice
    # streamed over its own back-to-back connection.
    slices = list(iter_audio_connection_slices(AUDIO_PCM_16K))
    total_connections = len(slices)

    print("[STARTUP] Starting streaming audio in Real-Time\n")
    if total_connections > 1:
        total_seconds = len(AUDIO_PCM_16K) / (TARGET_SAMPLE_RATE * 2)
        print_info(
            f"[INFO] Audio is {total_seconds:.0f}s long; it will be streamed over "
            f"{total_connections} connections to stay within the ~635s WebSocket limit."
        )

    for connection_index, _total, audio_slice in slices:
        await stream_audio_connection(
            audio_slice, conversation_phrases, analysis_queue, connection_index, total_connections
        )

    # All audio streamed: wait for the backlog of queued analyses to drain, then
    # stop the worker via the None sentinel.
    await analysis_queue.join()
    await analysis_queue.put(None)
    await worker_task

    print_info("[INFO] Process completed")


if __name__ == "__main__":
    try:
        asyncio.run(stream_audio())
    finally:
        if csv_writer:
            csv_writer.close()
            print(f"[INFO] Results written to {CSV_PATH}")
