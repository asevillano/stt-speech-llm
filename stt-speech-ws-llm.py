#pip install azure-cognitiveservices-speech websockets aiohttp
import asyncio
import websockets
import wave
import json
import uuid

from common_functions import (
    SPEECH_REGION, LANGUAGE, CHUNK_MS, TARGET_SAMPLE_RATE,
    SHOW_TIME, INTENTS, WS_URL,
    print_info, print_partial, parse_audio_file_arg,
    load_audio_16k_mono, create_aoai_clients, warmup_aoai, analyze_phrase,
    get_auth_token, create_speech_config_message,
    create_speech_context_message, create_audio_message,
    ResultsCsvWriter,
)

# Optional command-line arguments: audio file to process and an optional CSV file
# (--csv/-o) where the per-segment results are written. If no audio file is given,
# the default audio file is used; if no CSV is given, no CSV is written.
AUDIO_FILE, CSV_PATH = parse_audio_file_arg(
    "Real-time speech transcription with LLM analysis.", with_csv=True
)
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
print()


async def receive_messages(ws):
    """Task to receive messages from the server."""
    # Accumulates all completed phrases of the conversation
    conversation_phrases = []
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
                            print("-"*50)
                            print(f"[TRANSCRIPTION] {display_text}")

                            # Accumulate the phrase and analyze it with the LLM
                            if display_text.strip():
                                conversation_phrases.append(display_text)
                                full_conversation = " ".join(conversation_phrases)
                                # Run the (blocking) LLM call without blocking the event loop
                                analysis = await asyncio.to_thread(
                                    analyze_phrase, aoai_clients, display_text, full_conversation
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
                                    #print("-"*50)
                                    if csv_writer:
                                        csv_writer.write_row(display_text, analysis)
                                elif analysis and "error" in analysis:
                                    print(f"[ERROR] LLM analysis failed: {analysis['error']}")
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

    print("[STARTUP] Connecting to Azure Speech Service...")
    async with websockets.connect(WS_URL, additional_headers=headers, max_size=10**7) as ws:
        print_info("[INFO] Connected to the service")

        # 1. Send the configuration message
        print("[STARTUP] Sending configuration and speech context...")
        config_message = create_speech_config_message(request_id)
        await ws.send(config_message)
        print_info("[INFO] Configuration sent")

        # 1b. Send the speech.context message to enable Semantic segmentation
        context_message = create_speech_context_message(request_id)
        await ws.send(context_message)
        print_info("[INFO] Speech context (Semantic segmentation) sent")

        # Start the task to receive messages
        receive_task = asyncio.create_task(receive_messages(ws))

        # 2. Send audio in chunks (already converted to 16 kHz mono 16-bit PCM)
        print("[STARTUP] Starting streaming audio in Real-Time\n")
        # Bytes per chunk: 16-bit (2 bytes) * TARGET_SAMPLE_RATE * CHUNK_MS
        bytes_per_sample = 2
        bytes_per_chunk = int(TARGET_SAMPLE_RATE * (CHUNK_MS / 1000.0)) * bytes_per_sample

        print_info(f"[INFO] Sending audio from {AUDIO_FILE}")
        print_info(f"[INFO] Chunks of {bytes_per_chunk} bytes ({CHUNK_MS}ms)")

        chunk_count = 0
        for offset in range(0, len(AUDIO_PCM_16K), bytes_per_chunk):
            data = AUDIO_PCM_16K[offset:offset + bytes_per_chunk]
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

        print_info(f"[INFO] Full audio sent ({chunk_count} chunks)")
        
        # 3. Send an empty audio message to signal the end
        final_message = create_audio_message(request_id, b'')
        await ws.send(final_message)
        print_info("[INFO] End-of-stream message sent")
        
        # 4. Wait to receive all results
        await receive_task
        
        print_info("[INFO] Process completed")

if __name__ == "__main__":
    try:
        asyncio.run(stream_audio())
    finally:
        if csv_writer:
            csv_writer.close()
            print(f"[INFO] Results written to {CSV_PATH}")