# pip install azure-cognitiveservices-speech websockets aiohttp streamlit openai azure-identity python-dotenv
# Run with:  streamlit run stt-speech-ws-llm_app.py
#
# Design note (latency):
# The whole transcription + LLM pipeline (WebSocket streaming, speech recognition
# and Azure OpenAI intent/summary calls) runs in a DEDICATED BACKGROUND THREAD.
# Streamlit only READS finished results from a thread-safe queue and renders them.
# To keep the UI off the worker's critical path, the live results panel is an
# st.fragment that auto-refreshes on its own (run_every) every 0.5 s WITHOUT
# re-running the whole script. This avoids the previous full-script rerun loop,
# which re-rendered the entire page (sidebar, file listing, all phrases) every
# refresh and stole the GIL from the worker, inflating the measured LLM latency.
# Timing is reported two ways: 'server_elapsed_s' (server-side processing time
# from response headers, immune to client GIL contention) when available, and
# 'elapsed_s' (client wall-clock) as a fallback. Both are measured in the worker.

import asyncio
import websockets
import os
import json
import uuid
import queue
import threading

import streamlit as st

from common_functions import (
    SPEECH_REGION, LANGUAGE, CHUNK_MS, TARGET_SAMPLE_RATE,
    SHOW_INFO, SHOW_PARTIAL, SHOW_TIME, INTENTS, INTENT_DESCRIPTIONS, WS_URL,
    AOAI_INTENT_MODEL, AOAI_SUMMARY_MODEL, USE_SEPARATE_INTENT_SUMMARY,
    AOAI_INTENT_ENDPOINT, AOAI_SUMMARY_ENDPOINT,
    DEFAULT_AUDIO_FILE,
    load_audio_16k_mono, describe_audio,
    create_aoai_clients, warmup_aoai, analyze_phrase,
    get_auth_token, create_speech_config_message,
    create_speech_context_message, create_audio_message,
    warmup_speech_token, ResultsCsvWriter,
)

# Configuration
AUDIO_FILE = DEFAULT_AUDIO_FILE  # default selection

# Directory used both to list local .wav files and to store uploaded audio
AUDIO_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(AUDIO_DIR, "uploads")


@st.cache_resource(show_spinner=False)
def get_aoai_client():
    """Creates the Azure OpenAI client(s) (Entra ID auth) once per Streamlit session
    and warms them up. Cached so reruns never re-create or re-warm the clients,
    keeping the first real call fast and out of the UI critical path."""
    clients = create_aoai_clients()
    warmup_aoai(clients)
    # Pre-fetch the Speech service Entra ID token too, so pressing "Start" does not
    # pay the token round-trip before the WebSocket connection.
    warmup_speech_token()
    return clients


async def receive_messages(ws, aoai_client, event_queue):
    """Task to receive messages from the server. Pushes results to event_queue
    instead of printing, so the UI thread can render them without being in the
    recognition/LLM critical path."""
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

                            # Accumulate the phrase and analyze it with the LLM
                            if display_text.strip():
                                conversation_phrases.append(display_text)
                                full_conversation = " ".join(conversation_phrases)
                                # Run the (blocking) LLM call without blocking the event loop
                                analysis = await asyncio.to_thread(
                                    analyze_phrase, aoai_client, display_text, full_conversation
                                )
                                event_queue.put({
                                    "type": "phrase",
                                    "text": display_text,
                                    "analysis": analysis,
                                })
                        elif recognition_status == 'NoMatch':
                            if SHOW_INFO:
                                event_queue.put({"type": "info", "text": "No speech detected in this segment"})
                        else:
                            if SHOW_INFO:
                                event_queue.put({"type": "info", "text": f"Status: {recognition_status}"})

                    elif path == 'speech.hypothesis':
                        result = json.loads(body)
                        text = result.get('Text', '')
                        if SHOW_PARTIAL:
                            event_queue.put({"type": "partial", "text": text})

                    elif path == 'turn.end':
                        if SHOW_INFO:
                            event_queue.put({"type": "info", "text": "Turn ended"})
                        return

            else:
                # Binary message (not expected in response)
                pass

    except websockets.ConnectionClosed:
        if SHOW_INFO:
            event_queue.put({"type": "info", "text": "Connection closed by the server"})
    except Exception as e:
        event_queue.put({"type": "error", "text": f"Error receiving messages: {e}"})


async def stream_audio(aoai_client, event_queue, audio_file):
    """Connects to Azure Speech Service over WebSocket, streams the audio in
    real time and drives the receive/analysis loop."""
    token = await get_auth_token()
    request_id = str(uuid.uuid4()).replace('-', '')

    # Entra ID authentication via Authorization header
    headers = {
        "Authorization": f"Bearer {token}"
    }

    event_queue.put({"type": "status", "text": "Connected to Azure Speech Service"})
    async with websockets.connect(WS_URL, additional_headers=headers, max_size=10**7) as ws:
        # 1. Send the configuration message
        config_message = create_speech_config_message(request_id)
        await ws.send(config_message)

        # 1b. Send the speech.context message to enable Semantic segmentation
        context_message = create_speech_context_message(request_id)
        await ws.send(context_message)

        # Start the task to receive messages
        receive_task = asyncio.create_task(receive_messages(ws, aoai_client, event_queue))

        # 2. Send audio in chunks (converted to 16 kHz mono 16-bit PCM)
        event_queue.put({"type": "status", "text": "Streaming audio in real time..."})
        audio_pcm = load_audio_16k_mono(audio_file)
        # Bytes per chunk: 16-bit (2 bytes) * TARGET_SAMPLE_RATE * CHUNK_MS
        bytes_per_chunk = int(TARGET_SAMPLE_RATE * (CHUNK_MS / 1000.0)) * 2

        chunk_count = 0
        for offset in range(0, len(audio_pcm), bytes_per_chunk):
            data = audio_pcm[offset:offset + bytes_per_chunk]
            if not data:
                break

            # Create audio message with binary header
            audio_message = create_audio_message(request_id, data)
            await ws.send(audio_message)

            chunk_count += 1

            # Simulate real-time streaming
            await asyncio.sleep(CHUNK_MS / 1000.0)

        # 3. Send an empty audio message to signal the end
        final_message = create_audio_message(request_id, b'')
        await ws.send(final_message)

        # 4. Wait to receive all results
        await receive_task

        event_queue.put({"type": "status", "text": "Process completed"})


def run_pipeline(aoai_client, event_queue, audio_file):
    """Worker entry point: runs the asyncio pipeline in its own thread with its
    own event loop, fully decoupled from Streamlit's rerun loop."""
    try:
        asyncio.run(stream_audio(aoai_client, event_queue, audio_file))
    except Exception as e:
        event_queue.put({"type": "error", "text": str(e)})
    finally:
        event_queue.put({"type": "done"})


# ----------------------------- Streamlit UI ---------------------------------

def list_local_wav_files():
    """Returns the list of .wav file names found in the app directory (sorted)."""
    try:
        return sorted(f for f in os.listdir(AUDIO_DIR) if f.lower().endswith(".wav"))
    except OSError:
        return []


st.set_page_config(page_title="Speech-to-Text + Intent, Sentiment & Summary", layout="wide")
st.title("Real-Time Transcription, Intent, Sentiment & Summary")
st.caption(
    "Audio is streamed over WebSocket to Azure Speech; each final phrase is "
    "analyzed by Azure OpenAI. Processing runs in a background thread, so the "
    "UI never affects transcription or LLM latency."
)

# Warm up Azure OpenAI eagerly at page load (NOT on the Start click). This builds
# the client(s), pre-fetches the Entra ID token and opens the HTTPS connection(s)
# with a 'ping' call WHILE the user is still choosing the audio source, so pressing
# "Start transcription" no longer pays that latency. Cached via @st.cache_resource,
# so it runs only once per Streamlit session and is a no-op on later reruns.
if "aoai_warm" not in st.session_state:
    with st.spinner("Warming up Azure OpenAI (first load only)..."):
        get_aoai_client()
    st.session_state.aoai_warm = True

with st.sidebar:
    st.subheader("Configuration")
    st.text(f"Region: {SPEECH_REGION}")
    st.text(f"Language: {LANGUAGE}")
    if USE_SEPARATE_INTENT_SUMMARY:
        st.text(f"Intent model: {AOAI_INTENT_MODEL}")
        st.text(f"Intent endpoint: {AOAI_INTENT_ENDPOINT}")
        st.text(f"Summary model: {AOAI_SUMMARY_MODEL}")
        st.text(f"Summary endpoint: {AOAI_SUMMARY_ENDPOINT}")
    else:
        st.text(f"Model: {AOAI_INTENT_MODEL}")
        st.text(f"Endpoint: {AOAI_INTENT_ENDPOINT}")
    #st.text(f"Chunk: {CHUNK_MS} ms")

    st.markdown("---")
    st.subheader("Audio source")
    source = st.radio(
        "Select the audio source",
        ["Local file", "Select another file"],
        disabled=st.session_state.get("running", False),
    )

    selected_audio = None
    if source == "Local file":
        local_files = list_local_wav_files()
        if local_files:
            default_idx = local_files.index(AUDIO_FILE) if AUDIO_FILE in local_files else 0
            chosen = st.selectbox(
                "Local .wav files",
                local_files,
                index=default_idx,
                disabled=st.session_state.get("running", False),
            )
            selected_audio = os.path.join(AUDIO_DIR, chosen)
        else:
            st.warning("No .wav files found next to the app.")
    else:
        uploaded = st.file_uploader(
            "Upload a .wav file",
            type=["wav"],
            disabled=st.session_state.get("running", False),
        )
        if uploaded is not None:
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            selected_audio = os.path.join(UPLOAD_DIR, uploaded.name)
            with open(selected_audio, "wb") as f:
                f.write(uploaded.getbuffer())

    if selected_audio:
        desc, ok = describe_audio(selected_audio)
        st.caption(f"Selected: {os.path.basename(selected_audio)}")
        if ok:
            st.success(f"Format: {desc}")
        else:
            st.warning(f"Format: {desc} (required: 16000 Hz, mono, 16-bit PCM)")

    st.markdown("---")
    st.subheader("CSV output")
    csv_filename = st.text_input(
        "CSV file name (optional)",
        value=st.session_state.get("csv_filename", ""),
        placeholder="results.csv",
        help="If set, the per-segment results (segment, transcription, intent, "
             "sentiment, summary, time) are written to this CSV file. Relative "
             "names are saved next to the app.",
        disabled=st.session_state.get("running", False),
    )
    st.session_state.csv_filename = csv_filename

    st.markdown("---")
    st.markdown("**Possible intents**")
    # Show each intent with its description (loaded from intents.csv) when available.
    if any(INTENT_DESCRIPTIONS.get(name) for name in INTENTS):
        with st.expander("View intent descriptions"):
            for name in INTENTS:
                desc = INTENT_DESCRIPTIONS.get(name, "")
                st.markdown(f"- **{name}**: {desc}" if desc else f"- **{name}**")
    else:
        st.write(", ".join(INTENTS))

# Persistent state across reruns
if "results" not in st.session_state:
    st.session_state.results = []   # list of finished phrase events
if "status" not in st.session_state:
    st.session_state.status = "Idle"
if "running" not in st.session_state:
    st.session_state.running = False
if "finished" not in st.session_state:
    st.session_state.finished = False
if "event_queue" not in st.session_state:
    st.session_state.event_queue = None
if "worker" not in st.session_state:
    st.session_state.worker = None
if "csv_writer" not in st.session_state:
    st.session_state.csv_writer = None
if "csv_path" not in st.session_state:
    st.session_state.csv_path = None

col_start, col_reset = st.columns(2)
start_clicked = col_start.button("Start transcription", type="primary",
                                 disabled=st.session_state.running or not selected_audio)
reset_clicked = col_reset.button("Reset", disabled=st.session_state.running)

if reset_clicked:
    if st.session_state.csv_writer:
        st.session_state.csv_writer.close()
    st.session_state.csv_writer = None
    st.session_state.csv_path = None
    st.session_state.results = []
    st.session_state.status = "Idle"
    st.session_state.finished = False
    st.session_state.event_queue = None
    st.session_state.worker = None
    st.rerun()

if start_clicked and not st.session_state.running and selected_audio:
    # Build/warm the AOAI client once, then launch the background worker.
    aoai_client = get_aoai_client()
    st.session_state.results = []
    st.session_state.finished = False
    st.session_state.status = f"Starting... ({os.path.basename(selected_audio)})"
    st.session_state.event_queue = queue.Queue()
    # Open the CSV results file (if a name was provided). Relative names are saved
    # next to the app. The writer lives in session_state so the segment counter
    # survives reruns; rows are written by the UI thread while draining events.
    if st.session_state.csv_writer:
        st.session_state.csv_writer.close()
        st.session_state.csv_writer = None
        st.session_state.csv_path = None
    csv_name = (st.session_state.get("csv_filename") or "").strip()
    if csv_name:
        csv_path = csv_name if os.path.isabs(csv_name) else os.path.join(AUDIO_DIR, csv_name)
        try:
            st.session_state.csv_writer = ResultsCsvWriter(csv_path)
            st.session_state.csv_path = csv_path
        except OSError as ex:
            st.session_state.csv_writer = None
            st.session_state.csv_path = None
            st.warning(f"Could not open CSV file '{csv_path}': {ex}")
    worker = threading.Thread(
        target=run_pipeline,
        args=(aoai_client, st.session_state.event_queue, selected_audio),
        daemon=True,
    )
    worker.start()
    st.session_state.worker = worker
    st.session_state.running = True

# Drain any events produced by the background worker (non-blocking).
def drain_event_queue():
    """Moves any pending worker events into session_state. Returns True when the
    worker signalled completion during this drain."""
    if st.session_state.event_queue is None:
        return False
    q = st.session_state.event_queue
    just_finished = False
    while True:
        try:
            evt = q.get_nowait()
        except queue.Empty:
            break
        etype = evt.get("type")
        if etype == "phrase":
            st.session_state.results.append(evt)
            # Write the row to the CSV file (if enabled), skipping failed analyses.
            writer = st.session_state.csv_writer
            analysis = evt.get("analysis") or {}
            if writer and "error" not in analysis:
                writer.write_row(evt.get("text", ""), analysis)
        elif etype in ("status", "info"):
            st.session_state.status = evt["text"]
        elif etype == "error":
            st.session_state.status = f"Error: {evt['text']}"
        elif etype == "done":
            st.session_state.running = False
            st.session_state.finished = True
            just_finished = True
            if st.session_state.csv_writer:
                st.session_state.csv_writer.close()
                st.session_state.status = (
                    f"Process completed. Results written to {st.session_state.csv_path}"
                )
    return just_finished


def render_results():
    """Renders the status line and all finished phrases (latest first)."""
    st.info(f"Status: {st.session_state.status}")
    st.subheader("Conversation")
    if not st.session_state.results:
        st.write("_No phrases yet._")
        return
    for i, evt in enumerate(reversed(st.session_state.results), start=1):
        idx = len(st.session_state.results) - i + 1
        analysis = evt.get("analysis") or {}
        with st.container(border=True):
            st.markdown(f"**[{idx}] Transcription:** {evt['text']}")
            if "error" in analysis:
                st.error(f"LLM analysis failed: {analysis['error']}")
            else:
                cols = st.columns([1, 1, 3, 1]) if SHOW_TIME else st.columns([1, 1, 4])
                cols[0].markdown(f"**Intent:** {analysis.get('intent', 'unknown')}")
                cols[1].markdown(f"**Sentiment:** {analysis.get('sentiment', 'unknown')}")
                cols[2].markdown(f"**Summary:** {analysis.get('summary', '')}")
                if SHOW_TIME:
                    client_s = analysis.get("elapsed_s", 0) or 0
                    server_s = analysis.get("server_elapsed_s")
                    if server_s is not None:
                        cols[3].markdown(
                            f"**LLM time:** {server_s:.3f} s _(server)_\n\n"
                            f"_{client_s:.3f} s (client)_"
                        )
                    else:
                        cols[3].markdown(f"**LLM time:** {client_s:.3f} s")


# Live view: while the worker runs, this fragment auto-refreshes ON ITS OWN every
# 0.5 s WITHOUT re-running the whole script. That keeps the heavy widgets (sidebar,
# file listing, buttons) out of the refresh loop, so the UI no longer steals the
# GIL from the background worker and the measured LLM latency stays clean.
def _live_view():
    just_finished = drain_event_queue()
    render_results()
    # When the worker finishes, do ONE full rerun to stop the fragment's run_every
    # cadence and re-enable the Start/Reset buttons.
    if just_finished:
        st.rerun()


run_every = 0.5 if st.session_state.running else None
st.fragment(run_every=run_every)(_live_view)()

