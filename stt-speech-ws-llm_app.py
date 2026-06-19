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
import time
import uuid
import queue
import threading

import streamlit as st

import common_functions as cf
from common_functions import (
    SPEECH_REGION, LANGUAGE, CHUNK_MS, TARGET_SAMPLE_RATE,
    SHOW_INFO, SHOW_PARTIAL, SHOW_TIME, INTENTS, INTENT_DESCRIPTIONS, WS_URL,
    AOAI_INTENT_MODEL, AOAI_SUMMARY_MODEL, USE_SEPARATE_INTENT_SUMMARY,
    AOAI_INTENT_ENDPOINT, AOAI_SUMMARY_ENDPOINT,
    DEFAULT_AUDIO_FILE, MAX_CONNECTION_AUDIO_SECONDS,
    INCREMENTAL_SUMMARY, SUMMARY_REFRESH_EVERY,
    load_audio_16k_mono, describe_audio,
    create_aoai_clients, warmup_aoai, analyze_phrase,
    get_auth_token, create_speech_config_message,
    create_speech_context_message, create_audio_message,
    iter_audio_connection_slices,
    warmup_speech_token, ResultsCsvWriter,
)

# Configuration
AUDIO_FILE = DEFAULT_AUDIO_FILE  # default selection

# Directory used both to list local .wav files and to store uploaded audio
AUDIO_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(AUDIO_DIR, "uploads")

# Grace period (seconds) to wait for the final turn.end after a slice has been
# fully streamed. Guards against a connection that stops responding and would
# otherwise hang the worker forever; on timeout we reconnect/move on.
TURN_END_GRACE_SECONDS = float(os.getenv("TURN_END_GRACE_SECONDS", "60"))


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


class _StopRequested(Exception):
    """Raised inside the worker when the UI asks to stop transcription early."""


async def receive_messages(ws, aoai_client, event_queue, state):
    """Task to receive messages from the server. Pushes results to event_queue
    instead of printing, so the UI thread can render them without being in the
    recognition/LLM critical path.

    'state' holds the running conversation and the monotonic phrase id; it is owned
    by the caller and shared across reconnections, so transcripts and ids stay
    coherent when a long audio is streamed over several WebSocket connections."""
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
                                state["phrase_id"] += 1
                                phrase_id = state["phrase_id"]
                                state["conversation_phrases"].append(display_text)
                                full_conversation = " ".join(state["conversation_phrases"])
                                # Emit the transcription immediately so the UI can
                                # render it without waiting for the LLM response.
                                event_queue.put({
                                    "type": "transcription",
                                    "id": phrase_id,
                                    "text": display_text,
                                })
                                # Run the (blocking) LLM call without blocking the event loop
                                analysis = await asyncio.to_thread(
                                    analyze_phrase, aoai_client, display_text,
                                    full_conversation, state["previous_summary"], phrase_id
                                )
                                if analysis and "error" not in analysis:
                                    state["previous_summary"] = analysis.get(
                                        "summary", state["previous_summary"]
                                    )
                                # Emit the analysis as a follow-up event correlated
                                # by id, so the UI fills in the matching phrase.
                                event_queue.put({
                                    "type": "analysis",
                                    "id": phrase_id,
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


async def receive_segment(ws, event_queue):
    """Receiver used in FIXED-TIME SEGMENT mode: reads the messages of the CURRENT
    recognition turn and returns the list of final phrase texts once turn.end is
    received. The audio fed to this turn is an exact client-side slice, so its time
    boundaries are precise regardless of the service's internal segmentation."""
    phrases = []
    try:
        async for message in ws:
            if not isinstance(message, str):
                continue
            lines = message.split('\r\n\r\n', 1)
            if len(lines) <= 1:
                continue
            headers_text, body = lines[0], lines[1]
            path = None
            for line in headers_text.split('\r\n'):
                if line.startswith('Path:'):
                    path = line.split(':', 1)[1]
                    break

            if path == 'speech.phrase':
                result = json.loads(body)
                if result.get('RecognitionStatus') == 'Success':
                    display_text = result.get('DisplayText', '')
                    if display_text.strip():
                        phrases.append(display_text)
                elif result.get('RecognitionStatus') == 'NoMatch':
                    if SHOW_INFO:
                        event_queue.put({"type": "info", "text": "No speech detected in this segment"})
            elif path == 'speech.hypothesis':
                if SHOW_PARTIAL:
                    result = json.loads(body)
                    event_queue.put({"type": "partial", "text": result.get('Text', '')})
            elif path == 'turn.end':
                # End of this segment's turn: return everything collected.
                return phrases
    except websockets.ConnectionClosed:
        if SHOW_INFO:
            event_queue.put({"type": "info", "text": "Connection closed by the server"})
    return phrases


async def stream_audio(aoai_client, event_queue, audio_file, interval_seconds=None, stop_event=None):
    """Connects to Azure Speech Service over WebSocket, streams the audio in
    real time and drives the receive/analysis loop.

    When ``interval_seconds`` is None, Semantic segmentation is used and each final
    phrase is analyzed as it arrives (continuous turn). When it is a number, the
    audio is sliced client-side into exact N-second segments, each sent as its own
    recognition turn, and analyzed once fully transcribed.

    ``stop_event`` is a threading.Event the UI sets to request an early stop; it is
    checked while pacing the audio so the worker exits promptly and cleanly.

    Azure Speech closes a connection after ~635 s of total duration, so long audio
    is streamed over several back-to-back connections; the running conversation and
    phrase ids are shared across them so the LLM context and the UI ids stay
    coherent."""
    def _check_stop():
        """Raises _StopRequested when the UI has asked the worker to stop."""
        if stop_event is not None and stop_event.is_set():
            raise _StopRequested()

    # Pre-load and convert the audio once so it can be streamed in chunks.
    audio_pcm = load_audio_16k_mono(audio_file)
    # Bytes per chunk: 16-bit (2 bytes) * TARGET_SAMPLE_RATE * CHUNK_MS
    bytes_per_chunk = int(TARGET_SAMPLE_RATE * (CHUNK_MS / 1000.0)) * 2

    # Shared state across all connections (LLM context + monotonic phrase id).
    state = {"conversation_phrases": [], "phrase_id": 0, "previous_summary": ""}

    if interval_seconds is None:
        # ----------------------- SEMANTIC MODE (default) -----------------------
        # Long audio is split into back-to-back connections (each up to the
        # ~635s WebSocket limit), with semantic segmentation per connection. If a
        # connection drops mid-slice (duration limit reached early or a 1012
        # service restart), we reconnect and RESUME the same slice from the last
        # byte we managed to send, so already-transcribed audio is not re-sent
        # (no duplicate phrases) and no audio is lost. A bounded grace timeout on
        # the final turn.end guards against a connection that goes silent and
        # would otherwise hang the worker forever.
        slices = list(iter_audio_connection_slices(audio_pcm))
        total_connections = len(slices)
        for connection_index, _total, audio_slice in slices:
            sent_offset = 0
            attempts = 0
            while sent_offset < len(audio_slice):
                attempts += 1
                token = await get_auth_token()
                request_id = str(uuid.uuid4()).replace('-', '')
                headers = {"Authorization": f"Bearer {token}"}

                if total_connections > 1:
                    event_queue.put({"type": "status",
                                     "text": f"Connected to Azure Speech Service "
                                             f"(connection {connection_index}/{total_connections})"})
                else:
                    event_queue.put({"type": "status", "text": "Connected to Azure Speech Service"})

                ws = None
                receive_task = None
                try:
                    ws = await websockets.connect(WS_URL, additional_headers=headers, max_size=10**7)
                    # 1. Send the configuration message (connection-level, sent once).
                    await ws.send(create_speech_config_message(request_id))
                    # 1b. Send the speech.context message to enable Semantic segmentation
                    await ws.send(create_speech_context_message(request_id))

                    # Start the task to receive messages
                    receive_task = asyncio.create_task(
                        receive_messages(ws, aoai_client, event_queue, state)
                    )

                    # 2. Send audio in chunks (converted to 16 kHz mono 16-bit PCM),
                    # resuming from the last byte sent on previous attempts.
                    event_queue.put({"type": "status", "text": "Streaming audio in real time..."})
                    for offset in range(sent_offset, len(audio_slice), bytes_per_chunk):
                        _check_stop()
                        data = audio_slice[offset:offset + bytes_per_chunk]
                        if not data:
                            break
                        await ws.send(create_audio_message(request_id, data))
                        sent_offset = offset + len(data)
                        # Simulate real-time streaming
                        await asyncio.sleep(CHUNK_MS / 1000.0)

                    # 3. Send an empty audio message to signal the end of this turn
                    await ws.send(create_audio_message(request_id, b''))

                    # 4. Wait (bounded) to receive all results for this connection
                    await asyncio.wait_for(receive_task, timeout=TURN_END_GRACE_SECONDS)
                    # Slice fully streamed and finalized; move to the next slice.
                    break
                except (websockets.ConnectionClosed, asyncio.TimeoutError) as ex:
                    # The connection dropped mid-slice or stopped sending turn.end.
                    # Reconnect and resume this slice from the last byte sent.
                    if receive_task is not None:
                        receive_task.cancel()
                    reason = ("timed out waiting for turn.end"
                              if isinstance(ex, asyncio.TimeoutError) else "connection closed")
                    if attempts >= 5:
                        event_queue.put({"type": "error",
                                         "text": f"Giving up on connection {connection_index} "
                                                 f"after {attempts} attempts ({reason})"})
                        break
                    event_queue.put({"type": "status",
                                     "text": f"Reconnecting ({reason}); resuming "
                                             f"connection {connection_index}..."})
                finally:
                    if ws is not None:
                        try:
                            await ws.close()
                        except Exception:
                            pass
    else:
        # ----------------- SEGMENT MODE (exact X-second slices) ----------------
        # Slice the audio into exact interval_seconds segments, each sent as its
        # OWN recognition turn (fresh X-RequestId). Unlike the semantic mode, the
        # LLM analysis runs synchronously BETWEEN segments, so the connection's
        # wall-clock grows by audio time PLUS analysis time. We track the real
        # elapsed wall-clock per connection and open a fresh connection before it
        # would exceed the budget, and we retry a segment over a new connection if
        # the current one drops (duration limit or 1012 service restart).
        bytes_per_sample = 2
        bytes_per_segment = int(TARGET_SAMPLE_RATE * interval_seconds) * bytes_per_sample
        total_bytes = len(audio_pcm)
        seconds_per_byte = 1.0 / (TARGET_SAMPLE_RATE * bytes_per_sample)

        seg_starts = list(range(0, total_bytes, bytes_per_segment))

        # Current connection and the wall-clock time (monotonic) at which it opened.
        conn = {"ws": None, "start": 0.0, "index": 0}

        async def open_connection():
            token = await get_auth_token()
            conn_request_id = str(uuid.uuid4()).replace('-', '')
            headers = {"Authorization": f"Bearer {token}"}
            conn["index"] += 1
            event_queue.put({"type": "status",
                             "text": f"Connected to Azure Speech Service "
                                     f"(connection {conn['index']})"})
            conn["ws"] = await websockets.connect(WS_URL, additional_headers=headers, max_size=10**7)
            await conn["ws"].send(create_speech_config_message(conn_request_id))
            conn["start"] = time.monotonic()

        async def close_connection():
            if conn["ws"] is not None:
                try:
                    await conn["ws"].close()
                except Exception:
                    pass
                conn["ws"] = None

        event_queue.put({"type": "status", "text": "Streaming audio in real time..."})
        try:
            for seg_start in seg_starts:
                _check_stop()
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
                            (time.monotonic() - conn["start"] + interval_seconds) > MAX_CONNECTION_AUDIO_SECONDS):
                        await close_connection()
                        await open_connection()

                    segment_request_id = str(uuid.uuid4()).replace('-', '')
                    recv_task = asyncio.create_task(receive_segment(conn["ws"], event_queue))
                    try:
                        # Stream this segment's audio chunks, paced for real time.
                        for offset in range(seg_start, seg_end, bytes_per_chunk):
                            _check_stop()
                            data = audio_pcm[offset:min(offset + bytes_per_chunk, seg_end)]
                            if not data:
                                break
                            await conn["ws"].send(create_audio_message(segment_request_id, data))
                            await asyncio.sleep(CHUNK_MS / 1000.0)  # simulate real time

                        # End this turn so the service finalizes exactly this segment.
                        await conn["ws"].send(create_audio_message(segment_request_id, b''))
                        phrases = await recv_task
                    except websockets.ConnectionClosed:
                        # Connection dropped mid-segment (duration limit or 1012
                        # service restart). Reconnect and retry this same segment.
                        recv_task.cancel()
                        await close_connection()
                        if attempts >= 5:
                            event_queue.put({"type": "error",
                                             "text": f"Giving up on segment "
                                                     f"{start_s:g}-{end_s:g}s after {attempts} attempts"})
                            phrases = []
                        else:
                            event_queue.put({"type": "status",
                                             "text": f"Connection closed during segment "
                                                     f"{start_s:g}-{end_s:g}s; reconnecting..."})

                segment_text = " ".join(phrases).strip()
                if segment_text:
                    state["phrase_id"] += 1
                    phrase_id = state["phrase_id"]
                    state["conversation_phrases"].append(segment_text)
                    full_conversation = " ".join(state["conversation_phrases"])
                    # Emit the transcription immediately, with its exact time range.
                    event_queue.put({
                        "type": "transcription",
                        "id": phrase_id,
                        "text": segment_text,
                        "range": f"{start_s:g}-{end_s:g}s",
                    })
                    # Run the (blocking) LLM call without blocking the event loop.
                    analysis = await asyncio.to_thread(
                        analyze_phrase, aoai_client, segment_text,
                        full_conversation, state["previous_summary"], phrase_id
                    )
                    if analysis and "error" not in analysis:
                        state["previous_summary"] = analysis.get(
                            "summary", state["previous_summary"]
                        )
                    event_queue.put({
                        "type": "analysis",
                        "id": phrase_id,
                        "text": segment_text,
                        "analysis": analysis,
                    })
        finally:
            await close_connection()

    event_queue.put({"type": "status", "text": "Process completed"})


def run_pipeline(aoai_client, event_queue, audio_file, interval_seconds=None, stop_event=None):
    """Worker entry point: runs the asyncio pipeline in its own thread with its
    own event loop, fully decoupled from Streamlit's rerun loop."""
    try:
        asyncio.run(stream_audio(aoai_client, event_queue, audio_file,
                                 interval_seconds, stop_event))
    except _StopRequested:
        event_queue.put({"type": "status", "text": "Stopped by user"})
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
    st.subheader("Segmentation")
    semantic_segmentation = st.checkbox(
        "Semantic segmentation",
        value=st.session_state.get("semantic_segmentation", True),
        help="When enabled, the Speech service segments the conversation "
             "semantically. When disabled, the audio is sliced client-side into "
             "exact fixed-length segments.",
        disabled=st.session_state.get("running", False),
    )
    st.session_state.semantic_segmentation = semantic_segmentation
    segment_seconds = None
    if not semantic_segmentation:
        segment_seconds = st.number_input(
            "Segment length (seconds)",
            min_value=1.0,
            value=float(st.session_state.get("segment_seconds", 10.0)),
            step=1.0,
            help="Length of each fixed-time audio segment sent to the LLM.",
            disabled=st.session_state.get("running", False),
        )
        st.session_state.segment_seconds = segment_seconds

    st.markdown("---")
    st.subheader("Summary")
    incremental_summary = st.checkbox(
        "Incremental summary",
        value=st.session_state.get("incremental_summary", INCREMENTAL_SUMMARY),
        help="When enabled, the running summary is built from the previous summary "
             "plus the latest phrase (bounded input, lower latency on long calls). "
             "When disabled, the whole transcript is re-summarized on every phrase.",
        disabled=st.session_state.get("running", False),
    )
    st.session_state.incremental_summary = incremental_summary
    summary_refresh_every = 0
    if incremental_summary:
        regenerate = st.checkbox(
            "Regenerate full summary periodically",
            value=st.session_state.get("summary_regenerate", SUMMARY_REFRESH_EVERY > 0),
            help="Every N phrases, regenerate the summary from the whole transcript "
                 "to recover detail lost to incremental drift.",
            disabled=st.session_state.get("running", False),
        )
        st.session_state.summary_regenerate = regenerate
        if regenerate:
            summary_refresh_every = int(st.number_input(
                "Regenerate every N phrases",
                min_value=1,
                value=int(st.session_state.get("summary_refresh_every", SUMMARY_REFRESH_EVERY) or 10),
                step=1,
                help="Number of phrases between full re-summaries.",
                disabled=st.session_state.get("running", False),
            ))
            st.session_state.summary_refresh_every = summary_refresh_every

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
if "stop_event" not in st.session_state:
    st.session_state.stop_event = None

running = st.session_state.running

# Toggle button colours: red background / white text when idle ("Start"); grey
# background / black text when running ("Stop"). The button is keyed so the CSS
# can target its wrapper (Streamlit adds a .st-key-<key> class to it).
toggle_color_css = (
    """
    <style>
    div.st-key-startstop_btn button {
        background-color: #9e9e9e !important;
        color: #000000 !important;
        border-color: #9e9e9e !important;
    }
    </style>
    """
    if running else
    """
    <style>
    div.st-key-startstop_btn button {
        background-color: #e60000 !important;
        color: #ffffff !important;
        border-color: #e60000 !important;
    }
    </style>
    """
)
st.markdown(toggle_color_css, unsafe_allow_html=True)

col_start, col_reset = st.columns(2)
toggle_label = "Stop transcription" if running else "Start transcription"
toggle_clicked = col_start.button(
    toggle_label, key="startstop_btn",
    disabled=(not running and not selected_audio),
)
reset_clicked = col_reset.button("Reset", disabled=running)

start_clicked = False
if toggle_clicked:
    if running:
        # Ask the worker to stop; it checks the event while pacing the audio and
        # exits cleanly, emitting 'done' which flips 'running' back to False.
        if st.session_state.get("stop_event") is not None:
            st.session_state.stop_event.set()
        st.session_state.status = "Stopping..."
        st.rerun()
    else:
        start_clicked = True

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
    st.session_state.stop_event = None
    st.rerun()

if start_clicked and not st.session_state.running and selected_audio:
    # Build/warm the AOAI client once, then launch the background worker.
    aoai_client = get_aoai_client()
    st.session_state.results = []
    st.session_state.finished = False
    st.session_state.status = f"Starting... ({os.path.basename(selected_audio)})"
    st.session_state.event_queue = queue.Queue()
    stop_event = threading.Event()
    st.session_state.stop_event = stop_event
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
    # When Semantic segmentation is off, slice the audio into fixed-time segments.
    interval_seconds = None
    if not st.session_state.get("semantic_segmentation", True):
        interval_seconds = float(st.session_state.get("segment_seconds", 10.0))
    # Apply the summary options chosen in the sidebar so the worker's analyze_phrase
    # (which reads these module-level globals) uses them for this run.
    cf.INCREMENTAL_SUMMARY = bool(st.session_state.get("incremental_summary", INCREMENTAL_SUMMARY))
    cf.SUMMARY_REFRESH_EVERY = (
        int(st.session_state.get("summary_refresh_every", SUMMARY_REFRESH_EVERY))
        if st.session_state.get("incremental_summary", INCREMENTAL_SUMMARY)
        and st.session_state.get("summary_regenerate", SUMMARY_REFRESH_EVERY > 0)
        else 0
    )
    worker = threading.Thread(
        target=run_pipeline,
        args=(aoai_client, st.session_state.event_queue, selected_audio,
              interval_seconds, stop_event),
        daemon=True,
    )
    worker.start()
    st.session_state.worker = worker
    st.session_state.running = True
    # Rerun immediately so the toggle button re-renders as "Stop transcription"
    # (grey). The button is outside the auto-refreshing fragment, so without this
    # rerun it would keep showing "Start transcription" until the next interaction.
    st.rerun()

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
        if etype == "transcription":
            # Add the phrase immediately with a pending (None) analysis so the UI
            # shows the transcription right away.
            st.session_state.results.append({
                "id": evt["id"],
                "text": evt.get("text", ""),
                "range": evt.get("range"),
                "analysis": None,
            })
        elif etype == "analysis":
            # Fill in the matching phrase (by id) with the LLM result.
            analysis = evt.get("analysis") or {}
            for row in st.session_state.results:
                if row.get("id") == evt["id"]:
                    row["analysis"] = analysis
                    break
            # Write the row to the CSV file (if enabled), skipping failed analyses.
            writer = st.session_state.csv_writer
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
        analysis = evt.get("analysis")
        with st.container(border=True):
            time_range = evt.get("range")
            label = f"**[{idx}] Transcription:** {evt['text']}"
            if time_range:
                label = f"**[{idx}] Transcription** _({time_range})_**:** {evt['text']}"
            st.markdown(label)
            if analysis is None:
                # Transcription is ready but the LLM has not responded yet.
                st.caption(":hourglass_flowing_sand: Analyzing...")
            elif "error" in analysis:
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

