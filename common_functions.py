"""Shared configuration and helper functions for the STT + LLM programs.

This module centralizes everything that was previously duplicated across
stt-speech-ws-llm.py, stt-speech-llm.py and stt-speech-ws-llm_app.py:

- Configuration constants (Speech / Azure OpenAI / language / audio format).
- Audio conversion to 16 kHz mono 16-bit PCM.
- Azure OpenAI client creation, warm-up and phrase analysis (intent + summary).
- Speech service authentication.
- WebSocket protocol message builders.
"""

from dotenv import load_dotenv
import argparse
import csv
import json
import os
import struct
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
load_dotenv(override=True)


def _env_bool(name, default=False):
    """Reads a boolean flag from the environment (accepts true/1/yes/on)."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


SPEECH_REGION = os.getenv("SPEECH_REGION")
# Resource ID of the Speech resource (required for Entra ID authentication)
# Format: /subscriptions/<sub-id>/resourceGroups/<rg>/providers/Microsoft.CognitiveServices/accounts/<resource-name>
SPEECH_RESOURCE_ID = os.getenv("SPEECH_RESOURCE_ID")
# Module-level Entra ID bearer token provider for the Speech service. Created lazily
# on first use and reused so the underlying credential caches the token across calls.
_speech_token_provider = None
#LANGUAGE = "es-ES"
LANGUAGE = "en-GB"
DEFAULT_AUDIO_FILE = "customer-support-sample.wav"
CHUNK_MS = 100  # Duration of each chunk (ms)
TARGET_SAMPLE_RATE = 16000  # Required by the Speech service (16 kHz, mono, 16-bit PCM)

# CSV file (next to this module) holding the intent taxonomy used by the LLM
# prompts. It has two columns: "intent" and "description". Editing this file is
# the supported way to add, remove or refine intents without touching the code.
INTENTS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "intents.csv")

# Display control flags
SHOW_INFO = False       # Show "[INFO]" messages
SHOW_PARTIAL = False    # Show "[PARTIAL]" (intermediate hypothesis) messages
SHOW_TIME = True        # Show the time (s) the text model call takes
SHOW_DEBUG = True      # Show "[DEBUG]" messages about which model(s) are called

# Azure OpenAI configuration (Entra ID authentication)
AOAI_API_VERSION = "2025-04-01-preview"

# When SERVERLESS_MODELS is True the models are consumed as Azure AI Foundry
# serverless / Models-as-a-Service deployments through the Azure AI Inference
# SDK (chat completions API) instead of Azure OpenAI (Responses API). This is the
# usual way to deploy open models such as Qwen or GPT-OSS. Auth then uses an API
# key when provided, otherwise Entra ID (DefaultAzureCredential).
SERVERLESS_MODELS = _env_bool("SERVERLESS_MODELS", False)

# Endpoint/model for intent detection and for summary generation. The intent
# variables are required; the summary ones default to the intent values, so a
# single service can be configured with just AZURE_OPENAI_INTENT_*.
# If the endpoints or the models differ between intent and summary, the two are
# processed in parallel with two separate prompts; otherwise a single combined
# call is used.
AOAI_INTENT_ENDPOINT = os.environ["AZURE_OPENAI_INTENT_ENDPOINT"]
AOAI_INTENT_MODEL = os.environ["AZURE_OPENAI_INTENT_MODEL"]
AOAI_SUMMARY_ENDPOINT = os.getenv("AZURE_OPENAI_SUMMARY_ENDPOINT", AOAI_INTENT_ENDPOINT)
AOAI_SUMMARY_MODEL = os.getenv("AZURE_OPENAI_SUMMARY_MODEL", AOAI_INTENT_MODEL)

# Optional API keys used only when SERVERLESS_MODELS is True. If omitted, Entra
# ID (DefaultAzureCredential) is used for the serverless endpoints as well.
AOAI_INTENT_KEY = os.getenv("AZURE_OPENAI_INTENT_KEY")
AOAI_SUMMARY_KEY = os.getenv("AZURE_OPENAI_SUMMARY_KEY", AOAI_INTENT_KEY)

# True when intent and summary must be obtained from a different endpoint or model.
USE_SEPARATE_INTENT_SUMMARY = (
    AOAI_INTENT_ENDPOINT != AOAI_SUMMARY_ENDPOINT
    or AOAI_INTENT_MODEL != AOAI_SUMMARY_MODEL
)

# Possible intents to detect for each phrase, loaded from INTENTS_CSV at startup.
# INTENTS is the list of intent names (used for display); INTENT_DESCRIPTIONS keeps
# the per-intent descriptions used to build the LLM prompt.
def _load_intents(path):
    """Reads the intent taxonomy from a CSV with columns 'intent' and 'description'.
    Returns (names, descriptions_dict). Falls back to a built-in list if the file
    is missing or unreadable, so the app still runs."""
    fallback = [
        ("greetings", ""), ("request_info", ""), ("flight_details", ""),
        ("complain", ""), ("delay", ""), ("apology", ""), ("refund", ""),
        ("thanks", ""), ("closing", ""),
    ]
    rows = fallback
    try:
        with open(path, newline="", encoding="utf-8") as f:
            parsed = [
                (r["intent"].strip(), (r.get("description") or "").strip())
                for r in csv.DictReader(f)
                if r.get("intent") and r["intent"].strip()
            ]
        if parsed:
            rows = parsed
    except (OSError, KeyError) as ex:
        print(f"[STARTUP] Could not read intents from {path}: {ex}. Using defaults.")
    names = [name for name, _ in rows]
    descriptions = {name: desc for name, desc in rows}
    return names, descriptions


INTENTS, INTENT_DESCRIPTIONS = _load_intents(INTENTS_CSV)


def _format_intents_block():
    """Builds the bulleted intent list (with descriptions when available) injected
    into the prompts. Falls back to a comma-separated list when no descriptions."""
    if any(INTENT_DESCRIPTIONS.get(name) for name in INTENTS):
        lines = []
        for name in INTENTS:
            desc = INTENT_DESCRIPTIONS.get(name)
            lines.append(f"- {name}: {desc}" if desc else f"- {name}")
        return "\n".join(lines)
    return ", ".join(INTENTS)


_INTENTS_BLOCK = _format_intents_block()

# Possible sentiments to detect for each (latest) phrase. The sentiment is always
# evaluated on the LATEST phrase alone, never on the accumulated conversation.
SENTIMENTS = ["positive", "neutral", "negative"]

# System prompt for intent detection and running conversation summary (single call)
ANALYSIS_SYSTEM_PROMPT = (
    "You are an assistant that analyzes a customer support phone call transcription.\n"
    "For the latest customer phrase, identify its intent. The possible intents are:\n"
    f"{_INTENTS_BLOCK}\n"
    "If none clearly applies, use \"none\".\n"
    f"Also classify the sentiment of ONLY the latest phrase as one of: {', '.join(SENTIMENTS)}.\n"
    "Also produce a SHORT summary of the whole conversation so far. The summary must "
    "CONDENSE and PARAPHRASE the key points (who is calling, the issue, and what is "
    "being requested) in 1-2 sentences. Do NOT copy or repeat the transcription "
    "verbatim, and do NOT simply concatenate the phrases.\n"
    "Respond ONLY with a valid JSON object with exactly these keys: "
    '{"intent": "<one of the intents or none>", "sentiment": "<one of the sentiments>", "summary": "<concise summary>"}.'
)

# Separate prompts used when intent and summary run on different endpoints/models.
# Intent and sentiment are both per-latest-phrase, so they share this prompt/call.
INTENT_SYSTEM_PROMPT = (
    "You are an assistant that analyzes a customer support phone call transcription.\n"
    "For the latest customer phrase, identify its intent. The possible intents are:\n"
    f"{_INTENTS_BLOCK}\n"
    "If none clearly applies, use \"none\".\n"
    f"Also classify the sentiment of ONLY the latest phrase as one of: {', '.join(SENTIMENTS)}.\n"
    "Respond ONLY with a valid JSON object with exactly these keys: "
    '{"intent": "<one of the intents or none>", "sentiment": "<one of the sentiments>"}.'
)

SUMMARY_SYSTEM_PROMPT = (
    "You are an assistant that analyzes a customer support phone call transcription.\n"
    "Produce a SHORT summary of the whole conversation so far. The summary must "
    "CONDENSE and PARAPHRASE the key points (who is calling, the issue, and what is "
    "being requested) in 1-2 sentences. Do NOT copy or repeat the transcription "
    "verbatim, and do NOT simply concatenate the phrases.\n"
    "Respond ONLY with a valid JSON object with exactly this key: "
    '{"summary": "<concise summary>"}.'
)

# WebSocket service URL
WS_URL = f"wss://{SPEECH_REGION}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1?language={LANGUAGE}&format=detailed"


# --------------------------------------------------------------------------- #
# Display helpers
# --------------------------------------------------------------------------- #
def print_info(message):
    """Prints an [INFO] message if SHOW_INFO is enabled."""
    if SHOW_INFO:
        print(message)


def print_partial(message):
    """Prints a [PARTIAL] message if SHOW_PARTIAL is enabled."""
    if SHOW_PARTIAL:
        print(message)


# --------------------------------------------------------------------------- #
# Command-line argument parsing
# --------------------------------------------------------------------------- #
def parse_audio_file_arg(description, with_csv=False):
    """Parses an optional positional ``audio_file`` argument. If not provided,
    DEFAULT_AUDIO_FILE is used.

    When ``with_csv`` is True, an optional ``--csv``/``-o`` argument is also parsed
    and the function returns ``(audio_file, csv_path)`` (csv_path is None when not
    given). Otherwise only ``audio_file`` is returned (backward compatible)."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "audio_file",
        nargs="?",
        default=DEFAULT_AUDIO_FILE,
        help="Path to the audio file to process (default: %(default)s).",
    )
    if with_csv:
        parser.add_argument(
            "--csv", "-o",
            dest="csv_path",
            default=None,
            help="Path to a CSV file where the per-segment results are written "
                 "(columns: segment, transcription, intent, sentiment, summary, time). "
                 "If omitted, no CSV is written.",
        )
        args = parser.parse_args()
        return args.audio_file, args.csv_path
    return parser.parse_args().audio_file


# --------------------------------------------------------------------------- #
# CSV results writer
# --------------------------------------------------------------------------- #
CSV_COLUMNS = ["segment", "transcription", "intent", "sentiment", "summary", "time"]


class ResultsCsvWriter:
    """Writes per-segment analysis results to a CSV file with the columns:
    segment, transcription, intent, sentiment, summary, time.

    The ``segment`` column is an auto-incrementing counter (1, 2, 3, ...). The
    ``time`` column reports the server-side processing time in seconds when
    available, otherwise the client wall-clock time. Rows are flushed after each
    write so partial results survive an interrupted run."""

    def __init__(self, path):
        self.path = path
        self._file = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(CSV_COLUMNS)
        self._file.flush()
        self._segment = 0

    def write_row(self, transcription, analysis):
        """Appends one result row. ``analysis`` is the dict returned by
        analyze_phrase (intent/sentiment/summary/elapsed_s/server_elapsed_s)."""
        analysis = analysis or {}
        self._segment += 1
        time_s = analysis.get("server_elapsed_s")
        if time_s is None:
            time_s = analysis.get("elapsed_s")
        time_str = f"{time_s:.3f}" if isinstance(time_s, (int, float)) else ""
        self._writer.writerow([
            self._segment,
            transcription,
            analysis.get("intent", ""),
            analysis.get("sentiment", ""),
            analysis.get("summary", ""),
            time_str,
        ])
        self._file.flush()

    def close(self):
        """Closes the underlying file. Safe to call multiple times."""
        if self._file and not self._file.closed:
            self._file.close()


# --------------------------------------------------------------------------- #
# Audio conversion
# --------------------------------------------------------------------------- #
def load_audio_16k_mono(path):
    """Reads a WAV file and returns its audio as 16 kHz, mono, 16-bit PCM bytes.

    The Azure Speech service expects 16 kHz / mono / 16-bit PCM. If the source
    file has a different sample rate, number of channels or sample width, it is
    converted on the fly so any WAV file can be processed.
    """
    with wave.open(path, "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sampwidth = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())

    # Decode raw PCM samples into a float32 array in the range [-1, 1].
    if sampwidth == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 1:
        # 8-bit PCM is unsigned (0..255), centered at 128.
        samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sampwidth == 4:
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth * 8}-bit")

    # Downmix to mono by averaging channels.
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)

    # Resample to the target sample rate using linear interpolation.
    if sample_rate != TARGET_SAMPLE_RATE:
        duration = samples.shape[0] / sample_rate
        target_len = int(round(duration * TARGET_SAMPLE_RATE))
        if target_len > 0 and samples.shape[0] > 1:
            src_idx = np.linspace(0, samples.shape[0] - 1, num=target_len)
            samples = np.interp(src_idx, np.arange(samples.shape[0]), samples)

    # Encode back to 16-bit PCM bytes.
    pcm16 = np.clip(samples * 32768.0, -32768, 32767).astype(np.int16)
    return pcm16.tobytes()


def describe_audio(path):
    """Returns a human-readable description of the WAV file and whether it matches
    the required format (16000 Hz, mono, 16-bit PCM). Returns (text, ok)."""
    try:
        with wave.open(path, "rb") as wf:
            channels = wf.getnchannels()
            rate = wf.getframerate()
            bits = wf.getsampwidth() * 8
            ok = (rate == TARGET_SAMPLE_RATE and channels == 1 and bits == 16)
            text = f"{rate} Hz, {channels} ch, {bits}-bit"
            return text, ok
    except Exception as ex:
        return f"Could not read file: {ex}", False


# --------------------------------------------------------------------------- #
# Azure OpenAI client and phrase analysis
# --------------------------------------------------------------------------- #
def create_aoai_clients():
    """Creates the model client(s) needed for intent and summary analysis. Returns
    a dict with keys 'intent', 'summary' and 'token_provider'. Roles that share the
    same endpoint reuse the same client.

    When SERVERLESS_MODELS is True, Azure AI Inference ChatCompletionsClient is
    used (chat completions API) with an API key when provided, otherwise Entra ID.
    Otherwise, the Azure OpenAI client (Responses API) with Entra ID is used."""
    if SERVERLESS_MODELS:
        from azure.ai.inference import ChatCompletionsClient
        from azure.core.credentials import AzureKeyCredential

        token_credential = DefaultAzureCredential()
        _by_endpoint = {}

        def _client_for(endpoint, key):
            if endpoint not in _by_endpoint:
                credential = AzureKeyCredential(key) if key else token_credential
                _by_endpoint[endpoint] = ChatCompletionsClient(
                    endpoint=endpoint,
                    credential=credential,
                )
            return _by_endpoint[endpoint]

        return {
            "intent": _client_for(AOAI_INTENT_ENDPOINT, AOAI_INTENT_KEY),
            "summary": _client_for(AOAI_SUMMARY_ENDPOINT, AOAI_SUMMARY_KEY),
            "token_provider": None,
        }

    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
    )

    _by_endpoint = {}

    def _client_for(endpoint):
        if endpoint not in _by_endpoint:
            _by_endpoint[endpoint] = AzureOpenAI(
                api_version=AOAI_API_VERSION,
                azure_endpoint=endpoint,
                azure_ad_token_provider=token_provider,
            )
        return _by_endpoint[endpoint]

    return {
        "intent": _client_for(AOAI_INTENT_ENDPOINT),
        "summary": _client_for(AOAI_SUMMARY_ENDPOINT),
        "token_provider": token_provider,
    }


def warmup_aoai(clients):
    """Warm up the model client(s): pre-fetch the Entra ID token (Azure OpenAI only)
    and open the HTTPS connection(s) so the first real intent/summary call is fast."""
    try:
        if not SERVERLESS_MODELS:
            clients["token_provider"]()  # pre-fetch and cache the Entra ID token
        if USE_SEPARATE_INTENT_SUMMARY:
            targets = [("intent", AOAI_INTENT_MODEL), ("summary", AOAI_SUMMARY_MODEL)]
        else:
            # Same endpoint and model: a single combined call uses the intent client.
            targets = [("intent", AOAI_INTENT_MODEL)]

        warmed = set()
        for role, model in targets:
            client = clients[role]
            key = (id(client), model)
            if key in warmed:
                continue
            warmed.add(key)
            if SERVERLESS_MODELS:
                from azure.ai.inference.models import UserMessage
                client.complete(
                    model=model,
                    messages=[UserMessage(content="ping")],
                    max_tokens=16,
                )
            else:
                client.responses.create(
                    model=model,
                    instructions="ping",
                    input="ping",
                    max_output_tokens=16,
                )
    except Exception as ex:
        print(f"[STARTUP] Warm-up skipped: {ex}")


def _build_user_prompt(latest_phrase, full_conversation):
    return (
        f"Latest phrase: \"{latest_phrase}\"\n\n"
        f"Full conversation so far:\n\"{full_conversation}\""
    )


def _extract_json(raw):
    """Parses a JSON object from a model response, tolerating code fences and any
    reasoning/preamble text some open models (e.g. GPT-OSS) emit before the JSON."""
    raw = raw.strip()
    # Strip code fences if present
    if raw.startswith("```"):
        raw = raw.strip("`").lstrip("json").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fall back to the outermost {...} block.
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw[start:end + 1])
        raise


def _server_seconds_from_headers(headers):
    """Extracts the server-side processing time (in seconds) from response headers,
    or None when the service did not report it. Azure OpenAI exposes it via the
    'openai-processing-ms' header; some gateways use 'x-envoy-upstream-service-time'
    (already in ms). Returns a float in seconds or None."""
    if not headers:
        return None
    for key in ("openai-processing-ms", "x-envoy-upstream-service-time"):
        value = headers.get(key)
        if value:
            try:
                return float(value) / 1000.0
            except (TypeError, ValueError):
                continue
    return None


def _call_model(client, model, system_prompt, user_prompt):
    """Calls the model once and returns (parsed_json, elapsed_s, server_s).

    - elapsed_s: client-side wall-clock time of the call (always available).
    - server_s: server-side processing time reported by the service via response
      headers, or None when the service does not expose it.

    Uses the Azure AI Inference chat completions API when SERVERLESS_MODELS is True,
    otherwise the Azure OpenAI Responses API."""
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
        # with_raw_response gives access to HTTP headers (server timing) while
        # still parsing the typed response object.
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
            # Older SDKs without with_raw_response: fall back to a plain call.
            response = client.responses.create(
                model=model,
                instructions=system_prompt,
                input=user_prompt,
                temperature=0.0,
                max_output_tokens=512,
            )
        raw = response.output_text.strip()
    elapsed_s = time.perf_counter() - start
    return _extract_json(raw), elapsed_s, server_s


def analyze_phrase(clients, latest_phrase, full_conversation):
    """Detects the intent of the latest phrase and summarizes the conversation so far.

    When intent and summary share the same endpoint and model, a single combined
    call is made. When they differ (different endpoint or model), two separate
    prompts are issued IN PARALLEL, one per service.

    Returns a dict with 'intent', 'summary' and 'elapsed_s' (wall-clock time), or
    {'error': ...} on failure."""
    user_prompt = _build_user_prompt(latest_phrase, full_conversation)
    try:
        if not USE_SEPARATE_INTENT_SUMMARY:
            # Same endpoint and model: a single combined call.
            if SHOW_DEBUG:
                print(f"[DEBUG] Calling combined model '{AOAI_INTENT_MODEL}' "
                      f"@ {AOAI_INTENT_ENDPOINT} (intent + summary)")
            result, elapsed_s, server_s = _call_model(
                clients["intent"], AOAI_INTENT_MODEL, ANALYSIS_SYSTEM_PROMPT, user_prompt
            )
            result["elapsed_s"] = elapsed_s
            result["server_elapsed_s"] = server_s
            return result

        # Different endpoint/model: run intent and summary in parallel.
        if SHOW_DEBUG:
            print(f"[DEBUG] Calling intent model '{AOAI_INTENT_MODEL}' "
                  f"@ {AOAI_INTENT_ENDPOINT} and summary model '{AOAI_SUMMARY_MODEL}' "
                  f"@ {AOAI_SUMMARY_ENDPOINT} in parallel")
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as executor:
            intent_future = executor.submit(
                _call_model, clients["intent"], AOAI_INTENT_MODEL,
                INTENT_SYSTEM_PROMPT, user_prompt,
            )
            summary_future = executor.submit(
                _call_model, clients["summary"], AOAI_SUMMARY_MODEL,
                SUMMARY_SYSTEM_PROMPT, user_prompt,
            )
            intent_result, intent_elapsed, intent_server = intent_future.result()
            summary_result, summary_elapsed, summary_server = summary_future.result()
        elapsed_s = time.perf_counter() - start
        # Server-side time for the parallel case is the slower of the two calls
        # (they run concurrently), when both services report it.
        if intent_server is not None and summary_server is not None:
            server_s = max(intent_server, summary_server)
        else:
            server_s = None
        return {
            "intent": intent_result.get("intent", "unknown"),
            "sentiment": intent_result.get("sentiment", "unknown"),
            "summary": summary_result.get("summary", ""),
            "elapsed_s": elapsed_s,
            "server_elapsed_s": server_s,
            "intent_elapsed_s": intent_elapsed,
            "summary_elapsed_s": summary_elapsed,
            "intent_server_elapsed_s": intent_server,
            "summary_server_elapsed_s": summary_server,
        }
    except Exception as ex:
        return {"error": str(ex)}


# --------------------------------------------------------------------------- #
# Speech service authentication
# --------------------------------------------------------------------------- #
def get_speech_auth_token():
    """Returns the Speech service auth token in the format aad#{resourceId}#{aadToken}.

    The Entra ID bearer token provider is created once at module level and reused,
    so the underlying DefaultAzureCredential caches the token across calls. This
    lets callers pre-warm the token (e.g. at app start) so a later real call is
    fast instead of paying the Entra ID round-trip again."""
    if not SPEECH_RESOURCE_ID:
        raise Exception("SPEECH_RESOURCE_ID is not configured")
    global _speech_token_provider
    if _speech_token_provider is None:
        _speech_token_provider = get_bearer_token_provider(
            DefaultAzureCredential(),
            "https://cognitiveservices.azure.com/.default"
        )
    aad_token = _speech_token_provider()
    # The Speech service requires the format: aad#{resourceId}#{aadToken}
    return f"aad#{SPEECH_RESOURCE_ID}#{aad_token}"


def warmup_speech_token():
    """Pre-fetches and caches the Speech service Entra ID token so the first real
    WebSocket connection does not pay the token round-trip. Safe to call multiple
    times (the credential caches the token)."""
    try:
        get_speech_auth_token()
    except Exception as ex:
        print(f"[STARTUP] Speech token warm-up skipped: {ex}")


async def get_auth_token():
    """Async wrapper around get_speech_auth_token() for the WebSocket clients."""
    return get_speech_auth_token()


# --------------------------------------------------------------------------- #
# WebSocket protocol message builders
# --------------------------------------------------------------------------- #
def create_speech_config_message(request_id):
    """Creates the initial configuration message."""
    config = {
        "context": {
            "system": {
                "version": "1.0.0"
            },
            "os": {
                "platform": "Python",
                "name": "WebSocket Client",
                "version": "1.0"
            }
        }
    }

    payload = json.dumps(config)
    header = f"X-RequestId:{request_id}\r\n"
    header += f"Content-Type:application/json; charset=utf-8\r\n"
    header += f"Path:speech.config\r\n\r\n"

    return header + payload


def create_speech_context_message(request_id):
    """Creates the speech.context message that enables Semantic segmentation.
    This replicates the message the Speech SDK sends when
    Speech_SegmentationStrategy is set to "Semantic".
    """
    context = {
        "phraseDetection": {
            "mode": "CONVERSATION",
            "language": LANGUAGE,
            "CONVERSATION": {
                "segmentation": {
                    "mode": "Semantic"
                }
            }
        }
    }

    payload = json.dumps(context)
    header = f"X-RequestId:{request_id}\r\n"
    header += f"Content-Type:application/json; charset=utf-8\r\n"
    header += f"Path:speech.context\r\n\r\n"

    return header + payload


def create_audio_message(request_id, audio_data, is_last=False):
    """Creates an audio message with a binary header."""
    # Header in text format
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    header = f"X-RequestId:{request_id}\r\n"
    header += f"X-Timestamp:{timestamp}\r\n"
    header += f"Content-Type:audio/x-wav\r\n"
    header += f"Path:audio\r\n\r\n"

    header_bytes = header.encode('utf-8')
    header_length = len(header_bytes)

    # Build the complete binary message:
    # 2 bytes: header length (big-endian)
    # N bytes: header
    # M bytes: audio data
    message = struct.pack('>H', header_length) + header_bytes + audio_data

    return message
