# pip install azure-cognitiveservices-speech openai azure-identity python-dotenv
import threading

import azure.cognitiveservices.speech as speech_sdk

from common_functions import (
    SPEECH_REGION, LANGUAGE, TARGET_SAMPLE_RATE,
    SHOW_TIME, INTENTS,
    print_info, print_partial, parse_audio_file_arg,
    load_audio_16k_mono, create_aoai_clients, warmup_aoai, analyze_phrase,
    get_speech_auth_token,
    ResultsCsvWriter,
)

# Optional command-line arguments: audio file to process and an optional CSV file
# (--csv/-o) where the per-segment results are written. If no audio file is given,
# the default audio file is used; if no CSV is given, no CSV is written.
AUDIO_FILE, CSV_PATH = parse_audio_file_arg(
    "Continuous speech transcription with LLM analysis.", with_csv=True
)
csv_writer = ResultsCsvWriter(CSV_PATH) if CSV_PATH else None

# Azure OpenAI configuration (Entra ID authentication)
print("[STARTUP] Initializing services...")
print("[STARTUP] Initializing Azure OpenAI client...")
aoai_clients = create_aoai_clients()
warmup_aoai(aoai_clients)
print("[STARTUP] Azure OpenAI client ready.")


def get_speech_config():
    """Builds the SpeechConfig using Entra ID (Azure AD) authentication and
    enables Semantic segmentation. Replicates the auth used by the WebSocket version."""
    auth_token = get_speech_auth_token()

    speech_config = speech_sdk.SpeechConfig(
        auth_token=auth_token,
        region=SPEECH_REGION,
        speech_recognition_language=LANGUAGE,
    )
    # Enable Semantic segmentation (same behavior as the WebSocket speech.context message)
    speech_config.set_property(speech_sdk.PropertyId.Speech_SegmentationStrategy, "Semantic")
    return speech_config


def speech_recognize_continuous():
    """Performs continuous speech recognition from an audio file using the Speech SDK.
    For each recognized (final) phrase, analyzes it with the LLM."""
    speech_config = get_speech_config()

    # Convert the audio to 16 kHz mono 16-bit PCM and feed it via a push stream,
    # so any WAV (stereo, different sample rate, etc.) can be processed.
    audio_pcm = load_audio_16k_mono(AUDIO_FILE)
    stream_format = speech_sdk.audio.AudioStreamFormat(
        samples_per_second=TARGET_SAMPLE_RATE, bits_per_sample=16, channels=1
    )
    push_stream = speech_sdk.audio.PushAudioInputStream(stream_format)
    audio_config = speech_sdk.AudioConfig(stream=push_stream)
    speech_recognizer = speech_sdk.SpeechRecognizer(speech_config, audio_config)

    # Event to signal that the session has ended
    done = threading.Event()

    # Accumulates all completed phrases of the conversation
    conversation_phrases = []
    # Running summary + 1-based phrase count kept across phrases for
    # INCREMENTAL_SUMMARY / periodic full refresh (ignored otherwise).
    summary_state = {"previous": "", "turn": 0}

    def session_started_cb(evt: speech_sdk.SessionEventArgs):
        print_info(f"[INFO] Session started: {evt.session_id}")

    def recognizing_cb(evt: speech_sdk.SpeechRecognitionEventArgs):
        print_partial(f"[PARTIAL] {evt.result.text}")

    def recognized_cb(evt: speech_sdk.SpeechRecognitionEventArgs):
        if evt.result.reason == speech_sdk.ResultReason.RecognizedSpeech:
            display_text = evt.result.text
            if display_text.strip():
                print("-" * 50)
                print(f"[TRANSCRIPTION] {display_text}")

                # Accumulate the phrase and analyze it with the LLM
                conversation_phrases.append(display_text)
                full_conversation = " ".join(conversation_phrases)
                summary_state["turn"] += 1
                analysis = analyze_phrase(
                    aoai_clients, display_text, full_conversation,
                    summary_state["previous"], summary_state["turn"]
                )
                if analysis and "error" not in analysis:
                    summary_state["previous"] = analysis.get('summary', summary_state["previous"])
                    print(f"[INTENT] {analysis.get('intent', 'unknown')}")
                    print(f"[SENTIMENT] {analysis.get('sentiment', 'unknown')}")
                    print(f"[SUMMARY] {analysis.get('summary', '')}")
                    if SHOW_TIME:
                        print(f"[TIME] Text model call: {analysis.get('elapsed_s', 0):.3f} s")
                    if csv_writer:
                        csv_writer.write_row(display_text, analysis)
                elif analysis and "error" in analysis:
                    print(f"[ERROR] LLM analysis failed: {analysis['error']}")
        elif evt.result.reason == speech_sdk.ResultReason.NoMatch:
            print_info("[INFO] No speech detected in this segment")

    def canceled_cb(evt: speech_sdk.SpeechRecognitionCanceledEventArgs):
        print_info(f"[INFO] Recognition canceled: {evt.reason}")
        if evt.reason == speech_sdk.CancellationReason.Error:
            print(f"[ERROR] {evt.error_details}")
        done.set()

    def session_stopped_cb(evt: speech_sdk.SessionEventArgs):
        print_info("[INFO] Session stopped")
        done.set()

    # Connect callbacks to the events fired by the speech recognizer
    speech_recognizer.session_started.connect(session_started_cb)
    speech_recognizer.recognizing.connect(recognizing_cb)
    speech_recognizer.recognized.connect(recognized_cb)
    speech_recognizer.canceled.connect(canceled_cb)
    speech_recognizer.session_stopped.connect(session_stopped_cb)

    # Start continuous speech recognition
    print("[STARTUP] Starting continuous recognition\n")
    speech_recognizer.start_continuous_recognition()

    # Push the converted audio into the stream, then close it to signal the end.
    push_stream.write(audio_pcm)
    push_stream.close()

    # Wait until the session ends (file fully processed)
    done.wait()

    speech_recognizer.stop_continuous_recognition()
    print_info("[INFO] Process completed")


def main():
    # Prefined Intents
    print(f"Possible intents to detect: {', '.join(INTENTS)}")
    print()

    print("[STARTUP] Connecting to Azure Speech Service...")
    speech_recognize_continuous()


if __name__ == "__main__":
    try:
        main()
    finally:
        if csv_writer:
            csv_writer.close()
            print(f"[INFO] Results written to {CSV_PATH}")
