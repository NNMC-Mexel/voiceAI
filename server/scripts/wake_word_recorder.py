#!/usr/bin/env python3
"""
Wake-word recorder for VoiceMed.

Always-listening mode with offline wake-word detection (Vosk).
Doctor says "Джарвис" to start recording, "стоп Джарвис" to stop.
Recorded fragment is sent to /api/process for transcription + structuring.

Usage:
    python wake_word_recorder.py --model-path ./vosk-model-small-ru-0.22 --api-url http://localhost:3000
"""

import argparse
import collections
import json
import logging
import os
import struct
import sys
import tempfile
import threading
import time
import wave
from enum import Enum
from typing import Optional

import requests
import sounddevice as sd
from vosk import Model, KaldiRecognizer

# ─── Constants ───────────────────────────────────────────────────────────────

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_BLOCK_SIZE = 4000  # ~250ms at 16kHz
DEFAULT_WAKE_PHRASES = "джарвис"
DEFAULT_STOP_PHRASES = "стоп джарвис,конец записи"
DEFAULT_MAX_DURATION = 120  # seconds
PRE_BUFFER_SECONDS = 2  # keep last N seconds in memory

# ─── State machine ───────────────────────────────────────────────────────────

class State(Enum):
    IDLE = "idle"
    RECORDING = "recording"
    UPLOADING = "uploading"


# ─── Logger setup ────────────────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    logger = logging.getLogger("wake_word_recorder")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)s  %(message)s", datefmt="%H:%M:%S"
    ))
    logger.addHandler(handler)
    return logger


log = setup_logging()

# ─── Auth helper ─────────────────────────────────────────────────────────────

def obtain_token(api_url: str, password: str) -> str:
    """Login via /api/auth/login and return Bearer token."""
    url = f"{api_url.rstrip('/')}/api/auth/login"
    log.info("Logging in to %s ...", url)
    resp = requests.post(url, json={"password": password}, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("success"):
        raise RuntimeError(f"Auth failed: {data}")
    token = data["token"]
    log.info("Auth OK, token obtained.")
    return token


# ─── Upload helper ───────────────────────────────────────────────────────────

def upload_audio(api_url: str, wav_path: str, token: Optional[str]) -> dict:
    """Send WAV file to /api/process (multipart) and return response JSON."""
    url = f"{api_url.rstrip('/')}/api/process"
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    with open(wav_path, "rb") as f:
        files = {"file": (os.path.basename(wav_path), f, "audio/wav")}
        resp = requests.post(url, files=files, headers=headers, timeout=300)

    resp.raise_for_status()
    return resp.json()


# ─── Core recorder ───────────────────────────────────────────────────────────

class WakeWordRecorder:
    def __init__(self, args: argparse.Namespace):
        self.api_url: str = args.api_url
        self.token: Optional[str] = args.token
        self.auth_password: Optional[str] = args.auth_password
        self.sample_rate: int = args.sample_rate
        self.block_size: int = args.block_size
        self.max_duration: int = args.max_duration

        self.wake_phrases: list[str] = [
            p.strip().lower() for p in args.wake_phrases.split(",") if p.strip()
        ]
        self.stop_phrases: list[str] = [
            p.strip().lower() for p in args.stop_phrases.split(",") if p.strip()
        ]

        self.state = State.IDLE
        self.recording_frames: list[bytes] = []
        self.recording_start: float = 0.0

        # Pre-buffer: ring buffer of last N seconds
        pre_buffer_blocks = int(PRE_BUFFER_SECONDS * self.sample_rate / self.block_size)
        self.pre_buffer: collections.deque[bytes] = collections.deque(maxlen=max(pre_buffer_blocks, 1))

        # Load Vosk model
        log.info("Loading Vosk model from: %s", args.model_path)
        if not os.path.isdir(args.model_path):
            log.error("Model directory not found: %s", args.model_path)
            sys.exit(1)
        self.model = Model(args.model_path)
        self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
        self.recognizer.SetWords(False)
        log.info("Vosk model loaded OK.")

        # Authenticate if needed
        if not self.token and self.auth_password:
            try:
                self.token = obtain_token(self.api_url, self.auth_password)
            except Exception as e:
                log.warning("Auth failed, will retry on upload: %s", e)

    # ─── State transitions ───────────────────────────────────────────────

    def _set_state(self, new_state: State):
        old = self.state
        self.state = new_state
        log.info("State: %s -> %s", old.value, new_state.value)

    # ─── Phrase matching ─────────────────────────────────────────────────

    def _text_contains(self, text: str, phrases: list[str]) -> bool:
        """Check if any phrase appears in the recognized text."""
        text_lower = text.lower().strip()
        if not text_lower:
            return False
        for phrase in phrases:
            if phrase in text_lower:
                return True
        return False

    # ─── Audio callback ──────────────────────────────────────────────────

    def _audio_callback(self, indata, frames, time_info, status):
        """Called by sounddevice for each audio block."""
        if status:
            log.warning("Audio status: %s", status)

        # Convert float32 -> int16 PCM bytes
        raw = b"".join(struct.pack("<h", max(-32768, min(32767, int(s * 32767)))) for s in indata[:, 0])

        if self.state == State.IDLE:
            # Feed to Vosk for wake-word detection; keep pre-buffer
            self.pre_buffer.append(raw)
            if self.recognizer.AcceptWaveform(raw):
                result = json.loads(self.recognizer.Result())
                text = result.get("text", "")
                if text:
                    log.debug("Vosk (final): %s", text)
                    if self._text_contains(text, self.wake_phrases):
                        self._start_recording()
            else:
                partial = json.loads(self.recognizer.PartialResult())
                text = partial.get("partial", "")
                if text:
                    log.debug("Vosk (partial): %s", text)
                    if self._text_contains(text, self.wake_phrases):
                        self._start_recording()

        elif self.state == State.RECORDING:
            self.recording_frames.append(raw)

            # Check max duration
            elapsed = time.time() - self.recording_start
            if elapsed >= self.max_duration:
                log.warning("Max recording duration reached (%ds), stopping.", self.max_duration)
                self._stop_recording()
                return

            # Feed to Vosk for stop-phrase detection
            if self.recognizer.AcceptWaveform(raw):
                result = json.loads(self.recognizer.Result())
                text = result.get("text", "")
                if text:
                    log.debug("Vosk (final): %s", text)
                    if self._text_contains(text, self.stop_phrases):
                        self._stop_recording()
            else:
                partial = json.loads(self.recognizer.PartialResult())
                text = partial.get("partial", "")
                if text:
                    log.debug("Vosk (partial): %s", text)
                    if self._text_contains(text, self.stop_phrases):
                        self._stop_recording()

    # ─── Start / stop recording ──────────────────────────────────────────

    def _start_recording(self):
        """Transition to recording state."""
        self._set_state(State.RECORDING)
        self.recording_start = time.time()
        # Flush pre-buffer into recording
        self.recording_frames = list(self.pre_buffer)
        self.pre_buffer.clear()
        # Reset recognizer for clean stop-phrase detection
        self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
        self.recognizer.SetWords(False)
        log.info("🎙  Recording started (pre-buffer: %.1fs)", PRE_BUFFER_SECONDS)

    def _stop_recording(self):
        """Transition to uploading, save WAV, send to API."""
        self._set_state(State.UPLOADING)
        duration = time.time() - self.recording_start
        log.info("Recording stopped. Duration: %.1fs, frames: %d", duration, len(self.recording_frames))

        # Reset recognizer for next wake-word detection
        self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
        self.recognizer.SetWords(False)

        frames = self.recording_frames
        self.recording_frames = []

        # Upload in background thread so audio callback keeps working
        threading.Thread(target=self._save_and_upload, args=(frames,), daemon=True).start()

    def _save_and_upload(self, frames: list[bytes]):
        """Save frames as WAV and upload to /api/process."""
        wav_path = None
        try:
            # Write WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, prefix="voicemed_") as tmp:
                wav_path = tmp.name

            with wave.open(wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.sample_rate)
                wf.writeframes(b"".join(frames))

            file_size = os.path.getsize(wav_path)
            log.info("Saved WAV: %s (%.1f KB)", wav_path, file_size / 1024)

            # Re-auth if token missing
            if not self.token and self.auth_password:
                try:
                    self.token = obtain_token(self.api_url, self.auth_password)
                except Exception as e:
                    log.error("Auth failed: %s", e)

            # Upload
            log.info("Uploading to %s/api/process ...", self.api_url)
            result = upload_audio(self.api_url, wav_path, self.token)

            if result.get("success"):
                tx = result.get("transcription", {})
                doc = result.get("document", {})
                log.info("✅ Processing OK! Transcription length: %d chars",
                         len(tx.get("text", "")))
                # Print patient name if found
                patient = doc.get("patientFullName") or doc.get("patient", "")
                if patient:
                    log.info("   Patient: %s", patient)
                complaints = doc.get("complaints", "")
                if complaints:
                    log.info("   Complaints: %s", complaints[:100])
            else:
                log.error("Server returned error: %s", result.get("error", result))

        except requests.exceptions.ConnectionError as e:
            log.error("Connection error (server down?): %s", e)
        except requests.exceptions.Timeout:
            log.error("Upload timed out (server overloaded?)")
        except requests.exceptions.HTTPError as e:
            log.error("HTTP error: %s", e)
            if e.response is not None:
                log.error("Response: %s", e.response.text[:500])
        except Exception as e:
            log.error("Unexpected error during upload: %s", e, exc_info=True)
        finally:
            # Cleanup temp file
            if wav_path and os.path.exists(wav_path):
                try:
                    os.unlink(wav_path)
                except OSError:
                    pass
            self._set_state(State.IDLE)
            log.info("Listening for wake phrase: %s", ", ".join(self.wake_phrases))

    # ─── Main loop ───────────────────────────────────────────────────────

    def run(self):
        """Start the always-listening loop."""
        log.info("=" * 60)
        log.info("VoiceMed Wake-Word Recorder")
        log.info("=" * 60)
        log.info("API URL:      %s", self.api_url)
        log.info("Sample rate:  %d Hz", self.sample_rate)
        log.info("Block size:   %d samples", self.block_size)
        log.info("Pre-buffer:   %ds", PRE_BUFFER_SECONDS)
        log.info("Max duration: %ds", self.max_duration)
        log.info("Wake phrases: %s", self.wake_phrases)
        log.info("Stop phrases: %s", self.stop_phrases)
        log.info("Auth:         %s", "token" if self.token else ("password" if self.auth_password else "none"))
        log.info("-" * 60)
        log.info("Listening for wake phrase: %s", ", ".join(self.wake_phrases))
        log.info("Press Ctrl+C to exit.")
        log.info("")

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                channels=1,
                dtype="float32",
                callback=self._audio_callback,
            ):
                while True:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            log.info("\nExiting.")
        except Exception as e:
            log.error("Audio stream error: %s", e, exc_info=True)
            sys.exit(1)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="VoiceMed wake-word recorder (Vosk + sounddevice)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python wake_word_recorder.py --model-path ./vosk-model-small-ru-0.22 --api-url http://localhost:3000
  python wake_word_recorder.py --model-path ./vosk-model-small-ru-0.22 --api-url http://192.168.1.10:3000 --auth-password secret123
  python wake_word_recorder.py --model-path ./vosk-model-small-ru-0.22 --api-url http://localhost:3000 --wake-phrases "джарвис,начать запись" --stop-phrases "стоп джарвис,конец записи"
        """,
    )
    parser.add_argument("--model-path", required=True, help="Path to Vosk model directory")
    parser.add_argument("--api-url", required=True, help="VoiceMed API base URL (e.g. http://localhost:3000)")
    parser.add_argument("--token", default=None, help="Bearer token for API auth")
    parser.add_argument("--auth-password", default=None, help="Password for /api/auth/login (if no token)")
    parser.add_argument("--wake-phrases", default=DEFAULT_WAKE_PHRASES,
                        help=f"Comma-separated wake phrases (default: {DEFAULT_WAKE_PHRASES})")
    parser.add_argument("--stop-phrases", default=DEFAULT_STOP_PHRASES,
                        help=f"Comma-separated stop phrases (default: {DEFAULT_STOP_PHRASES})")
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE,
                        help=f"Audio sample rate in Hz (default: {DEFAULT_SAMPLE_RATE})")
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE,
                        help=f"Audio block size in samples (default: {DEFAULT_BLOCK_SIZE})")
    parser.add_argument("--max-duration", type=int, default=DEFAULT_MAX_DURATION,
                        help=f"Max recording duration in seconds (default: {DEFAULT_MAX_DURATION})")

    args = parser.parse_args()
    recorder = WakeWordRecorder(args)
    recorder.run()


if __name__ == "__main__":
    main()
