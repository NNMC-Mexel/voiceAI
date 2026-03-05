#!/usr/bin/env python3
"""Persistent Silero TTS HTTP server for МедДок.

Загружает модель Silero ОДИН РАЗ при старте.
Работает офлайн на GPU (PyTorch CUDA).

Установка (один раз):
  pip install torch --index-url https://download.pytorch.org/whl/cu128
  pip install soundfile omegaconf

Переменные окружения:
  TTS_SPEAKER      Голос: xenia|kseniya|baya (женские) aidar|eugene (мужские)
                   (default: xenia — наиболее естественный)
  TTS_SAMPLE_RATE  8000 | 24000 | 48000 (default: 48000)
  TTS_SERVER_PORT  HTTP порт (default: 5500)
  TTS_SERVER_HOST  Хост (default: 0.0.0.0)

Endpoints:
  GET  /health  → {"status":"ok","speaker":"...","sample_rate":48000}
  POST /tts     → body: {"text":"..."}
               ← {"audio_base64":"...","format":"wav","elapsed":0.3}
"""

import sys
import json
import os
import base64
import io
import re
import time
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger("tts-server")

SPEAKER     = os.environ.get("TTS_SPEAKER", "xenia")
SAMPLE_RATE = int(os.environ.get("TTS_SAMPLE_RATE", "48000"))
PORT        = int(os.environ.get("TTS_SERVER_PORT", "5500"))
HOST        = os.environ.get("TTS_SERVER_HOST", "0.0.0.0")

logger.info("Loading Silero TTS model ...")
t_start = time.time()

try:
    import torch
    import soundfile as sf  # type: ignore
except ImportError as exc:
    logger.error(f"Missing dependency: {exc}")
    logger.error("Run: pip install torch --index-url https://download.pytorch.org/whl/cu128")
    logger.error("     pip install soundfile omegaconf")
    sys.exit(1)

try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = torch.hub.load(  # type: ignore
        repo_or_dir="snakers4/silero-models",
        model="silero_tts",
        language="ru",
        speaker="v3_1_ru",
        trust_repo=True,
    )
    model.to(device)
except Exception as exc:
    logger.error(f"Failed to load Silero model: {exc}")
    sys.exit(1)

logger.info(f"Silero TTS loaded in {time.time() - t_start:.1f}s — server is ready.")
logger.info(f"Speaker: {SPEAKER}  |  Sample rate: {SAMPLE_RATE}  |  Device: {device}")


def _split_text(text: str, max_len: int = 500) -> list[str]:
    """Split long text into chunks at sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) + 1 <= max_len:
            current = (current + " " + sent).strip() if current else sent
        else:
            if current:
                chunks.append(current)
            # Single sentence longer than max_len — split at comma/space
            if len(sent) > max_len:
                for i in range(0, len(sent), max_len):
                    chunks.append(sent[i : i + max_len])
            else:
                current = sent
    if current:
        chunks.append(current)
    return chunks or [text[:max_len]]


def _synthesize_wav(text: str) -> bytes:
    chunks = _split_text(text)
    parts = []
    for chunk in chunks:
        audio = model.apply_tts(  # type: ignore
            text=chunk,
            speaker=SPEAKER,
            sample_rate=SAMPLE_RATE,
        )
        parts.append(audio)

    full_audio = torch.cat(parts) if len(parts) > 1 else parts[0]

    buf = io.BytesIO()
    sf.write(buf, full_audio.numpy(), SAMPLE_RATE, format="wav", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


class TtsHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(200, {
                "status": "ok",
                "speaker": SPEAKER,
                "sample_rate": SAMPLE_RATE,
                "device": str(device),
            })
        else:
            self._send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        if self.path == "/tts":
            self._handle_tts()
        else:
            self._send_json(404, {"error": "Not found"})

    def _handle_tts(self) -> None:
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body)
        except Exception:
            self._send_json(400, {"error": "Invalid JSON body"})
            return

        text = data.get("text", "").strip()
        if not text:
            self._send_json(400, {"error": "text is required"})
            return

        try:
            t0 = time.time()
            wav_bytes = _synthesize_wav(text)
            elapsed = time.time() - t0

            audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
            logger.info(f"OK  {elapsed:.2f}s | {len(text)} chars | {len(wav_bytes)} bytes wav")
            self._send_json(200, {
                "audio_base64": audio_b64,
                "format": "wav",
                "elapsed": elapsed,
            })
        except Exception as exc:
            logger.error(f"TTS failed: {exc}", exc_info=True)
            self._send_json(500, {"error": str(exc)})

    def _send_json(self, code: int, data: dict) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: object) -> None:
        pass


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Каждый HTTP-запрос обрабатывается в отдельном потоке."""
    daemon_threads = True


if __name__ == "__main__":
    server = ThreadedHTTPServer((HOST, PORT), TtsHandler)
    logger.info(f"TTS HTTP server → http://{HOST}:{PORT}")
    logger.info("Endpoints: GET /health   POST /tts")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down.")
        server.shutdown()
