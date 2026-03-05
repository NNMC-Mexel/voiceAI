#!/usr/bin/env python3
"""Persistent XTTS v2 HTTP server for МедДок.

Загружает модель XTTS v2 ОДИН РАЗ при старте.
Поддерживает клонирование голоса по WAV-семплу или встроенного диктора.

Переменные окружения:
  TTS_MODEL        Имя модели (default: tts_models/multilingual/multi-dataset/xtts_v2)
  TTS_SPEAKER_WAV  Путь к WAV-семплу голоса 6-10 сек (опционально, включает клонирование)
  TTS_SPEAKER      Встроенный диктор, если WAV не указан (default: Ana Florence)
  TTS_SERVER_PORT  HTTP порт (default: 5500)
  TTS_SERVER_HOST  Хост (default: 0.0.0.0)

Endpoints:
  GET  /health  → {"status":"ok","model":"...","speaker":"..."}
  POST /tts     → body: {"text":"...","language":"ru"}
               ← {"audio_base64":"...","format":"wav","elapsed":0.4}
"""

import sys
import json
import os
import base64
import tempfile
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

MODEL_NAME   = os.environ.get("TTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")
SPEAKER_WAV  = os.environ.get("TTS_SPEAKER_WAV", "").strip()
SPEAKER_NAME = os.environ.get("TTS_SPEAKER", "Ana Florence")
PORT         = int(os.environ.get("TTS_SERVER_PORT", "5500"))
HOST         = os.environ.get("TTS_SERVER_HOST", "0.0.0.0")

logger.info(f"Loading TTS model '{MODEL_NAME}' ...")
t_start = time.time()

try:
    from TTS.api import TTS  # type: ignore
    tts = TTS(MODEL_NAME, gpu=True)
except Exception as exc:
    logger.error(f"Failed to load TTS model: {exc}")
    sys.exit(1)

logger.info(f"TTS model loaded in {time.time() - t_start:.1f}s — server is ready.")
if SPEAKER_WAV and os.path.isfile(SPEAKER_WAV):
    logger.info(f"Voice cloning mode: {SPEAKER_WAV}")
else:
    logger.info(f"Built-in speaker mode: {SPEAKER_NAME}")
    if SPEAKER_WAV:
        logger.warning(f"TTS_SPEAKER_WAV set but file not found: {SPEAKER_WAV}")


class TtsHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/health":
            speaker = SPEAKER_WAV if (SPEAKER_WAV and os.path.isfile(SPEAKER_WAV)) else SPEAKER_NAME
            self._send_json(200, {"status": "ok", "model": MODEL_NAME, "speaker": speaker})
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
        language = data.get("language", "ru")

        if not text:
            self._send_json(400, {"error": "text is required"})
            return

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            t0 = time.time()

            if SPEAKER_WAV and os.path.isfile(SPEAKER_WAV):
                tts.tts_to_file(
                    text=text,
                    file_path=tmp_path,
                    speaker_wav=SPEAKER_WAV,
                    language=language,
                )
            else:
                tts.tts_to_file(
                    text=text,
                    file_path=tmp_path,
                    speaker=SPEAKER_NAME,
                    language=language,
                )

            elapsed = time.time() - t0

            with open(tmp_path, "rb") as f:
                audio_bytes = f.read()

            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
            logger.info(f"OK  {elapsed:.2f}s | {len(text)} chars | {len(audio_bytes)} bytes wav")
            self._send_json(200, {
                "audio_base64": audio_b64,
                "format": "wav",
                "elapsed": elapsed,
            })
        except Exception as exc:
            logger.error(f"TTS failed: {exc}", exc_info=True)
            self._send_json(500, {"error": str(exc)})
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

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
