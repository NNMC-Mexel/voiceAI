#!/usr/bin/env python3
"""Persistent faster-whisper HTTP server for МедДок.

Загружает модель Whisper ОДИН РАЗ при старте.
Устраняет задержку 15-30 сек на загрузку модели при каждом запросе.

Переменные окружения:
  WHISPER_MODEL_PATH    Имя модели или локальный путь  (default: large-v3)
  WHISPER_DEVICE        cuda | cpu                      (default: cuda)
  WHISPER_COMPUTE_TYPE  float16 | int8 | float32        (default: float16)
  WHISPER_DEVICE_INDEX  Индекс GPU                      (default: 0)
  WHISPER_SERVER_PORT   HTTP порт                       (default: 9000)
  WHISPER_SERVER_HOST   Хост для прослушивания          (default: 127.0.0.1)
  WHISPER_BEAM_SIZE     Размер beam search              (default: 5)

Endpoints:
  GET  /health      → {"status":"ok","model":"...","device":"..."}
  POST /transcribe  → body: {"audio_base64":"<base64>","language":"ru"}   # cross-machine
                             {"audio_path":"/abs/path","language":"ru"}    # same-machine only
                    ← {"text":"...","language":"ru","elapsed":1.23}
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
logger = logging.getLogger("whisper-server")

MODEL_PATH   = os.environ.get("WHISPER_MODEL_PATH", "large-v3")
DEVICE       = os.environ.get("WHISPER_DEVICE", "cuda")
COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "float16")
DEVICE_INDEX = int(os.environ.get("WHISPER_DEVICE_INDEX", "0"))
PORT         = int(os.environ.get("WHISPER_SERVER_PORT", "9000"))
HOST         = os.environ.get("WHISPER_SERVER_HOST", "127.0.0.1")
DEFAULT_BEAM = int(os.environ.get("WHISPER_BEAM_SIZE", "5"))

logger.info(f"Loading Whisper model '{MODEL_PATH}' on {DEVICE} ({COMPUTE_TYPE}) ...")
t_start = time.time()

try:
    from faster_whisper import WhisperModel  # type: ignore
    model = WhisperModel(
        MODEL_PATH,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        device_index=DEVICE_INDEX,
    )
except Exception as exc:
    logger.error(f"Failed to load Whisper model: {exc}")
    sys.exit(1)

logger.info(f"Model loaded in {time.time() - t_start:.1f}s — server is ready.")


class WhisperHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "model": MODEL_PATH, "device": DEVICE})
        else:
            self._send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        if self.path == "/transcribe":
            self._handle_transcribe()
        else:
            self._send_json(404, {"error": "Not found"})

    def _handle_transcribe(self) -> None:
        # Parse JSON body
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body)
        except Exception:
            self._send_json(400, {"error": "Invalid JSON body"})
            return

        language  = data.get("language", "ru")
        beam_size = int(data.get("beam_size", DEFAULT_BEAM))

        tmp_path = None
        try:
            if "audio_base64" in data:
                # Основной режим: аудио передаётся base64 (работает cross-machine,
                # например когда бэкенд на Coolify, а Whisper на ПК с GPU)
                try:
                    audio_bytes = base64.b64decode(data["audio_base64"])
                except Exception:
                    self._send_json(400, {"error": "Invalid base64 in audio_base64"})
                    return
                with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
                    tmp.write(audio_bytes)
                    tmp_path = tmp.name
                audio_path = tmp_path
            elif "audio_path" in data:
                # Режим совместимости: путь к файлу (работает только если сервер и клиент
                # на одной машине, например в docker-compose с общим volume)
                try:
                    audio_path = os.path.realpath(data["audio_path"])
                except Exception:
                    self._send_json(400, {"error": "Invalid audio_path"})
                    return
                if not os.path.isfile(audio_path):
                    self._send_json(400, {"error": f"File not found: {audio_path}"})
                    return
            else:
                self._send_json(400, {"error": "Provide audio_base64 or audio_path"})
                return

            # Транскрипция — модель уже загружена, начинаем сразу
            t0 = time.time()
            segments, info = model.transcribe(
                audio_path,
                language=language,
                beam_size=beam_size,
            )
            text = " ".join(s.text.strip() for s in segments)
            elapsed = time.time() - t0

            logger.info(
                f"OK  {elapsed:.2f}s | {len(text)} chars | lang={info.language}"
            )
            self._send_json(200, {
                "text": text,
                "language": info.language,
                "elapsed": elapsed,
            })
        except Exception as exc:
            logger.error(f"Transcription failed: {exc}", exc_info=True)
            self._send_json(500, {"error": str(exc)})
        finally:
            # Удаляем временный файл если он был создан
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
        # Подавляем стандартный access log — используем наш logger
        pass


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Каждый HTTP-запрос обрабатывается в отдельном потоке."""
    daemon_threads = True


if __name__ == "__main__":
    server = ThreadedHTTPServer((HOST, PORT), WhisperHandler)
    logger.info(f"Whisper HTTP server → http://{HOST}:{PORT}")
    logger.info("Endpoints: GET /health   POST /transcribe")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down.")
        server.shutdown()
