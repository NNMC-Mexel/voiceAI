#!/usr/bin/env python3
"""Persistent GigaAM v3 HTTP server для МедДок.

Загружает модель ОДИН РАЗ при старте.
Эндпоинты совместимы с whisper_server.py — можно переключаться без изменений бэкенда.

Переменные окружения:
  GIGAAM_MODEL       Имя модели: v3_ctc | v3_rnnt | v3_e2e_ctc | v3_e2e_rnnt  (default: v3_ctc)
  GIGAAM_DEVICE      cuda | cpu                                                 (default: cuda)
  GIGAAM_SERVER_PORT HTTP порт                                                  (default: 9001)
  GIGAAM_SERVER_HOST Хост для прослушивания                                     (default: 0.0.0.0)

Endpoints:
  GET  /health      → {"status":"ok","model":"v3_ctc","device":"cuda"}
  POST /transcribe  → body: {"audio_base64":"<base64>"}   или {"audio_path":"/abs/path"}
                   ← {"text":"...","language":"ru","elapsed":1.23}
"""

import sys
import json
import os
import base64
import subprocess
import tempfile
import time
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger("gigaam-server")

MODEL_NAME = os.environ.get("GIGAAM_MODEL", "v3_ctc")
DEVICE     = os.environ.get("GIGAAM_DEVICE", "cuda")
PORT       = int(os.environ.get("GIGAAM_SERVER_PORT", "9001"))
HOST       = os.environ.get("GIGAAM_SERVER_HOST", "0.0.0.0")

logger.info(f"Loading GigaAM model '{MODEL_NAME}' on {DEVICE} ...")
t_start = time.time()

try:
    import gigaam
    model = gigaam.load_model(MODEL_NAME, device=DEVICE)
    model.eval()
except Exception as exc:
    logger.error(f"Failed to load GigaAM model: {exc}")
    sys.exit(1)

logger.info(f"Model loaded in {time.time() - t_start:.1f}s — server is ready.")


def _check_ffmpeg() -> bool:
    try:
        subprocess.run(["ffprobe", "-version"], capture_output=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


HAS_FFMPEG = _check_ffmpeg()
if not HAS_FFMPEG:
    logger.warning("ffmpeg NOT found — audio conversion may fail for non-WAV input")


def convert_to_wav(input_path: str) -> str:
    """Convert any audio file to 16kHz mono WAV. Returns path to temp WAV file."""
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    subprocess.run(
        ["ffmpeg", "-y", "-i", input_path,
         "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav_path],
        capture_output=True, timeout=120, check=True,
    )
    return wav_path


def transcribe_audio(wav_path: str) -> str:
    """Transcribe WAV file using GigaAM. Chooses short/longform automatically."""
    from gigaam.model import LONGFORM_THRESHOLD, SAMPLE_RATE
    import soundfile as sf

    info = sf.info(wav_path)
    n_samples = int(info.duration * info.samplerate)

    if n_samples > LONGFORM_THRESHOLD:
        logger.info(f"Audio {info.duration:.1f}s > {LONGFORM_THRESHOLD/SAMPLE_RATE:.0f}s — using transcribe_longform")
        try:
            result = model.transcribe_longform(wav_path)
            return " ".join(seg.text.strip() for seg in result.segments if seg.text.strip())
        except ModuleNotFoundError as exc:
            if exc.name != "pyannote":
                raise
            logger.warning("pyannote is not installed; falling back to fixed 24s GigaAM chunks")
            return transcribe_audio_in_fixed_chunks(wav_path)
    else:
        result = model.transcribe(wav_path)
        return result.text.strip()


def transcribe_audio_in_fixed_chunks(wav_path: str) -> str:
    """Fallback for long audio when pyannote-based transcribe_longform is unavailable."""
    chunk_seconds = int(os.environ.get("GIGAAM_FALLBACK_CHUNK_SECONDS", "24"))
    chunk_seconds = max(5, min(chunk_seconds, 24))

    with tempfile.TemporaryDirectory(prefix="gigaam_chunks_") as chunk_dir:
        pattern = os.path.join(chunk_dir, "chunk_%04d.wav")
        subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", wav_path,
                "-f", "segment",
                "-segment_time", str(chunk_seconds),
                "-reset_timestamps", "1",
                "-ar", "16000",
                "-ac", "1",
                "-c:a", "pcm_s16le",
                pattern,
            ],
            capture_output=True,
            timeout=120,
            check=True,
        )

        chunks = [
            os.path.join(chunk_dir, name)
            for name in sorted(os.listdir(chunk_dir))
            if name.lower().endswith(".wav")
        ]
        if not chunks:
            raise RuntimeError("ffmpeg produced no GigaAM fallback chunks")

        parts = []
        logger.info(f"Fallback chunking produced {len(chunks)} chunks ({chunk_seconds}s each)")
        for index, chunk_path in enumerate(chunks, start=1):
            result = model.transcribe(chunk_path)
            text = result.text.strip()
            logger.info(f"chunk {index}/{len(chunks)}: {len(text)} chars")
            if text:
                parts.append(text)

        return " ".join(parts).strip()


class GigaAMHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "model": MODEL_NAME, "device": DEVICE})
        else:
            self._send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        if self.path == "/transcribe":
            self._handle_transcribe()
        else:
            self._send_json(404, {"error": "Not found"})

    def _handle_transcribe(self) -> None:
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body)
        except Exception:
            self._send_json(400, {"error": "Invalid JSON body"})
            return

        tmp_input = None
        tmp_wav   = None
        try:
            if "audio_base64" in data:
                try:
                    audio_bytes = base64.b64decode(data["audio_base64"])
                except Exception:
                    self._send_json(400, {"error": "Invalid base64 in audio_base64"})
                    return
                with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
                    f.write(audio_bytes)
                    tmp_input = f.name
                audio_path = tmp_input
            elif "audio_path" in data:
                audio_path = os.path.realpath(data["audio_path"])
                if not os.path.isfile(audio_path):
                    self._send_json(400, {"error": f"File not found: {audio_path}"})
                    return
            else:
                self._send_json(400, {"error": "Provide audio_base64 or audio_path"})
                return

            # Convert to 16kHz WAV (GigaAM требует WAV)
            t0 = time.time()
            try:
                tmp_wav = convert_to_wav(audio_path)
            except Exception as exc:
                self._send_json(500, {"error": f"Audio conversion failed: {exc}"})
                return

            text = transcribe_audio(tmp_wav)
            elapsed = time.time() - t0

            logger.info(f"OK  {elapsed:.2f}s | {len(text)} chars | model={MODEL_NAME}")
            self._send_json(200, {
                "text": text,
                "language": "ru",
                "elapsed": elapsed,
                "chunks": 1,
                "chunk_details": [],
                "avg_logprob": 0.0,
                "low_confidence": False,
            })
        except Exception as exc:
            logger.error(f"Transcription failed: {exc}", exc_info=True)
            self._send_json(500, {"error": str(exc)})
        finally:
            for p in [tmp_input, tmp_wav]:
                if p and os.path.exists(p):
                    try:
                        os.unlink(p)
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
    daemon_threads = True


if __name__ == "__main__":
    server = ThreadedHTTPServer((HOST, PORT), GigaAMHandler)
    logger.info(f"GigaAM HTTP server → http://{HOST}:{PORT}")
    logger.info("Endpoints: GET /health   POST /transcribe")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down.")
        server.shutdown()
