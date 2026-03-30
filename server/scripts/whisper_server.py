#!/usr/bin/env python3
"""Persistent faster-whisper HTTP server for МедДок.

Загружает модель Whisper ОДИН РАЗ при старте.
Устраняет задержку 15-30 сек на загрузку модели при каждом запросе.
Поддерживает автоматический chunking длинных записей для уменьшения галлюцинаций.

Переменные окружения:
  WHISPER_MODEL_PATH    Имя модели или локальный путь  (default: large-v3)
  WHISPER_DEVICE        cuda | cpu                      (default: cuda)
  WHISPER_COMPUTE_TYPE  float16 | int8 | float32        (default: float16)
  WHISPER_DEVICE_INDEX  Индекс GPU                      (default: 0)
  WHISPER_SERVER_PORT   HTTP порт                       (default: 9000)
  WHISPER_SERVER_HOST   Хост для прослушивания          (default: 127.0.0.1)
  WHISPER_BEAM_SIZE     Размер beam search              (default: 5)
  WHISPER_CHUNK_THRESHOLD  Порог в секундах для активации chunking (default: 40)
  WHISPER_CHUNK_TARGET     Целевая длина куска в секундах          (default: 30)

Endpoints:
  GET  /health      → {"status":"ok","model":"...","device":"..."}
  POST /transcribe  → body: {"audio_base64":"<base64>","language":"ru"}   # cross-machine
                             {"audio_path":"/abs/path","language":"ru"}    # same-machine only
                    ← {"text":"...","language":"ru","elapsed":1.23}
"""

import sys
import json
import os
import re
import base64
import subprocess
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
DEFAULT_BEAM    = int(os.environ.get("WHISPER_BEAM_SIZE", "5"))
CHUNK_THRESHOLD = int(os.environ.get("WHISPER_CHUNK_THRESHOLD", "40"))
CHUNK_TARGET    = int(os.environ.get("WHISPER_CHUNK_TARGET", "30"))
INITIAL_PROMPT  = os.environ.get(
    "WHISPER_INITIAL_PROMPT",
    "Пациент. Жалобы. Анамнез. Диагноз. Рекомендации. Объективно. АД мм рт.ст. ЧСС уд/мин. "
    "SpO2. ЧД. ИМТ. МРТ. КТ. УЗИ. ЭКГ. ОАК. ОАМ. СОЭ. СРБ. "
    "гипертоническая болезнь, сахарный диабет, ишемическая болезнь сердца, ХОБЛ, "
    "аускультация, перкуссия, пальпация, хрипы, одышка, тахикардия, брадикардия, "
    "амлодипин, метформин, омепразол, аспирин, метопролол, лизиноприл, аторвастатин.",
)

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


# ─── Audio Chunking ──────────────────────────────────────────────────────────

def _check_ffmpeg() -> bool:
    """Check if ffmpeg/ffprobe are available."""
    try:
        subprocess.run(["ffprobe", "-version"], capture_output=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False

HAS_FFMPEG = _check_ffmpeg()
if HAS_FFMPEG:
    logger.info(f"ffmpeg found — chunking enabled (threshold={CHUNK_THRESHOLD}s, target={CHUNK_TARGET}s)")
else:
    logger.warning("ffmpeg NOT found — chunking disabled, long audio may hallucinate")


def get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds using ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "json", audio_path],
        capture_output=True, text=True, timeout=30,
    )
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


def detect_silences(audio_path: str, noise_db: int = -30, min_silence_sec: float = 0.3) -> list:
    """Detect silence intervals using ffmpeg silencedetect filter.
    Returns list of {'start': float, 'end': float, 'mid': float}."""
    result = subprocess.run(
        ["ffmpeg", "-i", audio_path, "-af",
         f"silencedetect=noise={noise_db}dB:d={min_silence_sec}",
         "-f", "null", "-"],
        capture_output=True, text=True, timeout=120,
    )
    silences = []
    current: dict = {}
    for line in result.stderr.split("\n"):
        if "silence_start:" in line:
            m = re.search(r"silence_start:\s*([\d.]+)", line)
            if m:
                current = {"start": float(m.group(1))}
        elif "silence_end:" in line and current:
            m = re.search(r"silence_end:\s*([\d.]+)", line)
            if m:
                current["end"] = float(m.group(1))
                current["mid"] = (current["start"] + current["end"]) / 2
                silences.append(current)
                current = {}
    return silences


def compute_split_points(duration: float, silences: list, target_sec: int = 30) -> list:
    """Find optimal split points at silence boundaries near target chunk durations.
    Returns sorted list of time points (always starts with 0, ends with duration)."""
    if duration <= target_sec * 1.5:
        return [0.0, duration]

    split_points = [0.0]
    current_pos = 0.0

    while current_pos + target_sec * 1.3 < duration:
        target = current_pos + target_sec
        # Find the silence nearest to the target point
        best_mid = None
        best_distance = float("inf")

        for s in silences:
            mid = s["mid"]
            # Only consider silences after at least 10s from current position
            if mid <= current_pos + 10:
                continue
            # Only consider silences before 2x target from current position
            if mid > current_pos + target_sec * 2:
                continue
            dist = abs(mid - target)
            if dist < best_distance:
                best_distance = dist
                best_mid = mid

        if best_mid is not None and best_distance < target_sec * 0.6:
            split_point = best_mid
        else:
            # No good silence found — split at target anyway
            split_point = target

        split_points.append(split_point)
        current_pos = split_point

    split_points.append(duration)
    return split_points


def split_audio(audio_path: str, split_points: list) -> list:
    """Split audio file at given time points using ffmpeg.
    Returns list of temporary chunk file paths (caller must clean up)."""
    chunks = []
    for i in range(len(split_points) - 1):
        start = split_points[i]
        chunk_duration = split_points[i + 1] - start
        fd, chunk_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path,
             "-ss", f"{start:.3f}", "-t", f"{chunk_duration:.3f}",
             "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
             chunk_path],
            capture_output=True, timeout=60,
        )
        # Verify chunk was created and is non-empty
        if os.path.isfile(chunk_path) and os.path.getsize(chunk_path) > 1000:
            chunks.append(chunk_path)
        else:
            # Cleanup failed chunk
            try:
                os.unlink(chunk_path)
            except OSError:
                pass
    return chunks


def transcribe_chunked(audio_path: str, language: str, beam_size: int,
                       initial_prompt: str) -> tuple:
    """Transcribe long audio by splitting into chunks.
    Returns (text, info, n_chunks, chunk_details)."""
    duration = get_audio_duration(audio_path)

    if duration <= CHUNK_THRESHOLD or not HAS_FFMPEG:
        # Short audio or no ffmpeg — transcribe as-is
        segments, info = model.transcribe(
            audio_path, language=language, beam_size=beam_size,
            initial_prompt=initial_prompt,
        )
        text = " ".join(s.text.strip() for s in segments)
        return text, info, 1, [{"duration": duration, "chars": len(text)}]

    # Long audio — chunk it
    logger.info(f"Audio is {duration:.1f}s (>{CHUNK_THRESHOLD}s) — activating chunking")
    silences = detect_silences(audio_path)
    logger.info(f"Found {len(silences)} silence intervals")

    split_points = compute_split_points(duration, silences, CHUNK_TARGET)
    n_chunks = len(split_points) - 1
    logger.info(f"Splitting into {n_chunks} chunks at: {[f'{p:.1f}s' for p in split_points]}")

    chunk_paths = split_audio(audio_path, split_points)
    if not chunk_paths:
        # Fallback: split failed, transcribe whole file
        logger.warning("Chunk splitting failed — falling back to whole-file transcription")
        segments, info = model.transcribe(
            audio_path, language=language, beam_size=beam_size,
            initial_prompt=initial_prompt,
        )
        text = " ".join(s.text.strip() for s in segments)
        return text, info, 1, [{"duration": duration, "chars": len(text)}]

    texts = []
    chunk_details = []
    info = None
    try:
        for i, chunk_path in enumerate(chunk_paths):
            t_chunk = time.time()
            segments, chunk_info = model.transcribe(
                chunk_path, language=language, beam_size=beam_size,
                initial_prompt=initial_prompt,
            )
            chunk_text = " ".join(s.text.strip() for s in segments)
            chunk_elapsed = time.time() - t_chunk
            chunk_dur = split_points[i + 1] - split_points[i]

            texts.append(chunk_text)
            chunk_details.append({
                "chunk": i + 1,
                "duration": round(chunk_dur, 1),
                "chars": len(chunk_text),
                "elapsed": round(chunk_elapsed, 2),
            })
            logger.info(
                f"  Chunk {i+1}/{len(chunk_paths)}: "
                f"{chunk_dur:.1f}s → {len(chunk_text)} chars in {chunk_elapsed:.2f}s"
            )
            if info is None:
                info = chunk_info
    finally:
        # Cleanup chunk files
        for cp in chunk_paths:
            try:
                os.unlink(cp)
            except OSError:
                pass

    text = " ".join(t for t in texts if t)
    return text, info, len(chunk_paths), chunk_details


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
            initial_prompt = data.get("initial_prompt", INITIAL_PROMPT) or INITIAL_PROMPT

            text, info, n_chunks, chunk_details = transcribe_chunked(
                audio_path, language=language, beam_size=beam_size,
                initial_prompt=initial_prompt,
            )
            elapsed = time.time() - t0

            logger.info(
                f"OK  {elapsed:.2f}s | {len(text)} chars | "
                f"lang={info.language} | chunks={n_chunks}"
            )
            self._send_json(200, {
                "text": text,
                "language": info.language,
                "elapsed": elapsed,
                "chunks": n_chunks,
                "chunk_details": chunk_details,
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
