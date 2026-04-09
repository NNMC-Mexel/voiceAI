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

def _json_default(obj):
    """Handle numpy types that stdlib json can't serialize (numpy.bool_, numpy.float32, etc.)."""
    try:
        import numpy as np
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

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
    """Get audio duration in seconds using ffprobe.
    Falls back to stream duration or decoded frame count if format.duration is missing
    (e.g. for webm recordings from MediaRecorder that lack duration metadata)."""
    # Try format.duration first
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "json", audio_path],
        capture_output=True, text=True, timeout=30,
    )
    try:
        data = json.loads(result.stdout)
        dur = data.get("format", {}).get("duration")
        if dur is not None:
            return float(dur)
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Fallback: decode and count packets (works for webm without duration metadata)
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=duration", "-of",
         "default=noprint_wrappers=1:nokey=1", audio_path],
        capture_output=True, text=True, timeout=30,
    )
    try:
        val = result.stdout.strip()
        if val and val != "N/A":
            return float(val)
    except ValueError:
        pass

    # Last resort: full decode to measure duration
    result = subprocess.run(
        ["ffmpeg", "-i", audio_path, "-f", "null", "-"],
        capture_output=True, text=True, timeout=300,
    )
    m = re.search(r"time=(\d+):(\d+):([\d.]+)", result.stderr)
    if m:
        h, mi, s = float(m.group(1)), float(m.group(2)), float(m.group(3))
        return h * 3600 + mi * 60 + s

    raise RuntimeError(f"Could not determine duration of {audio_path}")


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
    Returns list of temporary chunk file paths (caller must clean up).
    On exception cleans up all created files to avoid leaking tmp space."""
    chunks = []
    chunk_path = None
    try:
        for i in range(len(split_points) - 1):
            start = split_points[i]
            chunk_duration = split_points[i + 1] - start
            fd, chunk_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-i", audio_path,
                     "-ss", f"{start:.3f}", "-t", f"{chunk_duration:.3f}",
                     "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                     chunk_path],
                    capture_output=True, timeout=60, check=False,
                )
            except subprocess.TimeoutExpired:
                logger.warning(f"ffmpeg timeout splitting chunk {i+1}, skipping")
                try:
                    os.unlink(chunk_path)
                except OSError:
                    pass
                chunk_path = None
                continue
            # Verify chunk was created and is non-empty
            if os.path.isfile(chunk_path) and os.path.getsize(chunk_path) > 1000:
                chunks.append(chunk_path)
            else:
                try:
                    os.unlink(chunk_path)
                except OSError:
                    pass
            chunk_path = None
        return chunks
    except Exception:
        # На любой непредвиденной ошибке — чистим всё что успели создать
        if chunk_path and os.path.isfile(chunk_path):
            try:
                os.unlink(chunk_path)
            except OSError:
                pass
        for cp in chunks:
            try:
                os.unlink(cp)
            except OSError:
                pass
        raise


SHORT_AUDIO_SEC = 300  # < 5 мин — короткое, можно держать contextual continuation


def _collect_segments(segments_iter):
    """Drain a faster-whisper segments generator into (text, avg_logprob).
    avg_logprob = duration-weighted mean of segment-level logprobs."""
    parts = []
    weighted = 0.0
    total_dur = 0.0
    for s in segments_iter:
        parts.append(s.text.strip())
        dur = max(0.001, float((s.end or 0) - (s.start or 0)))
        weighted += float(s.avg_logprob or 0) * dur
        total_dur += dur
    text = " ".join(p for p in parts if p)
    avg_logprob = float(weighted / total_dur) if total_dur > 0 else 0.0
    return text, avg_logprob


def transcribe_chunked(audio_path: str, language: str, beam_size: int,
                       initial_prompt: str, hotwords: str = "") -> tuple:
    """Transcribe long audio by splitting into chunks.
    Returns (text, info, n_chunks, chunk_details, avg_logprob)."""
    duration = get_audio_duration(audio_path)

    # Адаптивные параметры по длительности:
    # - короткое аудио (<5 мин): condition_on_previous_text=True для связности контекста
    # - длинное: False, чтобы галлюцинации не распространялись по сегментам
    is_short = duration <= SHORT_AUDIO_SEC

    common_kwargs = dict(
        language=language,
        beam_size=beam_size,
        initial_prompt=initial_prompt,
        # VAD внутри faster-whisper — режет тишину и улучшает сегментацию,
        # без внешнего ffmpeg-чанкинга по тишине
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        # Repetition penalty повышен с 1.2 → 1.3 для длинных медицинских записей
        # с естественными повторами («давление 140, давление 140»)
        repetition_penalty=1.3,
        condition_on_previous_text=is_short,
        temperature=0,
        no_speech_threshold=0.6,
        word_timestamps=True,
    )
    if hotwords:
        common_kwargs["hotwords"] = hotwords

    if duration <= CHUNK_THRESHOLD or not HAS_FFMPEG:
        # Short audio or no ffmpeg — transcribe as-is
        segments, info = model.transcribe(audio_path, **common_kwargs)
        text, avg_lp = _collect_segments(segments)
        return text, info, 1, [{"duration": float(duration), "chars": len(text), "avg_logprob": float(round(avg_lp, 3))}], float(avg_lp)

    # Длинное аудио — оставляем chunking как safety net на случай экстремально
    # длинных записей. Но т.к. включён vad_filter, чанков будет меньше.
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
        segments, info = model.transcribe(audio_path, **common_kwargs)
        text, avg_lp = _collect_segments(segments)
        return text, info, 1, [{"duration": float(duration), "chars": len(text), "avg_logprob": float(round(avg_lp, 3))}], float(avg_lp)

    texts = []
    chunk_details = []
    info = None
    weighted_lp = 0.0
    total_lp_dur = 0.0
    try:
        for i, chunk_path in enumerate(chunk_paths):
            t_chunk = time.time()
            segments, chunk_info = model.transcribe(chunk_path, **common_kwargs)
            chunk_text, chunk_lp = _collect_segments(segments)
            chunk_elapsed = time.time() - t_chunk
            chunk_dur = split_points[i + 1] - split_points[i]

            texts.append(chunk_text)
            chunk_details.append({
                "chunk": i + 1,
                "duration": float(round(chunk_dur, 1)),
                "chars": len(chunk_text),
                "elapsed": round(chunk_elapsed, 2),
                "avg_logprob": float(round(chunk_lp, 3)),
            })
            weighted_lp += chunk_lp * chunk_dur
            total_lp_dur += chunk_dur
            logger.info(
                f"  Chunk {i+1}/{len(chunk_paths)}: "
                f"{chunk_dur:.1f}s → {len(chunk_text)} chars in {chunk_elapsed:.2f}s "
                f"logprob={chunk_lp:.2f}"
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
    avg_lp = float(weighted_lp / total_lp_dur) if total_lp_dur > 0 else 0.0
    return text, info, len(chunk_paths), chunk_details, avg_lp


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
            hotwords = data.get("hotwords", "") or ""

            text, info, n_chunks, chunk_details, avg_logprob = transcribe_chunked(
                audio_path, language=language, beam_size=beam_size,
                initial_prompt=initial_prompt, hotwords=hotwords,
            )
            elapsed = time.time() - t0

            # avg_logprob — индикатор уверенности модели:
            #   > -0.5  отлично
            #   -0.5..-1.0  норма
            #   < -1.0  возможны галлюцинации, требует проверки
            confidence_warning = bool(avg_logprob < -1.0)
            logger.info(
                f"OK  {elapsed:.2f}s | {len(text)} chars | "
                f"lang={info.language} | chunks={n_chunks} | "
                f"avg_logprob={avg_logprob:.2f}"
                f"{'  [LOW CONFIDENCE]' if confidence_warning else ''}"
            )
            self._send_json(200, {
                "text": text,
                "language": info.language,
                "elapsed": elapsed,
                "chunks": n_chunks,
                "chunk_details": chunk_details,
                "avg_logprob": float(round(avg_logprob, 3)),
                "low_confidence": confidence_warning,
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
        # default=_json_default handles numpy types (numpy.bool_, numpy.float32, etc.)
        body = json.dumps(data, ensure_ascii=False, default=_json_default).encode("utf-8")
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
