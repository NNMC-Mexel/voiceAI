#!/usr/bin/env python3
"""Test chunking logic locally: split audio → send each chunk to Whisper server → compare.

Usage:
  python test_chunking.py <audio_file> [whisper_server_url]
"""

import sys
import json
import os
import re
import subprocess
import tempfile
import time
import base64
import urllib.request

WHISPER_SERVER = os.environ.get("WHISPER_SERVER_URL", "http://192.168.41.161:9000")
CHUNK_THRESHOLD = 40  # seconds
CHUNK_TARGET = 30     # seconds

INITIAL_PROMPT = (
    "Консультация кардиолога. Риск падения по шкале Морзе. Оценка боли. "
    "АД, ЧСС, ИМТ, SpO2, ЭКГ, ЭхоКГ, ХМЭКГ, КАГ, СМАД, УЗДГ БЦА, ОАК, ОАМ, ProBNP, "
    "ПМЖВ, ПКА, ОВ ЛКА, РСДЛА, ФВ, КДО, КСО, КДР, КСР, МЖП, "
    "NYHA, CCS, EHRA, CHADS2 VASc, МКБ-10, "
    "ИБС, ХСН, СССУ, НРС, ОНМК, ДГПЖ, АИТ, ДН, ОКС, ТЭЛА, "
    "фибрилляция предсердий, стенокардия напряжения, артериальная гипертензия, "
    "дилатационная кардиомиопатия, трикуспидальная регургитация, митральная регургитация, "
    "кардиоресинхронизирующее устройство, CRT-D, кардиовертер-дефибриллятор, "
    "Medtronic, Quadra Assura, Brava Quad, DDDR, VVIR, "
    "бисопролол, конкор, ксарелто, дигоксин, джардинс, форсига, "
    "мм.рт.ст., уд/мин, мкмоль/л, ммоль/л, г/л"
)

# ─── ffmpeg helpers (same as whisper_server.py) ─────────────────────────────

def find_ffmpeg():
    """Find ffmpeg/ffprobe executables."""
    # Try PATH first
    try:
        subprocess.run(["ffprobe", "-version"], capture_output=True, timeout=5)
        return "ffprobe", "ffmpeg"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    # Try winget install location
    winget_dir = os.path.expanduser(
        "~/AppData/Local/Microsoft/WinGet/Packages/"
        "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/"
        "ffmpeg-8.1-full_build/bin"
    )
    if os.path.isfile(os.path.join(winget_dir, "ffprobe.exe")):
        return os.path.join(winget_dir, "ffprobe.exe"), os.path.join(winget_dir, "ffmpeg.exe")
    raise FileNotFoundError("ffmpeg not found")

FFPROBE, FFMPEG = find_ffmpeg()
print(f"Using ffmpeg: {FFMPEG}")


def get_audio_duration(audio_path: str) -> float:
    result = subprocess.run(
        [FFPROBE, "-v", "quiet", "-show_entries", "format=duration",
         "-of", "json", audio_path],
        capture_output=True, text=True, timeout=30,
    )
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


def detect_silences(audio_path: str, noise_db: int = -30, min_silence_sec: float = 0.3) -> list:
    result = subprocess.run(
        [FFMPEG, "-i", audio_path, "-af",
         f"silencedetect=noise={noise_db}dB:d={min_silence_sec}",
         "-f", "null", "-"],
        capture_output=True, text=True, timeout=120,
    )
    silences = []
    current = {}
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
    if duration <= target_sec * 1.5:
        return [0.0, duration]

    split_points = [0.0]
    current_pos = 0.0

    while current_pos + target_sec * 1.3 < duration:
        target = current_pos + target_sec
        best_mid = None
        best_distance = float("inf")

        for s in silences:
            mid = s["mid"]
            if mid <= current_pos + 10:
                continue
            if mid > current_pos + target_sec * 2:
                continue
            dist = abs(mid - target)
            if dist < best_distance:
                best_distance = dist
                best_mid = mid

        if best_mid is not None and best_distance < target_sec * 0.6:
            split_point = best_mid
        else:
            split_point = target

        split_points.append(split_point)
        current_pos = split_point

    split_points.append(duration)
    return split_points


def split_audio(audio_path: str, split_points: list) -> list:
    chunks = []
    for i in range(len(split_points) - 1):
        start = split_points[i]
        chunk_duration = split_points[i + 1] - start
        fd, chunk_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        subprocess.run(
            [FFMPEG, "-y", "-i", audio_path,
             "-ss", f"{start:.3f}", "-t", f"{chunk_duration:.3f}",
             "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
             chunk_path],
            capture_output=True, timeout=60,
        )
        if os.path.isfile(chunk_path) and os.path.getsize(chunk_path) > 1000:
            chunks.append(chunk_path)
        else:
            try:
                os.unlink(chunk_path)
            except OSError:
                pass
    return chunks


# ─── Whisper server communication ──────────────────────────────────────────

def transcribe_via_server(audio_path: str, server_url: str) -> dict:
    """Send audio to Whisper HTTP server and get transcription."""
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    audio_base64 = base64.b64encode(audio_bytes).decode("ascii")

    payload = json.dumps({
        "audio_base64": audio_base64,
        "language": "ru",
        "beam_size": 1,
        "initial_prompt": INITIAL_PROMPT,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{server_url}/transcribe",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    audio_path = sys.argv[1] if len(sys.argv) > 1 else None
    server_url = sys.argv[2] if len(sys.argv) > 2 else WHISPER_SERVER

    if not audio_path or not os.path.isfile(audio_path):
        print(f"Usage: python {sys.argv[0]} <audio_file> [whisper_server_url]")
        sys.exit(1)

    print(f"Audio: {audio_path}")
    print(f"Whisper server: {server_url}")
    print()

    # Step 1: Get duration
    duration = get_audio_duration(audio_path)
    print(f"Duration: {duration:.1f}s")

    # Step 2: Detect silences
    print("Detecting silences...")
    silences = detect_silences(audio_path)
    print(f"Found {len(silences)} silence intervals")
    for i, s in enumerate(silences[:10]):
        print(f"  silence {i+1}: {s['start']:.2f}s - {s['end']:.2f}s (gap {s['end']-s['start']:.2f}s)")
    if len(silences) > 10:
        print(f"  ... and {len(silences)-10} more")
    print()

    # Step 3: Compute split points
    split_points = compute_split_points(duration, silences, CHUNK_TARGET)
    n_chunks = len(split_points) - 1
    print(f"Split points: {[f'{p:.1f}s' for p in split_points]}")
    print(f"Number of chunks: {n_chunks}")
    for i in range(n_chunks):
        chunk_dur = split_points[i+1] - split_points[i]
        print(f"  chunk {i+1}: {split_points[i]:.1f}s - {split_points[i+1]:.1f}s ({chunk_dur:.1f}s)")
    print()

    # Step 4: Split audio
    print("Splitting audio...")
    chunk_paths = split_audio(audio_path, split_points)
    print(f"Created {len(chunk_paths)} chunk files")
    for i, cp in enumerate(chunk_paths):
        size_kb = os.path.getsize(cp) / 1024
        print(f"  chunk {i+1}: {cp} ({size_kb:.0f} KB)")
    print()

    # Step 5: Transcribe each chunk
    print("=" * 60)
    print("CHUNKED TRANSCRIPTION")
    print("=" * 60)
    all_texts = []
    total_chars = 0
    t_total = time.time()

    for i, chunk_path in enumerate(chunk_paths):
        chunk_dur = split_points[i+1] - split_points[i]
        print(f"\n--- Chunk {i+1}/{len(chunk_paths)} ({chunk_dur:.1f}s) ---")
        t0 = time.time()
        try:
            result = transcribe_via_server(chunk_path, server_url)
            elapsed = time.time() - t0
            text = result["text"]
            print(f"  Time: {elapsed:.1f}s | Chars: {len(text)}")
            print(f"  Text: {text[:200]}{'...' if len(text) > 200 else ''}")
            all_texts.append(text)
            total_chars += len(text)
        except Exception as e:
            print(f"  ERROR: {e}")
        finally:
            try:
                os.unlink(chunk_path)
            except OSError:
                pass

    chunked_text = " ".join(t for t in all_texts if t)
    total_elapsed = time.time() - t_total
    print(f"\n{'=' * 60}")
    print(f"CHUNKED RESULT: {len(chunked_text)} chars in {total_elapsed:.1f}s ({n_chunks} chunks)")
    print(f"{'=' * 60}")

    # Step 6: Transcribe whole file for comparison
    print(f"\n{'=' * 60}")
    print("WHOLE-FILE TRANSCRIPTION (for comparison)")
    print(f"{'=' * 60}")
    t0 = time.time()
    try:
        result = transcribe_via_server(audio_path, server_url)
        whole_elapsed = time.time() - t0
        whole_text = result["text"]
        print(f"Time: {whole_elapsed:.1f}s | Chars: {len(whole_text)}")
    except Exception as e:
        print(f"ERROR: {e}")
        whole_text = ""
        whole_elapsed = 0

    # Step 7: Save results
    output_path = os.path.join(os.path.dirname(audio_path) or ".", "chunking_test_result.txt")
    # Also save to server/temp
    temp_output = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "temp", "qa_chunking.txt")

    for out_path in [output_path, temp_output]:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(f"=== CHUNKING TEST ===\n")
                f.write(f"Audio: {audio_path}\n")
                f.write(f"Duration: {duration:.1f}s\n")
                f.write(f"Chunks: {n_chunks}\n")
                f.write(f"Split points: {[f'{p:.1f}' for p in split_points]}\n\n")

                f.write(f"=== CHUNKED TRANSCRIPTION ({len(chunked_text)} chars, {total_elapsed:.1f}s) ===\n")
                f.write(chunked_text + "\n\n")

                for i, text in enumerate(all_texts):
                    chunk_dur = split_points[i+1] - split_points[i]
                    f.write(f"--- chunk {i+1} ({split_points[i]:.1f}s-{split_points[i+1]:.1f}s, {chunk_dur:.1f}s) [{len(text)} chars] ---\n")
                    f.write(text + "\n\n")

                f.write(f"=== WHOLE-FILE TRANSCRIPTION ({len(whole_text)} chars, {whole_elapsed:.1f}s) ===\n")
                f.write(whole_text + "\n")
            print(f"\nResults saved to: {out_path}")
        except Exception as e:
            print(f"Failed to save to {out_path}: {e}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Whole file: {len(whole_text)} chars in {whole_elapsed:.1f}s")
    print(f"  Chunked:    {len(chunked_text)} chars in {total_elapsed:.1f}s ({n_chunks} chunks)")
    diff_chars = len(chunked_text) - len(whole_text)
    diff_pct = (diff_chars / max(len(whole_text), 1)) * 100
    print(f"  Difference: {diff_chars:+d} chars ({diff_pct:+.1f}%)")
    if len(chunked_text) > len(whole_text):
        print(f"  → Chunking recovered ~{diff_chars} chars of content")
    else:
        print(f"  → Chunking produced less text (may indicate less hallucination)")


if __name__ == "__main__":
    main()
