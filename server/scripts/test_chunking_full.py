#!/usr/bin/env python3
"""Full pipeline test: chunked transcription → Node.js server (hallucination cleaner + dictionary + LLM).

Sends audio to Whisper in chunks, concatenates, then sends to /api/process-text
(or manually to /api/transcribe + /api/structure).

Usage:
  python test_chunking_full.py <audio_file>
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
NODE_SERVER = os.environ.get("NODE_SERVER_URL", "http://localhost:3001")
AUTH_PASSWORD = "meddok2026"

CHUNK_THRESHOLD = 40
CHUNK_TARGET = 30

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


def find_ffmpeg():
    try:
        subprocess.run(["ffprobe", "-version"], capture_output=True, timeout=5)
        return "ffprobe", "ffmpeg"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    winget_dir = os.path.expanduser(
        "~/AppData/Local/Microsoft/WinGet/Packages/"
        "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/"
        "ffmpeg-8.1-full_build/bin"
    )
    if os.path.isfile(os.path.join(winget_dir, "ffprobe.exe")):
        return os.path.join(winget_dir, "ffprobe.exe"), os.path.join(winget_dir, "ffmpeg.exe")
    raise FileNotFoundError("ffmpeg not found")


FFPROBE, FFMPEG = find_ffmpeg()


def get_audio_duration(audio_path):
    result = subprocess.run(
        [FFPROBE, "-v", "quiet", "-show_entries", "format=duration", "-of", "json", audio_path],
        capture_output=True, text=True, timeout=30,
    )
    return float(json.loads(result.stdout)["format"]["duration"])


def detect_silences(audio_path, noise_db=-30, min_silence_sec=0.3):
    result = subprocess.run(
        [FFMPEG, "-i", audio_path, "-af",
         f"silencedetect=noise={noise_db}dB:d={min_silence_sec}", "-f", "null", "-"],
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


def compute_split_points(duration, silences, target_sec=30):
    if duration <= target_sec * 1.5:
        return [0.0, duration]
    split_points = [0.0]
    current_pos = 0.0
    while current_pos + target_sec * 1.3 < duration:
        target = current_pos + target_sec
        best_mid, best_distance = None, float("inf")
        for s in silences:
            mid = s["mid"]
            if mid <= current_pos + 10 or mid > current_pos + target_sec * 2:
                continue
            dist = abs(mid - target)
            if dist < best_distance:
                best_distance = dist
                best_mid = mid
        if best_mid is not None and best_distance < target_sec * 0.6:
            split_points.append(best_mid)
            current_pos = best_mid
        else:
            split_points.append(target)
            current_pos = target
    split_points.append(duration)
    return split_points


def split_audio(audio_path, split_points):
    chunks = []
    for i in range(len(split_points) - 1):
        start = split_points[i]
        dur = split_points[i + 1] - start
        fd, chunk_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        subprocess.run(
            [FFMPEG, "-y", "-i", audio_path,
             "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
             "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", chunk_path],
            capture_output=True, timeout=60,
        )
        if os.path.isfile(chunk_path) and os.path.getsize(chunk_path) > 1000:
            chunks.append(chunk_path)
        else:
            try: os.unlink(chunk_path)
            except: pass
    return chunks


def transcribe_via_server(audio_path, server_url):
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    payload = json.dumps({
        "audio_base64": base64.b64encode(audio_bytes).decode("ascii"),
        "language": "ru", "beam_size": 1, "initial_prompt": INITIAL_PROMPT,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{server_url}/transcribe", data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())


def get_auth_token():
    payload = json.dumps({"password": AUTH_PASSWORD}).encode("utf-8")
    req = urllib.request.Request(
        f"{NODE_SERVER}/api/auth/login", data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())["token"]


def send_full_pipeline(audio_path, token):
    """Send audio through full /api/process pipeline (no chunking - for comparison)."""
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    boundary = "----FormBoundary7MA4YWxkTrZu0gW"
    filename = os.path.basename(audio_path)
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="audio"; filename="{filename}"\r\n'
        f"Content-Type: audio/ogg\r\n\r\n"
    ).encode("utf-8") + audio_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

    req = urllib.request.Request(
        f"{NODE_SERVER}/api/process", data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Authorization": f"Bearer {token}",
        },
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read())


def main():
    audio_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not audio_path or not os.path.isfile(audio_path):
        print(f"Usage: python {sys.argv[0]} <audio_file>")
        sys.exit(1)

    print(f"Audio: {audio_path}")
    token = get_auth_token()
    print(f"Auth token: {token[:20]}...")

    # Step 1: Chunked transcription
    duration = get_audio_duration(audio_path)
    print(f"Duration: {duration:.1f}s")

    silences = detect_silences(audio_path)
    split_points = compute_split_points(duration, silences, CHUNK_TARGET)
    n_chunks = len(split_points) - 1
    print(f"Chunks: {n_chunks}")
    chunk_paths = split_audio(audio_path, split_points)

    texts = []
    t_whisper = time.time()
    for i, cp in enumerate(chunk_paths):
        chunk_dur = split_points[i + 1] - split_points[i]
        try:
            result = transcribe_via_server(cp, WHISPER_SERVER)
            texts.append(result["text"])
            print(f"  chunk {i+1}/{n_chunks} ({chunk_dur:.0f}s): {len(result['text'])} chars")
        except Exception as e:
            print(f"  chunk {i+1} ERROR: {e}")
        finally:
            try: os.unlink(cp)
            except: pass

    chunked_text = " ".join(t for t in texts if t)
    whisper_elapsed = time.time() - t_whisper
    print(f"\nChunked transcription: {len(chunked_text)} chars in {whisper_elapsed:.1f}s")

    # Step 2: Send chunked text through Node.js for cleaning + LLM structuring
    # We'll use a custom endpoint or simulate by sending the text directly
    # Since /api/process expects audio, we need to send the chunked audio as a single file
    # OR we can test by sending the whole audio and comparing

    # For now: send whole audio through full pipeline for comparison
    print(f"\nSending whole audio through /api/process...")
    t_full = time.time()
    try:
        full_result = send_full_pipeline(audio_path, token)
        full_elapsed = time.time() - t_full
        print(f"Full pipeline: {full_elapsed:.1f}s")
    except Exception as e:
        print(f"Full pipeline ERROR: {e}")
        full_result = None
        full_elapsed = 0

    # Save results
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "temp", "qa_chunking_full.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"=== CHUNKING FULL TEST ===\n")
        f.write(f"Duration: {duration:.1f}s, Chunks: {n_chunks}\n\n")

        f.write(f"=== CHUNKED TRANSCRIPTION (raw, {len(chunked_text)} chars) ===\n")
        f.write(chunked_text + "\n\n")

        if full_result and full_result.get("success"):
            doc = full_result.get("document", {})
            f.write(f"=== FULL PIPELINE RESULT (whole audio) ===\n")
            for key in ["complaints", "anamnesis", "outpatientExams", "clinicalCourse",
                         "allergyHistory", "objectiveStatus", "diagnosis", "finalDiagnosis",
                         "conclusion", "doctorNotes", "recommendations", "diet"]:
                val = doc.get(key, "")
                f.write(f"\n--- {key} [{len(val)} chars] ---\n")
                f.write(val + "\n")

            timings = full_result.get("timings", {})
            f.write(f"\n=== TIMINGS ===\n")
            for k, v in timings.items():
                f.write(f"  {k}: {v}\n")

    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
