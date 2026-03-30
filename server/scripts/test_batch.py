#!/usr/bin/env python3
"""Batch test: send each audio file through full /api/process pipeline.
Saves structured results for analysis.

Usage: python test_batch.py <audio_dir_or_file> [audio_dir_or_file2 ...]
"""

import sys, json, os, time, urllib.request, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

NODE_SERVER = os.environ.get("NODE_SERVER_URL", "http://localhost:3001")
AUTH_PASSWORD = "meddok2026"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "temp")


def get_auth_token():
    payload = json.dumps({"password": AUTH_PASSWORD}).encode("utf-8")
    req = urllib.request.Request(
        f"{NODE_SERVER}/api/auth/login", data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())["token"]


def send_audio(audio_path, token):
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    boundary = "----FormBoundary7MA4YWxkTrZu0gW"
    filename = os.path.basename(audio_path)
    ext = os.path.splitext(filename)[1].lower()
    mime = {"ogg": "audio/ogg", "webm": "audio/webm", "wav": "audio/wav",
            "mp3": "audio/mpeg", "m4a": "audio/mp4"}.get(ext.lstrip("."), "audio/ogg")

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="audio"; filename="{filename}"\r\n'
        f"Content-Type: {mime}\r\n\r\n"
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


def format_result(result, audio_name, elapsed):
    doc = result.get("document", {})
    timings = result.get("timings", {})
    lines = []
    lines.append(f"=== {audio_name} ===")
    lines.append(f"Total: {elapsed:.1f}s | Whisper: {timings.get('whisper', '?')} | LLM: {timings.get('llm', '?')}")

    # Patient info
    patient = doc.get("patient", {})
    if any(patient.values()):
        lines.append(f"  patient: {patient.get('fullName', '')} | {patient.get('age', '')} | {patient.get('gender', '')}")

    # All text fields
    fields = ["complaints", "anamnesis", "outpatientExams", "clinicalCourse",
              "allergyHistory", "objectiveStatus", "neurologicalStatus",
              "diagnosis", "finalDiagnosis", "conclusion", "doctorNotes",
              "recommendations", "diet"]
    for field in fields:
        val = doc.get(field, "")
        lines.append(f"\n--- {field} [{len(val)} chars] ---")
        lines.append(val if val else "(empty)")

    # Risk assessment
    risk = doc.get("riskAssessment", {})
    if risk:
        lines.append(f"\n--- riskAssessment ---")
        lines.append(json.dumps(risk, ensure_ascii=False))

    return "\n".join(lines)


def main():
    paths = sys.argv[1:] if len(sys.argv) > 1 else []
    if not paths:
        print(f"Usage: python {sys.argv[0]} <audio_file_or_dir> ...")
        sys.exit(1)

    # Collect all audio files
    audio_files = []
    seen_sizes = set()
    for p in paths:
        if os.path.isdir(p):
            for f in sorted(os.listdir(p)):
                fp = os.path.join(p, f)
                if os.path.isfile(fp) and f.lower().endswith(('.ogg', '.webm', '.wav', '.mp3', '.m4a')):
                    # Deduplicate by size
                    size = os.path.getsize(fp)
                    if size not in seen_sizes:
                        seen_sizes.add(size)
                        audio_files.append(fp)
                    else:
                        print(f"  Skipping duplicate: {f}")
        elif os.path.isfile(p):
            audio_files.append(p)

    print(f"Testing {len(audio_files)} unique audio files")
    token = get_auth_token()

    all_results = []
    for i, audio_path in enumerate(audio_files):
        name = os.path.basename(audio_path)
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(audio_files)}] {name} ({os.path.getsize(audio_path)//1024} KB)")
        print(f"{'='*60}")

        t0 = time.time()
        try:
            result = send_audio(audio_path, token)
            elapsed = time.time() - t0

            if result.get("success"):
                formatted = format_result(result, name, elapsed)
                print(formatted[:2000])
                if len(formatted) > 2000:
                    print(f"  ... ({len(formatted)} chars total)")
                all_results.append(formatted)
            else:
                error_msg = result.get("error", "Unknown error")
                print(f"  ERROR: {error_msg}")
                all_results.append(f"=== {name} ===\nERROR: {error_msg}")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  EXCEPTION ({elapsed:.1f}s): {e}")
            all_results.append(f"=== {name} ===\nEXCEPTION: {e}")

    # Save all results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "qa_batch.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Batch test: {len(audio_files)} files\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for r in all_results:
            f.write(r + "\n\n" + "="*80 + "\n\n")

    print(f"\n\nAll results saved to: {out_path}")


if __name__ == "__main__":
    main()
