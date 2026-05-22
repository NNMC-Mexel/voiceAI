"""A/B harness for whisper_server temperature-fallback patch.
Sends a fixed audio file to the live server and reports quality signals.
Usage: python scripts/whisper_ab.py <audio_path> <label>
"""
import sys, json, base64, time, urllib.request, re

SERVER = "http://192.168.41.161:9000/transcribe"

def loop_count(text):
    # crude: phrases (3-8 words) repeated 3+ times back-to-back
    return len(re.findall(
        r"((?:[^\s]+\s+){2,7}[^\s]+)(?:\s+\1){2,}", text))

def latin_garbage(text):
    # sentences with 2+ latin words (non-medical heuristic)
    bad = 0
    for s in re.split(r"[.!?]", text):
        lw = re.findall(r"\b[a-zA-Z]{3,}\b", s)
        if len(lw) >= 2:
            bad += 1
    return bad

def main():
    audio_path, label = sys.argv[1], sys.argv[2]
    with open(audio_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    payload = json.dumps({"audio_base64": b64, "language": "ru", "beam_size": 1}).encode()
    req = urllib.request.Request(SERVER, data=payload,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        data = json.loads(r.read())
    wall = time.time() - t0
    text = data.get("text", "")
    out = {
        "label": label,
        "chars": len(text),
        "chunks": data.get("chunks"),
        "avg_logprob": round(data.get("avg_logprob", 0), 3),
        "server_elapsed_s": round(data.get("elapsed", 0), 1),
        "wall_s": round(wall, 1),
        "loop_phrases": loop_count(text),
        "latin_garbage_sentences": latin_garbage(text),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    with open(f"scripts/_ab_{label}.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[full text saved to scripts/_ab_{label}.txt]")

if __name__ == "__main__":
    main()
