"""One-shot transcription of a single audio file via the project's faster-whisper setup."""
import os, sys, time
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from faster_whisper import WhisperModel

audio = sys.argv[1]
ct2 = os.path.join(os.path.dirname(__file__), "..", "models", "whisper-hindi-ct2")

t0 = time.time()
model = WhisperModel(ct2, device="cuda", compute_type="float16")
print(f"[load] {time.time()-t0:.1f}s")

t0 = time.time()
segments, info = model.transcribe(audio, language="hi", task="transcribe", vad_filter=True)
segs = list(segments)
elapsed = time.time() - t0

print(f"[asr] {elapsed:.1f}s, lang={info.language} prob={info.language_probability:.2f}, dur={info.duration:.1f}s")
print("---")
for s in segs:
    print(f"[{s.start:5.1f}-{s.end:5.1f}] {s.text.strip()}")
print("---")
print("FULL:", " ".join(s.text.strip() for s in segs))
