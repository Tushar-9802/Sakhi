"""Run the full Sakhi pipeline (ASR → visit type → form + danger) on one audio file."""
import json, os, sys, time
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import transcribe_audio, detect_visit_type, extract_all, init_schemas

audio = sys.argv[1]
init_schemas()

t0 = time.time()
transcript = transcribe_audio(audio)
asr_s = time.time() - t0
print(f"\n[transcript] {transcript}\n")

visit_type = detect_visit_type(transcript)
print(f"[visit_type] {visit_type}\n")

t0 = time.time()
result = extract_all(transcript, visit_type)
ext_s = time.time() - t0

print("[form]")
print(json.dumps(result.get("form"), ensure_ascii=False, indent=2))
print("\n[danger]")
print(json.dumps(result.get("danger"), ensure_ascii=False, indent=2))
print(f"\n[timing] asr={asr_s:.1f}s extract={ext_s:.1f}s total={asr_s+ext_s:.1f}s")
