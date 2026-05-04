"""Run the production transcribe_audio() twice on the same file, diff the output."""
import os, sys
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import transcribe_audio

audio = sys.argv[1]
runs = [transcribe_audio(audio) for _ in range(2)]
print("\n--- run 1 ---\n", runs[0])
print("\n--- run 2 ---\n", runs[1])
print(f"\n[match] {runs[0] == runs[1]}")
