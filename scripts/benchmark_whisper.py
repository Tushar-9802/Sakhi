"""Benchmark different Whisper models on Hindi medical audio.

Usage:
    python scripts/benchmark_whisper.py

Tests each model on test_audio/*.mp3, reports:
  - Transcription output (first 200 chars)
  - Whether key medical values appear as digits
  - Time taken
  - VRAM usage
"""

import time
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.hindi_normalize import normalize_transcript

AUDIO_DIR = "test_audio"
AUDIO_FILES = [f for f in os.listdir(AUDIO_DIR) if f.endswith((".mp3", ".wav"))]

# Expected digit substrings in ANC normal transcript
ANC_NORMAL_EXPECT = ["110", "70", "58", "11.5", "24"]
ANC_DANGER_EXPECT = ["155", "100"]

MODELS = [
    ("vasista22/whisper-hindi-small", {"chunk_length_s": 30}),
    ("collabora/whisper-large-v2-hindi", {"chunk_length_s": 30}),
]


def get_vram_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def test_model(model_name, pipe_kwargs):
    from transformers import pipeline as hf_pipeline

    print(f"\n{'='*60}")
    print(f"MODEL: {model_name}")
    print(f"{'='*60}")

    vram_before = get_vram_mb()
    print(f"Loading... (VRAM before: {vram_before:.0f} MB)")

    t0 = time.time()
    pipe = hf_pipeline(
        "automatic-speech-recognition",
        model=model_name,
        device="cuda",
    )
    load_time = time.time() - t0
    vram_after = get_vram_mb()
    print(f"Loaded in {load_time:.1f}s (VRAM: {vram_after:.0f} MB, delta: {vram_after - vram_before:.0f} MB)")

    for audio_file in AUDIO_FILES:
        audio_path = os.path.join(AUDIO_DIR, audio_file)
        print(f"\n--- {audio_file} ---")

        t0 = time.time()
        result = pipe(audio_path, **pipe_kwargs)
        elapsed = time.time() - t0

        raw = result["text"].strip()
        normalized = normalize_transcript(raw)

        print(f"Time: {elapsed:.1f}s")
        print(f"Raw ({len(raw)} chars): {raw[:200]}")
        print(f"Normalized ({len(normalized)} chars): {normalized[:200]}")

        # Check for expected digits
        expect = ANC_NORMAL_EXPECT if "normal" in audio_file else ANC_DANGER_EXPECT
        for val in expect:
            found_raw = val in raw
            found_norm = val in normalized
            status = "RAW" if found_raw else ("NORM" if found_norm else "MISS")
            print(f"  {val}: {status}")

    # Free VRAM
    del pipe
    torch.cuda.empty_cache()
    import gc; gc.collect()


if __name__ == "__main__":
    for model_name, kwargs in MODELS:
        try:
            test_model(model_name, kwargs)
        except Exception as e:
            print(f"\nERROR with {model_name}: {e}")
