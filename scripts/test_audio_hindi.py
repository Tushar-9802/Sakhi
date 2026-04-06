"""
MedScribe v2 — Hindi Audio Input Test (Gate 1)
Tests Gemma 4 E4B's native audio input with Hindi speech.

CRITICAL CONSTRAINT: E4B has a 30-second audio limit (750 tokens at 25 tok/sec).
ASHA conversations are 10-15 minutes. This script tests:
  1. Single 30-sec chunk processing
  2. Audio chunking strategy for long conversations
  3. Hindi ASR quality baseline
  4. Whisper fallback if E4B Hindi ASR is insufficient

Usage:
  python scripts/01_test_audio_hindi.py --audio <path.wav>
  python scripts/01_test_audio_hindi.py --generate-test   # generate synthetic test audio
  python scripts/01_test_audio_hindi.py --whisper-fallback # test Whisper as ASR backup
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# ── Audio Chunking ──────────────────────────────────────────────────────────

CHUNK_DURATION_SEC = 28  # 2-sec margin under 30-sec limit
OVERLAP_SEC = 2          # overlap to avoid cutting mid-word
SAMPLE_RATE = 16000


def chunk_audio(audio_path: str, chunk_dir: str = None) -> list[dict]:
    """
    Split audio file into <=28-second chunks with 2-sec overlap.
    Returns list of {path, start_sec, end_sec, duration_sec}.
    """
    import librosa
    import soundfile as sf
    import numpy as np

    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    total_duration = len(y) / sr
    print(f"  Audio loaded: {total_duration:.1f}s, {sr}Hz, mono")

    if chunk_dir is None:
        chunk_dir = os.path.join(os.path.dirname(audio_path), "chunks")
    os.makedirs(chunk_dir, exist_ok=True)

    chunks = []
    step = CHUNK_DURATION_SEC - OVERLAP_SEC
    start = 0

    while start < total_duration:
        end = min(start + CHUNK_DURATION_SEC, total_duration)
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        chunk_audio = y[start_sample:end_sample]

        chunk_path = os.path.join(chunk_dir, f"chunk_{len(chunks):03d}.wav")
        sf.write(chunk_path, chunk_audio, sr)

        chunks.append({
            "path": chunk_path,
            "start_sec": start,
            "end_sec": end,
            "duration_sec": end - start,
        })
        start += step

    print(f"  Split into {len(chunks)} chunks ({CHUNK_DURATION_SEC}s each, {OVERLAP_SEC}s overlap)")
    return chunks


# ── Gemma 4 E4B Audio Processing ───────────────────────────────────────────

def test_e4b_audio(audio_path: str, device: str = "cuda"):
    """
    Test Gemma 4 E4B native audio input via Transformers.
    Returns transcription text.
    """
    import torch
    from transformers import AutoProcessor, AutoModelForCausalLM

    print(f"\n=== Testing Gemma 4 E4B Audio (Transformers) ===")
    print(f"  Audio: {audio_path}")

    # Load model
    print("  Loading Gemma 4 E4B...")
    t0 = time.time()
    model_id = "google/gemma-4-E4B-it"

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    # Load audio
    import librosa
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    duration = len(y) / sr
    print(f"  Audio duration: {duration:.1f}s ({int(duration * 25)} tokens)")

    if duration > 30:
        print(f"  WARNING: Audio is {duration:.1f}s — exceeds 30s limit. Use chunk_audio() first.")
        return None

    # Build message with audio
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": y.tolist()},
                {"type": "text", "text": (
                    "Transcribe the following Hindi/Hinglish speech exactly as spoken. "
                    "Preserve Hindi words in Devanagari script. "
                    "Include all medical terms and numbers precisely."
                )},
            ],
        }
    ]

    # Process
    print("  Running inference...")
    t0 = time.time()
    inputs = processor.apply_chat_template(
        messages, return_tensors="pt", tokenize=True
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )

    # Decode
    response = processor.decode(output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    elapsed = time.time() - t0
    print(f"  Inference time: {elapsed:.1f}s")
    print(f"  Transcription:\n    {response[:500]}")

    return response


def test_e4b_audio_chunked(audio_path: str, device: str = "cuda"):
    """
    Process long audio by chunking into 28-sec segments.
    Assembles full transcription from all chunks.
    """
    print(f"\n=== Chunked Audio Processing ===")

    import librosa
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    duration = len(y) / sr

    if duration <= 30:
        print(f"  Audio is {duration:.1f}s — no chunking needed")
        return test_e4b_audio(audio_path, device)

    chunk_dir = os.path.join("data", "temp", "chunks")
    chunks = chunk_audio(audio_path, chunk_dir)

    transcriptions = []
    for i, chunk in enumerate(chunks):
        print(f"\n  --- Chunk {i+1}/{len(chunks)} ({chunk['start_sec']:.0f}s-{chunk['end_sec']:.0f}s) ---")
        text = test_e4b_audio(chunk["path"], device)
        if text:
            transcriptions.append({
                "chunk_index": i,
                "start_sec": chunk["start_sec"],
                "end_sec": chunk["end_sec"],
                "text": text,
            })

    # Assemble (simple concatenation — overlap dedup can be added later)
    full_text = " ".join([t["text"] for t in transcriptions])
    print(f"\n  === Full Transcription ({len(transcriptions)} chunks) ===")
    print(f"  {full_text[:1000]}")

    # Save
    output_path = os.path.join("data", "temp", "transcription_result.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "audio_path": audio_path,
            "total_duration_sec": duration,
            "num_chunks": len(chunks),
            "transcriptions": transcriptions,
            "full_text": full_text,
        }, f, ensure_ascii=False, indent=2)
    print(f"  Saved to {output_path}")

    return full_text


# ── Whisper Fallback Test ──────────────────────────────────────────────────

def test_whisper_fallback(audio_path: str, device: str = "cuda"):
    """
    Test Whisper small/medium as Hindi ASR fallback.
    If E4B's native Hindi ASR is insufficient, we use:
      Whisper (Hindi ASR) → text → Gemma 4 E4B (extraction)
    This is two models but still better than v1's three-model chain.
    """
    import torch
    from transformers import pipeline

    print(f"\n=== Whisper Fallback Test (Hindi) ===")
    print(f"  Audio: {audio_path}")

    # Try whisper-small first (lighter), upgrade to medium if needed
    for model_id in ["openai/whisper-small", "openai/whisper-medium"]:
        print(f"\n  Testing {model_id}...")
        t0 = time.time()
        try:
            pipe = pipeline(
                "automatic-speech-recognition",
                model=model_id,
                device=device,
                torch_dtype=torch.float16,
            )
            result = pipe(
                audio_path,
                generate_kwargs={"language": "hindi", "task": "transcribe"},
                chunk_length_s=30,
                batch_size=8,
                return_timestamps=True,
            )
            elapsed = time.time() - t0
            print(f"  Time: {elapsed:.1f}s")
            print(f"  Transcription:\n    {result['text'][:500]}")

            if result.get("chunks"):
                print(f"  Timestamps: {len(result['chunks'])} segments")

            return result["text"]

        except Exception as e:
            print(f"  Failed: {e}")
            continue

    print("  All Whisper models failed.")
    return None


# ── Test Audio Generation ──────────────────────────────────────────────────

def generate_test_audio():
    """
    Generate a synthetic Hindi test audio using TTS or provide instructions.
    For now, creates a silent WAV as a placeholder and prints instructions
    for obtaining real Hindi test audio.
    """
    import numpy as np
    import soundfile as sf

    os.makedirs("data/raw", exist_ok=True)
    test_path = "data/raw/test_hindi_30s.wav"

    # Create 30-sec silent audio as structural test
    silence = np.zeros(SAMPLE_RATE * 30, dtype=np.float32)
    sf.write(test_path, silence, SAMPLE_RATE)
    print(f"  Created placeholder: {test_path} (30s silent)")
    print()
    print("  To test with real Hindi audio, you need one of:")
    print("  1. Record a Hindi conversation sample (phone/mic)")
    print("  2. Use Google TTS: gtts-cli 'नमस्ते, मेरा नाम सुनीता है' --lang hi -o test.mp3")
    print("  3. Download from Common Voice Hindi dataset")
    print("  4. Use a sample from Mozilla Common Voice (hindi split)")
    print()
    print("  Recommended test sentences (ASHA visit context):")
    print('  - "दीदी, मुझे सिर में बहुत दर्द हो रहा है और आँखों के सामने धुंधला दिख रहा है"')
    print('  - "बच्चे का वज़न 2.1 किलो है, दूध ठीक से नहीं पी रहा"')
    print('  - "पिछली बार बी.पी. 140/90 आया था, अभी भी पैर सूजे हुए हैं"')

    return test_path


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MedScribe v2 — Hindi Audio Test")
    parser.add_argument("--audio", type=str, help="Path to Hindi audio file")
    parser.add_argument("--generate-test", action="store_true", help="Generate test audio placeholder")
    parser.add_argument("--whisper-fallback", action="store_true", help="Test Whisper as backup ASR")
    parser.add_argument("--chunk-only", action="store_true", help="Only test audio chunking")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()

    if args.generate_test:
        generate_test_audio()
        return

    if not args.audio:
        print("Error: provide --audio <path> or --generate-test")
        sys.exit(1)

    if not os.path.exists(args.audio):
        print(f"Error: file not found: {args.audio}")
        sys.exit(1)

    if args.chunk_only:
        chunks = chunk_audio(args.audio)
        for c in chunks:
            print(f"  Chunk: {c['start_sec']:.0f}s-{c['end_sec']:.0f}s → {c['path']}")
        return

    if args.whisper_fallback:
        test_whisper_fallback(args.audio, args.device)
    else:
        # Try E4B native audio first
        import librosa
        y, sr = librosa.load(args.audio, sr=SAMPLE_RATE, mono=True)
        duration = len(y) / sr

        if duration > 30:
            test_e4b_audio_chunked(args.audio, args.device)
        else:
            test_e4b_audio(args.audio, args.device)


if __name__ == "__main__":
    main()
