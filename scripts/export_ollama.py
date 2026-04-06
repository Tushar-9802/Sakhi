"""
MedScribe v2 — Export to Ollama

Creates the Ollama model from the GGUF exported by 05_train_unsloth.py.
Also tests the model via Ollama API with a sample extraction.

Usage:
  python scripts/07_export_ollama.py --create    # Create Ollama model
  python scripts/07_export_ollama.py --test      # Test the model
  python scripts/07_export_ollama.py --create --test  # Both
"""
import argparse
import json
import os
import subprocess
import sys


MODELFILE_PATH = "configs/Modelfile"
MODEL_NAME = "medscribe-v2"
GGUF_DIR = "models/exported"

SAMPLE_TRANSCRIPT = """ASHA: नमस्ते बहन जी, कैसी तबीयत है?
Patient: ठीक हूँ दीदी, बस थोड़ी कमज़ोरी लग रही है।
ASHA: आखिरी बार पीरियड कब आया था?
Patient: करीब 7 महीने पहले, अब बच्चा होने वाला है।
ASHA: चलिए बी.पी. देखते हैं... 120/80 है, बिल्कुल नॉर्मल। वज़न 55 किलो।
Patient: आयरन की गोली खा रही हूँ रोज़।
ASHA: बहुत अच्छा। TT का टीका लगवाया?
Patient: हाँ, पहला लगवा लिया, दूसरा अगले महीने है।
ASHA: बच्चा हिल रहा है ठीक से?
Patient: हाँ दीदी, खूब हिलता है।
ASHA: अगली बार अस्पताल जाकर खून की जाँच करवा लेना। अगली विज़िट 2 हफ्ते बाद।"""


def create_model():
    """Create Ollama model from Modelfile."""
    # Check GGUF exists
    gguf_files = [f for f in os.listdir(GGUF_DIR) if f.endswith(".gguf")] if os.path.exists(GGUF_DIR) else []
    if not gguf_files:
        print(f"ABORT: No GGUF files found in {GGUF_DIR}")
        print("Run scripts/05_train_unsloth.py first.")
        sys.exit(1)

    print(f"Found GGUF: {gguf_files}")
    print(f"Creating Ollama model '{MODEL_NAME}' from {MODELFILE_PATH}...")

    result = subprocess.run(
        ["ollama", "create", MODEL_NAME, "-f", MODELFILE_PATH],
        capture_output=True, text=True, timeout=300,
    )

    if result.returncode == 0:
        print(f"Model '{MODEL_NAME}' created successfully")
        print(result.stdout)
    else:
        print(f"Failed to create model:")
        print(result.stderr)
        sys.exit(1)


def test_model():
    """Test the model with a sample ASHA transcript."""
    try:
        import ollama
    except ImportError:
        print("ollama package not installed")
        sys.exit(1)

    print(f"\nTesting '{MODEL_NAME}' with sample ANC transcript...")

    # Load ANC schema
    with open("configs/schemas/anc_visit.json", "r", encoding="utf-8") as f:
        schema = json.load(f)

    system_prompt = (
        "You are a clinical data extraction system for India's ASHA health worker program. "
        "Extract structured data from the Hindi/Hinglish conversation into JSON. "
        "ONLY extract what is explicitly stated. Use null for unmentioned fields."
    )

    user_prompt = (
        f"Extract data from this ASHA visit:\n\n{SAMPLE_TRANSCRIPT}\n\n"
        f"Schema:\n{json.dumps(schema, ensure_ascii=False)}"
    )

    import time
    t0 = time.time()
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    elapsed = time.time() - t0

    text = response.message.content
    print(f"\nResponse ({elapsed:.1f}s):\n{text[:2000]}")

    # Try to parse
    try:
        data = json.loads(text)
        print(f"\nValid JSON. Non-null fields: {_count_non_null(data)}")

        # Quick sanity checks
        checks = []
        vitals = data.get("vitals", {})
        if vitals.get("bp_systolic") == 120 and vitals.get("bp_diastolic") == 80:
            checks.append("BP extracted correctly")
        if vitals.get("weight_kg") == 55:
            checks.append("Weight extracted correctly")

        preg = data.get("pregnancy", {})
        if preg.get("gestational_weeks") and 28 <= preg["gestational_weeks"] <= 32:
            checks.append("Gestational weeks reasonable")

        for c in checks:
            print(f"  PASS: {c}")

    except json.JSONDecodeError:
        print("\nOutput is not valid JSON — model may need more fine-tuning")


def _count_non_null(d, count=0):
    if isinstance(d, dict):
        for v in d.values():
            count = _count_non_null(v, count)
    elif isinstance(d, list):
        count += len(d)
    elif d is not None:
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="MedScribe v2 — Ollama Export")
    parser.add_argument("--create", action="store_true", help="Create Ollama model")
    parser.add_argument("--test", action="store_true", help="Test model with sample")
    args = parser.parse_args()

    if not args.create and not args.test:
        print("Specify --create, --test, or both")
        sys.exit(1)

    if args.create:
        create_model()
    if args.test:
        test_model()


if __name__ == "__main__":
    main()
