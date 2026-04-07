"""
Sakhi — Augment training data with unlabeled (no speaker labels) transcript variants.

Simulates raw Whisper ASR output by stripping "ASHA:" / "Patient:" / "Mother:" labels
from existing training samples. This teaches the model to handle audio transcripts
where no speaker diarization is available.

Ground truth adjustments:
  - patient.name → null if the name was only a generic address ("दीदी", "बहन")
  - All other fields stay the same (vitals, danger signs, etc.)

Usage:
    python scripts/augment_unlabeled.py
    python scripts/augment_unlabeled.py --ratio 0.3  # 30% of samples become unlabeled
"""

import argparse
import copy
import json
import os
import random
import re
import sys

INPUT_FILE = "data/processed/training_data_raw.jsonl"
OUTPUT_FILE = "data/processed/training_data_raw_augmented.jsonl"

# Speaker label patterns to strip
SPEAKER_LABELS = re.compile(
    r'^(ASHA|Patient|Mother|Father|Husband|Doctor|ANM|Nurse|CHW|दाई)\s*:\s*',
    re.MULTILINE | re.IGNORECASE
)

# Generic address terms that are NOT real names
GENERIC_ADDRESSES = {
    "दीदी", "बहन", "बहनजी", "भाई", "भैया", "जी", "अम्मा", "माँ", "माताजी",
    "patient", "didi", "bahen", "amma",
}


def strip_speaker_labels(transcript):
    """Remove speaker labels like 'ASHA:', 'Patient:' from transcript."""
    # Remove speaker labels
    result = SPEAKER_LABELS.sub('', transcript)
    # Collapse multiple newlines
    result = re.sub(r'\n{2,}', '\n', result)
    # Remove leading/trailing whitespace per line
    result = '\n'.join(line.strip() for line in result.split('\n') if line.strip())
    return result


def fix_ground_truth(form_extraction):
    """Null out fields that would be hallucinated on unlabeled transcripts."""
    form = copy.deepcopy(form_extraction)

    # Walk through possible patient name locations
    for path in [
        ("patient", "name"),
        ("patient", "patient_name"),
        ("patient_details", "name"),
        ("mother_assessment", "patient_name"),
        ("visit_info", "patient_name"),
    ]:
        obj = form
        for key in path[:-1]:
            obj = obj.get(key, {}) if isinstance(obj, dict) else {}
        if isinstance(obj, dict) and path[-1] in obj:
            name = obj[path[-1]]
            if name and str(name).strip().lower() in GENERIC_ADDRESSES:
                obj[path[-1]] = None

    return form


def main():
    parser = argparse.ArgumentParser(description="Augment training data with unlabeled variants")
    parser.add_argument("--ratio", type=float, default=0.3,
                        help="Fraction of samples to create unlabeled variants for (default: 0.3)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    if not os.path.exists(INPUT_FILE):
        print(f"ABORT: {INPUT_FILE} not found. Run generate_training_data.py first.")
        sys.exit(1)

    # Load raw data
    raw_samples = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_samples.append(json.loads(line))

    print(f"Loaded {len(raw_samples)} raw samples")

    # Select samples to augment
    n_augment = int(len(raw_samples) * args.ratio)
    augment_indices = set(random.sample(range(len(raw_samples)), n_augment))
    print(f"Creating {n_augment} unlabeled variants ({args.ratio:.0%} of total)")

    # Create augmented dataset
    augmented = []
    n_labels_stripped = 0

    for i, sample in enumerate(raw_samples):
        # Always include original
        augmented.append(sample)

        if i in augment_indices:
            # Create unlabeled variant
            variant = copy.deepcopy(sample)
            original_transcript = variant["transcript"]
            variant["transcript"] = strip_speaker_labels(original_transcript)
            variant["id"] = f"{variant['id']}_unlabeled"
            variant["form_extraction"] = fix_ground_truth(variant["form_extraction"])

            if variant["transcript"] != original_transcript:
                n_labels_stripped += 1

            augmented.append(variant)

    # Write augmented dataset
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for sample in augmented:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\nResults:")
    print(f"  Original samples: {len(raw_samples)}")
    print(f"  Unlabeled variants added: {n_augment}")
    print(f"  Labels actually stripped: {n_labels_stripped}")
    print(f"  Total samples: {len(augmented)}")
    print(f"  Written to: {OUTPUT_FILE}")
    print(f"\nNext: Run prepare_training.py with --input {OUTPUT_FILE}")

    # Show a before/after example
    if augment_indices:
        idx = min(augment_indices)
        orig = raw_samples[idx]["transcript"]
        stripped = strip_speaker_labels(orig)
        print(f"\n--- Example (sample {idx}) ---")
        print(f"BEFORE (first 200 chars):\n  {orig[:200]}")
        print(f"\nAFTER (first 200 chars):\n  {stripped[:200]}")


if __name__ == "__main__":
    main()
