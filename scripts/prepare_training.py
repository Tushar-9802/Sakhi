"""
MedScribe v2 — Prepare Training Data for Unsloth

Converts raw generated data (JSONL from 03_generate_training_data.py) into
the chat-format JSONL that Unsloth's SFTTrainer expects for Gemma 4.

Each training example becomes a multi-turn conversation:
  system: extraction instructions
  user:   transcript + schema
  assistant: structured JSON output

Two training tasks per sample:
  1. Form extraction (ANC/PNC/delivery/child_health)
  2. Danger sign detection

Also handles train/val split and data quality stats.

Usage:
  python scripts/04_prepare_training.py
  python scripts/04_prepare_training.py --val-ratio 0.15
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================
DEFAULT_INPUT_FILE = "data/processed/training_data_raw.jsonl"
AUGMENTED_INPUT_FILE = "data/processed/training_data_raw_augmented.jsonl"
# Use augmented data if available, otherwise fall back to original
INPUT_FILE = AUGMENTED_INPUT_FILE if os.path.exists(AUGMENTED_INPUT_FILE) else DEFAULT_INPUT_FILE
TRAIN_FILE = "data/processed/train.jsonl"
VAL_FILE = "data/processed/val.jsonl"
STATS_FILE = "data/processed/data_stats.json"

FORM_SYSTEM_PROMPT = (
    "You are a clinical data extraction system for India's ASHA health worker program. "
    "Extract structured data from the Hindi/Hinglish home visit conversation into the requested JSON schema. "
    "ONLY extract information explicitly stated in the conversation. Use null for any field not mentioned.\n\n"
    "STRICT RULES:\n"
    "1. Do NOT invent names, dates, phone numbers, or addresses. If the patient is only called 'दीदी' or 'बहन', set name to null.\n"
    "2. If age is not explicitly stated as a number, set age to null. Do NOT guess from context.\n"
    "3. If blood group, HIV status, or other lab tests are not discussed, they MUST be null — never assume 'negative' or a default group.\n"
    "4. If the conversation has no speaker labels (ASHA/Patient), still extract data but be extra strict about nulls.\n"
    "5. Numbers may appear as Hindi words (e.g., 'एक सो दस बटा सत्तर' = 110/70). Convert them to digits.\n"
    "Return valid JSON only."
)

DANGER_SYSTEM_PROMPT = (
    "You are a clinical danger sign detection system for India's ASHA health worker program. "
    "Analyze the Hindi/Hinglish home visit conversation for NHM-defined danger signs. "
    "ONLY flag signs with DIRECT evidence from the conversation. "
    "Each danger sign MUST include utterance_evidence — the exact Hindi quote that triggered it. "
    "If NO danger signs are present, return an empty danger_signs array. "
    "Return valid JSON only."
)


def load_schema(name: str) -> dict:
    with open(f"configs/schemas/{name}.json", "r", encoding="utf-8") as f:
        return json.load(f)


def build_form_user_message(transcript: str, schema: dict) -> str:
    return (
        f"Extract structured data from this ASHA home visit conversation:\n\n"
        f"{transcript}\n\n"
        f"Output JSON schema:\n{json.dumps(schema, ensure_ascii=False)}"
    )


def build_danger_user_message(transcript: str, visit_type: str, schema: dict) -> str:
    return (
        f"Analyze this ASHA home visit conversation for danger signs.\n\n"
        f"Visit type: {visit_type}\n\n"
        f"{transcript}\n\n"
        f"Output JSON schema:\n{json.dumps(schema, ensure_ascii=False)}"
    )


def raw_to_training_examples(sample: dict, schemas: dict) -> list[dict]:
    """Convert one raw sample into 1-2 training examples (chat format)."""
    examples = []
    transcript = sample["transcript"]
    visit_type = sample["visit_type"]
    form_schema_name = sample["form_schema"]

    form_schema = schemas[form_schema_name]
    danger_schema = schemas["danger_signs"]

    # ── Example 1: Form extraction ──
    examples.append({
        "messages": [
            {"role": "system", "content": FORM_SYSTEM_PROMPT},
            {"role": "user", "content": build_form_user_message(transcript, form_schema)},
            {"role": "assistant", "content": json.dumps(sample["form_extraction"], ensure_ascii=False)},
        ],
        "metadata": {
            "task": "form_extraction",
            "visit_type": visit_type,
            "schema": form_schema_name,
            "has_danger_signs": sample["has_danger_signs"],
            "source_id": sample["id"],
        },
    })

    # ── Example 2: Danger sign detection ──
    examples.append({
        "messages": [
            {"role": "system", "content": DANGER_SYSTEM_PROMPT},
            {"role": "user", "content": build_danger_user_message(transcript, visit_type, danger_schema)},
            {"role": "assistant", "content": json.dumps(sample["danger_signs_extraction"], ensure_ascii=False)},
        ],
        "metadata": {
            "task": "danger_signs",
            "visit_type": visit_type,
            "has_danger_signs": sample["has_danger_signs"],
            "source_id": sample["id"],
        },
    })

    return examples


def main():
    parser = argparse.ArgumentParser(description="Prepare training data for Unsloth")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    if not os.path.exists(INPUT_FILE):
        print(f"ABORT: Input not found: {INPUT_FILE}")
        print(f"Run scripts/generate_training_data.py first.")
        sys.exit(1)

    # Load schemas
    schemas = {}
    for name in ["anc_visit", "pnc_visit", "delivery", "child_health", "danger_signs"]:
        schemas[name] = load_schema(name)

    # Load raw data
    raw_samples = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_samples.append(json.loads(line))

    print(f"Loaded {len(raw_samples)} raw samples")

    # Convert to training examples
    all_examples = []
    for sample in raw_samples:
        examples = raw_to_training_examples(sample, schemas)
        all_examples.extend(examples)

    print(f"Produced {len(all_examples)} training examples ({len(all_examples) // len(raw_samples)} per sample)")

    # ── Oversample positive (danger sign) examples to fix imbalance ──
    # Raw data is ~32% positive / 68% negative due to stricter validation
    # on danger sign scenarios. Oversample positive to reach ~45% target.
    danger_positive = [ex for ex in all_examples
                       if ex["metadata"]["task"] == "danger_signs" and ex["metadata"]["has_danger_signs"]]
    danger_negative = [ex for ex in all_examples
                       if ex["metadata"]["task"] == "danger_signs" and not ex["metadata"]["has_danger_signs"]]

    if danger_positive and danger_negative:
        current_ratio = len(danger_positive) / (len(danger_positive) + len(danger_negative))
        target_ratio = 0.45
        if current_ratio < target_ratio:
            # Calculate how many extra positive samples needed
            extra_needed = int((target_ratio * len(danger_negative)) / (1 - target_ratio)) - len(danger_positive)
            extra_needed = max(0, extra_needed)
            if extra_needed > 0:
                oversampled = random.choices(danger_positive, k=extra_needed)
                all_examples.extend(oversampled)
                new_pos = len(danger_positive) + extra_needed
                new_total = new_pos + len(danger_negative)
                print(f"Oversampled: +{extra_needed} positive danger sign examples "
                      f"({current_ratio:.0%} → {new_pos/new_total:.0%})")

    # Shuffle
    random.shuffle(all_examples)

    # Split
    val_count = max(1, int(len(all_examples) * args.val_ratio))
    val_examples = all_examples[:val_count]
    train_examples = all_examples[val_count:]

    print(f"Split: {len(train_examples)} train / {len(val_examples)} val ({args.val_ratio:.0%})")

    # Write
    for path, examples in [(TRAIN_FILE, train_examples), (VAL_FILE, val_examples)]:
        with open(path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"Wrote {path} ({len(examples)} examples)")

    # Stats
    stats = {
        "raw_samples": len(raw_samples),
        "total_examples": len(all_examples),
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "by_task": {},
        "by_visit_type": {},
        "danger_sign_balance": {"positive": 0, "negative": 0},
    }
    for ex in all_examples:
        meta = ex["metadata"]
        task = meta["task"]
        vtype = meta["visit_type"]
        stats["by_task"][task] = stats["by_task"].get(task, 0) + 1
        stats["by_visit_type"][vtype] = stats["by_visit_type"].get(vtype, 0) + 1
        if meta["has_danger_signs"]:
            stats["danger_sign_balance"]["positive"] += 1
        else:
            stats["danger_sign_balance"]["negative"] += 1

    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)

    # Print stats
    print(f"\nData Statistics:")
    print(f"  By task: {json.dumps(stats['by_task'])}")
    print(f"  By visit type: {json.dumps(stats['by_visit_type'])}")
    neg = stats['danger_sign_balance']['negative']
    pos = stats['danger_sign_balance']['positive']
    total = neg + pos
    print(f"  Danger sign balance: {pos} positive / {neg} negative ({neg/total*100:.0f}% negative)" if total else "")
    print(f"\nReady for training: python scripts/train_unsloth.py")


if __name__ == "__main__":
    main()
