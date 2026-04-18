"""
Sakhi — Prepare Training Data for Unsloth

Converts raw generated data into chat-format JSONL for SFTTrainer.
Fixes from v1: strips schema metadata from assistant outputs, uses trimmed
danger schema (matching production), correct system prompts.

Usage:
  python scripts/prepare_training.py
"""
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
INPUT_FILE = AUGMENTED_INPUT_FILE if os.path.exists(AUGMENTED_INPUT_FILE) else DEFAULT_INPUT_FILE
TRAIN_FILE = "data/processed/train.jsonl"
VAL_FILE = "data/processed/val.jsonl"
STATS_FILE = "data/processed/data_stats.json"

# Match production prompts exactly (from app.py)
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
    "Analyze the Hindi/Hinglish home visit conversation for NHM-defined danger signs.\n\n"
    "STRICT RULES:\n"
    "1. ONLY flag a danger sign if the EXACT words proving it appear in the conversation.\n"
    "2. utterance_evidence MUST be a verbatim copy-paste from the conversation — do NOT paraphrase or fabricate.\n"
    "3. If a vital sign is NORMAL (e.g., BP 110/70, temperature 37°C), that is NOT a danger sign.\n"
    "4. Most routine visits have ZERO danger signs. Return an empty danger_signs array when none exist.\n"
    "5. When in doubt, do NOT flag — a missed flag is better than a false alarm.\n"
    "Return valid JSON only."
)


def load_schema(name: str) -> dict:
    with open(f"configs/schemas/{name}.json", "r", encoding="utf-8") as f:
        return json.load(f)


def build_trimmed_danger_schema():
    """Match production: trimmed danger schema without checklists."""
    return {
        "type": "object",
        "properties": {
            "visit_type": {
                "type": "string",
                "enum": ["antenatal", "postnatal_mother", "newborn", "child_under5"],
            },
            "danger_signs": {
                "type": "array",
                "description": "Detected danger signs. Empty array [] if none found.",
                "items": {
                    "type": "object",
                    "properties": {
                        "sign": {"type": "string"},
                        "category": {"type": "string", "enum": ["immediate_referral", "urgent_care", "monitor_closely"]},
                        "clinical_value": {"type": ["string", "null"]},
                        "utterance_evidence": {"type": "string", "description": "REQUIRED: exact verbatim quote"},
                    },
                    "required": ["sign", "category", "utterance_evidence"],
                },
            },
            "referral_decision": {
                "type": "object",
                "properties": {
                    "decision": {"type": "string", "enum": ["refer_immediately", "refer_within_24h", "continue_monitoring", "routine_followup"]},
                    "reason": {"type": "string"},
                },
                "required": ["decision", "reason"],
            },
        },
        "required": ["visit_type", "danger_signs", "referral_decision"],
    }


def clean_form_output(form_data: dict) -> dict:
    """Strip any schema metadata from form extraction output."""
    # Remove JSON Schema metadata keys that GPT-4o sometimes includes
    for key in ("$schema", "title", "description", "$id", "$ref"):
        form_data.pop(key, None)
    return form_data


def clean_danger_output(danger_data: dict) -> dict:
    """Strip schema metadata and checklists — match production trimmed format."""
    # Remove schema metadata
    for key in ("$schema", "title", "description", "$id", "$ref"):
        danger_data.pop(key, None)

    # Remove checklists (production derives these programmatically)
    danger_data.pop("maternal_danger_signs_checklist", None)
    danger_data.pop("newborn_danger_signs_checklist", None)

    # Remove evidence_utterances from referral (production builds this from signs)
    ref = danger_data.get("referral_decision", {})
    ref.pop("evidence_utterances", None)
    ref.pop("recommended_facility", None)

    # Strip confidence from individual signs (not in trimmed schema)
    for sign in danger_data.get("danger_signs", []):
        sign.pop("confidence", None)

    return danger_data


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


def raw_to_training_examples(sample: dict, schemas: dict, danger_schema_trimmed: dict) -> list[dict]:
    """Convert one raw sample into 1-2 training examples (chat format)."""
    examples = []
    transcript = sample["transcript"]
    visit_type = sample["visit_type"]
    form_schema_name = sample["form_schema"]

    form_schema = schemas[form_schema_name]

    # ── Example 1: Form extraction ──
    form_output = clean_form_output(dict(sample["form_extraction"]))
    examples.append({
        "messages": [
            {"role": "system", "content": FORM_SYSTEM_PROMPT},
            {"role": "user", "content": build_form_user_message(transcript, form_schema)},
            {"role": "assistant", "content": json.dumps(form_output, ensure_ascii=False)},
        ],
        "metadata": {
            "task": "form_extraction",
            "visit_type": visit_type,
            "schema": form_schema_name,
            "has_danger_signs": sample["has_danger_signs"],
            "source_id": sample["id"],
        },
    })

    # ── Example 2: Danger sign detection (trimmed schema, matching production) ──
    danger_output = clean_danger_output(dict(sample["danger_signs_extraction"]))
    examples.append({
        "messages": [
            {"role": "system", "content": DANGER_SYSTEM_PROMPT},
            {"role": "user", "content": build_danger_user_message(transcript, visit_type, danger_schema_trimmed)},
            {"role": "assistant", "content": json.dumps(danger_output, ensure_ascii=False)},
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
    random.seed(42)

    if not os.path.exists(INPUT_FILE):
        print(f"ABORT: Input not found: {INPUT_FILE}")
        sys.exit(1)

    # Load schemas
    schemas = {}
    for name in ["anc_visit", "pnc_visit", "delivery", "child_health"]:
        schemas[name] = load_schema(name)
    danger_schema_trimmed = build_trimmed_danger_schema()

    # Load raw data
    raw_samples = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_samples.append(json.loads(line))

    print(f"Loaded {len(raw_samples)} raw samples from {INPUT_FILE}")

    # Convert to training examples
    all_examples = []
    schema_leak_fixed = 0
    for sample in raw_samples:
        # Count schema leakage fixes
        if "$schema" in sample.get("danger_signs_extraction", {}):
            schema_leak_fixed += 1
        if "$schema" in sample.get("form_extraction", {}):
            schema_leak_fixed += 1

        examples = raw_to_training_examples(sample, schemas, danger_schema_trimmed)
        all_examples.extend(examples)

    print(f"Produced {len(all_examples)} training examples")
    if schema_leak_fixed:
        print(f"Fixed schema leakage in {schema_leak_fixed} examples")

    # Verify no leakage remains
    leaked = 0
    for ex in all_examples:
        content = ex["messages"][2]["content"]
        if '"$schema"' in content or '"title": "' in content[:100]:
            leaked += 1
    if leaked:
        print(f"WARNING: {leaked} examples still have schema leakage!")
    else:
        print(f"Schema leakage check: CLEAN")

    # ── Oversample positive danger sign examples to ~45% ──
    danger_positive = [ex for ex in all_examples
                       if ex["metadata"]["task"] == "danger_signs" and ex["metadata"]["has_danger_signs"]]
    danger_negative = [ex for ex in all_examples
                       if ex["metadata"]["task"] == "danger_signs" and not ex["metadata"]["has_danger_signs"]]

    if danger_positive and danger_negative:
        current_ratio = len(danger_positive) / (len(danger_positive) + len(danger_negative))
        target_ratio = 0.45
        if current_ratio < target_ratio:
            extra_needed = int((target_ratio * len(danger_negative)) / (1 - target_ratio)) - len(danger_positive)
            extra_needed = max(0, extra_needed)
            if extra_needed > 0:
                oversampled = random.choices(danger_positive, k=extra_needed)
                all_examples.extend(oversampled)
                new_pos = len(danger_positive) + extra_needed
                new_total = new_pos + len(danger_negative)
                print(f"Oversampled: +{extra_needed} positive danger examples "
                      f"({current_ratio:.0%} -> {new_pos/new_total:.0%})")

    random.shuffle(all_examples)

    # Split
    val_count = max(1, int(len(all_examples) * 0.15))
    val_examples = all_examples[:val_count]
    train_examples = all_examples[val_count:]

    print(f"Split: {len(train_examples)} train / {len(val_examples)} val")

    # Write
    for path, examples in [(TRAIN_FILE, train_examples), (VAL_FILE, val_examples)]:
        with open(path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"Wrote {path}")

    # Stats
    stats = {
        "raw_samples": len(raw_samples),
        "total_examples": len(all_examples),
        "train": len(train_examples),
        "val": len(val_examples),
        "schema_leaks_fixed": schema_leak_fixed,
    }
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nReady for training: python scripts/train_unsloth.py")


if __name__ == "__main__":
    main()
