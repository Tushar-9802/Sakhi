"""
MedScribe v2 — Synthetic Training Data Generation

Generates paired (Hindi ASHA conversation, structured JSON extraction) training data
via GPT-4o Mini API. Each sample produces:
  1. A realistic Hindi ASHA home visit conversation transcript
  2. The correct structured JSON extraction (form fields + danger signs)

Adapted from MedScribe v1's proven generation pipeline with:
  - Budget cap + cost tracking
  - Per-sample validation (schema + clinical consistency)
  - Checkpoint/resume support
  - Batch quality monitoring with abort threshold
  - Dry-run mode
  - Negative examples (no danger signs, many null fields)

Usage:
  python scripts/generate_training_data.py --dry-run     # First 5 samples
  python scripts/generate_training_data.py               # Full generation
  python scripts/generate_training_data.py --resume      # Resume from checkpoint
  python scripts/generate_training_data.py --count 500   # Generate N samples
"""
import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ============================================================
# CONFIG
# ============================================================
OUTPUT_DIR = "data/processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "training_data_raw.jsonl")
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "generation_checkpoint.json")

MODEL = "gpt-4o-mini"
TEMPERATURE = 0.7       # Higher for diversity in conversations
JSON_TEMPERATURE = 0.1  # Lower for accurate structured extraction
MAX_TOKENS_TRANSCRIPT = 2500
MAX_TOKENS_EXTRACTION = 2000

BATCH_SIZE = 10
MAX_BUDGET = 10.0
FAILURE_RATE_ABORT = 0.30
MIN_BATCH_FOR_CHECK = 20

# ============================================================
# VISIT SCENARIO DEFINITIONS
# ============================================================
# Each scenario defines the visit type, clinical profile, and
# whether danger signs should be present. ~40% of samples should
# be NEGATIVE (no danger signs) to teach the model restraint.

SCENARIOS = [
    # ── ANC VISITS (40% of data) ──
    {
        "type": "anc_visit",
        "label": "ANC — Normal pregnancy, routine visit",
        "has_danger_signs": False,
        "weight": 8,
        "clinical_profile": (
            "Healthy primigravida, 24 weeks, all vitals normal, no complaints. "
            "BP 110/70, Hb 11.5, weight gain normal. Routine ANC visit. "
            "ASHA counsels on diet, IFA tablets, birth preparedness."
        ),
    },
    {
        "type": "anc_visit",
        "label": "ANC — Mild anemia, otherwise normal",
        "has_danger_signs": False,
        "weight": 5,
        "clinical_profile": (
            "Second pregnancy, 28 weeks, mild anemia (Hb 9.8). No other complaints. "
            "BP normal, fetal movements good. ASHA gives IFA, advises diet."
        ),
    },
    {
        "type": "anc_visit",
        "label": "ANC — Preeclampsia signs (HIGH RISK)",
        "has_danger_signs": True,
        "weight": 5,
        "clinical_profile": (
            "Third trimester (32-36 weeks), elevated BP (140-160/90-110), "
            "headache, blurred vision, facial/hand swelling, excessive weight gain. "
            "Possible preeclampsia. Needs immediate referral."
        ),
    },
    {
        "type": "anc_visit",
        "label": "ANC — Severe anemia (HIGH RISK)",
        "has_danger_signs": True,
        "weight": 3,
        "clinical_profile": (
            "Second trimester, severe anemia (Hb <7), extreme weakness, breathlessness, "
            "pallor, dizziness. Needs urgent referral for IV iron/transfusion."
        ),
    },
    {
        "type": "anc_visit",
        "label": "ANC — Reduced fetal movement",
        "has_danger_signs": True,
        "weight": 3,
        "clinical_profile": (
            "Late third trimester, patient reports baby moving less than usual "
            "for 2 days. Other vitals may be normal. Needs monitoring/referral."
        ),
    },
    {
        "type": "anc_visit",
        "label": "ANC — Vaginal bleeding",
        "has_danger_signs": True,
        "weight": 2,
        "clinical_profile": (
            "Any trimester, reports vaginal bleeding (spotting to heavy). "
            "Immediate danger sign requiring emergency referral."
        ),
    },
    {
        "type": "anc_visit",
        "label": "ANC — Gestational diabetes",
        "has_danger_signs": False,
        "weight": 2,
        "clinical_profile": (
            "28-32 weeks, elevated blood sugar found on testing. No acute danger signs "
            "but needs dietary counseling and follow-up. BP normal."
        ),
    },
    {
        "type": "anc_visit",
        "label": "ANC — Young primigravida, many null fields",
        "has_danger_signs": False,
        "weight": 3,
        "clinical_profile": (
            "First visit, early pregnancy (8-10 weeks). Minimal information available — "
            "no labs done yet, no vitals taken at this visit (home visit, no equipment beyond BP). "
            "Many form fields should be null. Just registration and counseling."
        ),
    },

    # ── PNC / HBNC VISITS (25% of data) ──
    {
        "type": "pnc_visit",
        "label": "PNC — Normal postpartum + healthy newborn",
        "has_danger_signs": False,
        "weight": 6,
        "clinical_profile": (
            "Day 7 HBNC visit. Mother recovering well, no fever, bleeding light. "
            "Baby weight 3.0 kg (birth weight 2.8), breastfeeding well, "
            "cord clean and dry, active, no jaundice."
        ),
    },
    {
        "type": "pnc_visit",
        "label": "PNC — Low birth weight newborn, needs monitoring",
        "has_danger_signs": False,
        "weight": 3,
        "clinical_profile": (
            "Day 14 visit. Baby 2.1 kg (birth weight 1.9), gaining slowly. "
            "Breastfeeding adequate but not frequent enough. Mother well. "
            "No acute danger signs but close monitoring needed."
        ),
    },
    {
        "type": "pnc_visit",
        "label": "PNC — Newborn not feeding well (DANGER)",
        "has_danger_signs": True,
        "weight": 3,
        "clinical_profile": (
            "Day 3 visit. Newborn lethargic, not latching, weak cry. "
            "May have fever or hypothermia. Mother reports baby sleeping too much. "
            "IMNCI danger signs present — urgent referral."
        ),
    },
    {
        "type": "pnc_visit",
        "label": "PNC — Postpartum hemorrhage signs (DANGER)",
        "has_danger_signs": True,
        "weight": 2,
        "clinical_profile": (
            "Day 1-3 visit. Mother reports heavy bleeding, soaking through cloth. "
            "Feeling dizzy and weak. Possible postpartum hemorrhage. "
            "Immediate referral needed."
        ),
    },
    {
        "type": "pnc_visit",
        "label": "PNC — Newborn jaundice",
        "has_danger_signs": True,
        "weight": 2,
        "clinical_profile": (
            "Day 3-5 visit. Newborn has yellow skin, possibly yellow palms/soles. "
            "Feeding okay or slightly reduced. Needs assessment for severity — "
            "jaundice within 24h of birth or palms/soles = severe."
        ),
    },

    # ── DELIVERY (10% of data) ──
    {
        "type": "delivery",
        "label": "Delivery — Normal institutional delivery",
        "has_danger_signs": False,
        "weight": 3,
        "clinical_profile": (
            "Full-term normal delivery at PHC/district hospital. "
            "Healthy baby boy/girl, cried immediately, breastfed within 1 hour. "
            "Birth weight 2.8-3.5 kg. Mother stable. Vaccines given at birth."
        ),
    },
    {
        "type": "delivery",
        "label": "Delivery — Home delivery (partial info)",
        "has_danger_signs": False,
        "weight": 2,
        "clinical_profile": (
            "Home delivery attended by dai/family. ASHA visiting after the fact. "
            "Limited info on delivery details. Baby seems okay. "
            "Need to check birth weight, breastfeeding, vaccines."
        ),
    },
    {
        "type": "delivery",
        "label": "Delivery — Preterm with complications (DANGER)",
        "has_danger_signs": True,
        "weight": 2,
        "clinical_profile": (
            "Preterm delivery (34-36 weeks). Low birth weight (<2.5 kg). "
            "Baby may have breathing difficulty. Mother may have had complications. "
            "Needs close monitoring or referral."
        ),
    },

    # ── CHILD HEALTH / HBYC (25% of data) ──
    {
        "type": "child_health",
        "label": "HBYC — Healthy 6-month-old, routine visit",
        "has_danger_signs": False,
        "weight": 5,
        "clinical_profile": (
            "6-month HBYC visit. Good weight gain, breastfeeding + complementary food started. "
            "Immunizations up to date. Milestones appropriate. No illness."
        ),
    },
    {
        "type": "child_health",
        "label": "HBYC — Underweight child, no acute danger",
        "has_danger_signs": False,
        "weight": 3,
        "clinical_profile": (
            "9-month visit. Child slightly underweight, pallor present (mild anemia). "
            "Eating poorly. No acute illness. ASHA counsels on feeding. "
            "Deworming due. Vitamin A due."
        ),
    },
    {
        "type": "child_health",
        "label": "HBYC — Diarrhea + dehydration (DANGER)",
        "has_danger_signs": True,
        "weight": 3,
        "clinical_profile": (
            "12-month visit. Child has watery diarrhea for 3 days, "
            "not drinking well, sunken eyes, lethargic. Possible severe dehydration. "
            "IMNCI general danger signs may be present."
        ),
    },
    {
        "type": "child_health",
        "label": "HBYC — Pneumonia signs (DANGER)",
        "has_danger_signs": True,
        "weight": 2,
        "clinical_profile": (
            "9-month visit. Child has cough for 5 days, fast breathing, "
            "chest indrawing visible. Fever present. Possible pneumonia. "
            "Needs urgent referral per IMNCI."
        ),
    },
    {
        "type": "child_health",
        "label": "HBYC — Severe malnutrition (DANGER)",
        "has_danger_signs": True,
        "weight": 2,
        "clinical_profile": (
            "15-month visit. Visible severe wasting, very low weight for age, "
            "child not active. Possible edema of feet. Severe acute malnutrition "
            "requiring NRC referral."
        ),
    },
]


# ============================================================
# PROMPTS
# ============================================================

SYSTEM_PROMPT_TRANSCRIPT = """You generate realistic Hindi conversations between ASHA health workers and patients during home visits in rural India. These will train a medical AI — clinical accuracy and natural dialogue are both critical.

OUTPUT FORMAT (strict):
- ONLY dialogue lines. Each line: "ASHA:", "Patient:", or "Mother:" followed by spoken words.
- ABSOLUTELY NO narration, stage directions, action descriptions, or parentheticals.
  WRONG: (BP चेक करते हुए) / [measures weight] / *takes temperature* / (बच्चे को देख कर)
  RIGHT: "ASHA: चलिए, BP देख लेती हूँ... 150/95 आ रहा है, ये तो ज़्यादा है।"
- ALL text MUST be in Devanagari script. Do NOT use Romanized Hindi (no "Aapka BP", no "theek hai").
  Only English words allowed: medical terms (BP, Hb, TT, IFA, ORS, kg, mg, PHC, CHC).

CLINICAL REQUIREMENTS:
- ASHA must verbally state EVERY measurement with its value in Devanagari: "आपका BP 140/90 आ रहा है", "बच्चे का वज़न 3.1 kg है", "Hb 9.5 आया है"
- Include at least 6 distinct clinical data points spoken naturally in dialogue (vitals, history, medications, plans).
- Visit-specific data the ASHA should cover:
  ANC: gestational age, BP, weight, Hb, IFA compliance, TT status, fetal movement, birth preparedness (transport, facility, money, blood donor).
  PNC/Newborn: mother's bleeding/fever/pain, baby weight, feeding pattern (frequency, latch), cord condition, jaundice check, vaccination status.
  Child health: age, weight, feeding/diet, immunization status, milestones, illness symptoms, deworming/Vitamin A.

DIALOGUE STYLE:
- Vary openings — sometimes ASHA calls from the door, sometimes patient greets first, sometimes mid-activity. Do NOT always start with "नमस्ते, कैसे हैं आप?"
- ASHA is warm but efficient — covers clinical ground without sounding like a form.
- Patient speaks colloquially: approximate dates ("लगभग 6 महीना"), local terms, sometimes vague or tangential.
- Patient may volunteer info, ask questions, express worry, or dismiss concerns.
- 20-30 dialogue turns with substantive content (not single-word responses).
- End with ASHA's concrete plan: next visit date, any referral, medications given, counseling summary."""

SYSTEM_PROMPT_FORM_EXTRACTION = """You are a clinical data extraction system. Extract structured medical data from an ASHA home visit conversation transcript into the provided JSON schema.

RULES:
1. Extract ONLY information EXPLICITLY stated or clearly implied in the conversation.
2. Use null for anything not mentioned — never guess or fill in "expected" values.
3. Numbers must match exactly as stated in conversation (BP, weight, Hb, temperature, age, etc.).
4. For array fields (symptoms_reported, counseling_provided), extract all relevant items mentioned.
5. If ASHA states a measurement value, record the exact number, not just "normal".
6. Patient's approximate statements: convert to best numeric estimate ("लगभग 6 महीना" → gestational_weeks: 24).
7. Return valid JSON matching the schema. No markdown formatting."""

SYSTEM_PROMPT_DANGER_EXTRACTION = """You are a clinical danger sign extraction system for Indian ASHA worker home visits. Extract danger signs from conversation transcripts with high precision.

CORE RULES:
1. ONLY flag danger signs with DIRECT, EXPLICIT evidence in the conversation text.
2. Each flag MUST include utterance_evidence — the exact Hindi quote that triggered it.
3. If NO danger signs exist in the conversation, return an empty danger_signs array. This is correct and expected for normal visits.
4. NEVER invent or hallucinate danger signs. When in doubt, do not flag.

CLASSIFICATION GUIDANCE:
- immediate_referral: life-threatening — heavy uncontrolled bleeding, convulsions, unconsciousness, BP ≥160/110, newborn not breathing
- urgent_care: serious — elevated BP with symptoms (headache/vision/swelling), Hb <7 with symptoms, signs of severe dehydration (sunken eyes + lethargic + not drinking), fast breathing with chest indrawing
- monitor_closely: borderline — isolated mild findings, low-grade fever (99-100°F), mild swelling alone

ANTI-HALLUCINATION:
- Do NOT flag normal values as danger signs. BP 110/70 is normal. Temperature 98.6°F is normal.
- Do NOT flag a sign just because the scenario suggests it — only flag what the CONVERSATION actually says.
- If a value is borderline (e.g., temp 99°F), classify as monitor_closely at most, NOT urgent_care.

REFERRAL LOGIC:
- Any immediate_referral sign → refer_immediately (district_hospital/FRU)
- Only urgent_care signs → refer_within_24h (PHC/CHC)
- Only monitor_closely → continue_monitoring
- No signs → routine_followup

Fill the relevant checklist (maternal or newborn): "detected" if found, "not_detected" if assessed as normal, "not_assessed" if not discussed.

Return valid JSON only."""


def build_transcript_prompt(scenario: dict) -> str:
    danger_instruction = (
        "Yes — include clear, unambiguous danger signs in the conversation"
        if scenario['has_danger_signs']
        else "No — this is a normal visit with no danger signs"
    )
    symptom_instruction = (
        "Patient describes symptoms matching danger signs in natural colloquial Hindi (not medical jargon)."
        if scenario['has_danger_signs']
        else "Patient has no concerning symptoms. Routine, healthy visit."
    )
    return f"""Generate an ASHA home visit conversation in Hindi (Devanagari only):

VISIT: {scenario['type']} — {scenario['label']}
CLINICAL DETAILS: {scenario['clinical_profile']}
DANGER SIGNS: {danger_instruction}

Rules:
- Pure dialogue, zero narration/parentheticals. ASHA speaks measurements aloud in Devanagari.
- {symptom_instruction}
- At least 6 clinical data points woven naturally into conversation.
- 20-30 turns. Feel like a real village home visit, not a medical interview."""


def build_extraction_prompt(transcript: str, visit_type: str, schema: dict) -> str:
    return f"""Extract structured data from this ASHA home visit conversation into the provided JSON schema.

CONVERSATION TRANSCRIPT:
{transcript}

OUTPUT JSON SCHEMA:
{json.dumps(schema, ensure_ascii=False, indent=2)}

Extract the data now. Return ONLY valid JSON matching the schema. Use null for fields not mentioned in the conversation."""


def build_danger_signs_prompt(transcript: str, visit_type: str, schema: dict) -> str:
    return f"""Analyze this ASHA home visit conversation for danger signs.

CONVERSATION TRANSCRIPT:
{transcript}

VISIT TYPE: {visit_type}

OUTPUT JSON SCHEMA:
{json.dumps(schema, ensure_ascii=False, indent=2)}

CRITICAL RULES:
- ONLY flag danger signs that have DIRECT evidence in the conversation
- Each danger sign MUST include utterance_evidence — the exact Hindi quote that triggered it
- If NO danger signs are present, return an empty danger_signs array
- Referral decision must be based ONLY on detected danger signs

Return ONLY valid JSON matching the schema."""


# ============================================================
# SCHEMA MAPPING
# ============================================================

def load_schema(name: str) -> dict:
    path = Path(f"configs/schemas/{name}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


VISIT_TYPE_TO_SCHEMA = {
    "anc_visit": "anc_visit",
    "pnc_visit": "pnc_visit",
    "delivery": "delivery",
    "child_health": "child_health",
}


# ============================================================
# VALIDATION
# ============================================================

def validate_transcript(text: str) -> dict:
    """Validate generated transcript quality."""
    result = {"valid": True, "issues": []}

    if not text or len(text) < 200:
        return {"valid": False, "issues": ["Transcript too short"]}

    # Must contain Hindi characters
    hindi_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    if hindi_chars < 50:
        result["valid"] = False
        result["issues"].append(f"Too few Hindi characters ({hindi_chars})")

    # Must have multiple dialogue turns
    turns = text.count("ASHA:") + text.count("Patient:") + text.count("Mother:")
    if turns < 8:
        result["valid"] = False
        result["issues"].append(f"Too few dialogue turns ({turns})")

    # Should not contain English narration
    narration_markers = ["[", "]", "(walks", "(checks", "(measures", "Scene:", "Setting:"]
    for marker in narration_markers:
        if marker.lower() in text.lower():
            result["issues"].append(f"Contains narration marker: {marker}")

    return result


def validate_extraction(data: dict, visit_type: str, has_danger_signs: bool) -> dict:
    """Validate extracted JSON quality and clinical consistency."""
    result = {"valid": True, "issues": []}

    if not isinstance(data, dict):
        return {"valid": False, "issues": ["Not a dict"]}

    # Check it's not empty
    non_null_count = _count_non_null(data)
    if non_null_count < 3:
        result["valid"] = False
        result["issues"].append(f"Almost empty extraction ({non_null_count} non-null fields)")

    return result


def validate_danger_signs(data: dict, has_danger_signs: bool) -> dict:
    """Validate danger sign extraction — the most critical validation."""
    result = {"valid": True, "issues": []}

    if not isinstance(data, dict):
        return {"valid": False, "issues": ["Not a dict"]}

    signs = data.get("danger_signs", [])

    # If we expect danger signs, there should be some
    if has_danger_signs and len(signs) == 0:
        result["valid"] = False
        result["issues"].append("Expected danger signs but got none")

    # If we DON'T expect danger signs, there should be none
    if not has_danger_signs and len(signs) > 0:
        result["valid"] = False
        result["issues"].append(f"Expected no danger signs but got {len(signs)} — hallucination in training data")

    # Every danger sign must have utterance_evidence
    for i, sign in enumerate(signs):
        if not sign.get("utterance_evidence"):
            result["valid"] = False
            result["issues"].append(f"Danger sign [{i}] '{sign.get('sign')}' missing utterance_evidence")

    # Referral decision consistency
    referral = data.get("referral_decision", {})
    decision = referral.get("decision", "")

    if has_danger_signs and decision in ("routine_followup", "continue_monitoring") and len(signs) > 0:
        # Has flags but says routine — inconsistent
        severity = [s.get("category") for s in signs]
        if "immediate_referral" in severity:
            result["valid"] = False
            result["issues"].append("Has immediate_referral signs but decision is not refer_immediately")

    if not has_danger_signs and decision in ("refer_immediately", "refer_within_24h"):
        result["valid"] = False
        result["issues"].append("No danger signs but referral decision is urgent — hallucination")

    return result


def _count_non_null(d, count=0):
    if isinstance(d, dict):
        for v in d.values():
            count = _count_non_null(v, count)
    elif isinstance(d, list):
        count += len(d)
    elif d is not None:
        count += 1
    return count


# ============================================================
# CHECKPOINT
# ============================================================

def load_checkpoint() -> dict:
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    return {
        "generated": 0, "valid": 0, "invalid": 0, "failed": 0,
        "total_cost": 0.0, "by_type": {}, "by_danger": {"positive": 0, "negative": 0},
    }


def save_checkpoint(cp: dict):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(cp, f, indent=2)


# ============================================================
# COST TRACKING
# ============================================================
# GPT-4o Mini pricing (as of 2026)
INPUT_COST_PER_M = 0.15   # $0.15 per 1M input tokens
OUTPUT_COST_PER_M = 0.60   # $0.60 per 1M output tokens


def estimate_cost(input_text: str, output_text: str) -> float:
    input_tokens = len(input_text) // 4
    output_tokens = len(output_text) // 4
    return (input_tokens * INPUT_COST_PER_M / 1_000_000) + (output_tokens * OUTPUT_COST_PER_M / 1_000_000)


# ============================================================
# WEIGHTED SCENARIO SAMPLING
# ============================================================

def sample_scenario() -> dict:
    """Sample a scenario weighted by the 'weight' field."""
    weights = [s["weight"] for s in SCENARIOS]
    return random.choices(SCENARIOS, weights=weights, k=1)[0]


# ============================================================
# GENERATION
# ============================================================

def generate_one_sample(client: OpenAI, scenario: dict, schemas: dict) -> dict | None:
    """
    Generate one complete training sample:
      1. Generate Hindi conversation transcript
      2. Extract structured form data
      3. Extract danger signs
      4. Validate all three
    Returns the sample dict or None if validation fails.
    """
    visit_type = scenario["type"]
    form_schema_name = VISIT_TYPE_TO_SCHEMA[visit_type]
    form_schema = schemas[form_schema_name]
    danger_schema = schemas["danger_signs"]

    total_cost = 0.0

    # ── Step 1: Generate transcript ──
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_TRANSCRIPT},
                {"role": "user", "content": build_transcript_prompt(scenario)},
            ],
            max_tokens=MAX_TOKENS_TRANSCRIPT,
            temperature=TEMPERATURE,
        )
        transcript = resp.choices[0].message.content.strip()
        total_cost += estimate_cost(SYSTEM_PROMPT_TRANSCRIPT + build_transcript_prompt(scenario), transcript)
    except Exception as e:
        return {"error": f"Transcript generation failed: {e}", "cost": total_cost}

    # Validate transcript
    tv = validate_transcript(transcript)
    if not tv["valid"]:
        return {"error": f"Invalid transcript: {tv['issues']}", "cost": total_cost}

    # ── Step 2: Extract form data ──
    try:
        extraction_prompt = build_extraction_prompt(transcript, visit_type, form_schema)
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_FORM_EXTRACTION},
                {"role": "user", "content": extraction_prompt},
            ],
            max_tokens=MAX_TOKENS_EXTRACTION,
            temperature=JSON_TEMPERATURE,
            response_format={"type": "json_object"},
        )
        form_text = resp.choices[0].message.content.strip()
        form_data = json.loads(form_text)
        total_cost += estimate_cost(SYSTEM_PROMPT_FORM_EXTRACTION + extraction_prompt, form_text)
    except json.JSONDecodeError as e:
        return {"error": f"Form extraction not valid JSON: {e}", "cost": total_cost}
    except Exception as e:
        return {"error": f"Form extraction failed: {e}", "cost": total_cost}

    # Validate form extraction
    ev = validate_extraction(form_data, visit_type, scenario["has_danger_signs"])
    if not ev["valid"]:
        return {"error": f"Invalid extraction: {ev['issues']}", "cost": total_cost}

    # ── Step 3: Extract danger signs ──
    try:
        danger_prompt = build_danger_signs_prompt(transcript, visit_type, danger_schema)
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_DANGER_EXTRACTION},
                {"role": "user", "content": danger_prompt},
            ],
            max_tokens=MAX_TOKENS_EXTRACTION,
            temperature=JSON_TEMPERATURE,
            response_format={"type": "json_object"},
        )
        danger_text = resp.choices[0].message.content.strip()
        danger_data = json.loads(danger_text)
        total_cost += estimate_cost(SYSTEM_PROMPT_DANGER_EXTRACTION + danger_prompt, danger_text)
    except json.JSONDecodeError as e:
        return {"error": f"Danger signs not valid JSON: {e}", "cost": total_cost}
    except Exception as e:
        return {"error": f"Danger signs extraction failed: {e}", "cost": total_cost}

    # Validate danger signs — this is the most critical validation
    dv = validate_danger_signs(danger_data, scenario["has_danger_signs"])
    if not dv["valid"]:
        return {"error": f"Invalid danger signs: {dv['issues']}", "cost": total_cost}

    # ── Success — build training sample ──
    return {
        "sample": {
            "id": None,  # assigned later
            "visit_type": visit_type,
            "scenario_label": scenario["label"],
            "has_danger_signs": scenario["has_danger_signs"],
            "transcript": transcript,
            "form_extraction": form_data,
            "danger_signs_extraction": danger_data,
            "form_schema": form_schema_name,
        },
        "validation": {
            "transcript": tv,
            "extraction": ev,
            "danger_signs": dv,
        },
        "cost": total_cost,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="MedScribe v2 — Training Data Generation")
    parser.add_argument("--dry-run", action="store_true", help="Generate 5 samples only")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--count", type=int, default=500, help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    target = 5 if args.dry_run else args.count
    print("=" * 60)
    print(f"Training Data Generation — {MODEL}" + (" [DRY RUN]" if args.dry_run else ""))
    print(f"Target: {target} samples")
    print("=" * 60)

    # ── Gate: API key ──
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\nABORT: OPENAI_API_KEY not found. Set in .env file.")
        sys.exit(1)

    client = OpenAI()

    # Quick API test
    try:
        client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=5,
        )
        print(f"API connection verified: {MODEL}")
    except Exception as e:
        print(f"\nABORT: API test failed: {e}")
        sys.exit(1)

    # ── Load schemas ──
    schemas = {}
    for name in ["anc_visit", "pnc_visit", "delivery", "child_health", "danger_signs"]:
        schemas[name] = load_schema(name)
    print(f"Loaded {len(schemas)} schemas")

    # ── Load checkpoint ──
    cp = load_checkpoint()
    start_idx = cp["generated"] if args.resume else 0
    if not args.resume:
        cp = {
            "generated": 0, "valid": 0, "invalid": 0, "failed": 0,
            "total_cost": 0.0, "by_type": {}, "by_danger": {"positive": 0, "negative": 0},
            "quality": {
                "narration_leaks": 0, "low_hindi": 0, "low_turns": 0,
                "polarity_errors": 0, "avg_clinical_density": 0.0,
                "total_clinical_density": 0, "total_scored": 0,
            },
        }

    # Open output file (append if resuming)
    mode = "a" if args.resume and os.path.exists(OUTPUT_FILE) else "w"
    outfile = open(OUTPUT_FILE, mode, encoding="utf-8")

    print(f"\nStarting from sample {start_idx + 1}...")
    if args.dry_run:
        print("DRY RUN: Generating 5 samples. Review output before full run.\n")

    batch_count = 0

    for i in range(start_idx, target):
        scenario = sample_scenario()
        vtype = scenario["type"]

        result = generate_one_sample(client, scenario, schemas)

        # Track cost regardless
        cp["total_cost"] += result.get("cost", 0)

        if "error" in result:
            cp["failed"] += 1
            cp["generated"] += 1
            status = f"FAIL: {result['error'][:80]}"
        elif "sample" in result:
            sample = result["sample"]
            sample["id"] = f"ms2_{i:04d}"

            # Write to JSONL
            outfile.write(json.dumps(sample, ensure_ascii=False) + "\n")
            outfile.flush()

            cp["valid"] += 1
            cp["generated"] += 1
            cp["by_type"][vtype] = cp["by_type"].get(vtype, 0) + 1
            if scenario["has_danger_signs"]:
                cp["by_danger"]["positive"] += 1
            else:
                cp["by_danger"]["negative"] += 1

            n_signs = len(sample["danger_signs_extraction"].get("danger_signs", []))
            status = f"OK [{vtype}] signs={n_signs}"

            # ── Real-time quality monitoring ──
            q = cp.setdefault("quality", {
                "narration_leaks": 0, "low_hindi": 0, "low_turns": 0,
                "polarity_errors": 0, "avg_clinical_density": 0.0,
                "total_clinical_density": 0, "total_scored": 0,
            })
            t = sample["transcript"]
            # Check narration (exclude single English words in parens like "(latch)")
            paren_matches = re.findall(r'\([^)]{5,}\)', t)  # only flag parens with 5+ chars
            bracket_matches = re.findall(r'\[[^\]]+\]', t)
            if paren_matches or bracket_matches or '*' in t:
                q["narration_leaks"] += 1
            # Check Hindi density
            hindi_chars = sum(1 for c in t if '\u0900' <= c <= '\u097F')
            if hindi_chars / max(len(t), 1) < 0.5:
                q["low_hindi"] += 1
            # Check turns
            turns = t.count("ASHA:") + t.count("Patient:") + t.count("Mother:")
            if turns < 15:
                q["low_turns"] += 1
            # Clinical density (count numbers near medical terms)
            clin_kws = ['bp', 'weight', 'वजन', 'kg', 'hb', 'हीमोग्लोबिन', 'तापमान',
                        'हफ्ता', 'महीना', 'tablet', 'ग्राम', 'डिग्री', 'किलो']
            nums = re.findall(r'\d+\.?\d*', t)
            clin_count = 0
            for n in nums:
                pos = t.find(n)
                ctx = t[max(0, pos-40):pos+40].lower()
                if any(kw in ctx for kw in clin_kws):
                    clin_count += 1
            q["total_clinical_density"] += clin_count
            q["total_scored"] += 1
            # Polarity check
            if scenario["has_danger_signs"] and n_signs == 0:
                q["polarity_errors"] += 1
            elif not scenario["has_danger_signs"] and n_signs > 0:
                q["polarity_errors"] += 1
        else:
            cp["failed"] += 1
            cp["generated"] += 1
            status = "UNKNOWN"

        batch_count += 1
        pct = cp["generated"] / target * 100
        print(f"  [{cp['generated']}/{target}] ({pct:.0f}%) ${cp['total_cost']:.4f} | {scenario['label'][:50]} | {status}")

        # ── Safety checks ──
        if cp["total_cost"] > MAX_BUDGET:
            print(f"\nABORT: Budget exceeded (${cp['total_cost']:.2f} > ${MAX_BUDGET})")
            break

        if cp["generated"] >= MIN_BATCH_FOR_CHECK:
            total_attempted = cp["valid"] + cp["failed"]
            if total_attempted > 0 and cp["failed"] / total_attempted > FAILURE_RATE_ABORT:
                print(f"\nABORT: Failure rate {cp['failed']}/{total_attempted} exceeds threshold")
                break

        # Checkpoint + quality report
        if batch_count >= BATCH_SIZE:
            save_checkpoint(cp)
            batch_count = 0

            # Periodic quality report every 50 samples
            q = cp.get("quality", {})
            scored = q.get("total_scored", 0)
            if scored > 0 and scored % 50 < BATCH_SIZE:
                avg_clin = q["total_clinical_density"] / scored
                print(f"\n  ┌── QUALITY REPORT (n={scored}) ──")
                print(f"  │ Narration leaks: {q['narration_leaks']} ({q['narration_leaks']/scored*100:.0f}%)")
                print(f"  │ Low Hindi:       {q['low_hindi']} ({q['low_hindi']/scored*100:.0f}%)")
                print(f"  │ Low turns (<15): {q['low_turns']} ({q['low_turns']/scored*100:.0f}%)")
                print(f"  │ Polarity errors: {q['polarity_errors']} ({q['polarity_errors']/scored*100:.0f}%)")
                print(f"  │ Avg clinical #s: {avg_clin:.1f} per sample")
                print(f"  └{'─' * 35}")

                # ABORT on polarity errors > 5%
                if scored >= 30 and q["polarity_errors"] / scored > 0.05:
                    print(f"\n  ABORT: Polarity error rate {q['polarity_errors']}/{scored} > 5%")
                    print(f"  This means the model is hallucinating danger signs or missing real ones.")
                    print(f"  Fix prompts before continuing.")
                    break

        # Rate limit
        time.sleep(0.3)

    outfile.close()
    save_checkpoint(cp)

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("GENERATION SUMMARY")
    print("=" * 60)
    print(f"  Total generated:  {cp['generated']}")
    print(f"  Valid samples:    {cp['valid']}")
    print(f"  Failed:           {cp['failed']}")
    print(f"  Total cost:       ${cp['total_cost']:.4f}")
    print(f"\n  By visit type:")
    for vt, count in sorted(cp["by_type"].items()):
        print(f"    {vt}: {count}")
    print(f"\n  Danger sign balance:")
    print(f"    Positive (has danger signs): {cp['by_danger']['positive']}")
    print(f"    Negative (no danger signs):  {cp['by_danger']['negative']}")
    total_with_labels = cp['by_danger']['positive'] + cp['by_danger']['negative']
    if total_with_labels > 0:
        neg_pct = cp['by_danger']['negative'] / total_with_labels * 100
        print(f"    Negative ratio: {neg_pct:.0f}% (target: ~40%)")

    # Quality summary
    q = cp.get("quality", {})
    scored = q.get("total_scored", 0)
    if scored > 0:
        avg_clin = q["total_clinical_density"] / scored
        print(f"\n  Quality metrics:")
        print(f"    Narration leaks: {q['narration_leaks']}/{scored} ({q['narration_leaks']/scored*100:.0f}%)")
        print(f"    Low Hindi:       {q['low_hindi']}/{scored} ({q['low_hindi']/scored*100:.0f}%)")
        print(f"    Low turns (<15): {q['low_turns']}/{scored} ({q['low_turns']/scored*100:.0f}%)")
        print(f"    Polarity errors: {q['polarity_errors']}/{scored} ({q['polarity_errors']/scored*100:.0f}%)")
        print(f"    Avg clinical #s: {avg_clin:.1f} per sample")

        if q['polarity_errors'] > 0:
            print(f"\n    WARNING: {q['polarity_errors']} polarity errors detected!")
            print(f"    Review these samples before training.")

    print(f"\n  Output: {OUTPUT_FILE}")

    if args.dry_run:
        print(f"\nDRY RUN complete. Review the output file.")
        print(f"If quality looks good, run: python scripts/generate_training_data.py --count {args.count}")

    if cp["valid"] >= 200:
        print(f"\nREADY for next step: python scripts/04_prepare_training.py")
    elif cp["valid"] > 0:
        print(f"\n{cp['valid']} samples generated. May need more for good fine-tune results.")

    print("=" * 60)


if __name__ == "__main__":
    main()
