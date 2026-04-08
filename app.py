"""
Sakhi (सखी) — ASHA Health Worker AI Companion
================================================
Hindi voice → structured MCTS/HMIS forms + danger sign detection
powered by Gemma 4 E4B (fine-tuned via Unsloth).

Tabs:
1. Voice to Form    — Hindi audio → transcript → form extraction + danger signs
2. Text to Form     — Paste Hindi transcript → extraction
3. About            — Architecture, ASHA context, offline deployment

Launch: python app.py
"""
import os
import re
import json
import time
import html as html_mod

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import gradio as gr

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_PATH = "./models/checkpoints/final"
MAX_SEQ_LENGTH = 4096

# Ollama config — set OLLAMA_MODEL to use Ollama instead of Unsloth
# Use "sakhi" once fine-tuned GGUF is registered, or base model for now
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:e4b-it-q4_K_M")
USE_OLLAMA = os.environ.get("USE_OLLAMA", "1") == "1"

# System prompts (same as training)
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

# ============================================================
# EXAMPLE TRANSCRIPTS (for demo)
# ============================================================
EXAMPLE_TRANSCRIPTS = [
    [
        "ANC Visit — Normal",
        (
            "ASHA: नमस्ते, कैसे हैं आप?\n"
            "Patient: नमस्ते दीदी, मैं ठीक हूँ।\n"
            "ASHA: अच्छा है। मैं आपका चेकअप करने आई हूँ। चलिए, पहले आपका BP चेक कर लेती हूँ।\n"
            "Patient: ठीक है।\n"
            "ASHA: आपका BP 110/70 है, बिल्कुल ठीक है। अब वजन देखती हूँ... 58 kg है। पिछली बार 56 था, तो अच्छा बढ़ रहा है।\n"
            "Patient: हाँ, मैं अच्छा खा रही हूँ।\n"
            "ASHA: बहुत अच्छा! Hb कितना आया था पिछली बार?\n"
            "Patient: डॉक्टर ने कहा था 11.5 है।\n"
            "ASHA: ये तो बहुत अच्छा है। IFA की गोलियाँ ले रही हैं?\n"
            "Patient: हाँ, रोज़ लेती हूँ।\n"
            "ASHA: TT का टीका लगा?\n"
            "Patient: हाँ, पहला लग गया है।\n"
            "ASHA: बच्चे की हलचल कैसी है?\n"
            "Patient: बहुत हिलता-डुलता है, ठीक है।\n"
            "ASHA: बहुत अच्छा। आप लगभग 24 हफ्ते की हैं। डिलीवरी के लिए कहाँ जाएँगी?\n"
            "Patient: PHC में।\n"
            "ASHA: गाड़ी का इंतज़ाम है?\n"
            "Patient: हाँ, पति की गाड़ी है।\n"
            "ASHA: ठीक है। अगली बार 2 हफ्ते बाद आऊँगी। कोई तकलीफ़ हो तो फ़ोन कर दीजिए।\n"
            "Patient: ठीक है दीदी, धन्यवाद।"
        ),
    ],
    [
        "ANC Visit — Preeclampsia (DANGER)",
        (
            "ASHA: नमस्ते दीदी, कैसे हैं?\n"
            "Patient: दीदी, मुझे बहुत सिरदर्द हो रहा है कल से।\n"
            "ASHA: अच्छा, और कोई तकलीफ़?\n"
            "Patient: हाँ, आँखों के सामने धुंधला दिखता है कभी-कभी। और चेहरे पर सूजन भी आ गई है।\n"
            "ASHA: ये तो ठीक नहीं है। मैं BP चेक करती हूँ... आपका BP 155/100 आ रहा है। ये बहुत ज़्यादा है।\n"
            "Patient: क्या करें दीदी?\n"
            "ASHA: आपको तुरंत PHC जाना होगा। ये गंभीर हो सकता है। आप कितने महीने की हैं?\n"
            "Patient: लगभग 8 महीने।\n"
            "ASHA: पैरों में सूजन है?\n"
            "Patient: हाँ, काफी सूजन है।\n"
            "ASHA: मैं अभी गाड़ी का इंतज़ाम करती हूँ। आपको आज ही PHC ले चलती हूँ।"
        ),
    ],
    [
        "PNC — Newborn not feeding (DANGER)",
        (
            "ASHA: नमस्ते, कैसे हैं? बच्चा कैसा है?\n"
            "Mother: दीदी, बच्चा बहुत सोता रहता है। दूध भी ठीक से नहीं पीता।\n"
            "ASHA: कब से ऐसा है?\n"
            "Mother: कल से। पहले ठीक था, अब लगभग 12 घंटे से दूध नहीं पिया।\n"
            "ASHA: बच्चे का रोना कैसा है?\n"
            "Mother: बहुत कमज़ोर आवाज़ में रोता है।\n"
            "ASHA: तापमान चेक करती हूँ... 100.5 डिग्री है। बुखार है। और बच्चा सुस्त लग रहा है।\n"
            "Mother: क्या करें?\n"
            "ASHA: ये IMNCI के danger signs हैं। बच्चे को तुरंत PHC ले जाना होगा। मैं गाड़ी बुलाती हूँ।"
        ),
    ],
    [
        "Child Health — Routine visit",
        (
            "ASHA: नमस्ते, बच्चा कैसा है?\n"
            "Mother: बिल्कुल ठीक है दीदी। खूब खाता है, खेलता है।\n"
            "ASHA: बहुत अच्छा! वजन देखती हूँ... 8.5 kg है। 9 महीने के लिए अच्छा है।\n"
            "Mother: हाँ, दाल-चावल, केला सब खाता है अब।\n"
            "ASHA: Vitamin A की दवाई दी थी पिछली बार?\n"
            "Mother: हाँ, 6 महीने में दी थी।\n"
            "ASHA: अच्छा। अब deworming भी देनी है। और टीके सब लगे हैं?\n"
            "Mother: हाँ, सब समय पर लगे हैं।\n"
            "ASHA: बहुत अच्छा। बच्चा बैठता है, घुटनों पर चलता है?\n"
            "Mother: हाँ, सब करता है। बोलने भी लगा है थोड़ा।\n"
            "ASHA: बढ़िया है। अगली बार 3 महीने बाद आऊँगी।"
        ),
    ],
]


# ============================================================
# SCHEMA LOADING
# ============================================================
def load_schema(name):
    with open(f"configs/schemas/{name}.json", "r", encoding="utf-8") as f:
        return json.load(f)


SCHEMAS = {}
VISIT_TYPE_MAP = {
    "anc_visit": "anc_visit",
    "pnc_visit": "pnc_visit",
    "delivery": "delivery",
    "child_health": "child_health",
}


def init_schemas():
    global SCHEMAS
    for name in ["anc_visit", "pnc_visit", "delivery", "child_health", "danger_signs"]:
        SCHEMAS[name] = load_schema(name)


# ============================================================
# MODEL LOADING
# ============================================================
_model = None
_tokenizer = None


def load_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    import torch
    torch._dynamo.config.suppress_errors = True
    from unsloth import FastLanguageModel

    print("[MODEL] Loading Gemma 4 E4B fine-tuned model...")
    _model, _tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(_model)
    print("[MODEL] Model loaded.")
    return _model, _tokenizer


# ============================================================
# TRANSCRIPT POST-PROCESSING (delegated to src/hindi_normalize)
# ============================================================
from src.hindi_normalize import normalize_transcript as postprocess_transcript


_whisper_model = None

def transcribe_audio(audio_path):
    """Transcribe audio using collabora/whisper-large-v2-hindi via faster-whisper (CTranslate2)."""
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        import os
        ct2_path = os.path.join(os.path.dirname(__file__), "models", "whisper-hindi-ct2")
        if os.path.exists(ct2_path):
            print(f"[ASR] Loading CTranslate2 model from {ct2_path}...")
            _whisper_model = WhisperModel(ct2_path, device="cuda", compute_type="float16")
        else:
            print("[ASR] CT2 model not found, loading from HuggingFace (slower)...")
            _whisper_model = WhisperModel("collabora/whisper-large-v2-hindi", device="cuda", compute_type="float16")
        print("[ASR] Whisper loaded.")

    print("[ASR] Transcribing...")
    segments, info = _whisper_model.transcribe(audio_path, language="hi", task="transcribe", vad_filter=True)
    transcript = " ".join(seg.text.strip() for seg in segments)

    transcript = postprocess_transcript(transcript)

    print(f"[ASR] Transcript ({len(transcript)} chars)")
    return transcript


def run_inference(system_prompt, user_prompt):
    """Run model inference via Ollama or Unsloth, return parsed JSON or raw text."""
    if USE_OLLAMA:
        return _run_inference_ollama(system_prompt, user_prompt)
    return _run_inference_unsloth(system_prompt, user_prompt)


def _run_inference_ollama(system_prompt, user_prompt):
    """Run inference via Ollama API — fast GGUF on GPU."""
    import ollama

    t0 = time.time()
    resp = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": 0.1, "num_ctx": 4096},
    )
    elapsed = time.time() - t0

    response = resp.message.content
    tok_s = resp.eval_count / (resp.eval_duration / 1e9) if resp.eval_duration else 0
    print(f"[LLM] Ollama: {elapsed:.1f}s ({resp.eval_count} tok, {tok_s:.0f} tok/s)")

    parsed = _parse_json_response(response)
    return {"raw": response, "parsed": parsed, "time_s": elapsed}


def _run_inference_unsloth(system_prompt, user_prompt):
    """Run inference via Unsloth/transformers — slower but works without Ollama."""
    import torch
    model, tokenizer = load_model()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text=[text], return_tensors="pt").to("cuda")

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=768, do_sample=False)
    elapsed = time.time() - t0

    response = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    parsed = _parse_json_response(response)
    return {"raw": response, "parsed": parsed, "time_s": elapsed}


def _parse_json_response(response):
    """Parse JSON from model response, handling markdown fences and quirks."""
    print(f"[DEBUG] raw response repr (first 80): {repr(response[:80])}")

    # Strip markdown fences — handle variations: ```json, ``` json, whitespace, BOM
    clean = response.strip().lstrip('\ufeff')
    clean = re.sub(r'^`{3,}\s*(?:json)?\s*[\r\n]*', '', clean, flags=re.IGNORECASE)
    clean = re.sub(r'[\r\n]*`{3,}\s*$', '', clean)
    clean = clean.strip()

    # Fix common model quirks
    if clean and clean[0] == '"' and not clean.startswith('{"') and not clean.startswith('["'):
        clean = "{" + clean
    if clean and clean[0] not in ('{', '['):
        first_brace = min(
            (clean.find("{") if clean.find("{") >= 0 else len(clean)),
            (clean.find("[") if clean.find("[") >= 0 else len(clean)),
        )
        if first_brace < len(clean):
            print(f"[DEBUG] skipped leading junk: {repr(clean[:first_brace])}")
            clean = clean[first_brace:]
    clean = re.sub(r'"{2,}([^"]+)"{2,}', r'"\1"', clean)
    clean = re.sub(r'(?<=: )"{2,}', '"', clean)
    clean = re.sub(r'"{2,}(?=\s*[,\}\]])', '"', clean)
    clean = re.sub(r',\s*([}\]])', r'\1', clean)

    print(f"[DEBUG] cleaned JSON (first 120): {repr(clean[:120])}")

    try:
        return json.loads(clean)
    except json.JSONDecodeError as e:
        print(f"[DEBUG] JSON parse failed: {e}")
        for end_pos in range(len(clean), max(0, len(clean) - 200), -1):
            if clean[end_pos - 1] in ('}', ']'):
                try:
                    parsed = json.loads(clean[:end_pos])
                    print(f"[DEBUG] recovered JSON by truncating at pos {end_pos}")
                    return parsed
                except json.JSONDecodeError:
                    continue

    print(f"[DEBUG] FULL raw response ({len(response)} chars):\n{response}\n---END---")
    return None


# ============================================================
# EXTRACTION PIPELINE
# ============================================================
def detect_visit_type(transcript):
    """Heuristic visit type detection from transcript content."""
    t = transcript.lower()
    # ANC keywords — check first (most common), broad matching
    if any(kw in t for kw in ["गर्भ", "प्रेग्नेंसी", "pregnancy", "anc", "पेट में बच्चा",
                               "हफ्ते की", "हफ्ते हो", "महीने की", "महीने हो",
                               "lmp", "edd", "bp चेक", "hb ", "ifa", "tt का टीका",
                               "बच्चे की हलचल", "fetal", "डिलीवरी कहाँ", "डिलीवरी के लिए",
                               "जन्म के लिए तैयारी", "birth preparedness"]):
        return "anc_visit"
    # PNC — check before delivery (delivery keywords overlap)
    if any(kw in t for kw in ["नवजात", "newborn", "दूध पीना", "दूध नहीं पीता", "दूध पीता",
                               "दूध पिला", "नाभि", "cord", "नाल", "स्तनपान",
                               "breastfeed", "imnci", "hbnc", "डिलीवरी के बाद",
                               "hbnc", "pnc"]):
        return "pnc_visit"
    # Delivery — only if actually about the delivery event
    if any(kw in t for kw in ["डिलीवरी हो गई", "डिलीवरी हुई", "delivery हुई",
                               "जन्म हुआ", "पैदा हुआ", "प्रसव हुआ",
                               "ऑपरेशन से हुई", "caesarean", "सिजेरियन",
                               "जन्म का वजन", "birth weight"]):
        return "delivery"
    if any(kw in t for kw in ["बच्चा", "child", "टीका", "vaccine", "deworming",
                               "vitamin a", "hbyc", "महीने का"]):
        return "child_health"
    return "anc_visit"


def build_trimmed_danger_schema():
    """Danger sign schema without checklists — much smaller output."""
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


# Maternal danger sign names that map to checklist fields
MATERNAL_CHECKLIST_SIGNS = {
    "severe_vaginal_bleeding": ["vaginal bleeding", "severe bleeding", "रक्तस्राव", "खून"],
    "convulsions": ["convulsion", "seizure", "दौरा", "अकड़न"],
    "severe_headache_blurred_vision": ["headache", "blurred vision", "सिरदर्द", "धुंधला"],
    "high_fever": ["high fever", "fever", "बुखार", "तेज़ बुखार"],
    "severe_abdominal_pain": ["abdominal pain", "पेट दर्द", "पेट में दर्द"],
    "fast_difficult_breathing": ["breathing", "साँस", "सांस"],
    "swelling_face_hands": ["swelling", "सूजन"],
    "reduced_fetal_movement": ["fetal movement", "reduced movement", "हलचल कम", "हिलता नहीं"],
    "water_break_prom": ["water break", "पानी टूट", "झिल्ली"],
    "foul_vaginal_discharge": ["discharge", "बदबूदार", "स्राव"],
}

NEWBORN_CHECKLIST_SIGNS = {
    "not_feeding_well": ["not feeding", "feeding", "दूध नहीं", "दूध पीना"],
    "convulsions": ["convulsion", "seizure", "दौरा"],
    "fast_breathing_gte60": ["fast breathing", "breathing", "साँस तेज़"],
    "severe_chest_indrawing": ["chest indrawing", "छाती धँसना"],
    "high_temperature": ["high temperature", "fever", "बुखार", "तापमान"],
    "low_temperature": ["low temperature", "ठंडा", "हाइपोथर्मिया"],
    "no_movement": ["no movement", "सुस्त", "हिलता नहीं"],
    "jaundice": ["jaundice", "पीलिया"],
    "umbilicus_red_pus": ["umbilicus", "नाभि", "cord"],
}


def derive_checklists(danger_signs, visit_type):
    """Derive maternal/newborn checklists from the danger_signs array."""
    maternal_ck = {k: "not_assessed" for k in MATERNAL_CHECKLIST_SIGNS}
    newborn_ck = {k: "not_assessed" for k in NEWBORN_CHECKLIST_SIGNS}

    if not danger_signs:
        return maternal_ck, newborn_ck

    # Check each detected sign against checklist keywords
    detected_signs_text = " ".join(
        f"{s.get('sign', '')} {s.get('utterance_evidence', '')}".lower()
        for s in danger_signs
    )

    for field, keywords in MATERNAL_CHECKLIST_SIGNS.items():
        if any(kw.lower() in detected_signs_text for kw in keywords):
            maternal_ck[field] = "detected"
        else:
            maternal_ck[field] = "not_detected"

    for field, keywords in NEWBORN_CHECKLIST_SIGNS.items():
        if any(kw.lower() in detected_signs_text for kw in keywords):
            newborn_ck[field] = "detected"
        else:
            newborn_ck[field] = "not_detected"

    return maternal_ck, newborn_ck


def validate_form_output(parsed, transcript):
    """Post-extraction validation: strip hallucinated fields, apply range checks.

    Common hallucination patterns on audio transcripts:
      - patient.name = "दीदी" / "बहन" / "Patient" (generic address, not a name)
      - patient.age = 30 (model's default guess)
      - lab_results.blood_group / hiv_status invented when not discussed
    """
    if not isinstance(parsed, dict):
        return parsed

    t_lower = transcript.lower() if transcript else ""

    # -- Name hallucination: generic Hindi address terms --
    FAKE_NAMES = {"दीदी", "बहन", "बहनजी", "patient", "दी दी", "didi", "bahen"}
    patient = parsed.get("patient") or {}
    name = patient.get("name") or patient.get("patient_name")
    if name and name.strip().lower() in FAKE_NAMES:
        if "patient" in parsed and isinstance(parsed["patient"], dict):
            for key in ("name", "patient_name"):
                if key in parsed["patient"]:
                    parsed["patient"][key] = None
                    print(f"[VALIDATE] Stripped hallucinated name: {name}")

    # -- Age hallucination: exactly 30 when not mentioned --
    age = patient.get("age") or patient.get("patient_age")
    if age == 30:
        # Check if "30" or "तीस" actually appears in transcript
        if "30" not in transcript and "तीस" not in transcript:
            if "patient" in parsed and isinstance(parsed["patient"], dict):
                for key in ("age", "patient_age"):
                    if key in parsed["patient"]:
                        parsed["patient"][key] = None
                        print(f"[VALIDATE] Stripped hallucinated age: 30")

    # -- Lab results hallucination: blood_group, HIV when not discussed --
    lab = parsed.get("lab_results") or {}
    BLOOD_GROUPS = {"a+", "a-", "b+", "b-", "ab+", "ab-", "o+", "o-"}
    bg = lab.get("blood_group")
    if bg and str(bg).strip().lower() in BLOOD_GROUPS:
        bg_mentioned = any(kw in t_lower for kw in ["blood group", "ब्लड ग्रुप", "खून का ग्रुप", "रक्त समूह"])
        if not bg_mentioned:
            parsed.setdefault("lab_results", {})["blood_group"] = None
            print(f"[VALIDATE] Stripped hallucinated blood_group: {bg}")

    hiv = lab.get("hiv_status") or lab.get("hiv")
    if hiv and str(hiv).strip().lower() in ("negative", "positive", "नेगेटिव", "पॉजिटिव"):
        hiv_mentioned = any(kw in t_lower for kw in ["hiv", "एचआईवी", "एड्स"])
        if not hiv_mentioned:
            for key in ("hiv_status", "hiv"):
                if key in parsed.get("lab_results", {}):
                    parsed["lab_results"][key] = None
                    print(f"[VALIDATE] Stripped hallucinated HIV: {hiv}")

    # -- Range checks on vital signs --
    RANGES = {
        "bp_systolic": (60, 250), "bp_diastolic": (30, 150),
        "weight_kg": (1, 200), "hemoglobin_gm_percent": (3, 20),
        "gestational_weeks": (1, 45), "temperature_f": (90, 110),
    }
    for section in [parsed, parsed.get("vitals", {}), parsed.get("pregnancy", {}),
                    parsed.get("anc_details", {}), parsed.get("newborn", {})]:
        if not isinstance(section, dict):
            continue
        for field, (lo, hi) in RANGES.items():
            val = section.get(field)
            if val is not None:
                try:
                    num = float(val)
                    if num < lo or num > hi:
                        section[field] = None
                        print(f"[VALIDATE] Out-of-range {field}={val} (valid: {lo}-{hi})")
                except (ValueError, TypeError):
                    pass

    return parsed


def extract_form(transcript, visit_type):
    """Extract structured form data from transcript."""
    schema = SCHEMAS.get(VISIT_TYPE_MAP.get(visit_type, "anc_visit"), SCHEMAS["anc_visit"])
    user_prompt = (
        f"Extract structured data from this ASHA home visit conversation:\n\n"
        f"{transcript}\n\n"
        f"Output JSON schema:\n{json.dumps(schema, ensure_ascii=False)}"
    )
    result = run_inference(FORM_SYSTEM_PROMPT, user_prompt)
    if result.get("parsed") and isinstance(result["parsed"], dict):
        result["parsed"] = validate_form_output(result["parsed"], transcript)
    return result


def extract_danger_signs(transcript, visit_type):
    """Extract danger signs using trimmed schema (no checklists) + post-validation."""
    schema = build_trimmed_danger_schema()
    user_prompt = (
        f"Analyze this ASHA home visit conversation for danger signs.\n\n"
        f"Visit type: {visit_type}\n\n"
        f"{transcript}\n\n"
        f"Output JSON schema:\n{json.dumps(schema, ensure_ascii=False)}"
    )
    result = run_inference(DANGER_SYSTEM_PROMPT, user_prompt)

    # Post-validation: drop danger signs whose evidence isn't in the transcript
    # or whose evidence is a generic ASHA phrase (not actual symptom description)
    GENERIC_PHRASES = [
        "कोई तकलीफ़ हो तो फ़ोन कर दीजिए",
        "कोई तकलीफ हो तो फोन कर दीजिए",
        "कोई समस्या हो तो तुरंत बताइए",
        "कोई समस्या हो तो फोन करें",
        "कोई दिक्कत हो तो",
        "अगली बार आऊँगी",
        "अगली विज़िट",
        "ठीक है दीदी, धन्यवाद",
        "ठीक है दीदी",
    ]

    # Normal vital sign readings that should NOT be flagged as danger signs
    NORMAL_INDICATORS = [
        "110/70", "120/80", "110/80", "118/76", "108/72",  # normal BP
        "बिल्कुल ठीक", "सामान्य", "नॉर्मल", "अच्छा है", "ठीक है",
        "बिल्कुल सामान्य",
    ]

    if result["parsed"] and "danger_signs" in result["parsed"]:
        validated_signs = []
        norm_transcript = re.sub(r'\s+', ' ', transcript.strip())

        for sign in result["parsed"]["danger_signs"]:
            evidence = sign.get("utterance_evidence", "")
            if not evidence or len(evidence) < 10:
                print(f"[DEBUG] dropped sign '{sign.get('sign','')}': evidence too short ({len(evidence)} chars)")
                continue

            norm_evidence = re.sub(r'\s+', ' ', evidence.strip())

            # Check against generic phrase blocklist
            is_generic = any(phrase in norm_evidence for phrase in GENERIC_PHRASES)
            if is_generic:
                print(f"[DEBUG] dropped sign '{sign.get('sign','')}': evidence is generic ASHA phrase")
                continue

            # Check if evidence describes a normal reading, not a danger sign
            is_normal = any(indicator in norm_evidence for indicator in NORMAL_INDICATORS)
            if is_normal:
                print(f"[DEBUG] dropped sign '{sign.get('sign','')}': evidence contains normal vital indicator")
                continue

            found = False
            if norm_evidence in norm_transcript:
                found = True
            elif len(norm_evidence) >= 20:
                min_chunk = min(30, len(norm_evidence))
                for i in range(0, len(norm_evidence) - min_chunk + 1):
                    chunk = norm_evidence[i:i + min_chunk]
                    if chunk in norm_transcript:
                        found = True
                        break

            if found:
                validated_signs.append(sign)
            else:
                print(f"[DEBUG] dropped sign '{sign.get('sign','')}': evidence not found in transcript")
                print(f"[DEBUG]   evidence: {repr(norm_evidence[:80])}")

        # If all remaining signs cite the same evidence, it's likely generic — drop all
        if len(validated_signs) > 1:
            evidences = set(s.get("utterance_evidence", "").strip() for s in validated_signs)
            if len(evidences) == 1:
                print(f"[DEBUG] dropped all {len(validated_signs)} signs: all cite same evidence (likely generic)")
                validated_signs = []

        dropped = len(result["parsed"]["danger_signs"]) - len(validated_signs)
        if dropped:
            print(f"[DEBUG] post-validation dropped {dropped}/{dropped + len(validated_signs)} danger signs")
        result["parsed"]["danger_signs"] = validated_signs

        if not validated_signs:
            result["parsed"]["referral_decision"] = {
                "decision": "routine_followup",
                "reason": "No danger signs detected in conversation",
            }

    # Derive checklists programmatically (instead of model generating them)
    if result["parsed"]:
        signs = result["parsed"].get("danger_signs", [])
        maternal_ck, newborn_ck = derive_checklists(signs, visit_type)
        result["parsed"]["maternal_danger_signs_checklist"] = maternal_ck
        result["parsed"]["newborn_danger_signs_checklist"] = newborn_ck

    return result


# ============================================================
# HTML FORMATTERS
# ============================================================
def status_pill(level, msg):
    css_class = {"ready": "ready", "error": "error", "processing": "processing"}.get(level, "")
    return f'<div class="status-pill {css_class}">{html_mod.escape(msg)}</div>'


def format_form_html(data, visit_type):
    """Render extracted form data as styled HTML."""
    if not data:
        return '<div class="result-card error">Failed to extract form data</div>'

    html = f'<div class="result-card"><h3>MCTS Form — {visit_type.replace("_", " ").title()}</h3>'
    html += render_dict_html(data)
    html += '</div>'
    return html


def render_dict_html(d, depth=0):
    """Recursively render a dict as nested HTML."""
    if not isinstance(d, dict):
        return f'<span class="field-value">{html_mod.escape(str(d))}</span>'

    html = '<div class="field-group">' if depth > 0 else ''
    for key, val in d.items():
        if key.startswith("$") or key in ("type", "description", "required", "items", "enum", "minimum", "maximum"):
            continue  # Skip schema metadata
        label = key.replace("_", " ").title()
        if isinstance(val, dict):
            html += f'<div class="field-row nested"><span class="field-label">{html_mod.escape(label)}</span>'
            html += render_dict_html(val, depth + 1)
            html += '</div>'
        elif isinstance(val, list):
            items = ", ".join(str(v) for v in val) if val else "—"
            html += f'<div class="field-row"><span class="field-label">{html_mod.escape(label)}</span><span class="field-value">{html_mod.escape(items)}</span></div>'
        elif val is None:
            html += f'<div class="field-row null"><span class="field-label">{html_mod.escape(label)}</span><span class="field-value null">—</span></div>'
        else:
            css = ""
            if isinstance(val, (int, float)):
                css = ' class="field-value numeric"'
            html += f'<div class="field-row"><span class="field-label">{html_mod.escape(label)}</span><span{css}>{html_mod.escape(str(val))}</span></div>'
    if depth > 0:
        html += '</div>'
    return html


def format_danger_html(data):
    """Render danger sign results as styled HTML with evidence."""
    if not data:
        return '<div class="result-card error">Failed to analyze danger signs</div>'

    signs = data.get("danger_signs", [])
    referral = data.get("referral_decision", {})
    decision = referral.get("decision", "routine_followup")

    # Referral banner
    colors = {
        "refer_immediately": ("#ef4444", "rgba(239,68,68,0.1)", "IMMEDIATE REFERRAL"),
        "refer_within_24h": ("#f59e0b", "rgba(245,158,11,0.1)", "REFER WITHIN 24H"),
        "continue_monitoring": ("#3b82f6", "rgba(59,130,246,0.1)", "CONTINUE MONITORING"),
        "routine_followup": ("#10b981", "rgba(16,185,129,0.1)", "ROUTINE FOLLOW-UP"),
    }
    color, bg, label = colors.get(decision, ("#6b7280", "rgba(107,114,128,0.1)", decision.upper()))

    html = f'<div class="danger-card" style="border-left: 3px solid {color};">'
    html += f'<div style="background: {bg}; color: {color}; padding: 10px 16px; border-radius: 10px; margin-bottom: 16px; font-weight: 700; font-size: 13px; letter-spacing: 0.5px; text-transform: uppercase;">{label}</div>'

    if referral.get("reason"):
        html += f'<div style="margin-bottom: 16px; color: #475569; font-size: 13px; line-height: 1.5;">{html_mod.escape(referral["reason"])}</div>'

    if signs:
        html += f'<div style="font-weight: 700; margin-bottom: 10px; color: #1e293b; font-size: 13px;">Danger Signs: {len(signs)}</div>'
        for s in signs:
            cat = s.get("category", "unknown")
            cat_colors = {
                "immediate_referral": "#ef4444",
                "urgent_care": "#f59e0b",
                "monitor_closely": "#3b82f6",
            }
            cat_color = cat_colors.get(cat, "#6b7280")
            sign_name = s.get("sign", "Unknown")
            evidence = s.get("utterance_evidence", "")
            confidence = s.get("confidence", 0)
            clinical_val = s.get("clinical_value", "")

            html += f'<div style="border-left: 2px solid {cat_color}; padding: 10px 14px; margin: 8px 0; background: #f8fafc; border-radius: 0 10px 10px 0;">'
            html += f'<div style="font-weight: 600; color: {cat_color}; font-size: 14px;">{html_mod.escape(sign_name)}</div>'
            html += f'<div style="font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.3px; margin-top: 2px;">{cat.replace("_", " ")}</div>'
            if clinical_val:
                html += f'<div style="font-size: 13px; color: #1e293b; margin-top: 6px; font-variant-numeric: tabular-nums;">Value: <strong>{html_mod.escape(str(clinical_val))}</strong></div>'
            if evidence:
                html += f'<div style="font-size: 12px; color: #64748b; margin-top: 6px; font-style: italic; border-left: 2px solid #e2e8f0; padding-left: 10px;">"{html_mod.escape(evidence)}"</div>'
            html += '</div>'
    else:
        html += '<div style="color: #059669; font-weight: 600; font-size: 13px; padding: 14px; background: #ecfdf5; border: 1px solid #a7f3d0; border-radius: 10px; text-align: center;">No danger signs detected</div>'

    # Checklists
    for ck_name, ck_label in [("maternal_danger_signs_checklist", "Maternal Checklist"), ("newborn_danger_signs_checklist", "Newborn Checklist")]:
        ck = data.get(ck_name, {})
        if ck and any(v != "not_assessed" for v in ck.values()):
            html += f'<div style="margin-top: 16px;"><div style="font-weight: 600; font-size: 12px; color: #94a3b8; text-transform: uppercase; margin-bottom: 6px;">{ck_label}</div>'
            for field, status in ck.items():
                icon = {"detected": "🔴", "not_detected": "🟢", "not_assessed": "⚪"}.get(status, "⚪")
                html += f'<div style="font-size: 12px; padding: 2px 0;">{icon} {field.replace("_", " ").title()}: {status.replace("_", " ")}</div>'
            html += '</div>'

    html += '</div>'
    return html


# ============================================================
# HANDLER FUNCTIONS
# ============================================================
def process_transcript(transcript, visit_type_override):
    """Main handler: transcript → form + danger signs (two-pass, trimmed schemas)."""
    if not transcript or not transcript.strip():
        return (
            status_pill("error", "No transcript provided"),
            "", "", ""
        )

    yield (
        status_pill("processing", "Detecting visit type..."),
        "", "", ""
    )

    # Detect visit type
    if visit_type_override and visit_type_override != "Auto-detect":
        visit_type = visit_type_override.lower().replace(" ", "_")
    else:
        visit_type = detect_visit_type(transcript)

    yield (
        status_pill("processing", f"Extracting form data ({visit_type})..."),
        "", "", ""
    )

    # Pass 1: Form extraction
    form_result = extract_form(transcript, visit_type)
    form_html = format_form_html(form_result["parsed"], visit_type)

    yield (
        status_pill("processing", "Analyzing danger signs..."),
        form_html, "", ""
    )

    # Pass 2: Danger signs (trimmed schema, no checklists)
    danger_result = extract_danger_signs(transcript, visit_type)
    danger_html = format_danger_html(danger_result["parsed"])

    total_time = form_result["time_s"] + danger_result["time_s"]
    elapsed = f'{total_time:.1f}s'

    yield (
        status_pill("ready", f"Complete — {visit_type.replace('_', ' ').title()} — {elapsed}"),
        form_html, danger_html, elapsed
    )


def process_audio(audio_path, visit_type_override):
    """Audio → transcript → form + danger signs using Gemma 4 E4B native audio."""
    if not audio_path:
        yield (status_pill("error", "No audio provided"), "", "", "", "")
        return

    yield (status_pill("processing", "Transcribing Hindi audio..."), "", "", "", "")

    try:
        transcript = transcribe_audio(audio_path)
        if not transcript or not transcript.strip():
            yield (status_pill("error", "Transcription returned empty — try speaking louder or closer to mic"), "", "", "", "")
            return
    except Exception as e:
        print(f"[ASR] Error: {e}")
        yield (status_pill("error", f"ASR failed: {e}"), "", "", "", "")
        return

    yield (status_pill("processing", "Extracting from transcript..."), transcript, "", "", "")

    # Run extraction pipeline
    for status, form_html, danger_html, total_time in process_transcript(transcript, visit_type_override):
        yield (status, transcript, form_html, danger_html, total_time)


def load_example(example_idx):
    """Load an example transcript."""
    if example_idx is None:
        return ""
    idx = int(example_idx)
    if 0 <= idx < len(EXAMPLE_TRANSCRIPTS):
        return EXAMPLE_TRANSCRIPTS[idx][1]
    return ""


def set_uploaded_audio(uploaded_file):
    """Normalize UploadButton payload to a filepath for gr.Audio(type='filepath')."""
    if not uploaded_file:
        return None
    if isinstance(uploaded_file, str):
        return uploaded_file
    file_name = getattr(uploaded_file, "name", None)
    if file_name:
        return file_name
    if isinstance(uploaded_file, dict):
        return uploaded_file.get("path") or uploaded_file.get("name")
    return None


# ============================================================
# CSS  — minimal: only styles custom HTML output cards
# Gradio 6 theme handles all native widget colours.
# ============================================================
CUSTOM_CSS = """
html {
    overflow-y: scroll !important;
}

/* ── App shell ── */
.gradio-container {
    max-width: 1120px !important;
    margin: 0 auto !important;
    padding: 18px 22px 26px !important;
}

/* ── Hero Header ── */
.hero-header {
    text-align: center;
    padding: 32px 20px 20px;
    margin-bottom: 10px;
    position: relative;
}
.hero-header::after {
    content: '';
    position: absolute;
    bottom: 0; left: 50%;
    transform: translateX(-50%);
    width: 80px; height: 3px;
    background: linear-gradient(90deg, transparent, #0d9488, transparent);
    border-radius: 2px;
}
.hero-header h1 {
    font-size: 34px;
    font-weight: 800;
    margin: 0;
    background: linear-gradient(135deg, #0d9488 0%, #059669 50%, #10b981 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.5px;
}
.hero-header .tagline {
    font-size: 14px;
    color: #475569;
    margin: 8px 0 0;
}
.hero-header .tech-badge {
    display: inline-block;
    margin-top: 10px;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    color: #0d9488;
    background: rgba(13,148,136,0.08);
    border: 1px solid rgba(16,185,129,0.2);
    letter-spacing: 0.5px;
}

/* ── Tabs ── */
.tabs {
    border-bottom: 1px solid #dbe4ed !important;
    margin-bottom: 14px !important;
}
.tabs button {
    border-radius: 10px 10px 0 0 !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 10px 14px !important;
}
.tabs button.selected {
    color: #0f766e !important;
    background: rgba(13,148,136,0.08) !important;
}

/* ── Section cards ── */
#voice-primary-card,
#voice-transcript-card,
#text-input-card,
#voice-results-card,
#text-results-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    padding: 14px;
}
#voice-primary-card,
#text-input-card {
    margin-top: 10px;
}
#voice-transcript-card,
#text-results-card {
    margin-top: 12px;
}

/* ── Status pills ── */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    border-radius: 20px;
    padding: 6px 16px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.3px;
    border: 1px solid #e2e8f0;
    background: #ffffff;
    color: #94a3b8;
}
.status-pill::before {
    content: '';
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #94a3b8;
}
.status-pill.ready { background:#ecfdf5; border-color:#a7f3d0; color:#065f46; }
.status-pill.ready::before { background: #059669; }
.status-pill.error { background:#fef2f2; border-color:#fecaca; color:#991b1b; }
.status-pill.error::before { background: #dc2626; }
.status-pill.processing {
    background: #eff6ff;
    border-color: #bfdbfe;
    color: #1e40af;
    animation: pulse-glow 2s ease-in-out infinite;
}
.status-pill.processing::before {
    background: #2563eb;
    animation: dot-pulse 1s ease-in-out infinite;
}
@keyframes pulse-glow {
    0%,100% { box-shadow: 0 0 0 0 rgba(37,99,235,0); }
    50%      { box-shadow: 0 0 12px 2px rgba(37,99,235,0.10); }
}
@keyframes dot-pulse {
    0%,100% { opacity:1; transform:scale(1); }
    50%     { opacity:.4; transform:scale(.7); }
}

/* ── Result / danger cards ── */
.result-card, .danger-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 20px 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,.06), 0 4px 16px rgba(0,0,0,.04);
}
.result-card h3 {
    margin: 0 0 16px 0;
    color: #0d9488;
    font-size: 14px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    padding-bottom: 12px;
    border-bottom: 1px solid #e2e8f0;
}
.result-card.error {
    border-color: rgba(239,68,68,.3);
    background: rgba(239,68,68,.05);
    color: #f87171;
}

/* ── Field rows (inside result cards) ── */
.field-row {
    display: flex;
    padding: 6px 0;
    font-size: 13px;
    border-bottom: 1px solid #f1f5f9;
}
.field-row.null { opacity: 0.3; }
.field-label {
    font-weight: 600;
    color: #475569;
    min-width: 200px;
    flex-shrink: 0;
    font-size: 12px;
}
.field-value { color: #1e293b; }
.field-value.null { color: #94a3b8; font-style: italic; }
.field-value.numeric { color: #0d9488; font-weight: 700; font-variant-numeric: tabular-nums; }
.field-group { margin-left: 16px; border-left: 2px solid #d1dde6; padding-left: 14px; }
.field-row.nested > .field-label {
    color: #0d9488; font-size: 12px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.3px; padding-top: 8px;
}

/* ── About section ── */
.about-section { font-size: 14px; line-height: 1.8; color: #475569; }
.about-section h2 { color: #0d9488; font-size: 22px; margin-top: 28px; font-weight: 700; }
.about-section h3 { color: #1e293b; font-size: 16px; font-weight: 600; margin-top: 20px; }
.about-section code {
    background: #f8fafb; padding: 2px 8px; border-radius: 6px;
    font-size: 12px; color: #0d9488; border: 1px solid #e2e8f0;
}
.about-section table {
    background: #fff; border-radius: 10px;
    overflow: hidden; width: 100%;
    border-collapse: separate; border-spacing: 0;
}
.about-section td { color: #475569; padding: 10px 14px !important; border-bottom: 1px solid #e2e8f0; }
.about-section tr:last-child td { border-bottom: none; }
.about-section td:first-child { color: #1e293b; font-weight: 600; }
.about-section strong { color: #1e293b; }
.about-section ol, .about-section ul { padding-left: 20px; }
.about-section li { margin: 6px 0; }

/* ── Voice tab polish (strict no-jump build) ── */
#voice-primary-card {
    width: min(100%, 760px) !important;
    max-width: 760px !important;
    margin: 10px auto 0 !important;
    overflow: hidden !important;
    min-width: 0 !important;
}
#audio-shell {
    width: 100% !important;
    max-width: 100% !important;
    min-width: 0 !important;
}
#audio-input {
    border: 1px solid #d7e1ea;
    border-radius: 12px;
    padding: 12px;
    background: #fbfdff;
    width: 100% !important;
    max-width: 100% !important;
    min-width: 0 !important;
    min-height: 128px;
    overflow: hidden !important;
    box-sizing: border-box !important;
}
#audio-input > div,
#audio-input [data-testid="waveform-container"],
#audio-input .wrap,
#audio-input .waveform-container {
    width: 100% !important;
    max-width: 100% !important;
    min-width: 0 !important;
    overflow: hidden !important;
}
#audio-input [data-testid="audio"] {
    width: 100% !important;
    max-width: 100% !important;
    min-width: 0 !important;
    min-height: 112px;
    height: auto !important;
    overflow: visible !important;
}
#audio-placeholder {
    height: 8px;
}
#audio-upload-wrap {
    margin-top: 8px;
    width: 100% !important;
    max-width: 100% !important;
    min-width: 0 !important;
}
#audio-upload-btn {
    width: 100% !important;
    max-width: 100% !important;
    min-width: 0 !important;
}
#audio-upload-btn button {
    min-height: 40px !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    width: 100% !important;
    max-width: 100% !important;
    min-width: 0 !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}

#voice-controls {
    align-items: stretch;
    gap: 12px;
    margin-top: 10px;
    width: 100% !important;
    max-width: 100% !important;
    min-width: 0 !important;
}
#voice-transcript-card,
#voice-results-card {
    width: min(100%, 760px) !important;
    max-width: 760px !important;
    margin-left: auto !important;
    margin-right: auto !important;
}
#visit-type-audio,
#process-audio-btn {
    min-height: 48px;
    min-width: 0 !important;
}
#visit-type-audio button,
#process-audio-btn button {
    min-height: 48px !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}
#process-audio-btn button {
    width: 100%;
    font-size: 15px;
}
#voice-primary-card .gr-row > *,
#voice-primary-card .gr-column > * {
    min-width: 0 !important;
    max-width: 100% !important;
}
#visit-type-audio > label,
#text-visit-type > label,
#text-example > label,
#audio-transcript > label {
    font-size: 12px !important;
    font-weight: 600 !important;
    color: #475569 !important;
}
#text-controls {
    gap: 10px;
}
#text-extract-btn button {
    min-height: 46px !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}

footer { opacity: 0.3; }
"""


# ============================================================
# BUILD APP
# ============================================================
def build_app():
    init_schemas()

    theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.teal,
        secondary_hue=gr.themes.colors.green,
        neutral_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("Inter"),
    ).set(
        body_background_fill="#f4f7fa",
        block_background_fill="#ffffff",
        block_border_color="#e2e8f0",
        input_background_fill="#f8fafb",
        button_primary_background_fill="linear-gradient(135deg, #0d9488 0%, #059669 100%)",
        button_primary_background_fill_hover="linear-gradient(135deg, #0f766e 0%, #047857 100%)",
        button_primary_text_color="#ffffff",
    )

    with gr.Blocks(
        title="Sakhi (सखी) — ASHA Health Companion",
        theme=theme,
        css=CUSTOM_CSS,
    ) as app:

        # ── Header ──
        gr.HTML("""
        <div class="hero-header">
            <h1>Sakhi (सखी)</h1>
            <div class="tagline">AI companion for India's 1 million ASHA health workers</div>
            <div class="tech-badge">Gemma 4 E4B  ·  Offline-First  ·  Hindi Voice</div>
        </div>
        """)

        with gr.Tabs():

            # ── TAB 1: Voice to Form ──
            with gr.Tab("🎙️ Voice to Form", id="voice"):
                gr.Markdown("Record or upload a Hindi ASHA home visit conversation.")

                with gr.Group(elem_id="voice-primary-card"):
                    with gr.Row():
                        with gr.Column(scale=1, min_width=0, elem_id="audio-shell"):
                            audio_input = gr.Audio(
                                label="Audio Input",
                                show_label=False,
                                type="filepath",
                                sources=["microphone"],
                                elem_id="audio-input",
                            )
                            gr.HTML("<div id='audio-placeholder'></div>")
                            with gr.Row(elem_id="audio-upload-wrap"):
                                audio_upload_btn = gr.UploadButton(
                                    "Upload Audio File",
                                    file_types=["audio"],
                                    file_count="single",
                                    elem_id="audio-upload-btn",
                                )
                    with gr.Row(elem_id="voice-controls"):
                        visit_type_audio = gr.Dropdown(
                            choices=["Auto-detect", "ANC Visit", "PNC Visit", "Delivery", "Child Health"],
                            value="Auto-detect",
                            label="Visit Type",
                            scale=2,
                            elem_id="visit-type-audio",
                        )
                        audio_btn = gr.Button(
                            "Process Audio",
                            variant="primary",
                            size="lg",
                            scale=1,
                            elem_id="process-audio-btn",
                        )
                    audio_status = gr.HTML(value=status_pill("ready", "Ready"))
                    audio_upload_btn.upload(
                        fn=set_uploaded_audio,
                        inputs=[audio_upload_btn],
                        outputs=[audio_input],
                    )

                with gr.Group(elem_id="voice-transcript-card"):
                    audio_transcript = gr.Textbox(label="Transcript", lines=5, interactive=False, elem_id="audio-transcript")
                audio_time = gr.Textbox(visible=False)

                # Outputs — side by side (these are wide enough to share a row)
                with gr.Row(equal_height=False, elem_id="voice-results-card"):
                    audio_form = gr.HTML(label="Form Extraction")
                    audio_danger = gr.HTML(label="Danger Signs")

                audio_btn.click(
                    fn=process_audio,
                    inputs=[audio_input, visit_type_audio],
                    outputs=[audio_status, audio_transcript, audio_form, audio_danger, audio_time],
                )

            # ── TAB 2: Text to Form ──
            with gr.Tab("📋 Text to Form", id="text"):
                gr.Markdown("Paste a Hindi ASHA home visit conversation transcript.")

                with gr.Group(elem_id="text-input-card"):
                    text_input = gr.Textbox(
                        label="Hindi Transcript",
                        placeholder="Paste ASHA home visit conversation here...",
                        lines=10,
                    )
                    with gr.Row(elem_id="text-controls"):
                        visit_type_text = gr.Dropdown(
                            choices=["Auto-detect", "ANC Visit", "PNC Visit", "Delivery", "Child Health"],
                            value="Auto-detect",
                            label="Visit Type",
                            scale=1,
                            elem_id="text-visit-type",
                        )
                        example_dropdown = gr.Dropdown(
                            choices=[("-- Select Example --", None)] + [(ex[0], i) for i, ex in enumerate(EXAMPLE_TRANSCRIPTS)],
                            value=None,
                            label="Load Example",
                            type="value",
                            scale=1,
                            elem_id="text-example",
                        )
                    text_btn = gr.Button("Extract Structured Form", variant="primary", size="lg", elem_id="text-extract-btn")
                    text_status = gr.HTML(value=status_pill("ready", "Ready"))
                text_time = gr.Textbox(visible=False)

                example_dropdown.change(fn=load_example, inputs=[example_dropdown], outputs=[text_input])

                with gr.Row(equal_height=False, elem_id="text-results-card"):
                    text_form = gr.HTML(label="Form Extraction")
                    text_danger = gr.HTML(label="Danger Signs")

                text_btn.click(
                    fn=process_transcript,
                    inputs=[text_input, visit_type_text],
                    outputs=[text_status, text_form, text_danger, text_time],
                )

            # ── TAB 3: About ──
            with gr.Tab("About", id="about"):
                gr.HTML("""
                <div class="about-section">
                <h2>Sakhi (सखी) — For India's 1 Million ASHA Workers</h2>
                <p>
                    India's <strong>Accredited Social Health Activists (ASHA)</strong> conduct millions of home visits
                    annually — antenatal care, postnatal checks, newborn assessments, child health visits. Each visit
                    requires filling government <strong>MCTS/HMIS forms</strong> and detecting <strong>NHM-defined danger signs</strong>
                    that require urgent referral.
                </p>
                <p>
                    Today, this is done with pen and paper. Critical danger signs are missed. Forms are filled hours
                    later from memory. Data never reaches the health system in time.
                </p>

                <h3>How It Works</h3>
                <p>
                    Sakhi uses <strong>Gemma 4 E4B</strong> with structured prompting
                    to extract structured data from Hindi home visit conversations:
                </p>
                <ol>
                    <li><strong>Audio Input</strong> — ASHA records the conversation on her phone</li>
                    <li><strong>Transcription</strong> — Hindi speech is transcribed (Whisper / Gemma 4 native audio)</li>
                    <li><strong>Form Extraction</strong> — Transcript → structured MCTS/HMIS JSON (patient info, vitals, history, birth preparedness)</li>
                    <li><strong>Danger Sign Detection</strong> — NHM-defined danger signs flagged with exact utterance evidence</li>
                    <li><strong>Referral Decision</strong> — Automatic triage: routine, monitor, refer within 24h, or refer immediately</li>
                </ol>

                <h3>Anti-Hallucination by Design</h3>
                <p>
                    Every danger sign <strong>MUST cite the exact Hindi utterance</strong> that triggered it.
                    No citation = no flag. The model was trained on 40% negative examples (no danger signs)
                    to teach restraint. Zero polarity errors across 402 validated training samples.
                </p>

                <h3>Architecture</h3>
                <table style="font-size: 13px; border-collapse: collapse; width: 100%;">
                    <tr style="border-bottom: 1px solid #e2e8f0;">
                        <td style="padding: 6px; font-weight: 600;">Base Model</td>
                        <td style="padding: 6px;">Gemma 4 E4B (4.5B effective / 8B total)</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #e2e8f0;">
                        <td style="padding: 6px; font-weight: 600;">Fine-tuning</td>
                        <td style="padding: 6px;">Unsloth LoRA (r=16, 42M params, 0.7% of total)</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #e2e8f0;">
                        <td style="padding: 6px; font-weight: 600;">Training Data</td>
                        <td style="padding: 6px;">402 validated Hindi ASHA conversations (GPT-4o Mini synthetic, $0.78)</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #e2e8f0;">
                        <td style="padding: 6px; font-weight: 600;">Training</td>
                        <td style="padding: 6px;">3 epochs, eval loss 0.71, train loss 3.01</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #e2e8f0;">
                        <td style="padding: 6px; font-weight: 600;">Output</td>
                        <td style="padding: 6px;">Structured JSON (5 schemas: ANC, PNC, delivery, child health, danger signs)</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #e2e8f0;">
                        <td style="padding: 6px; font-weight: 600;">Deployment</td>
                        <td style="padding: 6px;">4-bit quantized, runs on consumer GPU. Fully offline.</td>
                    </tr>
                </table>

                <h3>Prize Tracks</h3>
                <ul>
                    <li><strong>Main Track</strong> — Real-world impact for 1M+ health workers</li>
                    <li><strong>Health &amp; Sciences</strong> — Clinical extraction + danger sign detection</li>
                    <li><strong>Digital Equity</strong> — Hindi-first, rural, offline, closing the AI gap</li>
                    <li><strong>Unsloth</strong> — Fine-tuned with Unsloth LoRA for domain-specific task</li>
                    <li><strong>Ollama</strong> — Local deployment via Ollama</li>
                </ul>
                </div>
                """)

    return app


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )