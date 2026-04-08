"""
Hackathon-grade quality test: 15 diverse Hindi medical transcripts.

Tests form extraction + danger sign detection across all 4 visit types.
Checks: value accuracy, hallucination, false positives, false negatives,
code-switching, unlabeled audio, edge cases.

Each test uses the correct schema for its visit type.
"""
import json
import os
import re
import sys
import time

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")

import ollama

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
# 15 TEST CASES
# ============================================================
# Each: (name, visit_type, schema_name, transcript,
#         expected_form_checks, expected_danger_min, expected_danger_max,
#         expected_referral, hallucination_traps)
#
# expected_form_checks: dict of {json_path: expected_value}
#   use dotted paths like "vitals.bp_systolic"
# hallucination_traps: list of field paths that MUST be null

TESTS = [
    # ── ANC CASES ──
    # 1. ANC Normal — all vitals mentioned, labeled speakers
    (
        "ANC Normal — full vitals",
        "anc_visit", "anc_visit",
        (
            "ASHA: नमस्ते, कैसे हैं आप?\n"
            "Patient: नमस्ते दीदी, मैं ठीक हूँ।\n"
            "ASHA: आपका BP 110/70 है, बिल्कुल ठीक है। वजन 58 kg है। Hb 11.5 आया था।\n"
            "ASHA: आप 24 हफ्ते की हैं। IFA रोज़ ले रही हैं? TT पहला लग गया।\n"
            "Patient: हाँ दीदी। डिलीवरी PHC में करवाएँगे।"
        ),
        {
            "vitals.bp_systolic": 110, "vitals.bp_diastolic": 70,
            "vitals.weight_kg": 58, "vitals.hemoglobin_gm_percent": 11.5,
            "pregnancy.gestational_weeks": 24,
            "pregnancy.expected_delivery_place": "PHC",
        },
        0, 0, "routine_followup",
        ["patient.name", "patient.age", "lab_results.blood_group", "lab_results.hiv_status"],
    ),
    # 2. ANC Preeclampsia — multiple danger signs
    (
        "ANC Preeclampsia — multi-danger",
        "anc_visit", "anc_visit",
        (
            "ASHA: नमस्ते दीदी, कैसे हैं?\n"
            "Patient: दीदी, बहुत सिरदर्द हो रहा है। आँखों के सामने धुंधला दिखता है।\n"
            "Patient: चेहरे पर सूजन आ गई है।\n"
            "ASHA: BP चेक करती हूँ... 155/100 आ रहा है। बहुत ज़्यादा है।\n"
            "Patient: पैरों में भी काफी सूजन है।\n"
            "ASHA: आपको तुरंत PHC जाना होगा। आप 8 महीने की हैं।"
        ),
        {"vitals.bp_systolic": 155, "vitals.bp_diastolic": 100},
        2, 5, "refer_immediately",
        ["patient.name", "lab_results.blood_group"],
    ),
    # 3. ANC Severe anemia — low Hb
    (
        "ANC Severe Anemia",
        "anc_visit", "anc_visit",
        (
            "ASHA: Hb report आया?\n"
            "Patient: हाँ, 6.5 आया है। बहुत कम है। चक्कर आते हैं। साँस लेने में तकलीफ़ होती है।\n"
            "ASHA: BP 100/60 है। वजन 45 kg। आप 20 हफ्ते की हैं।\n"
            "ASHA: आपको PHC में आयरन injection लेना होगा।"
        ),
        {
            "vitals.bp_systolic": 100, "vitals.bp_diastolic": 60,
            "vitals.weight_kg": 45, "vitals.hemoglobin_gm_percent": 6.5,
            "pregnancy.gestational_weeks": 20,
        },
        1, 3, "refer_immediately",
        ["patient.name", "lab_results.blood_group"],
    ),
    # 4. ANC — only partial info mentioned
    (
        "ANC Partial Info — sparse transcript",
        "anc_visit", "anc_visit",
        (
            "ASHA: BP ठीक है, 118/76 है।\n"
            "Patient: ठीक है दीदी।"
        ),
        {"vitals.bp_systolic": 118, "vitals.bp_diastolic": 76},
        0, 0, "routine_followup",
        ["patient.name", "patient.age", "vitals.weight_kg", "vitals.hemoglobin_gm_percent",
         "pregnancy.gestational_weeks", "lab_results.blood_group", "lab_results.hiv_status"],
    ),
    # 5. ANC Unlabeled — no speaker labels (realistic ASR output)
    (
        "ANC Unlabeled ASR output",
        "anc_visit", "anc_visit",
        (
            "नमस्ते कैसे हैं BP check करती हूँ BP 120/80 है normal है "
            "weight 55 kg है Hb test करवाया था 10.2 आया था थोड़ा low है "
            "IFA रोज़ लेना गर्भ 28 weeks का है delivery के लिए district hospital जाएँगे"
        ),
        {
            "vitals.bp_systolic": 120, "vitals.bp_diastolic": 80,
            "vitals.weight_kg": 55, "vitals.hemoglobin_gm_percent": 10.2,
            "pregnancy.gestational_weeks": 28,
        },
        0, 0, "routine_followup",
        ["patient.name", "lab_results.blood_group"],
    ),
    # 6. ANC Hinglish heavy — code-switching
    (
        "ANC Hinglish heavy code-switch",
        "anc_visit", "anc_visit",
        (
            "ASHA: Hello didi, aaj check-up hai. BP check karti hoon. 130/85 hai, thoda high.\n"
            "Patient: Koi problem hai kya?\n"
            "ASHA: Abhi nahi, but monitor karna hoga. Weight 62 kg. Hb report mein 9.8 aaya.\n"
            "ASHA: Aap 32 weeks ki hain. Baby ki movement kaisi hai?\n"
            "Patient: Bahut move karta hai.\n"
            "ASHA: Good. Delivery ke liye district hospital ready hai?"
        ),
        {
            "vitals.bp_systolic": 130, "vitals.bp_diastolic": 85,
            "vitals.weight_kg": 62, "vitals.hemoglobin_gm_percent": 9.8,
            "pregnancy.gestational_weeks": 32,
        },
        0, 1, "routine_followup",  # BP 130/85 is borderline, 0-1 flags acceptable
        ["patient.name", "lab_results.blood_group"],
    ),
    # 7. ANC with named patient — name should be extracted
    (
        "ANC with patient name Sunita",
        "anc_visit", "anc_visit",
        (
            "ASHA: नमस्ते सुनीता जी, आज का चेकअप करते हैं।\n"
            "सुनीता: नमस्ते दीदी। मेरी उम्र 25 साल है।\n"
            "ASHA: BP 116/74 है। वजन 54 kg। Hb 12.0 है। बहुत अच्छा।\n"
            "ASHA: 30 हफ्ते की हैं। सब ठीक चल रहा है।"
        ),
        {
            "patient.name": "सुनीता",
            "patient.age": 25,
            "vitals.bp_systolic": 116, "vitals.bp_diastolic": 74,
            "vitals.weight_kg": 54, "vitals.hemoglobin_gm_percent": 12.0,
            "pregnancy.gestational_weeks": 30,
        },
        0, 0, "routine_followup",
        ["lab_results.blood_group", "lab_results.hiv_status"],
    ),

    # ── PNC CASES ──
    # 8. PNC Normal — mother and baby fine
    (
        "PNC Normal — day 7",
        "pnc_visit", "pnc_visit",
        (
            "ASHA: नमस्ते दीदी। डिलीवरी को 7 दिन हो गए। आप कैसे हैं?\n"
            "Mother: मैं ठीक हूँ। बच्चा अच्छे से दूध पी रहा है।\n"
            "ASHA: बच्चे का वजन 3.1 kg है। नाभि सूखी है। तापमान सामान्य है।\n"
            "ASHA: आपका BP 118/76 है। खून बहना बंद हो गया?\n"
            "Mother: हाँ, अब बहुत कम है।"
        ),
        {
            "visit_info.visit_day": 7,
            "infant_assessment.weight_kg": 3.1,
        },
        0, 0, "routine_followup",
        [],
    ),
    # 9. PNC Danger — newborn not feeding + fever
    (
        "PNC Danger — newborn not feeding",
        "pnc_visit", "pnc_visit",
        (
            "ASHA: बच्चा कैसा है?\n"
            "Mother: दीदी, बच्चा बहुत सोता रहता है। दूध ठीक से नहीं पीता। 12 घंटे से दूध नहीं पिया।\n"
            "ASHA: बच्चे का रोना कैसा है?\n"
            "Mother: बहुत कमज़ोर आवाज़ में रोता है।\n"
            "ASHA: तापमान 100.5 डिग्री है। बुखार है। बच्चा सुस्त लग रहा है।\n"
            "ASHA: ये danger signs हैं। तुरंत PHC ले जाना होगा।"
        ),
        {"infant_assessment.temperature": 100.5},
        1, 4, "refer_immediately",
        [],
    ),
    # 10. PNC — heavy postpartum bleeding (maternal danger)
    (
        "PNC Danger — postpartum bleeding",
        "pnc_visit", "pnc_visit",
        (
            "ASHA: डिलीवरी को 3 दिन हुए। कैसे हैं?\n"
            "Mother: दीदी, बहुत ज़्यादा खून आ रहा है। pad 1 घंटे में भीग जाता है।\n"
            "Mother: चक्कर भी आ रहे हैं। बहुत कमज़ोरी है।\n"
            "ASHA: ये बहुत गंभीर है। तुरंत hospital जाना होगा।"
        ),
        {"visit_info.days_since_delivery": 3},
        1, 3, "refer_immediately",
        [],
    ),

    # ── DELIVERY CASES ──
    # 11. Delivery — normal institutional
    (
        "Delivery Normal — institutional",
        "delivery", "delivery",
        (
            "ASHA: डिलीवरी कब हुई?\n"
            "Mother: कल रात 3 बजे। लड़का हुआ है।\n"
            "ASHA: कहाँ हुई डिलीवरी?\n"
            "Mother: PHC में। normal delivery थी।\n"
            "ASHA: बच्चे का वजन?\n"
            "Mother: 2.8 kg है।\n"
            "ASHA: स्तनपान शुरू किया?\n"
            "Mother: हाँ, तुरंत शुरू किया। एक घंटे के अंदर।"
        ),
        {
            "delivery.place": "PHC",
            "delivery.type": "normal",
            "infant.sex": "male",
            "infant.birth_weight_kg": 2.8,
            "infant.breastfed_within_1hr": True,
        },
        0, 0, "routine_followup",
        [],
    ),
    # 12. Delivery — home delivery, low birth weight
    (
        "Delivery — home, LBW baby",
        "delivery", "delivery",
        (
            "ASHA: बच्चा कहाँ हुआ?\n"
            "Mother: घर पर ही हो गया। दाई ने करवाया। लड़की हुई है।\n"
            "ASHA: बच्ची का वजन बहुत कम है, 1.8 kg। ये low birth weight है।\n"
            "Mother: हाँ, बच्ची बहुत छोटी है।\n"
            "ASHA: बच्ची ने जन्म के समय रोया?\n"
            "Mother: हाँ, रोई थी।\n"
            "ASHA: बच्ची को गर्म रखना ज़रूरी है। PHC में चेकअप करवाना होगा।"
        ),
        {
            "delivery.place": "home",
            "infant.sex": "female",
            "infant.birth_weight_kg": 1.8,
            "infant.cried_at_birth": True,
        },
        1, 2, "refer_immediately",
        [],
    ),

    # ── CHILD HEALTH CASES ──
    # 13. Child health — routine, healthy
    (
        "Child Health — routine 9 months",
        "child_health", "child_health",
        (
            "ASHA: बच्चा कैसा है?\n"
            "Mother: बिल्कुल ठीक है दीदी। खूब खाता है, खेलता है।\n"
            "ASHA: वजन 8.5 kg है। 9 महीने के लिए अच्छा है।\n"
            "ASHA: Vitamin A दी थी? हाँ, 6 महीने में पहली dose दी थी।\n"
            "ASHA: टीके सब लगे हैं। बच्चा बैठता है, घुटनों पर चलता है। बढ़िया।"
        ),
        {
            "child.age_months": 9,
            "growth_assessment.weight_kg": 8.5,
            "immunization.up_to_date": True,
        },
        0, 0, "routine_followup",
        [],
    ),
    # 14. Child health — sick child, diarrhea + dehydration
    (
        "Child Health — diarrhea danger",
        "child_health", "child_health",
        (
            "ASHA: बच्चे को क्या हुआ?\n"
            "Mother: 3 दिन से दस्त लग रहे हैं। बहुत पतले पानी जैसे।\n"
            "Mother: खाना-पीना बंद कर दिया है। बहुत सुस्त हो गया है।\n"
            "ASHA: बच्चे का वजन 6.2 kg है। 12 महीने का है।\n"
            "ASHA: आँखें धँसी हुई हैं। ये dehydration के signs हैं। तुरंत PHC जाना होगा।"
        ),
        {
            "child.age_months": 12,
            "growth_assessment.weight_kg": 6.2,
            "illness_assessment.diarrhea": True,
            "illness_assessment.diarrhea_duration_days": 3,
        },
        1, 3, "refer_immediately",
        [],
    ),

    # ── EDGE CASES ──
    # 15. ANC — normal visit with ZERO concerning findings (false positive trap)
    (
        "ANC Zero Findings — false positive trap",
        "anc_visit", "anc_visit",
        (
            "ASHA: सब ठीक है दीदी?\n"
            "Patient: हाँ दीदी, बिल्कुल ठीक हूँ। कोई तकलीफ़ नहीं।\n"
            "ASHA: बहुत अच्छा। अगली बार आऊँगी। कोई तकलीफ़ हो तो फ़ोन कर दीजिए।\n"
            "Patient: ठीक है दीदी, धन्यवाद।"
        ),
        {},  # No vitals to check — nothing was measured
        0, 0, "routine_followup",
        ["patient.name", "patient.age", "vitals.bp_systolic", "vitals.weight_kg",
         "vitals.hemoglobin_gm_percent", "pregnancy.gestational_weeks",
         "lab_results.blood_group", "lab_results.hiv_status"],
    ),
]


def load_schemas():
    schemas = {}
    for name in ["anc_visit", "pnc_visit", "delivery", "child_health", "danger_signs"]:
        with open(f"configs/schemas/{name}.json", encoding="utf-8") as f:
            schemas[name] = json.load(f)
    return schemas


def get_nested(d, path):
    """Get value from dict using dotted path like 'vitals.bp_systolic'."""
    parts = path.split(".")
    for p in parts:
        if not isinstance(d, dict):
            return None
        d = d.get(p)
    return d


def parse_json_response(raw):
    clean = raw.strip().lstrip('\ufeff')
    clean = re.sub(r'^`{3,}\s*(?:json)?\s*[\r\n]*', '', clean, flags=re.IGNORECASE)
    clean = re.sub(r'[\r\n]*`{3,}\s*$', '', clean).strip()
    clean = re.sub(r',\s*([}\]])', r'\1', clean)
    if clean and clean[0] not in ('{', '['):
        idx = min(
            (clean.find("{") if clean.find("{") >= 0 else len(clean)),
            (clean.find("[") if clean.find("[") >= 0 else len(clean)),
        )
        if idx < len(clean):
            clean = clean[idx:]
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        for end in range(len(clean), max(0, len(clean) - 200), -1):
            if clean[end - 1] in ('}', ']'):
                try:
                    return json.loads(clean[:end])
                except json.JSONDecodeError:
                    continue
    return None


def run_all_tests(model):
    schemas = load_schemas()
    total_pass = 0
    total_fail = 0
    total_time = 0
    issues = []

    for (name, visit_type, schema_name, transcript,
         expected_form, danger_min, danger_max, expected_referral,
         must_be_null) in TESTS:

        schema = schemas[schema_name]
        danger_schema = schemas["danger_signs"]

        # ── Form extraction ──
        form_user = (
            f"Extract structured data from this ASHA home visit conversation:\n\n"
            f"{transcript}\n\n"
            f"Output JSON schema:\n{json.dumps(schema, ensure_ascii=False)}"
        )

        t0 = time.time()
        resp = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": FORM_SYSTEM_PROMPT},
                {"role": "user", "content": form_user},
            ],
            options={"temperature": 0.0, "num_ctx": 4096},
        )
        form_time = time.time() - t0
        form_parsed = parse_json_response(resp.message.content)

        # ── Danger sign detection ──
        danger_user = (
            f"Analyze this ASHA home visit conversation for danger signs.\n\n"
            f"Visit type: {visit_type}\n\n"
            f"{transcript}\n\n"
            f"Output JSON schema:\n{json.dumps(danger_schema, ensure_ascii=False)}"
        )

        t0 = time.time()
        resp2 = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": DANGER_SYSTEM_PROMPT},
                {"role": "user", "content": danger_user},
            ],
            options={"temperature": 0.0, "num_ctx": 4096},
        )
        danger_time = time.time() - t0
        danger_parsed = parse_json_response(resp2.message.content)

        elapsed = form_time + danger_time
        total_time += elapsed

        test_issues = []

        # ── Check form values ──
        if form_parsed is None:
            test_issues.append("FORM_PARSE_FAIL")
        else:
            for path, expected_val in expected_form.items():
                got = get_nested(form_parsed, path)
                if got is None:
                    test_issues.append(f"MISSING {path} (expected {expected_val})")
                else:
                    try:
                        if isinstance(expected_val, bool):
                            if got != expected_val:
                                test_issues.append(f"WRONG {path}: {got} != {expected_val}")
                        elif isinstance(expected_val, (int, float)):
                            if abs(float(got) - float(expected_val)) > 0.5:
                                test_issues.append(f"WRONG {path}: {got} != {expected_val}")
                        elif isinstance(expected_val, str):
                            got_lower = str(got).lower().strip()
                            exp_lower = expected_val.lower().strip()
                            # Allow partial match for names and places
                            if exp_lower not in got_lower and got_lower not in exp_lower:
                                test_issues.append(f"WRONG {path}: {got} != {expected_val}")
                    except (ValueError, TypeError):
                        if str(got) != str(expected_val):
                            test_issues.append(f"WRONG {path}: {got} != {expected_val}")

            # ── Check hallucination traps ──
            for path in must_be_null:
                val = get_nested(form_parsed, path)
                if val is not None and str(val).lower() not in ("null", "none", ""):
                    test_issues.append(f"HALLUC {path}={val}")

        # ── Check danger signs ──
        if danger_parsed is None:
            test_issues.append("DANGER_PARSE_FAIL")
        else:
            signs = danger_parsed.get("danger_signs", [])
            n_signs = len(signs) if isinstance(signs, list) else 0
            if n_signs < danger_min:
                test_issues.append(f"FALSE_NEG: {n_signs} signs < {danger_min} expected")
            if n_signs > danger_max:
                test_issues.append(f"FALSE_POS: {n_signs} signs > {danger_max} expected")

            # Check referral
            ref = danger_parsed.get("referral_decision", {})
            ref_decision = ref.get("decision", "")
            # Group equivalent referral decisions
            SAFE_REFERRALS = {"routine_followup", "continue_monitoring"}
            URGENT_REFERRALS = {"refer_immediately", "refer_within_24h"}
            if expected_referral:
                exp_group = "safe" if expected_referral in SAFE_REFERRALS else "urgent"
                got_group = "safe" if ref_decision in SAFE_REFERRALS else "urgent"
                if exp_group != got_group:
                    test_issues.append(f"REFERRAL: {ref_decision} != {expected_referral}")

        # ── Verdict ──
        if test_issues:
            status = "FAIL"
            total_fail += 1
        else:
            status = "PASS"
            total_pass += 1

        issues_str = "; ".join(test_issues) if test_issues else "all checks OK"
        print(f"  {status} [{name}] ({elapsed:.1f}s) {issues_str}")

    print(f"\n  Score: {total_pass}/{total_pass + total_fail}, avg {total_time / (total_pass + total_fail):.1f}s/test")
    return total_pass, total_fail


def main():
    models = [
        "gemma4:e4b-it-q4_K_M",
        # "sakhi:latest",  # fine-tuned model — disabled, worse than base
    ]

    results = {}
    for model in models:
        print(f"\n{'=' * 70}")
        print(f" {model}")
        print(f"{'=' * 70}")
        p, f = run_all_tests(model)
        results[model] = (p, f)

    print(f"\n{'=' * 70}")
    print("FINAL SCORES")
    print(f"{'=' * 70}")
    for model, (p, f) in results.items():
        pct = p / (p + f) * 100
        print(f"  {p}/{p+f} ({pct:.0f}%)  {model}")


if __name__ == "__main__":
    main()
