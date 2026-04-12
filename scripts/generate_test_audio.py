"""
Generate synthetic Hindi ASHA conversation audio for end-to-end pipeline testing.

Each clip has known ground truth values, enabling automated validation.
Uses Google TTS (Hindi) — not perfectly natural but has correct words/numbers
that Whisper must transcribe, normalize, and the LLM must extract.
"""
import json
import os
from gtts import gTTS

OUT_DIR = "test_audio/synthetic"
os.makedirs(OUT_DIR, exist_ok=True)

# Each test case: (filename, transcript_hindi, expected_values)
# expected_values used by test_pipeline.py for validation
TEST_CASES = [
    # ── ANC CASES ──
    (
        "anc_normal_vitals",
        "नमस्ते दीदी, मैं आशा कार्यकर्ता हूँ। आज आपका चेकअप करती हूँ। "
        "आपका नाम रीना है ना? आपकी उम्र 22 साल है। "
        "चलिए BP चेक करते हैं। आपका BP 110 बटा 70 है, बिल्कुल ठीक है। "
        "वजन 55 किलो है। Hb रिपोर्ट में 11 पॉइंट 5 आया। "
        "आप 24 हफ्ते की गर्भवती हैं। IFA की गोलियाँ रोज़ लो। "
        "TT का पहला टीका लग गया है। "
        "डिलीवरी के लिए PHC में जाएँगी। सब ठीक चल रहा है।",
        {
            "visit_type": "anc_visit",
            "checks": {
                "patient.name": "रीना",
                "patient.age": 22,
                "vitals.bp_systolic": 110,
                "vitals.bp_diastolic": 70,
                "vitals.weight_kg": 55,
                "vitals.hemoglobin_gm_percent": 11.5,
                "pregnancy.gestational_weeks": 24,
            },
            "danger_count": [0, 0],
            "referral": "routine_followup",
            "must_be_null": ["lab_results.blood_group", "lab_results.hiv_status"],
        },
    ),
    (
        "anc_preeclampsia_danger",
        "दीदी आज मुझे बहुत सिरदर्द हो रहा है। आँखों के सामने धुंधला दिखता है। "
        "चेहरे पर सूजन आ गई है। पैरों में भी सूजन है। "
        "मैं आपका BP चेक करती हूँ। BP 160 बटा 105 है। ये बहुत ज़्यादा है। "
        "आप 32 हफ्ते की हैं। ये danger signs हैं। तुरंत hospital जाना होगा।",
        {
            "visit_type": "anc_visit",
            "checks": {
                "vitals.bp_systolic": 160,
                "vitals.bp_diastolic": 105,
                "pregnancy.gestational_weeks": 32,
            },
            "danger_count": [2, 5],
            "referral": "refer_immediately",
            "must_be_null": ["patient.name"],
        },
    ),
    (
        "anc_severe_anemia",
        "Hb की रिपोर्ट आई है। Hb 6 पॉइंट 2 है, बहुत कम है। "
        "मुझे बहुत चक्कर आते हैं। साँस लेने में तकलीफ़ होती है। "
        "BP 95 बटा 60 है। वजन 42 किलो है। आप 18 हफ्ते की हैं। "
        "आपको तुरंत PHC जाकर iron injection लेना होगा।",
        {
            "visit_type": "anc_visit",
            "checks": {
                "vitals.bp_systolic": 95,
                "vitals.bp_diastolic": 60,
                "vitals.weight_kg": 42,
                "vitals.hemoglobin_gm_percent": 6.2,
                "pregnancy.gestational_weeks": 18,
            },
            "danger_count": [1, 3],
            "referral": "refer_immediately",
            "must_be_null": ["patient.name", "lab_results.blood_group"],
        },
    ),
    (
        "anc_minimal_info",
        "BP ठीक है, 118 बटा 76 है। बाकी सब ठीक है दीदी।",
        {
            "visit_type": "anc_visit",
            "checks": {
                "vitals.bp_systolic": 118,
                "vitals.bp_diastolic": 76,
            },
            "danger_count": [0, 0],
            "referral": "routine_followup",
            "must_be_null": ["patient.name", "patient.age", "vitals.weight_kg",
                             "vitals.hemoglobin_gm_percent", "lab_results.blood_group"],
        },
    ),
    (
        "anc_hinglish_codeswitching",
        "Hello didi, aaj check-up hai. BP check karti hoon. "
        "BP 135 by 88 hai, thoda high side pe hai. "
        "Weight 64 kg hai. Hb report mein 9 point 5 aaya hai. "
        "Aap 30 weeks ki hain. Baby movement acchi hai. "
        "Delivery ke liye district hospital jaayengi.",
        {
            "visit_type": "anc_visit",
            "checks": {
                "vitals.bp_systolic": 135,
                "vitals.bp_diastolic": 88,
                "vitals.weight_kg": 64,
                "vitals.hemoglobin_gm_percent": 9.5,
                "pregnancy.gestational_weeks": 30,
            },
            "danger_count": [0, 1],
            "referral": "routine_followup",
            "must_be_null": ["patient.name", "lab_results.blood_group"],
        },
    ),
    # ── PNC CASES ──
    (
        "pnc_normal_day3",
        "नमस्ते दीदी, डिलीवरी को 3 दिन हो गए। कैसे हैं आप? "
        "मैं ठीक हूँ। बच्चा अच्छे से दूध पी रहा है। "
        "बच्चे का वजन 2 पॉइंट 9 kg है। नाभि सूखी है। "
        "बच्चे का तापमान सामान्य है। "
        "आपका BP 120 बटा 78 है। खून बहना बंद हो गया है।",
        {
            "visit_type": "pnc_visit",
            "checks": {
                "infant_assessment.weight_kg": 2.9,
            },
            "danger_count": [0, 0],
            "referral": "routine_followup",
            "must_be_null": [],
        },
    ),
    (
        "pnc_newborn_danger",
        "दीदी, बच्चा बहुत सुस्त है। दूध नहीं पी रहा है। "
        "12 घंटे से दूध नहीं पिया है। बहुत कमज़ोर आवाज़ में रोता है। "
        "बच्चे का तापमान 101 डिग्री है। बुखार है। "
        "ये danger signs हैं। तुरंत PHC ले जाना होगा।",
        {
            "visit_type": "pnc_visit",
            "checks": {},
            "danger_count": [1, 4],
            "referral": "refer_immediately",
            "must_be_null": [],
        },
    ),
    (
        "pnc_postpartum_hemorrhage",
        "डिलीवरी को 2 दिन हुए हैं। बहुत ज़्यादा खून आ रहा है। "
        "pad 1 घंटे में भीग जाता है। चक्कर आ रहे हैं। "
        "बहुत कमज़ोरी लग रही है। ये गंभीर है। तुरंत hospital जाना होगा।",
        {
            "visit_type": "pnc_visit",
            "checks": {},
            "danger_count": [1, 3],
            "referral": "refer_immediately",
            "must_be_null": [],
        },
    ),
    # ── DELIVERY CASES ──
    (
        "delivery_normal_institutional",
        "डिलीवरी कब हुई? कल रात 2 बजे। लड़का हुआ है। "
        "कहाँ हुई डिलीवरी? PHC में, normal delivery थी। "
        "बच्चे का वजन 3 पॉइंट 1 kg है। "
        "स्तनपान तुरंत शुरू किया, 1 घंटे के अंदर। "
        "बच्चे ने जन्म के समय रोया था।",
        {
            "visit_type": "delivery",
            "checks": {
                "infant.birth_weight_kg": 3.1,
            },
            "danger_count": [0, 0],
            "referral": "routine_followup",
            "must_be_null": [],
        },
    ),
    (
        "delivery_home_lbw",
        "बच्चा घर पर ही हो गया। दाई ने करवाया। लड़की हुई है। "
        "बच्ची का वजन 1 पॉइंट 7 kg है। बहुत कम है। ये low birth weight है। "
        "बच्ची ने जन्म के समय रोया था। "
        "बच्ची को गर्म रखना ज़रूरी है। PHC में चेकअप करवाना होगा।",
        {
            "visit_type": "delivery",
            "checks": {
                "infant.birth_weight_kg": 1.7,
            },
            "danger_count": [1, 2],
            "referral": "refer_immediately",
            "must_be_null": [],
        },
    ),
    # ── CHILD HEALTH CASES ──
    (
        "child_health_routine",
        "बच्चा कैसा है? बिल्कुल ठीक है दीदी। खूब खाता है, खेलता है। "
        "बच्चे का वजन 8 पॉइंट 2 kg है। उम्र 9 महीने है। "
        "Vitamin A की पहली dose 6 महीने में दी थी। "
        "सारे टीके लगे हैं। बच्चा बैठता है, घुटनों पर चलता है।",
        {
            "visit_type": "child_health",
            "checks": {
                "child.age_months": 9,
                "growth_assessment.weight_kg": 8.2,
            },
            "danger_count": [0, 0],
            "referral": "routine_followup",
            "must_be_null": [],
        },
    ),
    (
        "child_diarrhea_dehydration",
        "बच्चे को 4 दिन से दस्त लग रहे हैं। बहुत पतले पानी जैसे। "
        "खाना पीना बंद कर दिया है। बहुत सुस्त हो गया है। "
        "बच्चे का वजन 5 पॉइंट 8 kg है। उम्र 10 महीने है। "
        "आँखें धँसी हुई हैं। ये dehydration के signs हैं। तुरंत PHC जाना होगा।",
        {
            "visit_type": "child_health",
            "checks": {
                "child.age_months": 10,
                "growth_assessment.weight_kg": 5.8,
            },
            "danger_count": [1, 3],
            "referral": "refer_immediately",
            "must_be_null": [],
        },
    ),
    # ── EDGE CASES ──
    (
        "edge_zero_findings",
        "नमस्ते दीदी, सब कैसा है? बिल्कुल ठीक है, कोई तकलीफ़ नहीं। "
        "बहुत अच्छा। अगली बार आऊँगी। कोई तकलीफ़ हो तो फ़ोन कर दीजिए।",
        {
            "visit_type": "anc_visit",
            "checks": {},
            "danger_count": [0, 0],
            "referral": "routine_followup",
            "must_be_null": ["patient.name", "patient.age", "vitals.bp_systolic",
                             "vitals.weight_kg", "vitals.hemoglobin_gm_percent",
                             "lab_results.blood_group", "lab_results.hiv_status"],
        },
    ),
    (
        "edge_hindi_numbers_only",
        "आपका BP एक सौ बीस बटा अस्सी है। वजन बावन किलो है। "
        "Hb दस पॉइंट आठ है। आप छब्बीस हफ्ते की हैं।",
        {
            "visit_type": "anc_visit",
            "checks": {
                "vitals.bp_systolic": 120,
                "vitals.bp_diastolic": 80,
                "vitals.weight_kg": 52,
                "vitals.hemoglobin_gm_percent": 10.8,
                "pregnancy.gestational_weeks": 26,
            },
            "danger_count": [0, 0],
            "referral": "routine_followup",
            "must_be_null": [],
        },
    ),
    (
        "edge_noisy_nonmedical",
        "हाँ दीदी बच्चों को स्कूल भेजना है। राशन की दुकान बंद थी आज। "
        "बिजली भी नहीं आ रही। पानी का भी problem है गाँव में।",
        {
            "visit_type": "anc_visit",
            "checks": {},
            "danger_count": [0, 0],
            "referral": "routine_followup",
            "must_be_null": ["patient.name", "patient.age", "vitals.bp_systolic",
                             "vitals.weight_kg"],
        },
    ),
]


def main():
    manifest = []
    for filename, transcript, expected in TEST_CASES:
        out_path = os.path.join(OUT_DIR, f"{filename}.mp3")
        print(f"Generating {out_path}...")
        tts = gTTS(text=transcript, lang="hi", slow=False)
        tts.save(out_path)
        manifest.append({
            "file": f"{filename}.mp3",
            "transcript_source": transcript,
            "expected": expected,
        })

    manifest_path = os.path.join(OUT_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\nGenerated {len(manifest)} test audio files + manifest.json")


if __name__ == "__main__":
    main()
