"""Quick API test for Sakhi — hits the running Gradio app."""
import requests
import json
import sys

BASE = "http://localhost:7860"

# Example transcripts (same as in-app examples)
TESTS = {
    "ANC Normal": {
        "transcript": (
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
        "visit_type": "Auto-detect",
        "expect_danger": False,
        "expect_fields": {"bp_systolic": 110, "bp_diastolic": 70, "weight_kg": 58.0, "hemoglobin_gm_percent": 11.5},
    },
    "ANC Preeclampsia DANGER": {
        "transcript": (
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
        "visit_type": "Auto-detect",
        "expect_danger": True,
        "expect_fields": {"bp_systolic": 155, "bp_diastolic": 100},
    },
    "PNC Newborn DANGER": {
        "transcript": (
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
        "visit_type": "Auto-detect",
        "expect_danger": True,
        "expect_fields": {},
    },
    "Child Health Routine": {
        "transcript": (
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
        "visit_type": "Auto-detect",
        "expect_danger": False,
        "expect_fields": {"weight_kg": 8.5},
    },
}


def call_gradio_api(transcript, visit_type="Auto-detect"):
    """Call the Gradio app's process_transcript endpoint."""
    from gradio_client import Client
    client = Client(BASE)
    # Text to Form tab uses process_transcript
    result = client.predict(
        transcript=transcript,
        visit_type_override=visit_type,
        api_name="/process_transcript"
    )
    return result


def run_tests():
    print("=" * 60)
    print("SAKHI API TEST SUITE")
    print("=" * 60)

    try:
        from gradio_client import Client
    except ImportError:
        print("Installing gradio_client...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "gradio_client", "-q"])
        from gradio_client import Client

    client = Client(BASE, verbose=False)

    passed = 0
    failed = 0

    for name, test in TESTS.items():
        print(f"\n--- {name} ---")
        try:
            result = client.predict(
                transcript=test["transcript"],
                visit_type_override=test["visit_type"],
                api_name="/process_transcript"
            )

            # result is a tuple: (status_html, form_html, danger_html, time_str)
            status_html, form_html, danger_html, time_str = result

            # Check basics
            has_form = "result-card" in form_html and "error" not in form_html
            has_danger_section = "danger-card" in danger_html
            has_danger_signs = "Danger Signs Detected" in danger_html
            is_routine = "ROUTINE FOLLOW-UP" in danger_html
            is_referral = "REFERRAL" in danger_html

            print(f"  Form extracted: {'YES' if has_form else 'FAILED'}")
            print(f"  Danger signs: {'YES' if has_danger_signs else 'none'}")
            print(f"  Referral: {'REFERRAL' if is_referral else 'routine'}")
            print(f"  Time: {time_str}")

            # Validate expectations
            ok = True
            if test["expect_danger"] and not has_danger_signs:
                print(f"  FAIL: expected danger signs but got none")
                ok = False
            if not test["expect_danger"] and has_danger_signs:
                print(f"  FAIL: expected NO danger signs but got some")
                ok = False
            if not has_form:
                print(f"  FAIL: form extraction failed")
                ok = False

            # Check specific field values in form HTML
            for field, expected in test["expect_fields"].items():
                field_label = field.replace("_", " ").title()
                if str(expected) in form_html:
                    print(f"  {field_label}: {expected} OK")
                else:
                    print(f"  FAIL: {field_label} expected {expected}, not found in output")
                    ok = False

            if ok:
                print(f"  PASSED")
                passed += 1
            else:
                failed += 1

        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(TESTS)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run_tests()
