"""
End-to-end pipeline test suite for Sakhi.

Runs 15 synthetic Hindi audio files through the FULL pipeline:
  Audio → Whisper ASR → Hindi normalization → Form extraction → Danger sign detection

Validates against known ground truth values from manifest.json.
Tests: value accuracy, hallucination, danger sign detection, referral decisions.

Usage:
    python scripts/test_pipeline_e2e.py
"""
import json
import os
import sys
import time

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")
# Disable buffering for real-time output on Windows
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

# Ensure project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import (
    transcribe_audio,
    extract_form,
    extract_danger_signs,
    detect_visit_type,
    init_schemas,
)

AUDIO_DIR = "test_audio/synthetic"
MANIFEST = os.path.join(AUDIO_DIR, "manifest.json")

SAFE_REFERRALS = {"routine_followup", "continue_monitoring"}
URGENT_REFERRALS = {"refer_immediately", "refer_within_24h"}


def get_nested(d, path):
    """Get value from dict using dotted path like 'vitals.bp_systolic'."""
    parts = path.split(".")
    for p in parts:
        if not isinstance(d, dict):
            return None
        d = d.get(p)
    return d


def check_value(got, expected):
    """Check if extracted value matches expected, with tolerance for numbers."""
    if got is None:
        return False
    if isinstance(expected, bool):
        return got == expected
    if isinstance(expected, (int, float)):
        try:
            return abs(float(got) - float(expected)) <= 1.0
        except (ValueError, TypeError):
            return False
    if isinstance(expected, str):
        got_lower = str(got).lower().strip()
        exp_lower = expected.lower().strip()
        return exp_lower in got_lower or got_lower in exp_lower
    return str(got) == str(expected)


def run_test(test_case, test_num, total):
    """Run a single end-to-end test. Returns (pass, issues, timing)."""
    audio_file = os.path.join(AUDIO_DIR, test_case["file"])
    expected = test_case["expected"]
    name = test_case["file"].replace(".mp3", "")

    issues = []
    timing = {}

    # ── Step 1: ASR ──
    t0 = time.time()
    transcript = transcribe_audio(audio_file)
    timing["asr"] = round(time.time() - t0, 1)

    if not transcript or not transcript.strip():
        issues.append("ASR_EMPTY")
        print(f"  [{test_num}/{total}] FAIL [{name}] — ASR returned empty")
        return False, issues, timing

    # ── Step 2: Visit type detection ──
    detected_type = detect_visit_type(transcript)
    expected_type = expected["visit_type"]
    if detected_type != expected_type:
        issues.append(f"VISIT_TYPE: detected={detected_type} expected={expected_type}")

    # Use expected type for extraction (test extraction quality, not detection)
    visit_type = expected_type

    # ── Step 3: Form extraction ──
    t0 = time.time()
    form_result = extract_form(transcript, visit_type)
    timing["form"] = round(time.time() - t0, 1)
    form = form_result.get("parsed")

    if form is None:
        issues.append("FORM_PARSE_FAIL")
    else:
        # Check expected values
        for path, exp_val in expected.get("checks", {}).items():
            got = get_nested(form, path)
            if got is None:
                issues.append(f"MISSING {path} (expected {exp_val})")
            elif not check_value(got, exp_val):
                issues.append(f"WRONG {path}: got={got} expected={exp_val}")

        # Check hallucination traps
        for path in expected.get("must_be_null", []):
            val = get_nested(form, path)
            if val is not None and str(val).lower() not in ("null", "none", "", "—"):
                issues.append(f"HALLUC {path}={val}")

    # ── Step 4: Danger sign detection ──
    t0 = time.time()
    danger_result = extract_danger_signs(transcript, visit_type)
    timing["danger"] = round(time.time() - t0, 1)
    danger = danger_result.get("parsed")

    if danger is None:
        issues.append("DANGER_PARSE_FAIL")
    else:
        signs = danger.get("danger_signs", [])
        n_signs = len(signs) if isinstance(signs, list) else 0
        d_min, d_max = expected.get("danger_count", [0, 0])
        if n_signs < d_min:
            issues.append(f"FALSE_NEG: {n_signs} danger signs < {d_min} expected")
        if n_signs > d_max:
            issues.append(f"FALSE_POS: {n_signs} danger signs > {d_max} expected")

        # Check referral decision
        ref = danger.get("referral_decision", {})
        ref_decision = ref.get("decision", "")
        exp_ref = expected.get("referral", "")
        if exp_ref:
            exp_group = "safe" if exp_ref in SAFE_REFERRALS else "urgent"
            got_group = "safe" if ref_decision in SAFE_REFERRALS else "urgent"
            if exp_group != got_group:
                issues.append(f"REFERRAL: got={ref_decision} expected={exp_ref}")

    timing["total"] = round(sum(timing.values()), 1)
    passed = len(issues) == 0
    status = "PASS" if passed else "FAIL"
    detail = "all checks OK" if passed else "; ".join(issues)
    print(f"  [{test_num}/{total}] {status} [{name}] ({timing['total']:.1f}s) {detail}")

    return passed, issues, timing


def main():
    if not os.path.exists(MANIFEST):
        print(f"ERROR: {MANIFEST} not found. Run scripts/generate_test_audio.py first.")
        sys.exit(1)

    with open(MANIFEST, encoding="utf-8") as f:
        test_cases = json.load(f)

    print("Initializing schemas...")
    init_schemas()

    print(f"\n{'=' * 74}")
    print(f" Sakhi E2E Pipeline Test — {len(test_cases)} audio samples")
    print(f"{'=' * 74}")

    total_pass = 0
    total_fail = 0
    all_timings = []
    failures = []

    for i, tc in enumerate(test_cases, 1):
        passed, issues, timing = run_test(tc, i, len(test_cases))
        if passed:
            total_pass += 1
        else:
            total_fail += 1
            failures.append((tc["file"], issues))
        all_timings.append(timing)

    # ── Summary ──
    total = total_pass + total_fail
    pct = total_pass / total * 100 if total else 0
    avg_total = sum(t.get("total", 0) for t in all_timings) / len(all_timings)
    avg_asr = sum(t.get("asr", 0) for t in all_timings) / len(all_timings)
    avg_form = sum(t.get("form", 0) for t in all_timings) / len(all_timings)
    avg_danger = sum(t.get("danger", 0) for t in all_timings) / len(all_timings)

    print(f"\n{'=' * 74}")
    print(f" RESULTS: {total_pass}/{total} ({pct:.0f}%)")
    print(f" Avg timing: ASR {avg_asr:.1f}s | Form {avg_form:.1f}s | Danger {avg_danger:.1f}s | Total {avg_total:.1f}s")
    print(f"{'=' * 74}")

    if failures:
        print(f"\n FAILURES:")
        for fname, issues in failures:
            print(f"  {fname}: {'; '.join(issues)}")

    # Exit code for CI
    sys.exit(0 if total_fail == 0 else 1)


if __name__ == "__main__":
    main()
