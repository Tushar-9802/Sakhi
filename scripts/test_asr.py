"""
Rigorous ASR + normalization test suite for Sakhi.

Tests:
  1. Hindi number parser — edge cases, Whisper misspellings, compound numbers
  2. Medical term normalization — all abbreviations
  3. Full normalize_transcript — real Whisper output patterns
  4. Live ASR — transcribe test_audio/ files, verify medical values extracted
  5. Round-trip — ASR output → normalization → check expected values

Usage:
    python scripts/test_asr.py
    python scripts/test_asr.py --skip-gpu   # skip live ASR tests (no GPU needed)
"""

import sys
import os
import argparse
import time

os.environ["PYTHONIOENCODING"] = "utf-8"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.hindi_normalize import (
    parse_hindi_number, convert_numbers, normalize_transcript,
    WORD_TO_NUM, MEDICAL_TERMS
)

PASS = 0
FAIL = 0

def safe_print(s):
    """Print with fallback for Windows cp1252 encoding."""
    try:
        print(s)
    except UnicodeEncodeError:
        print(s.encode('ascii', errors='replace').decode('ascii'))


def check(name, got, expected, exact=True):
    global PASS, FAIL
    if exact:
        ok = got == expected
    else:
        # substring check
        ok = expected in str(got)
    if ok:
        PASS += 1
    else:
        FAIL += 1
        safe_print(f"  FAIL: {name}")
        safe_print(f"    got:      {got!r}")
        safe_print(f"    expected: {expected!r}")


def test_number_parser():
    """Test parse_hindi_number on individual and compound numbers."""
    print("\n=== 1. Number Parser ===")

    # Singles (0-99)
    singles = [
        ("शून्य", 0), ("एक", 1), ("दो", 2), ("तीन", 3), ("चार", 4),
        ("पांच", 5), ("पाँच", 5), ("छह", 6), ("सात", 7), ("आठ", 8),
        ("नौ", 9), ("दस", 10), ("ग्यारह", 11), ("बारह", 12), ("तेरह", 13),
        ("चौदह", 14), ("पंद्रह", 15), ("सोलह", 16), ("सत्रह", 17),
        ("अठारह", 18), ("उन्नीस", 19), ("बीस", 20), ("पच्चीस", 25),
        ("तीस", 30), ("पैंतीस", 35), ("चालीस", 40), ("पैंतालीस", 45),
        ("पचास", 50), ("पचपन", 55), ("अट्ठावन", 58), ("साठ", 60),
        ("पैंसठ", 65), ("सत्तर", 70), ("पचहत्तर", 75), ("अस्सी", 80),
        ("पचासी", 85), ("नब्बे", 90), ("पंचानवे", 95), ("निन्यानवे", 99),
    ]
    for word, expected in singles:
        check(f"parse({word})", parse_hindi_number(word), expected)

    # Whisper misspellings
    misspellings = [
        ("गयारह", 11), ("बारा", 12), ("पंद्रा", 15), ("इक्किस", 21),
        ("बाइस", 22), ("पचीस", 25), ("अठ्ठाईस", 28), ("बतीस", 32),
        ("चालिस", 40), ("अठावन", 58), ("सतर", 70), ("अठहतर", 78),
        ("उनासी", 79), ("अस्सि", 80), ("पचानवे", 95),
    ]
    for word, expected in misspellings:
        check(f"misspell({word})", parse_hindi_number(word), expected)

    # Compounds (100+)
    compounds = [
        ("सौ", 100), ("सो", 100),
        ("एक सौ", 100), ("एक सो", 100),
        ("एक सौ दस", 110), ("एक सौ बीस", 120),
        ("एक सौ पंद्रह", 115), ("एक सौ पच्चीस", 125),
        ("एक सौ तीस", 130), ("एक सौ पैंतीस", 135),
        ("एक सौ चालीस", 140), ("एक सौ पैंतालीस", 145),
        ("एक सौ पचास", 150), ("एक सौ पचपन", 155),
        ("एक सौ साठ", 160), ("एक सौ पैंसठ", 165),
        ("एक सौ सत्तर", 170), ("एक सौ अस्सी", 180),
        ("एक सौ नब्बे", 190),
        ("दो सौ", 200), ("तीन सौ", 300),
    ]
    for text, expected in compounds:
        check(f"compound({text})", parse_hindi_number(text), expected)

    # Common BP values
    bp_values = [
        ("एक सौ दस", 110), ("एक सौ बीस", 120), ("एक सौ तीस", 130),
        ("एक सौ चालीस", 140), ("एक सौ पचास", 150), ("एक सौ साठ", 160),
    ]
    for text, expected in bp_values:
        check(f"bp({text})", parse_hindi_number(text), expected)


def test_compound_splits():
    """Test that merged words like 'एकसो' are handled."""
    print("\n=== 2. Compound Word Splits ===")

    cases = [
        ("एकसो दस", "110"),
        ("एकसो बीस", "120"),
        ("एकसो पचपन", "155"),
        ("दोसो", "200"),
    ]
    for inp, expected in cases:
        result = convert_numbers(inp)
        check(f"split({inp})", expected in result, True)


def test_medical_terms():
    """Test all medical abbreviation conversions."""
    print("\n=== 3. Medical Terms ===")

    cases = [
        ("बीपी", "BP"), ("भीपी", "BP"), ("बी पी", "BP"), ("बी.पी.", "BP"),
        ("एचबी", "Hb"), ("हीमोग्लोबिन", "Hb"), ("एच बी", "Hb"),
        ("आईएफए", "IFA"), ("टीटी", "TT"), ("टी टी", "TT"),
        ("पीएचसी", "PHC"), ("पी एच सी", "PHC"),
        ("सीएचसी", "CHC"), ("बीसीजी", "BCG"), ("ओपीवी", "OPV"),
        ("किलो", "kg"), ("किलोग्राम", "kg"),
        ("बटा", "/"), ("दशमलव", "."),
    ]
    for hindi, expected in cases:
        result = normalize_transcript(f"test {hindi} test")
        check(f"med({hindi}→{expected})", expected in result, True)


def test_full_normalization():
    """Test normalize_transcript on realistic Whisper output patterns."""
    print("\n=== 4. Full Normalization (realistic patterns) ===")

    cases = [
        # BP values
        ("आपका बीपी एक सौ दस बटा सत्तर है", "110/70"),
        ("बीपी एकसो पचपन बटा सौ", "155/100"),
        ("बी पी एक सौ बीस बटा अस्सी है", "120/80"),
        # Weight
        ("वजन अट्ठावन किलो है", "58 kg"),
        ("वजन पचास किलोग्राम", "50 kg"),
        # Hemoglobin with decimal
        ("एचबी ग्यारह दशमलव पांच है", "Hb 11.5"),
        ("हीमोग्लोबिन बारह दशमलव दो", "Hb 12.2"),
        # Gestational weeks
        ("लगभग चौबीस हफ्ते", "24 हफ्ते"),
        ("बत्तीस हफ्ते की", "32 हफ्ते"),
        # Temperature
        ("तापमान सौ दशमलव पांच डिग्री", "100.5"),
        # Mixed digits and words
        ("बीपी 110 बटा सत्तर", "110/70"),
        ("बीपी 110/सतर", "110/70"),
        # Whisper repetition fix
        ("हाँ हाँ हाँ हाँ हाँ ठीक है", "हाँ ठीक है"),
        # Sentence boundaries
        ("पहला।  दूसरा।  तीसरा", "पहला।\nदूसरा।\nतीसरा"),
        # Already-digit pass-through
        ("BP 120/80 है, weight 55 kg", "120/80"),
    ]

    for inp, expect_substr in cases:
        result = normalize_transcript(inp)
        check(f"norm({inp[:40]}...→{expect_substr})", expect_substr in result, True)


def test_edge_cases():
    """Edge cases that could break the parser."""
    print("\n=== 5. Edge Cases ===")

    # Empty input
    check("empty", normalize_transcript(""), "")
    # Only numbers
    check("only_num", "25" in convert_numbers("पच्चीस"), True)
    # Number at start of text
    check("num_start", normalize_transcript("पच्चीस हफ्ते").startswith("25"), True)
    # Number at end of text
    check("num_end", normalize_transcript("हफ्ते पच्चीस").endswith("25"), True)
    # Adjacent numbers separated by non-number word
    result = normalize_transcript("वजन पचास और उम्र तीस")
    check("adjacent_nums", "50" in result and "30" in result, True)
    # Don't convert number words inside other Hindi words
    # "एकतरफ" should NOT become "1तरफ"
    result = convert_numbers("एकतरफ")
    check("no_partial", "1" not in result, True)
    # Very large numbers (should still work)
    check("nine_hundred", parse_hindi_number("नौ सौ निन्यानवे"), 999)


def test_real_whisper_transcripts():
    """Test on actual saved Whisper transcripts from previous sessions."""
    print("\n=== 6. Real Whisper Transcripts ===")

    test_files = {
        "postprocess_test.txt": ["110", "70"],       # BP should be found after norm
        "postprocess_test2.txt": ["110", "70"],       # Has "110/सतर"
        "pp_test.txt": ["110/70", "58 kg", "11.5"],   # Already clean
    }

    for fname, expected_values in test_files.items():
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), fname)
        if not os.path.exists(path):
            print(f"  SKIP: {fname} not found")
            continue
        raw = open(path, encoding="utf-8").read().strip()
        result = normalize_transcript(raw)
        for val in expected_values:
            check(f"file({fname}→{val})", val in result, True)


def test_live_asr():
    """Test actual ASR transcription on test audio files."""
    print("\n=== 7. Live ASR (GPU required) ===")

    audio_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_audio")
    if not os.path.exists(audio_dir):
        print("  SKIP: test_audio/ not found")
        return

    audio_files = [f for f in os.listdir(audio_dir) if f.endswith((".mp3", ".wav"))]
    if not audio_files:
        print("  SKIP: no audio files found")
        return

    # Expected values per file
    expectations = {
        "anc_normal.mp3": {
            "values": ["110", "70", "58", "11.5", "24"],
            "terms": ["BP", "kg", "Hb"],
        },
        "anc_danger.mp3": {
            "values": ["155", "100"],
            "terms": ["BP", "PHC"],
        },
    }

    from faster_whisper import WhisperModel
    ct2_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "whisper-hindi-ct2")

    if os.path.exists(ct2_path):
        print(f"  Loading CT2 model from {ct2_path}...")
        model = WhisperModel(ct2_path, device="cuda", compute_type="float16")
    else:
        print("  Loading collabora/whisper-large-v2-hindi from HuggingFace...")
        model = WhisperModel("collabora/whisper-large-v2-hindi", device="cuda", compute_type="float16")

    for fname in audio_files:
        fpath = os.path.join(audio_dir, fname)
        expect = expectations.get(fname, {"values": [], "terms": []})

        print(f"\n  --- {fname} ---")
        t0 = time.time()
        segments, info = model.transcribe(fpath, language="hi", task="transcribe")
        raw_text = " ".join(seg.text.strip() for seg in segments)
        asr_time = time.time() - t0

        t0 = time.time()
        normalized = normalize_transcript(raw_text)
        norm_time = time.time() - t0

        safe_print(f"  ASR time:  {asr_time:.1f}s")
        safe_print(f"  Norm time: {norm_time*1000:.0f}ms")
        safe_print(f"  Raw:  {raw_text[:150]}...")
        safe_print(f"  Norm: {normalized[:150]}...")

        for val in expect["values"]:
            found_raw = val in raw_text
            found_norm = val in normalized
            status = "RAW+NORM" if found_raw else ("NORM" if found_norm else "MISS")
            check(f"asr({fname}→{val})", found_norm, True)
            safe_print(f"    {val}: {status}")

        for term in expect["terms"]:
            check(f"asr({fname}→{term})", term in normalized, True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-gpu", action="store_true", help="Skip live ASR tests")
    args = parser.parse_args()

    print("=" * 60)
    print("SAKHI ASR + NORMALIZATION TEST SUITE")
    print("=" * 60)

    test_number_parser()
    test_compound_splits()
    test_medical_terms()
    test_full_normalization()
    test_edge_cases()
    test_real_whisper_transcripts()

    if not args.skip_gpu:
        test_live_asr()
    else:
        print("\n=== 7. Live ASR — SKIPPED (--skip-gpu) ===")

    print("\n" + "=" * 60)
    print(f"RESULTS: {PASS} passed, {FAIL} failed")
    print("=" * 60)

    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
