"""
Hindi text normalization for medical ASR transcripts.

Converts Hindi number words to digits and normalizes medical abbreviations.
Works as a standalone module — no external dependencies beyond Python stdlib.

Usage:
    from src.hindi_normalize import normalize_transcript

    raw = "आपका BP एक सौ दस बटा सत्तर है, वजन अट्ठावन kg"
    clean = normalize_transcript(raw)
    # → "आपका BP 110/70 है, वजन 58 kg"
"""

import re

# ============================================================
# HINDI NUMBER WORD → VALUE MAPPING (0-99 + 100)
# Includes common Whisper misspellings for each number.
# ============================================================

WORD_TO_NUM = {
    # 0-10
    "शून्य": 0, "एक": 1, "दो": 2, "तीन": 3, "चार": 4,
    "पांच": 5, "पाँच": 5, "पाच": 5, "छह": 6, "छः": 6,
    "सात": 7, "आठ": 8, "नौ": 9, "दस": 10,
    # 11-19
    "ग्यारह": 11, "गयारह": 11, "ग्यारा": 11,
    "बारह": 12, "बारा": 12,
    "तेरह": 13, "तेरा": 13,
    "चौदह": 14, "चौदा": 14,
    "पंद्रह": 15, "पन्द्रह": 15, "पंद्रा": 15,
    "सोलह": 16, "सोला": 16,
    "सत्रह": 17, "सत्तरह": 17,
    "अठारह": 18, "अठारा": 18,
    "उन्नीस": 19, "उन्निस": 19,
    # 20-29
    "बीस": 20, "इक्कीस": 21, "इक्किस": 21,
    "बाईस": 22, "बाइस": 22,
    "तेईस": 23, "तेइस": 23,
    "चौबीस": 24, "चौबिस": 24,
    "पच्चीस": 25, "पचीस": 25, "पच्चिस": 25,
    "छब्बीस": 26, "छब्बिस": 26,
    "सत्ताईस": 27, "सत्ताइस": 27,
    "अट्ठाईस": 28, "अट्ठाइस": 28, "अठ्ठाईस": 28,
    "उनतीस": 29, "उन्तीस": 29,
    # 30-39
    "तीस": 30, "इकतीस": 31, "इकत्तीस": 31,
    "बत्तीस": 32, "बतीस": 32,
    "तैंतीस": 33, "तेंतीस": 33,
    "चौंतीस": 34, "चौतीस": 34,
    "पैंतीस": 35, "पेंतीस": 35,
    "छत्तीस": 36, "छतीस": 36,
    "सैंतीस": 37, "सेंतीस": 37,
    "अड़तीस": 38, "अडतीस": 38,
    "उनतालीस": 39, "उन्तालीस": 39,
    # 40-49
    "चालीस": 40, "चालिस": 40,
    "इकतालीस": 41, "एकतालीस": 41,
    "बयालीस": 42, "बयालिस": 42,
    "तैंतालीस": 43, "तेंतालीस": 43,
    "चौवालीस": 44, "चवालीस": 44,
    "पैंतालीस": 45, "पेंतालीस": 45,
    "छियालीस": 46, "छयालीस": 46,
    "सैंतालीस": 47, "सेंतालीस": 47,
    "अड़तालीस": 48, "अडतालीस": 48,
    "उनचास": 49,
    # 50-59
    "पचास": 50,
    "इक्यावन": 51,
    "बावन": 52,
    "तिरपन": 53, "तिरेपन": 53,
    "चौवन": 54, "चौबन": 54,
    "पचपन": 55,
    "छप्पन": 56, "छपन": 56,
    "सत्तावन": 57, "सतावन": 57,
    "अट्ठावन": 58, "अठावन": 58, "अठ्ठावन": 58,
    "उनसठ": 59,
    # 60-69
    "साठ": 60, "साट": 60,
    "इकसठ": 61, "एकसठ": 61,
    "बासठ": 62, "बासट": 62,
    "तिरसठ": 63, "तिरेसठ": 63,
    "चौंसठ": 64, "चौसठ": 64,
    "पैंसठ": 65, "पेंसठ": 65,
    "छियासठ": 66, "छयासठ": 66,
    "सड़सठ": 67, "सडसठ": 67,
    "अड़सठ": 68, "अडसठ": 68,
    "उनहत्तर": 69, "उनहतर": 69,
    # 70-79
    "सत्तर": 70, "सतर": 70,
    "इकहत्तर": 71, "इकहतर": 71,
    "बहत्तर": 72, "बहतर": 72,
    "तिहत्तर": 73, "तिहतर": 73,
    "चौहत्तर": 74, "चौहतर": 74,
    "पचहत्तर": 75, "पचहतर": 75,
    "छिहत्तर": 76, "छिहतर": 76,
    "सतहत्तर": 77, "सतहतर": 77,
    "अठहत्तर": 78, "अठहतर": 78,
    "उन्यासी": 79, "उनासी": 79, "उन्नासी": 79,
    # 80-89
    "अस्सी": 80, "अस्सि": 80,
    "इक्यासी": 81, "एक्यासी": 81,
    "बयासी": 82, "ब्यासी": 82,
    "तिरासी": 83,
    "चौरासी": 84,
    "पचासी": 85,
    "छियासी": 86, "छयासी": 86,
    "सत्तासी": 87, "सतासी": 87,
    "अट्ठासी": 88, "अठासी": 88,
    "नवासी": 89, "नव्वासी": 89,
    # 90-99
    "नब्बे": 90, "नब्बें": 90,
    "इक्यानवे": 91,
    "बानवे": 92,
    "तिरानवे": 93,
    "चौरानवे": 94,
    "पंचानवे": 95, "पचानवे": 95,
    "छियानवे": 96,
    "सत्तानवे": 97, "सतानवे": 97,
    "अट्ठानवे": 98, "अठानवे": 98,
    "निन्यानवे": 99, "निन्नानवे": 99,
    # Hundred marker
    "सौ": 100, "सो": 100,
}

# ============================================================
# MEDICAL TERM NORMALIZATION
# ============================================================

MEDICAL_TERMS = {
    "बीपी": "BP", "भीपी": "BP", "बीबी": "BP", "बी पी": "BP", "बी.पी.": "BP",
    "एचबी": "Hb", "हबी": "Hb", "हीमोग्लोबिन": "Hb", "एच बी": "Hb",
    "आईएफए": "IFA", "आई एफ ए": "IFA",
    "टीटी": "TT", "टी टी": "TT",
    "पीएचसी": "PHC", "पी एच सी": "PHC", "पीएचसे": "PHC",
    "सीएचसी": "CHC", "सी एच सी": "CHC",
    "बीसीजी": "BCG", "ओपीवी": "OPV", "हेप बी": "Hep-B",
    "आईएमएनसीआई": "IMNCI",
    "किलो": "kg", "किलोग्राम": "kg",
    "बटा": "/", "बता": "/",
    "दशमलव": ".", "दशम्लव": ".", "दशम्लफ": ".",
    "डिग्री": "\u00b0",
}

# ============================================================
# NUMBER PARSING ENGINE
# ============================================================

# Sorted longest-first for greedy regex matching
_NUM_SORTED = sorted(WORD_TO_NUM.items(), key=lambda x: -len(x[0]))

# Devanagari character class for word boundary detection
# Covers base consonants, vowels, matras, nukta, virama, etc.
_DEVA = r'\u0900-\u097F'

# Regex matching any single Hindi number word
_NUM_WORD_INNER = r'(?:' + '|'.join(re.escape(w) for w, _ in _NUM_SORTED) + r')'

# Regex matching a sequence of Hindi number words separated by spaces,
# with Devanagari-aware word boundaries (not preceded/followed by Devanagari chars)
_NUM_SEQ_RE = re.compile(
    r'(?<![' + _DEVA + r'])' +
    _NUM_WORD_INNER + r'(?:\s+' + _NUM_WORD_INNER + r')*' +
    r'(?![' + _DEVA + r'])'
)


def _parse_one_number(words, start):
    """Parse one Hindi number expression starting at words[start].

    Returns (consumed_word_count, value) or (0, None) if no number begins here.

    Recognized patterns:
      [1-9] सौ [1-99]   →  एक सौ साठ = 160
      [1-9] सौ           →  दो सौ = 200
      सौ [1-99]          →  सौ दस = 110
      सौ                 →  सौ = 100
      [0-99]             →  अट्ठावन = 58, दस = 10

    Adjacent simple digits are NOT merged: "दो तीन" yields (1, 2) — caller is
    expected to advance and parse "तीन" as a separate number. This keeps
    "2-3 days" from collapsing into "5".
    """
    n = len(words)
    if start >= n:
        return 0, None
    v0 = WORD_TO_NUM.get(words[start])
    if v0 is None:
        return 0, None

    # Pattern: [1-9] सौ [optional 1-99]
    if 1 <= v0 < 10 and start + 1 < n and WORD_TO_NUM.get(words[start + 1]) == 100:
        total = v0 * 100
        if start + 2 < n:
            v2 = WORD_TO_NUM.get(words[start + 2])
            if v2 is not None and 0 < v2 < 100:
                return 3, total + v2
        return 2, total

    # Pattern: सौ [optional 1-99]
    if v0 == 100:
        if start + 1 < n:
            v1 = WORD_TO_NUM.get(words[start + 1])
            if v1 is not None and 0 < v1 < 100:
                return 2, 100 + v1
        return 1, 100

    # Pattern: any single number word (0-99)
    return 1, v0


def parse_hindi_number(text):
    """Parse a single Hindi number expression into an integer.

    For sequences of unrelated numbers (e.g. "दो तीन" — two then three),
    returns only the first parseable number (2). Use convert_numbers() to
    handle mixed sequences in real transcripts.

    Examples:
      "एक सौ दस"       → 110
      "एक सौ पचपन"    → 155
      "दो सौ"           → 200
      "सौ"              → 100
      "सत्तर"            → 70
      "अट्ठावन"          → 58
      "नौ सौ निन्यानवे" → 999
    """
    words = text.strip().split()
    if not words:
        return None
    consumed, val = _parse_one_number(words, 0)
    if consumed == 0:
        return None
    return val


# Whisper sometimes merges number words (e.g., "एकसो" instead of "एक सो").
# Split these before main parsing.
_COMPOUND_SPLITS = re.compile(
    r'(एकसो|दोसो|तीनसो|चारसो|पांचसो|पाँचसो|छहसो|सातसो|आठसो|नौसो)'
)
_COMPOUND_SPLIT_MAP = {
    "एकसो": "एक सो", "दोसो": "दो सो", "तीनसो": "तीन सो",
    "चारसो": "चार सो", "पांचसो": "पांच सो", "पाँचसो": "पाँच सो",
    "छहसो": "छह सो", "सातसो": "सात सो", "आठसो": "आठ सो", "नौसो": "नौ सो",
}


def convert_numbers(text):
    """Replace all Hindi number word sequences in text with digit strings.

    Within a matched sequence, parses one number at a time using
    _parse_one_number, so unrelated adjacent number words ("दो तीन")
    stay as separate digits ("2 3") instead of summing.
    """
    # Pre-split compound words like "एकसो" → "एक सो"
    text = _COMPOUND_SPLITS.sub(lambda m: _COMPOUND_SPLIT_MAP.get(m.group(0), m.group(0)), text)

    def _replace(m):
        words = m.group(0).split()
        out = []
        i = 0
        while i < len(words):
            consumed, val = _parse_one_number(words, i)
            if consumed == 0:
                out.append(words[i])
                i += 1
            else:
                out.append(str(val))
                i += consumed
        return " ".join(out)
    return _NUM_SEQ_RE.sub(_replace, text)


# ============================================================
# FULL TRANSCRIPT NORMALIZATION
# ============================================================

def normalize_transcript(transcript):
    """Full normalization pipeline for Whisper Hindi ASR output.

    Steps:
      1. Fix Whisper repetition artifacts
      2. Normalize medical abbreviations (बीपी → BP, etc.)
      3. Convert Hindi number words to digits (algorithmic)
      4. Clean up spacing around / and .
      5. Add line breaks at sentence boundaries (।)
    """
    # 1. Fix Whisper repetition bugs
    transcript = re.sub(r'(.{1,5}?)\1{3,}', r'\1', transcript)
    transcript = re.sub(r'(\b\S+\b)(\s+\1){3,}', r'\1', transcript)

    # 2. Normalize medical abbreviations (longest first to avoid partial matches)
    for hindi, eng in sorted(MEDICAL_TERMS.items(), key=lambda x: -len(x[0])):
        transcript = transcript.replace(hindi, eng)

    # 3. Convert Hindi number words to digits
    transcript = convert_numbers(transcript)

    # 4. Clean up spacing around / and .
    transcript = re.sub(r'\s*/\s*', '/', transcript)
    transcript = re.sub(r'(\d)\s*\.\s*(\d)', r'\1.\2', transcript)

    # 5. Add line breaks at sentence boundaries
    transcript = re.sub(r'[।](?:\s+)', '।\n', transcript)

    # 6. Clean up
    transcript = transcript.strip().rstrip(',. ')

    return transcript
