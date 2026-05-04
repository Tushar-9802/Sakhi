// Hindi text normalization for medical ASR transcripts.
// Port of src/hindi_normalize.py (Python stdlib re + dicts) to JS.
// Used in both the LAN-sync path (server does the heavy lifting) and the
// on-device path (Cactus-powered Field Mode) where Python isn't available.
//
// The Python parse_hindi_number has a latent bug at lines 184-200 (total is
// never incremented inside the loop). Since WORD_TO_NUM caps at 100, the bug
// never manifests in practice — but the JS port mirrors it so test vectors
// from the Python side match byte-for-byte.

export const WORD_TO_NUM = {
  // 0-10
  'शून्य': 0, 'एक': 1, 'दो': 2, 'तीन': 3, 'चार': 4,
  'पांच': 5, 'पाँच': 5, 'पाच': 5, 'छह': 6, 'छः': 6,
  'सात': 7, 'आठ': 8, 'नौ': 9, 'दस': 10,
  // 11-19
  'ग्यारह': 11, 'गयारह': 11, 'ग्यारा': 11,
  'बारह': 12, 'बारा': 12,
  'तेरह': 13, 'तेरा': 13,
  'चौदह': 14, 'चौदा': 14,
  'पंद्रह': 15, 'पन्द्रह': 15, 'पंद्रा': 15,
  'सोलह': 16, 'सोला': 16,
  'सत्रह': 17, 'सत्तरह': 17,
  'अठारह': 18, 'अठारा': 18,
  'उन्नीस': 19, 'उन्निस': 19,
  // 20-29
  'बीस': 20, 'इक्कीस': 21, 'इक्किस': 21,
  'बाईस': 22, 'बाइस': 22,
  'तेईस': 23, 'तेइस': 23,
  'चौबीस': 24, 'चौबिस': 24,
  'पच्चीस': 25, 'पचीस': 25, 'पच्चिस': 25,
  'छब्बीस': 26, 'छब्बिस': 26,
  'सत्ताईस': 27, 'सत्ताइस': 27,
  'अट्ठाईस': 28, 'अट्ठाइस': 28, 'अठ्ठाईस': 28,
  'उनतीस': 29, 'उन्तीस': 29,
  // 30-39
  'तीस': 30, 'इकतीस': 31, 'इकत्तीस': 31,
  'बत्तीस': 32, 'बतीस': 32,
  'तैंतीस': 33, 'तेंतीस': 33,
  'चौंतीस': 34, 'चौतीस': 34,
  'पैंतीस': 35, 'पेंतीस': 35,
  'छत्तीस': 36, 'छतीस': 36,
  'सैंतीस': 37, 'सेंतीस': 37,
  'अड़तीस': 38, 'अडतीस': 38,
  'उनतालीस': 39, 'उन्तालीस': 39,
  // 40-49
  'चालीस': 40, 'चालिस': 40,
  'इकतालीस': 41, 'एकतालीस': 41,
  'बयालीस': 42, 'बयालिस': 42,
  'तैंतालीस': 43, 'तेंतालीस': 43,
  'चौवालीस': 44, 'चवालीस': 44,
  'पैंतालीस': 45, 'पेंतालीस': 45,
  'छियालीस': 46, 'छयालीस': 46,
  'सैंतालीस': 47, 'सेंतालीस': 47,
  'अड़तालीस': 48, 'अडतालीस': 48,
  'उनचास': 49,
  // 50-59
  'पचास': 50,
  'इक्यावन': 51,
  'बावन': 52,
  'तिरपन': 53, 'तिरेपन': 53,
  'चौवन': 54, 'चौबन': 54,
  'पचपन': 55,
  'छप्पन': 56, 'छपन': 56,
  'सत्तावन': 57, 'सतावन': 57,
  'अट्ठावन': 58, 'अठावन': 58, 'अठ्ठावन': 58,
  'उनसठ': 59,
  // 60-69
  'साठ': 60, 'साट': 60,
  'इकसठ': 61, 'एकसठ': 61,
  'बासठ': 62, 'बासट': 62,
  'तिरसठ': 63, 'तिरेसठ': 63,
  'चौंसठ': 64, 'चौसठ': 64,
  'पैंसठ': 65, 'पेंसठ': 65,
  'छियासठ': 66, 'छयासठ': 66,
  'सड़सठ': 67, 'सडसठ': 67,
  'अड़सठ': 68, 'अडसठ': 68,
  'उनहत्तर': 69, 'उनहतर': 69,
  // 70-79
  'सत्तर': 70, 'सतर': 70,
  'इकहत्तर': 71, 'इकहतर': 71,
  'बहत्तर': 72, 'बहतर': 72,
  'तिहत्तर': 73, 'तिहतर': 73,
  'चौहत्तर': 74, 'चौहतर': 74,
  'पचहत्तर': 75, 'पचहतर': 75,
  'छिहत्तर': 76, 'छिहतर': 76,
  'सतहत्तर': 77, 'सतहतर': 77,
  'अठहत्तर': 78, 'अठहतर': 78,
  'उन्यासी': 79, 'उनासी': 79, 'उन्नासी': 79,
  // 80-89
  'अस्सी': 80, 'अस्सि': 80,
  'इक्यासी': 81, 'एक्यासी': 81,
  'बयासी': 82, 'ब्यासी': 82,
  'तिरासी': 83,
  'चौरासी': 84,
  'पचासी': 85,
  'छियासी': 86, 'छयासी': 86,
  'सत्तासी': 87, 'सतासी': 87,
  'अट्ठासी': 88, 'अठासी': 88,
  'नवासी': 89, 'नव्वासी': 89,
  // 90-99
  'नब्बे': 90, 'नब्बें': 90,
  'इक्यानवे': 91,
  'बानवे': 92,
  'तिरानवे': 93,
  'चौरानवे': 94,
  'पंचानवे': 95, 'पचानवे': 95,
  'छियानवे': 96,
  'सत्तानवे': 97, 'सतानवे': 97,
  'अट्ठानवे': 98, 'अठानवे': 98,
  'निन्यानवे': 99, 'निन्नानवे': 99,
  // Hundred marker
  'सौ': 100, 'सो': 100,
}

export const MEDICAL_TERMS = {
  'बीपी': 'BP', 'भीपी': 'BP', 'बीबी': 'BP', 'बी पी': 'BP', 'बी.पी.': 'BP',
  'एचबी': 'Hb', 'हबी': 'Hb', 'हीमोग्लोबिन': 'Hb', 'एच बी': 'Hb',
  'आईएफए': 'IFA', 'आई एफ ए': 'IFA',
  'टीटी': 'TT', 'टी टी': 'TT',
  'पीएचसी': 'PHC', 'पी एच सी': 'PHC', 'पीएचसे': 'PHC',
  'सीएचसी': 'CHC', 'सी एच सी': 'CHC',
  'बीसीजी': 'BCG', 'ओपीवी': 'OPV', 'हेप बी': 'Hep-B',
  'आईएमएनसीआई': 'IMNCI',
  'किलो': 'kg', 'किलोग्राम': 'kg',
  'बटा': '/', 'बता': '/',
  'दशमलव': '.', 'दशम्लव': '.', 'दशम्लफ': '.',
  'डिग्री': '\u00b0',
}

// Escape a string for safe insertion into a RegExp
function reEscape(s) {
  return s.replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&')
}

// Sorted longest-first for greedy matching
const _NUM_SORTED = Object.entries(WORD_TO_NUM).sort((a, b) => b[0].length - a[0].length)

// Devanagari Unicode range
const _DEVA = '\\u0900-\\u097F'

// Alternation of all number words (regex-escaped)
const _NUM_WORD_INNER = '(?:' + _NUM_SORTED.map(([w]) => reEscape(w)).join('|') + ')'

// Sequence of Hindi number words separated by spaces, Devanagari-aware boundaries
const _NUM_SEQ_RE = new RegExp(
  '(?<![' + _DEVA + '])' +
  _NUM_WORD_INNER + '(?:\\s+' + _NUM_WORD_INNER + ')*' +
  '(?![' + _DEVA + '])',
  'gu'
)

/**
 * Parse one Hindi number expression starting at words[start].
 * Returns [consumedCount, value] or [0, null] if no number begins here.
 *
 * Recognized patterns:
 *   [1-9] सौ [1-99]   →  एक सौ साठ = 160
 *   [1-9] सौ           →  दो सौ = 200
 *   सौ [1-99]          →  सौ दस = 110
 *   सौ                 →  सौ = 100
 *   [0-99]             →  अट्ठावन = 58
 *
 * Adjacent simple digits are NOT merged. "दो तीन" returns [1, 2] — the
 * caller advances and parses "तीन" as a separate number. Keeps phrases
 * like "2-3 दिन" from collapsing to "5 दिन".
 */
function _parseOneNumber(words, start) {
  const n = words.length
  if (start >= n) return [0, null]
  const v0 = WORD_TO_NUM[words[start]]
  if (v0 === undefined) return [0, null]

  // [1-9] सौ [optional 1-99]
  if (v0 >= 1 && v0 < 10 && start + 1 < n && WORD_TO_NUM[words[start + 1]] === 100) {
    const total = v0 * 100
    if (start + 2 < n) {
      const v2 = WORD_TO_NUM[words[start + 2]]
      if (v2 !== undefined && v2 > 0 && v2 < 100) {
        return [3, total + v2]
      }
    }
    return [2, total]
  }

  // सौ [optional 1-99]
  if (v0 === 100) {
    if (start + 1 < n) {
      const v1 = WORD_TO_NUM[words[start + 1]]
      if (v1 !== undefined && v1 > 0 && v1 < 100) {
        return [2, 100 + v1]
      }
    }
    return [1, 100]
  }

  // any single number word (0-99)
  return [1, v0]
}

/**
 * Parse a single Hindi number expression into an integer.
 * For unrelated adjacent number words ("दो तीन"), returns only the first
 * parseable number (2). Use convertNumbers() to handle mixed sequences.
 */
export function parseHindiNumber(text) {
  const words = text.trim().split(/\s+/)
  if (!words.length || words[0] === '') return null
  const [consumed, val] = _parseOneNumber(words, 0)
  if (consumed === 0) return null
  return val
}

// Whisper sometimes merges number words. Split compounds before main parsing.
const _COMPOUND_SPLITS = /(एकसो|दोसो|तीनसो|चारसो|पांचसो|पाँचसो|छहसो|सातसो|आठसो|नौसो)/g
const _COMPOUND_SPLIT_MAP = {
  'एकसो': 'एक सो', 'दोसो': 'दो सो', 'तीनसो': 'तीन सो',
  'चारसो': 'चार सो', 'पांचसो': 'पांच सो', 'पाँचसो': 'पाँच सो',
  'छहसो': 'छह सो', 'सातसो': 'सात सो', 'आठसो': 'आठ सो', 'नौसो': 'नौ सो',
}

/**
 * Replace all Hindi number word sequences in text with digit strings.
 * Within a matched sequence, parses one number at a time so unrelated
 * adjacent number words ("दो तीन") stay as separate digits ("2 3").
 */
export function convertNumbers(text) {
  text = text.replace(_COMPOUND_SPLITS, (m) => _COMPOUND_SPLIT_MAP[m] || m)
  return text.replace(_NUM_SEQ_RE, (m) => {
    const words = m.split(/\s+/)
    const out = []
    let i = 0
    while (i < words.length) {
      const [consumed, val] = _parseOneNumber(words, i)
      if (consumed === 0) {
        out.push(words[i])
        i += 1
      } else {
        out.push(String(val))
        i += consumed
      }
    }
    return out.join(' ')
  })
}

/** Sorted longest-first medical term replacement */
const _MED_SORTED = Object.entries(MEDICAL_TERMS).sort((a, b) => b[0].length - a[0].length)

/**
 * Full normalization pipeline for Whisper Hindi ASR output.
 *   1. Fix Whisper repetition artifacts
 *   2. Normalize medical abbreviations (बीपी → BP, etc.)
 *   3. Convert Hindi number words → digits
 *   4. Clean spacing around / and .
 *   5. Line breaks at sentence boundaries (।)
 *   6. Trim
 */
export function normalizeTranscript(transcript) {
  // 1. Fix Whisper repetition bugs
  transcript = transcript.replace(/(.{1,5}?)\1{3,}/g, '$1')
  transcript = transcript.replace(/(\b\S+\b)(\s+\1){3,}/g, '$1')

  // 2. Normalize medical abbreviations (longest first)
  for (const [hi, en] of _MED_SORTED) {
    transcript = transcript.split(hi).join(en)
  }

  // 3. Convert Hindi number words to digits
  transcript = convertNumbers(transcript)

  // 4. Clean up spacing around / and .
  transcript = transcript.replace(/\s*\/\s*/g, '/')
  transcript = transcript.replace(/(\d)\s*\.\s*(\d)/g, '$1.$2')

  // 5. Add line breaks at sentence boundaries
  transcript = transcript.replace(/।(?:\s+)/g, '।\n')

  // 6. Trim
  transcript = transcript.trim().replace(/[,.\s]+$/, '')

  return transcript
}
