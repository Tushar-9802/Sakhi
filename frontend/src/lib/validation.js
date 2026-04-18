// 6-layer anti-hallucination validation pipeline.
// Port of app.py validate_form_output (lines 772-847) and the danger-sign
// validation inside extract_danger_signs (lines 877-947).
//
// Layers:
//   Form validation
//     1. Name hallucination (दीदी/बहन/patient)
//     2. Default-age hallucination (age=30 when not in transcript)
//     3. Lab results hallucination (blood_group, hiv_status invented)
//     4. Numeric range checks (BP, weight, Hb, gestation, temp)
//   Danger-sign validation
//     5. Evidence length (<10 chars dropped)
//     6. Generic ASHA phrase blocklist
//     7. Normal value filter (BP 110/70, "ठीक है")
//     8. Transcript grounding (verbatim or 30-char chunk)
//     9. Duplicate-evidence dedup (all signs cite same evidence → drop all)

const FAKE_NAMES = new Set(['दीदी', 'बहन', 'बहनजी', 'patient', 'दी दी', 'didi', 'bahen'])
const BLOOD_GROUPS = new Set(['a+', 'a-', 'b+', 'b-', 'ab+', 'ab-', 'o+', 'o-'])
const HIV_VALUES = new Set(['negative', 'positive', 'नेगेटिव', 'पॉजिटिव'])
const BG_KEYWORDS = ['blood group', 'ब्लड ग्रुप', 'खून का ग्रुप', 'रक्त समूह']
const HIV_KEYWORDS = ['hiv', 'एचआईवी', 'एड्स']

const RANGES = {
  bp_systolic: [60, 250],
  bp_diastolic: [30, 150],
  weight_kg: [1, 200],
  hemoglobin_gm_percent: [3, 20],
  gestational_weeks: [1, 45],
  temperature_f: [90, 110],
}

const GENERIC_PHRASES = [
  'कोई तकलीफ़ हो तो फ़ोन कर दीजिए',
  'कोई तकलीफ हो तो फोन कर दीजिए',
  'कोई समस्या हो तो तुरंत बताइए',
  'कोई समस्या हो तो फोन करें',
  'कोई दिक्कत हो तो',
  'अगली बार आऊँगी',
  'अगली विज़िट',
  'ठीक है दीदी, धन्यवाद',
  'ठीक है दीदी',
]

const NORMAL_INDICATORS = [
  '110/70', '120/80', '110/80', '118/76', '108/72',
  'बिल्कुल ठीक', 'सामान्य', 'नॉर्मल', 'अच्छा है', 'ठीक है',
  'बिल्कुल सामान्य',
]

function isPlainObject(v) {
  return v !== null && typeof v === 'object' && !Array.isArray(v)
}

/**
 * Strip hallucinated fields + apply range checks on form output.
 * Mutates and returns `parsed`.
 */
export function validateFormOutput(parsed, transcript) {
  if (!isPlainObject(parsed)) return parsed
  const tLower = (transcript || '').toLowerCase()

  // Layer 1 — fake names
  const patient = isPlainObject(parsed.patient) ? parsed.patient : {}
  const name = patient.name ?? patient.patient_name
  if (name && FAKE_NAMES.has(String(name).trim().toLowerCase())) {
    if (isPlainObject(parsed.patient)) {
      for (const key of ['name', 'patient_name']) {
        if (key in parsed.patient) parsed.patient[key] = null
      }
    }
  }

  // Layer 2 — default-age hallucination
  const age = patient.age ?? patient.patient_age
  if (age === 30) {
    const t = transcript || ''
    if (!t.includes('30') && !t.includes('तीस')) {
      if (isPlainObject(parsed.patient)) {
        for (const key of ['age', 'patient_age']) {
          if (key in parsed.patient) parsed.patient[key] = null
        }
      }
    }
  }

  // Layer 3a — blood group invented
  const lab = isPlainObject(parsed.lab_results) ? parsed.lab_results : {}
  const bg = lab.blood_group
  if (bg && BLOOD_GROUPS.has(String(bg).trim().toLowerCase())) {
    const mentioned = BG_KEYWORDS.some((kw) => tLower.includes(kw))
    if (!mentioned) {
      if (!isPlainObject(parsed.lab_results)) parsed.lab_results = {}
      parsed.lab_results.blood_group = null
    }
  }

  // Layer 3b — HIV invented
  const hiv = lab.hiv_status ?? lab.hiv
  if (hiv && HIV_VALUES.has(String(hiv).trim().toLowerCase())) {
    const mentioned = HIV_KEYWORDS.some((kw) => tLower.includes(kw))
    if (!mentioned) {
      if (isPlainObject(parsed.lab_results)) {
        for (const key of ['hiv_status', 'hiv']) {
          if (key in parsed.lab_results) parsed.lab_results[key] = null
        }
      }
    }
  }

  // Layer 4 — numeric range checks
  const sections = [
    parsed,
    isPlainObject(parsed.vitals) ? parsed.vitals : null,
    isPlainObject(parsed.pregnancy) ? parsed.pregnancy : null,
    isPlainObject(parsed.anc_details) ? parsed.anc_details : null,
    isPlainObject(parsed.newborn) ? parsed.newborn : null,
  ].filter(Boolean)

  for (const section of sections) {
    for (const [field, [lo, hi]] of Object.entries(RANGES)) {
      const val = section[field]
      if (val == null) continue
      const num = Number(val)
      if (Number.isFinite(num) && (num < lo || num > hi)) {
        section[field] = null
      }
    }
  }

  return parsed
}

/**
 * Validate danger_signs array against transcript.
 * Input: { danger_signs: [...], ... }, transcript string.
 * Returns a new object with the danger_signs array filtered.
 */
export function validateDangerSigns(parsed, transcript) {
  if (!isPlainObject(parsed) || !Array.isArray(parsed.danger_signs)) return parsed

  const normTranscript = (transcript || '').replace(/\s+/g, ' ').trim()
  const validated = []

  for (const sign of parsed.danger_signs) {
    const evidence = sign.utterance_evidence || ''

    // Layer 5 — evidence length
    if (!evidence || evidence.length < 10) continue

    const normEvidence = evidence.replace(/\s+/g, ' ').trim()

    // Layer 6 — generic ASHA phrases
    if (GENERIC_PHRASES.some((p) => normEvidence.includes(p))) continue

    // Layer 7 — normal vital indicators
    if (NORMAL_INDICATORS.some((i) => normEvidence.includes(i))) continue

    // Layer 8 — transcript grounding
    let found = false
    if (normTranscript.includes(normEvidence)) {
      found = true
    } else if (normEvidence.length >= 20) {
      const minChunk = Math.min(30, normEvidence.length)
      for (let i = 0; i <= normEvidence.length - minChunk; i++) {
        if (normTranscript.includes(normEvidence.slice(i, i + minChunk))) {
          found = true
          break
        }
      }
    }
    if (!found) continue

    validated.push(sign)
  }

  // Layer 9 — all cite same evidence → drop all
  if (validated.length > 1) {
    const evidences = new Set(validated.map((s) => (s.utterance_evidence || '').trim()))
    if (evidences.size === 1) {
      return { ...parsed, danger_signs: [] }
    }
  }

  return { ...parsed, danger_signs: validated }
}
