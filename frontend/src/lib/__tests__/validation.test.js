import { test } from 'node:test'
import assert from 'node:assert/strict'
import { validateFormOutput, validateDangerSigns } from '../validation.js'

// -----------------------
// Form validation (Layers 1-4)
// -----------------------

test('L1 name hallucination: दीदी gets nulled', () => {
  const out = validateFormOutput(
    { patient: { name: 'दीदी', age: 25 } },
    'नमस्ते दीदी, कैसी हैं?',
  )
  assert.equal(out.patient.name, null)
  assert.equal(out.patient.age, 25)  // age untouched
})

test('L1 real names preserved', () => {
  const out = validateFormOutput(
    { patient: { name: 'सुनीता' } },
    'सुनीता जी, BP देख लेती हूँ',
  )
  assert.equal(out.patient.name, 'सुनीता')
})

test('L2 default-age 30 hallucinated → nulled when not in transcript', () => {
  const out = validateFormOutput(
    { patient: { age: 30 } },
    'कैसी हैं? BP ठीक है',  // no 30, no तीस
  )
  assert.equal(out.patient.age, null)
})

test('L2 age 30 preserved when transcript mentions it', () => {
  const out = validateFormOutput(
    { patient: { age: 30 } },
    '30 साल की हूँ',
  )
  assert.equal(out.patient.age, 30)
})

test('L2 age 30 preserved when transcript mentions तीस', () => {
  const out = validateFormOutput(
    { patient: { age: 30 } },
    'तीस साल की हूँ',
  )
  assert.equal(out.patient.age, 30)
})

test('L3a blood_group invented → nulled', () => {
  const out = validateFormOutput(
    { lab_results: { blood_group: 'O+' } },
    'BP 110/70 है',  // no blood group mention
  )
  assert.equal(out.lab_results.blood_group, null)
})

test('L3a blood_group preserved when mentioned', () => {
  const out = validateFormOutput(
    { lab_results: { blood_group: 'O+' } },
    'blood group O+ है',
  )
  assert.equal(out.lab_results.blood_group, 'O+')
})

test('L3b HIV invented → nulled', () => {
  const out = validateFormOutput(
    { lab_results: { hiv_status: 'negative' } },
    'वजन अच्छा है',
  )
  assert.equal(out.lab_results.hiv_status, null)
})

test('L3b HIV preserved when mentioned', () => {
  const out = validateFormOutput(
    { lab_results: { hiv_status: 'negative' } },
    'HIV test negative आया',
  )
  assert.equal(out.lab_results.hiv_status, 'negative')
})

test('L4 BP out of range → nulled', () => {
  const out = validateFormOutput(
    { vitals: { bp_systolic: 300, bp_diastolic: 80 } },
    'transcript',
  )
  assert.equal(out.vitals.bp_systolic, null)
  assert.equal(out.vitals.bp_diastolic, 80)  // in range
})

test('L4 weight out of range → nulled', () => {
  const out = validateFormOutput(
    { vitals: { weight_kg: 250 } },
    't',
  )
  assert.equal(out.vitals.weight_kg, null)
})

test('L4 gestation out of range → nulled', () => {
  const out = validateFormOutput(
    { pregnancy: { gestational_weeks: 50 } },
    't',
  )
  assert.equal(out.pregnancy.gestational_weeks, null)
})

test('L4 valid ranges preserved', () => {
  const out = validateFormOutput(
    { vitals: { bp_systolic: 120, bp_diastolic: 80, weight_kg: 58, hemoglobin_gm_percent: 11.5 } },
    't',
  )
  assert.equal(out.vitals.bp_systolic, 120)
  assert.equal(out.vitals.bp_diastolic, 80)
  assert.equal(out.vitals.weight_kg, 58)
  assert.equal(out.vitals.hemoglobin_gm_percent, 11.5)
})

test('non-object input returned as-is', () => {
  assert.equal(validateFormOutput(null, 't'), null)
  assert.equal(validateFormOutput('string', 't'), 'string')
  assert.deepEqual(validateFormOutput([1, 2], 't'), [1, 2])
})

// -----------------------
// Danger-sign validation (Layers 5-9)
// -----------------------

test('L5 evidence too short (<10 chars) → dropped', () => {
  const transcript = 'सिरदर्द हो रहा है, और चक्कर भी आ रहे हैं'
  const out = validateDangerSigns(
    { danger_signs: [{ sign: 'headache', utterance_evidence: 'दर्द' }] },
    transcript,
  )
  assert.deepEqual(out.danger_signs, [])
})

test('L6 generic ASHA phrase → dropped', () => {
  const transcript = 'कोई तकलीफ़ हो तो फ़ोन कर दीजिए, ठीक है'
  const out = validateDangerSigns(
    {
      danger_signs: [{
        sign: 'generic',
        utterance_evidence: 'कोई तकलीफ़ हो तो फ़ोन कर दीजिए',
      }],
    },
    transcript,
  )
  assert.deepEqual(out.danger_signs, [])
})

test('L7 normal vital indicator → dropped', () => {
  const transcript = 'BP 110/70 है, बिल्कुल ठीक है'
  const out = validateDangerSigns(
    {
      danger_signs: [{
        sign: 'hypertension',
        utterance_evidence: 'BP 110/70 है, बिल्कुल ठीक',
      }],
    },
    transcript,
  )
  assert.deepEqual(out.danger_signs, [])
})

test('L8 evidence not in transcript → dropped', () => {
  const transcript = 'BP चेक किया, सब ठीक है'
  const out = validateDangerSigns(
    {
      danger_signs: [{
        sign: 'seizure',
        utterance_evidence: 'मिर्गी के दौरे आए पिछले हफ्ते',
      }],
    },
    transcript,
  )
  assert.deepEqual(out.danger_signs, [])
})

test('L8 evidence in transcript → kept', () => {
  const transcript = 'सिर बहुत दर्द कर रहा है, और आँखों के सामने धुंधला हो रहा है'
  const out = validateDangerSigns(
    {
      danger_signs: [{
        sign: 'severe_headache',
        utterance_evidence: 'सिर बहुत दर्द कर रहा है',
      }],
    },
    transcript,
  )
  assert.equal(out.danger_signs.length, 1)
  assert.equal(out.danger_signs[0].sign, 'severe_headache')
})

test('L8 30-char chunk fallback matches', () => {
  const transcript = 'बहुत तेज़ सिरदर्द और उल्टी भी हो रही है कल से'
  // Evidence slightly paraphrased but 30-char chunks overlap
  const out = validateDangerSigns(
    {
      danger_signs: [{
        sign: 'headache_vomiting',
        utterance_evidence: 'बहुत तेज़ सिरदर्द और उल्टी भी हो रही है',
      }],
    },
    transcript,
  )
  assert.equal(out.danger_signs.length, 1)
})

test('L9 all signs cite same evidence → all dropped', () => {
  const transcript = 'सिर बहुत दर्द कर रहा है तीन दिन से'
  const out = validateDangerSigns(
    {
      danger_signs: [
        { sign: 'a', utterance_evidence: 'सिर बहुत दर्द कर रहा है तीन दिन से' },
        { sign: 'b', utterance_evidence: 'सिर बहुत दर्द कर रहा है तीन दिन से' },
        { sign: 'c', utterance_evidence: 'सिर बहुत दर्द कर रहा है तीन दिन से' },
      ],
    },
    transcript,
  )
  assert.deepEqual(out.danger_signs, [])
})

test('L9 different evidence → all kept', () => {
  const transcript = 'सिर में बहुत दर्द है और आँखों से धुंधला दिखता है'
  const out = validateDangerSigns(
    {
      danger_signs: [
        { sign: 'headache', utterance_evidence: 'सिर में बहुत दर्द है' },
        { sign: 'vision', utterance_evidence: 'आँखों से धुंधला दिखता है' },
      ],
    },
    transcript,
  )
  assert.equal(out.danger_signs.length, 2)
})

test('no danger_signs array → passthrough', () => {
  const input = { danger_signs: undefined }
  assert.equal(validateDangerSigns(input, 't'), input)
})

test('non-object input → passthrough', () => {
  assert.equal(validateDangerSigns(null, 't'), null)
  assert.equal(validateDangerSigns('x', 't'), 'x')
})
