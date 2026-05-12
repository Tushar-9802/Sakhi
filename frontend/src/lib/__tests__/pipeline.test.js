import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
  parseJsonLoose,
  extractForm,
  extractDangerSigns,
  runPipeline,
  applyMetadata,
  SCHEMAS,
} from '../pipeline.js'

// -----------------------
// JSON repair parser
// -----------------------

test('parseJsonLoose plain object', () => {
  assert.deepEqual(parseJsonLoose('{"a":1}'), { a: 1 })
})

test('parseJsonLoose strips ```json fences', () => {
  assert.deepEqual(parseJsonLoose('```json\n{"a":1}\n```'), { a: 1 })
  assert.deepEqual(parseJsonLoose('```\n{"a":1}\n```'), { a: 1 })
})

test('parseJsonLoose handles trailing commas', () => {
  assert.deepEqual(parseJsonLoose('{"a":1,"b":2,}'), { a: 1, b: 2 })
  assert.deepEqual(parseJsonLoose('{"a":[1,2,3,]}'), { a: [1, 2, 3] })
})

test('parseJsonLoose cuts prose around object', () => {
  const input = 'Here is the JSON:\n{"a":1}\nThat is the answer.'
  assert.deepEqual(parseJsonLoose(input), { a: 1 })
})

test('parseJsonLoose returns null on garbage', () => {
  assert.equal(parseJsonLoose(''), null)
  assert.equal(parseJsonLoose('not json at all'), null)
  assert.equal(parseJsonLoose(null), null)
  assert.equal(parseJsonLoose(undefined), null)
})

// -----------------------
// SCHEMAS — JSON imports work
// -----------------------

test('SCHEMAS loads all 4 visit-type schemas', () => {
  assert.ok(SCHEMAS.anc_visit)
  assert.ok(SCHEMAS.pnc_visit)
  assert.ok(SCHEMAS.delivery)
  assert.ok(SCHEMAS.child_health)
  assert.equal(SCHEMAS.anc_visit.title, 'ANC Visit Extraction')
})

// -----------------------
// Mock engine for pipeline tests
// -----------------------

function mockEngine({ formText, dangerText = '{"danger_signs":[],"referral_decision":null}' }) {
  let call = 0
  return {
    complete: async () => {
      call++
      if (call === 1) return { text: formText }
      return { text: dangerText }
    },
  }
}

// -----------------------
// extractForm
// -----------------------

test('extractForm happy path: valid JSON from engine', async () => {
  const engine = mockEngine({
    formText: '{"patient":{"name":"सुनीता","age":25},"vitals":{"bp_systolic":120,"bp_diastolic":80}}',
  })
  const out = await extractForm({ engine, transcript: 'सुनीता जी, BP 120/80 है', visitType: 'anc_visit' })
  assert.equal(out.form.patient.name, 'सुनीता')
  assert.equal(out.form.vitals.bp_systolic, 120)
})

test('extractForm validates: hallucinated दीदी nulled', async () => {
  const engine = mockEngine({
    formText: '{"patient":{"name":"दीदी","age":30}}',
  })
  const out = await extractForm({
    engine,
    transcript: 'नमस्ते दीदी',  // no age mention
    visitType: 'anc_visit',
  })
  assert.equal(out.form.patient.name, null)
  assert.equal(out.form.patient.age, null)
})

test('extractForm malformed JSON → returns error', async () => {
  const engine = mockEngine({ formText: 'not json at all' })
  const out = await extractForm({ engine, transcript: 't', visitType: 'anc_visit' })
  assert.equal(out.form, null)
  assert.equal(out.error, 'json-parse-failed')
})

// -----------------------
// extractDangerSigns
// -----------------------

test('extractDangerSigns parses JSON output (on-device path)', async () => {
  const transcript = 'सिर में बहुत दर्द हो रहा है और धुंधला दिख रहा है'
  const engine = {
    complete: async () => ({
      text: JSON.stringify({
        danger_signs: [{
          sign: 'severe_headache',
          category: 'immediate_referral',
          clinical_value: null,
          utterance_evidence: 'सिर में बहुत दर्द हो रहा है',
        }],
        referral_decision: { decision: 'refer_immediately', reason: 'preeclampsia suspected' },
      }),
    }),
  }
  const out = await extractDangerSigns({ engine, transcript, visitType: 'anc_visit' })
  assert.equal(out.danger.danger_signs.length, 1)
  assert.equal(out.danger.danger_signs[0].sign, 'severe_headache')
  assert.equal(out.danger.referral_decision.decision, 'refer_immediately')
  assert.ok(typeof out.raw === 'string')
})

test('extractDangerSigns handles fenced JSON', async () => {
  const engine = {
    complete: async () => ({
      text: '```json\n{"danger_signs":[{"sign":"severe_headache","category":"immediate_referral","utterance_evidence":"सिर में बहुत दर्द हो रहा है"}],"referral_decision":null}\n```',
    }),
  }
  const out = await extractDangerSigns({
    engine,
    transcript: 'सिर में बहुत दर्द हो रहा है',
    visitType: 'anc_visit',
  })
  assert.equal(out.danger.danger_signs.length, 1)
})

test('extractDangerSigns validates away ungrounded evidence', async () => {
  const engine = {
    complete: async () => ({
      text: JSON.stringify({
        danger_signs: [{
          sign: 'seizure',
          category: 'immediate_referral',
          utterance_evidence: 'मिर्गी के दौरे आए कल',  // not in transcript
        }],
        referral_decision: null,
      }),
    }),
  }
  const out = await extractDangerSigns({
    engine,
    transcript: 'BP normal है, कोई तकलीफ नहीं',
    visitType: 'anc_visit',
  })
  assert.equal(out.danger.danger_signs.length, 0)
})

test('extractDangerSigns malformed JSON → empty result with error flag', async () => {
  const engine = { complete: async () => ({ text: 'not json' }) }
  const out = await extractDangerSigns({ engine, transcript: 't', visitType: 'anc_visit' })
  assert.equal(out.danger.danger_signs.length, 0)
  assert.equal(out.error, 'json-parse-failed')
})

// -----------------------
// runPipeline (full)
// -----------------------

test('runPipeline end-to-end with mock engine', async () => {
  const transcript = 'सुनीता जी, आपका BP एक सौ बीस बटा अस्सी है, वजन अट्ठावन kg. 24 हफ्ते की हैं.'
  const engine = mockEngine({
    formText: '{"patient":{"name":"सुनीता"},"vitals":{"bp_systolic":120,"bp_diastolic":80,"weight_kg":58},"pregnancy":{"gestational_weeks":24}}',
    dangerToolCalls: [],  // no danger signs
  })
  const out = await runPipeline({ engine, transcript })
  assert.equal(out.visitType, 'anc_visit')
  assert.ok(out.transcript.includes('120/80'))
  assert.ok(out.transcript.includes('58 kg'))
  assert.equal(out.form.patient.name, 'सुनीता')
  assert.equal(out.form.vitals.bp_systolic, 120)
  assert.equal(out.danger.danger_signs.length, 0)
  assert.ok(out.timing.total_ms >= 0)
})

test('runPipeline respects hintedVisitType', async () => {
  const engine = mockEngine({ formText: '{"patient":{}}' })
  const out = await runPipeline({ engine, transcript: 'generic text', visitType: 'delivery' })
  assert.equal(out.visitType, 'delivery')
})

test('runPipeline falls back to auto-detect when hint is "auto"', async () => {
  const engine = mockEngine({ formText: '{"patient":{}}' })
  const out = await runPipeline({
    engine,
    transcript: 'नवजात दूध पी रहा है',  // → pnc_visit
    visitType: 'auto',
  })
  assert.equal(out.visitType, 'pnc_visit')
})

// -----------------------
// applyMetadata — mirrors app.py:apply_metadata
// -----------------------

test('applyMetadata anc: name + years-age + mobile override', () => {
  const form = { patient: { name: 'दीदी', age: 99 }, vitals: { bp_systolic: 120 } }
  const out = applyMetadata(form, 'anc_visit', {
    patient_name: 'सुनीता', patient_age: '24', age_unit: 'years',
    patient_mobile: '9876543210',
  })
  assert.equal(out.patient.name, 'सुनीता')
  assert.equal(out.patient.age, 24)
  assert.equal(out.patient.mobile, '9876543210')
  assert.equal(out.vitals.bp_systolic, 120) // unrelated field untouched
})

test('applyMetadata anc: months-unit does not write patient.age (ANC schema is years)', () => {
  const form = { patient: { name: null, age: null } }
  const out = applyMetadata(form, 'anc_visit', {
    patient_name: 'X', patient_age: '6', age_unit: 'months',
  })
  assert.equal(out.patient.name, 'X')
  assert.equal(out.patient.age, null)
})

test('applyMetadata child_health: years → age_months conversion + sex', () => {
  const form = { child: { name: 'सोनम', age_months: null, sex: 'female' } }
  const out = applyMetadata(form, 'child_health', {
    patient_name: 'आरव', patient_age: '2', age_unit: 'years', patient_sex: 'male',
  })
  assert.equal(out.child.name, 'आरव')
  assert.equal(out.child.age_months, 24)
  assert.equal(out.child.sex, 'male')
})

test('applyMetadata child_health: months passed through', () => {
  const form = { child: { name: null, age_months: null, sex: null } }
  const out = applyMetadata(form, 'child_health', {
    patient_name: 'आरव', patient_age: '14', age_unit: 'months', patient_sex: 'male',
  })
  assert.equal(out.child.age_months, 14)
})

test('applyMetadata pnc_visit: envelope-only, form untouched', () => {
  const form = { patient_id: null, mother: { vitals: { bp_systolic: 120 } } }
  const out = applyMetadata(form, 'pnc_visit', {
    patient_name: 'सुनीता', patient_age: '24', age_unit: 'years',
  })
  assert.deepEqual(out, form) // schema has no patient block, no merge
})

test('applyMetadata null metadata: pass-through', () => {
  const form = { patient: { name: 'X' } }
  assert.equal(applyMetadata(form, 'anc_visit', null), form)
})

test('applyMetadata null form: pass-through', () => {
  assert.equal(applyMetadata(null, 'anc_visit', { patient_name: 'X' }), null)
})

test('applyMetadata empty strings: ignored', () => {
  const form = { patient: { name: 'preserved', age: 30 } }
  const out = applyMetadata(form, 'anc_visit', {
    patient_name: '', patient_age: '', patient_mobile: '',
  })
  assert.equal(out.patient.name, 'preserved')
  assert.equal(out.patient.age, 30)
})

test('runPipeline returns metadata envelope and applies override', async () => {
  const engine = mockEngine({
    formText: '{"child":{"name":"सोनम","age_months":null,"sex":"female","weight_kg":null}}',
  })
  const out = await runPipeline({
    engine,
    transcript: 'बच्चे को 3 दिन से दस्त है',
    visitType: 'child_health',
    metadata: { patient_name: 'आरव', patient_age: '14', age_unit: 'months', patient_sex: 'male', asha_id: 'ASHA-1' },
  })
  assert.equal(out.form.child.name, 'आरव')
  assert.equal(out.form.child.age_months, 14)
  assert.equal(out.form.child.sex, 'male')
  assert.equal(out.metadata.patient_name, 'आरव')
  assert.equal(out.metadata.patient_age, 14) // string → number in envelope
  assert.equal(out.metadata.asha_id, 'ASHA-1')
})

test('runPipeline metadata envelope is null when no metadata passed', async () => {
  const engine = mockEngine({ formText: '{"patient":{}}' })
  const out = await runPipeline({ engine, transcript: 'generic' })
  assert.equal(out.metadata, null)
})
