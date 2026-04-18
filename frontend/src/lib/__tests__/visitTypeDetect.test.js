import { test } from 'node:test'
import assert from 'node:assert/strict'
import { detectVisitType } from '../visitTypeDetect.js'

test('delivery: explicit delivery phrase', () => {
  assert.equal(detectVisitType('कल रात डिलीवरी हो गई। लड़का हुआ'), 'delivery')
  assert.equal(detectVisitType('घर पर ही हो गया, दाई ने करवाई'), 'delivery')
  assert.equal(detectVisitType('सिजेरियन से हुई'), 'delivery')
})

test('anc_visit: pregnancy keywords', () => {
  assert.equal(detectVisitType('24 हफ्ते की हूँ। BP चेक कर लो'), 'anc_visit')
  assert.equal(detectVisitType('गर्भवती हूँ, TT का टीका लगाना है'), 'anc_visit')
  assert.equal(detectVisitType('IFA दे दी, बच्चे की हलचल ठीक है'), 'anc_visit')
})

test('pnc_visit: postpartum/newborn keywords', () => {
  assert.equal(detectVisitType('नवजात कैसा है? दूध पी रहा है'), 'pnc_visit')
  assert.equal(detectVisitType('नाभि सूख गई, PNC visit है'), 'pnc_visit')
  assert.equal(detectVisitType('स्तनपान कैसा है?'), 'pnc_visit')
})

test('child_health: older-child keywords', () => {
  assert.equal(detectVisitType('बच्चे को दस्त हैं 3 दिन से'), 'child_health')
  assert.equal(detectVisitType('8 महीने का है, टीका लगवाना है'), 'child_health')
  assert.equal(detectVisitType('बहुत सुस्त है, आँखें धँसी हुई हैं'), 'child_health')
})

test('default: unknown transcript → anc_visit', () => {
  assert.equal(detectVisitType('नमस्ते, कैसी हैं आप'), 'anc_visit')
  assert.equal(detectVisitType(''), 'anc_visit')
  assert.equal(detectVisitType(null), 'anc_visit')
})

test('ordering: delivery beats ANC when both keywords present', () => {
  // Mixed transcript: delivery mentioned alongside ANC concepts
  const t = 'पिछले हफ्ते डिलीवरी हो गई। पहले गर्भ के समय BP चेक किया था'
  assert.equal(detectVisitType(t), 'delivery')
})

test('ordering: ANC beats PNC when both present', () => {
  const t = 'गर्भवती हूँ, डिलीवरी कहाँ करूँ? दूध पीने वाला भाई भी है'
  assert.equal(detectVisitType(t), 'anc_visit')
})

test('case insensitive on English keywords', () => {
  assert.equal(detectVisitType('PREGNANCY चल रही है'), 'anc_visit')
  assert.equal(detectVisitType('PNC visit today'), 'pnc_visit')
})
