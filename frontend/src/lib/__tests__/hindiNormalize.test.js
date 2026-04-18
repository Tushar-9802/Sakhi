import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
  WORD_TO_NUM,
  parseHindiNumber,
  convertNumbers,
  normalizeTranscript,
} from '../hindiNormalize.js'

test('WORD_TO_NUM has 160+ entries covering 0-99 + 100', () => {
  assert.ok(Object.keys(WORD_TO_NUM).length >= 160)
  assert.equal(WORD_TO_NUM['शून्य'], 0)
  assert.equal(WORD_TO_NUM['एक'], 1)
  assert.equal(WORD_TO_NUM['दस'], 10)
  assert.equal(WORD_TO_NUM['सौ'], 100)
})

test('parseHindiNumber single-word lookups', () => {
  assert.equal(parseHindiNumber('शून्य'), 0)
  assert.equal(parseHindiNumber('एक'), 1)
  assert.equal(parseHindiNumber('दस'), 10)
  assert.equal(parseHindiNumber('सत्तर'), 70)
  assert.equal(parseHindiNumber('अट्ठावन'), 58)
  assert.equal(parseHindiNumber('सौ'), 100)
})

test('parseHindiNumber compound phrases', () => {
  assert.equal(parseHindiNumber('एक सौ दस'), 110)
  assert.equal(parseHindiNumber('एक सौ पचपन'), 155)
  assert.equal(parseHindiNumber('दो सौ'), 200)
  assert.equal(parseHindiNumber('पाँच सौ'), 500)
})

test('parseHindiNumber returns null on empty / non-number', () => {
  assert.equal(parseHindiNumber(''), null)
  assert.equal(parseHindiNumber('   '), null)
  assert.equal(parseHindiNumber('नमस्ते'), null)
})

test('parseHindiNumber stops at first non-number word (mirrors Python bug)', () => {
  // Python breaks the loop on unknown word, returns `total + current`.
  // Since `total` is never incremented, it returns `current` so far.
  assert.equal(parseHindiNumber('दस नमस्ते बीस'), 10)
})

test('convertNumbers replaces number words with digits', () => {
  assert.equal(convertNumbers('एक सौ दस'), '110')
  assert.equal(convertNumbers('एक सौ दस बटा सत्तर'), '110 बटा 70')
  assert.equal(convertNumbers('अट्ठावन kg'), '58 kg')
  // 'बटा' is a medical abbrev, normalizer replaces it with '/' — but convertNumbers alone doesn't.
})

test('convertNumbers handles compound splits (Whisper artifacts)', () => {
  // "एकसो" (merged) should split to "एक सो" and become 100
  assert.equal(convertNumbers('एकसो दस'), '110')
  assert.equal(convertNumbers('दोसो पचास'), '250')
})

test('normalizeTranscript full pipeline - BP reading', () => {
  const out = normalizeTranscript('आपका BP एक सौ दस बटा सत्तर है, वजन अट्ठावन kg')
  // After medical-term replace + number convert + space-around-slash cleanup
  assert.ok(out.includes('110/70'))
  assert.ok(out.includes('58 kg'))
})

test('normalizeTranscript converts बीपी → BP', () => {
  const out = normalizeTranscript('बीपी एक सौ दस')
  assert.ok(out.startsWith('BP '))
  assert.ok(out.includes('110'))
})

test('normalizeTranscript fixes repetition artifacts', () => {
  const out = normalizeTranscript('ठीकठीकठीकठीक है')
  // 4+ consecutive repeats should collapse to 1
  assert.ok(!/(ठीक){4,}/.test(out))
})

test('normalizeTranscript handles decimal via दशमलव', () => {
  const out = normalizeTranscript('ग्यारह दशमलव पाँच')
  // दशमलव → '.', numbers → digits, then digit-dot-digit whitespace cleanup
  assert.ok(out.includes('11'))
  assert.ok(out.includes('5'))
})

test('normalizeTranscript adds line break after ।', () => {
  const out = normalizeTranscript('वजन बढ़ रहा है। BP ठीक है।')
  assert.ok(out.includes('।\n'))
})

test('normalizeTranscript trims trailing punctuation/whitespace', () => {
  const out = normalizeTranscript('  ठीक है.  ')
  assert.equal(out, 'ठीक है')
})

test('normalizeTranscript preserves English medical terms', () => {
  const out = normalizeTranscript('BP ठीक, IFA दे दी')
  assert.ok(out.includes('BP'))
  assert.ok(out.includes('IFA'))
})
