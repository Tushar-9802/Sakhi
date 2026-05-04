// On-device pipeline orchestrator. Mirrors the server-side flow in api.py:
// normalize → detectVisit → formExtract → dangerExtract → validate.
//
// Engine is injected so the pipeline can run against:
//   - Cactus on-device (import cactus as engine)
//   - A test double (for node:test)
//   - A LAN proxy (future — if we want to unify code paths)
//
// Engine contract:
//   async complete({ messages, tools?, options? }) -> { text, toolCalls? }

import { normalizeTranscript } from './hindiNormalize.js'
import { detectVisitType } from './visitTypeDetect.js'
import { validateFormOutput, validateDangerSigns } from './validation.js'
import {
  FORM_SYSTEM_PROMPT,
  DANGER_SYSTEM_PROMPT,
  buildFormUserPrompt,
  buildDangerJsonUserPrompt,
} from './prompts.js'

import ancSchema from './schemas/anc_visit.json' with { type: 'json' }
import pncSchema from './schemas/pnc_visit.json' with { type: 'json' }
import deliverySchema from './schemas/delivery.json' with { type: 'json' }
import childSchema from './schemas/child_health.json' with { type: 'json' }

export const SCHEMAS = {
  anc_visit: ancSchema,
  pnc_visit: pncSchema,
  delivery: deliverySchema,
  child_health: childSchema,
}

/**
 * Repair + parse JSON output from a loosely-constrained LLM.
 * Handles: ```json fences, trailing commas, leading/trailing whitespace.
 */
export function parseJsonLoose(text) {
  if (!text || typeof text !== 'string') return null
  let s = text.trim()
  // Strip code fences
  s = s.replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/i, '')
  // Cut to outermost object braces if there's prose around it
  const first = s.indexOf('{')
  const last = s.lastIndexOf('}')
  if (first !== -1 && last !== -1 && last > first) {
    s = s.slice(first, last + 1)
  }
  // Trailing-comma cleanup
  s = s.replace(/,(\s*[}\]])/g, '$1')
  try {
    return JSON.parse(s)
  } catch {
    return null
  }
}

/**
 * Run form extraction via engine.complete, then validate.
 */
export async function extractForm({ engine, transcript, visitType }) {
  const schema = SCHEMAS[visitType] || SCHEMAS.anc_visit
  const res = await engine.complete({
    messages: [
      { role: 'system', content: FORM_SYSTEM_PROMPT },
      { role: 'user', content: buildFormUserPrompt(transcript, schema) },
    ],
    // 768 observed sufficient for the null-filled template output on all
    // visit types — E2B INT4 trimming ~30 s vs the earlier 1024 cap.
    options: { temperature: 0.1, max_tokens: 768 },
  })
  const parsed = parseJsonLoose(res.text)
  if (!parsed) {
    return { form: null, raw: res.text, error: 'json-parse-failed' }
  }
  return { form: validateFormOutput(parsed, transcript), raw: res.text }
}

/**
 * Run danger-sign extraction via engine.complete as plain JSON (on-device E2B).
 * E2B INT4 does not reliably emit OpenAI-style tool_calls; plain JSON with a
 * schema-shaped template is far more stable. Returns { danger, raw, error? }.
 */
export async function extractDangerSigns({ engine, transcript, visitType }) {
  const res = await engine.complete({
    messages: [
      { role: 'system', content: DANGER_SYSTEM_PROMPT },
      { role: 'user', content: buildDangerJsonUserPrompt(transcript, visitType) },
    ],
    options: { temperature: 0.1, max_tokens: 1024 },
  })
  const parsed = parseJsonLoose(res.text)
  if (!parsed) {
    return {
      danger: validateDangerSigns({ danger_signs: [], referral_decision: null }, transcript),
      raw: res.text,
      error: 'json-parse-failed',
    }
  }
  const normalized = {
    danger_signs: Array.isArray(parsed.danger_signs) ? parsed.danger_signs : [],
    referral_decision: parsed.referral_decision || null,
  }
  return { danger: validateDangerSigns(normalized, transcript), raw: res.text, error: null }
}

/**
 * Full pipeline. Input: raw Hindi transcript (already normalized OR raw).
 * Output: { transcript, visitType, form, danger, timing }.
 */
export async function runPipeline({ engine, transcript, visitType: hintedVisitType = null }) {
  const timing = {}
  const t0 = Date.now()

  const normalized = normalizeTranscript(transcript)
  timing.normalize_ms = Date.now() - t0

  const t1 = Date.now()
  const visitType = hintedVisitType && hintedVisitType !== 'auto'
    ? hintedVisitType
    : detectVisitType(normalized)
  timing.detect_ms = Date.now() - t1

  const t2 = Date.now()
  const { form, raw, error } = await extractForm({ engine, transcript: normalized, visitType })
  timing.form_ms = Date.now() - t2

  const t3 = Date.now()
  const dangerOut = await extractDangerSigns({ engine, transcript: normalized, visitType })
  timing.danger_ms = Date.now() - t3

  timing.total_ms = Date.now() - t0

  return {
    transcript: normalized,
    visitType,
    form,
    danger: dangerOut.danger,
    timing,
    _raw: {
      form: raw,
      formError: error || null,
      danger: dangerOut.raw,
      dangerError: dangerOut.error || null,
    },
  }
}
