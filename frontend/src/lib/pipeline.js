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
  DANGER_FC_SYSTEM_PROMPT,
  TOOL_FLAG_DANGER_SIGN,
  TOOL_ISSUE_REFERRAL,
  buildFormUserPrompt,
  buildDangerFCUserPrompt,
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
    options: { temperature: 0.1, max_tokens: 2048 },
  })
  const parsed = parseJsonLoose(res.text)
  if (!parsed) {
    return { form: null, raw: res.text, error: 'json-parse-failed' }
  }
  return { form: validateFormOutput(parsed, transcript), raw: res.text }
}

/**
 * Run danger-sign extraction via engine.complete with tools.
 * Parses toolCalls into { danger_signs, referral_decision } shape, then validates.
 */
export async function extractDangerSigns({ engine, transcript, visitType }) {
  const res = await engine.complete({
    messages: [
      { role: 'system', content: DANGER_FC_SYSTEM_PROMPT },
      { role: 'user', content: buildDangerFCUserPrompt(transcript, visitType) },
    ],
    tools: [TOOL_FLAG_DANGER_SIGN, TOOL_ISSUE_REFERRAL],
    options: { temperature: 0.1, max_tokens: 1024 },
  })

  const toolCalls = Array.isArray(res.toolCalls) ? res.toolCalls : []
  const dangerSigns = []
  let referral = null

  for (const tc of toolCalls) {
    const name = tc.name || tc.function?.name
    const args = tc.arguments || tc.function?.arguments
    const parsedArgs = typeof args === 'string' ? parseJsonLoose(args) : args
    if (!parsedArgs) continue
    if (name === 'flag_danger_sign') dangerSigns.push(parsedArgs)
    else if (name === 'issue_referral') referral = parsedArgs
  }

  const result = { danger_signs: dangerSigns, referral_decision: referral }
  return validateDangerSigns(result, transcript)
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
  const danger = await extractDangerSigns({ engine, transcript: normalized, visitType })
  timing.danger_ms = Date.now() - t3

  timing.total_ms = Date.now() - t0

  return {
    transcript: normalized,
    visitType,
    form,
    danger,
    timing,
    _raw: { form: raw, formError: error || null },
  }
}
