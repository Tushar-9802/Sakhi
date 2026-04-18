// System prompts + tool specs for on-device Gemma 4 inference.
// Verbatim port of Python constants in app.py — keep the wording identical
// to what the laptop pipeline uses so results are comparable.

// From app.py:38-49
export const FORM_SYSTEM_PROMPT = (
  "You are a clinical data extraction system for India's ASHA health worker program. " +
  "Extract structured data from the Hindi/Hinglish home visit conversation into the requested JSON schema. " +
  "ONLY extract information explicitly stated in the conversation. Use null for any field not mentioned.\n\n" +
  "STRICT RULES:\n" +
  "1. Do NOT invent names, dates, phone numbers, or addresses. If the patient is only called 'दीदी' or 'बहन', set name to null.\n" +
  "2. If age is not explicitly stated as a number, set age to null. Do NOT guess from context.\n" +
  "3. If blood group, HIV status, or other lab tests are not discussed, they MUST be null — never assume 'negative' or a default group.\n" +
  "4. If the conversation has no speaker labels (ASHA/Patient), still extract data but be extra strict about nulls.\n" +
  "5. Numbers may appear as Hindi words (e.g., 'एक सो दस बटा सत्तर' = 110/70). Convert them to digits.\n" +
  "6. Distinguish ABSOLUTE weight from weight CHANGE. 'वजन 55 किलो है' = 55 absolute; '3 किलो बढ़ गया' or 'गेन' = weight change — leave absolute weight_kg null.\n" +
  "7. If the patient says she is currently taking IFA / folic acid / calcium tablets ('ले रही हूँ'), set the corresponding *_given field to the stated value (e.g., 'daily') or a truthy string like 'yes'. Do not leave it null when she confirms taking them.\n" +
  "Return valid JSON only."
)

// From app.py:51-61
export const DANGER_SYSTEM_PROMPT = (
  "You are a clinical danger sign detection system for India's ASHA health worker program. " +
  "Analyze the Hindi/Hinglish home visit conversation for NHM-defined danger signs.\n\n" +
  "STRICT RULES:\n" +
  "1. ONLY flag a danger sign if the EXACT words proving it appear in the conversation.\n" +
  "2. utterance_evidence MUST be a verbatim copy-paste from the conversation — do NOT paraphrase or fabricate.\n" +
  "3. If a vital sign is NORMAL (e.g., BP 110/70, temperature 37°C), that is NOT a danger sign.\n" +
  "4. Most routine visits have ZERO danger signs. Return an empty danger_signs array when none exist.\n" +
  "5. When in doubt, do NOT flag — a missed flag is better than a false alarm.\n" +
  "Return valid JSON only."
)

/**
 * Convert a JSON Schema definition to a null-filled instance matching the
 * schema's shape. Smaller models (Gemma 4 E2B INT4) echo schema metadata
 * ($schema/title/description/type) back when given the raw schema — sending a
 * concrete null-valued template sidesteps that failure mode.
 */
function schemaToTemplate(schema) {
  if (!schema || typeof schema !== 'object') return null
  if (schema.type === 'object' && schema.properties) {
    const obj = {}
    for (const [key, prop] of Object.entries(schema.properties)) {
      obj[key] = schemaToTemplate(prop)
    }
    return obj
  }
  if (schema.type === 'array') return []
  // Primitives and union types default to null in the template.
  return null
}

/**
 * Build the user prompt for form extraction. On-device (E2B INT4) version —
 * sends a null-filled template rather than the raw JSON Schema envelope.
 * Mirrors app.py:853-857 in intent; the server-side path uses its own prompt.
 */
export function buildFormUserPrompt(transcript, schema) {
  const template = schemaToTemplate(schema)
  return (
    'Extract structured data from this ASHA home visit conversation.\n\n' +
    'Conversation:\n' + transcript + '\n\n' +
    'Fill in the JSON template below. Keep every key exactly as shown. ' +
    'Set a value only when the conversation explicitly states it — leave all ' +
    'other keys as null. Do NOT add extra keys. Do NOT include "$schema", ' +
    '"title", "description", "type", or "properties" in your output. Return ' +
    'only the filled-in JSON object.\n\n' +
    'Template:\n' +
    JSON.stringify(template, null, 2)
  )
}

/**
 * Build the user prompt for danger-sign plain-JSON extraction (on-device E2B).
 * E2B INT4 does not reliably emit OpenAI-style function calls — plain JSON with
 * a schema-shaped template is more stable.
 */
export function buildDangerJsonUserPrompt(transcript, visitType) {
  return (
    'Analyze this ASHA home visit conversation for danger signs.\n\n' +
    'Visit type: ' + visitType + '\n\n' +
    'Conversation:\n' + transcript + '\n\n' +
    'Return ONLY a JSON object with exactly these two keys:\n\n' +
    '{\n' +
    '  "danger_signs": [\n' +
    '    {\n' +
    '      "sign": "<short name, e.g. severe_preeclampsia, high_bp_with_symptoms, severe_anemia, swelling_face_hands>",\n' +
    '      "category": "immediate_referral" | "urgent_care" | "monitor_closely",\n' +
    '      "clinical_value": "<measured value like 150/95, or null if no number>",\n' +
    '      "utterance_evidence": "<exact verbatim Hindi/English quote from the conversation>"\n' +
    '    }\n' +
    '  ],\n' +
    '  "referral_decision": {\n' +
    '    "decision": "refer_immediately" | "refer_within_24h" | "continue_monitoring" | "routine_followup",\n' +
    '    "reason": "<brief clinical reason>"\n' +
    '  }\n' +
    '}\n\n' +
    'Rules:\n' +
    '- If no danger signs are present, set "danger_signs" to [] and "referral_decision" to null.\n' +
    '- utterance_evidence MUST be a verbatim copy-paste from the conversation.\n' +
    '- A vital sign in the normal range (BP 110/70, temperature 37°C, Hb > 11) is NOT a danger sign.\n' +
    '- Do NOT include any prose, markdown fences, or extra keys. Output ONLY the JSON object.'
  )
}
