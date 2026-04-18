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

// From app.py:346-358
export const DANGER_FC_SYSTEM_PROMPT = (
  "You are a clinical danger sign detection system for India's ASHA health worker program.\n\n" +
  "Analyze the conversation and use the provided tools:\n" +
  "1. flag_danger_sign — call ONCE per danger sign found. Evidence MUST be a verbatim quote from the conversation. " +
  "If NO danger signs exist, do NOT call any tool.\n" +
  "2. issue_referral — call only if danger signs warrant referral to a facility.\n\n" +
  "STRICT RULES:\n" +
  "- ONLY flag a danger sign if the EXACT words proving it appear in the conversation.\n" +
  "- utterance_evidence MUST be a verbatim copy-paste from the conversation — do NOT paraphrase.\n" +
  "- If a vital sign is NORMAL (e.g., BP 110/70, temperature 37°C), that is NOT a danger sign.\n" +
  "- Most routine visits have ZERO danger signs. Do NOT call any tools for normal visits.\n" +
  "- When in doubt, do NOT flag — a missed flag is better than a false alarm."
)

// From app.py:283-315
export const TOOL_FLAG_DANGER_SIGN = {
  type: 'function',
  function: {
    name: 'flag_danger_sign',
    description:
      'Flag a single danger sign detected in the patient conversation. ' +
      'Call once per danger sign found. Do NOT call if no danger signs exist. ' +
      'The evidence field MUST be an exact verbatim quote from the conversation.',
    parameters: {
      type: 'object',
      properties: {
        sign: {
          type: 'string',
          description: 'Standard NHM danger sign name (e.g., severe_preeclampsia, severe_anemia)',
        },
        category: {
          type: 'string',
          enum: ['immediate_referral', 'urgent_care', 'monitor_closely'],
        },
        clinical_value: {
          type: ['string', 'null'],
          description: "Measured value if applicable (e.g., '145/95', '38.5C')",
        },
        utterance_evidence: {
          type: 'string',
          description: 'REQUIRED: exact verbatim quote from conversation proving this sign',
        },
      },
      required: ['sign', 'category', 'utterance_evidence'],
    },
  },
}

// From app.py:317-344
export const TOOL_ISSUE_REFERRAL = {
  type: 'function',
  function: {
    name: 'issue_referral',
    description:
      'Issue a referral decision based on detected danger signs. ' +
      'Only call if danger signs warrant referral. Do NOT call for routine visits.',
    parameters: {
      type: 'object',
      properties: {
        urgency: {
          type: 'string',
          enum: ['immediate', 'within_24h', 'routine'],
        },
        facility: {
          type: ['string', 'null'],
          enum: ['PHC', 'CHC', 'district_hospital', 'FRU', null],
        },
        reason: {
          type: 'string',
          description: 'Brief clinical reasoning for referral',
        },
      },
      required: ['urgency', 'facility', 'reason'],
    },
  },
}

/**
 * Build the user prompt for form extraction.
 * Mirrors app.py:853-857.
 */
export function buildFormUserPrompt(transcript, schema) {
  return (
    'Extract structured data from this ASHA home visit conversation:\n\n' +
    transcript +
    '\n\nOutput JSON schema:\n' +
    JSON.stringify(schema)
  )
}

/**
 * Build the user prompt for danger-sign function-calling extraction.
 * Mirrors app.py:372-376.
 */
export function buildDangerFCUserPrompt(transcript, visitType) {
  return (
    'Analyze this ASHA home visit conversation for danger signs.\n\n' +
    'Visit type: ' + visitType + '\n\n' +
    'Conversation:\n' + transcript
  )
}
