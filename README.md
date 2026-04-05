# MedScribe v2 — Voice-to-Form for ASHA Workers

Offline tool that converts Hindi/Bhojpuri home visit conversations into filled government health forms and referral decisions for India's 1 million+ ASHA workers.

**Competition:** [Gemma 4 Good Hackathon](https://www.kaggle.com/competitions/gemma-4-good-hackathon) ($200K prize pool)

## Problem

ASHA workers conduct maternal/child health home visits across rural India. Every visit ends with 45 minutes of paper forms filled by memory, then physically carried to the Primary Health Centre. Half the danger signs observed never reach the system in time.

## Solution

Phone records the conversation → Gemma 4 E4B processes audio offline → Government MCTS/HMIS forms fill themselves → Danger signs flagged with evidence → Referral decision issued immediately.

No internet. No cloud. No physician required.

```
[Hindi/Bhojpuri Audio] → Gemma 4 E4B (single model) → [Structured JSON via function calling]
                                                         ├── Pre-filled MCTS form
                                                         ├── Danger sign flags + evidence
                                                         └── Referral decision
```

## Architecture

One model replaces the entire pipeline. Gemma 4 E4B has native audio input, native function calling, and 140+ language support. Structured JSON schemas enforce that every danger sign cites exact conversational evidence — hallucination is structurally blocked.

**Deployment (LENTERA pattern):** PHC supervisor's ₹15K tablet runs Ollama serving E4B. ASHA phones connect via local WiFi. No internet required at any point.

## Key Technical Choices

| Choice | Why |
|--------|-----|
| Gemma 4 E4B | Native audio + function calling + Hindi in one model |
| Function calling schemas | Structural anti-hallucination (evidence required per flag) |
| Unsloth LoRA fine-tune | Train on synthetic ASHA visit data, $2-5 cost |
| Ollama serving | Offline deployment on commodity hardware |
| 30-sec audio chunking | E4B audio limit = 30s; conversations chunked with overlap |

## Form Types Supported

- **ANC (Antenatal Care)** — pregnancy registration, vitals, lab results, birth preparedness
- **Delivery** — birth outcome, infant details, complications
- **PNC / HBNC** — postnatal mother + newborn assessment (days 1-42)
- **Child Health / HBYC** — growth, immunization, development, illness screening
- **Danger Signs** — NHM/IMNCI protocol with mandatory utterance evidence grounding

## Status

Phase 1 — Foundation (environment, schemas, baseline testing)
