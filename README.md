# Sakhi (सखी) — Voice-to-Form for ASHA Workers

Offline-first tool that converts Hindi home visit conversations into structured government health forms and real-time referral decisions for India's 1 million+ ASHA health workers.

**Competition:** [Gemma 4 Good Hackathon](https://www.kaggle.com/competitions/gemma-4-good-hackathon) ($200K prize pool)
**Tracks:** Health & Sciences (primary), Digital Equity (secondary), Ollama deployment

## Problem

India's ASHA workers conduct 50M+ maternal/child health home visits per year across rural areas. Every visit ends with paper forms filled from memory, then physically carried to the Primary Health Center. Danger signs observed in the field — preeclampsia, postpartum hemorrhage, neonatal distress — often never reach the system in time for intervention.

## Solution

Phone records the conversation → Whisper transcribes Hindi audio → Hindi normalization → Gemma 4 extracts structured form data + detects danger signs with evidence → Referral decision issued immediately.

No internet required. No cloud dependency. No physician needed at point of care.

```
[Hindi Audio] → Whisper ASR → Hindi Normalization → Gemma 4 E4B
                                                      ├── MCTS form (JSON)
                                                      ├── Danger sign flags + utterance evidence
                                                      └── Referral decision (immediate / 24h / routine)
```

### Why two models, not one?

Gemma 4's audio capability is multimodal *understanding*, not transcription. It interprets and responds to audio content. Whisper is a discriminative ASR model trained specifically for speech-to-text — delivering 5% WER on Hindi medical vocabulary and handling Hindi-English code-switching that ASHA conversations naturally contain. Conflating transcription and clinical reasoning into one model sacrifices accuracy on both tasks. The two-stage design is intentional: Whisper handles acoustics, Gemma 4 handles clinical reasoning and structured extraction.

## Architecture

| Component | Model | Size | Role |
|-----------|-------|------|------|
| ASR | collabora/whisper-large-v2-hindi | ~1.5 GB | Hindi speech → text via faster-whisper/CTranslate2 |
| Normalization | src/hindi_normalize.py | — | Hindi number words → digits, medical term mapping |
| Clinical Extraction | Gemma 4 E4B (Q4_K_M via Ollama) | ~5 GB | Transcript → structured JSON form + danger signs |

**Hindi number normalization:** Algorithmic parser covering all 0–999 Hindi number words with Whisper misspelling variants. Handles compound medical values: "एक सौ दस बटा सत्तर" → "110/70", "ग्यारह दशमलव पाँच" → "11.5", "तीन किलो दो सौ ग्राम" → "3.2 kg".

**Anti-hallucination pipeline (6 layers):**
1. Evidence length filter — danger signs with <10 char evidence dropped
2. Generic ASHA phrase blocklist — "कोई तकलीफ़ हो तो फ़ोन कर दीजिए" etc. filtered
3. Normal value filter — strips signs citing "110/70", "बिल्कुल ठीक", "सामान्य"
4. Transcript grounding — evidence must appear verbatim in the transcript
5. Deduplication across overlapping danger signs
6. Form validation — strips invented names (दीदी/बहन patterns), default ages, phantom lab results; range checks on BP (60–250/30–150), Hb (3–20), weight (1–200), gestational weeks (1–45)

## Deployment Model

```
Health Center (laptop, RTX GPU)         Field (any smartphone)
┌──────────────────────────────┐       ┌──────────────────────────┐
│  FastAPI server              │ WiFi  │  PWA (service worker     │
│  Whisper ASR (CTranslate2)   │◄─────►│    caches app offline)   │
│  Gemma 4 E4B (Ollama)        │ LAN   │  Field Mode: mic record  │
│  Static frontend serving     │       │  IndexedDB audio queue   │
└──────────────────────────────┘       └──────────────────────────┘
```

1. ASHA worker opens Sakhi on her phone over health center WiFi — first load caches the app via service worker
2. Goes to field — app works fully offline, records home visit conversations in Field Mode
3. Returns to health center — queued recordings sync over LAN and process through the pipeline
4. Structured forms + danger sign alerts ready for the ANM/medical officer

## Form Types

5 JSON schemas covering NHM/IMNCI protocol:

- **ANC (Antenatal Care)** — pregnancy registration, vitals, TT/IFA, lab results, birth preparedness
- **Delivery** — birth outcome, type (normal/C-section), infant details, complications, blood loss
- **PNC / HBNC** — postnatal mother + newborn assessment (days 1–42), lactation, cord care
- **Child Health / HBYC** — growth monitoring, immunization, developmental milestones, illness screening
- **Danger Signs** — 10 maternal + 9 newborn danger sign checklist with mandatory utterance evidence, referral decision

## Test Results

**Text extraction quality:** 14/15 tests pass (test_ollama_quality.py)
- 4/4 visit types: ANC, PNC, delivery, child health
- Zero false danger alarms on normal visits
- Correct referral escalation on danger cases

**End-to-end audio pipeline:** 13/15 tests pass (87%) — test_pipeline_e2e.py
- 15 synthetic Hindi audio samples through full pipeline
- 2 failures are TTS→ASR artifacts on BP values (synthetic audio, not real-world)
- All visit types pass, all danger sign tests pass, all edge cases pass
- Avg timing: ASR 3.4s | Form 15.5s | Danger 9.6s | **Total 28.6s**

**Hindi normalization:** 133 tests pass (test_asr.py)
- Covers 0–999 Hindi number words + Whisper misspelling variants
- Compound values (BP, weight, Hb), decimal points, fractions

**Training data:** 1,154 examples (981 train / 173 val) across 4 visit types, 458 positive danger sign cases

## Frontend

React + Vite PWA with five tabs:

| Tab | Purpose |
|-----|---------|
| Voice to Form | Record or upload audio, real-time SSE pipeline progress |
| Text to Form | Paste transcript, extract structured form with example loader |
| Field Mode | Offline recording queue — record, playback, sync when connected |
| About & Impact | Project context, ASHA program statistics |
| History | Past extractions with JSON/CSV export |

## Quick Start

```bash
# Prerequisites: Python 3.11+, Node 18+, Ollama, CUDA GPU (16GB VRAM recommended)

# Backend
pip install -r requirements.txt
ollama pull gemma4:e4b
python api.py                    # FastAPI on :8000

# Frontend
cd frontend && npm install
npm run dev                      # Vite dev server on :5173

# Tests
python scripts/test_ollama_quality.py    # Text extraction (14/15)
python scripts/test_pipeline_e2e.py      # Full E2E audio (13/15)
python scripts/test_asr.py               # Hindi normalization (133/133)
```

## Project Structure

```
api.py                              # FastAPI backend — 6 endpoints, SSE streaming
app.py                              # Core pipeline — ASR, extraction, danger signs, validation
src/hindi_normalize.py              # Hindi number/medical term normalization (160 number words)
configs/schemas/                    # 5 JSON schemas (ANC, PNC, delivery, child health, danger signs)
frontend/
  src/App.jsx                       # React app — all 5 tabs
  src/offlineQueue.js               # IndexedDB offline audio queue
  public/sw.js                      # Service worker for PWA offline caching
  public/manifest.json              # PWA manifest
scripts/
  test_ollama_quality.py            # Text extraction quality tests (14/15)
  test_pipeline_e2e.py              # End-to-end audio pipeline tests (13/15)
  test_asr.py                       # ASR + Hindi normalization tests (133/133)
  generate_training_data.py         # Synthetic ASHA conversation generation
  train_unsloth.py                  # LoRA fine-tuning via Unsloth
  export_ollama.py                  # GGUF → Ollama model export
data/
  processed/train.jsonl             # 981 training examples
  processed/val.jsonl               # 173 validation examples
```
