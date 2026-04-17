# Sakhi (सखी) — Voice-to-Form for ASHA Workers

Offline-first tool that converts Hindi home visit conversations into structured government health forms and real-time referral decisions for India's 1 million+ ASHA health workers.

**Competition:** [Gemma 4 Good Hackathon](https://www.kaggle.com/competitions/gemma-4-good-hackathon) ($200K prize pool)
**Tracks:** Health & Sciences | Ollama | Unsloth | Cactus (Android APK)

## Problem

India's ASHA workers conduct 50M+ maternal/child health home visits per year across rural areas. Every visit ends with paper forms filled from memory, then physically carried to the Primary Health Center. Danger signs observed in the field — preeclampsia, postpartum hemorrhage, neonatal distress — often never reach the system in time for intervention.

## Solution

Phone records the conversation → Whisper transcribes Hindi audio → Hindi normalization → Gemma 4 extracts structured form data, flags danger signs with evidence, and issues referral decisions — all through a single function calling invocation.

No internet required. No cloud dependency. No physician needed at point of care.

```
[Hindi Audio] → Whisper ASR → Hindi Normalization → Gemma 4 E4B (function calling)
                                                      ├── extract_form()      → structured MCTS JSON
                                                      ├── flag_danger_sign()  → per-sign with utterance evidence
                                                      └── issue_referral()    → urgency + facility + reasoning
```

### Why two models, not one?

Gemma 4's audio capability is multimodal *understanding*, not transcription. Whisper is a discriminative ASR model trained specifically for speech-to-text — delivering 5% WER on Hindi medical vocabulary and handling Hindi-English code-switching that ASHA conversations naturally contain. The two-stage design is intentional: Whisper handles acoustics, Gemma 4 handles clinical reasoning via native function calling.

## Function Calling

The pipeline uses Gemma 4's native function calling through Ollama's `tools=` parameter. A single LLM call invokes up to three tools:

| Tool | Purpose | When called |
|------|---------|-------------|
| `extract_form` | Fill visit-specific MCTS/HMIS schema with structured data | Every conversation |
| `flag_danger_sign` | Flag one NHM-defined danger sign with verbatim utterance evidence | Only when danger signs are present |
| `issue_referral` | Referral decision with urgency, facility level, and clinical reasoning | Only when danger signs warrant referral |

On a normal visit, only `extract_form` is called. On a high-risk visit (e.g., preeclampsia), the model calls all three — `extract_form` + multiple `flag_danger_sign` calls + `issue_referral` — in a single inference pass.

The pipeline uses a hybrid design: form extraction via `format="json"` (proven precision on structured schemas) and danger sign detection via native function calling. The model *decides* whether to flag danger signs and issue referrals — tool calls surface in the API response as `tool_calls` metadata.

## Architecture

| Component | Model | Size | Role |
|-----------|-------|------|------|
| ASR | collabora/whisper-large-v2-hindi | ~1.5 GB | Hindi speech → text via faster-whisper/CTranslate2 |
| Normalization | src/hindi_normalize.py | — | Hindi number words → digits, medical term mapping |
| Clinical Extraction | Gemma 4 E4B (Q4_K_M via Ollama) | ~5 GB | Function calling: form extraction + danger signs + referral |

**Hindi number normalization:** Algorithmic parser covering all 0–999 Hindi number words with Whisper misspelling variants. Handles compound medical values: "एक सौ दस बटा सत्तर" → "110/70", "ग्यारह दशमलव पाँच" → "11.5", "तीन किलो दो सौ ग्राम" → "3.2 kg".

**Anti-hallucination pipeline (6 layers):**
1. Evidence length filter — danger signs with <10 char evidence dropped
2. Generic ASHA phrase blocklist — "कोई तकलीफ़ हो तो फ़ोन कर दीजिए" etc. filtered
3. Normal value filter — strips signs citing "110/70", "बिल्कुल ठीक", "सामान्य"
4. Transcript grounding — evidence must appear verbatim in the transcript
5. Deduplication across overlapping danger signs
6. Form validation — strips invented names (दीदी/बहन patterns), default ages, phantom lab results; range checks on BP (60–250/30–150), Hb (3–20), weight (1–200), gestational weeks (1–45)

## Safety & Limitations

Sakhi is a decision-support tool, not a diagnostic system. All outputs require human review.

**What it catches:** Danger signs with explicit conversational evidence — elevated BP with symptoms, severe bleeding, neonatal distress indicators. The model only flags what was said in the conversation, grounded by verbatim utterance quotes.

**What it can miss:** Danger signs not discussed in conversation, subtle clinical findings that require physical examination, conditions that present atypically. The system cannot observe — it can only reason about what was spoken.

**False positive controls:** The 6-layer anti-hallucination pipeline aggressively filters ungrounded danger signs. On the test suite, normal visits produce zero false alarms.

**Human-in-the-loop:** Every referral decision is presented to the ANM/medical officer at the health center for review before action. The tool accelerates information flow from field to facility — it does not replace clinical judgment.

**Known gaps:** All current test data is synthetic (TTS-generated Hindi audio, LLM-generated training conversations). Real-world ASHA conversations will be noisier, more fragmented, and contain regional dialect variation not yet tested.

## Deployment Model

```
Health Center (laptop, RTX GPU)         Field (Android phone)
┌──────────────────────────────┐       ┌──────────────────────────┐
│  FastAPI server              │ WiFi  │  Native APK (Capacitor)  │
│  Whisper ASR (CTranslate2)   │◄─────►│  Field Mode: mic record  │
│  Gemma 4 E4B (Ollama)        │ LAN   │  IndexedDB chunk store   │
│  Static frontend serving     │       │  (crash-safe, every 5s)  │
└──────────────────────────────┘       └──────────────────────────┘
```

**Two delivery formats from the same codebase:**
- **PWA** — install via browser, works on any device with a modern browser
- **Native Android APK** — Capacitor-wrapped, installable `.apk`, same React UI, same offline chunk persistence. Verified end-to-end phone-to-laptop pipeline on Android.

1. ASHA worker installs Sakhi on her phone (APK or PWA) — first load over health center WiFi caches the app
2. Goes to field — app works fully offline, records home visit conversations in Field Mode
3. **Crash-safe recording:** audio chunks are persisted to IndexedDB every 5 seconds during a recording. If the browser tab closes, phone locks, or the app is killed mid-visit, the chunks survive — on reopen, an orange recovery banner offers to reassemble the partial recording
4. Returns to health center — queued recordings sync over LAN and process through the pipeline
5. Structured forms + danger sign alerts ready for the ANM/medical officer

## Form Types

5 JSON schemas covering NHM/IMNCI protocol:

- **ANC (Antenatal Care)** — pregnancy registration, vitals, TT/IFA, lab results, birth preparedness
- **Delivery** — birth outcome, type (normal/C-section), infant details, complications, blood loss
- **PNC / HBNC** — postnatal mother + newborn assessment (days 1–42), lactation, cord care
- **Child Health / HBYC** — growth monitoring, immunization, developmental milestones, illness screening
- **Danger Signs** — 10 maternal + 9 newborn danger sign checklist with mandatory utterance evidence, referral decision

## Test Results

**Text extraction quality (base Gemma 4 E4B):** 15/15 tests pass (test_ollama_quality.py)
- 4/4 visit types: ANC, PNC, delivery, child health
- Zero false danger alarms on normal visits
- Correct referral escalation on danger cases
- Avg 18.7s per test (form + danger sign extraction)

**End-to-end audio pipeline:** 13/15 tests pass (87%) — test_pipeline_e2e.py
- 15 synthetic Hindi audio samples through full pipeline
- 2 failures are TTS→ASR artifacts on BP values (synthetic audio, not real-world)
- All visit types pass, all danger sign tests pass, all edge cases pass
- Avg pipeline timing: ~15s per conversation (RTX 5070 Ti, warm Ollama, hybrid json+FC)

**Hindi normalization:** 133 tests pass (test_asr.py)
- Covers 0–999 Hindi number words + Whisper misspelling variants
- Compound values (BP, weight, Hb), decimal points, fractions

## Fine-Tuning (Unsloth Track)

We fine-tuned Gemma 4 E4B via Unsloth LoRA on 1,154 synthetic ASHA visit examples (981 train / 173 val) covering all 4 visit types and 458 positive danger sign cases. The resulting adapter is exported as a Q4_K_M GGUF and registered in Ollama as `sakhi:latest`.

**Configuration:** LR 5e-5, 1 epoch, LoRA r=16/alpha=32, dropout 0.05 — conservative hyperparameters to avoid overfitting on a small dataset.

**A/B comparison vs base** (see `RETRAIN_RESULTS.md`, `FIELD_COVERAGE_DIFF.md`):
- **Pass rate:** base 15/15 vs fine-tune 14/15 (single fail on heavy Hinglish code-switch → over-referral, a safer failure mode)
- **Latency:** base 18.7s vs fine-tune 19.0s avg — effectively tied
- **Schema normalization:** the fine-tune consistently translates Hindi symptom phrases into English schema labels ("दस्त" → "Diarrhea", "चक्कर आ रहे हैं" → "dizziness"), making downstream filtering easier. Base retains raw Hindi.
- **Unique field extractions:** fine-tune recovered 2 visit-type-specific fields the base missed (`anc_details.facility_or_home`, `visit_info.hbyc_visit_month`); base recovered 11 fields the fine-tune left null.

**Production choice:** we kept the base model in the live pipeline for its single-test accuracy edge. The fine-tune demonstrates the reproducible training pipeline and ships as an alternative for deployments that prefer consistent English schema values over raw transcription.

**Export pipeline (Windows):** the training script (`scripts/train_unsloth.py`) handles the full flow — data prep, LoRA training, auto-eval. For GGUF export we use a manual path (`scripts/export_merge.py`) that bypasses Unsloth's Windows mmap issues: load base + adapter via transformers, compute `delta_W = (B @ A) * (alpha/r)` per pair, then `llama.cpp/convert_hf_to_gguf.py` + `llama-quantize Q4_K_M`.

## Frontend

React + Vite PWA with five tabs:

| Tab | Purpose |
|-----|---------|
| Voice to Form | Record or upload audio, real-time SSE pipeline progress |
| Text to Form | Paste transcript, extract structured form with example loader |
| Field Mode | Crash-safe offline recording queue — chunks persisted every 5s, partial recordings recoverable after tab close |
| About & Impact | Project context, ASHA program statistics |
| History | Past extractions with JSON/CSV export |

## Quick Start

```bash
# Prerequisites: Python 3.11+, Node 18+, Ollama, CUDA GPU (16GB VRAM recommended)

# Backend
pip install -r requirements.txt
ollama pull gemma4:e4b
python api.py                    # FastAPI on 0.0.0.0:8000 (LAN-accessible)

# Frontend (PWA)
cd frontend && npm install
npm run dev                      # Vite dev server on :5173

# Android APK (Capacitor)
# Prerequisites: JDK 21 (Temurin), Android Studio with SDK
cd frontend
VITE_API_BASE_URL="http://<laptop-LAN-IP>:8000" npm run build
npx cap sync android
cd android && ./gradlew assembleDebug
# APK at: frontend/android/app/build/outputs/apk/debug/app-debug.apk

# Tests
python scripts/test_ollama_quality.py    # Text extraction (base 15/15, sakhi 14/15)
python scripts/test_pipeline_e2e.py      # Full E2E audio (13/15)
python scripts/test_asr.py               # Hindi normalization (133/133)

# Retrain + A/B eval (requires RTX GPU, cmake, llama.cpp binaries)
python scripts/train_unsloth.py                 # Full pipeline: prep, train, export, register, eval
python scripts/train_unsloth.py --export-only   # Skip training, just export saved adapter
python scripts/compare_field_coverage.py        # Field-level diff base vs sakhi
```

## Project Structure

```
api.py                              # FastAPI backend — SSE streaming, tool_calls metadata
app.py                              # Core pipeline — function calling, ASR, extraction, validation
src/hindi_normalize.py              # Hindi number/medical term normalization (160 number words)
configs/schemas/                    # 5 JSON schemas (ANC, PNC, delivery, child health, danger signs)
frontend/
  src/App.jsx                       # React app — all 5 tabs
  src/offlineQueue.js               # IndexedDB offline queue + crash-safe chunk persistence
  public/sw.js                      # Service worker for PWA offline caching
  public/manifest.json              # PWA manifest
  capacitor.config.json             # Capacitor config (appId com.sakhi.app, http scheme for LAN)
  android/                          # Native Android project — Capacitor-generated, produces APK
scripts/
  test_ollama_quality.py            # A/B quality tests (base 15/15, sakhi 14/15)
  test_pipeline_e2e.py              # End-to-end audio pipeline tests (13/15)
  test_asr.py                       # ASR + Hindi normalization tests (133/133)
  test_function_calling.py          # Gemma 4 function calling validation
  generate_training_data.py         # Synthetic ASHA conversation generation
  prepare_training.py               # Train/val split, schema cleanup, prompt matching
  train_unsloth.py                  # Full pipeline: prep, LoRA train, export, register, eval
  export_merge.py                   # Manual LoRA merge (bypasses Unsloth Windows mmap bug)
  compare_field_coverage.py         # Field-level diff base vs sakhi
data/
  processed/train.jsonl             # 981 training examples
  processed/val.jsonl               # 173 validation examples
models/
  checkpoints/final/                # Saved LoRA adapter (85MB)
  exported/sakhi-v2-q4_k_m.gguf     # Quantized fine-tune (5.3GB, registered in Ollama)
RETRAIN_RESULTS.md                  # A/B score summary
FIELD_COVERAGE_DIFF.md              # Field-level coverage diff
```
