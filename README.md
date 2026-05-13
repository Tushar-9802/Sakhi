# Sakhi (सखी) — Voice-to-Form for ASHA Workers

Offline-first tool that converts Hindi home visit conversations into structured government health forms and real-time referral decisions for India's 1 million+ ASHA health workers.

**Competition:** [Gemma 4 Good Hackathon](https://www.kaggle.com/competitions/gemma-4-good-hackathon) ($200K prize pool)
**Tracks:** Health & Sciences | Ollama | Unsloth | Cactus (Android APK)
**Partner frameworks:** [Gemma 4](https://blog.google/technology/developers/gemma-3/) (E2B + E4B), [Cactus SDK](https://github.com/cactus-compute/cactus) (on-device Android), [Ollama](https://ollama.ai) (workstation GPU), [Unsloth](https://unsloth.ai) (LoRA fine-tune), [Whisper](https://github.com/openai/whisper) (Hindi ASR via CTranslate2)

## Problem

India's ASHA workers conduct 50M+ maternal/child health home visits per year across rural areas. Every visit ends with paper forms filled from memory, then physically carried to the Primary Health Center. Danger signs observed in the field — preeclampsia, postpartum hemorrhage, neonatal distress — often never reach the system in time for intervention.

## Solution

One product, one extraction schema, one anti-hallucination pipeline — deployed two ways to match ASHA working reality:

- **Health-center mode (workstation + E4B via Ollama)** — sub-center / PHC / camp with a shared workstation. Phone records Hindi audio → LAN upload → Whisper ASR + Gemma 4 E4B on GPU with native function calling → structured JSON back to phone. Fast (~15 s) and accurate. This is the primary voice-to-form path.
- **Field mode (phone)** has two offline sub-paths:
  - **Record now, sync later** — ASHA records audio during home visits; chunks persist to IndexedDB every 5 s (crash-safe). When the phone is back on health-center WiFi, the queued recordings post to the workstation for full Whisper + E4B processing. This is the honest voice path — no on-device ASR attempted.
  - **Type a note for instant on-device extraction** — for when the ASHA wants structured output *right now* without network. A short Hindi note in a textarea runs through the full pipeline (normalize → detect visit type → extract form → detect danger signs) entirely on-device via Gemma 4 E2B INT4 on the Cactus SDK. Same schema, same validation as the workstation path. Pipeline latency is ≈ 5 min on a Snapdragon 8+ Gen 1 phone. This is acceptable against the clinical baseline: the status quo is an ASHA hand-filling the same form from memory (15–20 min), carrying it to the PHC (another walk), then waiting for a clinician to read and act on it (hours to days). A 5-minute wait for on-device structured extraction + flagged danger signs is a net time save, not a UX compromise — and it works with zero network, zero shared infrastructure.

```
Workstation path:
[Hindi Audio] → Whisper ASR → Hindi Normalization → Gemma 4 E4B (function calling)
                                                      ├── extract_form()      → structured MCTS JSON
                                                      ├── flag_danger_sign()  → per-sign with utterance evidence
                                                      └── issue_referral()    → urgency + facility + reasoning

On-device path (text-in):
[Hindi Text] → Hindi Normalization → Visit-type detect → Gemma 4 E2B (plain JSON)
                                                          ├── extract_form     → null-filled template filled in
                                                          └── detect_danger    → danger_signs + referral_decision
```

### Why not voice-to-form on-device too?

We looked into it — the honest answer is it doesn't work well enough yet for clinical Hindi. Cactus's transcribe API supports Whisper / Moonshine / Parakeet only (Gemma 4's audio conformer is for voice understanding in multimodal chat, not dedicated ASR). Cactus ships multilingual Whisper INT4 weights, but no Hindi-specific checkpoint — and published evidence (arXiv 2512.10967, Vistaar/Gramvaani) shows off-the-shelf Whisper on spontaneous rural Hindi hits 27% WER at best and 70%+ on clinical content, with a deletion-dominant error profile that silently drops numbers and symptoms. For an ASHA decision-support tool where a missed BP reading is a clinical harm, we chose to *not* ship an unreliable on-device voice path. Record-and-sync with Whisper-Large on the workstation keeps voice-in honest; the on-device LLM does what Gemma 4 is actually good at — Hindi text understanding.

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

| Component | Model | Size | Role | Deployment |
|-----------|-------|------|------|------------|
| ASR (workstation path only) | collabora/whisper-large-v2-hindi | ~1.5 GB | Hindi speech → text via faster-whisper/CTranslate2 | Workstation |
| Normalization | src/hindi_normalize.py | — | Hindi number words → digits, medical term mapping | Shared (Python server-side; JS port for phone) |
| Clinical Extraction (health-center mode, audio-in) | Gemma 4 E4B (Q4_K_M via Ollama) | ~5 GB | Function calling: form extraction + danger signs + referral | Workstation (GPU) |
| Clinical Extraction (field mode, text-in) | Gemma 4 E2B (INT4 via Cactus SDK) | ~4.4 GB download / ~6.3 GB on-device extracted (multimodal package includes audio + vision encoders that the text-in path does not use) | Same extraction schema, plain-JSON mode (E2B INT4 does not reliably emit OpenAI-style `tool_calls`) | Android (ARM, Snapdragon 7+ Gen 1 or newer, 8 GB RAM, ~7 GB free storage for the one-time install) |

**Patient demographics enter as a header, not from the audio.** Every clinical EMR works this way: identifiers typed once at intake, the conversation handled separately. The ASHA fills name / age / sex / mobile / ASHA-ID / visit-date in the header above the record button, and the LLM only extracts what was *said* during the visit — symptoms, vitals, counselling, next-visit date. This avoids a failure mode we hit in real-voice testing: Whisper-Hindi sometimes mishears patient names as different Hindi words, and a downstream LLM has no prior on what the name should be. Same merge logic runs on all three paths — `apply_metadata` in `app.py` for workstation audio and text, mirrored as a pure JS function in `pipeline.js` for on-device Cactus extraction — so server and phone produce identical envelopes for the same input. ANC fills `patient.{name, age, mobile}`; child_health fills `child.{name, age_months, sex}` with year→month conversion; PNC and delivery have no patient sub-object in their form, so the metadata travels in the response envelope only. `asha_id` is sticky across sessions via `localStorage`. For Field-mode recordings, the header is captured at record-start so later edits don't pollute earlier queue entries.

**Hindi number normalization:** Algorithmic parser covering all 0–999 Hindi number words with Whisper misspelling variants. Handles compound medical values: "एक सौ दस बटा सत्तर" → "110/70", "ग्यारह दशमलव पाँच" → "11.5", "तीन किलो दो सौ ग्राम" → "3.2 kg".

**Anti-hallucination pipeline (6 layers):**
1. Evidence length filter — danger signs with <10 char evidence dropped
2. Generic ASHA phrase blocklist — "कोई तकलीफ़ हो तो फ़ोन कर दीजिए" etc. filtered
3. Normal value filter — strips signs citing "110/70", "बिल्कुल ठीक", "सामान्य"
4. Transcript grounding — evidence must appear verbatim in the transcript
5. Deduplication across overlapping danger signs
6. Form validation — strips invented names (दीदी/बहन patterns), default ages, phantom lab results; range checks on BP (60–250/30–150), Hb (3–20), weight (1–200), gestational weeks (1–45)

## Reproducing the demo

Two reproduction paths, calibrated to how much friction the reviewer wants to accept.

**Path 1 — workstation, ~5 minutes (recommended for reviewers).** Runs the full pipeline (Whisper + Gemma 4 E4B via Ollama) on any CUDA workstation with ≥16 GB VRAM. No phone needed; same extraction code, same anti-hallucination validation, same form output. `pip install -r requirements.txt && ollama pull gemma4:e4b && python api.py` then open `http://localhost:8000`. Voice-to-form, text-to-form, and queue-and-sync flows all run here. This is sufficient to verify our engineering claims (function calling, normalization, 6-layer validation, schema correctness).

**Path 2 — on-device on Android, ~20-25 minutes total (for verifying the Cactus track).** Requires accepting the Cactus-Compute model license. Steps:
1. Accept terms at [huggingface.co/Cactus-Compute/gemma-4-E2B-it](https://huggingface.co/Cactus-Compute/gemma-4-E2B-it) (1 min, free HF account).
2. Download `gemma-4-e2b-it-int4.zip` (~4.4 GB) from that page.
3. Build + install the APK (`./gradlew assembleDebug && adb install -r ...`), or take the prebuilt APK from the GitHub Release.
4. Transfer the zip to the phone's `Downloads/` folder via USB MTP or USB-OTG drive. (WhatsApp won't work — 2 GB cap. Drive download to phone is fine if the file lands locally rather than streaming.)
5. Open Sakhi → Field Mode → On-Device Probe → **Import model (.zip)** → pick the zip from the system file picker. Wait ~3-5 minutes for extraction (progress bar + log card show live file count and MB written). Re-imports auto-evict the previous model — no manual cleanup, no risk of 12 GB accumulation.
6. **Load Model** → **Test Hindi** to confirm inference works.

**We do not redistribute the Cactus model.** It is gated under a custom Cactus-Compute license; hosting it on a public Drive link would violate that gating. The in-app SAF import flow exists precisely so reviewers who DO want to reproduce on-device can do so without us needing to host the weights ourselves and without needing developer mode or adb on their phone. The 3-minute demo video in the submission shows the full flow on a real phone, so the on-device claim can be verified without anyone needing to install the model themselves.

## Safety & Limitations

Sakhi is a decision-support tool, not a diagnostic system. All outputs require human review.

**What it catches:** Danger signs with explicit conversational evidence — elevated BP with symptoms, severe bleeding, neonatal distress indicators. The model only flags what was said in the conversation, grounded by verbatim utterance quotes.

**What it can miss:** Danger signs not discussed in conversation, subtle clinical findings that require physical examination, conditions that present atypically. The system cannot observe — it can only reason about what was spoken.

**False positive controls:** The 6-layer anti-hallucination pipeline aggressively filters ungrounded danger signs. On the test suite, normal visits produce zero false alarms.

**Human-in-the-loop:** Every referral decision is presented to the ANM/medical officer at the health center for review before action. The tool accelerates information flow from field to facility — it does not replace clinical judgment.

**Known gaps:** All current test data is synthetic (TTS-generated Hindi audio, LLM-generated training conversations). Real-world ASHA conversations will be noisier, more fragmented, and contain regional dialect variation not yet tested.

## Deployment Model

```
Health Center (workstation, RTX GPU)              Field (Android phone)
┌────────────────────────────────────┐       ┌──────────────────────────────────┐
│  python api.py  →  :8000           │◄─────►│  Native APK (Capacitor + React)  │
│  ├── /api/*   — pipeline endpoints │  WiFi │  ├── Health-center mode:         │
│  └── /        — React UI (dist/)   │  LAN  │  │   POST audio to workstation :8000  │
│                                    │       │  └── Field mode (offline):       │
│  Whisper ASR (CTranslate2)         │       │      (a) record + IDB queue +    │
│  Gemma 4 E4B (Ollama)              │       │          later sync to :8000     │
│                                    │       │      (b) type Hindi note →       │
│  Desktop browser UI:               │       │          Cactus + Gemma 4 E2B    │
│  http://localhost:8000             │       │          on-device text→form     │
└────────────────────────────────────┘       └──────────────────────────────────┘
```

**Three access points, same backend schema:**

1. **Workstation browser** — ANM/medical officer at the health center opens `http://localhost:8000` (or `http://<LAN-IP>:8000` from any workstation on the WiFi). FastAPI serves the built React UI at `/` and the pipeline endpoints at `/api/*`. One command (`python api.py`) starts everything.
2. **Phone, health-center mode** — APK records and posts to workstation's `:8000` over WiFi. Workstation does Whisper + E4B (fast, accurate). Best extraction quality available.
3. **Phone, field mode** — APK offers two offline paths. **(a)** Record audio during home visits — chunks stored crash-safely in IndexedDB every 5 s. Queued recordings sync to the health-center workstation when back on WiFi for full Whisper + E4B processing. **(b)** Type a short Hindi note in the "on-device text → form" card; the full extraction + danger-sign pipeline runs on the phone via Gemma 4 E2B on Cactus SDK. No network required. Total on-device pipeline latency ≈ 5 min on Snapdragon 8+ Gen 1 — suited for "tap and wait" use, not real-time.

**Crash-safe recording (Field Mode):** audio chunks are persisted to IndexedDB every 5 seconds during a recording. If the browser tab closes, the phone locks, or the app is killed mid-visit, the chunks survive — on reopen, an orange recovery banner offers to reassemble the partial recording.

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
- 2 failures are TTS→ASR artifacts on BP values (synthetic audio, not real-world). Root-cause walkthrough in [FAILURES.md](FAILURES.md).
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

One React + Vite codebase, shipped as both a browser UI (served by FastAPI at `/`) and a native Android APK (Capacitor-wrapped, same React bundle inside a WebView + native plugins):

| Tab | Purpose |
|-----|---------|
| Voice to Form | Record or upload audio, real-time SSE pipeline progress (workstation path). Patient & Visit Info header at the top (name / age / sex / ASHA-ID / visit-date) is posted alongside the audio so demographics don't depend on ASR. |
| Text to Form | Paste transcript, extract structured form with example loader (workstation path) |
| Field Mode | Offline-first: crash-safe audio recording queue (IndexedDB every 5 s) for later sync + **on-device text→form card** that runs the full pipeline through Gemma 4 E2B on Cactus SDK + **On-Device Probe** card for loading/health-checking the Cactus model. Same Patient & Visit Info header as the Voice tab; header values are snapshotted at record-start so later edits don't contaminate earlier queue entries. A "Developer view" toggle shows raw per-stage model output for verification. |
| About & Impact | Project context, ASHA program statistics |
| History | Past extractions with JSON/CSV export |

**JS pipeline port** (`frontend/src/lib/`) — the Python extraction pipeline (Hindi normalization, visit-type detection, form/danger prompts, 6-layer validation, demographics-header merge) has a full JS port so the phone can run the same logic against the on-device Cactus engine, engine-agnostic by design. 72/72 unit tests pass under `node --test`.

**On-device prompt design note:** E4B via Ollama handles a raw JSON Schema in the form-extraction prompt cleanly. E2B INT4 on Cactus doesn't — it echoes schema metadata (`$schema`, `title`, `description`, `type`) back as output data. The JS port sends a **null-filled instance template** instead (just the field shape with all values as null), and the model's job is to fill in the slots where the transcript says something. Similarly, danger-sign extraction on-device uses plain JSON (E2B doesn't reliably emit OpenAI-style `tool_calls` in Cactus's parseable shape). The workstation E4B path keeps native function calling.

## Quick Start

```bash
# Prerequisites: Python 3.11+, Node 18+, Ollama, CUDA GPU (16GB VRAM recommended)

# ── Health-center deployment (workstation, unified UI + API) ──
pip install -r requirements.txt
ollama pull gemma4:e4b
cd frontend && npm install && npm run build && cd ..
python api.py
# Browser: http://localhost:8000  (React UI)
# Phone APK (on same WiFi): posts to http://<workstation-LAN-IP>:8000

# ── Frontend dev mode (hot-reload) ──
cd frontend && npm run dev           # Vite on :5173, proxies /api to :8000

# ── Android APK (Capacitor, field-deployable) ──
# Prerequisites: JDK 21 (Temurin), Android Studio with SDK
cd frontend
VITE_API_BASE_URL="http://<workstation-LAN-IP>:8000" npm run build
npx cap sync android
cd android && ./gradlew assembleDebug
# APK at: frontend/android/app/build/outputs/apk/debug/app-debug.apk

# ── On-device Cactus model (for field mode) ──
# Two install paths. Pick one.
#
# (A) PRIMARY — judges / non-developers — no adb required:
#   1. Accept the Cactus-Compute terms at huggingface.co/Cactus-Compute/gemma-4-E2B-it
#   2. Download gemma-4-e2b-it-int4.zip (~4.4 GB) to a PC, then transfer to
#      the phone's Downloads folder via USB cable (MTP) or USB-OTG drive.
#      WhatsApp won't work (2 GB cap). Drive download to the phone also works
#      but Drive's content provider streams lazily, so prefer a downloaded copy.
#   3. Open Sakhi → Field Mode → On-Device Probe → Import model (.zip)
#      → pick the zip from the system file picker.
#   4. Wait ~3-5 min for extraction. Progress bar + log card show live
#      file count and MB written.
#   5. Tap Load Model → Test Hindi to confirm.
#   Re-imports automatically wipe the previous model dir — no manual cleanup,
#   no risk of accumulating multiple 6 GB models on the phone.
#
# (B) DEVELOPER — adb-based, scripted, faster on the same WiFi:
export HF_TOKEN=hf_...            # read token, repo must be accepted on HF UI
bash scripts/setup_cactus_model.sh
# Requires: adb on PATH, phone in USB debug mode authorised for this host,
# debuggable Sakhi APK installed (run-as-able). Full prerequisites +
# troubleshooting documented inside the script header.

# Tests
python scripts/test_ollama_quality.py    # Text extraction (base 15/15, sakhi 14/15)
python scripts/test_pipeline_e2e.py      # Full E2E audio (13/15)
python scripts/test_asr.py               # Hindi normalization (133/133)
cd frontend && npm test                  # JS pipeline port (72/72)

# Retrain + A/B eval (requires RTX GPU, cmake, llama.cpp binaries)
python scripts/train_unsloth.py                 # Full pipeline: prep, train, export, register, eval
python scripts/train_unsloth.py --export-only   # Skip training, just export saved adapter
python scripts/compare_field_coverage.py        # Field-level diff base vs sakhi
```

## Project Structure

```
api.py                              # FastAPI backend — SSE streaming + static mount of frontend/dist
app.py                              # Core pipeline — function calling, ASR, extraction, validation
src/hindi_normalize.py              # Hindi number/medical term normalization (160 number words)
configs/schemas/                    # 5 JSON schemas (ANC, PNC, delivery, child health, danger signs)
frontend/
  src/App.jsx                       # React app — all 5 tabs, on-device text-in card + Cactus probe in Field Mode
  src/offlineQueue.js               # IndexedDB offline queue + crash-safe chunk persistence
  src/lib/                          # JS port of Python pipeline (engine-agnostic)
    hindiNormalize.js               # Full port of src/hindi_normalize.py
    visitTypeDetect.js              # Visit-type keyword heuristic
    validation.js                   # 6-layer anti-hallucination
    prompts.js                      # FORM + DANGER prompts (template-based for on-device E2B)
    pipeline.js                     # Orchestrator (engine.complete({messages, options}) contract)
    cactus.js                       # Capacitor facade for Cactus SDK
    __tests__/                      # 62/62 assertions pass under node --test
  public/sw.js                      # Service worker for PWA offline caching (browser install)
  public/manifest.json              # PWA manifest
  capacitor.config.json             # Capacitor config (appId com.sakhi.app, http scheme for LAN)
  android/                          # Native Android project — Capacitor-generated, produces APK
    app/src/main/java/com/cactus/Cactus.kt             # Cactus SDK Kotlin wrapper (vendored from cactus-src; upstream publishes no Maven artifact)
    app/src/main/java/com/sakhi/app/CactusPlugin.kt    # Capacitor plugin bridging JS ↔ Cactus
    app/src/main/jniLibs/arm64-v8a/libcactus.so        # Cactus native library (66 MB, arm64-v8a). Committed to repo via .gitignore negation because the Cactus project publishes no prebuilt Android .so and no Maven artifact. Build provenance: compiled from github.com/cactus-compute/cactus via its upstream android/build.sh with NDK r27b + CMake 3.22.1 + Ninja on Windows Git Bash. To rebuild: clone cactus, set ANDROID_NDK_HOME + CMAKE_GENERATOR=Ninja, run `bash android/build.sh`. Output .so replaces this file.
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
  role_play_scripts.md              # Hindi role-play scripts for real-voice validation (4 scenarios)
models/
  checkpoints/final/                # Saved LoRA adapter (85MB)
  exported/sakhi-v2-q4_k_m.gguf     # Quantized fine-tune (5.3GB, registered in Ollama)
  cactus/gemma-4-e2b/               # INT4 on-device model for Cactus (not committed; HF-gated download)
RETRAIN_RESULTS.md                  # A/B score summary
FIELD_COVERAGE_DIFF.md              # Field-level coverage diff
```
