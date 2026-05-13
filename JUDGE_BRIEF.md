# Sakhi (सखी) — Judge Brief

*One-page version of the README. Full detail in [README.md](README.md).*

## The problem, in two sentences

India's 1 million+ ASHA health workers conduct 50M+ maternal and child home visits every year; every visit ends with a hand-filled paper form carried to the PHC. Danger signs observed in the field — preeclampsia, postpartum hemorrhage, neonatal distress — often don't reach the clinical system in time for intervention.

## What Sakhi does, in two sentences

Sakhi converts Hindi home-visit conversations (voice on a shared health-center workstation, text on the ASHA's phone offline) into structured NHM/MCTS forms + a function-calling-powered danger-sign triage that flags referrals with verbatim utterance evidence. Same pipeline, same anti-hallucination validation, two deployment modes: Whisper-Large + Gemma 4 E4B via Ollama on a workstation for accuracy, and Gemma 4 E2B via Cactus SDK on an Android phone for offline resilience.

![App screenshot placeholder — populated after Bareilly field trip](docs/screenshot-placeholder.png)

## Numbers a judge can check

| Measurement | Value | Source |
|---|---|---|
| Text extraction pass rate (base Gemma 4 E4B) | **15 / 15** | `scripts/test_ollama_quality.py` |
| End-to-end audio pipeline pass rate | **13 / 15** | `scripts/test_pipeline_e2e.py` (2 TTS→ASR artifacts, documented in FAILURES.md) |
| Hindi number / medical-term normalization | **133 / 133** | `scripts/test_asr.py` |
| On-device JS pipeline port (engine-agnostic) | **72 / 72** | `cd frontend && node --test src/lib/__tests__/` |
| False-alarm rate on routine visits | **0** | Strict evidence-grounding + 6-layer validation |
| Workstation pipeline latency (audio → form) | ~15–25 s | RTX 5070 Ti, warm Ollama |
| On-device pipeline latency (Hindi text → form) | ~5 min | OnePlus 11R / Snapdragon 8+ Gen 1, Gemma 4 E2B INT4 on Cactus |

The 5-minute on-device figure is tested against the `ms2_0425` ANC preeclampsia training transcript: the model correctly extracts BP 150/95, TT complete, IFA = yes, verbatim Hindi symptoms, and flags `high_bp_with_symptoms` (urgent_care) with the Hindi quote `"आपका BP 150/95 आ रहा है"` and a "Refer Immediately" decision. A 5-minute wait is a net time save against the 15–20 min baseline of hand-filling paper forms plus travel to the PHC.

## Why this is submitted to four tracks

| Track | What Sakhi brings |
|---|---|
| **Health & Sciences** | A clinical-decision-support tool with explicit human-in-the-loop design, 6-layer anti-hallucination, strict-evidence danger-sign grounding, demographics entered as a typed header (the way every clinical EMR does it, so identifiers don't depend on ASR), and a real ASHA workflow (health-center mode + field mode with later sync) — not a research demo. |
| **Ollama** | Native function calling via `tools=` parameter for `extract_form` + `flag_danger_sign` + `issue_referral` in a single inference pass, quantized Gemma 4 E4B Q4_K_M served on LAN to any phone on the same WiFi. One command (`python api.py`) starts the full stack. |
| **Unsloth** | Honest reproducible LoRA pipeline in `scripts/train_unsloth.py`: data prep → LoRA train → GGUF export → Ollama registration → A/B eval vs base. Published artifacts: `RETRAIN_RESULTS.md`, `FIELD_COVERAGE_DIFF.md`. Fine-tune didn't beat base on pass-rate — we shipped the base and documented the fine-tune's specific wins (English schema-label normalization, visit-type-specific field recovery) rather than inflate the narrative. |
| **Cactus** | Genuine on-device integration: custom Capacitor plugin bridging JS ↔ Cactus Kotlin SDK, JS pipeline port that drives either the Cactus engine or the workstation engine through a single `engine.complete()` contract, null-filled instance template prompting pattern that sidesteps E2B INT4's schema-echo failure mode, in-app SAF zip-import so a judge can install the 4.4 GB model without adb or developer tooling (single-pass extract with 1%/heartbeat progress events; auto-evicts stale model dirs on re-import), and a Developer-view toggle that shows raw per-stage model output for verifiable extraction. We investigated on-device voice-in via `cactusTranscribe` + Gemma; documented in the README why it's not shipped (Gemma 4 doesn't serve Cactus's ASR path, and off-the-shelf Whisper-Hindi INT4 has 27–70% WER on rural/clinical Hindi per arXiv 2512.10967 — shipping it would be demo-theater with clinical harm potential). |

## Reproduce in under 10 minutes

**Health-center mode (workstation only):**
```bash
pip install -r requirements.txt && ollama pull gemma4:e4b
cd frontend && npm install && npm run build && cd ..
python api.py        # browser: http://localhost:8000
```

**Field mode (phone + Cactus):**

> **We do not redistribute the Cactus-Compute model** — it is gated under a custom Cactus license. Reviewers verifying the Cactus track follow the documented path below. Most reviewers can verify the engineering claims via the workstation path above without ever installing on-device; the 3-minute demo video shows the full on-device flow on a real phone.

```bash
# Build + install the APK once. After this the model install is in-app, no adb.
cd frontend && npm run build && npx cap sync android && \
  cd android && ./gradlew assembleDebug && \
  adb install -r app/build/outputs/apk/debug/app-debug.apk

# Model install — primary path, no developer tooling needed:
#   1. Accept terms at huggingface.co/Cactus-Compute/gemma-4-E2B-it
#   2. Download gemma-4-e2b-it-int4.zip (~4.4 GB) to the PHONE'S Downloads
#      folder (USB MTP from PC, OTG drive, or direct Drive download to local).
#   3. Open Sakhi → Field Mode → On-Device Probe → Import model (.zip)
#      → pick the zip. Progress bar fills in ~3-5 min.
#   4. Tap Load Model → Test Hindi.
#
# Re-imports auto-evict the previous model — one model on disk at a time.

# Developer alternative (adb-based, no manual file picking):
#   export HF_TOKEN=hf_... && bash scripts/setup_cactus_model.sh
```

A sample Hindi transcript ready to paste is at `data/processed/train.jsonl` (line 1 = ANC preeclampsia case) or in the main README.

## What we'd do with $10K and six more months

- Partner with an ASHA training institute (Santosh Medical College / IIT Madras Bhashini) to collect 100+ hours of *real* ASHA home-visit audio — the current evaluation is entirely on synthetic TTS audio + LLM-generated conversations.
- Fine-tune an IndicWhisper variant on that real audio for the on-device voice-in path that we deliberately did not ship in this submission.
- Harden integration with the official MCTS API so forms post directly into the NHM system instead of being exported as JSON/CSV.
- Pilot with 10–20 ASHA workers in one block (Muradnagar / Loni-adjacent) with before/after time-and-accuracy measurement.

## Contact

Tushar J — tushar.j@cognavi.com — GitHub: [Tushar-9802/Sakhi](https://github.com/Tushar-9802/Sakhi)
