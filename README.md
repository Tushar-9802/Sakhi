# Sakhi (सखी) — Voice-to-Form for ASHA Workers

Offline tool that converts Hindi home visit conversations into filled government health forms and referral decisions for India's 1 million+ ASHA workers.

**Competition:** [Gemma 4 Good Hackathon](https://www.kaggle.com/competitions/gemma-4-good-hackathon) ($200K prize pool)

## Problem

ASHA workers conduct maternal/child health home visits across rural India. Every visit ends with paper forms filled by memory, then physically carried to the PHC. Danger signs observed in the field often never reach the system in time.

## Solution

Phone records the conversation → Whisper transcribes Hindi audio → Gemma 4 extracts structured form data + detects danger signs → Referral decision issued immediately.

No internet. No cloud. No physician required.

```
[Hindi Audio] → Whisper ASR → Hindi Normalization → Gemma 4 E4B
                                                      ├── MCTS form (JSON)
                                                      ├── Danger sign flags + evidence
                                                      └── Referral decision
```

## Architecture

Two-model pipeline optimized for offline deployment:

| Component | Model | Size | Role |
|-----------|-------|------|------|
| ASR | collabora/whisper-large-v2-hindi | ~1.5 GB | Hindi speech → text (5% WER) |
| Normalization | src/hindi_normalize.py | 0 MB | Hindi numbers → digits, medical terms → English |
| Extraction | Gemma 4 E4B (LoRA fine-tuned) | ~5 GB | Transcript → structured JSON form + danger signs |

**Hindi number conversion:** Algorithmic parser handles all 0-999 Hindi number words + Whisper misspellings. Converts "एक सौ दस बटा सत्तर" → "110/70", "ग्यारह दशमलव पाँच" → "11.5".

**Anti-hallucination:** 6-layer validation on danger signs (evidence grounding, generic phrase blocklist, normal value filter, dedup). Form validation strips invented names/ages/lab results.

## Form Types Supported

- **ANC (Antenatal Care)** — pregnancy registration, vitals, lab results, birth preparedness
- **Delivery** — birth outcome, infant details, complications
- **PNC / HBNC** — postnatal mother + newborn assessment (days 1-42)
- **Child Health / HBYC** — growth, immunization, development, illness screening
- **Danger Signs** — NHM/IMNCI protocol with mandatory utterance evidence

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Gradio app
python app.py
# Open http://localhost:7860

# Run ASR + normalization tests (no GPU needed for offline tests)
python scripts/test_asr.py --skip-gpu
```

## Project Structure

```
app.py                          # Gradio app — ASR + extraction + UI
src/hindi_normalize.py          # Hindi number/medical term normalization
configs/schemas/                # JSON schemas for 4 visit types + danger signs
configs/training.yaml           # Unsloth LoRA training config
scripts/
  generate_training_data.py     # Synthetic ASHA conversation generation
  prepare_training.py           # Raw → chat-format training data
  augment_unlabeled.py          # Add speaker-label-stripped variants
  train_unsloth.py              # LoRA fine-tuning via Unsloth
  evaluate.py                   # Evaluation metrics
  test_asr.py                   # ASR + normalization test suite (133 tests)
  export_ollama.py              # GGUF/Ollama export
```

## Status

- Text-to-form: 4/4 visit types passing, accurate extraction, zero false danger alarms
- Audio-to-form: Hindi ASR working (collabora whisper + CTranslate2), all medical values found
- Anti-hallucination: validated on unlabeled transcripts (no speaker labels)
- Speed: targeting <30s total pipeline (ASR + extraction)
