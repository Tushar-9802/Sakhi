# Retrain Results

**Date:** 2026-04-17 09:35
**Training config:** LR=5e-05, epochs=1, LoRA r=16, alpha=32, dropout=0.05
**Training data:** 981 examples (schema leakage fixed, trimmed danger schema)

## Scores

| Model | Score |
|-------|-------|
| gemma4:e4b-it-q4_K_M (base) | 15/15 |
| sakhi:latest (fine-tuned) | 14/15 |

## Verdict

**BASE MODEL WINS — keep using gemma4:e4b-it-q4_K_M**

Fine-tuning did not improve quality. Skip Unsloth track.

## Base Model Details

```

```

## Fine-Tuned Model Details

```

```

## Diagnostics

- No clear pattern in failures. The base model may simply be better at zero-shot extraction than a LoRA fine-tune on 981 examples can achieve.

## What was fixed in this retrain (vs previous 9/15 attempt)

1. **Schema leakage removed** — 454/981 training examples had `$schema`, `title`, `description` in assistant output. Stripped.
2. **Trimmed danger schema** — training now uses the same trimmed schema as production (no checklists).
3. **System prompts match production** — exact same prompts in training and inference.
4. **LR reduced** — 2e-4 -> 5e-5 (4x lower to prevent overfitting).
5. **Epochs reduced** — 3 -> 1 (less overfitting on small dataset).
6. **LoRA alpha doubled** — 16 -> 32 (alpha=2*r is standard practice).
7. **Dropout added** — 0.0 -> 0.05 (regularization).

## If results are still bad, next steps to try

- Further lower LR to 2e-5
- Use only form_extraction examples (skip danger sign training, let base model handle it)
- Increase training data to 2000+ examples with better diversity
- Try r=8 instead of r=16 (smaller adapter, less capacity to overfit)
