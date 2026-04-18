"""
Sakhi — Unsloth LoRA Fine-Tuning + Auto-Evaluation

Full pipeline:
  1. Prepare training data (clean schema leakage, match production prompts)
  2. LoRA fine-tune Gemma 4 E4B via Unsloth
  3. Export to GGUF
  4. Register in Ollama as "sakhi"
  5. Run quality test suite (15 tests) against both base and fine-tuned
  6. Print A/B comparison

Usage:
  python scripts/train_unsloth.py                    # Full pipeline
  python scripts/train_unsloth.py --dry-run          # 10 steps, skip eval
  python scripts/train_unsloth.py --eval-only        # Skip training, just evaluate
  python scripts/train_unsloth.py --config configs/training.yaml
"""
import argparse
import json
import os
import subprocess
import sys
import time

# Disable torch.compile — required for PyTorch nightly + Gemma 4 on SM 12.0
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

import yaml


def step_prepare_data():
    """Step 0: Re-prepare training data with cleaned pipeline."""
    print("\n" + "=" * 60)
    print("STEP 0: Preparing training data (cleaned)")
    print("=" * 60)
    result = subprocess.run(
        [sys.executable, "scripts/prepare_training.py"],
        capture_output=False, timeout=120,
    )
    if result.returncode != 0:
        print("ABORT: Data preparation failed")
        sys.exit(1)

    # Verify no schema leakage
    leaked = 0
    with open("data/processed/train.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            content = json.loads(line)["messages"][2]["content"]
            if '"$schema"' in content:
                leaked += 1
    if leaked:
        print(f"ABORT: {leaked} examples still have schema leakage after cleaning!")
        sys.exit(1)
    print("Data verification: CLEAN (no schema leakage)")


def step_train(config_path: str, dry_run: bool):
    """Step 1: LoRA fine-tuning via Unsloth."""
    import torch
    torch._dynamo.config.suppress_errors = True

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_name = config["model"]["base_model"]
    max_seq_length = config["model"]["max_seq_length"]
    load_in_4bit = config["model"]["load_in_4bit"]
    lora_config = config["lora"]
    train_config = config["training"]

    print("\n" + "=" * 60)
    print("STEP 1: LoRA Fine-Tuning")
    print(f"Model: {model_name}")
    print(f"LR: {train_config['learning_rate']}, Epochs: {train_config['num_train_epochs']}")
    print(f"LoRA r={lora_config['r']}, alpha={lora_config['lora_alpha']}, dropout={lora_config['lora_dropout']}")
    print("=" * 60)

    # Gate: training data exists
    train_file = config["data"]["train_file"]
    val_file = config["data"]["validation_file"]
    for f_path in [train_file, val_file]:
        if not os.path.exists(f_path):
            print(f"ABORT: {f_path} not found")
            sys.exit(1)

    with open(train_file, "r", encoding="utf-8") as f:
        train_count = sum(1 for _ in f)
    with open(val_file, "r", encoding="utf-8") as f:
        val_count = sum(1 for _ in f)
    print(f"Training: {train_count} examples, Validation: {val_count} examples")

    # Load model
    from unsloth import FastLanguageModel

    print(f"\nLoading {model_name}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        dtype=None,
    )

    # Apply LoRA
    print("Applying LoRA adapter...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_config["r"],
        lora_alpha=lora_config["lora_alpha"],
        lora_dropout=lora_config["lora_dropout"],
        target_modules=lora_config["target_modules"],
        bias=lora_config["bias"],
        use_gradient_checkpointing="unsloth",
        random_state=config["experiment"]["seed"],
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    # Load dataset
    from datasets import Dataset

    def load_jsonl(path):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data

    train_data = load_jsonl(train_file)
    val_data = load_jsonl(val_file)

    def format_for_sft(examples):
        formatted = []
        for ex in examples:
            text = tokenizer.apply_chat_template(
                ex["messages"], tokenize=False, add_generation_prompt=False,
            )
            formatted.append({"text": text})
        return formatted

    train_dataset = Dataset.from_list(format_for_sft(train_data))
    val_dataset = Dataset.from_list(format_for_sft(val_data))
    print(f"Formatted {len(train_dataset)} train / {len(val_dataset)} val")

    # Training
    from trl import SFTTrainer
    from transformers import TrainingArguments

    max_steps = 10 if dry_run else -1
    num_epochs = 1 if dry_run else train_config["num_train_epochs"]

    training_args = TrainingArguments(
        output_dir=train_config["output_dir"],
        per_device_train_batch_size=train_config["per_device_train_batch_size"],
        gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
        gradient_checkpointing=train_config["gradient_checkpointing"],
        optim=train_config["optim"],
        learning_rate=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"],
        max_grad_norm=train_config["max_grad_norm"],
        num_train_epochs=num_epochs,
        max_steps=max_steps,
        warmup_ratio=train_config["warmup_ratio"],
        lr_scheduler_type=train_config["lr_scheduler_type"],
        bf16=train_config["bf16"],
        tf32=train_config["tf32"],
        logging_steps=train_config["logging_steps"],
        save_strategy=train_config["save_strategy"],
        save_steps=train_config["save_steps"],
        save_total_limit=train_config["save_total_limit"],
        eval_strategy=train_config["evaluation_strategy"],
        eval_steps=train_config["eval_steps"],
        load_best_model_at_end=train_config["load_best_model_at_end"],
        metric_for_best_model=train_config["metric_for_best_model"],
        dataloader_num_workers=train_config["dataloader_num_workers"],
        dataloader_pin_memory=train_config["dataloader_pin_memory"],
        seed=config["experiment"]["seed"],
        report_to="tensorboard",
        logging_dir="./logs/tensorboard",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        packing=True,
    )

    tag = "(DRY RUN — 10 steps)" if dry_run else ""
    print(f"\nStarting training {tag}...")
    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0

    # Evaluate
    print("\nRunning evaluation...")
    eval_results = trainer.evaluate()
    print(f"Eval loss: {eval_results.get('eval_loss', 'N/A'):.4f}")
    print(f"Training time: {train_time/60:.1f} min")

    # Save LoRA adapter
    output_dir = os.path.join(train_config["output_dir"], "final")
    print(f"\nSaving adapter to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Export to GGUF
    if not dry_run:
        export_config = config.get("export", {})
        gguf_quant = export_config.get("gguf_quantization", "q4_k_m")
        gguf_dir = export_config.get("output_dir", "./models/exported")
        os.makedirs(gguf_dir, exist_ok=True)

        print(f"\nExporting to GGUF ({gguf_quant})...")
        model.save_pretrained_gguf(gguf_dir, tokenizer, quantization_method=gguf_quant)
        print(f"GGUF exported to {gguf_dir}")

    return not dry_run  # return whether we should proceed to Ollama registration


def step_export_only(config_path: str):
    """Load saved adapter and export to GGUF (no training)."""
    import torch
    torch._dynamo.config.suppress_errors = True

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_name = config["model"]["base_model"]
    max_seq_length = config["model"]["max_seq_length"]
    load_in_4bit = config["model"]["load_in_4bit"]
    lora_config = config["lora"]
    train_config = config["training"]
    adapter_dir = os.path.join(train_config["output_dir"], "final")

    print("\n" + "=" * 60)
    print("EXPORT-ONLY: Loading adapter -> GGUF")
    print(f"Adapter: {adapter_dir}")
    print("=" * 60)

    if not os.path.exists(os.path.join(adapter_dir, "adapter_model.safetensors")):
        print(f"ABORT: No adapter at {adapter_dir}")
        sys.exit(1)

    from unsloth import FastLanguageModel

    print(f"\nLoading base {model_name}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        dtype=None,
    )

    print(f"Applying saved LoRA from {adapter_dir}...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_config["r"],
        lora_alpha=lora_config["lora_alpha"],
        lora_dropout=lora_config["lora_dropout"],
        target_modules=lora_config["target_modules"],
        bias=lora_config["bias"],
        use_gradient_checkpointing="unsloth",
        random_state=config["experiment"]["seed"],
    )
    # Load adapter weights
    from peft import PeftModel
    model.load_adapter(adapter_dir, adapter_name="default")

    export_config = config.get("export", {})
    gguf_quant = export_config.get("gguf_quantization", "q4_k_m")
    gguf_dir = export_config.get("output_dir", "./models/exported")
    os.makedirs(gguf_dir, exist_ok=True)

    print(f"\nExporting to GGUF ({gguf_quant})...")
    model.save_pretrained_gguf(gguf_dir, tokenizer, quantization_method=gguf_quant)
    print(f"GGUF exported to {gguf_dir}")


def step_register_ollama(config_path: str):
    """Step 2: Register model in Ollama."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_name = config.get("export", {}).get("ollama_model_name", "sakhi")
    gguf_dir = config.get("export", {}).get("output_dir", "./models/exported")

    print("\n" + "=" * 60)
    print(f"STEP 2: Registering '{model_name}' in Ollama")
    print("=" * 60)

    # Find GGUF file
    gguf_files = [f for f in os.listdir(gguf_dir) if f.endswith(".gguf")]
    if not gguf_files:
        print(f"ABORT: No GGUF in {gguf_dir}")
        return False

    gguf_path = os.path.join(gguf_dir, gguf_files[0])
    print(f"GGUF: {gguf_path} ({os.path.getsize(gguf_path) / 1e9:.1f} GB)")

    # Write Modelfile pointing to GGUF
    modelfile_content = f"""FROM {gguf_path}

TEMPLATE \"\"\"{{{{ if .System }}}}<start_of_turn>system
{{{{ .System }}}}<end_of_turn>
{{{{ end }}}}{{{{ if .Prompt }}}}<start_of_turn>user
{{{{ .Prompt }}}}<end_of_turn>
<start_of_turn>model
{{{{ end }}}}{{{{ .Response }}}}<end_of_turn>\"\"\"

PARAMETER temperature 0.1
PARAMETER num_ctx 4096
PARAMETER stop "<end_of_turn>"
PARAMETER stop "<eos>"
"""
    modelfile_path = "configs/Modelfile"
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)

    # Remove old model if exists
    subprocess.run(["ollama", "rm", model_name], capture_output=True, timeout=30)

    # Create new model
    result = subprocess.run(
        ["ollama", "create", model_name, "-f", modelfile_path],
        capture_output=True, text=True, timeout=300,
    )

    if result.returncode == 0:
        print(f"Model '{model_name}' registered successfully")
        return True
    else:
        print(f"Failed: {result.stderr}")
        return False


def step_evaluate(config_path: str = "configs/training.yaml"):
    """Step 3: A/B evaluation — base vs fine-tuned on 15-test quality suite.
    Saves results to RETRAIN_RESULTS.md in project root."""
    print("\n" + "=" * 60)
    print("STEP 3: A/B Evaluation (base vs fine-tuned)")
    print("=" * 60)

    # Run the quality test suite with both models
    result = subprocess.run(
        [sys.executable, "-u", "scripts/test_ollama_quality.py"],
        capture_output=True, text=True, timeout=1200,
        env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"},
    )

    full_output = result.stdout
    print(full_output)
    if result.stderr:
        print(result.stderr)

    # Parse per-model results
    lines = full_output.strip().split("\n")
    scores = {}
    model_details = {}  # model -> list of PASS/FAIL lines
    current_model = None

    for line in lines:
        stripped = line.strip()
        if "gemma4:e4b" in stripped and "=" * 5 not in stripped:
            current_model = "gemma4:e4b-it-q4_K_M"
            model_details[current_model] = []
        elif "sakhi:" in stripped and "=" * 5 not in stripped:
            current_model = "sakhi:latest"
            model_details[current_model] = []
        elif current_model and ("PASS" in stripped or "FAIL" in stripped) and "[" in stripped:
            model_details[current_model].append(stripped)
        if "%" in stripped and ("gemma4" in stripped or "sakhi" in stripped):
            parts = stripped.split()
            for i, p in enumerate(parts):
                if "gemma4" in p or "sakhi" in p:
                    scores[p] = parts[0] if i > 0 else parts[i-1]
                    break

    # Load training config for the report
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    lora = config.get("lora", {})
    train = config.get("training", {})

    # Determine winner and diagnose
    base_n, sakhi_n = 0, 0
    for model, score in scores.items():
        try:
            n = int(score.split("/")[0])
            if "gemma4" in model:
                base_n = n
            elif "sakhi" in model:
                sakhi_n = n
        except ValueError:
            pass

    if sakhi_n > base_n:
        verdict = "FINE-TUNED MODEL WINS — switch production to sakhi:latest"
        action = "Set `OLLAMA_MODEL=sakhi:latest` in env or update app.py"
    elif sakhi_n == base_n:
        verdict = "TIE — fine-tuned matches base quality. May be faster (shorter outputs)."
        action = "Check timing above. If sakhi is faster, consider switching."
    else:
        verdict = "BASE MODEL WINS — keep using gemma4:e4b-it-q4_K_M"
        action = "Fine-tuning did not improve quality. Skip Unsloth track."

    # Diagnose failures
    sakhi_failures = [l for l in model_details.get("sakhi:latest", []) if "FAIL" in l]
    base_failures = [l for l in model_details.get("gemma4:e4b-it-q4_K_M", []) if "FAIL" in l]

    # Check for common failure patterns
    diagnostics = []
    sakhi_fail_text = "\n".join(sakhi_failures)
    if "MISSING" in sakhi_fail_text:
        diagnostics.append("Model is under-extracting (MISSING fields). Possible causes: LR too low (model didn't learn enough), or training data doesn't cover these patterns well.")
    if "HALLUC" in sakhi_fail_text:
        diagnostics.append("Model is hallucinating values. Possible causes: LR too high (overfitting to training data quirks), insufficient negative examples, or training data has noisy labels.")
    if "WRONG" in sakhi_fail_text:
        diagnostics.append("Model extracts wrong values. Could be: training data has mismatched transcript-extraction pairs, or model is confusing similar fields.")
    if "FALSE_POS" in sakhi_fail_text:
        diagnostics.append("Model over-flags danger signs. Need more negative (no-danger) training examples, or lower the danger sign oversampling ratio.")
    if "FALSE_NEG" in sakhi_fail_text:
        diagnostics.append("Model under-flags danger signs. Training data may not have enough diverse danger scenarios.")
    if not diagnostics and sakhi_n < base_n:
        diagnostics.append("No clear pattern in failures. The base model may simply be better at zero-shot extraction than a LoRA fine-tune on 981 examples can achieve.")

    # Build markdown report
    report = f"""# Retrain Results

**Date:** {time.strftime('%Y-%m-%d %H:%M')}
**Training config:** LR={train.get('learning_rate')}, epochs={train.get('num_train_epochs')}, LoRA r={lora.get('r')}, alpha={lora.get('lora_alpha')}, dropout={lora.get('lora_dropout')}
**Training data:** 981 examples (schema leakage fixed, trimmed danger schema)

## Scores

| Model | Score |
|-------|-------|
| gemma4:e4b-it-q4_K_M (base) | {scores.get('gemma4:e4b-it-q4_K_M', '?')} |
| sakhi:latest (fine-tuned) | {scores.get('sakhi:latest', '?')} |

## Verdict

**{verdict}**

{action}

## Base Model Details

```
{chr(10).join(model_details.get('gemma4:e4b-it-q4_K_M', ['No results']))}
```

## Fine-Tuned Model Details

```
{chr(10).join(model_details.get('sakhi:latest', ['No results']))}
```

## Diagnostics

"""
    if diagnostics:
        for d in diagnostics:
            report += f"- {d}\n"
    else:
        report += "No issues detected.\n"

    report += f"""
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
"""

    # Write report
    report_path = "RETRAIN_RESULTS.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nResults saved to {report_path}")
    print(f"\n>>> {verdict}")

    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Sakhi — Full Retrain Pipeline")
    parser.add_argument("--config", type=str, default="configs/training.yaml")
    parser.add_argument("--dry-run", action="store_true", help="10 steps only, skip export/eval")
    parser.add_argument("--eval-only", action="store_true", help="Skip training, just evaluate")
    parser.add_argument("--export-only", action="store_true", help="Load saved adapter → export GGUF → register → evaluate")
    args = parser.parse_args()

    start = time.time()

    if args.export_only:
        step_export_only(args.config)
        success = step_register_ollama(args.config)
        if success:
            _enable_sakhi_in_quality_test()
            step_evaluate(args.config)
        else:
            print("\nSkipping evaluation — Ollama registration failed")
    elif args.eval_only:
        # Enable both models in quality test
        _enable_sakhi_in_quality_test()
        step_evaluate(args.config)
    else:
        # Full pipeline
        step_prepare_data()
        should_export = step_train(args.config, args.dry_run)

        if should_export:
            success = step_register_ollama(args.config)
            if success:
                _enable_sakhi_in_quality_test()
                step_evaluate(args.config)
            else:
                print("\nSkipping evaluation — Ollama registration failed")
        else:
            print("\nDry run complete. Skipping export and evaluation.")

    elapsed = time.time() - start
    print(f"\nTotal pipeline time: {elapsed/60:.1f} min")


def _enable_sakhi_in_quality_test():
    """Temporarily enable sakhi in the quality test for A/B comparison."""
    test_path = "scripts/test_ollama_quality.py"
    with open(test_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Uncomment sakhi if it's commented out
    if '# "sakhi:latest"' in content:
        content = content.replace(
            '# "sakhi:latest"',
            '"sakhi:latest"',
        )
        with open(test_path, "w", encoding="utf-8") as f:
            f.write(content)
        print("Enabled sakhi:latest in quality test for A/B comparison")


if __name__ == "__main__":
    main()
