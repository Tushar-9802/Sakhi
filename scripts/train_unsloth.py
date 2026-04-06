"""
MedScribe v2 — Unsloth LoRA Fine-Tuning

Fine-tunes Gemma 4 E4B via Unsloth for ASHA visit clinical extraction.
Trains on the chat-format JSONL from 04_prepare_training.py.

Uses Unsloth's FastLanguageModel for 2-5x faster training and 50-80% less memory.
Exports to GGUF for Ollama deployment.

Usage:
  python scripts/05_train_unsloth.py                    # Full training
  python scripts/05_train_unsloth.py --dry-run          # 10 steps only
  python scripts/05_train_unsloth.py --config configs/training.yaml
"""
import argparse
import json
import os
import sys
import yaml

# Disable torch.compile — required for PyTorch nightly + Gemma 4 on SM 12.0
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
import torch
torch._dynamo.config.suppress_errors = True


def main():
    parser = argparse.ArgumentParser(description="MedScribe v2 — Unsloth Training")
    parser.add_argument("--config", type=str, default="configs/training.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Train for 10 steps only")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model_name = config["model"]["base_model"]
    max_seq_length = config["model"]["max_seq_length"]
    load_in_4bit = config["model"]["load_in_4bit"]
    lora_config = config["lora"]
    train_config = config["training"]

    print("=" * 60)
    print(f"MedScribe v2 — Unsloth LoRA Training")
    print(f"Model: {model_name}")
    print(f"Max seq length: {max_seq_length}")
    print(f"4-bit: {load_in_4bit}")
    print("=" * 60)

    # ── Gate: training data exists ──
    train_file = config["data"]["train_file"]
    val_file = config["data"]["validation_file"]
    for f_path in [train_file, val_file]:
        if not os.path.exists(f_path):
            print(f"ABORT: Training data not found: {f_path}")
            print("Run scripts/04_prepare_training.py first.")
            sys.exit(1)

    # Count samples
    with open(train_file, "r", encoding="utf-8") as f:
        train_count = sum(1 for _ in f)
    with open(val_file, "r", encoding="utf-8") as f:
        val_count = sum(1 for _ in f)
    print(f"Training samples: {train_count}")
    print(f"Validation samples: {val_count}")

    if train_count < 50:
        print(f"WARNING: Only {train_count} training samples. Results may be poor.")

    # ── Load model via Unsloth ──
    from unsloth import FastLanguageModel

    print(f"\nLoading {model_name} via Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        dtype=None,  # auto-detect
    )

    # ── Apply LoRA ──
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
    print(f"Trainable parameters: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    # ── Load dataset ──
    from datasets import load_dataset

    def load_jsonl_messages(path):
        """Load JSONL where each line has a 'messages' field."""
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line.strip())
                data.append(obj)
        return data

    train_data = load_jsonl_messages(train_file)
    val_data = load_jsonl_messages(val_file)

    # Convert to HF dataset format
    from datasets import Dataset

    def format_for_sft(examples):
        """Format messages into the chat template string for SFTTrainer."""
        formatted = []
        for ex in examples:
            text = tokenizer.apply_chat_template(
                ex["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            formatted.append({"text": text})
        return formatted

    train_formatted = format_for_sft(train_data)
    val_formatted = format_for_sft(val_data)

    train_dataset = Dataset.from_list(train_formatted)
    val_dataset = Dataset.from_list(val_formatted)

    print(f"Formatted {len(train_dataset)} train / {len(val_dataset)} val examples")

    # ── Training ──
    from trl import SFTTrainer
    from transformers import TrainingArguments

    max_steps = 10 if args.dry_run else -1
    num_epochs = 1 if args.dry_run else train_config["num_train_epochs"]

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

    print(f"\nStarting training {'(DRY RUN — 10 steps)' if args.dry_run else ''}...")
    trainer.train()

    # ── Evaluate ──
    print("\nRunning evaluation...")
    eval_results = trainer.evaluate()
    print(f"Eval loss: {eval_results.get('eval_loss', 'N/A')}")

    # ── Save ──
    output_dir = os.path.join(train_config["output_dir"], "final")
    print(f"\nSaving model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # ── Export to GGUF for Ollama ──
    if not args.dry_run:
        export_config = config.get("export", {})
        gguf_quant = export_config.get("gguf_quantization", "q4_k_m")
        gguf_dir = export_config.get("output_dir", "./models/exported")
        os.makedirs(gguf_dir, exist_ok=True)

        print(f"\nExporting to GGUF ({gguf_quant})...")
        model.save_pretrained_gguf(
            gguf_dir,
            tokenizer,
            quantization_method=gguf_quant,
        )
        print(f"GGUF exported to {gguf_dir}")
        print(f"\nTo serve via Ollama:")
        print(f"  ollama create medscribe-v2 -f configs/Modelfile")
    else:
        print("\nDRY RUN complete. Skipping GGUF export.")
        print(f"If training looks good, run without --dry-run")

    print("=" * 60)


if __name__ == "__main__":
    main()
