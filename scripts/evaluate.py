"""
MedScribe v2 — Evaluation

Evaluates the fine-tuned model on the validation set.
Measures:
  1. Form extraction accuracy (field-by-field)
  2. Danger sign precision / recall
  3. Hallucination rate (danger signs without evidence)
  4. Referral decision accuracy
  5. JSON validity rate

Usage:
  python scripts/06_evaluate.py --model models/checkpoints/final
  python scripts/06_evaluate.py --model ollama:medscribe-v2
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
import torch
torch._dynamo.config.suppress_errors = True


def load_val_data(path: str) -> list[dict]:
    """Load validation JSONL (raw format with ground truth)."""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def field_accuracy(predicted: dict, ground_truth: dict, prefix: str = "") -> dict:
    """
    Compare predicted vs ground truth field-by-field.
    Returns {field: {correct, total, accuracy}}.
    """
    results = {}
    if not isinstance(ground_truth, dict):
        return results

    for key, gt_val in ground_truth.items():
        field_name = f"{prefix}.{key}" if prefix else key
        pred_val = predicted.get(key) if isinstance(predicted, dict) else None

        if isinstance(gt_val, dict):
            sub = field_accuracy(pred_val, gt_val, field_name)
            results.update(sub)
        elif isinstance(gt_val, list):
            # For arrays, check set overlap
            gt_set = set(str(x) for x in gt_val) if gt_val else set()
            pred_set = set(str(x) for x in (pred_val or [])) if isinstance(pred_val, list) else set()
            overlap = len(gt_set & pred_set)
            total = max(len(gt_set), 1)
            results[field_name] = {"correct": overlap, "total": total}
        else:
            match = (pred_val == gt_val) or (pred_val is None and gt_val is None)
            results[field_name] = {"correct": 1 if match else 0, "total": 1}

    return results


def danger_sign_metrics(predicted: dict, ground_truth: dict) -> dict:
    """
    Compute precision, recall, F1 for danger sign detection.
    Also checks hallucination rate (signs without evidence).
    """
    gt_signs = {s["sign"] for s in ground_truth.get("danger_signs", [])}
    pred_signs_list = predicted.get("danger_signs", []) if isinstance(predicted, dict) else []
    pred_signs = {s.get("sign", "") for s in pred_signs_list}

    tp = len(gt_signs & pred_signs)
    fp = len(pred_signs - gt_signs)
    fn = len(gt_signs - pred_signs)

    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if len(gt_signs) == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if len(gt_signs) == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Hallucination check: predicted signs without utterance_evidence
    hallucinations = 0
    for s in pred_signs_list:
        if not s.get("utterance_evidence"):
            hallucinations += 1

    # Referral decision accuracy
    gt_decision = ground_truth.get("referral_decision", {}).get("decision", "")
    pred_decision = ""
    if isinstance(predicted, dict):
        pred_decision = predicted.get("referral_decision", {}).get("decision", "")
    referral_correct = gt_decision == pred_decision

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "hallucinations": hallucinations,
        "total_predicted": len(pred_signs_list),
        "referral_correct": referral_correct,
    }


def run_inference_ollama(transcript: str, system_prompt: str, model: str) -> str:
    """Run inference via Ollama."""
    import ollama
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcript},
        ],
    )
    return response.message.content


def run_inference_transformers(transcript: str, system_prompt: str, model_path: str) -> str:
    """Run inference via Unsloth-loaded model (handles LoRA + 4-bit)."""
    from unsloth import FastLanguageModel

    # Cache model loading
    if not hasattr(run_inference_transformers, "_model"):
        print("  Loading model via Unsloth...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=4096,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        run_inference_transformers._model = model
        run_inference_transformers._tokenizer = tokenizer

    model = run_inference_transformers._model
    tokenizer = run_inference_transformers._tokenizer

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": transcript},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text=[text], return_tensors="pt").to("cuda")
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=2048, do_sample=False)
    return tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="MedScribe v2 — Evaluation")
    parser.add_argument("--model", required=True, help="Model path or ollama:<name>")
    parser.add_argument("--val-file", default="data/processed/val.jsonl")
    parser.add_argument("--raw-file", default="data/processed/training_data_raw.jsonl",
                        help="Raw data with ground truth for danger sign eval")
    parser.add_argument("--limit", type=int, default=0, help="Limit eval to N samples")
    args = parser.parse_args()

    is_ollama = args.model.startswith("ollama:")
    model_ref = args.model.split(":", 1)[1] if is_ollama else args.model

    print("=" * 60)
    print(f"MedScribe v2 — Evaluation")
    print(f"Model: {args.model}")
    print("=" * 60)

    # Load validation data
    val_data = load_val_data(args.val_file)
    if args.limit > 0:
        val_data = val_data[:args.limit]
    print(f"Evaluating on {len(val_data)} samples")

    # Metrics accumulators
    json_valid = 0
    json_invalid = 0
    all_field_results = defaultdict(lambda: {"correct": 0, "total": 0})
    all_danger_metrics = []

    for i, sample in enumerate(val_data):
        messages = sample["messages"]
        system_msg = messages[0]["content"]
        user_msg = messages[1]["content"]
        gt_response = messages[2]["content"]

        # Run inference
        try:
            if is_ollama:
                pred_text = run_inference_ollama(user_msg, system_msg, model_ref)
            else:
                pred_text = run_inference_transformers(user_msg, system_msg, model_ref)
        except Exception as e:
            print(f"  [{i+1}] Inference error: {e}")
            json_invalid += 1
            continue

        # Parse JSON — strip markdown code fences if present
        pred_clean = pred_text.strip()
        if pred_clean.startswith("```"):
            # Remove ```json ... ``` wrapper
            lines = pred_clean.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            pred_clean = "\n".join(lines)
        try:
            pred_data = json.loads(pred_clean)
            gt_data = json.loads(gt_response)
            json_valid += 1
        except json.JSONDecodeError:
            json_invalid += 1
            print(f"  [{i+1}] Invalid JSON: {pred_clean[:100]}...")
            continue

        # Field accuracy
        field_results = field_accuracy(pred_data, gt_data)
        for field, res in field_results.items():
            all_field_results[field]["correct"] += res["correct"]
            all_field_results[field]["total"] += res["total"]

        # Danger sign metrics (if this is a danger sign task)
        task = sample.get("metadata", {}).get("task", "")
        if task == "danger_signs":
            dm = danger_sign_metrics(pred_data, gt_data)
            all_danger_metrics.append(dm)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(val_data)}] processed")

    # ── Results ──
    total = json_valid + json_invalid
    print(f"\n{'=' * 60}")
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(f"\n  JSON validity: {json_valid}/{total} ({json_valid/total*100:.0f}%)")

    # Field accuracy
    if all_field_results:
        total_correct = sum(v["correct"] for v in all_field_results.values())
        total_fields = sum(v["total"] for v in all_field_results.values())
        print(f"\n  Overall field accuracy: {total_correct}/{total_fields} ({total_correct/total_fields*100:.1f}%)")
        print(f"\n  Per-field accuracy (top 20):")
        sorted_fields = sorted(all_field_results.items(),
                              key=lambda x: x[1]["correct"]/max(x[1]["total"],1))
        for field, res in sorted_fields[:20]:
            acc = res["correct"] / max(res["total"], 1) * 100
            print(f"    {field}: {acc:.0f}% ({res['correct']}/{res['total']})")

    # Danger sign metrics
    if all_danger_metrics:
        avg_precision = sum(m["precision"] for m in all_danger_metrics) / len(all_danger_metrics)
        avg_recall = sum(m["recall"] for m in all_danger_metrics) / len(all_danger_metrics)
        avg_f1 = sum(m["f1"] for m in all_danger_metrics) / len(all_danger_metrics)
        total_hallucinations = sum(m["hallucinations"] for m in all_danger_metrics)
        total_predicted = sum(m["total_predicted"] for m in all_danger_metrics)
        referral_correct = sum(1 for m in all_danger_metrics if m["referral_correct"])

        hallucination_rate = total_hallucinations / max(total_predicted, 1) * 100

        print(f"\n  Danger Sign Detection:")
        print(f"    Precision: {avg_precision:.2f}")
        print(f"    Recall:    {avg_recall:.2f}")
        print(f"    F1:        {avg_f1:.2f}")
        print(f"    Hallucination rate: {hallucination_rate:.1f}% ({total_hallucinations}/{total_predicted})")
        print(f"    Referral accuracy:  {referral_correct}/{len(all_danger_metrics)} ({referral_correct/len(all_danger_metrics)*100:.0f}%)")

    # Save results
    output = {
        "json_validity": {"valid": json_valid, "invalid": json_invalid, "rate": json_valid / max(total, 1)},
        "field_accuracy": {k: {**v, "accuracy": v["correct"]/max(v["total"],1)} for k, v in all_field_results.items()},
    }
    if all_danger_metrics:
        output["danger_signs"] = {
            "precision": avg_precision, "recall": avg_recall, "f1": avg_f1,
            "hallucination_rate": hallucination_rate,
            "referral_accuracy": referral_correct / len(all_danger_metrics),
        }

    eval_path = "data/processed/eval_results.json"
    with open(eval_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {eval_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
