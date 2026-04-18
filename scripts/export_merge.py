"""Merge LoRA adapter into base model and save as HF checkpoint.

Bypasses Unsloth and PEFT's module-matching to avoid both:
  - Unsloth 2026.4.2 dropping `gemma-4-E4B-it` model name
  - PEFT's ValueError on Gemma4ClippableLinear wrappers

Manual merge: delta_W = (B @ A) * (alpha/r), added to base weights.
Output can then be converted to GGUF via llama.cpp's convert_hf_to_gguf.py.
"""
import json
import os
import sys

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

BASE_MODEL = "google/gemma-4-E4B-it"
ADAPTER_DIR = "./models/checkpoints/final"
MERGED_DIR = "./models/merged_fp16"

print("=" * 60)
print("Manual LoRA merge: base + adapter -> merged FP16")
print(f"Base:    {BASE_MODEL}")
print(f"Adapter: {ADAPTER_DIR}")
print(f"Output:  {MERGED_DIR}")
print("=" * 60)

if not os.path.exists(os.path.join(ADAPTER_DIR, "adapter_model.safetensors")):
    print(f"ABORT: No adapter at {ADAPTER_DIR}")
    sys.exit(1)

os.makedirs(MERGED_DIR, exist_ok=True)

# Read adapter config for r/alpha and target modules
with open(os.path.join(ADAPTER_DIR, "adapter_config.json"), "r") as f:
    ac = json.load(f)
r = ac["r"]
alpha = ac["lora_alpha"]
scale = alpha / r
print(f"\nAdapter: r={r}, alpha={alpha}, scale={scale:.3f}")
print(f"Target modules: {ac['target_modules']}")

print("\n[1/4] Loading base model in bfloat16 on GPU...")
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map="cuda",
    local_files_only=True,
)
print(f"  Loaded: {type(base).__name__}")

print("\n[2/4] Loading adapter weights...")
adapter_sd = load_file(os.path.join(ADAPTER_DIR, "adapter_model.safetensors"))
print(f"  Tensors: {len(adapter_sd)}")

# Pair up lora_A / lora_B by stripping suffixes
pairs = {}
for k in adapter_sd:
    if ".lora_A.weight" in k:
        base_key = k.replace(".lora_A.weight", "")
        pairs.setdefault(base_key, {})["A"] = adapter_sd[k]
    elif ".lora_B.weight" in k:
        base_key = k.replace(".lora_B.weight", "")
        pairs.setdefault(base_key, {})["B"] = adapter_sd[k]

print(f"  LoRA pairs: {len(pairs)}")

# Build a name -> module map for the base model. We need to match adapter keys
# like "base_model.model.model.layers.0.self_attn.q_proj" to the actual Linear
# weight in the model. Gemma 4 wraps Linear in Gemma4ClippableLinear.
name_to_module = dict(base.named_modules())

print("\n[3/4] Merging delta into base weights...")
merged = 0
skipped = 0
for key, ab in pairs.items():
    if "A" not in ab or "B" not in ab:
        skipped += 1
        continue
    # Strip the "base_model.model." prefix that PEFT adds
    target_path = key.replace("base_model.model.", "")

    module = name_to_module.get(target_path)
    if module is None:
        print(f"  MISS: {target_path}")
        skipped += 1
        continue

    # Find the actual weight tensor (could be module.weight or module.linear.weight)
    if hasattr(module, "weight") and isinstance(module.weight, torch.nn.Parameter):
        weight = module.weight
    elif hasattr(module, "linear") and hasattr(module.linear, "weight"):
        weight = module.linear.weight
    else:
        print(f"  NO_WEIGHT: {target_path} ({type(module).__name__})")
        skipped += 1
        continue

    A = ab["A"].to(weight.device, dtype=torch.float32)
    B = ab["B"].to(weight.device, dtype=torch.float32)
    delta = (B @ A) * scale

    with torch.no_grad():
        weight.add_(delta.to(weight.dtype))
    merged += 1

print(f"  Merged: {merged}, Skipped: {skipped}")

if merged == 0:
    print("ABORT: No LoRA pairs were merged")
    sys.exit(1)

print(f"\n[4/4] Saving merged model to {MERGED_DIR}...")
base.save_pretrained(MERGED_DIR, safe_serialization=True, max_shard_size="5GB")

try:
    processor = AutoProcessor.from_pretrained(BASE_MODEL, local_files_only=True)
    processor.save_pretrained(MERGED_DIR)
    print("  Processor saved")
except Exception as e:
    print(f"  Processor save skipped: {e}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True)
    tokenizer.save_pretrained(MERGED_DIR)
    print("  Tokenizer saved (fallback)")

print(f"\nDone. Merged model ready at: {MERGED_DIR}")
print("Next: convert_hf_to_gguf.py to produce GGUF")
