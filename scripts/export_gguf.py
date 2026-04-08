"""
Export fine-tuned LoRA adapter to GGUF for Ollama deployment.

Step 1: Unsloth merges LoRA into base model -> saves 16-bit safetensors
Step 2: llama.cpp converts safetensors -> GGUF (with quantization)
Step 3: Register with Ollama

Usage:
  python scripts/export_gguf.py
  python scripts/export_gguf.py --quant q8_0
"""
import argparse
import os
import subprocess
import sys

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

OLLAMA_EXE = os.path.expanduser("~/AppData/Local/Programs/Ollama/ollama.exe")
ADAPTER_PATH = "./models/checkpoints/final"
MERGED_DIR = "./models/merged_16bit"
EXPORT_DIR = "./models/exported"
MODELFILE_PATH = "./configs/Modelfile"


def step1_merge(adapter_path):
    """Load base + LoRA via Unsloth, merge, save 16-bit safetensors."""
    print("=" * 60)
    print("Step 1: Merging LoRA into base model (Unsloth)...")
    print("=" * 60)

    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=4096,
        load_in_4bit=True,
    )

    os.makedirs(MERGED_DIR, exist_ok=True)
    model.save_pretrained_merged(
        MERGED_DIR,
        tokenizer,
        save_method="merged_16bit",
    )
    print(f"Merged model saved to {MERGED_DIR}")

    # Free GPU memory before llama.cpp conversion
    del model, tokenizer
    import torch, gc
    gc.collect()
    torch.cuda.empty_cache()


def step2_convert_gguf(quant):
    """Convert merged safetensors to GGUF via llama.cpp."""
    print("=" * 60)
    print(f"Step 2: Converting to GGUF ({quant}) via llama.cpp...")
    print("=" * 60)

    convert_script = "./llama.cpp/convert_hf_to_gguf.py"
    if not os.path.exists(convert_script):
        print("Cloning llama.cpp...")
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp", "./llama.cpp"],
            check=True,
        )
        # Only install gguf package — full requirements.txt has torch version conflicts
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "gguf"],
            check=True,
        )

    os.makedirs(EXPORT_DIR, exist_ok=True)

    if quant in ("f16", "f32"):
        # Direct conversion, no quantization needed
        gguf_output = os.path.join(EXPORT_DIR, f"sakhi-e4b-{quant}.gguf")
        subprocess.run(
            [sys.executable, convert_script, MERGED_DIR, "--outfile", gguf_output, "--outtype", quant],
            check=True,
        )
    else:
        # Convert to f16 first, then quantize
        gguf_f16 = os.path.join(EXPORT_DIR, "sakhi-e4b-f16.gguf")
        gguf_output = os.path.join(EXPORT_DIR, f"sakhi-e4b-{quant}.gguf")

        print("Converting HF -> GGUF (f16)...")
        subprocess.run(
            [sys.executable, convert_script, MERGED_DIR, "--outfile", gguf_f16, "--outtype", "f16"],
            check=True,
        )

        # Find llama-quantize binary
        quantize_bin = None
        for candidate in [
            "./llama.cpp/build/bin/llama-quantize",
            "./llama.cpp/build/bin/Release/llama-quantize",
            "./llama.cpp/build/bin/Release/llama-quantize.exe",
            "./llama.cpp/build/bin/llama-quantize.exe",
        ]:
            if os.path.exists(candidate):
                quantize_bin = candidate
                break

        if quantize_bin is None:
            print("Building llama.cpp (needs cmake)...")
            subprocess.run(
                ["cmake", "-B", "build", "-DCMAKE_BUILD_TYPE=Release"],
                cwd="./llama.cpp", check=True,
            )
            subprocess.run(
                ["cmake", "--build", "build", "--config", "Release", "-j"],
                cwd="./llama.cpp", check=True,
            )
            # Re-check after build
            for candidate in [
                "./llama.cpp/build/bin/Release/llama-quantize.exe",
                "./llama.cpp/build/bin/llama-quantize.exe",
                "./llama.cpp/build/bin/llama-quantize",
            ]:
                if os.path.exists(candidate):
                    quantize_bin = candidate
                    break

        if quantize_bin is None:
            print("ERROR: llama-quantize not found after build!")
            print(f"F16 GGUF is at: {gguf_f16}")
            print("You can quantize manually later.")
            gguf_output = gguf_f16
        else:
            quant_type = quant.upper()
            print(f"Quantizing f16 -> {quant_type}...")
            subprocess.run([quantize_bin, gguf_f16, gguf_output, quant_type], check=True)
            os.remove(gguf_f16)

    print(f"GGUF: {gguf_output}")
    return gguf_output


def step3_ollama(gguf_path, model_name):
    """Create Ollama model from GGUF."""
    print("=" * 60)
    print(f"Step 3: Creating Ollama model '{model_name}'...")
    print("=" * 60)

    import re
    abs_gguf = os.path.abspath(gguf_path)

    with open(MODELFILE_PATH, "r") as f:
        modelfile_content = f.read()

    modelfile_content = re.sub(
        r'^FROM\s+.*$',
        f'FROM {abs_gguf}',
        modelfile_content,
        flags=re.MULTILINE,
    )

    updated_modelfile = os.path.join(EXPORT_DIR, "Modelfile")
    with open(updated_modelfile, "w") as f:
        f.write(modelfile_content)

    result = subprocess.run(
        [OLLAMA_EXE, "create", model_name, "-f", updated_modelfile],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        sys.exit(1)

    print(f"Done! Test with: ollama run {model_name} \"नमस्ते\"")


def main():
    parser = argparse.ArgumentParser(description="Export LoRA to GGUF for Ollama")
    parser.add_argument("--quant", default="q4_k_m", help="Quantization (default: q4_k_m)")
    parser.add_argument("--adapter", default=ADAPTER_PATH)
    parser.add_argument("--model-name", default="sakhi")
    parser.add_argument("--skip-merge", action="store_true", help="Skip step 1 (reuse existing merged)")
    parser.add_argument("--skip-ollama", action="store_true", help="Skip step 3")
    args = parser.parse_args()

    if not args.skip_merge:
        step1_merge(args.adapter)

    gguf_path = step2_convert_gguf(args.quant)

    if not args.skip_ollama:
        step3_ollama(gguf_path, args.model_name)
    else:
        print(f"\nGGUF ready at: {gguf_path}")
        print(f"Run: ollama create {args.model_name} -f models/exported/Modelfile")


if __name__ == "__main__":
    main()
