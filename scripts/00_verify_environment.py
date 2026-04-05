"""
MedScribe v2 — Environment Verification (Gate 0)
Run this FIRST. Catches GPU/CUDA/library issues before they cascade.

Usage: python scripts/00_verify_environment.py
"""
import sys
import importlib
import subprocess

# ── Formatting helpers ──────────────────────────────────────────────────────
PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"
INFO = "\033[94m[INFO]\033[0m"

failures = []
warnings = []


def check(name, condition, msg_pass, msg_fail, critical=True):
    if condition:
        print(f"  {PASS} {name}: {msg_pass}")
        return True
    else:
        marker = FAIL if critical else WARN
        print(f"  {marker} {name}: {msg_fail}")
        if critical:
            failures.append(f"{name}: {msg_fail}")
        else:
            warnings.append(f"{name}: {msg_fail}")
        return False


# ── 1. Python Version ───────────────────────────────────────────────────────
print("\n=== 1. Python Environment ===")
py_ver = sys.version_info
py_str = f"{py_ver.major}.{py_ver.minor}.{py_ver.micro}"
check("Python version", py_ver >= (3, 10),
      f"{py_str}", f"{py_str} — need 3.10+")
check("Python 3.14 compat", py_ver < (3, 15),
      f"{py_str} within supported range",
      f"{py_str} may have compatibility issues with some packages",
      critical=False)

# ── 2. PyTorch + CUDA ──────────────────────────────────────────────────────
print("\n=== 2. PyTorch + CUDA ===")
try:
    import torch
    check("PyTorch import", True, f"v{torch.__version__}", "")

    cuda_avail = torch.cuda.is_available()
    check("CUDA available", cuda_avail,
          "Yes", "No — GPU acceleration will not work")

    if cuda_avail:
        gpu_name = torch.cuda.get_device_name(0)
        check("GPU detected", True, gpu_name, "")

        cc = torch.cuda.get_device_capability(0)
        cc_str = f"{cc[0]}.{cc[1]}"
        # RTX 5070 Ti = SM 12.0 (Blackwell)
        check("Compute capability", cc[0] >= 8,
              f"SM {cc_str}", f"SM {cc_str} — may be too old for bf16")

        vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        check("VRAM", vram_gb >= 8,
              f"{vram_gb:.1f} GB", f"{vram_gb:.1f} GB — need at least 8GB")

        # Test bf16 support
        try:
            t = torch.tensor([1.0], dtype=torch.bfloat16, device="cuda")
            check("BFloat16 support", True, "Supported", "")
            del t
        except Exception as e:
            check("BFloat16 support", False, "", str(e))

        cuda_ver = torch.version.cuda
        check("CUDA version", True, cuda_ver, "", critical=False)

    # Check nightly build
    is_nightly = "dev" in torch.__version__ or "nightly" in torch.__version__
    check("PyTorch nightly", is_nightly,
          f"Nightly build ({torch.__version__})",
          f"Stable build ({torch.__version__}) — RTX 5070 Ti needs nightly for SM 12.0",
          critical=False)

except ImportError:
    check("PyTorch import", False, "", "torch not installed")

# ── 3. Torchaudio ──────────────────────────────────────────────────────────
print("\n=== 3. Audio Stack ===")
try:
    import torchaudio
    check("torchaudio", True, f"v{torchaudio.__version__}", "")
except ImportError:
    check("torchaudio", False, "", "Not installed — needed for audio processing")

try:
    import librosa
    check("librosa", True, f"v{librosa.__version__}", "")
except ImportError:
    check("librosa", False, "", "Not installed", critical=False)

try:
    import soundfile
    check("soundfile", True, "Installed", "")
except ImportError:
    check("soundfile", False, "", "Not installed", critical=False)

# ── 4. Transformers Stack ──────────────────────────────────────────────────
print("\n=== 4. Transformers Stack ===")
for pkg, min_ver, crit in [
    ("transformers", "4.51.0", True),
    ("accelerate", "1.5.0", True),
    ("peft", "0.14.0", True),
    ("sentencepiece", "0.2.0", True),
    ("safetensors", "0.4.0", True),
    ("datasets", "3.0.0", True),
    ("trl", "0.15.0", False),
]:
    try:
        mod = importlib.import_module(pkg)
        ver = getattr(mod, "__version__", "unknown")
        check(pkg, True, f"v{ver}", "", critical=crit)
    except ImportError:
        check(pkg, False, "", f"Not installed (need >={min_ver})", critical=crit)

# ── 5. Unsloth ─────────────────────────────────────────────────────────────
print("\n=== 5. Unsloth (Fine-Tuning) ===")
try:
    import unsloth
    ver = getattr(unsloth, "__version__", "unknown")
    check("unsloth", True, f"v{ver}", "")
except ImportError:
    check("unsloth", False, "",
          "Not installed — needed for LoRA fine-tuning. Install: pip install unsloth[cu128]",
          critical=False)

# ── 6. Quantization ───────────────────────────────────────────────────────
print("\n=== 6. Quantization ===")
try:
    import bitsandbytes
    ver = getattr(bitsandbytes, "__version__", "unknown")
    check("bitsandbytes", True, f"v{ver}", "")
except ImportError:
    check("bitsandbytes", False, "", "Not installed", critical=False)

# ── 7. Ollama ──────────────────────────────────────────────────────────────
print("\n=== 7. Ollama (Serving) ===")
try:
    result = subprocess.run(["ollama", "--version"], capture_output=True, text=True, timeout=10)
    if result.returncode == 0:
        ver = result.stdout.strip() or result.stderr.strip()
        check("Ollama CLI", True, ver, "")
    else:
        check("Ollama CLI", False, "", "Installed but returned error", critical=False)
except FileNotFoundError:
    check("Ollama CLI", False, "",
          "Not found in PATH — install from ollama.com", critical=False)
except Exception as e:
    check("Ollama CLI", False, "", str(e), critical=False)

# Check if Ollama has gemma4 pulled
try:
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
    if result.returncode == 0 and "gemma4" in result.stdout.lower():
        check("Gemma 4 model", True, "Available in Ollama", "")
    else:
        check("Gemma 4 model", False, "",
              "Not pulled yet. Run: ollama pull gemma4:e4b-it-q4_K_M", critical=False)
except Exception:
    check("Gemma 4 model", False, "", "Could not check", critical=False)

# ── 8. Other Utilities ────────────────────────────────────────────────────
print("\n=== 8. Utilities ===")
for pkg in ["openai", "gradio", "jsonschema", "rich", "yaml", "pandas", "numpy"]:
    import_name = "pyyaml" if pkg == "yaml" else pkg
    try:
        mod = importlib.import_module(pkg)
        ver = getattr(mod, "__version__", "ok")
        check(pkg, True, f"v{ver}", "", critical=False)
    except ImportError:
        check(pkg, False, "", "Not installed", critical=False)

# ── 9. Gemma 4 E4B Quick Load Test ────────────────────────────────────────
print("\n=== 9. Gemma 4 E4B Tokenizer Test ===")
try:
    from transformers import AutoTokenizer
    print(f"  {INFO} Attempting to load Gemma 4 E4B tokenizer (may download ~500MB first time)...")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E4B-it", trust_remote_code=True)
    check("Gemma 4 E4B tokenizer", True,
          f"Loaded, vocab_size={tokenizer.vocab_size}", "")
    del tokenizer
except Exception as e:
    check("Gemma 4 E4B tokenizer", False, "",
          f"Could not load: {e}", critical=False)

# ── Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
if failures:
    print(f"\n{FAIL} {len(failures)} CRITICAL FAILURE(S):")
    for f in failures:
        print(f"  - {f}")
    print("\nFix these before proceeding to any other script.")
    sys.exit(1)
elif warnings:
    print(f"\n{WARN} {len(warnings)} WARNING(S) (non-blocking):")
    for w in warnings:
        print(f"  - {w}")
    print(f"\n{PASS} Environment is usable. Address warnings when possible.")
    sys.exit(0)
else:
    print(f"\n{PASS} All checks passed. Environment ready for MedScribe v2.")
    sys.exit(0)
