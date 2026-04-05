"""
MedScribe v2 — Function Calling Test (Gate 2)
Tests Gemma 4 E4B's native function calling with MCTS extraction schemas.

Tests:
  1. Schema loading and validation
  2. Function calling with a sample Hindi transcript
  3. Output validation against JSON schema
  4. Danger sign evidence grounding check
  5. Ollama function calling (text-only path)
  6. Transformers function calling (for audio pipeline)

Usage:
  python scripts/02_test_function_calling.py --mode ollama
  python scripts/02_test_function_calling.py --mode transformers
  python scripts/02_test_function_calling.py --validate-only  # schema validation only
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# ── Sample Test Data ────────────────────────────────────────────────────────

SAMPLE_TRANSCRIPT_HINDI = """
ASHA: नमस्ते बहन जी, कैसी तबीयत है आपकी?
Patient: नमस्ते दीदी, ठीक हूँ, बस थोड़ा चक्कर आता है कभी-कभी।
ASHA: पिछली बार जब आई थी तब आपने बताया था कि पैर सूज रहे हैं, अभी कैसे हैं?
Patient: हाँ दीदी, अभी भी सूजे हुए हैं, खासकर शाम को ज्यादा सूज जाते हैं। हाथ भी थोड़े सूजे लग रहे हैं।
ASHA: अच्छा, चलिए बी.पी. चेक करते हैं... बहन जी आपका बी.पी. 145/95 आ रहा है, ये थोड़ा ज़्यादा है। पिछली बार 130/85 था।
Patient: अरे, ये तो बढ़ गया दीदी! क्या करना चाहिए?
ASHA: और सिर में दर्द तो नहीं हो रहा?
Patient: हाँ, कल रात से सिर में दर्द हो रहा है, और आँखों के सामने थोड़ा धुंधला भी दिखा।
ASHA: ये सुनकर मुझे थोड़ी चिंता हो रही है। आपकी प्रेगनेंसी का कितना महीना चल रहा है?
Patient: 8वां महीना है दीदी, करीब 33-34 हफ्ते हो गए।
ASHA: वज़न चेक करते हैं... 62 किलो है। पिछली बार 59 किलो था, 3 किलो बढ़ा है दो हफ्ते में। बच्चा हिल रहा है ठीक से?
Patient: हाँ, हिल तो रहा है, लेकिन पहले से कम लग रहा है।
ASHA: आयरन की गोली खा रही हैं?
Patient: हाँ दीदी, रोज़ खा रही हूँ। TT का दूसरा टीका भी लगवा लिया है।
ASHA: अच्छा, अस्पताल जाने का इंतज़ाम किया है? कौन ले जाएगा?
Patient: हाँ, पति जी का ऑटो है, वो ले जाएंगे। ज़िला अस्पताल जाएंगे।
"""

SAMPLE_TRANSCRIPT_NORMAL = """
ASHA: नमस्ते बहन जी, कैसी हैं? बच्चे की तबीयत कैसी है?
Patient: नमस्ते दीदी, मैं ठीक हूँ, बच्चा भी ठीक है।
ASHA: बच्चे को दूध पिला रही हैं?
Patient: हाँ दीदी, सिर्फ अपना दूध दे रही हूँ, ऊपर का कुछ नहीं दिया।
ASHA: बहुत अच्छा! बच्चे का वज़न देखते हैं... 3.2 किलो है, जन्म के समय 2.8 था। अच्छा बढ़ रहा है।
Patient: हाँ, ठीक से पी रहा है, हर 2-3 घंटे में।
ASHA: नाभि कैसी है?
Patient: सूख गई है दीदी, साफ है।
ASHA: बच्चे की BCG और OPV की पहली खुराक लगवा ली थी ना?
Patient: हाँ, अस्पताल में ही लगा दी थी जन्म के समय।
ASHA: अगला टीका 6 हफ्ते पर लगेगा, याद रखना। और आप आयरन की गोली खा रही हैं?
Patient: हाँ दीदी, रोज़ खा रही हूँ।
"""


# ── Schema Loading ─────────────────────────────────────────────────────────

def load_schemas() -> dict:
    """Load all MCTS extraction schemas from configs/schemas/."""
    schema_dir = Path("configs/schemas")
    schemas = {}
    for f in schema_dir.glob("*.json"):
        with open(f, "r", encoding="utf-8") as fh:
            schemas[f.stem] = json.load(fh)
    print(f"  Loaded {len(schemas)} schemas: {list(schemas.keys())}")
    return schemas


def validate_schema(schema: dict) -> bool:
    """Validate that a schema is well-formed JSON Schema."""
    try:
        import jsonschema
        jsonschema.Draft7Validator.check_schema(schema)
        return True
    except Exception as e:
        print(f"  Schema validation error: {e}")
        return False


# ── Tool Definitions for Gemma 4 ──────────────────────────────────────────

def build_tool_definitions(schemas: dict) -> list:
    """
    Convert JSON schemas into Gemma 4 function calling tool definitions.
    Format follows the HuggingFace apply_chat_template(tools=...) pattern.
    """
    tools = []
    for name, schema in schemas.items():
        tool = {
            "type": "function",
            "function": {
                "name": f"extract_{name}",
                "description": schema.get("description", f"Extract {name} data from conversation"),
                "parameters": schema,
            }
        }
        tools.append(tool)
    return tools


# ── Ollama Function Calling Test ──────────────────────────────────────────

def test_ollama_function_calling(transcript: str, schemas: dict):
    """Test function calling via Ollama (text-only path)."""
    try:
        import ollama
    except ImportError:
        print("  ollama package not installed. Skipping.")
        return None

    print(f"\n=== Ollama Function Calling Test ===")
    tools = build_tool_definitions(schemas)

    # Use ANC + danger signs schemas for this test
    test_tools = [t for t in tools if t["function"]["name"] in ("extract_anc_visit", "extract_danger_signs")]

    system_prompt = (
        "You are a clinical data extraction system for India's ASHA health worker program. "
        "Extract structured data from the Hindi/Hinglish conversation transcript. "
        "ONLY extract information explicitly stated in the conversation. "
        "Use null for any field not mentioned. "
        "For danger signs, you MUST provide the exact utterance from the conversation as evidence. "
        "If no danger signs are present, return an empty danger_signs array."
    )

    t0 = time.time()
    try:
        response = ollama.chat(
            model="gemma4:e4b-it-q4_K_M",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Extract clinical data from this ASHA home visit conversation:\n\n{transcript}"},
            ],
            tools=test_tools,
        )
        elapsed = time.time() - t0
        print(f"  Response time: {elapsed:.1f}s")

        # Parse tool calls from response
        if hasattr(response, "message") and hasattr(response.message, "tool_calls"):
            tool_calls = response.message.tool_calls
            print(f"  Tool calls: {len(tool_calls)}")
            results = []
            for tc in tool_calls:
                print(f"\n  Function: {tc.function.name}")
                args = tc.function.arguments
                if isinstance(args, str):
                    args = json.loads(args)
                print(f"  Output:\n{json.dumps(args, ensure_ascii=False, indent=2)[:2000]}")
                results.append({"function": tc.function.name, "arguments": args})
            return results
        else:
            # Model responded with text instead of tool call
            text = response.message.content if hasattr(response, "message") else str(response)
            print(f"  Model returned text (no tool call):\n  {text[:500]}")
            return None

    except Exception as e:
        print(f"  Error: {e}")
        return None


# ── Transformers Function Calling Test ─────────────────────────────────────

def test_transformers_function_calling(transcript: str, schemas: dict, device: str = "cuda"):
    """Test function calling via HuggingFace Transformers (needed for audio pipeline)."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"\n=== Transformers Function Calling Test ===")

    model_id = "google/gemma-4-E4B-it"
    print(f"  Loading {model_id}...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    tools = build_tool_definitions(schemas)
    test_tools = [t for t in tools if t["function"]["name"] in ("extract_anc_visit", "extract_danger_signs")]

    messages = [
        {"role": "system", "content": (
            "You are a clinical data extraction system for India's ASHA health worker program. "
            "Extract structured data from the Hindi/Hinglish conversation transcript. "
            "ONLY extract information explicitly stated in the conversation. "
            "Use null for any field not mentioned. "
            "For danger signs, you MUST provide the exact utterance as evidence."
        )},
        {"role": "user", "content": f"Extract clinical data:\n\n{transcript}"},
    ]

    # Apply chat template with tools
    t0 = time.time()
    inputs = tokenizer.apply_chat_template(
        messages,
        tools=test_tools,
        return_tensors="pt",
        return_dict=True,
    ).to(device)

    print(f"  Input tokens: {inputs['input_ids'].shape[-1]}")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,
        )

    response = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    elapsed = time.time() - t0
    print(f"  Inference time: {elapsed:.1f}s")
    print(f"  Raw response:\n{response[:2000]}")

    # Try to parse structured output
    try:
        # Gemma 4 function calls use special format
        parsed = json.loads(response)
        print(f"\n  Parsed JSON successfully")
        return parsed
    except json.JSONDecodeError:
        print(f"\n  Response is not pure JSON — may need parsing of tool call format")
        return response


# ── Validation ─────────────────────────────────────────────────────────────

def validate_extraction(result: dict, schema: dict, schema_name: str) -> dict:
    """Validate extraction result against schema. Returns validation report."""
    import jsonschema

    report = {"schema": schema_name, "valid": True, "errors": [], "warnings": []}

    try:
        jsonschema.validate(result, schema)
    except jsonschema.ValidationError as e:
        report["valid"] = False
        report["errors"].append(str(e.message))

    # Custom validation: danger sign evidence check
    if schema_name == "danger_signs":
        danger_signs = result.get("danger_signs", [])
        for ds in danger_signs:
            if not ds.get("utterance_evidence"):
                report["valid"] = False
                report["errors"].append(
                    f"Danger sign '{ds.get('sign')}' has no utterance_evidence — HALLUCINATION"
                )

        # Check referral decision has evidence
        referral = result.get("referral_decision", {})
        if referral.get("decision") in ("refer_immediately", "refer_within_24h"):
            if not referral.get("evidence_utterances"):
                report["valid"] = False
                report["errors"].append("Referral decision has no evidence utterances")

    # Null field check: count how many fields are non-null
    non_null = 0
    total = 0
    for key, val in _flatten(result).items():
        total += 1
        if val is not None:
            non_null += 1
    if total > 0:
        fill_rate = non_null / total * 100
        report["fill_rate"] = f"{fill_rate:.0f}%"
        if fill_rate < 10:
            report["warnings"].append(f"Very low fill rate ({fill_rate:.0f}%) — model may not be extracting")

    return report


def _flatten(d, parent_key="", sep="."):
    """Flatten nested dict for field counting."""
    items = []
    if isinstance(d, dict):
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(_flatten(v, new_key, sep).items())
            elif isinstance(v, list):
                items.append((new_key, v if v else None))
            else:
                items.append((new_key, v))
    return dict(items)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MedScribe v2 — Function Calling Test")
    parser.add_argument("--mode", choices=["ollama", "transformers"], default="ollama")
    parser.add_argument("--validate-only", action="store_true", help="Only validate schemas")
    parser.add_argument("--transcript", type=str, help="Custom transcript file (UTF-8)")
    parser.add_argument("--normal", action="store_true", help="Use normal (no danger signs) transcript")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Load and validate schemas
    print("=== Schema Validation ===")
    schemas = load_schemas()
    all_valid = True
    for name, schema in schemas.items():
        valid = validate_schema(schema)
        status = "\033[92m[VALID]\033[0m" if valid else "\033[91m[INVALID]\033[0m"
        print(f"  {status} {name}")
        if not valid:
            all_valid = False

    if not all_valid:
        print("\nSchema validation failed. Fix schemas before testing model.")
        sys.exit(1)

    if args.validate_only:
        print("\nAll schemas valid.")
        return

    # Select transcript
    if args.transcript:
        with open(args.transcript, "r", encoding="utf-8") as f:
            transcript = f.read()
    elif args.normal:
        transcript = SAMPLE_TRANSCRIPT_NORMAL
        print("\n  Using NORMAL transcript (expect no danger signs)")
    else:
        transcript = SAMPLE_TRANSCRIPT_HINDI
        print("\n  Using HIGH-RISK transcript (expect danger signs)")

    # Run test
    if args.mode == "ollama":
        results = test_ollama_function_calling(transcript, schemas)
    else:
        results = test_transformers_function_calling(transcript, schemas, args.device)

    if results:
        print("\n=== Extraction Results ===")
        # Save results
        output_path = "data/temp/function_calling_test_result.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"  Saved to {output_path}")
    else:
        print("\n  No structured results returned. Model may need fine-tuning for this task.")


if __name__ == "__main__":
    main()
