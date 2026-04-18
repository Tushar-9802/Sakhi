"""Field-level diff: base vs sakhi on the same 15 transcripts.

The existing quality harness only checks `expected_form_checks` (pass/fail on
specific fields). This script captures FULL form JSON from both models and
diffs every leaf path, so we can identify cases where the fine-tune extracted
information the base model missed (or vice versa).
"""
import json
import os
import sys
import time

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_ollama_quality import (
    DANGER_SYSTEM_PROMPT,
    FORM_SYSTEM_PROMPT,
    TESTS,
    load_schemas,
    parse_json_response,
)

import ollama

MODELS = ["gemma4:e4b-it-q4_K_M", "sakhi:latest"]
OUT_PATH = "FIELD_COVERAGE_DIFF.md"


def flatten(d, prefix=""):
    """Return {dotted_path: value} for all leaves."""
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            p = f"{prefix}.{k}" if prefix else k
            out.update(flatten(v, p))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            out.update(flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = d
    return out


def is_null(v):
    return v is None or (isinstance(v, str) and v.strip().lower() in ("", "null", "none"))


def run_one(model, transcript, schema, danger_schema, visit_type):
    form_user = (
        f"Extract structured data from this ASHA home visit conversation:\n\n"
        f"{transcript}\n\n"
        f"Output JSON schema:\n{json.dumps(schema, ensure_ascii=False)}"
    )
    r1 = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": FORM_SYSTEM_PROMPT},
            {"role": "user", "content": form_user},
        ],
        options={"temperature": 0.0, "num_ctx": 4096},
    )
    form = parse_json_response(r1.message.content) or {}

    danger_user = (
        f"Analyze this ASHA home visit conversation for danger signs.\n\n"
        f"Visit type: {visit_type}\n\n"
        f"{transcript}\n\n"
        f"Output JSON schema:\n{json.dumps(danger_schema, ensure_ascii=False)}"
    )
    r2 = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": DANGER_SYSTEM_PROMPT},
            {"role": "user", "content": danger_user},
        ],
        options={"temperature": 0.0, "num_ctx": 4096},
    )
    danger = parse_json_response(r2.message.content) or {}
    return form, danger


def main():
    schemas = load_schemas()
    results = []

    for idx, test in enumerate(TESTS, 1):
        (name, visit_type, schema_name, transcript,
         expected_form, danger_min, danger_max, expected_referral,
         must_be_null) = test
        schema = schemas[schema_name]
        danger_schema = schemas["danger_signs"]
        print(f"\n[{idx}/{len(TESTS)}] {name}")

        outputs = {}
        for model in MODELS:
            t0 = time.time()
            form, danger = run_one(model, transcript, schema, danger_schema, visit_type)
            outputs[model] = {"form": form, "danger": danger, "elapsed": time.time() - t0}
            print(f"  {model}: {outputs[model]['elapsed']:.1f}s")

        results.append({"name": name, "outputs": outputs, "expected_form": expected_form,
                        "must_be_null": must_be_null})

    # Analyze diffs
    sakhi_only_count = 0
    base_only_count = 0
    diff_rows = []

    lines = ["# Field Coverage Diff: base vs sakhi\n"]
    lines.append(f"Date: {time.strftime('%Y-%m-%d %H:%M')}\n")
    lines.append("Captures every form leaf path, filtering out fields already covered by "
                 "the pass/fail harness (`expected_form_checks` + `hallucination_traps`).\n")

    for r in results:
        base = flatten(r["outputs"]["gemma4:e4b-it-q4_K_M"]["form"])
        sakhi = flatten(r["outputs"]["sakhi:latest"]["form"])
        tested_paths = set(r["expected_form"].keys()) | set(r["must_be_null"])

        sakhi_only = []
        base_only = []
        differ = []
        for path in set(base) | set(sakhi):
            if path in tested_paths:
                continue
            b, s = base.get(path), sakhi.get(path)
            if is_null(b) and not is_null(s):
                sakhi_only.append((path, s))
            elif is_null(s) and not is_null(b):
                base_only.append((path, b))
            elif not is_null(b) and not is_null(s) and b != s:
                differ.append((path, b, s))

        sakhi_only_count += len(sakhi_only)
        base_only_count += len(base_only)

        if sakhi_only or base_only or differ:
            lines.append(f"\n## {r['name']}\n")
            if sakhi_only:
                lines.append(f"**Sakhi extracted, base returned null** ({len(sakhi_only)}):")
                for p, v in sorted(sakhi_only):
                    lines.append(f"- `{p}` = `{v}`")
                lines.append("")
            if base_only:
                lines.append(f"**Base extracted, sakhi returned null** ({len(base_only)}):")
                for p, v in sorted(base_only):
                    lines.append(f"- `{p}` = `{v}`")
                lines.append("")
            if differ:
                lines.append(f"**Differ** ({len(differ)}):")
                for p, b, s in sorted(differ):
                    lines.append(f"- `{p}`: base=`{b}`, sakhi=`{s}`")
                lines.append("")

    summary = (
        f"\n## Summary\n\n"
        f"- Sakhi extracted fields base left null: **{sakhi_only_count}**\n"
        f"- Base extracted fields sakhi left null: **{base_only_count}**\n"
    )
    lines.insert(2, summary)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nSummary: sakhi_extra={sakhi_only_count}, base_extra={base_only_count}")
    print(f"Written to {OUT_PATH}")


if __name__ == "__main__":
    main()
