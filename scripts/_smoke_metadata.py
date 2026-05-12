"""Smoke test: POST a real recording with metadata override to /api/process-audio-stream,
print the final extracted form.child / patient / metadata envelope so we can confirm
apply_metadata kicked in (i.e. names/age/sex came from header, not the LLM).

Usage: python scripts/_smoke_metadata.py <audio_path> [--visit-type child_health]
"""
from __future__ import annotations
import argparse
import json
import sys
import time
import urllib.request
import urllib.error
import uuid

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def post_multipart(url: str, audio_path: str, fields: dict) -> str:
    boundary = f"----sakhi-smoke-{uuid.uuid4().hex}"
    body = bytearray()
    for k, v in fields.items():
        body += f"--{boundary}\r\n".encode()
        body += f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode()
        body += str(v).encode("utf-8") + b"\r\n"
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    fname = audio_path.replace("\\", "/").rsplit("/", 1)[-1]
    body += f"--{boundary}\r\n".encode()
    body += f'Content-Disposition: form-data; name="audio"; filename="{fname}"\r\n'.encode()
    body += b"Content-Type: application/octet-stream\r\n\r\n"
    body += audio_bytes + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        url, data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    last_complete = None
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as resp:
        # Stream SSE
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            if not line.startswith("data: "):
                continue
            try:
                evt = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            if evt.get("stage") == "complete":
                last_complete = evt
                print(f"[smoke] complete at {time.time()-t0:.1f}s")
                break
            else:
                stage = evt.get("stage") or evt.get("error") or "?"
                status = evt.get("status") or ""
                print(f"[smoke] {time.time()-t0:5.1f}s  {stage:<10} {status}")
    return last_complete


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--visit-type", default="child_health")
    ap.add_argument("--patient-name", default="आरव")
    ap.add_argument("--patient-age", default="14")
    ap.add_argument("--age-unit", default="months")
    ap.add_argument("--patient-sex", default="male")
    ap.add_argument("--asha-id", default="ASHA-9999")
    ap.add_argument("--visit-date", default="2026-05-13")
    ap.add_argument("--url", default="http://localhost:8000/api/process-audio-stream")
    args = ap.parse_args()

    fields = {
        "visit_type": args.visit_type,
        "patient_name": args.patient_name,
        "patient_age": args.patient_age,
        "age_unit": args.age_unit,
        "patient_sex": args.patient_sex,
        "asha_id": args.asha_id,
        "visit_date": args.visit_date,
    }
    print(f"[smoke] POST {args.url}  audio={args.audio}")
    print(f"[smoke] fields={fields}")
    evt = post_multipart(args.url, args.audio, fields)
    if not evt:
        print("[smoke] no complete event received")
        sys.exit(2)

    print("\n=== detected visit_type:", evt.get("visit_type"))
    print("=== metadata envelope:")
    print(json.dumps(evt.get("metadata"), ensure_ascii=False, indent=2))
    form = evt.get("form") or {}
    print("=== form.child:")
    print(json.dumps(form.get("child"), ensure_ascii=False, indent=2))
    print("=== form.patient:")
    print(json.dumps(form.get("patient"), ensure_ascii=False, indent=2))
    print("=== timing:")
    print(json.dumps(evt.get("timing"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
