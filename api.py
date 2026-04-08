"""
Sakhi API — FastAPI backend for React frontend.

Endpoints:
  POST /api/process-audio   — Upload audio file → transcript + form + danger signs
  POST /api/process-text    — Submit transcript text → form + danger signs
  GET  /api/health          — Health check
  GET  /api/examples        — List example transcripts

Runs on port 8000. React frontend runs on port 3000.
"""
import os
import json
import time
import tempfile

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# Import pipeline functions from app.py
from app import (
    transcribe_audio,
    extract_form,
    extract_danger_signs,
    detect_visit_type,
    init_schemas,
    validate_form_output,
    postprocess_transcript,
)

app = FastAPI(title="Sakhi API", version="1.0.0")

# CORS for React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load schemas on startup — models load lazily on first request (like Gradio)
@app.on_event("startup")
def startup():
    init_schemas()


# ── Models ──
class TextRequest(BaseModel):
    transcript: str
    visit_type: Optional[str] = "auto"


class ExtractionResult(BaseModel):
    visit_type: str
    form: Optional[dict] = None
    danger: Optional[dict] = None
    transcript: Optional[str] = None
    timing: dict = {}
    error: Optional[str] = None


# ── Endpoints ──
@app.get("/api/health")
def health():
    return {"status": "ok", "model": os.environ.get("OLLAMA_MODEL", "gemma4:e4b-it-q4_K_M")}


@app.get("/api/examples")
def examples():
    from app import EXAMPLE_TRANSCRIPTS
    return [
        {"label": ex[0], "transcript": ex[1], "default": i == 1}
        for i, ex in enumerate(EXAMPLE_TRANSCRIPTS)
    ]
    # index 1 = "ANC Visit — Preeclampsia (DANGER)" — best for demo (has danger signs)


@app.post("/api/process-text", response_model=ExtractionResult)
def process_text(req: TextRequest):
    t_total = time.time()

    transcript = req.transcript.strip()
    if not transcript:
        return ExtractionResult(visit_type="unknown", error="Empty transcript")

    # Detect visit type
    if req.visit_type and req.visit_type != "auto":
        visit_type = req.visit_type.lower().replace(" ", "_")
    else:
        visit_type = detect_visit_type(transcript)

    # Form extraction
    t0 = time.time()
    form_result = extract_form(transcript, visit_type)
    form_time = time.time() - t0

    # Danger sign detection
    t1 = time.time()
    danger_result = extract_danger_signs(transcript, visit_type)
    danger_time = time.time() - t1

    total = time.time() - t_total

    return ExtractionResult(
        visit_type=visit_type,
        form=form_result.get("parsed"),
        danger=danger_result.get("parsed"),
        timing={
            "form_s": round(form_time, 1),
            "danger_s": round(danger_time, 1),
            "total_s": round(total, 1),
        },
    )


@app.post("/api/process-audio", response_model=ExtractionResult)
async def process_audio(
    audio: UploadFile = File(...),
    visit_type: str = Form("auto"),
):
    t_total = time.time()

    # Save uploaded audio to temp file
    suffix = os.path.splitext(audio.filename or "audio.wav")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # ASR
        t0 = time.time()
        transcript = transcribe_audio(tmp_path)
        asr_time = time.time() - t0

        if not transcript or not transcript.strip():
            return ExtractionResult(
                visit_type="unknown",
                error="Transcription returned empty",
                timing={"asr_s": round(asr_time, 1)},
            )

        # Detect visit type
        if visit_type and visit_type != "auto":
            vtype = visit_type.lower().replace(" ", "_")
        else:
            vtype = detect_visit_type(transcript)

        # Form extraction
        t1 = time.time()
        form_result = extract_form(transcript, vtype)
        form_time = time.time() - t1

        # Danger sign detection
        t2 = time.time()
        danger_result = extract_danger_signs(transcript, vtype)
        danger_time = time.time() - t2

        total = time.time() - t_total

        return ExtractionResult(
            visit_type=vtype,
            form=form_result.get("parsed"),
            danger=danger_result.get("parsed"),
            transcript=transcript,
            timing={
                "asr_s": round(asr_time, 1),
                "form_s": round(form_time, 1),
                "danger_s": round(danger_time, 1),
                "total_s": round(total, 1),
            },
        )
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
