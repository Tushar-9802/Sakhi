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

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

# Import pipeline functions from app.py
from app import (
    transcribe_audio,
    extract_form,
    extract_danger_signs,
    extract_all,
    detect_visit_type,
    init_schemas,
    validate_form_output,
    postprocess_transcript,
)

app = FastAPI(title="Sakhi API", version="1.0.0")

# CORS for React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load schemas on startup — models load lazily on first request (like Gradio)
@app.on_event("startup")
def startup():
    init_schemas()


# ── Models ──
class PatientMetadata(BaseModel):
    """ASHA-entered patient identifier fields. All optional — pipeline still runs without them.
    When provided, override LLM-extracted name/age/sex in the form (see apply_metadata in app.py)."""
    patient_name: Optional[str] = None
    patient_age: Optional[int] = None
    age_unit: Optional[str] = None        # "years" | "months"
    patient_sex: Optional[str] = None     # "male" | "female"
    patient_mobile: Optional[str] = None
    asha_id: Optional[str] = None
    visit_date: Optional[str] = None      # ISO date string


class TextRequest(BaseModel):
    transcript: str
    visit_type: Optional[str] = "auto"
    metadata: Optional[PatientMetadata] = None


class ExtractionResult(BaseModel):
    visit_type: str
    form: Optional[dict] = None
    danger: Optional[dict] = None
    metadata: Optional[dict] = None
    transcript: Optional[str] = None
    timing: dict = {}
    tool_calls: Optional[list] = None
    error: Optional[str] = None


def _metadata_dict(meta):
    """Coerce a PatientMetadata or None into a dict (or None if empty)."""
    if meta is None:
        return None
    d = meta.dict() if hasattr(meta, "dict") else dict(meta)
    # Drop all-None entries so apply_metadata short-circuits cleanly
    return {k: v for k, v in d.items() if v is not None and v != ""} or None


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

    metadata = _metadata_dict(req.metadata)
    result = extract_all(transcript, visit_type, metadata=metadata)

    total = time.time() - t_total
    timing = result.get("timing", {})
    timing["total_s"] = round(total, 1)

    return ExtractionResult(
        visit_type=visit_type,
        form=result["form"],
        danger=result["danger"],
        metadata=result.get("metadata"),
        timing=timing,
        tool_calls=result.get("tool_calls"),
    )


@app.post("/api/process-audio", response_model=ExtractionResult)
async def process_audio(
    audio: UploadFile = File(...),
    visit_type: str = Form("auto"),
    patient_name: Optional[str] = Form(None),
    patient_age: Optional[int] = Form(None),
    age_unit: Optional[str] = Form(None),
    patient_sex: Optional[str] = Form(None),
    patient_mobile: Optional[str] = Form(None),
    asha_id: Optional[str] = Form(None),
    visit_date: Optional[str] = Form(None),
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

        metadata = _metadata_dict(PatientMetadata(
            patient_name=patient_name, patient_age=patient_age, age_unit=age_unit,
            patient_sex=patient_sex, patient_mobile=patient_mobile,
            asha_id=asha_id, visit_date=visit_date,
        ))
        result = extract_all(transcript, vtype, metadata=metadata)

        total = time.time() - t_total
        timing = result.get("timing", {})
        timing["asr_s"] = round(asr_time, 1)
        timing["total_s"] = round(total, 1)

        return ExtractionResult(
            visit_type=vtype,
            form=result["form"],
            danger=result["danger"],
            metadata=result.get("metadata"),
            transcript=transcript,
            timing=timing,
            tool_calls=result.get("tool_calls"),
        )
    finally:
        os.unlink(tmp_path)


def _sse_event(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


@app.post("/api/process-text-stream")
async def process_text_stream(req: TextRequest):
    def generate():
        t_total = time.time()
        transcript = req.transcript.strip()
        if not transcript:
            yield _sse_event({"error": "Empty transcript"})
            return

        # Detect visit type
        yield _sse_event({"stage": "detect", "status": "running"})
        if req.visit_type and req.visit_type != "auto":
            visit_type = req.visit_type.lower().replace(" ", "_")
        else:
            visit_type = detect_visit_type(transcript)
        yield _sse_event({"stage": "detect", "status": "done", "visit_type": visit_type})

        metadata = _metadata_dict(req.metadata)

        # Unified extraction (form + danger in one LLM call via function calling)
        yield _sse_event({"stage": "form", "status": "running"})
        t0 = time.time()
        result = extract_all(transcript, visit_type, metadata=metadata)
        extract_time = time.time() - t0
        yield _sse_event({"stage": "form", "status": "done", "time": round(extract_time, 1)})

        # Danger stage is instant (already done in same call)
        yield _sse_event({"stage": "danger", "status": "done", "time": 0.0})

        total = time.time() - t_total
        timing = result.get("timing", {})
        timing["total_s"] = round(total, 1)
        yield _sse_event({
            "stage": "complete",
            "visit_type": visit_type,
            "form": result["form"],
            "danger": result["danger"],
            "metadata": result.get("metadata"),
            "tool_calls": result.get("tool_calls"),
            "timing": timing,
        })

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/process-audio-stream")
async def process_audio_stream(
    audio: UploadFile = File(...),
    visit_type: str = Form("auto"),
    patient_name: Optional[str] = Form(None),
    patient_age: Optional[int] = Form(None),
    age_unit: Optional[str] = Form(None),
    patient_sex: Optional[str] = Form(None),
    patient_mobile: Optional[str] = Form(None),
    asha_id: Optional[str] = Form(None),
    visit_date: Optional[str] = Form(None),
):
    # Save uploaded audio to temp file before streaming
    suffix = os.path.splitext(audio.filename or "audio.wav")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    metadata = _metadata_dict(PatientMetadata(
        patient_name=patient_name, patient_age=patient_age, age_unit=age_unit,
        patient_sex=patient_sex, patient_mobile=patient_mobile,
        asha_id=asha_id, visit_date=visit_date,
    ))

    def generate():
        t_total = time.time()
        try:
            # ASR
            yield _sse_event({"stage": "asr", "status": "running"})
            t0 = time.time()
            transcript = transcribe_audio(tmp_path)
            asr_time = time.time() - t0
            yield _sse_event({"stage": "asr", "status": "done", "time": round(asr_time, 1)})

            if not transcript or not transcript.strip():
                yield _sse_event({"error": "Transcription returned empty"})
                return

            # Normalize
            yield _sse_event({"stage": "normalize", "status": "running"})
            transcript = postprocess_transcript(transcript)
            yield _sse_event({"stage": "normalize", "status": "done", "transcript": transcript})

            # Detect visit type
            yield _sse_event({"stage": "detect", "status": "running"})
            if visit_type and visit_type != "auto":
                vtype = visit_type.lower().replace(" ", "_")
            else:
                vtype = detect_visit_type(transcript)
            yield _sse_event({"stage": "detect", "status": "done", "visit_type": vtype})

            # Unified extraction (form + danger in one LLM call via function calling)
            yield _sse_event({"stage": "form", "status": "running"})
            t1 = time.time()
            result = extract_all(transcript, vtype, metadata=metadata)
            extract_time = time.time() - t1
            yield _sse_event({"stage": "form", "status": "done", "time": round(extract_time, 1)})

            # Danger stage is instant (already done in same call)
            yield _sse_event({"stage": "danger", "status": "done", "time": 0.0})

            total = time.time() - t_total
            timing = result.get("timing", {})
            timing["asr_s"] = round(asr_time, 1)
            timing["total_s"] = round(total, 1)
            yield _sse_event({
                "stage": "complete",
                "visit_type": vtype,
                "form": result["form"],
                "danger": result["danger"],
                "metadata": result.get("metadata"),
                "transcript": transcript,
                "tool_calls": result.get("tool_calls"),
                "timing": timing,
            })
        finally:
            os.unlink(tmp_path)

    return StreamingResponse(generate(), media_type="text/event-stream")


# Serve built React frontend at / when dist exists (unified desktop UI for health centers).
# Must be mounted AFTER all /api/* routes so they take priority.
_FRONTEND_DIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "dist")
if os.path.isdir(_FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=_FRONTEND_DIST, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
