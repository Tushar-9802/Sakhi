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
class TextRequest(BaseModel):
    transcript: str
    visit_type: Optional[str] = "auto"


class ExtractionResult(BaseModel):
    visit_type: str
    form: Optional[dict] = None
    danger: Optional[dict] = None
    transcript: Optional[str] = None
    timing: dict = {}
    tool_calls: Optional[list] = None
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

    # Unified extraction (function calling if enabled, else separate calls)
    result = extract_all(transcript, visit_type)

    total = time.time() - t_total
    timing = result.get("timing", {})
    timing["total_s"] = round(total, 1)

    return ExtractionResult(
        visit_type=visit_type,
        form=result["form"],
        danger=result["danger"],
        timing=timing,
        tool_calls=result.get("tool_calls"),
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

        # Unified extraction
        result = extract_all(transcript, vtype)

        total = time.time() - t_total
        timing = result.get("timing", {})
        timing["asr_s"] = round(asr_time, 1)
        timing["total_s"] = round(total, 1)

        return ExtractionResult(
            visit_type=vtype,
            form=result["form"],
            danger=result["danger"],
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

        # Unified extraction (form + danger in one LLM call via function calling)
        yield _sse_event({"stage": "form", "status": "running"})
        t0 = time.time()
        result = extract_all(transcript, visit_type)
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
            "tool_calls": result.get("tool_calls"),
            "timing": timing,
        })

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/process-audio-stream")
async def process_audio_stream(
    audio: UploadFile = File(...),
    visit_type: str = Form("auto"),
):
    # Save uploaded audio to temp file before streaming
    suffix = os.path.splitext(audio.filename or "audio.wav")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

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
            result = extract_all(transcript, vtype)
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
                "transcript": transcript,
                "tool_calls": result.get("tool_calls"),
                "timing": timing,
            })
        finally:
            os.unlink(tmp_path)

    return StreamingResponse(generate(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
