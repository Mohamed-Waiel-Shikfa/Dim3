import asyncio
import json
import os
import queue
import re
import shutil
import threading
import traceback
import uuid
from pathlib import Path
from typing import List

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from processing.pipeline import ProcessingPipeline
from training.trainer import run_training
from evaluation.routes import eval_router

# ── App Setup ────────────────────────────────────────────────────────────────
app = FastAPI()

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "processed"
TRAINING_DIR = BASE_DIR / "training_sessions"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
TRAINING_DIR.mkdir(exist_ok=True)

training_jobs: dict = {}

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.mount("/data", StaticFiles(directory=str(OUTPUT_DIR)), name="data")

pipeline = ProcessingPipeline(str(UPLOAD_DIR), str(OUTPUT_DIR))

app.state.pipeline = pipeline
app.include_router(eval_router)

# ── Page Routes ──────────────────────────────────────────────────────────────
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="base.html")

@app.get("/pages/data_processing")
async def data_processing(request: Request):
    return templates.TemplateResponse(request=request, name="data_processing.html")

@app.get("/pages/model_training")
async def model_training(request: Request):
    return templates.TemplateResponse(request=request, name="model_training.html")

@app.get("/pages/model_evaluation")
async def model_evaluation(request: Request):
    return templates.TemplateResponse(request=request, name="model_evaluation.html")


# ── API: Upload & Process ────────────────────────────────────────────────────
@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a 3D file and create a new processing session."""
    try:
        session_id = pipeline.new_session()
        save_path = UPLOAD_DIR / f"{session_id}_{file.filename}"

        with open(save_path, "wb") as f:
            content = await file.read()
            f.write(content)

        return JSONResponse({"session_id": session_id, "filename": file.filename, "path": str(save_path)})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/process/ingest")
async def process_ingest(file_path: str = Form(...)):
    """Run mesh ingestion (merge + export OBJ)."""
    try:
        result = pipeline.ingest(file_path)
        return JSONResponse(result)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/process/normalize")
async def process_normalize(voxel_size: float = Form(0.01)):
    """Run normalization and cleanup."""
    try:
        result = pipeline.normalize(voxel_size=voxel_size)
        return JSONResponse(result)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/process/normalize_low_poly")
async def process_normalize_low_poly(voxel_size: float = Form(0.05)):
    """Create low-poly version for GNN."""
    try:
        result = pipeline.normalize_low_poly(voxel_size=voxel_size)
        return JSONResponse(result)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/process/voxelize")
async def process_voxelize(grid_size: int = Form(32)):
    """Run voxelization."""
    try:
        result = pipeline.voxelize(grid_size=grid_size)
        return JSONResponse(result)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/process/sample")
async def process_sample(
    num_points: int = Form(1024),
    method: str = Form("fps")
):
    """Run point cloud sampling."""
    try:
        result = pipeline.sample_points(num_points=num_points, method=method)
        return JSONResponse(result)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/process/wireframe")
async def process_wireframe():
    """Extract wireframe from low-poly mesh."""
    try:
        result = pipeline.extract_wireframe()
        return JSONResponse(result)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/process/graph")
async def process_graph():
    """Extract graph topology."""
    try:
        result = pipeline.extract_graph()
        return JSONResponse(result)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


# ── API: Model Training ───────────────────────────────────────────────────────

@app.post("/api/train/upload")
async def train_upload(files: List[UploadFile] = File(...)):
    """Upload training files; class is derived from the relative path (directory name)."""
    try:
        job_id = str(uuid.uuid4())[:8]
        session_dir = TRAINING_DIR / job_id
        session_dir.mkdir(parents=True, exist_ok=True)

        files_by_class: dict = {}

        for file in files:
            # file.filename may be a relative path like "dataset/chair/001.obj"
            rel = file.filename or "unlabeled/file"
            parts = Path(rel).parts

            # Extract class from directory structure
            if len(parts) >= 3:
                cls = parts[1].lower()
            else:
                stem = re.sub(r"\.[^.]+$", "", parts[-1])
                m = re.match(r"^([a-zA-Z]+)", stem)
                cls = m.group(1).lower() if m else "unlabeled"

            # Skip hidden/system files
            if parts[-1].startswith("."):
                continue

            fname = Path(parts[-1]).name or f"file_{uuid.uuid4().hex[:6]}"

            cls_dir = session_dir / cls
            cls_dir.mkdir(exist_ok=True)

            # Avoid collisions with a short unique prefix
            save_path = cls_dir / f"{uuid.uuid4().hex[:6]}_{fname}"
            content = await file.read()
            save_path.write_bytes(content)
            files_by_class.setdefault(cls, []).append(str(save_path))

        return JSONResponse({
            "job_id": job_id,
            "files_by_class": {k: len(v) for k, v in files_by_class.items()},
        })
    except Exception as exc:
        traceback.print_exc()
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/api/train/start")
async def train_start(request: Request):
    """Launch a background training thread for the given job_id."""
    config = await request.json()
    job_id = config.get("job_id")
    if not job_id:
        return JSONResponse({"error": "No job_id"}, status_code=400)

    session_dir = TRAINING_DIR / job_id
    if not session_dir.exists():
        return JSONResponse({"error": "Session not found"}, status_code=404)

    files_by_class: dict = {}
    for cls_dir in session_dir.iterdir():
        if cls_dir.is_dir():
            paths = [str(f) for f in cls_dir.iterdir() if f.is_file()]
            if paths:
                files_by_class[cls_dir.name] = paths

    q           = queue.Queue()
    pause_event = threading.Event()
    stop_event  = threading.Event()

    thread = threading.Thread(
        target=run_training,
        args=(job_id, str(session_dir), files_by_class, config, q, pause_event, stop_event),
        daemon=True,
    )
    training_jobs[job_id] = {
        "queue":       q,
        "pause_event": pause_event,
        "stop_event":  stop_event,
        "thread":      thread,
        "session_dir": str(session_dir),
    }
    thread.start()
    return JSONResponse({"job_id": job_id, "status": "started"})


@app.get("/api/train/progress/{job_id}")
async def train_progress(job_id: str):
    """Server-Sent Events stream of training progress."""
    if job_id not in training_jobs:
        return JSONResponse({"error": "Job not found"}, status_code=404)

    q = training_jobs[job_id]["queue"]

    async def event_gen():
        try:
            while True:
                try:
                    event = q.get_nowait()
                    yield f"data: {json.dumps(event)}\n\n"
                    if event.get("type") in ("completed", "error"):
                        break
                except queue.Empty:
                    await asyncio.sleep(0.1)
        except GeneratorExit:
            pass

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/train/pause/{job_id}")
async def train_pause(job_id: str):
    if job_id in training_jobs:
        training_jobs[job_id]["pause_event"].set()
    return JSONResponse({"status": "paused"})


@app.post("/api/train/resume/{job_id}")
async def train_resume(job_id: str):
    if job_id in training_jobs:
        training_jobs[job_id]["pause_event"].clear()
    return JSONResponse({"status": "resumed"})


@app.post("/api/train/stop/{job_id}")
async def train_stop(job_id: str):
    if job_id in training_jobs:
        training_jobs[job_id]["stop_event"].set()
    return JSONResponse({"status": "stopped"})


@app.get("/api/train/download/{job_id}")
async def train_download(job_id: str):
    if job_id not in training_jobs:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    model_path = Path(training_jobs[job_id]["session_dir"]) / f"model_{job_id}.pt"
    if not model_path.exists():
        return JSONResponse({"error": "Model not ready"}, status_code=404)
    return FileResponse(
        str(model_path),
        filename=f"dim3_model_{job_id}.pt",
        media_type="application/octet-stream",
    )
