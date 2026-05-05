import asyncio
import csv
import io
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

# ── App Setup ────────────────────────────────────────────────────────────────
app = FastAPI()

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "processed"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

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
