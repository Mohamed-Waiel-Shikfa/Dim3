import zipfile
import uuid
import json
import traceback
import torch
from pathlib import Path
from typing import List

from fastapi import APIRouter, UploadFile, File, Request
from fastapi.responses import JSONResponse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error

# Import the exact model builder and feature extractors from trainer.py
from training.trainer import (
    _build_model,
    _load_mesh,
    _voxelize,
    _point_features,
    _graph_features
)

eval_router = APIRouter(prefix="/api/eval", tags=["Evaluation"])

# Define paths relative to this file's location (two folders up is the root)
BASE_DIR = Path(__file__).parent.parent
EVALUATION_DIR = BASE_DIR / "evaluation_sessions"
EVALUATION_DIR.mkdir(exist_ok=True)

@eval_router.post("/upload_model")
async def eval_upload_model(request: Request, file: UploadFile = File(...)):
    """Uploads a zip package containing config.json, metrics.csv, and a .pt model"""
    try:
        session_id = str(uuid.uuid4())[:8]
        session_dir = EVALUATION_DIR / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        zip_path = session_dir / file.filename
        with open(zip_path, "wb") as f:
            f.write(await file.read())

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(session_dir)

        try:
            config_file = next(session_dir.rglob("config.json"))
            metrics_file = next(session_dir.rglob("metrics.csv"))
        except StopIteration:
            raise FileNotFoundError("Could not find 'config.json' or 'metrics.csv' inside the uploaded ZIP.")

        with open(config_file, "r") as f:
            config = json.load(f)

        with open(metrics_file, "r") as f:
            metrics_csv = f.read()

        # Store the active session config path in the app state
        request.app.state.eval_model_dir = config_file.parent

        return JSONResponse({
            "status": "success",
            "config": config,
            "metrics_csv": metrics_csv
        })

    except Exception as e:
        print(f"--- ERROR IN UPLOAD MODEL ---")
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@eval_router.post("/predict")
async def eval_predict(request: Request, file: UploadFile = File(...)):
    """Runs a single 3D mesh through the Blender pipeline and into the PyTorch model"""
    pipeline = request.app.state.pipeline  # Fetch pipeline from app state
    try:
        session_id = pipeline.new_session()
        file_ext = Path(file.filename).suffix.lower()
        save_path = pipeline.upload_dir / f"{session_id}_eval_input{file_ext}"

        with open(save_path, "wb") as f:
            f.write(await file.read())

        pipeline.ingest(str(save_path))

        converted_files = list(pipeline.session_dir.glob("*.obj"))
        target_path = converted_files[0] if converted_files else save_path

        eval_model_dir = request.app.state.eval_model_dir
        config_path = eval_model_dir / "config.json"
        with open(config_path, "r") as f:
            config = json.load(f)

        pt_files = list(eval_model_dir.glob("*.pt"))
        if not pt_files:
            raise FileNotFoundError("No .pt model file found.")

        checkpoint = torch.load(pt_files[0], map_location="cpu")

        classes = checkpoint.get("class_names", config.get("class_names", ["Unknown"]))
        model_type = checkpoint.get("model_type", config.get("model_type", "cnn"))
        in_dim = checkpoint.get("in_dim", config.get("arch_param", 32))
        n_classes = checkpoint.get("n_classes", len(classes))
        layer_cfgs = config.get("layers", [])

        model = _build_model(model_type, layer_cfgs, in_dim, n_classes)
        model.load_state_dict(checkpoint.get('model_state', checkpoint))
        model.eval()

        mesh = _load_mesh(str(target_path))
        arch_param = config.get("arch_param", 32)

        if model_type == 'cnn':
            feat = _voxelize(mesh, arch_param)
            X = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        elif model_type == 'pointnet':
            feat = _point_features(mesh, arch_param)
            X = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
        else: # gnn
            feat = _graph_features(mesh)
            X = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            outputs = model(X)
            probabilities = torch.nn.functional.softmax(outputs, dim=1).squeeze(0)
            confidence, pred_idx = torch.max(probabilities, dim=0)

        predicted_class = classes[pred_idx.item()] if pred_idx.item() < len(classes) else "Unknown"

        return JSONResponse({
            "class": predicted_class,
            "confidence": confidence.item()
        })

    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        pipeline.cleanup_session()


@eval_router.post("/batch")
async def eval_batch(request: Request, files: List[UploadFile] = File(...)):
    """Processes a test directory dataset and calculates classic ML Metrics"""
    pipeline = request.app.state.pipeline  # Fetch pipeline from app state
    try:
        eval_model_dir = request.app.state.eval_model_dir
        config_path = eval_model_dir / "config.json"
        with open(config_path, "r") as f:
            config = json.load(f)

        pt_files = list(eval_model_dir.glob("*.pt"))
        checkpoint = torch.load(pt_files[0], map_location="cpu")

        classes = checkpoint.get("class_names", config.get("class_names", []))
        model_type = checkpoint.get("model_type", config.get("model_type", "cnn"))
        in_dim = checkpoint.get("in_dim", config.get("arch_param", 32))
        n_classes = checkpoint.get("n_classes", len(classes))
        layer_cfgs = config.get("layers", [])
        arch_param = config.get("arch_param", 32)

        model = _build_model(model_type, layer_cfgs, in_dim, n_classes)
        model.load_state_dict(checkpoint.get('model_state', checkpoint))
        model.eval()

        y_true = []
        y_pred = []

        for file in files:
            rel = file.filename or "unlabeled/file"
            parts = Path(rel).parts

            if len(parts) >= 2:
                true_class = parts[-2].lower()
            else:
                continue

            if true_class not in classes:
                continue

            true_idx = classes.index(true_class)

            session_id = pipeline.new_session()
            file_ext = Path(file.filename).suffix.lower()
            save_path = pipeline.upload_dir / f"{session_id}_batch{file_ext}"

            with open(save_path, "wb") as f:
                f.write(await file.read())

            try:
                pipeline.ingest(str(save_path))
                converted_files = list(pipeline.session_dir.glob("*.obj"))
                target_path = converted_files[0] if converted_files else save_path

                mesh = _load_mesh(str(target_path))

                if model_type == 'cnn':
                    feat = _voxelize(mesh, arch_param)
                    X = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                elif model_type == 'pointnet':
                    feat = _point_features(mesh, arch_param)
                    X = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
                else:
                    feat = _graph_features(mesh)
                    X = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)

                with torch.no_grad():
                    outputs = model(X)
                    pred_idx = torch.argmax(outputs, dim=1).item()

                y_true.append(true_idx)
                y_pred.append(pred_idx)

            except Exception as e:
                print(f"Failed to evaluate {file.filename}: {e}")
            finally:
                pipeline.cleanup_session()

        if len(y_true) == 0:
            raise ValueError("No valid files found belonging to the training classes.")

        return JSONResponse({
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average='macro', zero_division=0),
            "recall": recall_score(y_true, y_pred, average='macro', zero_division=0),
            "f1": f1_score(y_true, y_pred, average='macro', zero_division=0),
            "mse": mean_squared_error(y_true, y_pred)
        })

    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)
