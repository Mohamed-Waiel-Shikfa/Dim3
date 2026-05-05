import json
import numpy as np
import torch
from pathlib import Path

from .trainer import _load_mesh, _voxelize, _point_features, _graph_features, _build_model


class ModelSession:
    def __init__(self, model_dir):
        model_dir = Path(model_dir)

        config_files = sorted(model_dir.glob("config*.json"))
        if not config_files:
            raise FileNotFoundError(f"config.json not found in {model_dir}")
        with open(config_files[0]) as f:
            self.config = json.load(f)

        model_files = sorted(model_dir.glob("model*.pt"))
        if not model_files:
            raise FileNotFoundError(f"model.pt not found in {model_dir}")
        ckpt = torch.load(str(model_files[0]), map_location="cpu")

        self.class_names  = ckpt["class_names"]
        self.model_type   = ckpt["model_type"]
        self.in_dim       = ckpt["in_dim"]
        self.n_classes    = ckpt["n_classes"]
        self.arch_param   = self.config.get("arch_param", 32)

        fm = ckpt.get("feature_mean")
        fs = ckpt.get("feature_std")
        self.feature_mean = np.array(fm, dtype=np.float32) if fm else None
        self.feature_std  = np.array(fs, dtype=np.float32) if fs else None

        self.model = _build_model(
            self.model_type,
            self.config.get("layers", []),
            self.in_dim,
            self.n_classes,
        )
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

    def _extract(self, path):
        mesh = _load_mesh(path)
        if self.model_type == "cnn":
            feat = _voxelize(mesh, self.in_dim)
            return torch.tensor(feat, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        feat = _point_features(mesh, self.arch_param) if self.model_type == "pointnet" \
               else _graph_features(mesh)
        if self.feature_mean is not None:
            feat = (feat - self.feature_mean) / (self.feature_std + 1e-8)
        return torch.tensor(feat, dtype=torch.float32).unsqueeze(0)

    def predict(self, path):
        with torch.no_grad():
            logits = self.model(self._extract(path))
            probs  = torch.softmax(logits, dim=-1).squeeze().tolist()
        if isinstance(probs, float):
            probs = [probs]
        return {cls: float(p) for cls, p in zip(self.class_names, probs)}

    def evaluate(self, files_by_class, progress_cb=None):
        label_map  = {c: i for i, c in enumerate(self.class_names)}
        all_true, all_pred = [], []
        total = sum(len(v) for v in files_by_class.values())
        done  = 0

        for cls, paths in files_by_class.items():
            true_idx = label_map.get(cls)
            if true_idx is None:
                continue
            for path in paths:
                try:
                    probs    = self.predict(path)
                    pred_idx = max(range(self.n_classes), key=lambda i: probs.get(self.class_names[i], 0))
                    all_true.append(true_idx)
                    all_pred.append(pred_idx)
                except Exception:
                    pass
                done += 1
                if progress_cb:
                    progress_cb(done, total)

        n = len(all_true)
        if n == 0:
            return {"error": "No files could be processed"}

        accuracy = sum(t == p for t, p in zip(all_true, all_pred)) / n

        cm = np.zeros((self.n_classes, self.n_classes), dtype=int)
        for t, p in zip(all_true, all_pred):
            cm[t][p] += 1

        per_class = {}
        for i, cls in enumerate(self.class_names):
            tp = int(cm[i][i])
            fp = int(cm[:, i].sum() - tp)
            fn = int(cm[i].sum() - tp)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            per_class[cls] = {
                "precision": round(prec, 4),
                "recall":    round(rec,  4),
                "f1":        round(f1,   4),
                "support":   int(cm[i].sum()),
                "correct":   tp,
            }

        return {
            "accuracy":         round(float(accuracy), 4),
            "confusion_matrix": cm.tolist(),
            "class_names":      self.class_names,
            "per_class":        per_class,
            "total_samples":    n,
        }
