import csv
import json
import time
import queue
import threading
import numpy as np
import torch
import torch.nn as nn
import trimesh
from pathlib import Path

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

_ACT_NAMES = ['ReLU', 'LeakyReLU', 'Sigmoid', 'Tanh', 'Softmax', 'None']


def _get_act(idx):
    name = _ACT_NAMES[min(int(idx), len(_ACT_NAMES) - 1)]
    if name == 'ReLU':      return nn.ReLU()
    if name == 'LeakyReLU': return nn.LeakyReLU()
    if name == 'Sigmoid':   return nn.Sigmoid()
    if name == 'Tanh':      return nn.Tanh()
    if name == 'Softmax':   return nn.Softmax(dim=-1)
    return None


def _memory_mb():
    if _HAS_PSUTIL:
        return psutil.Process().memory_info().rss / (1024 * 1024)
    return 0.0


def _load_mesh(path):
    mesh = trimesh.load(str(path), force='mesh')
    mesh.vertices -= mesh.vertices.mean(axis=0)
    mx = mesh.extents.max()
    if mx > 0:
        mesh.vertices *= 2.0 / mx
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.fix_normals()
    return mesh


def _random_rotate(verts):
    a = np.random.uniform(0, 2 * np.pi, 3)
    cx, sx = np.cos(a[0]), np.sin(a[0])
    cy, sy = np.cos(a[1]), np.sin(a[1])
    cz, sz = np.cos(a[2]), np.sin(a[2])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return verts @ (Rz @ Ry @ Rx).T


def _voxelize(mesh, grid_size):
    mx = mesh.extents.max()
    if mx == 0:
        return np.zeros((grid_size,) * 3, dtype=np.float32)
    try:
        vox = mesh.voxelized(pitch=mx / grid_size)
        mat = vox.matrix.astype(np.float32)
    except Exception:
        return np.zeros((grid_size,) * 3, dtype=np.float32)
    out = np.zeros((grid_size,) * 3, dtype=np.float32)
    ms, gs = np.array(mat.shape), np.array([grid_size] * 3)
    mn = np.minimum(ms, gs)
    si, oi = (gs - mn) // 2, (ms - mn) // 2
    out[si[0]:si[0]+mn[0], si[1]:si[1]+mn[1], si[2]:si[2]+mn[2]] = \
        mat[oi[0]:oi[0]+mn[0], oi[1]:oi[1]+mn[1], oi[2]:oi[2]+mn[2]]
    return out


def _point_features(mesh, n_points):
    from scipy.spatial import KDTree
    dense, _ = trimesh.sample.sample_surface(mesh, max(n_points * 5, 2000))
    if len(dense) > n_points:
        sel = [int(np.random.randint(len(dense)))]
        dist = np.linalg.norm(dense - dense[sel[0]], axis=1)
        for _ in range(n_points - 1):
            sel.append(int(np.argmax(dist)))
            dist = np.minimum(dist, np.linalg.norm(dense - dense[sel[-1]], axis=1))
        pts = dense[sel]
    else:
        pts = dense

    feats = []
    for ax in range(3):
        col = pts[:, ax]
        feats += [col.mean(), col.std(), col.min(), col.max()]  # 12

    cov = np.cov(pts.T)
    ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
    feats += list(ev / (ev.sum() + 1e-9))  # 3

    bb = pts.max(axis=0) - pts.min(axis=0)
    feats += list(bb)  # 3

    tree = KDTree(pts)
    dd, _ = tree.query(pts, k=2)
    feats.append(float(dd[:, 1].mean()))  # 1

    vol = abs(mesh.volume) if mesh.is_watertight else (bb.prod() + 1e-9)
    sa = mesh.area + 1e-9
    feats.append(float(sa / (vol ** (2 / 3) + 1e-9)))      # sphericity
    feats.append(float(vol / (bb.prod() + 1e-9)))           # compactness
    feats.append(float(bb.max() / (bb.min() + 1e-9)))       # elongation
    feats.append(float(bb.min() / (bb.max() + 1e-9)))       # flatness  → 23

    c = pts.mean(axis=0)
    cd = np.linalg.norm(pts - c, axis=1)
    feats += [float(cd.mean()), float(cd.std())]            # 25

    fn_std = float(mesh.face_normals[:, 2].std()) if len(mesh.face_normals) > 0 else 0.0
    feats.append(fn_std)                                    # 26

    return np.array(feats, dtype=np.float32)


def _graph_features(mesh):
    nodes, edges = mesh.vertices, mesh.edges_unique
    n, e = len(nodes), len(edges)
    if n == 0 or e == 0:
        return np.zeros(16, dtype=np.float32)
    deg = np.zeros(n, dtype=int)
    for s, t in edges:
        deg[s] += 1; deg[t] += 1
    el = np.linalg.norm(nodes[edges[:, 0]] - nodes[edges[:, 1]], axis=1)
    max_e = n * (n - 1) / 2
    return np.array([
        float(deg.mean()), float(deg.std()), float(deg.min()), float(deg.max()),
        float(np.median(deg)),
        float((deg == 1).sum()) / (n + 1e-9),
        float(el.mean()), float(el.std()), float(el.min()), float(el.max()),
        float(e) / (n + 1e-9),
        float(e) / (max_e + 1e-9),
        float(n), float(e),
        float(e) / (n * np.log(n + 2)),
        float(deg.mean()) / (float(deg.max()) + 1e-9),
    ], dtype=np.float32)


def _build_model(model_type, layer_configs, in_dim_or_grid, n_classes):
    if model_type == 'cnn':
        grid = in_dim_or_grid
        mods, ch, sp, cur = [], 1, grid, None
        for cfg in layer_configs:
            typ = cfg['type']
            n = max(1, int(cfg.get('nodes', 64)))
            act = _get_act(cfg.get('actIndex', 0))
            if cur is None:
                if typ == 'Conv3D':
                    mods.append(nn.Conv3d(ch, n, 3, padding=1))
                    if act: mods.append(act)
                    ch = n
                elif typ == 'MaxPool' and sp > 1:
                    mods.append(nn.MaxPool3d(2))
                    sp = max(1, sp // 2)
                elif typ == 'Linear':
                    cur = ch * sp * sp * sp
                    mods += [nn.Flatten(), nn.Linear(cur, n)]
                    if act: mods.append(act)
                    cur = n
            elif typ == 'Linear':
                mods.append(nn.Linear(cur, n))
                if act: mods.append(act)
                cur = n
        if cur is None:
            cur = ch * sp * sp * sp
            mods.append(nn.Flatten())
        mods.append(nn.Linear(cur, n_classes))
        return nn.Sequential(*mods)
    else:
        dim = in_dim_or_grid
        mods = []
        for cfg in layer_configs:
            n = max(1, int(cfg.get('nodes', 64)))
            act = _get_act(cfg.get('actIndex', 0))
            mods.append(nn.Linear(dim, n))
            if act: mods.append(act)
            dim = n
        mods.append(nn.Linear(dim, n_classes))
        return nn.Sequential(*mods)


def run_training(job_id, session_dir, files_by_class, config, progress_q, pause_event, stop_event):
    try:
        model_type   = config.get('model_type', 'cnn')
        arch_param   = max(4, int(config.get('arch_param', 32)))
        epochs       = max(1, int(config.get('epochs', 10)))
        batch_size   = max(1, int(config.get('batch_size', 16)))
        lr           = float(config.get('lr', 0.001))
        momentum     = float(config.get('momentum', 0.9))
        val_split    = float(config.get('val_split', 20)) / 100.0
        rotations    = max(1, int(config.get('rotations', 3)))
        loss_fn_name = config.get('loss_fn', 'CrossEntropyLoss')
        layer_cfgs   = config.get('layers', [])

        class_names = sorted(files_by_class.keys())
        n_classes   = len(class_names)
        label_map   = {c: i for i, c in enumerate(class_names)}

        if n_classes < 2:
            progress_q.put({'type': 'error', 'message': 'Need at least 2 classes (subdirectories) to train.'})
            return

        # Persist training config for the evaluation page
        sd = Path(session_dir)
        config_save = {**config, 'class_names': class_names, 'n_classes': n_classes}
        (sd / f'config_{job_id}.json').write_text(json.dumps(config_save, indent=2))

        metrics_path = sd / f'metrics_{job_id}.csv'
        with open(metrics_path, 'w', newline='') as mf:
            csv.writer(mf).writerow(['epoch', 'train_loss', 'val_loss', 'accuracy', 'memory_mb'])

        all_paths = [(cls, p) for cls, paths in files_by_class.items() for p in paths]
        total_files = len(all_paths)
        X_list, y_list = [], []

        for idx, (cls, path) in enumerate(all_paths):
            if stop_event.is_set():
                return
            progress_q.put({
                'type': 'extracting',
                'current': idx + 1,
                'total': total_files,
                'file': Path(path).name,
                'progress': float(idx) / total_files * 30.0,
            })
            try:
                mesh  = _load_mesh(path)
                label = label_map[cls]
                for rot in range(rotations):
                    if model_type == 'cnn':
                        m2 = mesh.copy()
                        if rot > 0:
                            m2.vertices = _random_rotate(mesh.vertices.copy())
                        feat = _voxelize(m2, arch_param)
                    elif model_type == 'pointnet':
                        m2 = mesh.copy()
                        if rot > 0:
                            m2.vertices = _random_rotate(mesh.vertices.copy())
                        feat = _point_features(m2, arch_param)
                    else:
                        feat = _graph_features(mesh)  # rotation-invariant
                    X_list.append(feat)
                    y_list.append(label)
            except Exception as exc:
                progress_q.put({'type': 'log', 'message': f'Skipped {Path(path).name}: {exc}'})

        if not X_list:
            progress_q.put({'type': 'error', 'message': 'No valid meshes could be processed.'})
            return

        if model_type == 'cnn':
            X = torch.tensor(np.stack(X_list), dtype=torch.float32).unsqueeze(1)
            in_dim = arch_param
        else:
            X = torch.tensor(np.stack(X_list), dtype=torch.float32)
            in_dim = X.shape[1]
        y = torch.tensor(y_list, dtype=torch.long)

        n      = len(X)
        n_val  = max(0, min(int(n * val_split), n - n_classes))
        perm   = torch.randperm(n)
        X_tr, y_tr = X[perm[n_val:]], y[perm[n_val:]]
        X_val, y_val = X[perm[:n_val]], y[perm[:n_val]]

        feature_mean = feature_std = None
        if model_type != 'cnn' and len(X_tr) > 0:
            feature_mean = X_tr.mean(0)
            feature_std  = X_tr.std(0) + 1e-8
            X_tr  = (X_tr  - feature_mean) / feature_std
            if len(X_val) > 0:
                X_val = (X_val - feature_mean) / feature_std

        model = _build_model(model_type, layer_cfgs, in_dim, n_classes)
        optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

        ce   = nn.CrossEntropyLoss()
        mse  = nn.MSELoss()
        nll  = nn.NLLLoss()

        def compute_loss(out, yb):
            if loss_fn_name == 'MSELoss':
                oh = torch.zeros(len(yb), n_classes)
                oh.scatter_(1, yb.unsqueeze(1), 1)
                return mse(out, oh)
            if loss_fn_name == 'NLLLoss':
                return nll(torch.log_softmax(out, dim=-1), yb)
            return ce(out, yb)

        for epoch in range(1, epochs + 1):
            while pause_event.is_set():
                if stop_event.is_set(): return
                time.sleep(0.1)
            if stop_event.is_set(): return

            model.train()
            perm2   = torch.randperm(len(X_tr))
            tr_loss = 0.0
            correct = 0

            for i in range(0, len(X_tr), batch_size):
                bi  = perm2[i:i+batch_size]
                xb, yb = X_tr[bi], y_tr[bi]
                optim.zero_grad()
                out  = model(xb)
                loss = compute_loss(out, yb)
                loss.backward()
                optim.step()
                tr_loss += loss.item() * len(xb)
                correct += (out.argmax(1) == yb).sum().item()

            tr_loss /= max(len(X_tr), 1)
            acc      = correct / max(len(X_tr), 1)

            v_loss = tr_loss
            if len(X_val) > 0:
                model.eval()
                with torch.no_grad():
                    v_loss = compute_loss(model(X_val), y_val).item()

            mem = _memory_mb()
            with open(metrics_path, 'a', newline='') as mf:
                csv.writer(mf).writerow([epoch, f'{tr_loss:.6f}', f'{v_loss:.6f}', f'{acc:.6f}', f'{mem:.1f}'])

            progress_q.put({
                'type':       'epoch',
                'epoch':      epoch,
                'total':      epochs,
                'train_loss': float(tr_loss),
                'val_loss':   float(v_loss),
                'accuracy':   float(acc),
                'progress':   30.0 + 70.0 * epoch / epochs,
                'memory_mb':  mem,
            })

        model_path = sd / f'model_{job_id}.pt'
        torch.save({
            'model_state':  model.state_dict(),
            'class_names':  class_names,
            'model_type':   model_type,
            'in_dim':       in_dim,
            'n_classes':    n_classes,
            'feature_mean': feature_mean.tolist() if feature_mean is not None else None,
            'feature_std':  feature_std.tolist()  if feature_std  is not None else None,
        }, str(model_path))
        progress_q.put({'type': 'completed', 'job_id': job_id})

    except Exception as exc:
        import traceback
        progress_q.put({'type': 'error', 'message': f'{exc}\n{traceback.format_exc()}'})
