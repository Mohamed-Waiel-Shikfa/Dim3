# =============================================================================
# DIM3 — Processing Pipeline
# Handles mesh ingestion, normalization, voxelization, point cloud sampling,
# and graph extraction using trimesh (no Blender dependency for the web server).
# =============================================================================
import os
import json
import shutil
import uuid
import numpy as np
import trimesh
from pathlib import Path
import platform


class ProcessingPipeline:
    """Stateful pipeline for a single uploaded model."""

    def __init__(self, upload_dir: str, output_dir: str):
        self.upload_dir = Path(upload_dir)
        self.output_dir = Path(output_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = None
        self.session_dir = None

    def new_session(self) -> str:
        self.session_id = str(uuid.uuid4())[:8]
        self.session_dir = self.output_dir / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        return self.session_id

    def _get_session_dir(self):
        if not self.session_dir:
            raise ValueError("No active session. Call new_session() first.")
        return self.session_dir

    def _find_blender(self) -> list:
        """Finds the Blender executable based on the operating system."""
        system = platform.system()

        # 1. Check if 'blender' is already in the system PATH
        path_executable = shutil.which("blender")
        if path_executable:
            return [path_executable, "--background", "--python"]

        # 2. Fallback to common OS-specific installation paths
        if system == "Darwin":  # macOS
            mac_path = "/Applications/Blender.app/Contents/MacOS/Blender"
            if Path(mac_path).exists():
                return [mac_path, "--background", "--python"]

        elif system == "Windows":
            win_path = r"C:\Program Files\Blender Foundation\Blender\blender.exe"
            if Path(win_path).exists():
                return [win_path, "--background", "--python"]

        elif system == "Linux":
            linux_path = "/usr/bin/blender"
            if Path(linux_path).exists():
                return [linux_path, "--background", "--python"]

        # 3. Last resort/Error handling
        raise EnvironmentError(
            "Blender executable not found. Please ensure it is installed and in your PATH."
        )

    # ── Step 1: Ingest ──────────────────────────────────────────────────────
    def ingest(self, file_path: str) -> dict:
        import subprocess
        sd = self._get_session_dir()

        orig_name = Path(file_path).stem.split("_", 1)[-1]
        cleaned_obj = sd / f"{orig_name}_cleaned.obj"
        high_poly_obj = sd / f"{orig_name}_remesh_high.obj"
        low_poly_obj = sd / f"{orig_name}_remesh_low.obj"

        # Dynamically determine the command based on OS
        BLENDER_CMD = self._find_blender()

        # Dynamically handle S_ROOT (relative to the script location)
        S_ROOT = Path(__file__).parent.absolute() / "../../../scripts"

        print("Running 3d_file_to_obj...")
        subprocess.run(BLENDER_CMD + [f"{S_ROOT}/3d_file_to_obj.py", "--", "--input_file", str(file_path), "--output_file", str(cleaned_obj)], check=True)

        print("Running mesh_cleanup HIGH...")
        subprocess.run(BLENDER_CMD + [f"{S_ROOT}/mesh_cleanup.py", "--", "--input_file", str(cleaned_obj), "--output_file", str(high_poly_obj), "--voxel_size", "0.01"], check=True)

        print("Running mesh_cleanup LOW...")
        subprocess.run(BLENDER_CMD + [f"{S_ROOT}/mesh_cleanup.py", "--", "--input_file", str(cleaned_obj), "--output_file", str(low_poly_obj), "--voxel_size", "0.05"], check=True)

        return {
            "obj_url": f"/data/{self.session_id}/{orig_name}_cleaned.obj",
            "high_poly_url": f"/data/{self.session_id}/{orig_name}_remesh_high.obj",
            "low_poly_url": f"/data/{self.session_id}/{orig_name}_remesh_low.obj"
        }

    # ── Step 2: Normalize ───────────────────────────────────────────────────
    def normalize(self, voxel_size: float = 0.01) -> dict:
        """Center, scale to 2x2x2, and optionally remesh."""
        sd = self._get_session_dir()
        in_path = sd / "ingested.obj"

        mesh = trimesh.load(str(in_path), force='mesh')

        # Center at origin
        mesh.vertices -= mesh.centroid

        # Scale to fit 2x2x2
        max_dim = mesh.extents.max()
        if max_dim > 0:
            mesh.vertices *= 2.0 / max_dim

        # Clean up
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.merge_vertices()
        mesh.fix_normals()

        out_path = sd / "normalized.obj"
        mesh.export(str(out_path), file_type='obj')

        return {
            "obj_url": f"/data/{self.session_id}/normalized.obj",
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),
            "bounds": mesh.bounds.tolist(),
            "is_watertight": mesh.is_watertight,
        }

    # ── Step 3: Voxelize (CNN) ──────────────────────────────────────────────
    def voxelize(self, grid_size: int = 32) -> dict:
        """Create voxel grid from normalized mesh."""
        sd = self._get_session_dir()
        in_path = sd / "normalized.obj"

        mesh = trimesh.load(str(in_path), force='mesh')

        max_extent = mesh.extents.max()
        pitch = max_extent / grid_size
        voxels = mesh.voxelized(pitch=pitch)

        # Get the occupancy matrix
        matrix = voxels.matrix.astype(np.int8)

        # Pad/crop to exact grid_size
        result = self._pad_or_crop(matrix, grid_size)

        # Save matrix as JSON-friendly format
        voxel_data = {
            "matrix": result.flatten().tolist(),
            "shape": [grid_size, grid_size, grid_size],
            "origin": voxels.transform[:3, 3].tolist() if hasattr(voxels, 'transform') else [0, 0, 0],
            "pitch": float(pitch),
            "filled_count": int(result.sum()),
            "total_count": int(result.size),
            "density": float(result.sum() / result.size),
        }

        # Save as JSON
        json_path = sd / "voxels.json"
        with open(json_path, 'w') as f:
            json.dump(voxel_data, f)

        # Also export the voxel mesh for 3D viewing
        try:
            voxel_mesh = voxels.as_boxes()
            voxel_obj_path = sd / "voxels.obj"
            voxel_mesh.export(str(voxel_obj_path), file_type='obj')
            voxel_data["voxel_obj_url"] = f"/data/{self.session_id}/voxels.obj"
        except Exception:
            pass

        voxel_data["json_url"] = f"/data/{self.session_id}/voxels.json"
        voxel_data["obj_url"] = f"/data/{self.session_id}/normalized.obj"

        return voxel_data

    # ── Step 4: Sample Point Cloud ──────────────────────────────────────────
    def sample_points(self, num_points: int = 1024, method: str = "fps") -> dict:
        """Sample point cloud from normalized mesh."""
        sd = self._get_session_dir()
        in_path = sd / "normalized.obj"

        mesh = trimesh.load(str(in_path), force='mesh')

        # Dense pre-sampling
        dense_count = max(num_points * 10, 20000)
        dense_points, face_indices = trimesh.sample.sample_surface(mesh, count=dense_count)

        if method == "fps":
            points = self._farthest_point_sampling(dense_points, num_points)
            method_desc = (
                "Farthest Point Sampling (FPS) guarantees maximum spatial coverage by "
                "iteratively selecting the point farthest from all previously chosen points. "
                "This produces an evenly distributed subset ideal for uniform representation."
            )
        elif method == "poisson":
            points = self._poisson_disk_sampling(dense_points, num_points)
            method_desc = (
                "Poisson Disk Sampling creates a blue-noise distribution where no two points "
                "are closer than a minimum radius. This mimics natural sampling patterns and "
                "avoids clustering artifacts common in random sampling."
            )
        elif method == "curvature":
            points = self._curvature_sampling(mesh, dense_points, face_indices, num_points)
            method_desc = (
                "Curvature-Based Sampling concentrates more points in high-curvature regions "
                "(edges, corners, fine details) while using fewer in flat areas. This preserves "
                "geometric detail where it matters most for shape understanding."
            )
        else:
            points = dense_points[:num_points]
            method_desc = "Random uniform surface sampling."

        points_list = points.tolist()

        point_data = {
            "points": points_list,
            "count": len(points_list),
            "method": method,
            "method_description": method_desc,
            "obj_url": f"/data/{self.session_id}/normalized.obj",
        }

        json_path = sd / "points.json"
        with open(json_path, 'w') as f:
            json.dump(point_data, f)

        point_data["json_url"] = f"/data/{self.session_id}/points.json"
        return point_data

    # ── Step 5: Extract Graph (GNN) ─────────────────────────────────────────
    def extract_graph(self) -> dict:
        """Extract graph topology from the low-poly mesh."""
        sd = self._get_session_dir()
        in_path = sd / "low_poly.obj"
        if not in_path.exists():
            in_path = sd / "normalized.obj"

        mesh = trimesh.load(str(in_path), force='mesh')

        nodes = mesh.vertices.astype(np.float32)
        edges = mesh.edges_unique
        edge_weights = np.linalg.norm(
            nodes[edges[:, 0]] - nodes[edges[:, 1]], axis=1
        ).tolist()

        # Export wireframe OBJ (edges only, no faces)
        wireframe_path = sd / "wireframe.obj"
        with open(wireframe_path, 'w') as f:
            for v in nodes:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for e in edges:
                f.write(f"l {e[0]+1} {e[1]+1}\n")

        graph_data = {
            "nodes": nodes.tolist(),
            "edges": edges.tolist(),
            "weights": edge_weights,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "wireframe_url": f"/data/{self.session_id}/wireframe.obj",
            "obj_url": f"/data/{self.session_id}/normalized.obj",
        }

        json_path = sd / "graph.json"
        with open(json_path, 'w') as f:
            json.dump(graph_data, f)

        graph_data["json_url"] = f"/data/{self.session_id}/graph.json"
        return graph_data

    # ── GNN Step: Normalize with large voxel (low poly) ─────────────────────
    def normalize_low_poly(self, voxel_size: float = 0.05) -> dict:
        """Create a low-poly version for GNN (larger voxel size)."""
        sd = self._get_session_dir()
        in_path = sd / "ingested.obj"

        mesh = trimesh.load(str(in_path), force='mesh')

        # Center and scale
        mesh.vertices -= mesh.centroid
        max_dim = mesh.extents.max()
        if max_dim > 0:
            mesh.vertices *= 2.0 / max_dim

        # Simplify via vertex clustering (no external dep)
        target_faces = max(200, len(mesh.faces) // 10)
        if len(mesh.faces) > target_faces:
            # Use voxel-based vertex clustering as a decimation fallback
            pitch = mesh.extents.max() / 30  # ~30 voxel divisions = low poly
            vox = mesh.voxelized(pitch)
            try:
                mesh = vox.marching_cubes
            except Exception:
                pass  # Keep original if marching cubes fails

        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.fix_normals()

        out_path = sd / "low_poly.obj"
        mesh.export(str(out_path), file_type='obj')

        return {
            "obj_url": f"/data/{self.session_id}/low_poly.obj",
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),
        }

    # ── GNN Step: Remove faces (wireframe only) ─────────────────────────────
    def extract_wireframe(self) -> dict:
        """Extract edges-only from the low-poly mesh."""
        sd = self._get_session_dir()
        in_path = sd / "low_poly.obj"

        mesh = trimesh.load(str(in_path), force='mesh')
        nodes = mesh.vertices.astype(np.float32)
        edges = mesh.edges_unique

        wireframe_path = sd / "wireframe.obj"
        with open(wireframe_path, 'w') as f:
            for v in nodes:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for e in edges:
                f.write(f"l {e[0]+1} {e[1]+1}\n")

        return {
            "wireframe_url": f"/data/{self.session_id}/wireframe.obj",
            "node_count": len(nodes),
            "edge_count": len(edges),
        }

    # ── Helpers ──────────────────────────────────────────────────────────────
    @staticmethod
    def _pad_or_crop(matrix, target_size):
        current_size = np.array(matrix.shape)
        result = np.zeros((target_size, target_size, target_size), dtype=np.int8)
        min_dims = np.minimum(current_size, target_size)
        start_idx = (target_size - min_dims) // 2
        orig_start = (current_size - min_dims) // 2
        result[
            start_idx[0]:start_idx[0]+min_dims[0],
            start_idx[1]:start_idx[1]+min_dims[1],
            start_idx[2]:start_idx[2]+min_dims[2]
        ] = matrix[
            orig_start[0]:orig_start[0]+min_dims[0],
            orig_start[1]:orig_start[1]+min_dims[1],
            orig_start[2]:orig_start[2]+min_dims[2]
        ]
        return result

    @staticmethod
    def _farthest_point_sampling(points, n_samples):
        if len(points) <= n_samples:
            return points
        selected = np.zeros((n_samples, 3))
        selected[0] = points[np.random.randint(len(points))]
        distances = np.linalg.norm(points - selected[0], axis=1)
        for i in range(1, n_samples):
            selected[i] = points[np.argmax(distances)]
            distances = np.minimum(distances, np.linalg.norm(points - selected[i], axis=1))
        return selected

    @staticmethod
    def _poisson_disk_sampling(points, n_samples):
        """Approximate Poisson disk sampling via dart throwing."""
        if len(points) <= n_samples:
            return points

        # Estimate minimum radius
        from scipy.spatial import KDTree
        tree = KDTree(points)
        # Find radius that gives approximately the right number of points
        volume = np.prod(points.max(axis=0) - points.min(axis=0))
        radius = (volume / n_samples) ** (1/3) * 0.5

        selected = []
        indices = np.random.permutation(len(points))

        for idx in indices:
            if len(selected) >= n_samples:
                break
            pt = points[idx]
            if len(selected) == 0:
                selected.append(pt)
            else:
                dists = np.linalg.norm(np.array(selected) - pt, axis=1)
                if dists.min() >= radius:
                    selected.append(pt)

        # If not enough, relax and add random
        while len(selected) < n_samples:
            idx = np.random.randint(len(points))
            selected.append(points[idx])

        return np.array(selected[:n_samples])

    @staticmethod
    def _curvature_sampling(mesh, dense_points, face_indices, n_samples):
        """Sample more points in high-curvature regions."""
        if len(dense_points) <= n_samples:
            return dense_points

        # Estimate per-face curvature using face normals and adjacency
        try:
            face_normals = mesh.face_normals
            # Use face adjacency to estimate curvature
            adjacency = mesh.face_adjacency
            adj_angles = mesh.face_adjacency_angles

            face_curvature = np.zeros(len(mesh.faces))
            for i, (f1, f2) in enumerate(adjacency):
                c = adj_angles[i]
                face_curvature[f1] += c
                face_curvature[f2] += c

            # Map curvature to sample points
            point_curvature = face_curvature[face_indices]

            # Normalize to probability distribution
            point_curvature = point_curvature - point_curvature.min()
            point_curvature = point_curvature ** 2  # Emphasize high curvature
            probs = point_curvature / (point_curvature.sum() + 1e-10)

            chosen_indices = np.random.choice(
                len(dense_points), size=n_samples, replace=False, p=probs
            )
            return dense_points[chosen_indices]

        except Exception:
            # Fallback to FPS if curvature estimation fails
            return ProcessingPipeline._farthest_point_sampling(dense_points, n_samples)

    def cleanup_session(self):
        """Remove session directory."""
        if self.session_dir and self.session_dir.exists():
            shutil.rmtree(self.session_dir)
