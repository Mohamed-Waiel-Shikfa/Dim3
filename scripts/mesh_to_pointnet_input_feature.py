import os
import argparse
import numpy as np
import trimesh

def farthest_point_sampling(points, n_samples):
    """FPS algorithm to guarantee even spatial coverage."""
    farthest_pts = np.zeros((n_samples, 3))
    farthest_pts[0] = points[np.random.randint(len(points))]
    distances = np.linalg.norm(points - farthest_pts[0], axis=1)

    for i in range(1, n_samples):
        farthest_pts[i] = points[np.argmax(distances)]
        distances = np.minimum(distances, np.linalg.norm(points - farthest_pts[i], axis=1))

    return farthest_pts

def process_pointclouds(input_dir, output_dir, num_points):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.endswith('.obj'): continue

        filepath = os.path.join(input_dir, filename)
        mesh = trimesh.load(filepath, force='mesh')
        base_name = os.path.splitext(filename)[0]

        print(f"Sampling Point Cloud for {base_name}...")

        # Pre-sample dense cloud
        dense_points, _ = trimesh.sample.sample_surface(mesh, count=20000)

        # Downsample via FPS
        final_points = farthest_point_sampling(dense_points, num_points)

        # 1. Output for Model (Numpy Array)
        np.save(os.path.join(output_dir, f"{base_name}_model.npy"), final_points)

        # 2. Output for Human (PLY File)
        pc = trimesh.PointCloud(final_points)
        pc.export(os.path.join(output_dir, f"{base_name}_human.ply"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--points", type=int, default=4096)
    args = parser.parse_args()

    process_pointclouds(args.input, args.output, args.points)
