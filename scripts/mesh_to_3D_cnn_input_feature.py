import os
import argparse
import numpy as np
import trimesh

def pad_or_crop_volume(voxel_matrix, target_size):
    """Ensures the 3D matrix is exactly (target_size x target_size x target_size)."""
    current_size = np.array(voxel_matrix.shape)
    result = np.zeros((target_size, target_size, target_size), dtype=np.int8)

    min_dims = np.minimum(current_size, target_size)
    start_idx = (target_size - min_dims) // 2
    orig_start = (current_size - min_dims) // 2

    result[start_idx[0]:start_idx[0]+min_dims[0],
           start_idx[1]:start_idx[1]+min_dims[1],
           start_idx[2]:start_idx[2]+min_dims[2]] = \
    voxel_matrix[orig_start[0]:orig_start[0]+min_dims[0],
                 orig_start[1]:orig_start[1]+min_dims[1],
                 orig_start[2]:orig_start[2]+min_dims[2]]
    return result

def process_voxels(input_dir, output_dir, grid_size):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.endswith('.obj'): continue

        filepath = os.path.join(input_dir, filename)
        mesh = trimesh.load(filepath, force='mesh')
        base_name = os.path.splitext(filename)[0]

        print(f"Voxelizing {base_name}...")

        # Calculate Pitch
        max_extent = mesh.extents.max()
        pitch = max_extent / grid_size

        # Voxelize (Trimesh defaults to surface/hollow voxelization automatically)
        voxels = mesh.voxelized(pitch=pitch)

        # 1. Output for Model (Numpy Array)
        matrix = voxels.matrix.astype(np.int8)
        final_tensor = pad_or_crop_volume(matrix, grid_size)
        np.save(os.path.join(output_dir, f"{base_name}_model.npy"), final_tensor)

        # 2. Output for Human (3D Cubes OBJ)
        voxel_mesh = voxels.as_boxes()
        voxel_mesh.export(os.path.join(output_dir, f"{base_name}_human.obj"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--grid", type=int, default=32)
    args = parser.parse_args()

    process_voxels(args.input, args.output, args.grid)
