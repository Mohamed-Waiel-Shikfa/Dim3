import os
import argparse
import numpy as np
import trimesh

def process_graphs(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.endswith('.obj'): continue

        filepath = os.path.join(input_dir, filename)
        mesh = trimesh.load(filepath, force='mesh')
        base_name = os.path.splitext(filename)[0]

        print(f"Extracting Graph for {base_name}...")

        # Extract Vertices (Nodes) and Edges
        nodes = mesh.vertices.astype(np.float32)
        edges = mesh.edges_unique

        # Calculate Edge Weights (Euclidean Distance)
        edge_attr = np.linalg.norm(nodes[edges[:, 0]] - nodes[edges[:, 1]], axis=1).astype(np.float32)

        # 1. Output for Model (NPZ Archive)
        # We transpose edges to shape [2, num_edges] as this is the standard PyTorch Geometric format
        np.savez(os.path.join(output_dir, f"{base_name}_model.npz"),
                 x=nodes,
                 edge_index=edges.T,
                 edge_attr=edge_attr)

        # 2. Output for Human (Wireframe OBJ)
        # Manually writing a line-only OBJ so it displays as a pure graph
        human_out_path = os.path.join(output_dir, f"{base_name}_human.obj")
        with open(human_out_path, 'w') as f:
            # Write nodes
            for v in nodes:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            # Write edges (OBJ files are 1-indexed, so we add 1 to the arrays)
            for e in edges:
                f.write(f"l {e[0]+1} {e[1]+1}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    process_graphs(args.input, args.output)
