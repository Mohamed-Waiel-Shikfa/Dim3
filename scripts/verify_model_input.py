import numpy as np
import os

# Set numpy to print cleanly without scientific notation
np.set_printoptions(precision=4, suppress=True, edgeitems=3, linewidth=100)

def verify_fruit_data(base_name="apple_040"):
    print(f"=========================================")
    print(f" 🔍 VERIFYING TENSORS FOR: {base_name}")
    print(f"=========================================\n")

    # ---------------------------------------------------------
    # 1. Verify Voxels
    # ---------------------------------------------------------
    voxel_path = f'data/fruit_voxels/{base_name}_model.npy'
    if os.path.exists(voxel_path):
        voxels = np.load(voxel_path)
        active_voxels = np.sum(voxels)
        total_voxels = voxels.size

        print("🧱 3D CNN (VOXELS)")
        print(f"  Shape: {voxels.shape}")
        print(f"  Density: {active_voxels} filled out of {total_voxels} total ({active_voxels/total_voxels:.2%})")

        # Show a 2D slice right through the middle of the 3D grid
        mid_z = voxels.shape[2] // 2
        print(f"  Middle Slice (Z={mid_z}) preview (0=Empty, 1=Filled):")
        # We print a small representation of the 2D plane
        print(f"  {voxels[..., mid_z]}")
        print("\n" + "-"*40 + "\n")
    else:
        print(f"❌ Missing Voxel file: {voxel_path}\n")

    # ---------------------------------------------------------
    # 2. Verify Point Cloud
    # ---------------------------------------------------------
    points_path = f'data/fruit_points/{base_name}_model.npy'
    if os.path.exists(points_path):
        points = np.load(points_path)

        print("☁️ POINTNET (POINT CLOUD)")
        print(f"  Shape: {points.shape} -> (Num Points, X/Y/Z)")
        print("  First 3 coordinates:")
        for i in range(3):
            print(f"    Point {i+1}: {points[i]}")
        print("\n" + "-"*40 + "\n")
    else:
        print(f"❌ Missing Point Cloud file: {points_path}\n")

    # ---------------------------------------------------------
    # 3. Verify Graph
    # ---------------------------------------------------------
    graph_path = f'data/fruit_graph/{base_name}_model.npz'
    if os.path.exists(graph_path):
        graph_data = np.load(graph_path)
        nodes = graph_data['x']
        edges = graph_data['edge_index']
        weights = graph_data['edge_attr']

        print("🕸️ GNN (GRAPH)")
        print(f"  Nodes Shape:   {nodes.shape} -> (Total Vertices, 3D Coords)")
        print(f"  Edges Shape:   {edges.shape} -> (2, Total Connections)")
        print(f"  Weights Shape: {weights.shape} -> (Total Connections,)")

        print("\n  First 3 Graph Connections:")
        for i in range(3):
            node_a = edges[0, i]
            node_b = edges[1, i]
            dist = weights[i]
            print(f"    Node {node_a:<4} is connected to Node {node_b:<4} | Euclidean Distance: {dist:.4f}")
        print("\n=========================================\n")
    else:
        print(f"❌ Missing Graph file: {graph_path}\n")

if __name__ == "__main__":
    # You can change this string to "banan" or "Kiwi" to inspect others
    verify_fruit_data("apple_040")
