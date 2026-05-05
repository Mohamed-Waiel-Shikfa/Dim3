import os
import trimesh
import numpy as np
import scipy.spatial.transform as st
import random
import re
import sys # Added for CLI args
import shutil # Moved to top for consistency
from tqdm import tqdm
import sys

# Check for minimum required arguments
if len(sys.argv) < 4:
    print("Usage: python script.py [SOURCE_DIR] [TARGET_DIR] [TARGET_COUNT]")
    sys.exit(1) # Stops the script gracefully

SOURCE_DIR = sys.argv[1]
TARGET_DIR = sys.argv[2]
TARGET_COUNT = int(sys.argv[3])

#ensure target dir exists
os.makedirs(TARGET_DIR, exist_ok=True)

def get_random_rotation():
    #generate random rotation matrix
    angles = np.random.uniform(0, 360, 3)
    r = st.Rotation.from_euler('xyz', angles, degrees=True)
    matrix = np.eye(4)
    matrix[:3, :3] = r.as_matrix()
    return matrix

def augment_class(label, files):
    #calculate how many copies per file to reach target
    orig_count = len(files)
    if orig_count == 0: return

    copies_needed = TARGET_COUNT - orig_count
    if copies_needed <= 0:
        #if already have enough, just copy originals
        for i, f in enumerate(files):
            src_path = os.path.join(SOURCE_DIR, f)
            dst_path = os.path.join(TARGET_DIR, f)
            if i < TARGET_COUNT:
                shutil.copy2(src_path, dst_path)
        return

    #first copy originals
    for f in files:
        shutil.copy2(os.path.join(SOURCE_DIR, f), os.path.join(TARGET_DIR, f))

    #generate augmented copies
    copies_per_file = (copies_needed // orig_count) + 1
    current_idx = orig_count + 1

    print(f"augmenting {label}: {orig_count} original -> ~{TARGET_COUNT}")

    for f in tqdm(files, desc=label):
        if current_idx > TARGET_COUNT: break

        src_path = os.path.join(SOURCE_DIR, f)
        ext = os.path.splitext(f)[1]

        try:
            #load mesh
            scene_or_mesh = trimesh.load(src_path)

            #handle scenes vs meshes
            if isinstance(scene_or_mesh, trimesh.Scene):
                original_scene = scene_or_mesh
            else:
                original_scene = trimesh.Scene(scene_or_mesh)

            for _ in range(copies_per_file):
                if current_idx > TARGET_COUNT: break

                #clone scene
                new_scene = original_scene.copy()

                #apply random rotation to all geometry nodes
                new_scene.apply_transform(get_random_rotation())

                #save new file
                new_filename = f"{label}{current_idx}{ext}"
                new_path = os.path.join(TARGET_DIR, new_filename)

                new_scene.export(new_path)
                current_idx += 1

        except Exception as e:
            print(f"error processing {f}: {e}")

#main
if __name__ == "__main__":
    #group files by class
    all_files = [f for f in os.listdir(SOURCE_DIR) if os.path.isfile(os.path.join(SOURCE_DIR, f))]
    class_map = {}

    for f in all_files:
        #extract label (text before first digit)
        match = re.search(r'^([a-zA-Z_]+)', f)
        if match:
            label = match.group(1).rstrip('0123456789')
            if label not in class_map:
                class_map[label] = []
            class_map[label].append(f)

    #process each class
    for label, files in class_map.items():
        augment_class(label, files)

    print(f"augmentation complete saved in {TARGET_DIR}")
