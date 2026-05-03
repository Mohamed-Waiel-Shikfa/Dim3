import bpy
import os
import sys
import argparse

def clear_scene():
    """Deletes all objects from the default Blender scene."""
    if bpy.context.active_object and bpy.context.active_object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def import_file(filepath):
    """Imports a 3D file using Blender 5.1.1+ operators."""
    ext = os.path.splitext(filepath)[1].lower()

    # Record the number of meshes before import to verify success
    meshes_before = len(bpy.data.meshes)

    try:
        # --- Modern Blender C++ Importers (Blender 4.0 - 5.1+) ---
        if ext == '.obj':
            # Note: This will log a C++ console error if the .mtl is missing,
            # but it still imports the geometry. We verify success below.
            bpy.ops.wm.obj_import(filepath=filepath)

        elif ext == '.stl':
            # STL was also moved to the C++ wm namespace in recent versions
            bpy.ops.wm.stl_import(filepath=filepath)

        elif ext == '.ply':
            bpy.ops.wm.ply_import(filepath=filepath)

        # --- Standard Python Importers ---
        elif ext == '.fbx':
            bpy.ops.import_scene.fbx(filepath=filepath)
        elif ext in ['.glb', '.gltf']:
            bpy.ops.import_scene.gltf(filepath=filepath)

        # --- Native Blender Files ---
        elif ext == '.blend':
            with bpy.data.libraries.load(filepath) as (data_from, data_to):
                data_to.objects = data_from.objects
            for obj in data_to.objects:
                if obj is not None and obj.type == 'MESH':
                    bpy.context.collection.objects.link(obj)
        else:
            print(f"Skipping unsupported extension: {ext}")
            return False

        # Validate that new geometry was actually added to the scene
        if len(bpy.data.meshes) > meshes_before:
            return True
        else:
            print(f"Failed to import {filepath}: File parsed but contained no valid mesh data.")
            return False

    except Exception as e:
        print(f"Failed to import {filepath}: {e}")
        return False

def join_all_meshes():
    """Joins all mesh objects in the scene into a single object."""
    bpy.ops.object.select_all(action='DESELECT')

    # Filter out cameras, lights, etc., keeping only meshes
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']

    if not meshes:
        print("No meshes found after import.")
        return False

    for mesh in meshes:
        mesh.select_set(True)

    bpy.context.view_layer.objects.active = meshes[0]

    if len(meshes) > 1:
        bpy.ops.object.join()

    return True

def export_obj(output_path):
    """Exports the active scene to an OBJ file (Blender 5.1.1+ API)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    bpy.ops.wm.obj_export(
        filepath=output_path,
        export_selected_objects=True,
        export_materials=False, # We strictly don't need materials for Dim3
        export_normals=True,
        export_uv=False
    )

def process_single_file(filepath, output_path):
    print(f"--- Processing single file: {filepath} ---")
    clear_scene()
    if not import_file(filepath):
        return False
    if not join_all_meshes():
        return False
    export_obj(output_path)
    print(f"Successfully unified and exported to: {output_path}")
    return True

def process_directory(input_dir, output_dir):
    print(f"Starting conversion from {input_dir} to {output_dir}...")

    for filename in os.listdir(input_dir):
        filepath = os.path.join(input_dir, filename)

        if not os.path.isfile(filepath) or filename.startswith('.'):
            continue

        print(f"\n--- Processing: {filename} ---")

        clear_scene()

        if not import_file(filepath):
            continue

        if not join_all_meshes():
            continue

        base_name = os.path.splitext(filename)[0]
        out_filepath = os.path.join(output_dir, f"{base_name}.obj")

        export_obj(out_filepath)
        print(f"Successfully unified and exported to: {out_filepath}")

if __name__ == "__main__":
    argv = sys.argv
    if "--" not in argv:
        print("Error: Please pass arguments after '--'.")
        sys.exit(1)

    args = argv[argv.index("--") + 1:]

    parser = argparse.ArgumentParser(description="Unify and convert 3D files to OBJ.")
    parser.add_argument("--input", required=False, help="Input directory containing raw 3D files.")
    parser.add_argument("--input_file", required=False, help="Single input file.")
    parser.add_argument("--output", required=False, help="Output directory for unified OBJ files.")
    parser.add_argument("--output_file", required=False, help="Single output file.")

    parsed_args = parser.parse_args(args)

    if parsed_args.input_file and parsed_args.output_file:
        process_single_file(parsed_args.input_file, parsed_args.output_file)
    elif parsed_args.input and parsed_args.output:
        process_directory(parsed_args.input, parsed_args.output)
    else:
        print("Provide either --input_file/--output_file OR --input/--output")
        sys.exit(1)
