import bpy
import os
import sys
import argparse
import mathutils
import time

def clear_scene():
    """Deletes all objects from the default Blender scene."""
    if bpy.context.active_object and bpy.context.active_object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def normalize_transform(obj):
    """Centers the object and scales it to fit exactly within a 2x2x2 cube."""
    print("  -> Running Normalization...", end="")
    step_start = time.time()
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # Apply existing transforms first
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # Calculate bounding box dimensions and center
    local_bbox_center = 0.125 * sum((mathutils.Vector(b) for b in obj.bound_box), mathutils.Vector())
    global_bbox_center = obj.matrix_world @ local_bbox_center

    # Center the object
    obj.location = obj.location - global_bbox_center
    bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)

    # Scale object to fit max dimension of 2.0
    dims = obj.dimensions
    max_dim = max(dims.x, dims.y, dims.z)
    if max_dim > 0:
        scale_factor = 2.0 / max_dim
        obj.scale = (scale_factor, scale_factor, scale_factor)

    # Apply final scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    print(f"     Done in {time.time() - step_start:.2f} seconds")

def apply_boolean_manifold(obj):
    """Intersects the object with a 2x2x2 bounding cube."""
    print("  -> Running boolean_manifold...", end="")
    step_start = time.time()
    bpy.ops.mesh.primitive_cube_add(size=2.1, location=(0, 0, 0)) # Slightly larger than 2 to avoid Z-fighting
    cube = bpy.context.active_object

    bpy.context.view_layer.objects.active = obj

    # Add Boolean Modifier
    bool_mod = obj.modifiers.new(name="Manifold_Bool", type='BOOLEAN')
    bool_mod.operation = 'INTERSECT'
    bool_mod.object = cube
    bool_mod.solver = 'EXACT'

    bpy.ops.object.modifier_apply(modifier="Manifold_Bool")

    # Delete the bounding cube
    bpy.data.objects.remove(cube, do_unlink=True)
    print(f"     Done in {time.time() - step_start:.2f} seconds")

def process_mesh(filepath, output_path, voxel_size):
    """Re-mesh pipeline. voxel_size of 0.01 for a high res mesh; 0.05 for a low res mesh"""

    total_start = time.time()
    clear_scene()

    # 1. Import
    bpy.ops.wm.obj_import(filepath=filepath)
    meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    if not meshes:
        print(f"Skipping {filepath}: No mesh data.")
        return

    obj = meshes[0]
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # 2. Normalize (Scale and center to fit 2x2x2)
    normalize_transform(obj)

    # 3. First Boolean Manifold pass (as requested)
    # apply_boolean_manifold(obj)

    # 4. Make a copy for Shrinkwrap target
    copy_mesh_data = obj.data.copy()
    copy_obj = obj.copy()
    copy_obj.data = copy_mesh_data
    copy_obj.name = "Original_Copy"
    bpy.context.collection.objects.link(copy_obj)

    # Re-select the working object
    bpy.context.view_layer.objects.active = obj
    copy_obj.select_set(False)
    obj.select_set(True)

    # 5. Voxel Remesh
    print("  -> Running Voxel Remesh...", end="")
    step_start = time.time()
    obj.data.remesh_voxel_size = voxel_size
    obj.data.use_remesh_fix_poles = True
    obj.data.use_remesh_preserve_volume = True
    bpy.ops.object.voxel_remesh()
    print(f"     Done in {time.time() - step_start:.2f} seconds")

    # # 6. Quadriflow Remesh
    # # try:
    # #     print("  -> Running Quadriflow Remesh...", end="")
    # #     step_start = time.time()
    # #     bpy.ops.object.quadriflow_remesh(target_faces=16000)
    # #     print(f"     Done in {time.time() - step_start:.2f} seconds")
    # # except Exception as e:
    # #     print(f"Quadriflow failed on {filepath}, continuing with voxel mesh. Error: {e}")

    # 7. Merge by distance
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles(threshold=0.001)
    bpy.ops.object.mode_set(mode='OBJECT')

    # 8. Shrinkwrap modifier
    print("  -> Running Shrinkwrap modifier...", end="")
    step_start = time.time()
    shrink_mod = obj.modifiers.new(name="Shrinkwrap", type='SHRINKWRAP')
    shrink_mod.target = copy_obj
    shrink_mod.wrap_method = 'NEAREST_SURFACEPOINT'
    bpy.ops.object.modifier_apply(modifier="Shrinkwrap")
    print(f"     Done in {time.time() - step_start:.2f} seconds")

    # 9. Boolean Manifold pass
    apply_boolean_manifold(obj)

    # 10. Recalculate Normals (Outside)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')

    # 11. Delete the copy
    bpy.data.objects.remove(copy_obj, do_unlink=True)

    # 12. Export
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.wm.obj_export(
        filepath=output_path,
        export_selected_objects=True,
        export_materials=False,
        export_normals=True,
        export_uv=False
    )
    print(f"Cleaned and exported: {output_path} (Total time: {time.time() - total_start:.2f} seconds)\n")
if __name__ == "__main__":
    argv = sys.argv
    if "--" not in argv:
        sys.exit(1)

    args = argv[argv.index("--") + 1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.01,
        help="Size of voxels (default: 0.01; use 0.05 for a low poly result)"
    )
    parsed_args = parser.parse_args(args)

    for filename in os.listdir(parsed_args.input):
        if filename.endswith(".obj"):
            in_path = os.path.join(parsed_args.input, filename)
            out_path = os.path.join(parsed_args.output, filename)
            process_mesh(in_path, out_path, voxel_size)
