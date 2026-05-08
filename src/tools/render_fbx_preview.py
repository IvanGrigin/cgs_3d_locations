# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


def _parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Render a standalone preview of an FBX/OBJ/GLB asset.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--render", required=True)
    ap.add_argument("--save-blend", default=None)
    ap.add_argument("--resolution", type=int, default=1200)
    return ap.parse_args(argv)


def _clear_scene(bpy) -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def _import_asset(bpy, path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(path))
    elif suffix == ".obj":
        if hasattr(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(filepath=str(path))
        else:
            bpy.ops.import_scene.obj(filepath=str(path))
    elif suffix in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(path))
    else:
        raise RuntimeError(f"Unsupported asset format: {suffix}")


def _content_meshes(bpy):
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]


def _bbox(bpy):
    from mathutils import Vector

    meshes = _content_meshes(bpy)
    if not meshes:
        return None
    mins = Vector((float("inf"), float("inf"), float("inf")))
    maxs = Vector((float("-inf"), float("-inf"), float("-inf")))
    for obj in meshes:
        for corner in obj.bound_box:
            world = obj.matrix_world @ Vector(corner)
            mins.x = min(mins.x, world.x)
            mins.y = min(mins.y, world.y)
            mins.z = min(mins.z, world.z)
            maxs.x = max(maxs.x, world.x)
            maxs.y = max(maxs.y, world.y)
            maxs.z = max(maxs.z, world.z)
    return mins, maxs


def _center_and_scale(bpy) -> None:
    bbox = _bbox(bpy)
    if not bbox:
        return
    mins, maxs = bbox
    center = (mins + maxs) * 0.5
    dims = maxs - mins
    longest = max(float(dims.x), float(dims.y), float(dims.z), 1e-6)
    scale = 2.4 / longest
    meshes = _content_meshes(bpy)
    for obj in meshes:
        obj.location -= center
        obj.scale = tuple(float(v) * scale for v in obj.scale)
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
    if meshes:
        bpy.context.view_layer.objects.active = meshes[0]
        bpy.ops.object.transform_apply(location=True, rotation=False, scale=True)

    bbox = _bbox(bpy)
    if not bbox:
        return
    mins, _maxs = bbox
    for obj in meshes:
        obj.location.z -= float(mins.z)


def _make_imported_materials_visible(bpy) -> None:
    mat = bpy.data.materials.new("OriginalFBX_GeometryPreview")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = (0.72, 0.70, 0.66, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.62
        bsdf.inputs["Alpha"].default_value = 1.0
    mat.blend_method = "OPAQUE"
    mat.use_backface_culling = False

    for obj in _content_meshes(bpy):
        obj.show_wire = True
        obj.show_in_front = True
        if obj.data.materials:
            for i in range(len(obj.data.materials)):
                obj.data.materials[i] = mat
        else:
            obj.data.materials.append(mat)


def _setup_render(bpy, resolution: int) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 64
    scene.render.resolution_x = int(resolution)
    scene.render.resolution_y = int(resolution)
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium High Contrast"
    scene.world = scene.world or bpy.data.worlds.new("World")
    scene.world.color = (0.78, 0.78, 0.78)

    bbox = _bbox(bpy)
    if bbox:
        mins, maxs = bbox
        cx, cy, cz = (mins + maxs) * 0.5
        radius = max(float((maxs - mins).length), 1.0)
    else:
        cx = cy = cz = 0.0
        radius = 3.0

    bpy.ops.object.light_add(type="AREA", location=(0.0, -3.5, 4.2))
    light = bpy.context.object
    light.name = "Preview_Key_Area"
    light.data.energy = 550
    light.data.size = 4.0

    bpy.ops.object.camera_add(location=(2.8, -3.6, 2.2), rotation=(math.radians(62), 0.0, math.radians(38)))
    cam = bpy.context.object
    direction = cam.location - cam.location.__class__((cx, cy, cz))
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    cam.data.lens = 50
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = max(radius * 0.72, 2.2)
    scene.camera = cam


def main() -> None:
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    args = _parse_args(argv)

    import bpy  # type: ignore

    input_path = Path(args.input).expanduser().resolve()
    render_path = Path(args.render).expanduser().resolve()
    blend_path = Path(args.save_blend).expanduser().resolve() if args.save_blend else None

    _clear_scene(bpy)
    _import_asset(bpy, input_path)
    _center_and_scale(bpy)
    _make_imported_materials_visible(bpy)
    _setup_render(bpy, int(args.resolution))

    render_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(render_path)
    bpy.ops.render.render(write_still=True)
    if blend_path:
        blend_path.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))


if __name__ == "__main__":
    main()
