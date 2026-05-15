#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Headless Blender renderer for GLB Creator.

Run directly for argument validation, or through Blender:
  blender -b --python src/tools/glb_creator_render_glb_blender.py -- --glb model.glb --out-dir renders
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def _argv_after_double_dash() -> list[str]:
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1 :]
    return sys.argv[1:]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glb", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--resolution", type=int, default=1024)
    return ap.parse_args(_argv_after_double_dash())


def _bbox_world(objects):
    import mathutils

    coords = []
    for obj in objects:
        if obj.type not in {"MESH", "CURVE", "SURFACE", "FONT"}:
            continue
        for corner in obj.bound_box:
            coords.append(obj.matrix_world @ mathutils.Vector(corner))
    if not coords:
        return mathutils.Vector((0, 0, 0)), mathutils.Vector((1, 1, 1))
    mn = mathutils.Vector((min(v.x for v in coords), min(v.y for v in coords), min(v.z for v in coords)))
    mx = mathutils.Vector((max(v.x for v in coords), max(v.y for v in coords), max(v.z for v in coords)))
    return mn, mx


def _look_at(obj, target):
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def main() -> None:
    args = _parse_args()
    glb_path = Path(args.glb).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    import bpy
    import mathutils

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    bpy.ops.import_scene.gltf(filepath=str(glb_path))
    imported = list(bpy.context.scene.objects)
    mesh_objects = [o for o in imported if o.type == "MESH"]
    if not mesh_objects:
        raise RuntimeError(f"No mesh objects imported from {glb_path}")

    mn, mx = _bbox_world(mesh_objects)
    center = (mn + mx) * 0.5
    dims = mx - mn
    radius = max(dims.x, dims.y, dims.z, 0.1)

    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.samples = 64
    bpy.context.scene.cycles.use_denoising = False
    if hasattr(bpy.context.view_layer, "cycles"):
        bpy.context.view_layer.cycles.use_denoising = False
    bpy.context.scene.render.resolution_x = int(args.resolution)
    bpy.context.scene.render.resolution_y = int(args.resolution)
    bpy.context.scene.view_settings.view_transform = "Filmic"
    bpy.context.scene.world = bpy.data.worlds.new("World") if bpy.context.scene.world is None else bpy.context.scene.world
    bpy.context.scene.world.color = (1.0, 1.0, 1.0)

    light_data = bpy.data.lights.new("Key_Area", type="AREA")
    light = bpy.data.objects.new("Key_Area", light_data)
    bpy.context.collection.objects.link(light)
    light.location = center + mathutils.Vector((radius * 1.6, -radius * 2.0, radius * 2.2))
    light.data.energy = 500
    light.data.size = max(radius * 2.0, 1.0)

    fill_data = bpy.data.lights.new("Fill_Area", type="AREA")
    fill = bpy.data.objects.new("Fill_Area", fill_data)
    bpy.context.collection.objects.link(fill)
    fill.location = center + mathutils.Vector((-radius * 2.0, radius * 1.4, radius * 1.5))
    fill.data.energy = 120
    fill.data.size = max(radius * 2.5, 1.0)

    cam_data = bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam.data.lens = 70
    cam.data.sensor_width = 32

    views = [
        ("front", 0),
        ("left", 90),
        ("right", -90),
        ("three_quarter", 45),
    ]
    manifest = {"glb": str(glb_path), "renders": []}
    distance = radius * 2.5
    z = center.z + max(dims.z * 0.25, radius * 0.18)

    for view, azimuth in views:
        rad = math.radians(azimuth)
        cam.location = mathutils.Vector((center.x + math.sin(rad) * distance, center.y - math.cos(rad) * distance, z))
        _look_at(cam, center)
        out_path = out_dir / f"render_{view}.png"
        bpy.context.scene.render.filepath = str(out_path)
        bpy.ops.render.render(write_still=True)
        manifest["renders"].append({"view": view, "path": str(out_path), "azimuth_deg": azimuth})

    (out_dir / "render_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
