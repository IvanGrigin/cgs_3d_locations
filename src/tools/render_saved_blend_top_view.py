#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import bpy
import mathutils


def _visible_mesh_bounds() -> tuple[mathutils.Vector, mathutils.Vector]:
    mins: list[mathutils.Vector] = []
    maxs: list[mathutils.Vector] = []
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH" or obj.hide_get() or obj.hide_render:
            continue
        name = obj.name.lower()
        if "camera" in name or "bbox" in name or "axis" in name:
            continue
        for corner in obj.bound_box:
            world = obj.matrix_world @ mathutils.Vector(corner)
            mins.append(world)
            maxs.append(world)
    if not mins:
        return mathutils.Vector((-3.0, -3.0, 0.0)), mathutils.Vector((3.0, 3.0, 3.0))
    return (
        mathutils.Vector((min(v.x for v in mins), min(v.y for v in mins), min(v.z for v in mins))),
        mathutils.Vector((max(v.x for v in maxs), max(v.y for v in maxs), max(v.z for v in maxs))),
    )


def _hide_ceiling_caps() -> int:
    rows: list[tuple[bpy.types.Object, float, float, float, float]] = []
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH" or obj.hide_get() or obj.hide_render:
            continue
        corners = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
        if not corners:
            continue
        z_min = min(v.z for v in corners)
        z_max = max(v.z for v in corners)
        x_span = max(v.x for v in corners) - min(v.x for v in corners)
        y_span = max(v.y for v in corners) - min(v.y for v in corners)
        rows.append((obj, z_min, z_max, x_span, y_span))
    if not rows:
        return 0
    scene_z_max = max(row[2] for row in rows)
    hidden = 0
    for obj, z_min, z_max, x_span, y_span in rows:
        name = obj.name.lower()
        z_span = z_max - z_min
        area = x_span * y_span
        is_named_ceiling = "ceiling" in name or "roof" in name
        is_room_shell_duplicate = (
            (name.endswith("/0") or name.endswith(".meshed") or name.endswith(".exterior"))
            and "living-room" in name
            and area >= 20.0
            and z_span >= 2.0
        )
        is_top_cap = z_span <= 0.35 and z_max >= scene_z_max - 0.55 and area >= 4.0
        if is_named_ceiling or is_room_shell_duplicate or is_top_cap:
            obj.hide_render = True
            obj.hide_set(True)
            hidden += 1
    return hidden


def _look_at(obj: bpy.types.Object, target: mathutils.Vector) -> None:
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--azimuth-deg", type=float, default=-135.0)
    parser.add_argument("--elevation-deg", type=float, default=52.0)
    parser.add_argument("--radius-mult", type=float, default=1.18)
    parser.add_argument("--lens", type=float, default=24.0)
    parser.add_argument("--resolution-x", type=int, default=1400)
    parser.add_argument("--resolution-y", type=int, default=1050)
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    args = parser.parse_args(argv)

    scene = bpy.context.scene
    hidden_ceilings = _hide_ceiling_caps()
    print(f"Hidden ceiling/top cap objects: {hidden_ceilings}")
    bb_min, bb_max = _visible_mesh_bounds()
    center = (bb_min + bb_max) * 0.5
    dims = bb_max - bb_min
    xy_span = max(float(dims.x), float(dims.y), 1.0)
    radius = xy_span * float(args.radius_mult)
    az = math.radians(args.azimuth_deg)
    elev = math.radians(args.elevation_deg)
    target = mathutils.Vector((center.x, center.y, bb_min.z + max(float(dims.z) * 0.32, 1.0)))

    cam = bpy.data.objects.get("CGS_TopInspectionCamera")
    if cam is None or cam.type != "CAMERA":
        cam_data = bpy.data.cameras.new("CGS_TopInspectionCamera")
        cam = bpy.data.objects.new("CGS_TopInspectionCamera", cam_data)
        scene.collection.objects.link(cam)
    cam.location = (
        target.x + radius * math.cos(az),
        target.y + radius * math.sin(az),
        target.z + math.tan(elev) * radius,
    )
    cam.data.type = "PERSP"
    cam.data.lens = float(args.lens)
    cam.data.clip_start = 0.02
    cam.data.clip_end = 300.0
    _look_at(cam, target)
    scene.camera = cam

    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except Exception:
        scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = int(args.resolution_x)
    scene.render.resolution_y = int(args.resolution_y)
    scene.render.resolution_percentage = 100
    try:
        scene.eevee.taa_render_samples = 32
    except Exception:
        pass
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.render.filepath = str(Path(args.out).expanduser().resolve())
    bpy.ops.render.render(write_still=True)
    print(f"Saved render: {scene.render.filepath}")


if __name__ == "__main__":
    main()
