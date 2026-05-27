#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import bpy
from mathutils import Vector


def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Render orbit PNG frames from an opened .blend scene.")
    p.add_argument("--frames-dir", required=True)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=640)
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--yaw-step", type=float, default=120.0)
    p.add_argument("--elevations", default="35,70")
    p.add_argument("--margin", type=float, default=1.35)
    p.add_argument("--frame-index", type=int, default=None)
    p.add_argument("--hide-room-shell", action="store_true")
    p.add_argument("--hide-outliers", action="store_true")
    return p


def argv_after_blender_separator() -> list[str]:
    argv = sys.argv
    if "--" in argv:
        return argv[argv.index("--") + 1 :]
    return argv[1:]


def parse_elevations(raw: str) -> list[float]:
    vals = [float(x.strip()) for x in str(raw).split(",") if x.strip()]
    if not vals:
        raise RuntimeError("No orbit elevations passed")
    return vals


def world_bbox_points(obj: bpy.types.Object) -> list[Vector]:
    return [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]


def hide_object_family(root: bpy.types.Object) -> None:
    stack = [root]
    while stack:
        obj = stack.pop()
        obj.hide_render = True
        obj.hide_viewport = True
        stack.extend(list(obj.children))


def shell_name_part(name: str) -> str:
    low = str(name or "").strip().lower()
    return low.rsplit("__", 1)[-1]


def is_floor_shell_name(name: str) -> bool:
    part = shell_name_part(name)
    return (
        part.endswith(".floor")
        or part == "floor"
        or "room_floor" in part
        or "preview_floor" in part
        or part.startswith("floor_")
    )


def matches_room_shell_name(name: str) -> bool:
    part = shell_name_part(name)
    if is_floor_shell_name(name):
        return False
    if any(
        token in part
        for token in (
            "ceilinglight",
            "ceiling_light",
            "lamp_ceiling",
            "flat_ceiling_light",
            "wall_light",
            "wall_sconce",
            "wall_art",
            "wall_mounted",
            "wall_mount",
            "wall_cabinet",
            "wall_shelf",
            "wall_unit",
        )
    ):
        return False
    return (
        "room_wall" in part
        or "room_ceiling" in part
        or "room_exterior" in part
        or "room_wallpaper" in part
        or "wallpaper_supplieroverlay" in part
        or part.endswith(".exterior")
        or part.endswith(".ceiling")
        or part.endswith(".wall")
        or part.endswith(".meshed")
        or part.endswith("/0")
        or ".wall." in part
        or "/wall" in part
        or "ceiling" in part
        or "exterior" in part
    )


def looks_like_wall_or_ceiling_by_geometry(obj: bpy.types.Object) -> bool:
    if obj.type != "MESH" or obj.hide_render:
        return False
    part = shell_name_part(obj.name)
    if is_floor_shell_name(obj.name):
        return False
    if any(
        token in part
        for token in (
            "ceilinglight",
            "ceiling_light",
            "lamp_ceiling",
            "flat_ceiling_light",
            "wall_light",
            "wall_sconce",
            "wall_art",
            "wall_mounted",
            "wall_mount",
            "wall_cabinet",
            "wall_shelf",
            "wall_unit",
        )
    ):
        return False
    pts = world_bbox_points(obj)
    if not pts:
        return False
    size_x = max(p.x for p in pts) - min(p.x for p in pts)
    size_y = max(p.y for p in pts) - min(p.y for p in pts)
    size_z = max(p.z for p in pts) - min(p.z for p in pts)
    min_z = min(p.z for p in pts)
    max_z = max(p.z for p in pts)
    wide_xy = max(size_x, size_y)
    is_wall_panel = size_z >= 1.6 and wide_xy >= 0.8 and min(size_x, size_y) <= 0.18
    is_ceiling_cap = size_z <= 0.18 and wide_xy >= 0.8 and min_z >= 1.8 and max_z >= 1.9
    return is_wall_panel or is_ceiling_cap


def hide_room_shell_objects() -> int:
    bpy.context.view_layer.update()
    hidden = 0
    for obj in list(bpy.data.objects):
        if matches_room_shell_name(obj.name) or looks_like_wall_or_ceiling_by_geometry(obj):
            hide_object_family(obj)
            hidden += 1
    print(f"[orbit_render] hidden_room_shell_objects={hidden}")
    return hidden


def object_world_bounds(obj: bpy.types.Object):
    if obj.type != "MESH" or obj.hide_render:
        return None
    pts = world_bbox_points(obj)
    if not pts:
        return None
    bmin = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    bmax = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return bmin, bmax


def hide_outlier_objects(max_abs: float = 100.0, max_size: float = 30.0) -> int:
    bpy.context.view_layer.update()
    hidden = 0
    for obj in list(bpy.data.objects):
        bounds = object_world_bounds(obj)
        if bounds is None:
            continue
        bmin, bmax = bounds
        size = bmax - bmin
        max_dim = max(abs(size.x), abs(size.y), abs(size.z))
        max_coord = max(abs(v) for v in [bmin.x, bmax.x, bmin.y, bmax.y, bmin.z, bmax.z])
        if max_coord > max_abs or max_dim > max_size:
            hide_object_family(obj)
            hidden += 1
    print(f"[orbit_render] hidden_outlier_objects={hidden}")
    return hidden


def collect_scene_bounds() -> dict:
    bpy.context.view_layer.update()
    mesh_objects = [obj for obj in bpy.data.objects if obj.type == "MESH" and not obj.hide_render]
    if not mesh_objects:
        raise RuntimeError("No renderable MESH objects")
    pts: list[Vector] = []
    for obj in mesh_objects:
        pts.extend(world_bbox_points(obj))
    min_x = min(p.x for p in pts)
    max_x = max(p.x for p in pts)
    min_y = min(p.y for p in pts)
    max_y = max(p.y for p in pts)
    min_z = min(p.z for p in pts)
    max_z = max(p.z for p in pts)
    center = Vector((0.5 * (min_x + max_x), 0.5 * (min_y + max_y), 0.5 * (min_z + max_z)))
    size_x = max_x - min_x
    size_y = max_y - min_y
    size_z = max_z - min_z
    return {
        "center": center,
        "min_z": min_z,
        "max_z": max_z,
        "radius_xy": max(size_x, size_y) * 0.5,
        "size_x": size_x,
        "size_y": size_y,
        "size_z": size_z,
    }


def ensure_camera(scene: bpy.types.Scene) -> bpy.types.Object:
    if scene.camera and scene.camera.type == "CAMERA":
        return scene.camera
    cam_data = bpy.data.cameras.new("OrbitRenderCamera")
    cam_obj = bpy.data.objects.new("OrbitRenderCamera", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj
    return cam_obj


def ensure_target(scene: bpy.types.Scene, center: Vector) -> bpy.types.Object:
    target = bpy.data.objects.get("OrbitRenderTarget")
    if target is None:
        target = bpy.data.objects.new("OrbitRenderTarget", None)
        scene.collection.objects.link(target)
    target.location = center
    return target


def ensure_track_to(camera: bpy.types.Object, target: bpy.types.Object) -> None:
    for constraint in camera.constraints:
        if constraint.type == "TRACK_TO" and constraint.target == target:
            return
    constraint = camera.constraints.new(type="TRACK_TO")
    constraint.target = target
    constraint.track_axis = "TRACK_NEGATIVE_Z"
    constraint.up_axis = "UP_Y"


def setup_render(scene: bpy.types.Scene, width: int, height: int, samples: int) -> None:
    scene.render.image_settings.file_format = "PNG"
    scene.render.resolution_x = int(width)
    scene.render.resolution_y = int(height)
    scene.render.resolution_percentage = 100
    if scene.render.engine == "CYCLES":
        scene.cycles.samples = int(samples)
        scene.cycles.use_denoising = True
    elif scene.render.engine in {"BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"} and hasattr(scene, "eevee"):
        if hasattr(scene.eevee, "taa_render_samples"):
            scene.eevee.taa_render_samples = int(samples)


def camera_distance(camera: bpy.types.Object, bounds: dict, pitch_deg: float, margin: float) -> float:
    angle = camera.data.angle if getattr(camera.data, "angle", 0.0) > 1e-6 else math.radians(50.0)
    half_fov = max(angle * 0.5, math.radians(10.0))
    radius_xy = max(float(bounds["radius_xy"]), 0.5)
    size_z = max(float(bounds["size_z"]), 0.5)
    dist_xy = radius_xy / math.tan(half_fov)
    dist_z = (0.5 * size_z) / math.tan(half_fov)
    correction = 1.0 + 0.25 * abs(math.sin(math.radians(pitch_deg)))
    return max(dist_xy, dist_z) * float(margin) * correction + 0.5 * max(
        float(bounds["size_x"]),
        float(bounds["size_y"]),
        float(bounds["size_z"]),
    )


def place_camera(camera: bpy.types.Object, center: Vector, distance: float, yaw_deg: float, pitch_deg: float) -> None:
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    camera.location = Vector(
        (
            center.x + distance * math.cos(pitch) * math.cos(yaw),
            center.y + distance * math.cos(pitch) * math.sin(yaw),
            center.z + distance * math.sin(pitch),
        )
    )


def main() -> None:
    args = build_cli().parse_args(argv_after_blender_separator())
    elevations = parse_elevations(args.elevations)
    frame_indices = {int(args.frame_index)} if args.frame_index is not None else None
    if args.hide_room_shell:
        hide_room_shell_objects()
    if args.hide_outliers:
        hide_outlier_objects()

    scene = bpy.context.scene
    setup_render(scene, args.width, args.height, args.samples)
    bounds = collect_scene_bounds()
    center = bounds["center"]
    camera = ensure_camera(scene)
    target = ensure_target(scene, center)
    ensure_track_to(camera, target)

    os.makedirs(args.frames_dir, exist_ok=True)
    frame_idx = 0
    rendered = 0
    for pitch_deg in elevations:
        distance = camera_distance(camera, bounds, pitch_deg, args.margin)
        yaw = 0.0
        while yaw < 360.0 - 1e-9:
            if frame_indices is None or frame_idx in frame_indices:
                place_camera(camera, center, distance, yaw, pitch_deg)
                scene.camera = camera
                out_path = os.path.join(args.frames_dir, f"frame_{frame_idx:03d}.png")
                scene.render.filepath = out_path
                bpy.ops.render.render(write_still=True)
                print(f"[orbit_render] rendered {out_path}")
                rendered += 1
            frame_idx += 1
            yaw += float(args.yaw_step)

    print(json.dumps({"frame_count": frame_idx, "rendered_count": rendered}, ensure_ascii=False))


if __name__ == "__main__":
    main()
