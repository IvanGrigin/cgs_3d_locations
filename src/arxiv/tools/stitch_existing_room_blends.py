#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"

BLENDER_SCRIPT = r'''
import json
import math
from pathlib import Path

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def inverse_room_point(point, frame):
    ox, oy = frame.get("offset_xy") or [0.0, 0.0]
    gx, gy = frame.get("origin_xy") or [0.0, 0.0]
    a = float(frame.get("rotation_rad") or 0.0)
    x = float(point[0]) - float(ox)
    y = float(point[1]) - float(oy)
    return (
        x * math.cos(a) - y * math.sin(a) + float(gx),
        x * math.sin(a) + y * math.cos(a) + float(gy),
    )


def estimate_apartment_min(apartment, rooms):
    doors = (((apartment.get("room") or {}).get("meta") or {}).get("door_graph") or {}).get("doors") or []
    estimates = []
    by_id = {r["room_id"]: r for r in rooms}
    for door in doors:
        room_id = str(door.get("to") or "")
        center = door.get("center_xy")
        if room_id not in by_id or not isinstance(center, list) or len(center) < 2:
            continue
        room = by_id[room_id]["room"]
        frame = ((room.get("meta") or {}).get("coordinate_frame") or {})
        room_doors = room.get("doors") or []
        if not frame or not room_doors:
            continue
        seg = (room_doors[0] or {}).get("segment") or {}
        if not {"x1", "x2", "y1", "y2"} <= set(seg):
            continue
        local = ((float(seg["x1"]) + float(seg["x2"])) * 0.5, (float(seg["y1"]) + float(seg["y2"])) * 0.5)
        gx, gy = inverse_room_point(local, frame)
        estimates.append((gx - float(center[0]), gy - float(center[1])))
    if estimates:
        return (
            sorted(x for x, _ in estimates)[len(estimates) // 2],
            sorted(y for _, y in estimates)[len(estimates) // 2],
        )
    poly = (apartment.get("room") or {}).get("floor_polygon") or []
    if poly:
        return (min(float(p.get("x", 0.0)) for p in poly), min(float(p.get("y", 0.0)) for p in poly))
    return (0.0, 0.0)


def room_matrix(frame, apt_min):
    off = frame.get("offset_xy") or [0.0, 0.0]
    origin = frame.get("origin_xy") or [0.0, 0.0]
    angle = float(frame.get("rotation_rad") or 0.0)
    return (
        mathutils.Matrix.Translation((float(origin[0]) - apt_min[0], float(origin[1]) - apt_min[1], 0.0))
        @ mathutils.Matrix.Rotation(angle, 4, "Z")
        @ mathutils.Matrix.Translation((-float(off[0]), -float(off[1]), 0.0))
    )


def append_room_objects(blend_path, room_id, transform):
    blend_path = str(Path(blend_path).resolve())
    with bpy.data.libraries.load(blend_path, link=False) as (src, dst):
        dst.objects = list(src.objects)

    objects = [obj for obj in dst.objects if obj is not None]
    collection = bpy.data.collections.new(room_id)
    bpy.context.scene.collection.children.link(collection)

    object_set = set(objects)
    for obj in objects:
        obj.name = f"{room_id}__{obj.name}"
        try:
            collection.objects.link(obj)
        except RuntimeError:
            pass
        for old_collection in list(obj.users_collection):
            if old_collection != collection:
                try:
                    old_collection.objects.unlink(obj)
                except RuntimeError:
                    pass

    roots = [obj for obj in objects if obj.parent not in object_set]
    for obj in roots:
        obj.matrix_world = transform @ obj.matrix_world

    return {"room_id": room_id, "blend": blend_path, "objects": len(objects), "roots": len(roots)}


def add_camera_and_light():
    mesh_objects = [obj for obj in bpy.data.objects if obj.type == "MESH" and not obj.hide_render]
    if not mesh_objects:
        return
    pts = []
    for obj in mesh_objects:
        pts.extend(obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box)
    min_v = mathutils.Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    max_v = mathutils.Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    center = (min_v + max_v) * 0.5
    dims = max_v - min_v
    radius = max(float(dims.x), float(dims.y), 1.0)

    cam_data = bpy.data.cameras.new("ApartmentCamera")
    cam = bpy.data.objects.new("ApartmentCamera", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    cam.location = (center.x, center.y - radius * 1.15, max_v.z + radius * 0.9)
    direction = center - cam.location
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    cam.data.lens = 24
    cam.data.clip_end = 300.0
    bpy.context.scene.camera = cam

    light_data = bpy.data.lights.new("ApartmentAreaLight", type="AREA")
    light = bpy.data.objects.new("ApartmentAreaLight", light_data)
    bpy.context.scene.collection.objects.link(light)
    light.location = (center.x, center.y, max_v.z + 3.0)
    light.data.energy = 650.0
    light.data.size = max(radius, 3.0)


def configure_render(render_path):
    scene = bpy.context.scene
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except Exception:
        try:
            scene.render.engine = "BLENDER_EEVEE"
        except Exception:
            pass
    scene.render.resolution_x = 1600
    scene.render.resolution_y = 1200
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(Path(render_path).resolve())


def main():
    args = json.loads(Path(bpy.app.driver_namespace["stitch_args"]).read_text(encoding="utf-8"))
    apt_dir = Path(args["apt_dir"]).resolve()
    manifest = read_json(apt_dir / "manifest.json")
    apartment = read_json(apt_dir / "apartment.json")

    rooms = []
    for entry in manifest.get("rooms") or []:
        room_id = str(entry.get("room_id") or "")
        room_json = Path(entry.get("room_json") or "")
        if not room_id or not room_json.is_file():
            continue
        rooms.append({"room_id": room_id, "room": read_json(room_json).get("room") or {}, "entry": entry})

    apt_min = estimate_apartment_min(apartment, rooms)
    clear_scene()

    report = {
        "apartment_dir": str(apt_dir),
        "apartment_global_min_xy": [round(apt_min[0], 6), round(apt_min[1], 6)],
        "rooms": [],
    }
    for room_info in rooms:
        room_id = room_info["room_id"]
        frame = ((room_info["room"].get("meta") or {}).get("coordinate_frame") or {})
        blend = args["room_blends"].get(room_id)
        if not frame or not blend or not Path(blend).is_file():
            report["rooms"].append({"room_id": room_id, "status": "skipped", "blend": blend})
            continue
        item = append_room_objects(blend, room_id, room_matrix(frame, apt_min))
        item["status"] = "ok"
        report["rooms"].append(item)

    add_camera_and_light()
    if args.get("render"):
        configure_render(args["render"])
        bpy.ops.render.render(write_still=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(Path(args["out_blend"]).resolve()), compress=True)
    Path(args["report"]).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


main()
'''


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_blender(path: str | None) -> str:
    if path:
        p = Path(path).expanduser()
        if p.is_file():
            return str(p.resolve())
    if Path(DEFAULT_BLENDER).is_file():
        return DEFAULT_BLENDER
    found = shutil.which("blender")
    if found:
        return found
    raise RuntimeError("Blender executable not found")


def choose_room_blend(apt_dir: Path, room_id: str, room_type: str) -> Path | None:
    room_dir = apt_dir / "rooms" / room_id
    if room_type == "kitchen":
        kitchen = room_dir / "kitchen" / f"{room_id}.blend"
        if kitchen.is_file():
            return kitchen
    candidates = [
        room_dir / "pipeline" / "optimal" / "scene_infinigen_clean_supplier.optimal.blend",
        room_dir / "pipeline" / "optimal" / "scene_infinigen_clean.blend",
        room_dir / "pipeline" / "optimal" / "infinigen_clean_scene.blend",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def build_room_blend_map(apt_dir: Path) -> dict[str, str]:
    manifest = read_json(apt_dir / "manifest.json")
    out: dict[str, str] = {}
    for entry in manifest.get("rooms") or []:
        room_id = str(entry.get("room_id") or "")
        blend = choose_room_blend(apt_dir, room_id, str(entry.get("room_type") or ""))
        if room_id and blend:
            out[room_id] = str(blend.resolve())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Stitch already-rendered room .blend files into one apartment .blend.")
    parser.add_argument("apt_dir")
    parser.add_argument("--blender", default=None)
    parser.add_argument("--out-blend", default=None)
    parser.add_argument("--render", default=None)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    apt_dir = Path(args.apt_dir).expanduser().resolve()
    out_dir = apt_dir / "apartment_pipeline" / "optimal"
    out_blend = Path(args.out_blend).expanduser().resolve() if args.out_blend else out_dir / "scene_apartment.stitched_existing_rooms.blend"
    render = Path(args.render).expanduser().resolve() if args.render else out_dir / "render_apartment.stitched_existing_rooms.png"
    report = Path(args.report).expanduser().resolve() if args.report else out_dir / "stitched_existing_rooms.report.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    room_blends = build_room_blend_map(apt_dir)
    payload = {
        "apt_dir": str(apt_dir),
        "out_blend": str(out_blend),
        "render": str(render),
        "report": str(report),
        "room_blends": room_blends,
    }

    blender = resolve_blender(args.blender)
    tmp_root = Path("/private/tmp")
    script = tmp_root / f"cgs_stitch_existing_room_blends_{os.getpid()}.py"
    payload_path = tmp_root / f"cgs_stitch_existing_room_blends_{os.getpid()}.json"
    script.write_text(BLENDER_SCRIPT, encoding="utf-8")
    payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        base_blend = room_blends.get("kv_1165_floor_2_bathroom_0048") or next(iter(room_blends.values()), None)
        cmd = [blender]
        if base_blend:
            cmd.append(base_blend)
        expr = (
            "import bpy, mathutils; "
            f"code = open(r'{script}', encoding='utf-8').read(); "
            f"bpy.app.driver_namespace['stitch_args'] = r'{payload_path}'; "
            "exec(code)"
        )
        cmd += ["-b", "--python-expr", expr]
        print("RUN:", " ".join(cmd))
        subprocess.run(cmd, check=True)
    finally:
        script.unlink(missing_ok=True)
        payload_path.unlink(missing_ok=True)

    print(f"blend = {out_blend}")
    print(f"render = {render}")
    print(f"report = {report}")


if __name__ == "__main__":
    main()
