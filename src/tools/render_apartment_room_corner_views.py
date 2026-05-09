#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import bpy
from mathutils import Vector


def argv_after_blender_separator() -> list[str]:
    argv = sys.argv
    if "--" in argv:
        return argv[argv.index("--") + 1 :]
    return argv[1:]


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, data: Any) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def polygon_xy(raw: Any) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    if not isinstance(raw, list):
        return out
    for point in raw:
        if not isinstance(point, dict):
            continue
        try:
            out.append((float(point.get("x", 0.0)), float(point.get("y", point.get("z", 0.0)))))
        except Exception:
            continue
    return out


def polygon_area(poly: list[tuple[float, float]]) -> float:
    if len(poly) < 3:
        return 0.0
    acc = 0.0
    for idx, (x1, y1) in enumerate(poly):
        x2, y2 = poly[(idx + 1) % len(poly)]
        acc += x1 * y2 - x2 * y1
    return acc * 0.5


def polygon_centroid(poly: list[tuple[float, float]]) -> tuple[float, float]:
    if not poly:
        return (0.0, 0.0)
    area = polygon_area(poly)
    if abs(area) < 1e-8:
        return (sum(x for x, _ in poly) / len(poly), sum(y for _, y in poly) / len(poly))
    cx = 0.0
    cy = 0.0
    for idx, (x1, y1) in enumerate(poly):
        x2, y2 = poly[(idx + 1) % len(poly)]
        cross = x1 * y2 - x2 * y1
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    factor = 1.0 / (6.0 * area)
    return (cx * factor, cy * factor)


def polygon_bbox(poly: list[tuple[float, float]]) -> dict[str, float]:
    if not poly:
        return {"x_min": 0.0, "x_max": 3.0, "y_min": 0.0, "y_max": 3.0}
    xs = [x for x, _ in poly]
    ys = [y for _, y in poly]
    return {"x_min": min(xs), "x_max": max(xs), "y_min": min(ys), "y_max": max(ys)}


def point_in_polygon(point: tuple[float, float], poly: list[tuple[float, float]], eps: float = 0.0) -> bool:
    if len(poly) < 3:
        return True
    x, y = point
    inside = False
    j = len(poly) - 1
    for i, (xi, yi) in enumerate(poly):
        xj, yj = poly[j]
        if abs((yi - y) * (xj - x) - (yj - y) * (xi - x)) <= eps:
            if min(xi, xj) - eps <= x <= max(xi, xj) + eps and min(yi, yj) - eps <= y <= max(yi, yj) + eps:
                return True
        crosses = (yi > y) != (yj > y)
        if crosses:
            x_at_y = (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
            if x < x_at_y + eps:
                inside = not inside
        j = i
    return inside


def move_toward(src: tuple[float, float], dst: tuple[float, float], distance: float) -> tuple[float, float]:
    dx = dst[0] - src[0]
    dy = dst[1] - src[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return dst
    step = min(distance, length * 0.82)
    return (src[0] + dx / length * step, src[1] + dy / length * step)


def camera_local_corners(poly: list[tuple[float, float]]) -> list[tuple[str, tuple[float, float]]]:
    box = polygon_bbox(poly)
    center = polygon_centroid(poly)
    width = max(box["x_max"] - box["x_min"], 0.1)
    depth = max(box["y_max"] - box["y_min"], 0.1)
    inset = min(0.48, max(0.18, min(width, depth) * 0.16))
    raw = [
        ("corner_00_sw", (box["x_min"], box["y_min"])),
        ("corner_01_se", (box["x_max"], box["y_min"])),
        ("corner_02_ne", (box["x_max"], box["y_max"])),
        ("corner_03_nw", (box["x_min"], box["y_max"])),
    ]
    out: list[tuple[str, tuple[float, float]]] = []
    for name, corner in raw:
        chosen = None
        for mul in (1.0, 1.5, 2.0, 2.8, 3.6):
            candidate = move_toward(corner, center, inset * mul)
            if point_in_polygon(candidate, poly, eps=0.03):
                chosen = candidate
                break
        if chosen is None:
            nearest = min(poly, key=lambda p: math.hypot(p[0] - corner[0], p[1] - corner[1])) if poly else corner
            chosen = move_toward(nearest, center, inset)
        out.append((name, chosen))
    return out


def apartment_min(apt_dir: Path, apartment_scene: Path | None) -> tuple[float, float]:
    if apartment_scene and apartment_scene.is_file():
        meta = read_json(apartment_scene).get("meta") or {}
        raw = meta.get("apartment_global_min_xy") or []
        if isinstance(raw, list) and len(raw) >= 2:
            return (float(raw[0]), float(raw[1]))
    apt_json = apt_dir / "apartment.json"
    if apt_json.is_file():
        poly = polygon_xy((read_json(apt_json).get("room") or {}).get("floor_polygon") or [])
        if poly:
            return (min(x for x, _ in poly), min(y for _, y in poly))
    return (0.0, 0.0)


def transform_local_point(point: tuple[float, float], z: float, frame: dict[str, Any], apt_min: tuple[float, float]) -> Vector:
    off = frame.get("offset_xy") or [0.0, 0.0]
    origin = frame.get("origin_xy") or [0.0, 0.0]
    angle = float(frame.get("rotation_rad") or math.radians(float(frame.get("rotation_deg") or 0.0)))
    px = float(point[0]) - float(off[0])
    py = float(point[1]) - float(off[1])
    c = math.cos(angle)
    s = math.sin(angle)
    return Vector(
        (
            float(origin[0]) - apt_min[0] + c * px - s * py,
            float(origin[1]) - apt_min[1] + s * px + c * py,
            float(z),
        )
    )


def room_summaries(apt_dir: Path) -> list[dict[str, Any]]:
    manifest_path = apt_dir / "manifest.json"
    rooms: list[dict[str, Any]] = []
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        for entry in manifest.get("rooms") or []:
            if not isinstance(entry, dict):
                continue
            room_id = str(entry.get("room_id") or "")
            summary_path = Path(str(entry.get("summary_json") or ""))
            if not summary_path.is_file() and room_id:
                summary_path = apt_dir / "rooms" / room_id / "room_summary.json"
            if summary_path.is_file():
                item = read_json(summary_path)
                item.setdefault("id", room_id)
                item.setdefault("room_type", entry.get("room_type") or entry.get("prompt_room_type"))
                rooms.append(item)
    if rooms:
        return rooms
    for path in sorted((apt_dir / "rooms").glob("*/room_summary.json")):
        item = read_json(path)
        item.setdefault("id", path.parent.name)
        rooms.append(item)
    return rooms


def ensure_camera(scene: bpy.types.Scene) -> bpy.types.Object:
    cam = bpy.data.objects.get("RoomCornerCamera")
    if cam is None or cam.type != "CAMERA":
        cam_data = bpy.data.cameras.new("RoomCornerCamera")
        cam = bpy.data.objects.new("RoomCornerCamera", cam_data)
        scene.collection.objects.link(cam)
    scene.camera = cam
    cam.data.type = "PERSP"
    cam.data.angle = math.radians(74.0)
    cam.data.clip_start = 0.02
    cam.data.clip_end = 250.0
    return cam


def ensure_target(scene: bpy.types.Scene) -> bpy.types.Object:
    target = bpy.data.objects.get("RoomCornerTarget")
    if target is None:
        target = bpy.data.objects.new("RoomCornerTarget", None)
        scene.collection.objects.link(target)
    return target


def aim_camera(camera: bpy.types.Object, target: Vector) -> None:
    direction = target - camera.location
    if direction.length <= 1e-6:
        return
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def ensure_fill_light(scene: bpy.types.Scene, center: Vector, ceiling_height: float) -> None:
    light = bpy.data.objects.get("RoomCornerFillLight")
    if light is None or light.type != "LIGHT":
        light_data = bpy.data.lights.new("RoomCornerFillLight", type="AREA")
        light = bpy.data.objects.new("RoomCornerFillLight", light_data)
        scene.collection.objects.link(light)
    light.location = (center.x, center.y, max(ceiling_height + 0.4, 2.8))
    light.data.energy = 320.0
    light.data.size = 4.5


def setup_render(scene: bpy.types.Scene, width: int, height: int, samples: int) -> None:
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except Exception:
        try:
            scene.render.engine = "BLENDER_EEVEE"
        except Exception:
            pass
    if hasattr(scene, "eevee"):
        for attr in ("taa_render_samples", "taa_samples"):
            if hasattr(scene.eevee, attr):
                try:
                    setattr(scene.eevee, attr, int(samples))
                except Exception:
                    pass
    if scene.render.engine == "CYCLES":
        scene.cycles.samples = int(samples)
        scene.cycles.use_denoising = True
    scene.render.resolution_x = int(width)
    scene.render.resolution_y = int(height)
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1


def write_markdown(report: dict[str, Any], out_path: Path) -> None:
    root = out_path.parent
    lines = [
        "# Apartment room corner renders",
        "",
        f"- apartment: `{report['apartment_dir']}`",
        f"- blend: `{report['blend']}`",
        f"- rooms: {len(report['rooms'])}",
        "",
    ]
    for room in report["rooms"]:
        lines.append(f"## {room['room_id']}")
        lines.append("")
        lines.append(f"- type: `{room.get('room_type') or ''}`")
        lines.append("")
        for render in room.get("renders") or []:
            png = Path(render["path"])
            try:
                rel = png.relative_to(root)
            except ValueError:
                rel = png
            lines.append(f"![{render['corner']}]({rel.as_posix()})")
            lines.append("")
    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render four upper-corner views for every room in an assembled apartment blend.")
    parser.add_argument("--apt-dir", required=True)
    parser.add_argument("--mode", default="optimal")
    parser.add_argument("--apartment-scene", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--report-json", default=None)
    parser.add_argument("--report-md", default=None)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--samples", type=int, default=16)
    return parser


def main() -> None:
    args = build_cli().parse_args(argv_after_blender_separator())
    apt_dir = Path(args.apt_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else apt_dir / "apartment_pipeline" / args.mode / "room_corner_renders"
    report_json = Path(args.report_json).expanduser().resolve() if args.report_json else apt_dir / "apartment_pipeline" / args.mode / "room_corner_renders.report.json"
    report_md = Path(args.report_md).expanduser().resolve() if args.report_md else apt_dir / "apartment_pipeline" / args.mode / "room_corner_renders.report.md"
    apartment_scene = Path(args.apartment_scene).expanduser().resolve() if args.apartment_scene else apt_dir / "apartment_pipeline" / args.mode / "scene_apartment.requirements.v1.json"
    apartment_blend = apt_dir / "apartment_pipeline" / args.mode / "scene_apartment.requirements.blend"
    current_blend = Path(bpy.data.filepath).resolve() if bpy.data.filepath else None
    if apartment_blend.is_file() and current_blend != apartment_blend.resolve():
        bpy.ops.wm.open_mainfile(filepath=str(apartment_blend))
    out_dir.mkdir(parents=True, exist_ok=True)

    scene = bpy.context.scene
    setup_render(scene, args.width, args.height, args.samples)
    camera = ensure_camera(scene)
    target = ensure_target(scene)
    apt_min = apartment_min(apt_dir, apartment_scene)

    report: dict[str, Any] = {
        "apartment_dir": str(apt_dir),
        "blend": str(Path(bpy.data.filepath).resolve()) if bpy.data.filepath else None,
        "mode": args.mode,
        "apartment_global_min_xy": [round(apt_min[0], 6), round(apt_min[1], 6)],
        "output_dir": str(out_dir),
        "width": int(args.width),
        "height": int(args.height),
        "samples": int(args.samples),
        "rooms": [],
    }

    for summary in room_summaries(apt_dir):
        room_id = str(summary.get("id") or summary.get("room_id") or "room")
        poly = polygon_xy(summary.get("floor_polygon") or [])
        if len(poly) < 3:
            width = float(summary.get("width_m") or 3.0)
            depth = float(summary.get("depth_m") or 3.0)
            poly = [(0.0, 0.0), (width, 0.0), (width, depth), (0.0, depth)]
        frame = summary.get("coordinate_frame") or {}
        if not frame:
            room_scene = apt_dir / "rooms" / room_id / "pipeline" / args.mode / "scene_requirements.v1.json"
            if room_scene.is_file():
                frame = (((read_json(room_scene).get("room") or {}).get("meta") or {}).get("coordinate_frame") or {})
        if not frame:
            report["rooms"].append({"room_id": room_id, "status": "missing_coordinate_frame", "renders": []})
            continue

        ceiling = float(summary.get("ceiling_height_m") or 2.8)
        centroid_local = polygon_centroid(poly)
        camera_z = max(1.85, ceiling - 0.18)
        target_z = min(max(1.05, ceiling * 0.42), 1.35)
        target_world = transform_local_point(centroid_local, target_z, frame, apt_min)
        target.location = target_world
        ensure_fill_light(scene, target_world, ceiling)

        room_report = {
            "room_id": room_id,
            "room_type": summary.get("room_type") or summary.get("source_room_type"),
            "status": "ok",
            "target_world": [round(float(target_world.x), 4), round(float(target_world.y), 4), round(float(target_world.z), 4)],
            "renders": [],
        }
        room_dir = out_dir / room_id
        room_dir.mkdir(parents=True, exist_ok=True)
        for idx, (corner_name, local_xy) in enumerate(camera_local_corners(poly)):
            camera.location = transform_local_point(local_xy, camera_z, frame, apt_min)
            aim_camera(camera, target_world)
            out_png = room_dir / f"{idx:02d}_{corner_name}.png"
            scene.render.filepath = str(out_png)
            bpy.ops.render.render(write_still=True)
            cam_world = camera.location
            room_report["renders"].append(
                {
                    "corner": corner_name,
                    "path": str(out_png),
                    "camera_world": [round(float(cam_world.x), 4), round(float(cam_world.y), 4), round(float(cam_world.z), 4)],
                    "camera_local_xy": [round(float(local_xy[0]), 4), round(float(local_xy[1]), 4)],
                }
            )
        report["rooms"].append(room_report)

    write_json(report_json, report)
    write_markdown(report, report_md)
    print(json.dumps({"report_json": str(report_json), "report_md": str(report_md), "rooms": len(report["rooms"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
