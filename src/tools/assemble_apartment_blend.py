#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import mathutils


ROOM_BLEND_CANDIDATES = (
    "pipeline/optimal/scene_infinigen_clean_supplier.requirements.blend",
    "pipeline/optimal/scene_kitchen_requirements.blend",
    "pipeline/optimal/scene_infinigen_clean_supplier.optimal.memfix.blend",
    "pipeline/optimal/scene_infinigen_clean_supplier.optimal.blend",
    "kitchen/{room_id}.blend",
)


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for block in (
        bpy.data.meshes,
        bpy.data.curves,
        bpy.data.lights,
        bpy.data.cameras,
        bpy.data.materials,
        bpy.data.images,
    ):
        for item in list(block):
            if getattr(item, "users", 0) == 0:
                block.remove(item)


def room_scene_path(room_dir: Path, mode: str) -> Path:
    return room_dir / "pipeline" / mode / "scene_requirements.v1.json"


def find_room_blend(room_dir: Path, room_id: str, mode: str) -> Path | None:
    for rel in ROOM_BLEND_CANDIDATES:
        path = room_dir / rel.format(room_id=room_id)
        if path.is_file():
            return path
    return None


def apt_min_from_scene(apartment_scene: Path | None, apartment_json: Path) -> tuple[float, float]:
    if apartment_scene and apartment_scene.is_file():
        data = read_json(apartment_scene)
        raw = ((data.get("meta") or {}).get("apartment_global_min_xy") or [])
        if isinstance(raw, list) and len(raw) >= 2:
            return float(raw[0]), float(raw[1])
    apt = read_json(apartment_json)
    poly = (apt.get("room") or {}).get("floor_polygon") or []
    if not poly:
        return 0.0, 0.0
    return (
        min(float(p.get("x", 0.0)) for p in poly if isinstance(p, dict)),
        min(float(p.get("y", 0.0)) for p in poly if isinstance(p, dict)),
    )


def transform_from_frame(frame: dict, apt_min: tuple[float, float]) -> mathutils.Matrix:
    off = frame.get("offset_xy") or [0.0, 0.0]
    origin = frame.get("origin_xy") or [0.0, 0.0]
    angle = float(frame.get("rotation_rad") or math.radians(float(frame.get("rotation_deg") or 0.0)))
    return (
        mathutils.Matrix.Translation((float(origin[0]) - apt_min[0], float(origin[1]) - apt_min[1], 0.0))
        @ mathutils.Matrix.Rotation(angle, 4, "Z")
        @ mathutils.Matrix.Translation((-float(off[0]), -float(off[1]), 0.0))
    )


def link_loaded_objects(objects: list[bpy.types.Object], collection: bpy.types.Collection) -> None:
    for obj in objects:
        if obj is None:
            continue
        if not any(existing == obj for existing in collection.objects):
            collection.objects.link(obj)


def rename_room_objects(objects: list[bpy.types.Object], room_id: str) -> None:
    for obj in objects:
        if obj is None:
            continue
        obj.name = f"{room_id}__{obj.name}"
        if obj.data is not None and hasattr(obj.data, "name"):
            obj.data.name = f"{room_id}__{obj.data.name}"


def looks_like_architectural_door_name(name: str) -> bool:
    low = str(name or "").strip().lower()
    if not low:
        return False
    if any(token in low for token in ("base_", "cabinet", "wardrobe", "cupboard")) and "_door_" in low:
        return False
    return (
        low in {"door", "doors"}
        or low.endswith("_door")
        or low.endswith("__door")
        or low.endswith("_doors")
        or low.endswith("__doors")
        or "__door." in low
        or "__doors." in low
        or "door_internal" in low
        or "doorfactory" in low
        or "paneldoorfactory" in low
        or "louverdoorfactory" in low
        or "glasspaneldoorfactory" in low
    )


def looks_like_bbox_helper_name(name: str) -> bool:
    low = str(name or "").strip().lower()
    return (
        "bbox_placeholder" in low
        or "spawn_placeholder" in low
        or low.endswith("_aabb")
        or "_aabb." in low
        or low.startswith("invalid_")
        or "__invalid_" in low
    )


def looks_like_functional_light_helper(obj: bpy.types.Object) -> bool:
    low = str(getattr(obj, "name", "") or "").lower()
    return bool(obj.get("cgs_functional_light")) or "cgs_functionallight_" in low


def remove_object_family(root: bpy.types.Object) -> int:
    removed = 0
    stack = [root]
    seen: set[bpy.types.Object] = set()
    while stack:
        obj = stack.pop()
        if obj is None or obj in seen:
            continue
        seen.add(obj)
        stack.extend(list(obj.children))
        try:
            bpy.data.objects.remove(obj, do_unlink=True)
            removed += 1
        except Exception:
            hide_object_family(obj)
    return removed


def cleanup_room_visual_helpers(objects: list[bpy.types.Object]) -> dict[str, int]:
    report = {
        "hidden_architectural_door_objects": 0,
        "removed_bbox_helper_objects": 0,
        "removed_functional_light_helpers": 0,
    }
    for obj in list(objects):
        if obj.name not in bpy.data.objects:
            continue
        if looks_like_architectural_door_name(obj.name):
            report["hidden_architectural_door_objects"] += hide_object_family(obj)
            continue
        if looks_like_bbox_helper_name(obj.name):
            report["removed_bbox_helper_objects"] += remove_object_family(obj)
            continue
        if looks_like_functional_light_helper(obj):
            report["removed_functional_light_helpers"] += remove_object_family(obj)
    return report


def append_room_blend(blend_path: Path, room_id: str, xform: mathutils.Matrix, root_collection: bpy.types.Collection) -> dict:
    before = set(bpy.data.objects)
    with bpy.data.libraries.load(str(blend_path), link=False) as (data_from, data_to):
        data_to.objects = list(data_from.objects)
    loaded = [obj for obj in bpy.data.objects if obj not in before]
    removed_cameras = 0
    for obj in list(loaded):
        if obj.type == "CAMERA":
            bpy.data.objects.remove(obj, do_unlink=True)
            loaded.remove(obj)
            removed_cameras += 1

    room_coll = bpy.data.collections.new(room_id)
    root_collection.children.link(room_coll)
    link_loaded_objects(loaded, room_coll)
    rename_room_objects(loaded, room_id)
    bpy.context.view_layer.update()

    loaded_set = set(loaded)
    roots = [obj for obj in loaded if obj.parent not in loaded_set]
    for obj in roots:
        obj.matrix_world = xform @ obj.matrix_world
    for obj in loaded:
        obj["source_room_id"] = room_id
        obj["source_room_blend"] = str(blend_path)
    bpy.context.view_layer.update()
    cleanup_report = cleanup_room_visual_helpers(loaded)
    return {
        "room_id": room_id,
        "blend": str(blend_path.resolve()),
        "objects_loaded": len(loaded),
        "root_objects_transformed": len(roots),
        "removed_cameras": removed_cameras,
        **cleanup_report,
    }


def world_bbox(objects: list[bpy.types.Object]) -> tuple[mathutils.Vector, mathutils.Vector]:
    pts: list[mathutils.Vector] = []
    for obj in objects:
        if obj.type != "MESH" or obj.hide_render:
            continue
        pts.extend(obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box)
    if not pts:
        return mathutils.Vector((0, 0, 0)), mathutils.Vector((5, 5, 3))
    return (
        mathutils.Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts))),
        mathutils.Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts))),
    )


def object_world_bbox(obj: bpy.types.Object) -> tuple[mathutils.Vector, mathutils.Vector] | None:
    if obj.type != "MESH" or obj.hide_render:
        return None
    pts = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
    if not pts:
        return None
    return (
        mathutils.Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts))),
        mathutils.Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts))),
    )


def hide_object_family(root: bpy.types.Object) -> int:
    hidden = 0
    stack = [root]
    while stack:
        obj = stack.pop()
        if not obj.hide_render or not obj.hide_viewport:
            hidden += 1
        obj.hide_render = True
        obj.hide_viewport = True
        stack.extend(list(obj.children))
    return hidden


def hide_outlier_meshes(max_abs: float = 100.0, max_size: float = 30.0) -> list[dict]:
    """Hide imported supplier geometry that has broken transforms and ruins camera bounds."""
    bpy.context.view_layer.update()
    hidden: list[dict] = []
    for obj in list(bpy.data.objects):
        bounds = object_world_bbox(obj)
        if bounds is None:
            continue
        bmin, bmax = bounds
        size = bmax - bmin
        max_dim = max(float(abs(size.x)), float(abs(size.y)), float(abs(size.z)))
        max_coord = max(abs(float(v)) for v in [bmin.x, bmax.x, bmin.y, bmax.y, bmin.z, bmax.z])
        if max_coord <= max_abs and max_dim <= max_size:
            continue
        hide_object_family(obj)
        hidden.append(
            {
                "name": obj.name,
                "source_room_id": str(obj.get("source_room_id") or ""),
                "max_abs": round(max_coord, 4),
                "max_size": round(max_dim, 4),
                "min": [round(float(bmin.x), 4), round(float(bmin.y), 4), round(float(bmin.z), 4)],
                "max": [round(float(bmax.x), 4), round(float(bmax.y), 4), round(float(bmax.z), 4)],
            }
        )
    if hidden:
        print(f"[assemble_apartment] hidden_outlier_meshes={len(hidden)}")
    return hidden


def setup_camera_and_light() -> None:
    mesh_objects = [obj for obj in bpy.data.objects if obj.type == "MESH" and not obj.hide_render]
    bmin, bmax = world_bbox(mesh_objects)
    center = (bmin + bmax) * 0.5
    size = bmax - bmin
    radius = max(size.x, size.y, size.z, 1.0)

    cam_data = bpy.data.cameras.new("ApartmentCamera")
    cam = bpy.data.objects.new("ApartmentCamera", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    cam.location = (center.x + radius * 0.75, center.y - radius * 0.95, center.z + radius * 0.72)

    target = bpy.data.objects.new("ApartmentCameraTarget", None)
    bpy.context.scene.collection.objects.link(target)
    target.location = center
    con = cam.constraints.new(type="TRACK_TO")
    con.target = target
    con.track_axis = "TRACK_NEGATIVE_Z"
    con.up_axis = "UP_Y"
    cam.data.lens = 24
    bpy.context.scene.camera = cam

    light_data = bpy.data.lights.new("ApartmentKeyArea", type="AREA")
    light = bpy.data.objects.new("ApartmentKeyArea", light_data)
    bpy.context.scene.collection.objects.link(light)
    light.location = (center.x, center.y, bmax.z + max(radius * 0.35, 2.0))
    light.data.energy = 500
    light.data.size = max(radius, 3.0)


def setup_render(width: int, height: int, samples: int) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
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


def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Assemble ready room .blend files into one apartment .blend.")
    p.add_argument("--apt-dir", required=True)
    p.add_argument("--mode", default="optimal")
    p.add_argument("--apartment-scene", default=None)
    p.add_argument("--save-blend", required=True)
    p.add_argument("--render", default=None)
    p.add_argument("--build-report", default=None)
    p.add_argument("--width", type=int, default=1400)
    p.add_argument("--height", type=int, default=1000)
    p.add_argument("--samples", type=int, default=64)
    return p


def main() -> None:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = argv[1:]
    args = build_cli().parse_args(argv)
    apt_dir = Path(args.apt_dir).expanduser().resolve()
    manifest = read_json(apt_dir / "manifest.json")
    apartment_json = apt_dir / "apartment.json"
    apartment_scene = Path(args.apartment_scene).expanduser().resolve() if args.apartment_scene else None
    apt_min = apt_min_from_scene(apartment_scene, apartment_json)

    clear_scene()
    root_collection = bpy.data.collections.new("ApartmentRooms")
    bpy.context.scene.collection.children.link(root_collection)

    reports = []
    for room_meta in manifest.get("rooms") or []:
        room_id = str(room_meta.get("room_id") or "")
        if not room_id:
            continue
        room_dir = apt_dir / "rooms" / room_id
        scene_path = room_scene_path(room_dir, args.mode)
        if not scene_path.is_file():
            reports.append({"room_id": room_id, "status": "missing_scene_requirements", "path": str(scene_path)})
            continue
        blend_path = find_room_blend(room_dir, room_id, args.mode)
        if blend_path is None:
            reports.append({"room_id": room_id, "status": "missing_room_blend"})
            continue
        scene = read_json(scene_path)
        frame = (((scene.get("room") or {}).get("meta") or {}).get("coordinate_frame") or {})
        if not frame:
            reports.append({"room_id": room_id, "status": "missing_coordinate_frame", "blend": str(blend_path)})
            continue
        report = append_room_blend(blend_path, room_id, transform_from_frame(frame, apt_min), root_collection)
        report["status"] = "ok"
        reports.append(report)

    hidden_outlier_objects = hide_outlier_meshes()
    setup_camera_and_light()
    setup_render(args.width, args.height, args.samples)

    save_path = Path(args.save_blend).expanduser().resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(save_path))

    render_path = None
    if args.render:
        render_path = Path(args.render).expanduser().resolve()
        render_path.parent.mkdir(parents=True, exist_ok=True)
        bpy.context.scene.render.filepath = str(render_path)
        bpy.ops.render.render(write_still=True)

    if args.build_report:
        bmin, bmax = world_bbox([obj for obj in bpy.data.objects if obj.type == "MESH"])
        report_path = Path(args.build_report).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(
                {
                    "apartment_dir": str(apt_dir),
                    "save_blend": str(save_path),
                    "render": str(render_path) if render_path else None,
                    "apartment_global_min_xy": list(apt_min),
                    "room_reports": reports,
                    "hidden_outlier_object_count": len(hidden_outlier_objects),
                    "hidden_outlier_objects": hidden_outlier_objects[:200],
                    "object_count": len(bpy.data.objects),
                    "mesh_count": sum(1 for obj in bpy.data.objects if obj.type == "MESH"),
                    "bounds": {
                        "x_min": round(float(bmin.x), 4),
                        "x_max": round(float(bmax.x), 4),
                        "y_min": round(float(bmin.y), 4),
                        "y_max": round(float(bmax.y), 4),
                        "z_min": round(float(bmin.z), 4),
                        "z_max": round(float(bmax.z), 4),
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
