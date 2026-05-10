#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import bpy
import mathutils

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    from topview_vlm_orientation_repair import collect_scene_objects, filter_target_objects
except Exception:
    collect_scene_objects = None
    filter_target_objects = None


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


def _norm_name(value: str) -> str:
    value = str(value or "").strip().lower()
    value = re.sub(r"\.\d{3}$", "", value)
    value = re.sub(r"[^0-9a-zа-яё]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _descendant_meshes(root: bpy.types.Object) -> list[bpy.types.Object]:
    out: list[bpy.types.Object] = []
    stack = [root]
    seen: set[str] = set()
    while stack:
        obj = stack.pop()
        if obj.name in seen:
            continue
        seen.add(obj.name)
        if obj.type == "MESH":
            out.append(obj)
        stack.extend(list(obj.children))
    return out


def _objects_for_scene_ref(ref) -> list[bpy.types.Object]:
    object_id = str(getattr(ref, "object_id", "") or "")
    name = str(getattr(ref, "name", "") or "")
    candidates: list[bpy.types.Object] = []

    for direct_name in (object_id, f"{object_id}_supplier_root", name):
        if direct_name and direct_name in bpy.data.objects:
            candidates.append(bpy.data.objects[direct_name])

    prop_keys = {
        "supplier_dining_item_id",
        "cgs_item_id",
        "cgs_object_id",
        "cgs_placement_id",
        "placement_id",
        "item_id",
        "object_id",
        "source_item_id",
        "scene_item_id",
    }
    for obj in bpy.context.scene.objects:
        for key in obj.keys():
            if key in prop_keys or key.endswith("_item_id") or key.endswith("_object_id"):
                if str(obj.get(key)) == object_id:
                    candidates.append(obj.parent if obj.parent is not None else obj)
                    break

    norm_name = _norm_name(name)
    if norm_name:
        for obj in bpy.context.scene.objects:
            if _norm_name(obj.name) == norm_name:
                candidates.append(obj)

    unique: list[bpy.types.Object] = []
    seen: set[str] = set()
    for obj in candidates:
        if obj is None:
            continue
        if obj.name in seen:
            continue
        seen.add(obj.name)
        unique.append(obj)
    return unique


def _apply_scene_orientations(
    scene_json: Path,
    target_ids: set[str],
    target_scope: str,
    include_armchairs: bool,
    report_path: Path | None,
) -> dict:
    report = {
        "scene_json": str(scene_json),
        "target_ids": sorted(target_ids),
        "applied": [],
        "skipped": [],
    }
    if collect_scene_objects is None or filter_target_objects is None:
        report["skipped"].append({"reason": "topview_vlm_orientation_repair_import_failed"})
        if report_path:
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report
    data = json.loads(scene_json.read_text(encoding="utf-8"))
    refs = collect_scene_objects(data, max_objects=10000)
    target_refs = filter_target_objects(refs, scope=target_scope, include_armchairs=include_armchairs)
    for ref in target_refs:
        object_id = str(ref.object_id)
        if target_ids and object_id not in target_ids:
            continue
        yaw_deg = ref.yaw_deg
        if yaw_deg is None:
            report["skipped"].append({"object_id": object_id, "reason": "missing_yaw"})
            continue
        roots = _objects_for_scene_ref(ref)
        if not roots:
            report["skipped"].append(
                {
                    "object_id": object_id,
                    "name": ref.name,
                    "reason": "blend_object_not_found",
                }
            )
            continue
        for root in roots:
            old_yaw = math.degrees(float(root.rotation_euler.z))
            root.rotation_euler.z = math.radians(float(yaw_deg))
            root["cgs_topview_vlm_object_id"] = object_id
            root["cgs_topview_vlm_applied_yaw_deg"] = float(yaw_deg)
            report["applied"].append(
                {
                    "object_id": object_id,
                    "name": ref.name,
                    "blend_object": root.name,
                    "old_yaw_deg": old_yaw % 360.0,
                    "new_yaw_deg": float(yaw_deg) % 360.0,
                }
            )
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _highlight_scene_targets(scene_json: Path, target_ids: set[str], target_scope: str, include_armchairs: bool) -> int:
    if not target_ids or collect_scene_objects is None or filter_target_objects is None:
        return 0
    data = json.loads(scene_json.read_text(encoding="utf-8"))
    refs = filter_target_objects(
        collect_scene_objects(data, max_objects=10000),
        scope=target_scope,
        include_armchairs=include_armchairs,
    )
    material = bpy.data.materials.get("CGS_Topview_Target_Red")
    if material is None:
        material = bpy.data.materials.new("CGS_Topview_Target_Red")
        material.diffuse_color = (1.0, 0.08, 0.02, 1.0)
    count = 0
    for ref in refs:
        if str(ref.object_id) not in target_ids:
            continue
        for root in _objects_for_scene_ref(ref):
            for mesh in _descendant_meshes(root):
                mesh.data.materials.clear()
                mesh.data.materials.append(material)
                count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--azimuth-deg", type=float, default=-135.0)
    parser.add_argument("--elevation-deg", type=float, default=52.0)
    parser.add_argument("--radius-mult", type=float, default=1.18)
    parser.add_argument("--lens", type=float, default=24.0)
    parser.add_argument("--resolution-x", type=int, default=1400)
    parser.add_argument("--resolution-y", type=int, default=1050)
    parser.add_argument("--scene-json", type=Path, default=None)
    parser.add_argument("--target-ids", default="")
    parser.add_argument("--target-scope", choices=["chairs", "all"], default="chairs")
    parser.add_argument("--include-armchairs", action="store_true")
    parser.add_argument("--apply-scene-orientations", action="store_true")
    parser.add_argument("--highlight-targets", action="store_true")
    parser.add_argument("--orientation-report", type=Path, default=None)
    parser.add_argument("--save-blend", type=Path, default=None)
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    args = parser.parse_args(argv)

    scene = bpy.context.scene
    target_ids = {item.strip() for item in str(args.target_ids or "").split(",") if item.strip()}
    if args.scene_json and args.apply_scene_orientations:
        report = _apply_scene_orientations(
            args.scene_json.expanduser().resolve(),
            target_ids,
            str(args.target_scope),
            bool(args.include_armchairs),
            args.orientation_report.expanduser().resolve() if args.orientation_report else None,
        )
        print(f"Applied scene orientations: {len(report.get('applied') or [])}")
    if args.scene_json and args.highlight_targets:
        highlighted = _highlight_scene_targets(
            args.scene_json.expanduser().resolve(),
            target_ids,
            str(args.target_scope),
            bool(args.include_armchairs),
        )
        print(f"Highlighted target mesh objects: {highlighted}")
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
    if args.save_blend:
        save_path = args.save_blend.expanduser().resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(save_path))
        print(f"Saved blend: {save_path}")


if __name__ == "__main__":
    main()
