#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import bpy
import bmesh
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


def _hide_exterior_shell_objects() -> int:
    hidden = 0
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH" or obj.hide_get() or obj.hide_render:
            continue
        name = obj.name.lower()
        is_exterior = (
            ".exterior" in name
            or name.endswith("/0.exterior")
            or name.endswith("_exterior")
            or "room_exterior" in name
            or "preview_exterior" in name
        )
        if is_exterior:
            obj.hide_render = True
            obj.hide_set(True)
            hidden += 1
    return hidden


def _delete_large_top_cap_faces(bb_min: mathutils.Vector, bb_max: mathutils.Vector) -> int:
    scene_z_max = float(bb_max.z)
    xy_area = max(float((bb_max.x - bb_min.x) * (bb_max.y - bb_min.y)), 1e-6)
    removed = 0
    for obj in list(bpy.context.scene.objects):
        if obj.type != "MESH" or obj.hide_get() or obj.hide_render:
            continue
        name = obj.name.lower()
        if any(token in name for token in ("floor", "rug", "carpet", "mattress", "pillow", "blanket", "lamp", "chair", "table", "desk", "cabinet", "shelf")):
            continue
        corners = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
        if not corners:
            continue
        z_max = max(float(v.z) for v in corners)
        if z_max < scene_z_max - 0.35:
            continue

        mesh = obj.data
        bm = bmesh.new()
        try:
            bm.from_mesh(mesh)
            bm.faces.ensure_lookup_table()
            normal_matrix = obj.matrix_world.to_3x3().inverted().transposed()
            faces_to_delete = []
            for face in bm.faces:
                world_center = obj.matrix_world @ face.calc_center_median()
                if float(world_center.z) < scene_z_max - 0.35:
                    continue
                world_normal = (normal_matrix @ face.normal).normalized()
                if abs(float(world_normal.z)) < 0.72:
                    continue
                world_verts = [obj.matrix_world @ vert.co for vert in face.verts]
                x_span = max(float(v.x) for v in world_verts) - min(float(v.x) for v in world_verts)
                y_span = max(float(v.y) for v in world_verts) - min(float(v.y) for v in world_verts)
                if x_span * y_span >= min(0.35, xy_area * 0.03):
                    faces_to_delete.append(face)
            if faces_to_delete and len(faces_to_delete) < len(bm.faces):
                new_mesh = mesh.copy()
                bmesh.ops.delete(bm, geom=faces_to_delete, context="FACES")
                bm.to_mesh(new_mesh)
                new_mesh.update()
                obj.data = new_mesh
                removed += len(faces_to_delete)
            elif faces_to_delete and len(faces_to_delete) == len(bm.faces):
                obj.hide_render = True
                obj.hide_set(True)
                removed += len(faces_to_delete)
        finally:
            bm.free()
    return removed


def _is_wall_render_candidate(obj: bpy.types.Object) -> bool:
    if obj.type != "MESH":
        return False
    name = obj.name.lower()
    if any(token in name for token in ("floor", "ceiling", "roof", "topview_label", "camera", "bbox", "axis")):
        return False
    if any(token in name for token in ("wall_art", "wallart", "wall lamp", "walllamp", "wall_light")):
        return False
    is_room_shell = (
        name.endswith("/0")
        or name.endswith("/0.meshed")
        or name.endswith("/0.wall")
        or name.endswith("/0.exterior")
        or ".wall" in name
        or ".exterior" in name
    )
    return (
        is_room_shell
        or
        name.startswith("preview_wall_")
        or
        "room_wall" in name
        or "room_wallpaper" in name
        or "/0.wall" in name
        or name.endswith(".wall")
        or name.endswith("_wall")
        or name == "wall"
    )


def _capture_wall_state() -> dict[str, tuple[bool, bool, bpy.types.Mesh]]:
    state: dict[str, tuple[bool, bool, bpy.types.Mesh]] = {}
    for obj in bpy.context.scene.objects:
        if not _is_wall_render_candidate(obj):
            continue
        state[obj.name] = (bool(obj.hide_get()), bool(obj.hide_render), obj.data)
    return state


def _restore_wall_state(state: dict[str, tuple[bool, bool, bpy.types.Mesh]]) -> None:
    for name, (hide_viewport, hide_render, mesh) in state.items():
        obj = bpy.data.objects.get(name)
        if obj is None:
            continue
        obj.data = mesh
        obj.hide_render = hide_render
        obj.hide_set(hide_viewport)


def _hide_preview_wall_ids(wall_ids: set[str]) -> int:
    if not wall_ids:
        return 0
    hidden = 0
    prefixes = tuple(f"preview_wall_{wall_id}_" for wall_id in wall_ids)
    for obj in bpy.context.scene.objects:
        name = obj.name
        if not name.startswith(prefixes):
            continue
        obj.hide_render = True
        obj.hide_set(True)
        hidden += 1
    return hidden


def _side_key_for_point(
    point: mathutils.Vector,
    center: mathutils.Vector,
    half_x: float,
    half_y: float,
) -> str:
    rel_x = (point.x - center.x) / max(half_x, 1e-6)
    rel_y = (point.y - center.y) / max(half_y, 1e-6)
    if abs(rel_x) >= abs(rel_y):
        return "x_pos" if rel_x >= 0.0 else "x_neg"
    return "y_pos" if rel_y >= 0.0 else "y_neg"


def _nearest_wall_side_keys(camera_location: mathutils.Vector, center: mathutils.Vector) -> set[str]:
    return {
        "x_pos" if camera_location.x >= center.x else "x_neg",
        "y_pos" if camera_location.y >= center.y else "y_neg",
    }


def _point_is_on_hidden_side(
    point: mathutils.Vector,
    center: mathutils.Vector,
    half_x: float,
    half_y: float,
    hidden_sides: set[str],
    threshold: float = 0.30,
) -> bool:
    rel_x = (point.x - center.x) / max(half_x, 1e-6)
    rel_y = (point.y - center.y) / max(half_y, 1e-6)
    return (
        ("x_pos" in hidden_sides and rel_x >= threshold)
        or ("x_neg" in hidden_sides and rel_x <= -threshold)
        or ("y_pos" in hidden_sides and rel_y >= threshold)
        or ("y_neg" in hidden_sides and rel_y <= -threshold)
    )


def _hide_nearest_room_walls(
    camera_location: mathutils.Vector,
    bb_min: mathutils.Vector,
    bb_max: mathutils.Vector,
) -> int:
    center = (bb_min + bb_max) * 0.5
    half_x = max(float(bb_max.x - bb_min.x) * 0.5, 1e-6)
    half_y = max(float(bb_max.y - bb_min.y) * 0.5, 1e-6)
    hidden_sides = _nearest_wall_side_keys(camera_location, center)
    hidden_count = 0

    for obj in list(bpy.context.scene.objects):
        if not _is_wall_render_candidate(obj) or obj.hide_get() or obj.hide_render:
            continue
        corners = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
        if not corners:
            continue
        obj_center = mathutils.Vector(
            (
                (min(v.x for v in corners) + max(v.x for v in corners)) * 0.5,
                (min(v.y for v in corners) + max(v.y for v in corners)) * 0.5,
                (min(v.z for v in corners) + max(v.z for v in corners)) * 0.5,
            )
        )
        if (
            _point_is_on_hidden_side(obj_center, center, half_x, half_y, hidden_sides, threshold=0.40)
            and max(v.z for v in corners) - min(v.z for v in corners) > 0.45
            and max(max(v.x for v in corners) - min(v.x for v in corners), max(v.y for v in corners) - min(v.y for v in corners)) <= max(half_x, half_y) * 1.25
        ):
            obj.hide_render = True
            obj.hide_set(True)
            hidden_count += 1
            continue

        mesh = obj.data
        bm = bmesh.new()
        try:
            bm.from_mesh(mesh)
            bm.faces.ensure_lookup_table()
            faces_to_delete = []
            normal_matrix = obj.matrix_world.to_3x3().inverted().transposed()
            for face in bm.faces:
                face_center = obj.matrix_world @ face.calc_center_median()
                if face_center.z < bb_min.z + 0.25:
                    continue
                world_normal = (normal_matrix @ face.normal).normalized()
                if abs(float(world_normal.z)) > 0.45:
                    continue
                world_verts = [obj.matrix_world @ vert.co for vert in face.verts]
                side_hits = sum(
                    1
                    for vert in world_verts
                    if _point_is_on_hidden_side(vert, center, half_x, half_y, hidden_sides, threshold=0.30)
                )
                if _point_is_on_hidden_side(face_center, center, half_x, half_y, hidden_sides, threshold=0.22) or side_hits >= max(1, len(world_verts) // 2):
                    faces_to_delete.append(face)
            if faces_to_delete and len(faces_to_delete) < len(bm.faces):
                new_mesh = mesh.copy()
                bmesh.ops.delete(bm, geom=faces_to_delete, context="FACES")
                bm.to_mesh(new_mesh)
                new_mesh.update()
                obj.data = new_mesh
                hidden_count += len(faces_to_delete)
            elif faces_to_delete and len(faces_to_delete) == len(bm.faces):
                obj.hide_render = True
                obj.hide_set(True)
                hidden_count += len(faces_to_delete)
        finally:
            bm.free()
    return hidden_count


def _look_at(obj: bpy.types.Object, target: mathutils.Vector) -> None:
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _relative_xy_point(
    center: mathutils.Vector,
    xy_span: float,
    azimuth_deg: float,
    radius_mult: float,
) -> mathutils.Vector:
    az = math.radians(float(azimuth_deg))
    radius = float(xy_span) * float(radius_mult)
    return mathutils.Vector((
        center.x + radius * math.cos(az),
        center.y + radius * math.sin(az),
        center.z,
    ))


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


def _bounds_for_roots(roots: list[bpy.types.Object]) -> tuple[mathutils.Vector, mathutils.Vector] | None:
    corners: list[mathutils.Vector] = []
    for root in roots:
        meshes = _descendant_meshes(root)
        if not meshes and root.type == "MESH":
            meshes = [root]
        for mesh in meshes:
            for corner in mesh.bound_box:
                corners.append(mesh.matrix_world @ mathutils.Vector(corner))
    if not corners:
        locs = [root.matrix_world.translation.copy() for root in roots if root is not None]
        if not locs:
            return None
        corners = locs
    return (
        mathutils.Vector((min(v.x for v in corners), min(v.y for v in corners), min(v.z for v in corners))),
        mathutils.Vector((max(v.x for v in corners), max(v.y for v in corners), max(v.z for v in corners))),
    )


def _material(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    material = bpy.data.materials.get(name)
    if material is None:
        material = bpy.data.materials.new(name)
    material.diffuse_color = color
    return material


def _add_target_label(label: str, roots: list[bpy.types.Object], scene_center: mathutils.Vector) -> bool:
    bounds = _bounds_for_roots(roots)
    if bounds is None:
        return False
    bb_min, bb_max = bounds
    obj_center = mathutils.Vector(((bb_min.x + bb_max.x) * 0.5, (bb_min.y + bb_max.y) * 0.5, bb_max.z + 0.12))
    label_upper = str(label).strip().upper()
    if label_upper.startswith("T"):
        loc = obj_center
        plane_size = 0.34
        font_size = 0.20
    else:
        direction = mathutils.Vector((obj_center.x - scene_center.x, obj_center.y - scene_center.y, 0.0))
        if direction.length < 1e-5:
            direction = mathutils.Vector((1.0, 0.0, 0.0))
        direction.normalize()
        xy_span = max(float(bb_max.x - bb_min.x), float(bb_max.y - bb_min.y), 0.2)
        loc = obj_center + direction * (xy_span * 0.75 + 0.34)
        plane_size = 0.46
        font_size = 0.27

    bg_mat = _material("CGS_Topview_Label_Background", (1.0, 0.92, 0.08, 1.0))
    text_mat = _material("CGS_Topview_Label_Text", (0.0, 0.0, 0.0, 1.0))

    bpy.ops.mesh.primitive_plane_add(size=plane_size, location=(loc.x, loc.y, loc.z - 0.006), rotation=(0.0, 0.0, 0.0))
    bg = bpy.context.object
    bg.name = f"CGS_Topview_Label_BG_{label}"
    bg.data.materials.append(bg_mat)

    text_data = bpy.data.curves.new(f"CGS_Topview_Label_{label}", "FONT")
    text_data.body = label
    text_data.align_x = "CENTER"
    text_data.align_y = "CENTER"
    text_data.size = font_size
    text_obj = bpy.data.objects.new(f"CGS_Topview_Label_{label}", text_data)
    text_obj.location = loc
    text_obj.data.materials.append(text_mat)
    bpy.context.scene.collection.objects.link(text_obj)
    return True


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


def _highlight_scene_targets(
    scene_json: Path,
    target_ids: set[str],
    target_scope: str,
    include_armchairs: bool,
    label_by_id: dict[str, str] | None = None,
    highlight_style: str = "material",
) -> int:
    if not target_ids or collect_scene_objects is None or filter_target_objects is None:
        return 0
    data = json.loads(scene_json.read_text(encoding="utf-8"))
    refs = filter_target_objects(
        collect_scene_objects(data, max_objects=10000),
        scope=target_scope,
        include_armchairs=include_armchairs,
    )
    material = _material("CGS_Topview_Target_Red", (1.0, 0.08, 0.02, 1.0))
    scene_bb_min, scene_bb_max = _visible_mesh_bounds()
    scene_center = (scene_bb_min + scene_bb_max) * 0.5
    count = 0
    for ref in refs:
        if str(ref.object_id) not in target_ids:
            continue
        roots = _objects_for_scene_ref(ref)
        if highlight_style == "material":
            for root in roots:
                for mesh in _descendant_meshes(root):
                    mesh.data.materials.clear()
                    mesh.data.materials.append(material)
                    count += 1
        elif highlight_style in {"label_only", "none"}:
            count += len(roots)
        else:
            raise ValueError(f"Unsupported highlight_style: {highlight_style}")
        if label_by_id and highlight_style != "none":
            _add_target_label(label_by_id.get(str(ref.object_id), str(ref.object_id)), roots, scene_center)
    return count


def _label_scene_refs(
    scene_json: Path,
    label_ids: set[str],
    label_by_id: dict[str, str],
) -> int:
    if not label_ids or collect_scene_objects is None:
        return 0
    data = json.loads(scene_json.read_text(encoding="utf-8"))
    refs = collect_scene_objects(data, max_objects=10000)
    scene_bb_min, scene_bb_max = _visible_mesh_bounds()
    scene_center = (scene_bb_min + scene_bb_max) * 0.5
    count = 0
    for ref in refs:
        object_id = str(ref.object_id)
        if object_id not in label_ids:
            continue
        roots = _objects_for_scene_ref(ref)
        if not roots:
            continue
        if _add_target_label(label_by_id.get(object_id, object_id), roots, scene_center):
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--azimuth-deg", type=float, default=-135.0)
    parser.add_argument("--elevation-deg", type=float, default=52.0)
    parser.add_argument("--radius-mult", type=float, default=1.18)
    parser.add_argument("--lens", type=float, default=24.0)
    parser.add_argument(
        "--view-specs-json",
        type=Path,
        default=None,
        help="Optional JSON list of {name,out,azimuth_deg,elevation_deg,radius_mult,lens} views to render in one Blender process.",
    )
    parser.add_argument("--resolution-x", type=int, default=1400)
    parser.add_argument("--resolution-y", type=int, default=1050)
    parser.add_argument(
        "--render-engine",
        choices=["eevee", "workbench"],
        default="eevee",
        help="Render engine for review frames. workbench uses material-color viewport preview style.",
    )
    parser.add_argument("--transparent-background", action="store_true")
    parser.add_argument("--hide-nearest-walls", action="store_true")
    parser.add_argument("--hide-exterior", action="store_true")
    parser.add_argument("--hide-wall-ids", default="", help="Comma-separated preview wall ids to hide, e.g. w0,w1. Window/door opening meshes are preserved.")
    parser.add_argument("--scene-json", type=Path, default=None)
    parser.add_argument("--target-ids", default="")
    parser.add_argument("--label-ids", default="")
    parser.add_argument("--target-label-map", type=Path, default=None)
    parser.add_argument("--target-scope", choices=["chairs", "all"], default="chairs")
    parser.add_argument("--include-armchairs", action="store_true")
    parser.add_argument("--apply-scene-orientations", action="store_true")
    parser.add_argument("--highlight-targets", action="store_true")
    parser.add_argument("--highlight-style", choices=["material", "label_only", "none"], default="material")
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
    label_ids = {item.strip() for item in str(args.label_ids or "").split(",") if item.strip()}
    label_by_id: dict[str, str] = {}
    if args.target_label_map and args.target_label_map.is_file():
        raw_map = json.loads(args.target_label_map.read_text(encoding="utf-8"))
        if isinstance(raw_map, dict):
            label_by_id = {str(object_id): str(label) for label, object_id in raw_map.items()}
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
            None if label_ids else label_by_id,
            str(args.highlight_style),
        )
        print(f"Highlighted target mesh objects: {highlighted}")
    if args.scene_json and label_ids:
        labeled = _label_scene_refs(
            args.scene_json.expanduser().resolve(),
            label_ids,
            label_by_id,
        )
        print(f"Labeled scene objects: {labeled}")
    if args.hide_exterior:
        hidden_exteriors = _hide_exterior_shell_objects()
        print(f"Hidden exterior shell objects: {hidden_exteriors}")
    hidden_ceilings = _hide_ceiling_caps()
    print(f"Hidden ceiling/top cap objects: {hidden_ceilings}")
    hidden_wall_ids = {item.strip() for item in str(args.hide_wall_ids or "").split(",") if item.strip()}
    hidden_explicit_walls = _hide_preview_wall_ids(hidden_wall_ids)
    if hidden_explicit_walls:
        print(f"Hidden explicit preview wall objects: {hidden_explicit_walls} ({','.join(sorted(hidden_wall_ids))})")
    bb_min, bb_max = _visible_mesh_bounds()
    deleted_top_faces = _delete_large_top_cap_faces(bb_min, bb_max)
    if deleted_top_faces:
        print(f"Deleted ceiling/top cap faces: {deleted_top_faces}")
        bb_min, bb_max = _visible_mesh_bounds()
    wall_state = _capture_wall_state()
    center = (bb_min + bb_max) * 0.5
    dims = bb_max - bb_min
    xy_span = max(float(dims.x), float(dims.y), 1.0)
    target = mathutils.Vector((center.x, center.y, bb_min.z + max(float(dims.z) * 0.32, 1.0)))

    cam = bpy.data.objects.get("CGS_TopInspectionCamera")
    if cam is None or cam.type != "CAMERA":
        cam_data = bpy.data.cameras.new("CGS_TopInspectionCamera")
        cam = bpy.data.objects.new("CGS_TopInspectionCamera", cam_data)
        scene.collection.objects.link(cam)
    if str(args.render_engine) == "workbench":
        scene.render.engine = "BLENDER_WORKBENCH"
        try:
            scene.display.shading.light = "STUDIO"
            scene.display.shading.color_type = "MATERIAL"
            scene.display.shading.show_xray = False
            scene.display.shading.show_shadows = True
            scene.display.shading.show_cavity = True
        except Exception:
            pass
    else:
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
    scene.render.film_transparent = bool(args.transparent_background)
    try:
        scene.view_settings.view_transform = "Filmic"
        scene.view_settings.look = "Medium High Contrast"
        scene.view_settings.exposure = 0.0
        scene.view_settings.gamma = 1.0
    except Exception:
        pass

    def render_view(
        out_path: Path,
        azimuth_deg: float,
        elevation_deg: float,
        radius_mult: float,
        lens: float,
        hide_nearest_walls: bool = False,
        spec: dict | None = None,
    ) -> None:
        spec = spec or {}
        _restore_wall_state(wall_state)
        camera_mode = str(spec.get("camera_mode", "orbit")).strip().lower()
        if camera_mode == "interior":
            camera_z = float(spec.get("camera_z", bb_min.z + max(float(dims.z) * 0.46, 1.35)))
            target_z = float(spec.get("target_z", bb_min.z + max(float(dims.z) * 0.30, 0.95)))
            cam_xy = _relative_xy_point(
                center,
                xy_span,
                float(azimuth_deg),
                float(spec.get("camera_radius_mult", radius_mult)),
            )
            look_azimuth = float(spec.get("look_azimuth_deg", float(azimuth_deg) + 180.0))
            look_xy = _relative_xy_point(
                center,
                xy_span,
                look_azimuth,
                float(spec.get("look_radius_mult", 0.14)),
            )
            cam.location = (cam_xy.x, cam_xy.y, camera_z)
            view_target = mathutils.Vector((look_xy.x, look_xy.y, target_z))
        else:
            radius = xy_span * float(radius_mult)
            az = math.radians(float(azimuth_deg))
            elev = math.radians(float(elevation_deg))
            cam.location = (
                target.x + radius * math.cos(az),
                target.y + radius * math.sin(az),
                target.z + math.tan(elev) * radius,
            )
            view_target = target
        if float(elevation_deg) >= 78.0 and camera_mode != "interior":
            cam.data.type = "ORTHO"
            cam.data.ortho_scale = max(xy_span * 1.12, 1.0)
        else:
            cam.data.type = "PERSP"
            cam.data.lens = float(lens)
        cam.data.clip_start = 0.02
        cam.data.clip_end = 300.0
        _look_at(cam, view_target)
        if hide_nearest_walls:
            hidden_walls = _hide_nearest_room_walls(cam.location, bb_min, bb_max)
            print(f"Hidden nearest wall side elements: {hidden_walls}")
        scene.camera = cam
        scene.render.filepath = str(out_path.expanduser().resolve())
        if str(args.render_engine) == "workbench":
            try:
                bpy.ops.render.opengl(write_still=True, view_context=False)
            except Exception:
                bpy.ops.render.render(write_still=True)
        else:
            bpy.ops.render.render(write_still=True)
        print(f"Saved render: {scene.render.filepath}")
        _restore_wall_state(wall_state)

    if args.view_specs_json:
        specs = json.loads(args.view_specs_json.expanduser().read_text(encoding="utf-8"))
        if not isinstance(specs, list):
            raise ValueError("--view-specs-json must contain a JSON list")
        for spec in specs:
            if not isinstance(spec, dict):
                continue
            render_view(
                Path(spec.get("out") or args.out),
                float(spec.get("azimuth_deg", args.azimuth_deg)),
                float(spec.get("elevation_deg", args.elevation_deg)),
                float(spec.get("radius_mult", args.radius_mult)),
                float(spec.get("lens", args.lens)),
                bool(spec.get("hide_nearest_walls", False)),
                spec,
            )
    else:
        render_view(
            Path(args.out),
            float(args.azimuth_deg),
            float(args.elevation_deg),
            float(args.radius_mult),
            float(args.lens),
            bool(args.hide_nearest_walls),
            None,
        )
    if args.save_blend:
        save_path = args.save_blend.expanduser().resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(save_path))
        print(f"Saved blend: {save_path}")


if __name__ == "__main__":
    main()
