# -*- coding: utf-8 -*-
# src/Plasement/blender_scene_builder.py
#
# Scene builder: room-spec / scene.v1 + placement JSON.
# Главная цель: корректные текстуры.
# Логика:
#   - если keep_existing_mats=True (по умолчанию), НЕ затираем авторские материалы OBJ/MTL
#   - пытаемся восстановить текстуры: relink TEX_IMAGE -> map_Kd из MTL -> largest-image fallback
#   - если материалов нет, создаём PBR (Principled) и назначаем
#
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import random
import traceback
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, cast

import bpy
import bmesh
import mathutils


# ============================================================
# CLI
# ============================================================

def _parse_argv(argv: List[str]) -> argparse.Namespace:
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    ap = argparse.ArgumentParser()
    ap.add_argument("--glb", default=None, help="Compat arg, ignored by builder")
    ap.add_argument("--json", required=True)

    ap.add_argument("--project-root", default=None)  # compat, unused
    ap.add_argument("--import-glb", action="store_true", help="Compat flag, ignored by builder")
    ap.add_argument("--save-blend", default=None)
    ap.add_argument("--render", default=None)
    ap.add_argument("--draw-aabb", action="store_true")
    ap.add_argument(
        "--no-bbox-fallback",
        action="store_true",
        help="Disable default bbox fallback for items whose mesh could not be imported or resolved.",
    )
    ap.add_argument("--reference-blend", default=None)
    ap.add_argument("--overlay-bbox-only", action="store_true")
    ap.add_argument("--background", action="store_true")  # compat
    ap.add_argument("--force-tint", action="store_true")
    ap.add_argument("--highlight-item-ids", default=None, help="Comma-separated placement/item ids to draw black bbox around.")
    ap.add_argument("--hide-room-shell", action="store_true", help="Hide walls, ceiling and exterior shell objects before render.")
    ap.add_argument("--turntable-render-dir", default=None, help="Render turntable PNG sequence to directory.")
    ap.add_argument("--turntable-frames", type=int, default=24, help="Frame count for turntable render sequence.")
    ap.add_argument("--turntable-elevation-deg", type=float, default=30.0, help="Camera elevation angle above floor-parallel orbit plane.")

    # Texture policy
    ap.add_argument(
        "--rebuild-materials",
        action="store_true",
        help="Force rebuild PBR materials even if OBJ has materials. Default: keep existing.",
    )
    ap.add_argument(
        "--no-keep-existing-mats",
        action="store_true",
        help="Alias: disable keeping existing materials (same as --rebuild-materials).",
    )

    ap.add_argument("--no-pack-assets", action="store_true", help="Do not pack external files into .blend")
    ap.add_argument("--verbose", action="store_true")

    # Room environment textures
    ap.add_argument(
        "--env-textures-dir",
        default="data/sourse",
        help="Directory with environment textures (floor_*.jpg, wall_*.jpg, window_*.jpg, door_*.jpg)",
    )
    ap.add_argument(
        "--style-text",
        default=None,
        help="Optional text description for future texture selection (placeholder; currently random).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for env texture selection (0 = derive from json path).",
    )

    return ap.parse_args(argv)


# ============================================================
# Logging
# ============================================================

def _log(verbose: bool, msg: str) -> None:
    if verbose:
        print(msg)


# ============================================================
# Scene utils
# ============================================================

def reset_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)

    scene = bpy.context.scene
    try:
        scene.unit_settings.system = "METRIC"
        scene.unit_settings.scale_length = 1.0
    except Exception:
        pass

    try:
        if "BLENDER_EEVEE_NEXT" in bpy.app.build_options:
            scene.render.engine = "BLENDER_EEVEE_NEXT"
        else:
            scene.render.engine = "BLENDER_EEVEE"
    except Exception:
        pass

    for obj in list(bpy.data.objects):
        try:
            bpy.data.objects.remove(obj, do_unlink=True)
        except Exception:
            pass
    for me in list(bpy.data.meshes):
        try:
            bpy.data.meshes.remove(me, do_unlink=True)
        except Exception:
            pass


def ensure_collection(name: str) -> bpy.types.Collection:
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(coll)
    return coll


def clear_collection_objects(coll: bpy.types.Collection) -> None:
    for obj in list(coll.objects):
        try:
            bpy.data.objects.remove(obj, do_unlink=True)
        except Exception:
            pass


def _unlink_from_all_collections(obj: bpy.types.Object) -> None:
    for c in list(obj.users_collection):
        try:
            c.objects.unlink(obj)
        except Exception:
            pass


def _world_bounds_mesh_objects(objs: List[bpy.types.Object]) -> Tuple[mathutils.Vector, mathutils.Vector]:
    bb_min = mathutils.Vector((+1e30, +1e30, +1e30))
    bb_max = mathutils.Vector((-1e30, -1e30, -1e30))

    deps = bpy.context.evaluated_depsgraph_get()
    for o in objs:
        if o.type != "MESH":
            continue
        eo = o.evaluated_get(deps)
        me = eo.to_mesh()
        if not me:
            continue
        mat = eo.matrix_world
        for v in me.vertices:
            p = mat @ v.co
            bb_min.x = min(bb_min.x, p.x)
            bb_min.y = min(bb_min.y, p.y)
            bb_min.z = min(bb_min.z, p.z)
            bb_max.x = max(bb_max.x, p.x)
            bb_max.y = max(bb_max.y, p.y)
            bb_max.z = max(bb_max.z, p.z)
        eo.to_mesh_clear()

    if bb_min.x > bb_max.x:
        z = mathutils.Vector((0.0, 0.0, 0.0))
        return z, z
    return bb_min, bb_max


def _world_bounds_single_mesh_object(obj: bpy.types.Object) -> Tuple[mathutils.Vector, mathutils.Vector]:
    bb_min = mathutils.Vector((+1e30, +1e30, +1e30))
    bb_max = mathutils.Vector((-1e30, -1e30, -1e30))

    if obj.type != "MESH":
        z = mathutils.Vector((0.0, 0.0, 0.0))
        return z, z

    deps = bpy.context.evaluated_depsgraph_get()
    eo = obj.evaluated_get(deps)
    me = eo.to_mesh()
    if not me:
        z = mathutils.Vector((0.0, 0.0, 0.0))
        return z, z

    try:
        mat = eo.matrix_world
        for v in me.vertices:
            p = mat @ v.co
            bb_min.x = min(bb_min.x, p.x)
            bb_min.y = min(bb_min.y, p.y)
            bb_min.z = min(bb_min.z, p.z)
            bb_max.x = max(bb_max.x, p.x)
            bb_max.y = max(bb_max.y, p.y)
            bb_max.z = max(bb_max.z, p.z)
    finally:
        eo.to_mesh_clear()

    if bb_min.x > bb_max.x:
        z = mathutils.Vector((0.0, 0.0, 0.0))
        return z, z
    return bb_min, bb_max


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(float(v) for v in values)
    n = len(values_sorted)
    mid = n // 2
    if n % 2 == 1:
        return values_sorted[mid]
    return 0.5 * (values_sorted[mid - 1] + values_sorted[mid])


def _filter_imported_mesh_outliers(
    objs: List[bpy.types.Object],
) -> Tuple[List[bpy.types.Object], List[Dict[str, float | str]]]:
    mesh_objs = [o for o in objs if o.type == "MESH"]
    if len(mesh_objs) < 3:
        return objs, []

    stats: List[Dict[str, float | str | bpy.types.Object]] = []
    eps = 1e-9
    for obj in mesh_objs:
        bmin, bmax = _world_bounds_single_mesh_object(obj)
        size = bmax - bmin
        longest = max(float(size.x), float(size.y), float(size.z), 0.0)
        shortest = min(float(size.x), float(size.y), float(size.z), 0.0)
        if longest <= eps:
            continue
        area_xy = max(float(size.x) * float(size.y), 0.0)
        area_xz = max(float(size.x) * float(size.z), 0.0)
        area_yz = max(float(size.y) * float(size.z), 0.0)
        stats.append(
            {
                "obj": obj,
                "name": obj.name,
                "sx": float(size.x),
                "sy": float(size.y),
                "sz": float(size.z),
                "diag": float(size.length),
                "longest": longest,
                "shortest": shortest,
                "thin_ratio": shortest / max(longest, eps),
                "footprint": max(area_xy, area_xz, area_yz),
                "volume": max(float(size.x) * float(size.y) * float(size.z), 0.0),
            }
        )

    if len(stats) < 3:
        return objs, []

    median_diag = max(_median([float(s["diag"]) for s in stats]), eps)
    median_longest = max(_median([float(s["longest"]) for s in stats]), eps)
    median_footprint = max(_median([float(s["footprint"]) for s in stats]), eps)
    median_volume = max(_median([float(s["volume"]) for s in stats]), eps)

    keep_meshes: List[bpy.types.Object] = []
    dropped: List[Dict[str, float | str]] = []
    for stat in stats:
        thin_ratio = float(stat["thin_ratio"])
        too_large = (
            float(stat["diag"]) > median_diag * 10.0
            or float(stat["longest"]) > median_longest * 8.0
            or float(stat["footprint"]) > median_footprint * 20.0
        )
        looks_like_helper_plane = (
            thin_ratio < 0.025
            and (
                float(stat["longest"]) > median_longest * 5.0
                or float(stat["footprint"]) > median_footprint * 10.0
            )
        )
        looks_like_flat_outlier = (
            thin_ratio < 0.06
            and float(stat["volume"]) < median_volume * 0.05
            and float(stat["footprint"]) > median_footprint * 8.0
        )

        if too_large and (looks_like_helper_plane or looks_like_flat_outlier):
            dropped.append(
                {
                    "name": str(stat["name"]),
                    "diag": float(stat["diag"]),
                    "longest": float(stat["longest"]),
                    "footprint": float(stat["footprint"]),
                    "thin_ratio": thin_ratio,
                }
            )
            continue

        keep_meshes.append(stat["obj"])

    if not keep_meshes or not dropped:
        return objs, []

    keep_ids = {id(obj) for obj in keep_meshes}
    filtered_objs = [o for o in objs if o.type != "MESH" or id(o) in keep_ids]
    return filtered_objs, dropped


def _drop_material_preview_meshes(
    objs: List[bpy.types.Object],
) -> Tuple[List[bpy.types.Object], List[str]]:
    mesh_objs = [o for o in objs if o.type == "MESH"]
    if len(mesh_objs) < 2:
        return objs, []

    non_preview_meshes = []
    dropped_names: List[str] = []
    for obj in mesh_objs:
        name_l = obj.name.lower().strip()
        if (
            name_l.startswith("mat_")
            or name_l.startswith("material")
            or name_l.startswith("swatch")
        ):
            dropped_names.append(obj.name)
            continue
        non_preview_meshes.append(obj)

    if not non_preview_meshes or not dropped_names:
        return objs, []

    keep_ids = {id(obj) for obj in non_preview_meshes}
    filtered_objs = [o for o in objs if o.type != "MESH" or id(o) in keep_ids]
    return filtered_objs, dropped_names


def _keep_primary_import_cluster(
    objs: List[bpy.types.Object],
) -> Tuple[List[bpy.types.Object], List[Dict[str, float | str]]]:
    mesh_objs = [o for o in objs if o.type == "MESH"]
    if len(mesh_objs) < 3:
        return objs, []

    eps = 1e-9
    stats: List[Dict[str, float | str | bpy.types.Object | mathutils.Vector]] = []
    for obj in mesh_objs:
        bmin, bmax = _world_bounds_single_mesh_object(obj)
        size = bmax - bmin
        longest = max(float(size.x), float(size.y), float(size.z), 0.0)
        if longest <= eps:
            continue
        stats.append(
            {
                "obj": obj,
                "name": obj.name,
                "center": (bmin + bmax) * 0.5,
                "longest": longest,
                "diag": float(size.length),
                "volume": max(float(size.x) * float(size.y) * float(size.z), 0.0),
            }
        )

    if len(stats) < 3:
        return objs, []

    median_longest = max(_median([float(s["longest"]) for s in stats]), eps)
    median_diag = max(_median([float(s["diag"]) for s in stats]), eps)
    base_link_dist = max(median_longest * 3.5, median_diag * 2.5)

    adjacency: Dict[int, List[int]] = {idx: [] for idx in range(len(stats))}
    for i in range(len(stats)):
        ci = stats[i]["center"]
        li = float(stats[i]["longest"])
        for j in range(i + 1, len(stats)):
            cj = stats[j]["center"]
            lj = float(stats[j]["longest"])
            assert isinstance(ci, mathutils.Vector)
            assert isinstance(cj, mathutils.Vector)
            link_dist = max(base_link_dist, max(li, lj) * 2.5)
            if (ci - cj).length <= link_dist:
                adjacency[i].append(j)
                adjacency[j].append(i)

    clusters: List[List[int]] = []
    seen: set[int] = set()
    for start in range(len(stats)):
        if start in seen:
            continue
        queue = [start]
        comp: List[int] = []
        seen.add(start)
        while queue:
            cur = queue.pop(0)
            comp.append(cur)
            for nxt in adjacency[cur]:
                if nxt in seen:
                    continue
                seen.add(nxt)
                queue.append(nxt)
        clusters.append(comp)

    if len(clusters) <= 1:
        return objs, []

    def _cluster_score(indices: List[int]) -> Tuple[int, float, float]:
        total_volume = 0.0
        total_mass = 0.0
        weighted_center = mathutils.Vector((0.0, 0.0, 0.0))
        for idx in indices:
            stat = stats[idx]
            center = stat["center"]
            longest = float(stat["longest"])
            volume = max(float(stat["volume"]), longest ** 3 * 0.02)
            total_volume += volume
            total_mass += volume
            assert isinstance(center, mathutils.Vector)
            weighted_center += center * volume
        if total_mass > eps:
            weighted_center /= total_mass
        return (len(indices), total_volume, -float(weighted_center.length))

    best_cluster = max(clusters, key=_cluster_score)
    if len(best_cluster) == len(stats):
        return objs, []

    keep_ids = {id(stats[idx]["obj"]) for idx in best_cluster}
    dropped: List[Dict[str, float | str]] = []
    for idx, stat in enumerate(stats):
        if idx in best_cluster:
            continue
        dropped.append(
            {
                "name": str(stat["name"]),
                "diag": float(stat["diag"]),
                "longest": float(stat["longest"]),
                "distance_to_origin": float(cast(mathutils.Vector, stat["center"]).length),
            }
        )

    if not dropped:
        return objs, []

    filtered_objs = [o for o in objs if o.type != "MESH" or id(o) in keep_ids]
    return filtered_objs, dropped


def _iter_object_with_descendants(root: bpy.types.Object) -> List[bpy.types.Object]:
    out = [root]
    queue = list(root.children)
    while queue:
        cur = queue.pop(0)
        out.append(cur)
        queue.extend(list(cur.children))
    return out


def _remove_object_family(root: Optional[bpy.types.Object]) -> None:
    if root is None:
        return
    for obj in reversed(_iter_object_with_descendants(root)):
        try:
            bpy.data.objects.remove(obj, do_unlink=True)
        except Exception:
            pass


def _aabb_from_blend_object_name(blend_object_name: str) -> Optional[Dict[str, float]]:
    name = str(blend_object_name or "").strip()
    if not name:
        return None
    obj = bpy.data.objects.get(name)
    if obj is None:
        return None

    family = _iter_object_with_descendants(obj)
    bmin, bmax = _world_bounds_mesh_objects(family)
    if bmin == bmax:
        try:
            corners = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
        except Exception:
            corners = []
        if corners:
            xs = [p.x for p in corners]
            ys = [p.y for p in corners]
            zs = [p.z for p in corners]
            bmin = mathutils.Vector((min(xs), min(ys), min(zs)))
            bmax = mathutils.Vector((max(xs), max(ys), max(zs)))

    if bmin == bmax:
        return None

    return {
        "x_min": float(bmin.x),
        "x_max": float(bmax.x),
        "y_min": float(bmin.y),
        "y_max": float(bmax.y),
        "z_min": float(bmin.z),
        "z_max": float(bmax.z),
    }


def _aabb_from_object_family_root(obj: bpy.types.Object) -> Optional[Dict[str, float]]:
    family = _iter_object_with_descendants(obj)
    bmin, bmax = _world_bounds_mesh_objects(family)
    if bmin == bmax:
        try:
            corners = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
        except Exception:
            corners = []
        if corners:
            xs = [p.x for p in corners]
            ys = [p.y for p in corners]
            zs = [p.z for p in corners]
            bmin = mathutils.Vector((min(xs), min(ys), min(zs)))
            bmax = mathutils.Vector((max(xs), max(ys), max(zs)))
    if bmin == bmax:
        return None
    return {
        "x_min": float(bmin.x),
        "x_max": float(bmax.x),
        "y_min": float(bmin.y),
        "y_max": float(bmax.y),
        "z_min": float(bmin.z),
        "z_max": float(bmax.z),
    }


def _blend_source_name_from_item(item: Dict) -> str:
    source = item.get("source") or {}
    if isinstance(source, dict):
        name = str(source.get("blend_object_name") or "").strip()
        if name:
            return name
    meta = item.get("meta") or {}
    meta_source = meta.get("source") or {}
    if isinstance(meta_source, dict):
        return str(meta_source.get("blend_object_name") or "").strip()
    return ""


def _get_scene_source_object(blend_object_name: str) -> Optional[bpy.types.Object]:
    name = str(blend_object_name or "").strip()
    if not name:
        return None
    return bpy.data.objects.get(name)


def _hide_object_family(root: bpy.types.Object) -> None:
    for obj in _iter_object_with_descendants(root):
        try:
            obj.hide_set(True)
        except Exception:
            pass
        try:
            obj.hide_render = True
        except Exception:
            pass


def _duplicate_light_objects_from_family(root: bpy.types.Object) -> int:
    kept_count = 0
    scene_coll = bpy.context.scene.collection
    for obj in _iter_object_with_descendants(root):
        if getattr(obj, "type", None) != "LIGHT":
            continue
        try:
            dup = obj.copy()
            if getattr(obj, "data", None) is not None:
                dup.data = obj.data.copy()
            dup.parent = None
            dup.matrix_world = obj.matrix_world.copy()
            dup.name = f"{obj.name}__kept"
            scene_coll.objects.link(dup)
            dup.hide_render = False
            try:
                dup.hide_set(False)
            except Exception:
                pass
            kept_count += 1
        except Exception:
            continue
    return kept_count


def _move_object_family_to_target_aabb(
    root: bpy.types.Object,
    target_aabb: Dict[str, float],
    *,
    align_bottom: bool = True,
) -> bool:
    family = _iter_object_with_descendants(root)
    bmin, bmax = _world_bounds_mesh_objects(family)
    if bmin == bmax:
        return False

    cur_center = (bmin + bmax) * 0.5
    tgt_center = mathutils.Vector(
        (
            0.5 * (float(target_aabb["x_min"]) + float(target_aabb["x_max"])),
            0.5 * (float(target_aabb["y_min"]) + float(target_aabb["y_max"])),
            0.5 * (float(target_aabb["z_min"]) + float(target_aabb["z_max"])),
        )
    )
    delta = tgt_center - cur_center
    if align_bottom:
        delta.z += float(target_aabb["z_min"]) - float(bmin.z + delta.z)

    for obj in family:
        try:
            obj.location += delta
        except Exception:
            pass
    bpy.context.view_layer.update()
    return True


def _translate_object_family(root: bpy.types.Object, delta: mathutils.Vector) -> None:
    for obj in _iter_object_with_descendants(root):
        try:
            obj.location += delta
        except Exception:
            pass
    bpy.context.view_layer.update()


def _move_object_family_to_exact_aabb(
    root: bpy.types.Object,
    current_aabb: Dict[str, float],
    target_aabb: Dict[str, float],
) -> Optional[Dict[str, float]]:
    cur_center = mathutils.Vector(
        (
            0.5 * (float(current_aabb["x_min"]) + float(current_aabb["x_max"])),
            0.5 * (float(current_aabb["y_min"]) + float(current_aabb["y_max"])),
            float(current_aabb["z_min"]),
        )
    )
    tgt_center = mathutils.Vector(
        (
            0.5 * (float(target_aabb["x_min"]) + float(target_aabb["x_max"])),
            0.5 * (float(target_aabb["y_min"]) + float(target_aabb["y_max"])),
            float(target_aabb["z_min"]),
        )
    )
    _translate_object_family(root, tgt_center - cur_center)
    return _aabb_from_object_family_root(root)


def _aabb_xy_overlap_area(a: Dict[str, float], b: Dict[str, float]) -> float:
    ox = max(0.0, min(float(a["x_max"]), float(b["x_max"])) - max(float(a["x_min"]), float(b["x_min"])))
    oy = max(0.0, min(float(a["y_max"]), float(b["y_max"])) - max(float(a["y_min"]), float(b["y_min"])))
    return ox * oy


def _mesh_face_support_plane_candidates(obj: bpy.types.Object) -> List[Dict[str, float]]:
    deps = bpy.context.evaluated_depsgraph_get()
    eo = obj.evaluated_get(deps)
    me = eo.to_mesh()
    if not me:
        return []

    planes: List[Dict[str, float]] = []
    try:
        world = eo.matrix_world
        normal_world = world.to_3x3()
        verts = me.vertices
        face_boxes: List[Dict[str, float]] = []
        for poly in me.polygons:
            if poly.area <= 5e-4:
                continue
            n = normal_world @ poly.normal
            if n.length <= 1e-6:
                continue
            n.normalize()
            if float(n.z) < 0.82:
                continue

            points = [world @ verts[idx].co for idx in poly.vertices]
            if len(points) < 3:
                continue
            xs = [float(p.x) for p in points]
            ys = [float(p.y) for p in points]
            zs = [float(p.z) for p in points]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            z_avg = sum(zs) / len(zs)
            sx = x_max - x_min
            sy = y_max - y_min
            if sx < 0.04 or sy < 0.04:
                continue
            face_boxes.append(
                {
                    "x_min": x_min,
                    "x_max": x_max,
                    "y_min": y_min,
                    "y_max": y_max,
                    "z": z_avg,
                    "area": float(poly.area),
                }
            )

        for face in sorted(face_boxes, key=lambda item: item["z"], reverse=True):
            merged = False
            for plane in planes:
                same_level = abs(face["z"] - plane["z"]) <= 0.02
                overlap = _aabb_xy_overlap_area(face, plane)
                near_xy = overlap > 1e-4 or (
                    abs(face["x_min"] - plane["x_max"]) <= 0.03
                    or abs(face["x_max"] - plane["x_min"]) <= 0.03
                    or abs(face["y_min"] - plane["y_max"]) <= 0.03
                    or abs(face["y_max"] - plane["y_min"]) <= 0.03
                )
                if not (same_level and near_xy):
                    continue
                prev_area = max(float(plane["area"]), 1e-6)
                plane["x_min"] = min(float(plane["x_min"]), float(face["x_min"]))
                plane["x_max"] = max(float(plane["x_max"]), float(face["x_max"]))
                plane["y_min"] = min(float(plane["y_min"]), float(face["y_min"]))
                plane["y_max"] = max(float(plane["y_max"]), float(face["y_max"]))
                plane["z"] = (float(plane["z"]) * prev_area + float(face["z"]) * float(face["area"])) / (prev_area + float(face["area"]))
                plane["area"] = prev_area + float(face["area"])
                merged = True
                break
            if not merged:
                planes.append(dict(face))
    finally:
        eo.to_mesh_clear()

    filtered: List[Dict[str, float]] = []
    for plane in planes:
        sx = float(plane["x_max"]) - float(plane["x_min"])
        sy = float(plane["y_max"]) - float(plane["y_min"])
        if sx < 0.06 or sy < 0.06:
            continue
        coverage = sx * sy
        if coverage < 0.012:
            continue
        plane["area"] = float(max(float(plane["area"]), coverage))
        filtered.append(plane)
    return filtered


def _annotate_support_plane_clearance(planes: List[Dict[str, float]]) -> List[Dict[str, float]]:
    annotated: List[Dict[str, float]] = []
    for idx, plane in enumerate(planes):
        next_gap = 10.0
        for jdx, other in enumerate(planes):
            if idx == jdx or float(other["z"]) <= float(plane["z"]) + 0.01:
                continue
            overlap = _aabb_xy_overlap_area(plane, other)
            if overlap <= 0.004:
                continue
            gap = float(other["z"]) - float(plane["z"])
            next_gap = min(next_gap, gap)
        enriched = dict(plane)
        enriched["clearance_height"] = float(next_gap)
        annotated.append(enriched)
    annotated.sort(key=lambda item: (item["z"], item["area"]), reverse=True)
    return annotated


def _extract_support_planes_from_object_family(root: bpy.types.Object) -> List[Dict[str, float]]:
    face_planes: List[Dict[str, float]] = []
    for obj in _iter_mesh_children(root):
        face_planes.extend(_mesh_face_support_plane_candidates(obj))

    planes = face_planes
    if not planes:
        for obj in _iter_mesh_children(root):
            bmin, bmax = _world_bounds_single_mesh_object(obj)
            if bmin == bmax:
                continue
            sx = max(float(bmax.x - bmin.x), 0.0)
            sy = max(float(bmax.y - bmin.y), 0.0)
            sz = max(float(bmax.z - bmin.z), 0.0)
            if sx < 0.06 or sy < 0.06:
                continue
            if sx * sy < 0.015:
                continue
            thin_horizontal = sz <= 0.16 or sz <= 0.35 * max(min(sx, sy), 1e-6)
            if not thin_horizontal:
                continue
            planes.append(
                {
                    "x_min": float(bmin.x),
                    "x_max": float(bmax.x),
                    "y_min": float(bmin.y),
                    "y_max": float(bmax.y),
                    "z": float(bmax.z),
                    "area": float(sx * sy),
                }
            )

    planes.sort(key=lambda item: (item["z"], item["area"]), reverse=True)
    deduped: List[Dict[str, float]] = []
    for plane in planes:
        duplicate = False
        for kept in deduped:
            if (
                abs(plane["z"] - kept["z"]) <= 0.015
                and abs(plane["x_min"] - kept["x_min"]) <= 0.03
                and abs(plane["x_max"] - kept["x_max"]) <= 0.03
                and abs(plane["y_min"] - kept["y_min"]) <= 0.03
                and abs(plane["y_max"] - kept["y_max"]) <= 0.03
            ):
                duplicate = True
                break
        if not duplicate:
            deduped.append(plane)
    return _annotate_support_plane_clearance(deduped)


def _infer_support_planes_from_anchor_item(anchor_item: Dict, anchor_aabb: Dict[str, float]) -> List[Dict[str, float]]:
    meta = anchor_item.get("meta") or {}
    semantic = str((meta.get("supplier_candidate") or {}).get("semantic_group") or "").strip().lower()
    name = str(anchor_item.get("name") or "").strip().lower()
    category = str(anchor_item.get("category") or "").strip().lower()
    if semantic not in {"shelf", "dresser", "nightstand", "tv_stand", "wardrobe"} and not any(
        token in f"{name} {category}" for token in ("shelf", "bookcase", "cabinet", "dresser", "тумб", "стеллаж", "шкаф", "комод", "полка")
    ):
        return []

    x1 = float(anchor_aabb["x_min"])
    x2 = float(anchor_aabb["x_max"])
    y1 = float(anchor_aabb["y_min"])
    y2 = float(anchor_aabb["y_max"])
    z1 = float(anchor_aabb["z_min"])
    z2 = float(anchor_aabb["z_max"])
    sx = x2 - x1
    sy = y2 - y1
    sz = z2 - z1
    mx = min(max(0.06 * sx, 0.015), sx * 0.2)
    my = min(max(0.06 * sy, 0.015), sy * 0.2)

    levels = [z2]
    if sz >= 0.45:
        levels.extend([z1 + sz * frac for frac in (0.28, 0.52, 0.76)])
    elif sz >= 0.22:
        levels.extend([z1 + sz * frac for frac in (0.4, 0.72)])

    planes: List[Dict[str, float]] = []
    for z in sorted(set(round(v, 4) for v in levels), reverse=True):
        planes.append(
            {
                "x_min": x1 + mx,
                "x_max": x2 - mx,
                "y_min": y1 + my,
                "y_max": y2 - my,
                "z": float(z),
                "area": max((sx - 2 * mx) * (sy - 2 * my), 1e-4),
            }
        )
    return _annotate_support_plane_clearance(planes)


def _choose_support_plane(
    item_aabb: Dict[str, float],
    planes: List[Dict[str, float]],
    *,
    mode: str,
) -> Optional[Dict[str, float]]:
    cx = 0.5 * (float(item_aabb["x_min"]) + float(item_aabb["x_max"]))
    cy = 0.5 * (float(item_aabb["y_min"]) + float(item_aabb["y_max"]))
    item_z_min = float(item_aabb["z_min"])
    candidates: List[Tuple[Tuple[float, ...], Dict[str, float]]] = []

    for plane in planes:
        overlap = _aabb_xy_overlap_area(item_aabb, plane)
        center_inside = (
            float(plane["x_min"]) - 0.03 <= cx <= float(plane["x_max"]) + 0.03
            and float(plane["y_min"]) - 0.03 <= cy <= float(plane["y_max"]) + 0.03
        )
        if not center_inside and overlap <= 1e-4:
            continue

        dz = item_z_min - float(plane["z"])
        if mode == "top":
            score = (
                1.0 if center_inside else 0.0,
                overlap,
                float(plane["z"]),
                float(plane["area"]),
            )
            candidates.append((score, plane))
            continue

        if dz < -0.08 or dz > 0.9:
            continue
        score = (
            1.0 if center_inside else 0.0,
            overlap,
            -abs(dz),
            float(plane["z"]),
        )
        candidates.append((score, plane))

    if not candidates and mode != "top":
        return _choose_support_plane(item_aabb, planes, mode="top")
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _snap_object_family_to_support_plane(
    root: bpy.types.Object,
    item_aabb: Dict[str, float],
    anchor_root: bpy.types.Object,
    *,
    mode: str,
) -> Optional[Dict[str, float]]:
    planes = _extract_support_planes_from_object_family(anchor_root)
    if not planes:
        return None
    plane = _choose_support_plane(item_aabb, planes, mode=mode)
    if plane is None:
        return None
    delta_z = float(plane["z"]) + 0.004 - float(item_aabb["z_min"])
    if abs(delta_z) < 1e-4:
        return item_aabb
    _translate_object_family(root, mathutils.Vector((0.0, 0.0, delta_z)))
    return _aabb_from_object_family_root(root)


def _translate_aabb_xy(aabb: Dict[str, float], dx: float, dy: float) -> Dict[str, float]:
    return {
        "x_min": float(aabb["x_min"]) + dx,
        "x_max": float(aabb["x_max"]) + dx,
        "y_min": float(aabb["y_min"]) + dy,
        "y_max": float(aabb["y_max"]) + dy,
        "z_min": float(aabb["z_min"]),
        "z_max": float(aabb["z_max"]),
    }


def _aabb_inside_xy(inner: Dict[str, float], outer: Dict[str, float], margin: float = 0.01) -> bool:
    return (
        float(inner["x_min"]) >= float(outer["x_min"]) - margin
        and float(inner["x_max"]) <= float(outer["x_max"]) + margin
        and float(inner["y_min"]) >= float(outer["y_min"]) - margin
        and float(inner["y_max"]) <= float(outer["y_max"]) + margin
    )


class MLSupportSolver:
    """
    A model-based support solver for books, trinkets, and lamps on replaced furniture.
    It searches candidate support planes and XY placements inside the anchor bounds.
    The objective penalizes floor penetration, leaving the support footprint, and collisions.
    This is deterministic but structured like a scored inference pass rather than hard-coded snapping.
    """

    def __init__(self, *, room_floor_z: float, clearance: float = 0.004) -> None:
        self.room_floor_z = float(room_floor_z)
        self.clearance = float(clearance)

    def _candidate_centers(
        self,
        item_size_xy: Tuple[float, float],
        plane: Dict[str, float],
        current_center_xy: Tuple[float, float],
    ) -> List[Tuple[float, float]]:
        sx, sy = item_size_xy
        px1, px2 = float(plane["x_min"]), float(plane["x_max"])
        py1, py2 = float(plane["y_min"]), float(plane["y_max"])
        mx = 0.5 * sx + 0.01
        my = 0.5 * sy + 0.01
        low_x, high_x = px1 + mx, px2 - mx
        low_y, high_y = py1 + my, py2 - my
        if low_x > high_x or low_y > high_y:
            return []

        cx_cur = min(max(float(current_center_xy[0]), low_x), high_x)
        cy_cur = min(max(float(current_center_xy[1]), low_y), high_y)
        cx_mid = 0.5 * (low_x + high_x)
        cy_mid = 0.5 * (low_y + high_y)
        xs = sorted({low_x, cx_cur, cx_mid, high_x})
        ys = sorted({low_y, cy_cur, cy_mid, high_y})
        out: List[Tuple[float, float]] = []
        for x in xs:
            for y in ys:
                out.append((x, y))
        return out

    def _candidate_aabb(
        self,
        item_aabb: Dict[str, float],
        *,
        center_xy: Tuple[float, float],
        support_z: float,
    ) -> Dict[str, float]:
        sx = float(item_aabb["x_max"]) - float(item_aabb["x_min"])
        sy = float(item_aabb["y_max"]) - float(item_aabb["y_min"])
        sz = float(item_aabb["z_max"]) - float(item_aabb["z_min"])
        cx, cy = center_xy
        z_min = float(support_z) + self.clearance
        return {
            "x_min": cx - 0.5 * sx,
            "x_max": cx + 0.5 * sx,
            "y_min": cy - 0.5 * sy,
            "y_max": cy + 0.5 * sy,
            "z_min": z_min,
            "z_max": z_min + sz,
        }

    def _collision_penalty(
        self,
        candidate: Dict[str, float],
        occupied_aabbs: List[Dict[str, float]],
    ) -> float:
        penalty = 0.0
        for occ in occupied_aabbs:
            ox = max(0.0, min(float(candidate["x_max"]), float(occ["x_max"])) - max(float(candidate["x_min"]), float(occ["x_min"])))
            oy = max(0.0, min(float(candidate["y_max"]), float(occ["y_max"])) - max(float(candidate["y_min"]), float(occ["y_min"])))
            oz = max(0.0, min(float(candidate["z_max"]), float(occ["z_max"])) - max(float(candidate["z_min"]), float(occ["z_min"])))
            if ox > 0.0 and oy > 0.0 and oz > 0.0:
                penalty += 1000.0 + (ox * oy * oz * 5000.0)
        return penalty

    def solve(
        self,
        *,
        item_aabb: Dict[str, float],
        anchor_aabb: Dict[str, float],
        planes: List[Dict[str, float]],
        occupied_aabbs: List[Dict[str, float]],
        mode: str,
    ) -> Optional[Dict[str, float]]:
        if not planes:
            return None

        cur_cx = 0.5 * (float(item_aabb["x_min"]) + float(item_aabb["x_max"]))
        cur_cy = 0.5 * (float(item_aabb["y_min"]) + float(item_aabb["y_max"]))
        cur_z = float(item_aabb["z_min"])
        sx = float(item_aabb["x_max"]) - float(item_aabb["x_min"])
        sy = float(item_aabb["y_max"]) - float(item_aabb["y_min"])
        sz = float(item_aabb["z_max"]) - float(item_aabb["z_min"])

        best_score = float("inf")
        best_aabb: Optional[Dict[str, float]] = None

        for plane in planes:
            plane_z = float(plane["z"])
            clearance_height = float(plane.get("clearance_height") or 10.0)
            if clearance_height < sz + 0.012:
                continue
            plane_pref_penalty = 0.0 if mode == "top" else abs(cur_z - plane_z) * 8.0
            for center_xy in self._candidate_centers((sx, sy), plane, (cur_cx, cur_cy)):
                candidate = self._candidate_aabb(item_aabb, center_xy=center_xy, support_z=plane_z)
                score = 0.0
                if candidate["z_min"] < self.room_floor_z - 0.002:
                    score += 5000.0
                if not _aabb_inside_xy(candidate, anchor_aabb, margin=0.02):
                    score += 2000.0
                overlap_area = _aabb_xy_overlap_area(candidate, plane)
                footprint_area = max(sx * sy, 1e-6)
                support_ratio = overlap_area / footprint_area
                score += max(0.0, 0.85 - support_ratio) * 600.0
                score += self._collision_penalty(candidate, occupied_aabbs)
                score += plane_pref_penalty
                score += math.hypot(center_xy[0] - cur_cx, center_xy[1] - cur_cy) * 4.0
                if mode == "top":
                    score -= plane_z * 0.5
                if score < best_score:
                    best_score = score
                    best_aabb = candidate

        return best_aabb


def _item_has_existing_mesh_file(item: Dict, json_dir: Path) -> bool:
    raw = _item_mesh_path_raw(item)
    resolved = _resolve_path_maybe(json_dir, raw)
    return bool(resolved) and os.path.isfile(str(resolved))


def _aabb_to_center_size(aabb: Dict[str, float]) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    x1, x2 = float(aabb["x_min"]), float(aabb["x_max"])
    y1, y2 = float(aabb["y_min"]), float(aabb["y_max"])
    z1, z2 = float(aabb["z_min"]), float(aabb["z_max"])
    cx, cy, cz = 0.5 * (x1 + x2), 0.5 * (y1 + y2), 0.5 * (z1 + z2)
    sx, sy, sz = (x2 - x1), (y2 - y1), (z2 - z1)
    return (cx, cy, cz), (sx, sy, sz)


def _add_aabb_box(aabb: Dict[str, float], name: str, collection: bpy.types.Collection) -> bpy.types.Object:
    (cx, cy, cz), (sx, sy, sz) = _aabb_to_center_size(aabb)
    verts = [
        (cx - 0.5 * sx, cy - 0.5 * sy, cz - 0.5 * sz),
        (cx + 0.5 * sx, cy - 0.5 * sy, cz - 0.5 * sz),
        (cx + 0.5 * sx, cy + 0.5 * sy, cz - 0.5 * sz),
        (cx - 0.5 * sx, cy + 0.5 * sy, cz - 0.5 * sz),
        (cx - 0.5 * sx, cy - 0.5 * sy, cz + 0.5 * sz),
        (cx + 0.5 * sx, cy - 0.5 * sy, cz + 0.5 * sz),
        (cx + 0.5 * sx, cy + 0.5 * sy, cz + 0.5 * sz),
        (cx - 0.5 * sx, cy + 0.5 * sy, cz + 0.5 * sz),
    ]
    faces = [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ]
    obj = _make_mesh_object(f"{name}_AABB", verts, faces, collection)
    obj.display_type = "WIRE"
    obj.show_in_front = True
    return obj


def _parse_id_set(raw: Optional[str]) -> set[str]:
    text = str(raw or "").strip()
    if not text:
        return set()
    return {chunk.strip() for chunk in text.split(",") if chunk.strip()}


def _add_aabb_label(aabb: Dict[str, float], text: str, collection: bpy.types.Collection) -> Optional[bpy.types.Object]:
    label = str(text or "").strip()
    if not label:
        return None
    x = 0.5 * (float(aabb["x_min"]) + float(aabb["x_max"]))
    y = 0.5 * (float(aabb["y_min"]) + float(aabb["y_max"]))
    z = float(aabb["z_max"]) + 0.08

    curve = bpy.data.curves.new(name=f"{label}_LabelCurve", type="FONT")
    curve.body = label
    curve.align_x = "CENTER"
    curve.size = 0.18

    obj = bpy.data.objects.new(f"{label}_Label", curve)
    bpy.context.scene.collection.objects.link(obj)
    _unlink_from_all_collections(obj)
    collection.objects.link(obj)
    obj.location = (x, y, z)
    obj.show_in_front = True
    return obj


def _should_skip_placeholder_bbox(item: Dict) -> bool:
    meta = item.get("meta") or {}
    if not meta.get("placeholder_bbox"):
        return False

    collections = {str(x).strip().lower() for x in (meta.get("collections") or []) if str(x).strip()}
    if collections & {"door_base_elements", "scatters", "assets:fruit"}:
        return True

    name = str(_item_name(item) or "").strip().lower()
    if not name:
        return False

    if name.startswith("scatter:"):
        return True

    return name in {
        "béziercurve",
        "beziercurve",
        "cube",
        "cube.001",
        "plane",
        "fruitfactory",
        "scatter:fruit",
    }


def _ensure_world() -> None:
    scene = bpy.context.scene
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    try:
        scene.world.use_nodes = True
        nt = scene.world.node_tree
        nodes = nt.nodes
        links = nt.links
        bg = next((n for n in nodes if n.type == "BACKGROUND"), None)
        out = next((n for n in nodes if n.type == "OUTPUT_WORLD"), None)
        if not out:
            out = nodes.new("ShaderNodeOutputWorld")
        if not bg:
            bg = nodes.new("ShaderNodeBackground")
        try:
            links.new(bg.outputs["Background"], out.inputs["Surface"])
        except Exception:
            pass
        bg.inputs["Strength"].default_value = 1.0
    except Exception:
        pass


def _force_material_preview_if_ui() -> None:
    wm = bpy.context.window_manager
    if not wm or not getattr(wm, "windows", None):
        return
    for win in wm.windows:
        scr = win.screen
        if not scr:
            continue
        for area in scr.areas:
            if area.type != "VIEW_3D":
                continue
            sp = area.spaces.active
            if not hasattr(sp, "shading"):
                continue
            try:
                sp.shading.type = "MATERIAL"
                if hasattr(sp.shading, "use_scene_lights"):
                    sp.shading.use_scene_lights = True
                if hasattr(sp.shading, "use_scene_world"):
                    sp.shading.use_scene_world = False
            except Exception:
                pass


def _frame_camera_on_bounds(bb_min: mathutils.Vector, bb_max: mathutils.Vector) -> None:
    center = (bb_min + bb_max) * 0.5
    dims = bb_max - bb_min
    radius = max(dims.x, dims.y, dims.z, 0.1)

    cam_data = bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.scene.collection.objects.link(cam)

    cam.location = (center.x - 1.6 * radius, center.y - 1.6 * radius, center.z + 1.2 * radius)
    cam.data.lens = 35
    direction = mathutils.Vector((center.x, center.y, center.z)) - cam.location
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

    bpy.context.scene.camera = cam


def _visible_mesh_bounds(default_min: mathutils.Vector, default_max: mathutils.Vector) -> Tuple[mathutils.Vector, mathutils.Vector]:
    visible = [obj for obj in bpy.data.objects if obj.type == "MESH" and not bool(getattr(obj, "hide_render", False))]
    if not visible:
        return default_min.copy(), default_max.copy()
    bb_min, bb_max = _world_bounds_mesh_objects(visible)
    if bb_min == bb_max:
        return default_min.copy(), default_max.copy()
    return bb_min, bb_max


def _store_scene_room_bounds(bb_min: mathutils.Vector, bb_max: mathutils.Vector) -> None:
    scene = bpy.context.scene
    scene["cgs_room_bounds"] = {
        "x_min": float(bb_min.x),
        "y_min": float(bb_min.y),
        "z_min": float(bb_min.z),
        "x_max": float(bb_max.x),
        "y_max": float(bb_max.y),
        "z_max": float(bb_max.z),
    }


def _scene_room_bounds() -> Optional[Tuple[mathutils.Vector, mathutils.Vector]]:
    raw = bpy.context.scene.get("cgs_room_bounds")
    if not isinstance(raw, dict):
        return None
    try:
        bb_min = mathutils.Vector((float(raw["x_min"]), float(raw["y_min"]), float(raw["z_min"])))
        bb_max = mathutils.Vector((float(raw["x_max"]), float(raw["y_max"]), float(raw["z_max"])))
    except Exception:
        return None
    if bb_min == bb_max:
        return None
    return bb_min, bb_max


def _item_semantic_group(item: Dict) -> str:
    meta = item.get("meta") or {}
    supplier_candidate = meta.get("supplier_candidate") if isinstance(meta.get("supplier_candidate"), dict) else {}
    group = str(
        supplier_candidate.get("semantic_group")
        or item.get("semantic_group")
        or ""
    ).strip().lower()
    if group:
        return group

    category = str(item.get("category") or "").strip().lower()
    name = str(_item_name(item) or "").strip().lower()
    text = f"{category} {name}"
    mapping = (
        ("desklampfactory", "lamp_table"),
        ("floorlampfactory", "lamp_floor"),
        ("ceilinglightfactory", "lamp_ceiling"),
        ("simpledeskfactory", "desk"),
        ("singleshelfactory", "shelf"),
        ("simplebookcasefactory", "shelf"),
        ("cellshelffactory", "shelf"),
        ("singlecabinetfactory", "wardrobe"),
        ("wardrobe", "wardrobe"),
        ("dresser", "dresser"),
        ("nightstand", "nightstand"),
        ("bedfactory", "bed"),
    )
    for token, group_name in mapping:
        if token in text:
            return group_name

    if "настоль" in text or ("lamp" in text and "desk" in text):
        return "lamp_table"
    if "торшер" in text or "floor lamp" in text:
        return "lamp_floor"
    if "люстр" in text or "pendant" in text or "chandelier" in text:
        return "lamp_ceiling"
    return ""


def _item_mount_mode(item: Dict) -> str:
    constraints = item.get("constraints") or {}
    mount_type = str((constraints or {}).get("mount_type") or "").strip().lower()
    if mount_type in {"ceiling", "wall", "floor"}:
        return mount_type

    semantic_group = _item_semantic_group(item)
    meta = item.get("meta") or {}
    if bool(meta.get("supplier_support_reanchored")) or semantic_group == "lamp_table":
        return "support"
    if semantic_group == "lamp_ceiling" or bool((constraints or {}).get("under_ceiling")):
        return "ceiling"
    if semantic_group in {
        "lamp_floor",
        "bed",
        "desk",
        "dresser",
        "nightstand",
        "side_table",
        "coffee_table",
        "shelf",
        "wardrobe",
        "tv_stand",
        "chair",
        "armchair",
        "sofa",
        "mirror",
        "plant",
    }:
        return "floor"
    if isinstance((constraints or {}).get("touch_floor"), dict) and (constraints["touch_floor"].get("side") == "bottom"):
        return "floor"
    return "support"


def _rotation_candidates_for_semantic_group(base_deg: float, semantic_group: str) -> List[float]:
    orientable_groups = {
        "bed",
        "desk",
        "dresser",
        "nightstand",
        "side_table",
        "coffee_table",
        "shelf",
        "wardrobe",
        "tv_stand",
    }
    base = float(base_deg or 0.0) % 360.0
    if semantic_group not in orientable_groups:
        return [base]
    candidates: List[float] = []
    for delta in (0.0, 90.0, 180.0, 270.0):
        value = (base + delta) % 360.0
        if all(abs(((value - existing + 180.0) % 360.0) - 180.0) > 1e-4 for existing in candidates):
            candidates.append(value)
    return candidates


def _nearest_room_wall_context(aabb: Dict[str, float]) -> Optional[Tuple[mathutils.Vector, mathutils.Vector, float]]:
    room_bounds = _scene_room_bounds()
    if room_bounds is None:
        return None
    room_min, room_max = room_bounds
    center = mathutils.Vector(
        (
            0.5 * (float(aabb["x_min"]) + float(aabb["x_max"])),
            0.5 * (float(aabb["y_min"]) + float(aabb["y_max"])),
            0.0,
        )
    )
    room_center = mathutils.Vector(
        (
            0.5 * (float(room_min.x) + float(room_max.x)),
            0.5 * (float(room_min.y) + float(room_max.y)),
            0.0,
        )
    )
    distances = [
        (abs(center.x - float(room_min.x)), mathutils.Vector((-1.0, 0.0, 0.0))),
        (abs(float(room_max.x) - center.x), mathutils.Vector((1.0, 0.0, 0.0))),
        (abs(center.y - float(room_min.y)), mathutils.Vector((0.0, -1.0, 0.0))),
        (abs(float(room_max.y) - center.y), mathutils.Vector((0.0, 1.0, 0.0))),
    ]
    wall_dist, wall_dir = min(distances, key=lambda item: item[0])
    room_dir = room_center - center
    if room_dir.length > 1e-6:
        room_dir.normalize()
    return wall_dir, room_dir, float(wall_dist)


def _matches_room_shell_name(name: str) -> bool:
    low = str(name or "").strip().lower()
    if not low:
        return False
    patterns = (
        r"(^|[./_])wall($|[./_0-9])",
        r"(^|[./_])ceiling($|[./_0-9])",
        r"(^|[./_])exterior($|[./_0-9])",
    )
    return any(re.search(pattern, low) for pattern in patterns)


def _hide_room_shell_objects() -> int:
    hidden = 0
    for obj in list(bpy.data.objects):
        if not _matches_room_shell_name(getattr(obj, "name", "")):
            continue
        if obj.type == "LIGHT":
            continue
        try:
            obj.hide_set(True)
        except Exception:
            pass
        try:
            obj.hide_render = True
        except Exception:
            pass
        hidden += 1
    return hidden


def _looks_like_overlay_helper_name(name: str) -> bool:
    low = str(name or "").strip().lower()
    if not low:
        return False
    return low.endswith("_label") or ("_aabb" in low)


def _set_overlay_helpers_render_visibility(show: bool) -> int:
    changed = 0
    for obj in list(bpy.data.objects):
        if not _looks_like_overlay_helper_name(getattr(obj, "name", "")):
            continue
        try:
            obj.hide_render = not show
            changed += 1
        except Exception:
            pass
    return changed


def _add_basic_lights(bb_min: mathutils.Vector, bb_max: mathutils.Vector) -> None:
    center = (bb_min + bb_max) * 0.5
    dims = bb_max - bb_min
    radius = max(dims.x, dims.y, dims.z, 0.1)

    sun_data = bpy.data.lights.new("Sun", "SUN")
    sun_data.energy = 3.0
    sun = bpy.data.objects.new("Sun", sun_data)
    bpy.context.scene.collection.objects.link(sun)
    sun.location = (center.x + 3.0 * radius, center.y + 2.5 * radius, center.z + 3.0 * radius)

    try:
        area_data = bpy.data.lights.new("Key", "AREA")
        area_data.energy = 500.0
        area_data.size = 2.0 * radius
        area = bpy.data.objects.new("Key", area_data)
        bpy.context.scene.collection.objects.link(area)
        area.location = (center.x - 1.5 * radius, center.y + 1.5 * radius, center.z + 1.8 * radius)
        direction = mathutils.Vector((center.x, center.y, center.z)) - area.location
        area.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    except Exception:
        pass


def _pack_assets_best_effort() -> None:
    try:
        bpy.ops.file.make_paths_relative()
    except Exception:
        pass
    try:
        bpy.ops.file.pack_all()
    except Exception:
        pass


def _configure_fast_render(scene: bpy.types.Scene) -> None:
    try:
        scene.render.engine = "CYCLES"
    except Exception:
        pass

    try:
        scene.render.resolution_percentage = 100
    except Exception:
        pass

    cycles = getattr(scene, "cycles", None)
    if cycles is None:
        return

    # Visualizer renders should stay interactive-fast even when a heavy
    # reference .blend carries over production settings like 8192 samples.
    for attr, value in (
        ("samples", 64),
        ("preview_samples", 16),
        ("max_bounces", 4),
        ("diffuse_bounces", 2),
        ("glossy_bounces", 2),
        ("transmission_bounces", 2),
        ("transparent_max_bounces", 2),
        ("volume_bounces", 0),
    ):
        try:
            setattr(cycles, attr, value)
        except Exception:
            pass

    for attr, value in (
        ("use_adaptive_sampling", True),
        ("use_denoising", True),
        ("use_preview_denoising", True),
    ):
        try:
            setattr(cycles, attr, value)
        except Exception:
            pass


def _configure_turntable_render(scene: bpy.types.Scene) -> None:
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except Exception:
        try:
            scene.render.engine = "BLENDER_EEVEE"
        except Exception:
            _configure_fast_render(scene)
            cycles = getattr(scene, "cycles", None)
            if cycles is None:
                return
            for attr, value in (
                ("samples", 16),
                ("preview_samples", 8),
                ("max_bounces", 3),
                ("diffuse_bounces", 1),
                ("glossy_bounces", 1),
                ("transmission_bounces", 1),
            ):
                try:
                    setattr(cycles, attr, value)
                except Exception:
                    pass
            return

    eevee = getattr(scene, "eevee", None)
    if eevee is not None:
        for attr, value in (
            ("taa_render_samples", 16),
            ("taa_samples", 8),
            ("use_gtao", True),
        ):
            try:
                setattr(eevee, attr, value)
            except Exception:
                pass

    try:
        scene.render.resolution_percentage = 100
    except Exception:
        pass


def _ensure_scene_camera(bb_min: mathutils.Vector, bb_max: mathutils.Vector) -> bpy.types.Object:
    scene = bpy.context.scene
    cam = bpy.data.objects.get("CGS_TurntableCamera")
    if cam is None or getattr(cam, "type", None) != "CAMERA":
        cam_data = bpy.data.cameras.new("CGS_TurntableCamera")
        cam = bpy.data.objects.new("CGS_TurntableCamera", cam_data)
        scene.collection.objects.link(cam)

    data = getattr(cam, "data", None)
    if data is not None:
        try:
            data.type = "PERSP"
        except Exception:
            pass
        try:
            data.lens = 28
        except Exception:
            pass
        try:
            data.clip_start = 0.01
            data.clip_end = 200.0
        except Exception:
            pass
        try:
            data.dof.use_dof = False
        except Exception:
            pass

    for constraint in list(getattr(cam, "constraints", [])):
        try:
            cam.constraints.remove(constraint)
        except Exception:
            pass

    scene.camera = cam
    return cam


def _render_turntable_sequence(
    out_dir: Path,
    frame_count: int,
    bb_min: mathutils.Vector,
    bb_max: mathutils.Vector,
    elevation_deg: float = 30.0,
    room_bb_min: Optional[mathutils.Vector] = None,
    room_bb_max: Optional[mathutils.Vector] = None,
) -> None:
    scene = bpy.context.scene
    cam = _ensure_scene_camera(bb_min, bb_max)
    visible_center = (bb_min + bb_max) * 0.5
    visible_dims = bb_max - bb_min
    room_min = room_bb_min.copy() if room_bb_min is not None else bb_min.copy()
    room_max = room_bb_max.copy() if room_bb_max is not None else bb_max.copy()
    room_center = (room_min + room_max) * 0.5
    room_dims = room_max - room_min
    xy_span = max(float(room_dims.x), float(room_dims.y), 0.8)
    z_span = max(float(visible_dims.z), 0.8)
    orbit_radius = max(1.9, xy_span * 0.68)
    elev_rad = math.radians(float(elevation_deg or 0.0))
    target = mathutils.Vector(
        (
            room_center.x * 0.72 + visible_center.x * 0.28,
            room_center.y * 0.72 + visible_center.y * 0.28,
            max(float(room_min.z) + 0.78, min(float(room_min.z) + float(room_dims.z) * 0.40, float(bb_min.z) + z_span * 0.28)),
        )
    )
    orbit_z = target.z + math.tan(elev_rad) * orbit_radius

    out_dir.mkdir(parents=True, exist_ok=True)
    prev_path = str(getattr(scene.render, "filepath", "") or "")
    prev_cam = scene.camera
    scene.camera = cam
    try:
        cam.data.lens = 28
    except Exception:
        pass

    try:
        for idx in range(max(int(frame_count), 1)):
            angle = (2.0 * math.pi * idx) / max(int(frame_count), 1)
            cam.location = (
                target.x + orbit_radius * math.cos(angle),
                target.y + orbit_radius * math.sin(angle),
                orbit_z,
            )
            direction = target - cam.location
            cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
            scene.render.filepath = str((out_dir / f"frame_{idx:03d}.png").resolve())
            bpy.ops.render.render(write_still=True)
    finally:
        scene.render.filepath = prev_path
        scene.camera = prev_cam or cam


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def clamp_item_aabb_to_room_bounds(
    aabb: Dict[str, float],
    bb_min: mathutils.Vector,
    bb_max: mathutils.Vector,
    margin: float = 0.02,
) -> Dict[str, float]:
    x1, x2 = float(aabb["x_min"]), float(aabb["x_max"])
    y1, y2 = float(aabb["y_min"]), float(aabb["y_max"])
    z1, z2 = float(aabb["z_min"]), float(aabb["z_max"])

    sx = max(1e-6, x2 - x1)
    sy = max(1e-6, y2 - y1)
    sz = max(1e-6, z2 - z1)

    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    cz = 0.5 * (z1 + z2)

    cx = _clamp(cx, float(bb_min.x) + margin + sx * 0.5, float(bb_max.x) - margin - sx * 0.5)
    cy = _clamp(cy, float(bb_min.y) + margin + sy * 0.5, float(bb_max.y) - margin - sy * 0.5)
    cz = _clamp(cz, float(bb_min.z) + margin + sz * 0.5, float(bb_max.z) - margin - sz * 0.5)

    return {
        "x_min": cx - sx * 0.5, "x_max": cx + sx * 0.5,
        "y_min": cy - sy * 0.5, "y_max": cy + sy * 0.5,
        "z_min": cz - sz * 0.5, "z_max": cz + sz * 0.5,
    }


# ============================================================
# Path helpers
# ============================================================

def _resolve_path_maybe(base_dir: Path, p: Optional[str]) -> Optional[str]:
    if not p:
        return None
    s = str(p).strip().strip('"').strip("'")
    if not s:
        return None
    pp = Path(s).expanduser()
    if pp.is_absolute():
        return str(pp)
    return str((base_dir / pp).resolve())


def _item_asset_dict(it: dict) -> dict:
    asset = it.get("asset")
    return asset if isinstance(asset, dict) else {}


def _item_mesh_path_raw(it: dict) -> Optional[str]:
    asset = _item_asset_dict(it)
    return (
        it.get("mesh_path")
        or it.get("obj_path")
        or asset.get("mesh_path")
        or asset.get("obj_path")
    )


def _item_supplier_candidate_pool(it: dict) -> List[dict]:
    meta = it.get("meta") or {}
    raw_pool = meta.get("supplier_candidate_pool")
    if not isinstance(raw_pool, list):
        raw_pool = []
    pool: List[dict] = []
    seen: set[str] = set()
    primary = meta.get("supplier_candidate") if isinstance(meta.get("supplier_candidate"), dict) else None
    for candidate in [primary, *raw_pool]:
        if not isinstance(candidate, dict):
            continue
        unique_key = str(candidate.get("unique_key") or "").strip()
        if unique_key and unique_key in seen:
            continue
        if unique_key:
            seen.add(unique_key)
        pool.append(candidate)
    return pool


def _candidate_mesh_path_raw(candidate: dict, fallback_item: dict) -> Optional[str]:
    return (
        candidate.get("asset_local_path")
        or candidate.get("mesh_path")
        or candidate.get("obj_path")
        or _item_mesh_path_raw(fallback_item)
    )


def _item_mesh_texture_dirs_raw(it: dict) -> List[str]:
    asset = _item_asset_dict(it)
    raw = it.get("mesh_texture_dirs")
    if isinstance(raw, list) and raw:
        return [str(x) for x in raw]

    raw = asset.get("mesh_texture_dirs")
    if isinstance(raw, list) and raw:
        return [str(x) for x in raw]

    return []


def _item_mesh_fit_mode(it: dict) -> str:
    asset = _item_asset_dict(it)
    return str(it.get("mesh_fit_mode") or asset.get("mesh_fit_mode") or "stretch")


def _item_name(it: dict) -> str:
    return str(it.get("name") or it.get("category") or it.get("id") or "Item")


def _build_obj_import_override() -> Optional[dict]:
    wm = bpy.context.window_manager
    windows = getattr(wm, "windows", None) if wm else None
    if not windows:
        return None
    win = windows[0]
    scr = win.screen if win else None
    if not scr:
        return None

    area = None
    for a in scr.areas:
        if a.type == "VIEW_3D":
            area = a
            break
    if area is None and scr.areas:
        area = scr.areas[0]

    region = None
    if area:
        for r in area.regions:
            if r.type == "WINDOW":
                region = r
                break

    if not area or not region:
        return None

    space_data = area.spaces.active if getattr(area, "spaces", None) else None

    return {
        "window": win,
        "screen": scr,
        "area": area,
        "region": region,
        "space_data": space_data,
        "scene": bpy.context.scene,
        "view_layer": bpy.context.view_layer,
    }


def import_obj(mesh_path: str) -> List[bpy.types.Object]:
    before = set(bpy.data.objects)

    try:
        if bpy.context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        pass

    override = _build_obj_import_override()
    if hasattr(bpy.ops.wm, "obj_import"):
        try:
            if override:
                with bpy.context.temp_override(**override):
                    bpy.ops.wm.obj_import(filepath=mesh_path)
            else:
                with bpy.context.temp_override(scene=bpy.context.scene, view_layer=bpy.context.view_layer):
                    bpy.ops.wm.obj_import(filepath=mesh_path)
        except Exception:
            if hasattr(bpy.ops.import_scene, "obj"):
                try:
                    if override:
                        with bpy.context.temp_override(**override):
                            bpy.ops.import_scene.obj(filepath=mesh_path)
                    else:
                        bpy.ops.import_scene.obj(filepath=mesh_path)
                except Exception as e2:
                    raise RuntimeError(str(e2))
            else:
                raise RuntimeError("OBJ import failed (no suitable importer / context).")
    else:
        if hasattr(bpy.ops.import_scene, "obj"):
            try:
                if override:
                    with bpy.context.temp_override(**override):
                        bpy.ops.import_scene.obj(filepath=mesh_path)
                else:
                    bpy.ops.import_scene.obj(filepath=mesh_path)
            except Exception as e:
                raise RuntimeError(str(e))
        else:
            raise RuntimeError("No OBJ importer available in this Blender build.")

    after = [o for o in bpy.data.objects if o not in before]
    return after


def import_fbx(mesh_path: str) -> List[bpy.types.Object]:
    before = set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=mesh_path)
    after = [o for o in bpy.data.objects if o not in before]
    return after


def import_gltf(mesh_path: str) -> List[bpy.types.Object]:
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=mesh_path)
    after = [o for o in bpy.data.objects if o not in before]
    return after


def import_supported_mesh(mesh_path: str) -> List[bpy.types.Object]:
    ext = os.path.splitext(mesh_path)[1].lower()
    if ext == ".obj":
        return import_obj(mesh_path)
    if ext == ".fbx":
        return import_fbx(mesh_path)
    if ext in {".glb", ".gltf"}:
        return import_gltf(mesh_path)
    raise RuntimeError(f"Unsupported mesh format: {ext}")


_SUPPORTED_MESH_EXTS = (".glb", ".gltf", ".obj", ".fbx")


def _mesh_ext_priority(ext: str) -> int:
    try:
        return _SUPPORTED_MESH_EXTS.index(ext.lower())
    except ValueError:
        return len(_SUPPORTED_MESH_EXTS)


def _discover_mesh_import_candidates(mesh_path: str) -> List[str]:
    base = Path(mesh_path).expanduser().resolve()
    if not base.exists():
        return [str(base)]

    seen: set[str] = set()
    out: List[str] = []

    def add_candidate(path: Path) -> None:
        try:
            resolved = str(path.expanduser().resolve())
        except Exception:
            return
        if resolved in seen:
            return
        if not Path(resolved).is_file():
            return
        if Path(resolved).suffix.lower() not in _SUPPORTED_MESH_EXTS:
            return
        seen.add(resolved)
        out.append(resolved)

    add_candidate(base)

    same_dir = base.parent
    same_stem = sorted(
        same_dir.glob(f"{base.stem}.*"),
        key=lambda p: (_mesh_ext_priority(p.suffix), str(p.name).lower()),
    )
    for candidate in same_stem:
        add_candidate(candidate)

    asset_root = detect_asset_root(str(base))
    recursive: List[Path] = []
    for root in [asset_root, asset_root.parent]:
        if not root.exists() or not root.is_dir():
            continue
        try:
            for path in root.rglob("*"):
                if path.is_file() and path.suffix.lower() in _SUPPORTED_MESH_EXTS:
                    recursive.append(path)
        except Exception:
            continue

    def rank_path(path: Path) -> tuple[int, int, int, str]:
        same_parent = 0 if path.parent == same_dir else 1
        same_stem_rank = 0 if path.stem == base.stem else 1
        return (
            same_parent,
            same_stem_rank,
            _mesh_ext_priority(path.suffix),
            str(path).lower(),
        )

    for candidate in sorted(recursive, key=rank_path):
        add_candidate(candidate)

    return out


def _safe_import_supported_mesh(mesh_path: str) -> tuple[List[bpy.types.Object], Optional[str]]:
    before = set(bpy.data.objects)
    try:
        objs = import_supported_mesh(mesh_path)
        return objs, None
    except Exception as exc:
        after = set(bpy.data.objects)
        created = [o for o in bpy.data.objects if o in (after - before)]
        for obj in created:
            try:
                bpy.data.objects.remove(obj, do_unlink=True)
            except Exception:
                pass
        return [], f"{type(exc).__name__}: {exc}"


# ============================================================
# Texture search / indexing
# ============================================================

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tga", ".bmp", ".tif", ".tiff", ".exr", ".webp"}
_IGNORE_DIRS = {".git", "__pycache__", ".venv", "node_modules", "textures_cache"}
_TRASH_TOKENS = {"preview", "render", "thumb", "thumbnail", "icon"}


def _is_hashlike_name(name: str) -> bool:
    if not name:
        return False
    if len(name) < 16:
        return False
    if any(ch.isspace() for ch in name):
        return False
    if re.search(r"[А-Яа-я]", name):
        return False
    alnum = sum(1 for ch in name.lower() if ("a" <= ch <= "z") or ("0" <= ch <= "9"))
    return alnum / max(len(name), 1) >= 0.85


def detect_asset_root(mesh_path: str) -> Path:
    model_dir = Path(mesh_path).resolve().parent
    if _is_hashlike_name(model_dir.name):
        return model_dir.parent
    return model_dir


def build_search_dirs(mesh_path: Optional[str], mesh_texture_dirs: Optional[List[str]]) -> List[str]:
    out: List[str] = []
    seen = set()

    def add_dir(p: Optional[Path]) -> None:
        if not p:
            return
        try:
            pp = str(p.resolve())
        except Exception:
            return
        if pp in seen:
            return
        if not p.exists() or not p.is_dir():
            return
        out.append(pp)
        seen.add(pp)

    if mesh_path:
        mp = Path(mesh_path).resolve()
        model_dir = mp.parent
        asset_root = detect_asset_root(str(mp))
        add_dir(model_dir)
        add_dir(asset_root)
        add_dir(asset_root.parent)

    for d in (mesh_texture_dirs or []):
        try:
            dd = Path(d).expanduser()
            if not dd.is_absolute() and mesh_path:
                dd = (Path(mesh_path).resolve().parent / dd).resolve()
            add_dir(dd)
        except Exception:
            pass

    return out


def _walk_images_limited(base_dir: str, max_depth: int, max_files: int) -> Iterable[Tuple[str, str]]:
    base = Path(base_dir)
    if not base.exists() or not base.is_dir():
        return

    emitted = 0
    base_parts = len(base.resolve().parts)

    for root, dirs, files in os.walk(str(base)):
        try:
            depth = len(Path(root).resolve().parts) - base_parts
        except Exception:
            depth = 0
        if depth > max_depth:
            dirs[:] = []
            continue

        dirs[:] = [d for d in dirs if d not in _IGNORE_DIRS]

        for fn in files:
            if emitted >= max_files:
                return
            ext = os.path.splitext(fn)[1].lower()
            if ext not in IMG_EXTS:
                continue
            fn_l = fn.lower()
            if any(tok in fn_l for tok in _TRASH_TOKENS):
                continue
            emitted += 1
            yield (fn_l, os.path.join(root, fn))


def build_image_index(search_dirs: List[str], max_depth: int = 4, max_files: int = 15000) -> Dict[str, str]:
    idx: Dict[str, str] = {}
    for d in search_dirs:
        for fn_l, full in _walk_images_limited(d, max_depth=max_depth, max_files=max_files):
            if fn_l not in idx:
                idx[fn_l] = full
    return idx


# ============================================================
# MTL parsing / resolving
# ============================================================

_MTL_MAP_RE = re.compile(r"^\s*(map_Kd|map_Bump|bump|map_d|map_Ks|map_Ka)\s+(.*)$", re.IGNORECASE)
_OBJ_MTLLIB_RE = re.compile(r"^\s*mtllib\s+(.*)$", re.IGNORECASE)


def _strip_mtl_opts(s: str) -> str:
    s = s.strip()
    if '"' in s:
        parts = re.findall(r'"([^"]+)"', s)
        if parts:
            return parts[-1].strip()
    toks = s.split()
    if not toks:
        return ""
    return toks[-1].strip()


def parse_obj_mtl_files(obj_path: str) -> List[str]:
    res: List[str] = []
    try:
        lines = Path(obj_path).read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return res

    for line in lines:
        m = _OBJ_MTLLIB_RE.match(line)
        if not m:
            continue
        tail = m.group(1).strip()
        parts = tail.split()
        for p in parts:
            p = p.strip().strip('"').strip("'")
            if p:
                res.append(p)

    out: List[str] = []
    seen = set()
    for x in res:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def parse_mtl_refs(mtl_path: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    try:
        lines = Path(mtl_path).read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return out

    for line in lines:
        m = _MTL_MAP_RE.match(line)
        if not m:
            continue
        key = m.group(1).lower()
        val = _strip_mtl_opts(m.group(2))
        if not val:
            continue

        if key == "map_kd":
            out.setdefault("basecolor_ref", val)
        elif key in ("map_bump", "bump"):
            out.setdefault("bump_ref", val)
        elif key == "map_d":
            out.setdefault("opacity_ref", val)
        elif key == "map_ks":
            out.setdefault("specular_ref", val)
        elif key == "map_ka":
            out.setdefault("ambient_ref", val)

    return out


def parse_mtl_materials(mtl_path: str) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    current: Optional[str] = None

    try:
        lines = Path(mtl_path).read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return out

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        low = line.lower()
        if low.startswith("newmtl "):
            current = line[7:].strip()
            if current:
                out.setdefault(current, {})
            continue

        if current is None:
            continue

        m = _MTL_MAP_RE.match(line)
        if not m:
            continue

        key = m.group(1).lower()
        val = _strip_mtl_opts(m.group(2))
        if not val:
            continue

        slot = out[current]
        if key == "map_kd":
            slot["basecolor_ref"] = val
        elif key in ("map_bump", "bump"):
            slot["bump_ref"] = val
        elif key == "map_d":
            slot["opacity_ref"] = val
        elif key == "map_ks":
            slot["specular_ref"] = val
        elif key == "map_ka":
            slot["ambient_ref"] = val

    return out


def resolve_texture_ref(ref: str, model_dir: Path, idx: Dict[str, str]) -> Optional[str]:
    if not ref:
        return None
    s = ref.strip().strip('"').strip("'")
    if not s:
        return None

    p = Path(s).expanduser()
    if p.is_absolute() and p.is_file():
        return str(p)

    cand = (model_dir / p).resolve()
    if cand.is_file():
        return str(cand)

    base = os.path.basename(s).lower()
    if base in idx:
        return idx[base]
    return None


# ============================================================
# PBR material building
# ============================================================

def _hsv_to_rgb(h: float, s: float, v: float) -> Tuple[float, float, float]:
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i %= 6
    if i == 0:
        return (v, t, p)
    if i == 1:
        return (q, v, p)
    if i == 2:
        return (p, v, t)
    if i == 3:
        return (p, q, v)
    if i == 4:
        return (t, p, v)
    return (v, p, q)


def _auto_color_rgb(key: str) -> Tuple[float, float, float]:
    h = (hash(key) & 0xFFFFFFFF) / 0xFFFFFFFF
    return _hsv_to_rgb(h, 0.45, 0.75)


def _named_color_rgb(value: Any) -> Optional[Tuple[float, float, float]]:
    text = str(value or "").strip().lower().replace("ё", "е")
    if not text:
        return None

    color_map = {
        "зелен": (0.33, 0.55, 0.36),
        "green": (0.33, 0.55, 0.36),
        "olive": (0.42, 0.46, 0.22),
        "sage": (0.56, 0.65, 0.52),
        "emerald": (0.12, 0.50, 0.34),
        "forest": (0.18, 0.39, 0.22),
        "бирюз": (0.23, 0.58, 0.54),
        "teal": (0.20, 0.50, 0.45),
        "черн": (0.10, 0.10, 0.10),
        "black": (0.10, 0.10, 0.10),
        "корич": (0.38, 0.27, 0.19),
        "brown": (0.38, 0.27, 0.19),
        "беж": (0.76, 0.70, 0.60),
        "beige": (0.76, 0.70, 0.60),
        "сер": (0.55, 0.55, 0.55),
        "gray": (0.55, 0.55, 0.55),
        "grey": (0.55, 0.55, 0.55),
        "бел": (0.85, 0.85, 0.85),
        "white": (0.85, 0.85, 0.85),
    }

    for token, rgb in color_map.items():
        if token in text:
            return rgb
    return None


def _supplier_candidate_tint_rgb(it: dict, fallback_name: str) -> Tuple[float, float, float]:
    meta = it.get("meta") if isinstance(it.get("meta"), dict) else {}
    supplier_candidate = meta.get("supplier_candidate") if isinstance(meta.get("supplier_candidate"), dict) else {}

    named = _named_color_rgb(supplier_candidate.get("color"))
    if named:
        return named

    color = it.get("color")
    if isinstance(color, (list, tuple)) and len(color) >= 3:
        try:
            return (float(color[0]), float(color[1]), float(color[2]))
        except Exception:
            pass

    return _auto_color_rgb(str(fallback_name))


@dataclass
class PBRMaps:
    basecolor: Optional[str] = None
    normal: Optional[str] = None
    roughness: Optional[str] = None
    metallic: Optional[str] = None
    ao: Optional[str] = None
    height: Optional[str] = None
    emissive: Optional[str] = None
    opacity: Optional[str] = None


def _set_image_colorspace(img: bpy.types.Image, is_data: bool) -> None:
    try:
        img.colorspace_settings.name = "Non-Color" if is_data else "sRGB"
    except Exception:
        pass


def _is_placeholder_image(img: Optional[bpy.types.Image]) -> bool:
    if img is None:
        return True
    try:
        name = str(getattr(img, "name", "") or "").strip().lower()
        filepath = str(getattr(img, "filepath", "") or "").strip()
        filepath_raw = str(getattr(img, "filepath_raw", "") or "").strip()
    except Exception:
        return True

    if name.startswith("map #") and not filepath and not filepath_raw:
        return True
    if filepath.startswith("/Map #") or filepath_raw.startswith("/Map #"):
        return True
    return False


def _socket_chain_has_real_image(socket) -> bool:
    if socket is None or not getattr(socket, "is_linked", False):
        return False

    visited = set()
    stack = [link.from_node for link in socket.links]

    while stack:
        node = stack.pop()
        if node is None:
            continue
        ptr = node.as_pointer()
        if ptr in visited:
            continue
        visited.add(ptr)

        if node.type == "TEX_IMAGE":
            img = getattr(node, "image", None)
            if img is None or _is_placeholder_image(img):
                continue
            try:
                if getattr(img, "size", (0, 0))[0] > 0:
                    return True
            except Exception:
                pass
            fp = getattr(img, "filepath", "") or getattr(img, "filepath_raw", "") or ""
            if fp:
                return True

        for inp in getattr(node, "inputs", []):
            if inp.is_linked:
                for link in inp.links:
                    stack.append(link.from_node)

    return False


def _should_apply_tint_rgb(tint_rgb: Optional[Tuple[float, float, float]]) -> bool:
    if not tint_rgb:
        return False
    try:
        r, g, b = [float(x) for x in tint_rgb[:3]]
    except Exception:
        return False
    return max(abs(r - 0.7), abs(g - 0.7), abs(b - 0.7)) > 0.03


def _blend_rgba(base_rgba, tint_rgb: Tuple[float, float, float], strength: float) -> Tuple[float, float, float, float]:
    base = tuple(float(x) for x in (base_rgba or (0.7, 0.7, 0.7, 1.0))[:4])
    s = max(0.0, min(1.0, float(strength)))
    return (
        (1.0 - s) * base[0] + s * float(tint_rgb[0]),
        (1.0 - s) * base[1] + s * float(tint_rgb[1]),
        (1.0 - s) * base[2] + s * float(tint_rgb[2]),
        base[3],
    )


def _apply_tint_to_material_nodes(
    mat: bpy.types.Material,
    tint_rgb: Optional[Tuple[float, float, float]],
    strength: float = 0.35,
) -> bool:
    if not mat or not _should_apply_tint_rgb(tint_rgb):
        return False

    mat.use_nodes = True
    nt = mat.node_tree
    if nt is None:
        return False

    nodes = nt.nodes
    links = nt.links
    bsdf = next((n for n in nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        return False

    base_input = bsdf.inputs.get("Base Color")
    if base_input is None:
        return False

    s = max(0.0, min(1.0, float(strength)))
    tint_node = nodes.new("ShaderNodeRGB")
    tint_node.label = "SUPPLIER_TINT_RGB"
    tint_node.location = (getattr(bsdf, "location", (300, 0))[0] - 420, getattr(bsdf, "location", (300, 0))[1] + 120)
    tint_node.outputs["Color"].default_value = (float(tint_rgb[0]), float(tint_rgb[1]), float(tint_rgb[2]), 1.0)

    mix = nodes.new("ShaderNodeMixRGB")
    mix.label = "SUPPLIER_TINT_MIX"
    mix.location = (getattr(bsdf, "location", (300, 0))[0] - 180, getattr(bsdf, "location", (300, 0))[1] + 20)
    mix.blend_type = "MIX"
    mix.inputs["Fac"].default_value = s

    incoming = [l for l in list(links) if l.to_node == bsdf and l.to_socket == base_input]
    if incoming:
        if not _socket_chain_has_real_image(base_input):
            for link in incoming:
                links.remove(link)
            try:
                base_input.default_value = (float(tint_rgb[0]), float(tint_rgb[1]), float(tint_rgb[2]), 1.0)
                return True
            except Exception:
                return False
        src_socket = incoming[0].from_socket
        for link in incoming:
            links.remove(link)
        try:
            links.new(src_socket, mix.inputs["Color1"])
            links.new(tint_node.outputs["Color"], mix.inputs["Color2"])
            links.new(mix.outputs["Color"], base_input)
            return True
        except Exception:
            return False

    try:
        base_input.default_value = _blend_rgba(base_input.default_value, tint_rgb, s)
        return True
    except Exception:
        return False


def _apply_tint_to_existing_materials(
    parent: bpy.types.Object,
    tint_rgb: Optional[Tuple[float, float, float]],
    strength: float = 0.35,
) -> int:
    if not _should_apply_tint_rgb(tint_rgb):
        return 0
    applied = 0
    for o in _iter_mesh_children(parent):
        me = getattr(o, "data", None)
        mats = getattr(me, "materials", None) if me else None
        for mat in (mats or []):
            if mat and _apply_tint_to_material_nodes(mat, tint_rgb=tint_rgb, strength=strength):
                applied += 1
    return applied


def _make_pbr_material(name: str, maps: PBRMaps, tint_rgb: Optional[Tuple[float, float, float]], tex_scale: float) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True

    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    nodes.clear()

    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (980, 0)

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (720, 0)
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    texcoord = nodes.new("ShaderNodeTexCoord")
    texcoord.location = (0, 0)
    mapping = nodes.new("ShaderNodeMapping")
    mapping.location = (220, 0)
    links.new(texcoord.outputs["Generated"], mapping.inputs["Vector"])
    mapping.inputs["Scale"].default_value = (float(tex_scale), float(tex_scale), float(tex_scale))

    def add_tex_image(path: str, loc: Tuple[int, int], is_data: bool) -> bpy.types.Node:
        n = nodes.new("ShaderNodeTexImage")
        n.location = loc
        try:
            img = bpy.data.images.load(path, check_existing=True)
            n.image = img
            _set_image_colorspace(img, is_data=is_data)
        except Exception:
            pass
        links.new(mapping.outputs["Vector"], n.inputs["Vector"])
        return n

    if maps.basecolor:
        base = add_tex_image(maps.basecolor, (440, 240), is_data=False)
        base_out = base.outputs["Color"]
        if maps.ao:
            ao = add_tex_image(maps.ao, (440, 460), is_data=True)
            mix = nodes.new("ShaderNodeMixRGB")
            mix.location = (600, 340)
            mix.blend_type = "MULTIPLY"
            mix.inputs["Fac"].default_value = 1.0
            links.new(base.outputs["Color"], mix.inputs["Color1"])
            links.new(ao.outputs["Color"], mix.inputs["Color2"])
            base_out = mix.outputs["Color"]

        if _should_apply_tint_rgb(tint_rgb):
            tint = nodes.new("ShaderNodeRGB")
            tint.location = (600, 120)
            tint.outputs["Color"].default_value = (float(tint_rgb[0]), float(tint_rgb[1]), float(tint_rgb[2]), 1.0)
            tint_mix = nodes.new("ShaderNodeMixRGB")
            tint_mix.location = (780, 220)
            tint_mix.blend_type = "MIX"
            tint_mix.inputs["Fac"].default_value = 0.35
            links.new(base_out, tint_mix.inputs["Color1"])
            links.new(tint.outputs["Color"], tint_mix.inputs["Color2"])
            base_out = tint_mix.outputs["Color"]

        links.new(base_out, bsdf.inputs["Base Color"])
    elif tint_rgb:
        bsdf.inputs["Base Color"].default_value = (float(tint_rgb[0]), float(tint_rgb[1]), float(tint_rgb[2]), 1.0)

    if maps.roughness:
        r = add_tex_image(maps.roughness, (440, -40), is_data=True)
        links.new(r.outputs["Color"], bsdf.inputs["Roughness"])

    if maps.metallic:
        m = add_tex_image(maps.metallic, (440, -260), is_data=True)
        links.new(m.outputs["Color"], bsdf.inputs["Metallic"])

    if maps.normal:
        ntex = add_tex_image(maps.normal, (440, -480), is_data=True)
        nmap = nodes.new("ShaderNodeNormalMap")
        nmap.location = (620, -480)
        links.new(ntex.outputs["Color"], nmap.inputs["Color"])
        links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])

    if maps.height:
        h = add_tex_image(maps.height, (440, -700), is_data=True)
        disp = nodes.new("ShaderNodeDisplacement")
        disp.location = (720, -700)
        links.new(h.outputs["Color"], disp.inputs["Height"])
        links.new(disp.outputs["Displacement"], out.inputs["Displacement"])

    if maps.emissive:
        e = add_tex_image(maps.emissive, (440, 660), is_data=False)
        if "Emission" in bsdf.inputs:
            links.new(e.outputs["Color"], bsdf.inputs["Emission"])
        elif "Emission Color" in bsdf.inputs:
            links.new(e.outputs["Color"], bsdf.inputs["Emission Color"])
        if "Emission Strength" in bsdf.inputs:
            bsdf.inputs["Emission Strength"].default_value = 1.0

    if maps.opacity and "Alpha" in bsdf.inputs:
        o = add_tex_image(maps.opacity, (440, 880), is_data=True)
        links.new(o.outputs["Color"], bsdf.inputs["Alpha"])
        try:
            mat.blend_method = "HASHED"
            mat.shadow_method = "HASHED"
        except Exception:
            pass

    return mat


# ============================================================
# Heuristic guessing
# ============================================================

def _pick_best_match(stem: str, files_lower: Dict[str, str], keys: List[str]) -> Optional[str]:
    stem_l = (stem or "").lower()
    candidates: List[Tuple[int, str]] = []

    for fn_l, full in files_lower.items():
        if stem_l and stem_l not in fn_l:
            continue

        score = 0
        for k in keys:
            if k in fn_l:
                score += 12

        if any(tok in fn_l for tok in _TRASH_TOKENS):
            score -= 50

        score += max(0, 60 - len(fn_l))
        if score > 0:
            candidates.append((score, full))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _guess_maps_from_scan(mesh_path: Optional[str], search_dirs: List[str], explicit_base: Optional[str]) -> PBRMaps:
    idx = build_image_index(search_dirs, max_depth=4, max_files=15000)
    if not idx:
        return PBRMaps(basecolor=(explicit_base if (explicit_base and os.path.isfile(explicit_base)) else None))

    base_keys = ["basecolor", "base_color", "albedo", "diffuse", "dif", "color", "col", "base", "tex", "texture"]
    nor_keys = ["normal", "norm", "nrm", "nmap"]
    rou_keys = ["roughness", "rough", "rgh"]
    met_keys = ["metallic", "metalness", "metal", "mtl", "met"]
    ao_keys = ["ambientocclusion", "ambient_occlusion", "occlusion", "occ", "ao"]
    hgt_keys = ["height", "displacement", "disp"]
    emi_keys = ["emissive", "emission", "emit", "glow"]
    op_keys = ["opacity", "alpha", "transparency"]

    stems: List[str] = []
    if mesh_path:
        mp = Path(mesh_path)
        stems.append(mp.stem)
        ar = detect_asset_root(str(mp))
        if not _is_hashlike_name(ar.name):
            stems.append(ar.name)
    stems.append("")

    out = PBRMaps()

    if explicit_base and os.path.isfile(explicit_base):
        out.basecolor = explicit_base

    for st in stems:
        if not out.basecolor:
            out.basecolor = _pick_best_match(st, idx, base_keys)
        if not out.normal:
            out.normal = _pick_best_match(st, idx, nor_keys)
        if not out.roughness:
            out.roughness = _pick_best_match(st, idx, rou_keys)
        if not out.metallic:
            out.metallic = _pick_best_match(st, idx, met_keys)
        if not out.ao:
            out.ao = _pick_best_match(st, idx, ao_keys)
        if not out.height:
            out.height = _pick_best_match(st, idx, hgt_keys)
        if not out.emissive:
            out.emissive = _pick_best_match(st, idx, emi_keys)
        if not out.opacity:
            out.opacity = _pick_best_match(st, idx, op_keys)

    return out


# ============================================================
# Existing materials
# ============================================================

def _iter_mesh_children(root: bpy.types.Object) -> List[bpy.types.Object]:
    stack = [root]
    out: List[bpy.types.Object] = []
    while stack:
        o = stack.pop()
        if o.type == "MESH":
            out.append(o)
        stack.extend(list(o.children))
    return out


def _object_has_any_material_slots(parent: bpy.types.Object) -> bool:
    for o in _iter_mesh_children(parent):
        me = getattr(o, "data", None)
        mats = getattr(me, "materials", None) if me else None
        if mats and len(mats) > 0 and any(m is not None for m in mats):
            return True
    return False


def _material_has_effective_basecolor_texture(mat: bpy.types.Material) -> bool:
    if not mat or not getattr(mat, "use_nodes", False) or not mat.node_tree:
        return False

    nt = mat.node_tree
    nodes = nt.nodes

    bsdf = next((n for n in nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        return False

    base_input = bsdf.inputs.get("Base Color")
    if base_input is None or not base_input.is_linked:
        return False

    return _socket_chain_has_real_image(base_input)


def _has_loaded_textures(parent: bpy.types.Object) -> bool:
    for o in _iter_mesh_children(parent):
        me = getattr(o, "data", None)
        mats = getattr(me, "materials", None) if me else None
        for mat in (mats or []):
            if _material_has_effective_basecolor_texture(mat):
                return True
    return False


def _apply_image_to_material_nodes(
    mat: bpy.types.Material,
    basecolor_img: Optional[bpy.types.Image] = None,
    normal_img: Optional[bpy.types.Image] = None,
    opacity_img: Optional[bpy.types.Image] = None,
) -> None:
    if not mat:
        return

    mat.use_nodes = True
    nt = mat.node_tree
    if nt is None:
        return

    nodes = nt.nodes
    links = nt.links

    out = next((n for n in nodes if n.type == "OUTPUT_MATERIAL"), None)
    bsdf = next((n for n in nodes if n.type == "BSDF_PRINCIPLED"), None)

    if out is None:
        out = nodes.new("ShaderNodeOutputMaterial")
        out.location = (500, 0)

    if bsdf is None:
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.location = (180, 0)
        try:
            links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
        except Exception:
            pass

    def remove_links_to_socket(socket_name: str) -> None:
        for l in list(links):
            try:
                if l.to_node == bsdf and l.to_socket and l.to_socket.name == socket_name:
                    links.remove(l)
            except Exception:
                pass

    if basecolor_img is not None:
        tex = nodes.new("ShaderNodeTexImage")
        tex.image = basecolor_img
        tex.location = (-180, 120)
        try:
            _set_image_colorspace(basecolor_img, is_data=False)
        except Exception:
            pass
        remove_links_to_socket("Base Color")
        try:
            links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
        except Exception:
            pass

    if normal_img is not None:
        tex_n = nodes.new("ShaderNodeTexImage")
        tex_n.image = normal_img
        tex_n.location = (-180, -180)
        try:
            _set_image_colorspace(normal_img, is_data=True)
        except Exception:
            pass

        nmap = nodes.new("ShaderNodeNormalMap")
        nmap.location = (0, -180)

        remove_links_to_socket("Normal")
        try:
            links.new(tex_n.outputs["Color"], nmap.inputs["Color"])
            links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
        except Exception:
            pass

    if opacity_img is not None and "Alpha" in bsdf.inputs:
        tex_a = nodes.new("ShaderNodeTexImage")
        tex_a.image = opacity_img
        tex_a.location = (-180, -420)
        try:
            _set_image_colorspace(opacity_img, is_data=True)
        except Exception:
            pass

        remove_links_to_socket("Alpha")
        try:
            links.new(tex_a.outputs["Color"], bsdf.inputs["Alpha"])
            mat.blend_method = "HASHED"
            mat.shadow_method = "HASHED"
        except Exception:
            pass


def _ensure_principled_basecolor_image(mat: bpy.types.Material, img: bpy.types.Image) -> None:
    mat.use_nodes = True
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links

    out = next((n for n in nodes if n.type == "OUTPUT_MATERIAL"), None)
    bsdf = next((n for n in nodes if n.type == "BSDF_PRINCIPLED"), None)

    if out is None:
        out = nodes.new("ShaderNodeOutputMaterial")
        out.location = (420, 0)

    if bsdf is None:
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.location = (120, 0)
        try:
            links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
        except Exception:
            pass

    tex = nodes.new("ShaderNodeTexImage")
    tex.image = img
    tex.location = (-200, 0)

    try:
        for l in list(links):
            if l.to_node == bsdf and l.to_socket and l.to_socket.name == "Base Color":
                links.remove(l)
        links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    except Exception:
        pass


def _relink_missing_images(parent: bpy.types.Object, idx: Dict[str, str]) -> Tuple[int, List[str]]:
    used: List[str] = []
    fixed = 0
    if not idx:
        return 0, used

    for o in _iter_mesh_children(parent):
        me = getattr(o, "data", None)
        mats = getattr(me, "materials", None) if me else None
        for mat in (mats or []):
            if not mat or not getattr(mat, "use_nodes", False) or not mat.node_tree:
                continue
            for n in mat.node_tree.nodes:
                if n.type != "TEX_IMAGE":
                    continue

                img = getattr(n, "image", None)
                if img and not _is_placeholder_image(img) and getattr(img, "size", (0, 0))[0] > 0:
                    continue

                want = None
                if img and not _is_placeholder_image(img) and getattr(img, "filepath", ""):
                    want = os.path.basename(img.filepath)
                if not want:
                    want = (getattr(n, "label", "") or getattr(n, "name", "") or "").strip()
                if not want:
                    continue

                cand = idx.get(os.path.basename(want).lower())
                if not cand:
                    continue

                try:
                    new_img = bpy.data.images.load(cand, check_existing=True)
                    n.image = new_img
                    fixed += 1
                    used.append(cand)
                except Exception:
                    pass

    return fixed, used


def _apply_mtl_map_kd_to_existing_mats(parent: bpy.types.Object, mesh_path: str, idx: Dict[str, str], verbose: bool) -> bool:
    obj_path = Path(mesh_path).resolve()
    model_dir = obj_path.parent

    mtl_names = parse_obj_mtl_files(str(obj_path))
    if not mtl_names:
        cand = next(iter(model_dir.glob("*.mtl")), None)
        if cand:
            mtl_names = [cand.name]

    if not mtl_names:
        return False

    merged_mtl: Dict[str, Dict[str, str]] = {}
    for mtl_name in mtl_names:
        mtl_path = (model_dir / mtl_name).resolve()
        if not mtl_path.is_file():
            rp = _resolve_path_maybe(model_dir, mtl_name)
            if rp and Path(rp).is_file():
                mtl_path = Path(rp)
            else:
                continue

        parsed = parse_mtl_materials(str(mtl_path))
        for k, v in parsed.items():
            merged_mtl[k] = v

    if not merged_mtl:
        return False

    def load_img(tex_ref: Optional[str], is_data: bool) -> Optional[bpy.types.Image]:
        if not tex_ref:
            return None
        resolved = resolve_texture_ref(tex_ref, model_dir, idx)
        if not resolved or not os.path.isfile(resolved):
            return None
        try:
            img = bpy.data.images.load(resolved, check_existing=True)
            _set_image_colorspace(img, is_data=is_data)
            return img
        except Exception:
            return None

    applied_any = False

    for o in _iter_mesh_children(parent):
        me = getattr(o, "data", None)
        mats = getattr(me, "materials", None) if me else None
        if not mats:
            continue

        for mat in mats:
            if not mat:
                continue

            mtl_info = merged_mtl.get(mat.name)

            if mtl_info is None:
                mat_name_l = mat.name.lower().strip()
                for mk, mv in merged_mtl.items():
                    mk_l = mk.lower().strip()
                    if mk_l == mat_name_l or mk_l in mat_name_l or mat_name_l in mk_l:
                        mtl_info = mv
                        break

            if mtl_info is None:
                continue

            base_img = load_img(mtl_info.get("basecolor_ref"), is_data=False)
            if base_img is None:
                base_img = load_img(mtl_info.get("ambient_ref"), is_data=False)

            normal_img = load_img(mtl_info.get("bump_ref"), is_data=True)
            opacity_img = load_img(mtl_info.get("opacity_ref"), is_data=True)

            if base_img is None and normal_img is None and opacity_img is None:
                continue

            _apply_image_to_material_nodes(
                mat=mat,
                basecolor_img=base_img,
                normal_img=normal_img,
                opacity_img=opacity_img,
            )
            applied_any = True

            if verbose:
                print(
                    f"[MTL] {parent.name} | mat={mat.name} | "
                    f"base={getattr(base_img, 'filepath', None)} | "
                    f"normal={getattr(normal_img, 'filepath', None)} | "
                    f"alpha={getattr(opacity_img, 'filepath', None)}"
                )

    return applied_any


def _fallback_apply_largest_image_existing_mats(parent: bpy.types.Object, idx: Dict[str, str], verbose: bool) -> bool:
    if not idx:
        return False

    best_path = None
    best_size = -1
    for p in idx.values():
        try:
            sz = os.path.getsize(p)
        except Exception:
            continue
        if sz > best_size:
            best_size = sz
            best_path = p

    if not best_path:
        return False

    try:
        img = bpy.data.images.load(best_path, check_existing=True)
    except Exception:
        return False

    applied = False
    for o in _iter_mesh_children(parent):
        me = getattr(o, "data", None)
        mats = getattr(me, "materials", None) if me else None
        for mat in (mats or []):
            if mat:
                _ensure_principled_basecolor_image(mat, img)
                applied = True

    if applied:
        _log(verbose, f"[Textures] {parent.name}: fallback largest={best_path}")
    return applied


def _fallback_apply_flat_tint_existing_mats(
    parent: bpy.types.Object,
    tint_rgb: Optional[Tuple[float, float, float]],
    verbose: bool,
) -> bool:
    if not _should_apply_tint_rgb(tint_rgb):
        return False

    mat = _make_pbr_material(
        name=f"Mat_{parent.name}_FlatTint",
        maps=PBRMaps(),
        tint_rgb=tint_rgb,
        tex_scale=1.0,
    )

    applied = False
    for o in _iter_mesh_children(parent):
        me = getattr(o, "data", None)
        mats = getattr(me, "materials", None) if me else None
        if mats is None:
            continue
        if not mats:
            mats.append(mat)
            applied = True
            continue
        for i in range(len(mats)):
            mats[i] = mat
            applied = True

    if applied:
        _log(verbose, f"[Textures] {parent.name}: fallback flat tint applied")
    return applied


def _ensure_textures(
    parent: bpy.types.Object,
    mesh_path: Optional[str],
    mesh_texture_dirs: Optional[List[str]],
    texture_path: Optional[str],
    texture_files: Optional[List[str]],
    texture_scale: float,
    tint_rgb: Optional[Tuple[float, float, float]],
    keep_existing_mats: bool,
    verbose: bool,
) -> None:
    search_dirs = build_search_dirs(mesh_path, mesh_texture_dirs)
    idx = build_image_index(search_dirs, max_depth=4, max_files=15000)

    if keep_existing_mats:
        if _has_loaded_textures(parent):
            _apply_tint_to_existing_materials(parent, tint_rgb=tint_rgb, strength=0.35)
            _log(verbose, f"[Textures] {parent.name}: effective basecolor texture already exists -> keep")
            return

        fixed, _ = _relink_missing_images(parent, idx)
        if fixed > 0 and _has_loaded_textures(parent):
            _apply_tint_to_existing_materials(parent, tint_rgb=tint_rgb, strength=0.35)
            _log(verbose, f"[Textures] {parent.name}: relink fixed={fixed}")
            return

        if mesh_path and os.path.isfile(mesh_path):
            if _apply_mtl_map_kd_to_existing_mats(parent, mesh_path, idx, verbose):
                if _has_loaded_textures(parent):
                    _apply_tint_to_existing_materials(parent, tint_rgb=tint_rgb, strength=0.35)
                    _log(verbose, f"[Textures] {parent.name}: restored from MTL")
                    return

        if _object_has_any_material_slots(parent):
            if _fallback_apply_largest_image_existing_mats(parent, idx, verbose):
                _apply_tint_to_existing_materials(parent, tint_rgb=tint_rgb, strength=0.35)
                if _has_loaded_textures(parent):
                    return
            if _fallback_apply_flat_tint_existing_mats(parent, tint_rgb=tint_rgb, verbose=verbose):
                return
            _log(verbose, f"[Textures] {parent.name}: materials exist, but MTL restore failed")
            return

    maps = PBRMaps()
    explicit_base = texture_path if (texture_path and os.path.isfile(texture_path)) else None

    if mesh_path and os.path.isfile(mesh_path):
        model_dir = Path(mesh_path).resolve().parent
        mtl_names = parse_obj_mtl_files(mesh_path)
        if not mtl_names:
            cand = next(iter(model_dir.glob("*.mtl")), None)
            if cand:
                mtl_names = [cand.name]

        for mtl_name in mtl_names:
            mtl_path = (model_dir / mtl_name).resolve()
            if not mtl_path.is_file():
                rp = _resolve_path_maybe(model_dir, mtl_name)
                if rp and Path(rp).is_file():
                    mtl_path = Path(rp)
                else:
                    continue

            refs = parse_mtl_refs(str(mtl_path))
            if not maps.basecolor and "basecolor_ref" in refs:
                maps.basecolor = resolve_texture_ref(refs["basecolor_ref"], model_dir, idx)
            if not maps.normal and "bump_ref" in refs:
                maps.normal = resolve_texture_ref(refs["bump_ref"], model_dir, idx)
            if not maps.opacity and "opacity_ref" in refs:
                maps.opacity = resolve_texture_ref(refs["opacity_ref"], model_dir, idx)
            if not maps.basecolor and "ambient_ref" in refs:
                maps.basecolor = resolve_texture_ref(refs["ambient_ref"], model_dir, idx)

            if any([maps.basecolor, maps.normal, maps.opacity]):
                break

    base_dir = Path(mesh_path).resolve().parent if mesh_path else Path.cwd()
    resolved_files: List[str] = []
    for tf in (texture_files or []):
        rp = _resolve_path_maybe(base_dir, tf)
        if rp and os.path.isfile(rp):
            resolved_files.append(rp)
        else:
            bn = os.path.basename(str(tf)).lower()
            if bn in idx:
                resolved_files.append(idx[bn])

    for p in resolved_files:
        bn = os.path.basename(p).lower()
        if not maps.basecolor and any(k in bn for k in ["basecolor", "base_color", "albedo", "diffuse", "color", "col", "tex", "texture"]):
            maps.basecolor = p
        if not maps.normal and any(k in bn for k in ["normal", "norm", "nrm", "nmap"]):
            maps.normal = p
        if not maps.roughness and any(k in bn for k in ["roughness", "rough", "rgh"]):
            maps.roughness = p
        if not maps.metallic and any(k in bn for k in ["metallic", "metalness", "metal", "mtl", "met"]):
            maps.metallic = p
        if not maps.ao and any(k in bn for k in ["ao", "occlusion", "occ", "ambientocclusion"]):
            maps.ao = p

    if not maps.basecolor and explicit_base:
        maps.basecolor = explicit_base

    if not any([maps.basecolor, maps.normal, maps.roughness, maps.metallic, maps.ao, maps.height, maps.emissive, maps.opacity]):
        maps = _guess_maps_from_scan(mesh_path, search_dirs, explicit_base=explicit_base)
    else:
        guessed = _guess_maps_from_scan(mesh_path, search_dirs, explicit_base=explicit_base)
        maps.roughness = maps.roughness or guessed.roughness
        maps.metallic = maps.metallic or guessed.metallic
        maps.ao = maps.ao or guessed.ao
        maps.height = maps.height or guessed.height
        maps.emissive = maps.emissive or guessed.emissive
        maps.opacity = maps.opacity or guessed.opacity

    _log(verbose, f"[Textures] {parent.name}: build PBR maps={maps}")

    mat = _make_pbr_material(
        name=f"Mat_{parent.name}",
        maps=maps,
        tint_rgb=tint_rgb,
        tex_scale=float(texture_scale or 1.0),
    )

    for o in _iter_mesh_children(parent):
        me = o.data
        if not me.materials:
            me.materials.append(mat)
        else:
            for i in range(len(me.materials)):
                me.materials[i] = mat


# ============================================================
# Room environment textures
# ============================================================

def _list_env_textures(env_dir: str) -> Dict[str, List[str]]:
    d = str(Path(env_dir).expanduser().resolve())
    exts = ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.tif", "*.tiff"]

    def collect(prefixes: List[str]) -> List[str]:
        out: List[str] = []
        for pref in prefixes:
            for e in exts:
                out.extend(glob(os.path.join(d, f"{pref}{e}")))
        out = sorted({str(Path(p).resolve()) for p in out})
        return out

    return {
        "floor": collect(["floor_", "floor"]),
        "wall": collect(["wall_", "wall"]),
        "window": collect(["window_", "window"]),
        "door": collect(["door_", "door"]),
    }


def _choose_texture(kind: str, candidates: List[str], style_text: Optional[str], rng: random.Random) -> Optional[str]:
    del kind
    del style_text
    if not candidates:
        return None
    return rng.choice(candidates)


def _make_image_material(name: str, image_path: str, uv_scale: float = 1.0, is_data: bool = False) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    nodes.clear()

    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (700, 0)

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (450, 0)
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    texcoord = nodes.new("ShaderNodeTexCoord")
    texcoord.location = (0, 0)

    mapping = nodes.new("ShaderNodeMapping")
    mapping.location = (170, 0)
    links.new(texcoord.outputs["UV"], mapping.inputs["Vector"])
    mapping.inputs["Scale"].default_value = (float(uv_scale), float(uv_scale), float(uv_scale))

    tex = nodes.new("ShaderNodeTexImage")
    tex.location = (340, 0)
    try:
        img = bpy.data.images.load(image_path, check_existing=True)
        tex.image = img
        _set_image_colorspace(img, is_data=is_data)
    except Exception:
        pass
    links.new(mapping.outputs["Vector"], tex.inputs["Vector"])
    links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])

    if "Roughness" in bsdf.inputs:
        bsdf.inputs["Roughness"].default_value = 0.85
    if "Specular" in bsdf.inputs:
        bsdf.inputs["Specular"].default_value = 0.10

    return mat


def _assign_material_to_object(obj: bpy.types.Object, mat: bpy.types.Material) -> None:
    if obj.type != "MESH":
        return
    me = obj.data
    try:
        me.materials.clear()
    except Exception:
        pass
    if len(me.materials) == 0:
        me.materials.append(mat)
    else:
        for i in range(len(me.materials)):
            me.materials[i] = mat


def _make_bbox_wire_material(name: str, rgba: Tuple[float, float, float, float]) -> bpy.types.Material:
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    nodes.clear()

    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (320, 0)

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (40, 0)
    try:
        bsdf.inputs["Base Color"].default_value = rgba
    except Exception:
        pass
    try:
        bsdf.inputs["Roughness"].default_value = 0.25
    except Exception:
        pass
    try:
        bsdf.inputs["Specular"].default_value = 0.0
    except Exception:
        pass
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    try:
        mat.shadow_method = "NONE"
    except Exception:
        pass
    return mat


def _make_renderable_bbox_box(
    aabb: Dict[str, float],
    name: str,
    collection: bpy.types.Collection,
    rgba: Tuple[float, float, float, float] = (0.02, 0.02, 0.02, 1.0),
    thickness: float = 0.02,
) -> bpy.types.Object:
    obj = _add_aabb_box(aabb, name, collection)
    obj.display_type = "WIRE"
    obj.show_in_front = True
    try:
        obj.color = rgba
    except Exception:
        pass
    try:
        mod = obj.modifiers.new(name="BBoxWireframe", type="WIREFRAME")
        mod.thickness = max(float(thickness), 0.002)
        mod.use_replace = True
        mod.use_even_offset = True
    except Exception:
        pass
    mat = _make_bbox_wire_material("MAT_REPLACED_BBOX", rgba)
    _assign_material_to_object(obj, mat)
    return obj


def _make_glass_material(name: str, image_path: Optional[str]) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    nodes.clear()

    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (700, 0)

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (430, 0)
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    if "Transmission Weight" in bsdf.inputs:
        bsdf.inputs["Transmission Weight"].default_value = 1.0
    elif "Transmission" in bsdf.inputs:
        bsdf.inputs["Transmission"].default_value = 1.0

    if "Roughness" in bsdf.inputs:
        bsdf.inputs["Roughness"].default_value = 0.05
    if "IOR" in bsdf.inputs:
        bsdf.inputs["IOR"].default_value = 1.45

    if image_path and os.path.isfile(image_path):
        texcoord = nodes.new("ShaderNodeTexCoord")
        texcoord.location = (0, 0)

        mapping = nodes.new("ShaderNodeMapping")
        mapping.location = (170, 0)
        links.new(texcoord.outputs["UV"], mapping.inputs["Vector"])

        tex = nodes.new("ShaderNodeTexImage")
        tex.location = (340, 0)
        img = bpy.data.images.load(image_path, check_existing=True)
        tex.image = img
        _set_image_colorspace(img, is_data=False)

        links.new(mapping.outputs["Vector"], tex.inputs["Vector"])
        links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])

        if "Alpha" in bsdf.inputs:
            links.new(tex.outputs["Alpha"], bsdf.inputs["Alpha"])
            try:
                mat.blend_method = "HASHED"
                mat.shadow_method = "HASHED"
            except Exception:
                pass
        else:
            try:
                mat.blend_method = "HASHED"
                mat.shadow_method = "HASHED"
            except Exception:
                pass

    return mat


# ============================================================
# Room spec mesh building
# ============================================================

def _synthesize_walls_from_floor_polygon(room_dict: dict) -> list[dict]:
    """
    Если в room-spec нет room["walls"], строим их автоматически
    по соседним вершинам floor_polygon:
        0->1, 1->2, ..., n-1->0
    """
    poly = room_dict.get("floor_polygon") or []
    if not isinstance(poly, list) or len(poly) < 3:
        return []

    walls = []
    n = len(poly)
    for i in range(n):
        walls.append({
            "id": f"w{i}",
            "from_vertex": i,
            "to_vertex": (i + 1) % n,
        })
    return walls


def _room_spec_from_bounds(
    room_engine: dict,
    default_bounds: Tuple[float, float, float, float, float, float],
) -> dict:
    x1, x2, y1, y2, z1, z2 = default_bounds
    if room_engine and all(k in room_engine for k in ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]):
        x1, x2 = float(room_engine["x_min"]), float(room_engine["x_max"])
        y1, y2 = float(room_engine["y_min"]), float(room_engine["y_max"])
        z1, z2 = float(room_engine["z_min"]), float(room_engine["z_max"])

    return {
        "room": {
            "floor_z": z1,
            "ceiling_height": max(z2 - z1, 0.1),
            "floor_polygon": [
                {"x": x1, "y": y1},
                {"x": x2, "y": y1},
                {"x": x2, "y": y2},
                {"x": x1, "y": y2},
            ],
            "walls": [
                {"id": "w0", "from_vertex": 0, "to_vertex": 1},
                {"id": "w1", "from_vertex": 1, "to_vertex": 2},
                {"id": "w2", "from_vertex": 2, "to_vertex": 3},
                {"id": "w3", "from_vertex": 3, "to_vertex": 0},
            ],
            "doors": [],
            "windows": [],
        }
    }

def _poly_signed_area_xy(pts: List[Tuple[float, float]]) -> float:
    a = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return 0.5 * a


def _normalize2(x: float, y: float) -> Tuple[float, float]:
    l = math.hypot(x, y)
    if l < 1e-12:
        return (1.0, 0.0)
    return (x / l, y / l)


def _make_mesh_object(name: str, verts: List[Tuple[float, float, float]], faces: List[Tuple[int, ...]], coll) -> bpy.types.Object:
    me = bpy.data.meshes.new(name + "_MESH")
    me.from_pydata(verts, [], faces)
    me.update()

    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    _unlink_from_all_collections(obj)
    coll.objects.link(obj)
    return obj


def _ensure_uv_layer(me: bpy.types.Mesh, name: str = "UVMap") -> None:
    if not me.uv_layers:
        me.uv_layers.new(name=name)


def _set_uvs_floor_xy(obj: bpy.types.Object, uv_scale: float = 1.0) -> None:
    me = obj.data
    _ensure_uv_layer(me)
    uv = me.uv_layers.active.data

    for poly in me.polygons:
        for li in poly.loop_indices:
            vi = me.loops[li].vertex_index
            x, y, _z = me.vertices[vi].co
            uv[li].uv = (x * uv_scale, y * uv_scale)


def _set_uvs_wall_sz(obj: bpy.types.Object, origin_xy: Tuple[float, float], dir_xy: Tuple[float, float], uv_scale: float = 1.0) -> None:
    me = obj.data
    _ensure_uv_layer(me)
    uv = me.uv_layers.active.data

    ox, oy = origin_xy
    dx, dy = dir_xy

    for poly in me.polygons:
        for li in poly.loop_indices:
            vi = me.loops[li].vertex_index
            x, y, z = me.vertices[vi].co
            s = (x - ox) * dx + (y - oy) * dy
            uv[li].uv = (s * uv_scale, z * uv_scale)


def _make_floor_from_polygon(name: str, poly_xy: List[Tuple[float, float]], z: float, coll) -> bpy.types.Object:
    bm = bmesh.new()
    verts = [bm.verts.new((x, y, z)) for (x, y) in poly_xy]
    bm.faces.new(verts)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])

    me = bpy.data.meshes.new(name + "_MESH")
    bm.to_mesh(me)
    bm.free()
    me.update()

    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    _unlink_from_all_collections(obj)
    coll.objects.link(obj)
    return obj


def _make_wall_quad(name: str, p0: Tuple[float, float], p1: Tuple[float, float], z0: float, z1: float, coll) -> bpy.types.Object:
    x0, y0 = p0
    x1, y1 = p1
    verts = [
        (x0, y0, z0),
        (x1, y1, z0),
        (x1, y1, z1),
        (x0, y0, z1),
        (x0, y0, z0),
        (x0, y0, z1),
        (x1, y1, z1),
        (x1, y1, z0),
    ]
    # Two coincident quads with opposite winding keep walls visible from both
    # sides while remaining infinitely thin.
    faces = [(0, 1, 2, 3), (4, 5, 6, 7)]
    return _make_mesh_object(name, verts, faces, coll)


def _make_decal_on_wall(
    name: str,
    wall_p0: Tuple[float, float],
    wall_p1: Tuple[float, float],
    inward_n: Tuple[float, float],
    s0: float,
    width: float,
    z0: float,
    height: float,
    coll,
    offset: float = 0.002,
) -> bpy.types.Object:
    x0, y0 = wall_p0
    x1, y1 = wall_p1
    ex, ey = (x1 - x0, y1 - y0)
    ex, ey = _normalize2(ex, ey)

    sc = s0 + 0.5 * width
    cx = x0 + ex * sc
    cy = y0 + ey * sc

    nx, ny = inward_n
    cx += nx * offset
    cy += ny * offset

    u0 = -0.5 * width
    u1 = +0.5 * width
    v0 = z0
    v1 = z0 + height

    verts = [
        (cx + ex * u0, cy + ey * u0, v0),
        (cx + ex * u1, cy + ey * u1, v0),
        (cx + ex * u1, cy + ey * u1, v1),
        (cx + ex * u0, cy + ey * u0, v1),
    ]
    faces = [(0, 1, 2, 3)]
    return _make_mesh_object(name, verts, faces, coll)

def _make_double_sided_decal_on_wall(
    name: str,
    wall_p0,
    wall_p1,
    inward_n,
    s0: float,
    width: float,
    z0: float,
    height: float,
    coll,
    offset: float = 0.003,
):
    """
    Создаёт две тонкие декали на стене:
    - со стороны комнаты
    - со внешней стороны

    inward_n — нормаль, направленная внутрь комнаты.
    """
    nx, ny = inward_n

    obj_in = _make_decal_on_wall(
        name=f"{name}_Inner",
        wall_p0=wall_p0,
        wall_p1=wall_p1,
        inward_n=(nx, ny),
        s0=s0,
        width=width,
        z0=z0,
        height=height,
        coll=coll,
        offset=offset,
    )

    obj_out = _make_decal_on_wall(
        name=f"{name}_Outer",
        wall_p0=wall_p0,
        wall_p1=wall_p1,
        inward_n=(-nx, -ny),
        s0=s0,
        width=width,
        z0=z0,
        height=height,
        coll=coll,
        offset=offset,
    )

    return [obj_in, obj_out]

def build_room_from_spec(
    room_json: dict,
    coll_room: bpy.types.Collection,
    env_textures_dir: str,
    style_text: Optional[str],
    seed: int,
    verbose: bool,
) -> Tuple[List[bpy.types.Object], mathutils.Vector, mathutils.Vector]:
    r = room_json["room"]
    floor_z = float(r.get("floor_z", r.get("z_min", 0.0)))
    if "z_max" in r:
        H = float(r["z_max"]) - floor_z
    else:
        H = float(r.get("ceiling_height", 2.8))
    H = max(H, 0.1)

    poly = r["floor_polygon"]
    poly_xy = [(float(p["x"]), float(p["y"])) for p in poly]

    area = _poly_signed_area_xy(poly_xy)
    ccw = (area > 0.0)

    floor_tex = wall_tex = door_tex = window_tex = None
    if env_textures_dir and os.path.isdir(env_textures_dir):
        tex = _list_env_textures(env_textures_dir)
        rng = random.Random(
            int(seed) if int(seed) != 0 else (hash(json.dumps(r, sort_keys=True)) & 0xFFFFFFFF)
        )

        floor_tex = _choose_texture("floor", tex.get("floor", []), style_text, rng)
        wall_tex = _choose_texture("wall", tex.get("wall", []), style_text, rng)
        door_tex = _choose_texture("door", tex.get("door", []), style_text, rng)
        window_tex = _choose_texture("window", tex.get("window", []), style_text, rng)

    if verbose:
        print(f"[RoomSpec] ccw={ccw} H={H}")
        print(f"[RoomSpec] floor_tex={floor_tex}")
        print(f"[RoomSpec] wall_tex ={wall_tex}")
        print(f"[RoomSpec] door_tex ={door_tex}")
        print(f"[RoomSpec] win_tex  ={window_tex}")

    floor_mat = _make_image_material("MAT_ROOM_FLOOR", floor_tex, uv_scale=1.0) if floor_tex else None
    wall_mat = _make_image_material("MAT_ROOM_WALL", wall_tex, uv_scale=1.0) if wall_tex else None
    door_mat = _make_image_material("MAT_ROOM_DOOR", door_tex, uv_scale=1.0) if door_tex else None
    win_mat = _make_glass_material("MAT_ROOM_WINDOW", window_tex) if window_tex else _make_glass_material("MAT_ROOM_WINDOW", None)

    out_objs: List[bpy.types.Object] = []

    floor_obj = _make_floor_from_polygon("Room_Floor", poly_xy, z=floor_z, coll=coll_room)
    _set_uvs_floor_xy(floor_obj, uv_scale=1.0)
    if floor_mat:
        _assign_material_to_object(floor_obj, floor_mat)
    out_objs.append(floor_obj)

    walls = r.get("walls", [])
    if not walls:
        walls = _synthesize_walls_from_floor_polygon(r)
        if verbose:
            print(f"[RoomSpec] walls missing -> synthesized {len(walls)} walls from floor_polygon")

    wall_by_id = {}

    for w in walls:
        wid = str(w.get("id", f"w{len(wall_by_id)}"))
        i0 = int(w["from_vertex"])
        i1 = int(w["to_vertex"])

        if i0 < 0 or i0 >= len(poly_xy) or i1 < 0 or i1 >= len(poly_xy):
            if verbose:
                print(f"[RoomSpec] skip invalid wall {wid}: from={i0} to={i1}")
            continue

        p0 = poly_xy[i0]
        p1 = poly_xy[i1]

        ex, ey = (p1[0] - p0[0], p1[1] - p0[1])
        ex, ey = _normalize2(ex, ey)

        if abs(ex) < 1e-12 and abs(ey) < 1e-12:
            if verbose:
                print(f"[RoomSpec] skip degenerate wall {wid}")
            continue

        if ccw:
            nx, ny = (-ey, ex)
        else:
            nx, ny = (ey, -ex)

        wall_obj = _make_wall_quad(
            f"Room_Wall_{wid}",
            p0,
            p1,
            z0=floor_z,
            z1=floor_z + H,
            coll=coll_room,
        )
        _set_uvs_wall_sz(wall_obj, origin_xy=p0, dir_xy=(ex, ey), uv_scale=1.0)

        if wall_mat:
            _assign_material_to_object(wall_obj, wall_mat)

        out_objs.append(wall_obj)

        wall_by_id[wid] = {
            "p0": p0,
            "p1": p1,
            "dir": (ex, ey),
            "in": (nx, ny),
        }

    for d in r.get("doors", []):
        wid = d["wall_id"]
        info = wall_by_id.get(wid)
        if not info:
            continue

        s0 = float(d["s"])
        width = float(d["width"])
        z0 = floor_z + float(d.get("z0", 0.0))
        height = float(d.get("height", 2.0))

        door_objs = _make_double_sided_decal_on_wall(
            name=f"Room_Door_{d['id']}",
            wall_p0=info["p0"],
            wall_p1=info["p1"],
            inward_n=info["in"],
            s0=s0,
            width=width,
            z0=z0,
            height=height,
            coll=coll_room,
            offset=0.003,
        )

        for door_obj in door_objs:
            _set_uvs_wall_sz(door_obj, origin_xy=info["p0"], dir_xy=info["dir"], uv_scale=1.0)
            if door_mat:
                _assign_material_to_object(door_obj, door_mat)
            out_objs.append(door_obj)

    for w in r.get("windows", []):
        wid = w["wall_id"]
        info = wall_by_id.get(wid)
        if not info:
            continue

        s0 = float(w["s"])
        width = float(w["width"])
        z0 = floor_z + float(w.get("z0", 0.9))
        height = float(w.get("height", 1.2))

        win_objs = _make_double_sided_decal_on_wall(
            name=f"Room_Window_{w['id']}",
            wall_p0=info["p0"],
            wall_p1=info["p1"],
            inward_n=info["in"],
            s0=s0,
            width=width,
            z0=z0,
            height=height,
            coll=coll_room,
            offset=0.004,
        )

        for win_obj in win_objs:
            _set_uvs_wall_sz(win_obj, origin_xy=info["p0"], dir_xy=info["dir"], uv_scale=1.0)
            if win_mat:
                _assign_material_to_object(win_obj, win_mat)
            out_objs.append(win_obj)

    bmin, bmax = _world_bounds_mesh_objects(out_objs)
    return out_objs, bmin, bmax
# ============================================================
# Placement
# ============================================================

def place_in_aabb(
    objs: List[bpy.types.Object],
    aabb: Dict[str, float],
    rotation_deg_engine: float,
    fit_mode: str,
    parent_name: str,
    collection: bpy.types.Collection,
    snap_to_floor: bool,
    floor_offset: float,
    semantic_group: str = "",
    snap_to_ceiling: bool = False,
    ceiling_offset: float = 0.0,
) -> Optional[bpy.types.Object]:
    objs, dropped_material_previews = _drop_material_preview_meshes(objs)
    if dropped_material_previews:
        for name in dropped_material_previews:
            print(f"[DBG] dropping imported material preview mesh {name}")
            obj = bpy.data.objects.get(name)
            if obj is not None:
                try:
                    bpy.data.objects.remove(obj, do_unlink=True)
                except Exception:
                    pass

    objs, dropped_meshes = _filter_imported_mesh_outliers(objs)
    if dropped_meshes:
        for meta in dropped_meshes:
            print(
                "[DBG] dropping imported outlier mesh "
                f"{meta['name']} diag={meta['diag']:.4f} "
                f"longest={meta['longest']:.4f} "
                f"footprint={meta['footprint']:.4f} "
                f"thin_ratio={meta['thin_ratio']:.5f}"
            )
            obj = bpy.data.objects.get(str(meta["name"]))
            if obj is not None:
                try:
                    bpy.data.objects.remove(obj, do_unlink=True)
                except Exception:
                    pass

    objs, dropped_clusters = _keep_primary_import_cluster(objs)
    if dropped_clusters:
        for meta in dropped_clusters:
            print(
                "[DBG] dropping imported distant cluster mesh "
                f"{meta['name']} diag={meta['diag']:.4f} "
                f"longest={meta['longest']:.4f} "
                f"distance_to_origin={meta['distance_to_origin']:.4f}"
            )
            obj = bpy.data.objects.get(str(meta["name"]))
            if obj is not None:
                try:
                    bpy.data.objects.remove(obj, do_unlink=True)
                except Exception:
                    pass

    mesh_objs = [o for o in objs if o.type == "MESH"]
    if not mesh_objs:
        return None

    def _fit_parent_once() -> None:
        bpy.context.view_layer.update()

        bmin, bmax = _world_bounds_mesh_objects(mesh_objs)
        cur = bmax - bmin

        tgt = mathutils.Vector(
            (
                float(aabb["x_max"]) - float(aabb["x_min"]),
                float(aabb["y_max"]) - float(aabb["y_min"]),
                float(aabb["z_max"]) - float(aabb["z_min"]),
            )
        )

        eps = 1e-9
        cur = mathutils.Vector((max(cur.x, eps), max(cur.y, eps), max(cur.z, eps)))
        sx, sy, sz = tgt.x / cur.x, tgt.y / cur.y, tgt.z / cur.z

        if (fit_mode or "stretch").lower() == "uniform":
            k = min(sx, sy, sz)
            parent.scale = (parent.scale.x * k, parent.scale.y * k, parent.scale.z * k)
        else:
            parent.scale = (parent.scale.x * sx, parent.scale.y * sy, parent.scale.z * sz)

        bpy.context.view_layer.update()

        bmin2, bmax2 = _world_bounds_mesh_objects(mesh_objs)
        cur_center = (bmin2 + bmax2) * 0.5
        tgt_center = mathutils.Vector(
            (
                0.5 * (float(aabb["x_min"]) + float(aabb["x_max"])),
                0.5 * (float(aabb["y_min"]) + float(aabb["y_max"])),
                0.5 * (float(aabb["z_min"]) + float(aabb["z_max"])),
            )
        )
        delta = tgt_center - cur_center

        if snap_to_floor and not snap_to_ceiling:
            bottom_after_center = bmin2.z + delta.z
            delta.z += (float(aabb["z_min"]) - bottom_after_center) + float(floor_offset)
        elif snap_to_ceiling and not snap_to_floor:
            top_after_center = bmax2.z + delta.z
            delta.z += (float(aabb["z_max"]) - top_after_center) - float(ceiling_offset)

        parent.location += delta
        bpy.context.view_layer.update()

    def _placement_metrics(rotation_deg: float) -> Tuple[bool, str, float]:
        bpy.context.view_layer.update()
        bmin, bmax = _world_bounds_mesh_objects(mesh_objs)
        cur = bmax - bmin
        tgt = mathutils.Vector(
            (
                float(aabb["x_max"]) - float(aabb["x_min"]),
                float(aabb["y_max"]) - float(aabb["y_min"]),
                float(aabb["z_max"]) - float(aabb["z_min"]),
            )
        )
        cur_center = (bmin + bmax) * 0.5
        tgt_center = mathutils.Vector(
            (
                0.5 * (float(aabb["x_min"]) + float(aabb["x_max"])),
                0.5 * (float(aabb["y_min"]) + float(aabb["y_max"])),
                0.5 * (float(aabb["z_min"]) + float(aabb["z_max"])),
            )
        )

        max_ratio = 1.85
        axis_failures: List[str] = []
        oversize_penalty = 0.0
        for axis_name, cur_val, tgt_val in (
            ("x", float(cur.x), float(tgt.x)),
            ("y", float(cur.y), float(tgt.y)),
            ("z", float(cur.z), float(tgt.z)),
        ):
            if tgt_val <= 1e-6:
                continue
            ratio = cur_val / tgt_val
            oversize_penalty += abs(ratio - 1.0)
            if ratio > max_ratio:
                axis_failures.append(f"{axis_name}:{ratio:.3f}")

        center_delta = cur_center - tgt_center
        max_target = max(float(tgt.x), float(tgt.y), float(tgt.z), 1e-6)
        center_ratio = center_delta.length / max_target
        if axis_failures:
            return False, "oversize " + ",".join(axis_failures), float("inf")
        if center_ratio > 0.65:
            return False, f"center_offset:{center_ratio:.3f}", float("inf")

        wall_penalty = 0.0
        if semantic_group in {"bed", "desk", "dresser", "nightstand", "side_table", "coffee_table", "shelf", "wardrobe", "tv_stand"}:
            wall_ctx = _nearest_room_wall_context(aabb)
            if wall_ctx is not None:
                wall_dir, room_dir, wall_dist = wall_ctx
                footprint = max(float(tgt.x), float(tgt.y), 1e-6)
                proximity = max(0.0, min(1.0, 1.0 - (wall_dist / max(0.6, footprint * 0.9 + 0.35))))
                if proximity > 1e-6:
                    rot_mat = mathutils.Matrix.Rotation(math.radians(float(rotation_deg)), 4, "Z")
                    back_vec = (rot_mat @ mathutils.Vector((0.0, -1.0, 0.0))).to_2d()
                    front_vec = (-back_vec).copy()
                    if back_vec.length > 1e-6:
                        back_vec.normalize()
                    wall_align = max(-1.0, min(1.0, back_vec.dot(wall_dir.to_2d())))
                    front_align = 0.0
                    if room_dir.length > 1e-6 and front_vec.length > 1e-6:
                        front_vec.normalize()
                        front_align = max(-1.0, min(1.0, front_vec.dot(room_dir.to_2d())))
                    wall_penalty = proximity * ((1.0 - wall_align) * 0.75 + (1.0 - front_align) * 0.35)

        fit_penalty = oversize_penalty + center_ratio * 2.5 + wall_penalty
        return True, "ok", fit_penalty

    parent = bpy.data.objects.new(parent_name, None)
    bpy.context.scene.collection.objects.link(parent)

    for o in objs:
        _unlink_from_all_collections(o)
        collection.objects.link(o)
        o.parent = parent

    bpy.context.view_layer.update()

    best_state: Optional[Tuple[float, mathutils.Vector, mathutils.Euler, float]] = None
    best_score = float("inf")
    last_failure_reason = "no_rotation_candidates"
    for candidate_rotation in _rotation_candidates_for_semantic_group(rotation_deg_engine, semantic_group):
        parent.location = (0.0, 0.0, 0.0)
        parent.scale = (1.0, 1.0, 1.0)
        parent.rotation_euler = (0.0, 0.0, math.radians(float(candidate_rotation or 0.0)))
        bpy.context.view_layer.update()
        _fit_parent_once()
        _fit_parent_once()
        placement_ok, placement_reason, placement_score = _placement_metrics(candidate_rotation)
        if not placement_ok:
            last_failure_reason = placement_reason
            continue
        if placement_score < best_score:
            best_score = placement_score
            best_state = (
                float(candidate_rotation),
                parent.location.copy(),
                parent.scale.copy(),
                float(placement_score),
            )

    if best_state is None:
        print(f"[DBG] rejecting unreasonable placement {parent_name}: {last_failure_reason}")
        for o in list(objs):
            try:
                bpy.data.objects.remove(o, do_unlink=True)
            except Exception:
                pass
        try:
            bpy.data.objects.remove(parent, do_unlink=True)
        except Exception:
            pass
        return None

    chosen_rotation, chosen_location, chosen_scale, chosen_score = best_state
    parent.location = chosen_location
    parent.scale = chosen_scale
    parent.rotation_euler = (0.0, 0.0, math.radians(chosen_rotation))
    parent["cgs_placement_score"] = float(chosen_score)
    parent["cgs_placement_confidence"] = "low" if chosen_score > 1.35 else "medium" if chosen_score > 0.7 else "high"
    parent["cgs_placement_rotation_deg"] = float(chosen_rotation)
    bpy.context.view_layer.update()
    return parent


# ============================================================
# Main build
# ============================================================

def _has_room_spec(data: dict) -> bool:
    if not isinstance(data, dict):
        return False
    room = data.get("room")
    if not isinstance(room, dict):
        return False
    poly = room.get("floor_polygon")
    return isinstance(poly, list) and len(poly) >= 3


def build_scene(
    json_path: str,
    draw_aabb: bool,
    force_tint: bool,
    keep_existing_mats: bool,
    verbose: bool,
    env_textures_dir: str,
    style_text: Optional[str],
    seed: int,
    reference_blend: Optional[str] = None,
    overlay_bbox_only: bool = False,
    bbox_fallback_missing_mesh: bool = True,
    highlight_item_ids: Optional[set[str]] = None,
) -> None:
    use_reference_scene = bool(reference_blend)
    if not use_reference_scene:
        reset_scene()
        _ensure_world()

    json_file = Path(json_path).expanduser().resolve()
    json_dir = json_file.parent

    with open(str(json_file), "r", encoding="utf-8") as f:
        data = json.load(f)

    coll_room = ensure_collection("Room")
    coll_items = ensure_collection("Items" if not overlay_bbox_only else "BBoxOverlay")
    if overlay_bbox_only:
        clear_collection_objects(coll_items)

    room_mode = "room_spec"
    room_objs: List[bpy.types.Object] = []

    bb_min = mathutils.Vector((0.0, 0.0, 0.0))
    bb_max = mathutils.Vector((5.0, 5.0, 3.0))

    room_bb_min: Optional[mathutils.Vector] = None
    room_bb_max: Optional[mathutils.Vector] = None

    if "items" in data:
        room_engine = data.get("room") or {}
        items = data["items"] or []
    else:
        room_engine = data.get("room") or data.get("room_spec") or {}
        items = data.get("placements") or data.get("items") or []
    items_by_id: Dict[str, Dict] = {
        str(item.get("id") or "").strip(): item
        for item in items
        if isinstance(item, dict) and str(item.get("id") or "").strip()
    }

    source_name_to_items: Dict[str, List[Dict]] = {}
    for item in items:
        source_name = _blend_source_name_from_item(item)
        if source_name:
            source_name_to_items.setdefault(source_name, []).append(item)
    hidden_reference_sources: set[str] = set()
    item_actual_aabbs: Dict[str, Dict[str, float]] = {}
    item_roots: Dict[str, bpy.types.Object] = {}
    item_issue_reasons: Dict[str, List[str]] = {}

    room_data_for_build = data if _has_room_spec(data) else _room_spec_from_bounds(
        room_engine=room_engine if isinstance(room_engine, dict) else {},
        default_bounds=(0.0, 5.0, 0.0, 5.0, 0.0, 3.0),
    )
    if not _has_room_spec(data):
        room_mode = "bounds_room_spec"
        _log(verbose, "[Room] room-spec missing -> synthesized thin-wall room from bounds")

    if not use_reference_scene:
        room_objs, room_bb_min, room_bb_max = build_room_from_spec(
            room_json=room_data_for_build,
            coll_room=coll_room,
            env_textures_dir=env_textures_dir,
            style_text=style_text,
            seed=seed,
            verbose=verbose,
        )
        bb_min = room_bb_min.copy()
        bb_max = room_bb_max.copy()
        _log(verbose, f"[Room] bounds: min={tuple(bb_min)}, max={tuple(bb_max)}")
    else:
        room = room_data_for_build.get("room") or {}
        poly = room.get("floor_polygon") or []
        pts = [(float(p["x"]), float(p["y"])) for p in poly if isinstance(p, dict) and "x" in p and "y" in p]
        floor_z = float(room.get("floor_z", 0.0))
        ceil_h = float(room.get("ceiling_height", 2.8))
        if pts:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            room_bb_min = mathutils.Vector((min(xs), min(ys), floor_z))
            room_bb_max = mathutils.Vector((max(xs), max(ys), floor_z + ceil_h))
            bb_min = room_bb_min.copy()
            bb_max = room_bb_max.copy()
    if room_bb_min is not None and room_bb_max is not None:
        _store_scene_room_bounds(room_bb_min, room_bb_max)

    z_floor = float(bb_min.z)
    z_ceil = float(bb_max.z)

    def _quantize_rot_0_90_180_270(deg: float) -> float:
        a = float(deg or 0.0) % 360.0
        allowed = (0.0, 90.0, 180.0, 270.0)
        best = min(allowed, key=lambda t: (abs(((a - t + 180.0) % 360.0) - 180.0), t))
        return best

    for it in items:
        aabb_eng = dict(it.get("aabb") or it.get("bbox") or {})
        if not aabb_eng:
            _log(verbose, f"⚠️ item without aabb: {it}")
            continue

        item_id = str(it.get("id") or "").strip()
        name = _item_name(it)
        constraints = it.get("constraints") or {}
        name_l = name.lower()
        meta = it.get("meta") or {}
        supplier_reanchored_support = bool(meta.get("supplier_support_reanchored"))
        force_placeholder_bbox = bool(meta.get("placeholder_bbox"))
        preserve_raw_aabb = bool(overlay_bbox_only or force_placeholder_bbox)
        source = it.get("source") or {}
        source_blend_name = _blend_source_name_from_item(it)
        source_scene_obj = _get_scene_source_object(source_blend_name) if use_reference_scene else None
        preserve_reference_vertical_anchor = bool(
            use_reference_scene
            and meta.get("supplier_binding_applied")
            and source_blend_name
        )
        semantic_group = _item_semantic_group(it)
        mount_mode = _item_mount_mode(it)

        if overlay_bbox_only and use_reference_scene:
            blend_aabb = _aabb_from_blend_object_name(source.get("blend_object_name"))
            if blend_aabb is not None:
                aabb_eng = blend_aabb

        is_ceiling_item = mount_mode == "ceiling"
        is_floor_item = mount_mode == "floor"

        # Клампим в текущую геометрию комнаты.
        if (not preserve_raw_aabb) and room_bb_min is not None and room_bb_max is not None:
            try:
                aabb_eng = clamp_item_aabb_to_room_bounds(aabb_eng, room_bb_min, room_bb_max, margin=0.05)
            except Exception as e:
                _log(verbose, f"[Clamp] failed for {name}: {e}")

        if not preserve_raw_aabb and not preserve_reference_vertical_anchor:
            if is_floor_item:
                if float(aabb_eng.get("z_min", 0.0)) < float(z_floor):
                    dz_fix = float(z_floor) - float(aabb_eng["z_min"])
                    aabb_eng["z_min"] = float(z_floor)
                    aabb_eng["z_max"] = float(aabb_eng["z_max"]) + dz_fix

                sz = float(aabb_eng["z_max"]) - float(aabb_eng["z_min"])
                aabb_eng["z_min"] = float(z_floor)
                aabb_eng["z_max"] = float(z_floor) + sz
            elif is_ceiling_item:
                sz = float(aabb_eng["z_max"]) - float(aabb_eng["z_min"])
                aabb_eng["z_max"] = float(z_ceil)
                aabb_eng["z_min"] = float(z_ceil) - sz

        rot_raw = float(it.get("rotation", it.get("yaw_deg", it.get("rotation_deg", 0.0))) or 0.0)
        rot_deg = _quantize_rot_0_90_180_270(rot_raw)

        fit_mode = _item_mesh_fit_mode(it)

        mesh_path_raw = _item_mesh_path_raw(it)
        mesh_path = _resolve_path_maybe(json_dir, mesh_path_raw)
        supplier_candidate_pool = _item_supplier_candidate_pool(it)

        mesh_tex_dirs_raw = _item_mesh_texture_dirs_raw(it)
        mesh_tex_dirs: List[str] = []
        for d in mesh_tex_dirs_raw:
            rr = _resolve_path_maybe(json_dir, str(d))
            if rr:
                mesh_tex_dirs.append(rr)

        texture_path = _resolve_path_maybe(json_dir, it.get("texture_path"))
        texture_files = [str(x) for x in (it.get("texture_files") or [])]

        texture_scale = float(it.get("texture_scale", 1.0) or 1.0)
        tint_rgb = _supplier_candidate_tint_rgb(it, fallback_name=str(name))

        placed_ok = False
        using_reference_object = False

        if not overlay_bbox_only:
            candidate_specs: List[Tuple[int, dict, str]] = []
            seen_mesh_paths: set[str] = set()
            for rank_idx, candidate in enumerate(supplier_candidate_pool):
                candidate_mesh_path = _resolve_path_maybe(json_dir, _candidate_mesh_path_raw(candidate, it))
                if not candidate_mesh_path or not os.path.isfile(candidate_mesh_path):
                    continue
                candidate_key = str(Path(candidate_mesh_path).resolve())
                if candidate_key in seen_mesh_paths:
                    continue
                seen_mesh_paths.add(candidate_key)
                candidate_specs.append((rank_idx, candidate, candidate_mesh_path))
            if mesh_path and os.path.isfile(mesh_path):
                default_key = str(Path(mesh_path).resolve())
                if default_key not in seen_mesh_paths:
                    candidate_specs.append((-1, {}, mesh_path))
                    seen_mesh_paths.add(default_key)

            if candidate_specs:
                print(f"[DBG] placement name={name}")
                print(f"[DBG] supplier placement candidates for {name}: {len(candidate_specs)}")

                best_parent: Optional[bpy.types.Object] = None
                best_mesh_path: Optional[str] = None
                best_imported_from: Optional[str] = None
                best_candidate: Optional[dict] = None
                best_total_score = float("inf")
                last_error: Optional[str] = None

                for rank_idx, supplier_candidate, candidate_mesh_path in candidate_specs:
                    print(f"[DBG] trying candidate rank={rank_idx + 1 if rank_idx >= 0 else 1} mesh={candidate_mesh_path}")
                    mesh_candidates = _discover_mesh_import_candidates(candidate_mesh_path)
                    print(f"[DBG] mesh_candidates for {name}: {mesh_candidates[:8]}")

                    imported_from: Optional[str] = None
                    objs: List[bpy.types.Object] = []
                    for candidate_path in mesh_candidates:
                        ext = os.path.splitext(candidate_path)[1].lower()
                        if ext not in _SUPPORTED_MESH_EXTS:
                            continue

                        before_names = set(o.name for o in bpy.data.objects)
                        objs, error = _safe_import_supported_mesh(candidate_path)
                        after_names = set(o.name for o in bpy.data.objects)
                        created_names = sorted(after_names - before_names)

                        if objs:
                            imported_from = candidate_path
                            print(f"[DBG] import_supported_mesh returned {len(objs)} objects for {name}")
                            print(f"[DBG] bpy created {len(created_names)} objects for {name}: {created_names[:20]}")
                            break

                        last_error = error
                        print(f"⚠️ {name}: import mesh failed for {candidate_path}: {error}")

                    if not objs:
                        continue

                    parent = place_in_aabb(
                        objs=objs,
                        aabb=aabb_eng,
                        rotation_deg_engine=rot_deg,
                        fit_mode=fit_mode,
                        parent_name=name,
                        collection=coll_items,
                        snap_to_floor=is_floor_item,
                        floor_offset=(-0.001 if is_floor_item else 0.0),
                        semantic_group=semantic_group,
                        snap_to_ceiling=is_ceiling_item,
                        ceiling_offset=0.0,
                    )
                    if parent is None:
                        continue

                    placement_score = float(parent.get("cgs_placement_score") or 999999.0)
                    total_score = placement_score + (max(rank_idx, 0) * 0.08)
                    if total_score < best_total_score:
                        _remove_object_family(best_parent)
                        best_parent = parent
                        best_mesh_path = candidate_mesh_path
                        best_imported_from = imported_from
                        best_candidate = supplier_candidate if isinstance(supplier_candidate, dict) else None
                        best_total_score = total_score
                    else:
                        _remove_object_family(parent)

                if best_parent is not None:
                    parent = best_parent
                    placed_ok = True
                    if item_id:
                        item_roots[item_id] = parent
                        actual_aabb = _aabb_from_object_family_root(parent)
                        if actual_aabb is not None:
                            item_actual_aabbs[item_id] = actual_aabb
                    if item_id and str(parent.get("cgs_placement_confidence") or "") == "low":
                        item_issue_reasons.setdefault(item_id, []).append("low_confidence_replacement")
                    if item_id and best_candidate is not None:
                        primary_key = str(((meta.get("supplier_candidate") or {}) if isinstance(meta.get("supplier_candidate"), dict) else {}).get("unique_key") or "").strip()
                        selected_key = str(best_candidate.get("unique_key") or "").strip()
                        if selected_key and primary_key and selected_key != primary_key:
                            item_issue_reasons.setdefault(item_id, []).append(f"used_alternative_candidate:{selected_key}")

                    if source_scene_obj is not None and source_blend_name not in hidden_reference_sources:
                        shared_items = source_name_to_items.get(source_blend_name) or []
                        can_hide_source = len(shared_items) <= 1 or all(
                            _item_has_existing_mesh_file(shared_it, json_dir)
                            for shared_it in shared_items
                        )
                        if can_hide_source:
                            kept_light_count = _duplicate_light_objects_from_family(source_scene_obj)
                            if kept_light_count:
                                print(f"[DBG] kept {kept_light_count} source lights for {name}: {source_blend_name}")
                            _hide_object_family(source_scene_obj)
                            hidden_reference_sources.add(source_blend_name)

                    effective_mesh_path = best_imported_from or best_mesh_path or mesh_path
                    if effective_mesh_path:
                        if force_tint:
                            mat = _make_pbr_material(
                                name=f"Mat_{name}_Tint",
                                maps=PBRMaps(),
                                tint_rgb=tint_rgb,
                                tex_scale=1.0,
                            )
                            for o in _iter_mesh_children(parent):
                                me = o.data
                                if not me.materials:
                                    me.materials.append(mat)
                                else:
                                    for i in range(len(me.materials)):
                                        me.materials[i] = mat
                        else:
                            _ensure_textures(
                                parent=parent,
                                mesh_path=effective_mesh_path,
                                mesh_texture_dirs=mesh_tex_dirs,
                                texture_path=texture_path,
                                texture_files=texture_files,
                                texture_scale=texture_scale,
                                tint_rgb=tint_rgb,
                                keep_existing_mats=keep_existing_mats,
                                verbose=verbose,
                            )
                else:
                    print(f"⚠️ {name}: Не удалось импортировать модель: {mesh_path}; last_error={last_error}")
                if (not placed_ok) and source_scene_obj is not None and meta.get("supplier_binding_applied"):
                    using_reference_object = True
                    placed_ok = True
                    print(f"[DBG] supplier mesh fallback to reference scene object for {name}: {source_blend_name}")
                    if item_id:
                        item_roots[item_id] = source_scene_obj
                        actual_aabb = _aabb_from_object_family_root(source_scene_obj)
                        if actual_aabb is not None:
                            item_actual_aabbs[item_id] = actual_aabb
            elif source_scene_obj is not None:
                using_reference_object = True
                placed_ok = True
                if supplier_reanchored_support:
                    moved = _move_object_family_to_target_aabb(
                        source_scene_obj,
                        aabb_eng,
                        align_bottom=True,
                    )
                    if not moved:
                        placed_ok = False
                print(f"[DBG] using reference scene object for {name}: {source_blend_name}")
                if placed_ok and item_id:
                    item_roots[item_id] = source_scene_obj
                    actual_aabb = _aabb_from_object_family_root(source_scene_obj)
                    if actual_aabb is not None:
                        item_actual_aabbs[item_id] = actual_aabb
            else:
                print(f"⚠️ {name}: mesh_path не найден в placement/asset или файл отсутствует: {mesh_path_raw}")

        force_bbox = overlay_bbox_only or force_placeholder_bbox
        highlight_bbox = bool(item_id and item_id in (highlight_item_ids or set()))
        want_bbox_fallback = bool(bbox_fallback_missing_mesh and (not placed_ok) and (not using_reference_object))
        skip_placeholder_bbox = _should_skip_placeholder_bbox(it)
        if highlight_bbox:
            _make_renderable_bbox_box(aabb_eng, name, coll_items)
        if (not placed_ok) and (draw_aabb or force_bbox or want_bbox_fallback) and (not skip_placeholder_bbox):
            _add_aabb_box(aabb_eng, name, coll_items)
            if force_bbox:
                _add_aabb_label(aabb_eng, name, coll_items)

    support_solver = MLSupportSolver(room_floor_z=z_floor)
    support_items: List[Tuple[float, str, Dict]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("id") or "").strip()
        meta = item.get("meta") or {}
        anchor_id = str(meta.get("supplier_support_anchor_target_id") or "").strip()
        if not item_id or not meta.get("supplier_support_reanchored") or not anchor_id:
            continue
        if item_id not in item_roots or anchor_id not in item_roots:
            continue
        item_aabb = item_actual_aabbs.get(item_id)
        if item_aabb is None:
            item_aabb = _aabb_from_object_family_root(item_roots[item_id])
        if item_aabb is None:
            continue
        footprint = max(
            1e-6,
            (float(item_aabb["x_max"]) - float(item_aabb["x_min"]))
            * (float(item_aabb["y_max"]) - float(item_aabb["y_min"])),
        )
        support_items.append((-footprint, item_id, item))

    occupied_support_aabbs: Dict[str, List[Dict[str, float]]] = {}
    for _, item_id, item in sorted(support_items):
        meta = item.get("meta") or {}
        anchor_id = str(meta.get("supplier_support_anchor_target_id") or "").strip()
        support_mode = str(meta.get("supplier_support_mode") or "top").strip().lower()
        item_root = item_roots.get(item_id)
        anchor_root = item_roots.get(anchor_id)
        if item_root is None or anchor_root is None or item_root == anchor_root:
            continue

        item_aabb = item_actual_aabbs.get(item_id) or _aabb_from_object_family_root(item_root)
        anchor_aabb = item_actual_aabbs.get(anchor_id) or _aabb_from_object_family_root(anchor_root)
        if item_aabb is None or anchor_aabb is None:
            continue

        planes = _extract_support_planes_from_object_family(anchor_root)
        if not planes:
            anchor_item = items_by_id.get(anchor_id) or {}
            planes = _infer_support_planes_from_anchor_item(anchor_item, anchor_aabb)
        if not planes:
            continue

        solved_aabb = support_solver.solve(
            item_aabb=item_aabb,
            anchor_aabb=anchor_aabb,
            planes=planes,
            occupied_aabbs=occupied_support_aabbs.get(anchor_id, []),
            mode=support_mode,
        )
        if solved_aabb is None:
            continue

        moved_aabb = _move_object_family_to_exact_aabb(item_root, item_aabb, solved_aabb)
        if moved_aabb is None:
            continue
        item_actual_aabbs[item_id] = moved_aabb
        occupied_support_aabbs.setdefault(anchor_id, []).append(moved_aabb)

    def _aabb_overlap_3d(a: Dict[str, float], b: Dict[str, float], margin: float = 0.01) -> bool:
        return (
            float(a["x_max"]) > float(b["x_min"]) + margin
            and float(a["x_min"]) < float(b["x_max"]) - margin
            and float(a["y_max"]) > float(b["y_min"]) + margin
            and float(a["y_min"]) < float(b["y_max"]) - margin
            and float(a["z_max"]) > float(b["z_min"]) + margin
            and float(a["z_min"]) < float(b["z_max"]) - margin
        )

    diagnostic_ids: List[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("id") or "").strip()
        meta = item.get("meta") or {}
        if not item_id:
            continue
        if meta.get("supplier_binding_applied") or meta.get("supplier_support_reanchored"):
            diagnostic_ids.append(item_id)

    for item_id in diagnostic_ids:
        aabb = item_actual_aabbs.get(item_id)
        if aabb is None:
            continue
        if room_bb_min is not None and room_bb_max is not None:
            margin = 0.02
            if (
                float(aabb["x_min"]) < float(room_bb_min.x) - margin
                or float(aabb["x_max"]) > float(room_bb_max.x) + margin
                or float(aabb["y_min"]) < float(room_bb_min.y) - margin
                or float(aabb["y_max"]) > float(room_bb_max.y) + margin
                or float(aabb["z_min"]) < float(room_bb_min.z) - margin
                or float(aabb["z_max"]) > float(room_bb_max.z) + margin
            ):
                item_issue_reasons.setdefault(item_id, []).append("out_of_bounds")

    for idx, item_id_a in enumerate(diagnostic_ids):
        aabb_a = item_actual_aabbs.get(item_id_a)
        if aabb_a is None:
            continue
        meta_a = (items_by_id.get(item_id_a) or {}).get("meta") or {}
        anchor_a = str(meta_a.get("supplier_support_anchor_target_id") or "").strip()
        for item_id_b in diagnostic_ids[idx + 1 :]:
            aabb_b = item_actual_aabbs.get(item_id_b)
            if aabb_b is None:
                continue
            meta_b = (items_by_id.get(item_id_b) or {}).get("meta") or {}
            anchor_b = str(meta_b.get("supplier_support_anchor_target_id") or "").strip()
            if anchor_a and anchor_a == item_id_b:
                continue
            if anchor_b and anchor_b == item_id_a:
                continue
            if _aabb_overlap_3d(aabb_a, aabb_b, margin=0.012):
                item_issue_reasons.setdefault(item_id_a, []).append(f"collision:{item_id_b}")
                item_issue_reasons.setdefault(item_id_b, []).append(f"collision:{item_id_a}")

    for item_id, reasons in item_issue_reasons.items():
        if not reasons:
            continue
        diagnostic_reasons = [reason for reason in sorted(set(reasons)) if not reason.startswith("used_alternative_candidate:")]
        if not diagnostic_reasons:
            continue
        aabb = item_actual_aabbs.get(item_id)
        if aabb is None:
            continue
        _make_renderable_bbox_box(
            aabb,
            f"INVALID_{item_id}",
            coll_items,
            rgba=(0.85, 0.05, 0.05, 1.0),
            thickness=0.03,
        )
        print(f"[DBG] invalid placement {item_id}: {diagnostic_reasons}")

    _log(verbose, f"[Room] final room mode = {room_mode}")

    if not use_reference_scene:
        _frame_camera_on_bounds(bb_min, bb_max)
        _add_basic_lights(bb_min, bb_max)
        _force_material_preview_if_ui()

# ============================================================
# RUN
# ============================================================

def main() -> None:
    args = _parse_argv(sys.argv)

    json_path = str(Path(args.json).expanduser().resolve())

    keep_existing = not (args.rebuild_materials or args.no_keep_existing_mats)

    build_scene(
        json_path=json_path,
        draw_aabb=bool(args.draw_aabb),
        force_tint=bool(args.force_tint),
        keep_existing_mats=keep_existing,
        verbose=bool(args.verbose),
        env_textures_dir=str(Path(args.env_textures_dir).expanduser().resolve()),
        style_text=args.style_text,
        seed=int(args.seed or 0),
        reference_blend=args.reference_blend,
        overlay_bbox_only=bool(args.overlay_bbox_only),
        bbox_fallback_missing_mesh=not bool(args.no_bbox_fallback),
        highlight_item_ids=_parse_id_set(args.highlight_item_ids),
    )

    if args.hide_room_shell:
        _hide_room_shell_objects()

    if not args.no_pack_assets:
        _pack_assets_best_effort()

    if not args.draw_aabb and not _parse_id_set(args.highlight_item_ids):
        _set_overlay_helpers_render_visibility(False)

    if args.save_blend:
        out_blend = str(Path(args.save_blend).expanduser().resolve())
        os.makedirs(os.path.dirname(out_blend), exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=out_blend)

    if args.render:
        out_png = str(Path(args.render).expanduser().resolve())
        os.makedirs(os.path.dirname(out_png), exist_ok=True)

        scene = bpy.context.scene
        _configure_fast_render(scene)
        scene.render.image_settings.file_format = "PNG"
        scene.render.filepath = out_png

        try:
            scene.render.film_transparent = False
        except Exception:
            pass

        bpy.ops.render.render(write_still=True)

    if args.turntable_render_dir:
        scene = bpy.context.scene
        _configure_turntable_render(scene)
        scene.render.image_settings.file_format = "PNG"
        try:
            scene.render.film_transparent = False
        except Exception:
            pass
        room_bounds = _scene_room_bounds()
        bb_min, bb_max = _visible_mesh_bounds(
            mathutils.Vector((0.0, 0.0, 0.0)),
            mathutils.Vector((5.0, 5.0, 3.0)),
        )
        _render_turntable_sequence(
            Path(args.turntable_render_dir).expanduser().resolve(),
            int(args.turntable_frames or 24),
            bb_min,
            bb_max,
            elevation_deg=float(args.turntable_elevation_deg or 0.0),
            room_bb_min=(room_bounds[0] if room_bounds else None),
            room_bb_max=(room_bounds[1] if room_bounds else None),
        )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
