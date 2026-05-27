#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export scene.v1 / placement.v1 layout distributions.

The script is intentionally append-safe: every run writes to a new timestamped
directory unless --run-name points to a new directory name.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


csv.field_size_limit(1024 * 1024 * 1024)

EPS = 1e-9
DEFAULT_GRID_SIZES = [5, 10, 20, 40]
DEFAULT_OUT_ROOT = Path("out/layout_distribution_analysis")


OBJECT_FIELDS = [
    "run_id",
    "run_dir",
    "artifact_variant",
    "source_file",
    "source_kind",
    "scene_file",
    "placement_file",
    "dataset_role",
    "creator_family",
    "creator_variant",
    "creator_raw",
    "creator_source_field",
    "room_id",
    "room_type",
    "room_width_m",
    "room_depth_m",
    "room_area_m2",
    "room_x_min",
    "room_x_max",
    "room_y_min",
    "room_y_max",
    "object_index",
    "object_id",
    "name",
    "category",
    "class_name",
    "center_x_m",
    "center_y_m",
    "center_z_m",
    "size_x_m",
    "size_y_m",
    "size_z_m",
    "yaw_deg",
    "yaw_rad",
    "rotation_deg",
    "aabb_x_min",
    "aabb_x_max",
    "aabb_y_min",
    "aabb_y_max",
    "aabb_z_min",
    "aabb_z_max",
    "x_norm_bbox",
    "y_norm_bbox",
    "x_norm",
    "y_norm",
    "inside_room_bbox",
    "inside_floor_polygon",
    "distance_to_nearest_wall_m",
    "distance_to_nearest_corner_m",
    "is_near_wall",
    "is_near_corner",
    "is_center_zone",
    "has_valid_aabb",
    "is_small_object",
    "is_trackable_for_distribution",
    "asset_model_id",
    "asset_mesh_path",
    "raw_source_json",
    "raw_meta_json",
]


ROOM_FIELDS = [
    "run_id",
    "run_dir",
    "artifact_variant",
    "source_file",
    "scene_file",
    "placement_file",
    "dataset_role",
    "creator_family",
    "creator_variant",
    "creator_raw",
    "room_id",
    "room_type",
    "room_width_m",
    "room_depth_m",
    "room_area_m2",
    "room_x_min",
    "room_x_max",
    "room_y_min",
    "room_y_max",
    "floor_polygon_json",
    "n_placements",
    "n_trackable_placements",
]


RUN_FIELDS = [
    "run_id",
    "run_dir",
    "artifact_variant",
    "scene_file",
    "placement_file",
    "dataset_role",
    "creator_family",
    "creator_variant",
    "creator_raw",
    "creator_source_field",
    "room_id",
    "room_type",
    "n_scene_placements",
    "n_placement_placements",
    "n_exported_objects",
    "status",
    "error",
]


HIST_FIELDS = [
    "grouping",
    "dataset_role",
    "creator_family",
    "creator_variant",
    "room_type",
    "class_name",
    "grid_size",
    "cell_x",
    "cell_y",
    "count",
    "probability",
    "n_objects",
    "n_rooms",
]


METRIC_FIELDS = [
    "grouping",
    "compare_level",
    "grid_size",
    "segment",
    "left_group",
    "right_group",
    "left_n_objects",
    "right_n_objects",
    "left_n_rooms",
    "right_n_rooms",
    "hist_l1_mean",
    "hist_l2_mean",
    "jensen_shannon",
    "cosine_similarity",
    "coverage_left",
    "coverage_right",
    "coverage_intersection",
]


FEATURE_SUMMARY_FIELDS = [
    "grouping",
    "dataset_role",
    "creator_family",
    "creator_variant",
    "room_type",
    "class_name",
    "n_objects",
    "n_rooms",
    "mean_x_norm",
    "mean_y_norm",
    "mean_distance_to_wall_m",
    "mean_distance_to_corner_m",
    "near_wall_rate",
    "near_corner_rate",
    "center_zone_rate",
    "small_object_rate",
]


ROOM_COUNT_FIELDS = [
    "grouping",
    "dataset_role",
    "creator_family",
    "creator_variant",
    "room_type",
    "n_rooms",
    "mean_objects_per_room",
    "median_objects_per_room",
    "min_objects_per_room",
    "max_objects_per_room",
    "mean_trackable_objects_per_room",
]


def as_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def as_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return ""


def abs_path(path: Path | None) -> str:
    if path is None:
        return ""
    return str(path.expanduser().resolve())


def artifact_variant_from_name(name: str, prefix: str) -> str | None:
    if name in {f"{prefix}.v1.json", f"{prefix}_v1.json"}:
        return "v1"
    if not name.endswith(".json"):
        return None
    if name.startswith(f"{prefix}_"):
        variant = name[len(prefix) + 1:-5]
    elif name.startswith(f"{prefix}."):
        variant = name[len(prefix) + 1:-5]
    else:
        return None
    lowered = variant.lower()
    if any(skip in lowered for skip in ("build_report", "gif_job", "appraisal")):
        return None
    return variant or None


def infer_creator_from_schema(payload: dict[str, Any] | None) -> tuple[str, str]:
    if not isinstance(payload, dict):
        return "", ""
    schema = as_str(payload.get("schema"))
    if schema:
        lowered = schema.lower()
        if lowered.startswith("scene_gt"):
            return "3dfront_processed_gt", "scene.schema"
    room = payload.get("room") if isinstance(payload.get("room"), dict) else {}
    meta = room.get("meta") if isinstance(room.get("meta"), dict) else {}
    source_subset = as_str(meta.get("source_subset"))
    source_dir = as_str(meta.get("source_dir"))
    if "3dfront" in source_subset.lower() or "3dfront" in source_dir.lower() or "3dfront" in schema.lower():
        return "3dfront_processed_gt", "room.meta"
    return "", ""


def parse_points(raw: Any) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    if not isinstance(raw, list):
        return points
    for item in raw:
        if isinstance(item, dict):
            x = as_float(item.get("x"))
            y = as_float(item.get("y", item.get("z")))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            x = as_float(item[0])
            y = as_float(item[1])
        else:
            continue
        if x is not None and y is not None:
            points.append((x, y))
    return points


def polygon_bounds(poly: list[tuple[float, float]]) -> tuple[float, float, float, float] | None:
    if not poly:
        return None
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), max(xs), min(ys), max(ys)


def point_in_polygon(x: float, y: float, poly: list[tuple[float, float]]) -> bool | None:
    if len(poly) < 3:
        return None
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        dy = yj - yi
        intersects = False
        if abs(dy) > EPS and ((yi > y) != (yj > y)):
            intersects = x < (xj - xi) * (y - yi) / dy + xi
        if intersects:
            inside = not inside
        j = i
    return inside


def point_segment_distance(x: float, y: float, a: tuple[float, float], b: tuple[float, float]) -> float:
    ax, ay = a
    bx, by = b
    vx = bx - ax
    vy = by - ay
    wx = x - ax
    wy = y - ay
    denom = vx * vx + vy * vy
    if denom <= EPS:
        return math.hypot(x - ax, y - ay)
    t = max(0.0, min(1.0, (wx * vx + wy * vy) / denom))
    px = ax + t * vx
    py = ay + t * vy
    return math.hypot(x - px, y - py)


def nearest_wall_distance(x: float, y: float, poly: list[tuple[float, float]]) -> float | None:
    if len(poly) < 2:
        return None
    distances = []
    for i, p0 in enumerate(poly):
        p1 = poly[(i + 1) % len(poly)]
        distances.append(point_segment_distance(x, y, p0, p1))
    return min(distances) if distances else None


def nearest_corner_distance(x: float, y: float, poly: list[tuple[float, float]]) -> float | None:
    if not poly:
        return None
    return min(math.hypot(x - px, y - py) for px, py in poly)


def room_info(scene: dict[str, Any] | None) -> dict[str, Any]:
    room = scene.get("room") if isinstance(scene, dict) else None
    if not isinstance(room, dict):
        return {
            "room_id": "",
            "room_type": "",
            "width": None,
            "depth": None,
            "area": None,
            "x_min": None,
            "x_max": None,
            "y_min": None,
            "y_max": None,
            "polygon": [],
        }

    poly = parse_points(room.get("floor_polygon")) or parse_points(room.get("floor_polygon_xz"))
    bounds = polygon_bounds(poly)

    x_min = as_float(room.get("x_min"))
    x_max = as_float(room.get("x_max"))
    y_min = as_float(room.get("y_min"))
    y_max = as_float(room.get("y_max"))
    if None in (x_min, x_max, y_min, y_max) and bounds is not None:
        x_min, x_max, y_min, y_max = bounds

    width = as_float(room.get("width_m", room.get("width")))
    depth = as_float(room.get("depth_m", room.get("depth")))
    if width is None and x_min is not None and x_max is not None:
        width = x_max - x_min
    if depth is None and y_min is not None and y_max is not None:
        depth = y_max - y_min
    if x_min is None and width is not None:
        x_min, x_max = 0.0, width
    if y_min is None and depth is not None:
        y_min, y_max = 0.0, depth
    area = as_float(room.get("area_m2"))
    if area is None and width is not None and depth is not None:
        area = width * depth

    return {
        "room_id": as_str(room.get("id") or room.get("name")),
        "room_type": as_str(room.get("room_type") or room.get("type") or room.get("type_hint")),
        "width": width,
        "depth": depth,
        "area": area,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "polygon": poly,
    }


def placements_from_payload(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    raw = payload.get("placements")
    if not isinstance(raw, list):
        raw = payload.get("items")
    return [x for x in raw if isinstance(x, dict)] if isinstance(raw, list) else []


def center_from_aabb(aabb: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    x0 = as_float(aabb.get("x_min"))
    x1 = as_float(aabb.get("x_max"))
    y0 = as_float(aabb.get("y_min"))
    y1 = as_float(aabb.get("y_max"))
    z0 = as_float(aabb.get("z_min"))
    z1 = as_float(aabb.get("z_max"))
    return (
        None if x0 is None or x1 is None else 0.5 * (x0 + x1),
        None if y0 is None or y1 is None else 0.5 * (y0 + y1),
        None if z0 is None or z1 is None else 0.5 * (z0 + z1),
    )


def infer_class_name(obj: dict[str, Any]) -> str:
    source = obj.get("source") if isinstance(obj.get("source"), dict) else {}
    for key in ("server_class_name", "class_name", "semantic_class"):
        value = source.get(key) or obj.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    text = f"{obj.get('name','')} {obj.get('category','')}".lower()
    rules = [
        ("double_bed", ["double bed"]),
        ("bed", ["bed", "blanket", "pillow"]),
        ("nightstand", ["nightstand", "bedside", "side table"]),
        ("wardrobe", ["wardrobe", "closet"]),
        ("cabinet", ["cabinet", "drawer", "shelf", "storage"]),
        ("sofa", ["sofa", "couch"]),
        ("chair", ["chair", "armchair", "stool"]),
        ("table", ["table", "desk"]),
        ("lamp", ["lamp", "light"]),
        ("sink", ["sink"]),
        ("toilet", ["toilet"]),
        ("bath", ["bath", "tub"]),
        ("refrigerator", ["refrigerator", "fridge"]),
        ("stove", ["stove", "oven", "cooktop"]),
        ("tv", ["television", "tv"]),
        ("decor", ["book", "vase", "plant", "picture", "mirror", "rug"]),
    ]
    for label, needles in rules:
        if any(n in text for n in needles):
            return label
    category = as_str(obj.get("category") or obj.get("name"), "unknown").strip()
    return re.sub(r"[^a-zA-Z0-9_]+", "_", category.lower()).strip("_") or "unknown"


def raw_creator_from_obj(obj: dict[str, Any]) -> tuple[str, str]:
    source = obj.get("source") if isinstance(obj.get("source"), dict) else {}
    meta = obj.get("meta") if isinstance(obj.get("meta"), dict) else {}
    meta_source = meta.get("source") if isinstance(meta.get("source"), dict) else {}
    for field, value in [
        ("object.source.placement_source", source.get("placement_source")),
        ("object.source.generator", source.get("generator")),
        ("object.meta.source.placement_source", meta_source.get("placement_source")),
        ("object.meta.placement_source", meta.get("placement_source")),
    ]:
        if isinstance(value, str) and value.strip():
            return value.strip(), field
    return "", ""


def normalize_creator(raw: str, path_text: str = "") -> tuple[str, str, str]:
    text = f"{raw} {path_text}".lower()
    if "procedural_room_stage" in text or "procedural_room" in text:
        return "procedural", "procedural_room_stage", raw
    if "procedural" in text:
        return "procedural", "procedural", raw
    if "3dfront" in text or "3d-front" in text or "scene_gt" in text:
        return "3dfront", "3dfront_processed_gt", raw
    if "infinigen" in text:
        return "infinigen", "infinigen_clean" if "clean" in text else "infinigen", raw
    if "m3dlayout" in text:
        if "diffusion" in text:
            return "m3dlayout", "m3dlayout_diffusion", raw
        if "_ar" in text or "autoregressive" in text:
            return "m3dlayout", "m3dlayout_ar", raw
        return "m3dlayout", "m3dlayout", raw
    if "diffuscene" in text:
        return "diffuscene", "diffuscene_remote" if "remote" in text else "diffuscene", raw
    if "ollama" in text or "llm" in text:
        return "ollama_llm", "ollama_llm", raw
    if "retrieval" in text or "knn" in text:
        return "retrieval", "retrieval_knn_scene", raw
    if "relaxed" in text:
        return "relaxed", "relaxed", raw
    if "cube" in text:
        return "cube", "cube", raw
    if "random" in text:
        return "random", "random", raw
    return "unknown", raw or "unknown", raw


def dataset_role_for(creator_family: str, path_text: str, role_rules: list[tuple[str, str]]) -> str:
    path_l = path_text.lower()
    for pattern, role in role_rules:
        if pattern.lower() in path_l:
            return role
    if "real" in path_l or "project" in path_l or "kvartirografiya" in path_l:
        return "real_project"
    if creator_family == "3dfront":
        return "gt_dataset"
    if creator_family != "unknown":
        return "generated"
    return "unknown"


def top_level_creator(scene: dict[str, Any] | None, placement: dict[str, Any] | None, run_dir: Path) -> tuple[str, str]:
    schema_creator, schema_field = infer_creator_from_schema(scene)
    if schema_creator:
        return schema_creator, schema_field
    if isinstance(placement, dict):
        for key in ("placer", "mode", "creator", "generator"):
            value = placement.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip(), f"placement.{key}"
    if isinstance(scene, dict):
        meta = scene.get("meta") if isinstance(scene.get("meta"), dict) else {}
        for key in ("placer", "mode", "creator", "generator"):
            value = meta.get(key) or scene.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip(), f"scene.{key}"
    return "", "path"


def discover_runs(roots: list[Path]) -> list[dict[str, Any]]:
    runs: dict[tuple[Path, str], dict[str, Any]] = {}
    for root in roots:
        for json_path in root.rglob("*.json"):
            name = json_path.name
            scene_variant = artifact_variant_from_name(name, "scene")
            if scene_variant is not None:
                rec = runs.setdefault((json_path.parent, scene_variant), {"run_dir": json_path.parent, "variant": scene_variant})
                rec["scene"] = json_path
                continue
            placement_variant = artifact_variant_from_name(name, "placement")
            if placement_variant is not None:
                rec = runs.setdefault((json_path.parent, placement_variant), {"run_dir": json_path.parent, "variant": placement_variant})
                rec["placement"] = json_path
    return [
        rec for _, rec in sorted(
            runs.items(),
            key=lambda kv: (str(kv[0][0]), kv[0][1]),
        )
    ]


def bool_int(value: bool | None) -> int:
    return 1 if value is True else 0


def object_row(
    *,
    run_id: str,
    run_dir: Path,
    artifact_variant: str,
    source_file: Path,
    source_kind: str,
    scene_file: Path | None,
    placement_file: Path | None,
    dataset_role: str,
    creator_family: str,
    creator_variant: str,
    creator_raw: str,
    creator_source_field: str,
    info: dict[str, Any],
    obj: dict[str, Any],
    object_index: int,
) -> dict[str, Any]:
    aabb = obj.get("aabb") if isinstance(obj.get("aabb"), dict) else {}
    pos = obj.get("position_m") if isinstance(obj.get("position_m"), list) else None
    size = obj.get("size_m") if isinstance(obj.get("size_m"), list) else None
    cx = as_float(pos[0]) if pos and len(pos) >= 1 else None
    cy = as_float(pos[1]) if pos and len(pos) >= 2 else None
    cz = as_float(pos[2]) if pos and len(pos) >= 3 else None
    if cx is None or cy is None or cz is None:
        ax, ay, az = center_from_aabb(aabb)
        cx = cx if cx is not None else ax
        cy = cy if cy is not None else ay
        cz = cz if cz is not None else az

    x_min = info["x_min"]
    x_max = info["x_max"]
    y_min = info["y_min"]
    y_max = info["y_max"]
    width = info["width"]
    depth = info["depth"]
    x_norm = None
    y_norm = None
    inside_bbox = None
    if None not in (cx, cy, x_min, y_min, width, depth) and width and depth:
        x_norm = (float(cx) - float(x_min)) / max(float(width), EPS)
        y_norm = (float(cy) - float(y_min)) / max(float(depth), EPS)
        inside_bbox = 0.0 <= x_norm <= 1.0 and 0.0 <= y_norm <= 1.0

    poly = info["polygon"]
    inside_poly = point_in_polygon(float(cx), float(cy), poly) if cx is not None and cy is not None else None
    wall_dist = nearest_wall_distance(float(cx), float(cy), poly) if cx is not None and cy is not None else None
    corner_dist = nearest_corner_distance(float(cx), float(cy), poly) if cx is not None and cy is not None else None
    near_wall_threshold = 0.35
    near_corner_threshold = 0.55
    is_center = False
    if x_norm is not None and y_norm is not None:
        is_center = 0.35 <= x_norm <= 0.65 and 0.35 <= y_norm <= 0.65

    sx = as_float(size[0]) if size and len(size) >= 1 else None
    sy = as_float(size[1]) if size and len(size) >= 2 else None
    sz = as_float(size[2]) if size and len(size) >= 3 else None
    ax0 = as_float(aabb.get("x_min"))
    ax1 = as_float(aabb.get("x_max"))
    ay0 = as_float(aabb.get("y_min"))
    ay1 = as_float(aabb.get("y_max"))
    az0 = as_float(aabb.get("z_min"))
    az1 = as_float(aabb.get("z_max"))
    valid_aabb = None not in (ax0, ax1, ay0, ay1, az0, az1) and ax1 >= ax0 and ay1 >= ay0 and az1 >= az0
    footprint = None
    if sx is not None and sy is not None:
        footprint = abs(sx * sy)
    elif valid_aabb:
        footprint = abs((ax1 - ax0) * (ay1 - ay0))
    is_small = footprint is not None and footprint < 0.05

    asset = obj.get("asset") if isinstance(obj.get("asset"), dict) else {}
    source = obj.get("source") if isinstance(obj.get("source"), dict) else {}
    meta = obj.get("meta") if isinstance(obj.get("meta"), dict) else {}
    class_name = infer_class_name(obj)
    trackable = (
        x_norm is not None
        and y_norm is not None
        and 0.0 <= x_norm < 1.0
        and 0.0 <= y_norm < 1.0
        and class_name not in {"unknown"}
    )

    return {
        "run_id": run_id,
        "run_dir": abs_path(run_dir),
        "artifact_variant": artifact_variant,
        "source_file": abs_path(source_file),
        "source_kind": source_kind,
        "scene_file": abs_path(scene_file),
        "placement_file": abs_path(placement_file),
        "dataset_role": dataset_role,
        "creator_family": creator_family,
        "creator_variant": creator_variant,
        "creator_raw": creator_raw,
        "creator_source_field": creator_source_field,
        "room_id": info["room_id"],
        "room_type": info["room_type"],
        "room_width_m": width,
        "room_depth_m": depth,
        "room_area_m2": info["area"],
        "room_x_min": x_min,
        "room_x_max": x_max,
        "room_y_min": y_min,
        "room_y_max": y_max,
        "object_index": object_index,
        "object_id": as_str(obj.get("id"), f"obj_{object_index:04d}"),
        "name": as_str(obj.get("name")),
        "category": as_str(obj.get("category")),
        "class_name": class_name,
        "center_x_m": cx,
        "center_y_m": cy,
        "center_z_m": cz,
        "size_x_m": sx,
        "size_y_m": sy,
        "size_z_m": sz,
        "yaw_deg": as_float(obj.get("yaw_deg")),
        "yaw_rad": as_float(obj.get("yaw_rad")),
        "rotation_deg": as_float(obj.get("rotation_deg")),
        "aabb_x_min": ax0,
        "aabb_x_max": ax1,
        "aabb_y_min": ay0,
        "aabb_y_max": ay1,
        "aabb_z_min": az0,
        "aabb_z_max": az1,
        "x_norm_bbox": x_norm,
        "y_norm_bbox": y_norm,
        "x_norm": x_norm,
        "y_norm": y_norm,
        "inside_room_bbox": bool_int(inside_bbox),
        "inside_floor_polygon": bool_int(inside_poly),
        "distance_to_nearest_wall_m": wall_dist,
        "distance_to_nearest_corner_m": corner_dist,
        "is_near_wall": bool_int(wall_dist is not None and wall_dist <= near_wall_threshold),
        "is_near_corner": bool_int(corner_dist is not None and corner_dist <= near_corner_threshold),
        "is_center_zone": bool_int(is_center),
        "has_valid_aabb": bool_int(valid_aabb),
        "is_small_object": bool_int(is_small),
        "is_trackable_for_distribution": bool_int(trackable),
        "asset_model_id": as_str(asset.get("model_id")),
        "asset_mesh_path": as_str(asset.get("mesh_path")),
        "raw_source_json": json_dumps(source),
        "raw_meta_json": json_dumps(meta),
    }


def build_rows(
    runs: list[dict[str, Any]],
    role_rules: list[tuple[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    run_rows: list[dict[str, Any]] = []
    room_rows: list[dict[str, Any]] = []
    object_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for idx, files in enumerate(runs, start=1):
        run_id = f"run_{idx:06d}"
        run_dir = Path(files["run_dir"])
        artifact_variant = as_str(files.get("variant"), "unknown")
        scene_file = files.get("scene")
        placement_file = files.get("placement")
        scene = None
        placement = None
        status = "ok"
        error = ""
        try:
            if scene_file is not None:
                scene = load_json(scene_file)
            if placement_file is not None:
                placement = load_json(placement_file)
            info = room_info(scene)
            raw_creator, creator_field = top_level_creator(scene, placement, run_dir)
            family, variant, raw_norm = normalize_creator(raw_creator, f"{run_dir} {artifact_variant}")
            role = dataset_role_for(family, str(run_dir), role_rules)

            scene_placements = placements_from_payload(scene)
            placement_placements = placements_from_payload(placement)
            source_placements = scene_placements if scene_placements else placement_placements
            source_file = scene_file if scene_placements and scene_file is not None else placement_file
            source_kind = "scene" if scene_placements else "placement"
            exported = []
            for obj_index, obj in enumerate(source_placements):
                obj_raw_creator, obj_creator_field = raw_creator_from_obj(obj)
                obj_family, obj_variant, obj_raw_norm = normalize_creator(obj_raw_creator or raw_norm, f"{run_dir} {artifact_variant}")
                row = object_row(
                    run_id=run_id,
                    run_dir=run_dir,
                    artifact_variant=artifact_variant,
                    source_file=source_file or run_dir,
                    source_kind=source_kind,
                    scene_file=scene_file,
                    placement_file=placement_file,
                    dataset_role=role,
                    creator_family=obj_family if obj_family != "unknown" else family,
                    creator_variant=obj_variant if obj_family != "unknown" else variant,
                    creator_raw=obj_raw_norm or raw_norm,
                    creator_source_field=obj_creator_field or creator_field,
                    info=info,
                    obj=obj,
                    object_index=obj_index,
                )
                exported.append(row)
            object_rows.extend(exported)
            n_trackable = sum(int(r["is_trackable_for_distribution"]) for r in exported)

            room_rows.append({
                "run_id": run_id,
                "run_dir": abs_path(run_dir),
                "artifact_variant": artifact_variant,
                "source_file": abs_path(scene_file or placement_file),
                "scene_file": abs_path(scene_file),
                "placement_file": abs_path(placement_file),
                "dataset_role": role,
                "creator_family": family,
                "creator_variant": variant,
                "creator_raw": raw_norm,
                "room_id": info["room_id"],
                "room_type": info["room_type"],
                "room_width_m": info["width"],
                "room_depth_m": info["depth"],
                "room_area_m2": info["area"],
                "room_x_min": info["x_min"],
                "room_x_max": info["x_max"],
                "room_y_min": info["y_min"],
                "room_y_max": info["y_max"],
                "floor_polygon_json": json_dumps(info["polygon"]),
                "n_placements": len(source_placements),
                "n_trackable_placements": n_trackable,
            })
        except Exception as exc:
            status = "error"
            error = repr(exc)
            failures.append({
                "run_id": run_id,
                "run_dir": abs_path(run_dir),
                "artifact_variant": artifact_variant,
                "scene_file": abs_path(scene_file),
                "placement_file": abs_path(placement_file),
                "error": error,
            })
        run_rows.append({
            "run_id": run_id,
            "run_dir": abs_path(run_dir),
            "artifact_variant": artifact_variant,
            "scene_file": abs_path(scene_file),
            "placement_file": abs_path(placement_file),
            "dataset_role": room_rows[-1]["dataset_role"] if status == "ok" and room_rows else "",
            "creator_family": room_rows[-1]["creator_family"] if status == "ok" and room_rows else "",
            "creator_variant": room_rows[-1]["creator_variant"] if status == "ok" and room_rows else "",
            "creator_raw": room_rows[-1]["creator_raw"] if status == "ok" and room_rows else "",
            "creator_source_field": creator_field if status == "ok" else "",
            "room_id": room_rows[-1]["room_id"] if status == "ok" and room_rows else "",
            "room_type": room_rows[-1]["room_type"] if status == "ok" and room_rows else "",
            "n_scene_placements": len(placements_from_payload(scene)) if isinstance(scene, dict) else 0,
            "n_placement_placements": len(placements_from_payload(placement)) if isinstance(placement, dict) else 0,
            "n_exported_objects": len(exported) if status == "ok" else 0,
            "status": status,
            "error": error,
        })
    return run_rows, room_rows, object_rows, failures


def clean_segment(value: str) -> str:
    return value if value else "__all__"


def group_key(row: dict[str, Any], grouping: str) -> tuple[str, str, str, str, str]:
    room_type = clean_segment(as_str(row.get("room_type"))) if "room_type" in grouping else "__all__"
    class_name = clean_segment(as_str(row.get("class_name"))) if "class" in grouping else "__all__"
    return (
        grouping,
        clean_segment(as_str(row.get("dataset_role"))),
        clean_segment(as_str(row.get("creator_family"))),
        clean_segment(as_str(row.get("creator_variant"))),
        room_type,
        class_name,
    )


def histogram_rows(
    objects: list[dict[str, Any]],
    grid_sizes: list[int],
    min_objects: int,
) -> list[dict[str, Any]]:
    trackable = [r for r in objects if int(r.get("is_trackable_for_distribution") or 0) == 1]
    groupings = ["overall", "by_room_type", "by_class", "by_room_type_class"]
    out: list[dict[str, Any]] = []
    for grid in grid_sizes:
        for grouping in groupings:
            buckets: dict[tuple[str, str, str, str, str, str], Counter[tuple[int, int]]] = defaultdict(Counter)
            rooms: dict[tuple[str, str, str, str, str, str], set[str]] = defaultdict(set)
            for row in trackable:
                x = as_float(row.get("x_norm"))
                y = as_float(row.get("y_norm"))
                if x is None or y is None or not (0.0 <= x < 1.0 and 0.0 <= y < 1.0):
                    continue
                key = group_key(row, grouping)
                cx = min(grid - 1, max(0, int(math.floor(x * grid))))
                cy = min(grid - 1, max(0, int(math.floor(y * grid))))
                buckets[key][(cx, cy)] += 1
                rooms[key].add(as_str(row.get("run_id")))
            for key, counts in buckets.items():
                total = sum(counts.values())
                if total < min_objects:
                    continue
                grouping_name, role, family, variant, room_type, class_name = key
                n_rooms = len(rooms[key])
                for cy in range(grid):
                    for cx in range(grid):
                        count = counts[(cx, cy)]
                        out.append({
                            "grouping": grouping_name,
                            "dataset_role": role,
                            "creator_family": family,
                            "creator_variant": variant,
                            "room_type": room_type,
                            "class_name": class_name,
                            "grid_size": grid,
                            "cell_x": cx,
                            "cell_y": cy,
                            "count": count,
                            "probability": count / max(total, 1),
                            "n_objects": total,
                            "n_rooms": n_rooms,
                        })
    return out


def vec_from_hist(rows: list[dict[str, Any]], grid: int) -> list[float]:
    vec = [0.0] * (grid * grid)
    for row in rows:
        x = int(row["cell_x"])
        y = int(row["cell_y"])
        vec[y * grid + x] = float(row["probability"])
    return vec


def js_divergence(p: list[float], q: list[float]) -> float:
    def kl(a: list[float], b: list[float]) -> float:
        total = 0.0
        for av, bv in zip(a, b):
            if av > 0 and bv > 0:
                total += av * math.log(av / bv, 2)
        return total

    m = [(a + b) * 0.5 for a, b in zip(p, q)]
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def cosine_similarity(p: list[float], q: list[float]) -> float:
    dot = sum(a * b for a, b in zip(p, q))
    np = math.sqrt(sum(a * a for a in p))
    nq = math.sqrt(sum(b * b for b in q))
    if np <= EPS or nq <= EPS:
        return 0.0
    return dot / (np * nq)


def metric_rows(hists: list[dict[str, Any]], min_objects: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str, str], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in hists:
        grouping = as_str(row["grouping"])
        grid = int(row["grid_size"])
        segment = f"room_type={row['room_type']}|class={row['class_name']}"
        for compare_level in ("creator_family", "creator_variant", "dataset_role"):
            left = clean_segment(as_str(row.get(compare_level)))
            key = (grouping, grid, compare_level, segment)
            grouped[key][left].append(row)

    out: list[dict[str, Any]] = []
    for (grouping, grid, compare_level, segment), by_group in grouped.items():
        names = sorted(by_group)
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                lrows = by_group[left]
                rrows = by_group[right]
                ln = int(lrows[0]["n_objects"]) if lrows else 0
                rn = int(rrows[0]["n_objects"]) if rrows else 0
                if ln < min_objects or rn < min_objects:
                    continue
                lv = vec_from_hist(lrows, grid)
                rv = vec_from_hist(rrows, grid)
                diff = [a - b for a, b in zip(lv, rv)]
                l_occ = {i for i, v in enumerate(lv) if v > 0}
                r_occ = {i for i, v in enumerate(rv) if v > 0}
                out.append({
                    "grouping": grouping,
                    "compare_level": compare_level,
                    "grid_size": grid,
                    "segment": segment,
                    "left_group": left,
                    "right_group": right,
                    "left_n_objects": ln,
                    "right_n_objects": rn,
                    "left_n_rooms": int(lrows[0]["n_rooms"]) if lrows else 0,
                    "right_n_rooms": int(rrows[0]["n_rooms"]) if rrows else 0,
                    "hist_l1_mean": sum(abs(x) for x in diff) / len(diff),
                    "hist_l2_mean": math.sqrt(sum(x * x for x in diff) / len(diff)),
                    "jensen_shannon": js_divergence(lv, rv),
                    "cosine_similarity": cosine_similarity(lv, rv),
                    "coverage_left": len(l_occ) / max(grid * grid, 1),
                    "coverage_right": len(r_occ) / max(grid * grid, 1),
                    "coverage_intersection": len(l_occ & r_occ) / max(grid * grid, 1),
                })
    return out


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def median(values: list[float]) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    mid = len(xs) // 2
    if len(xs) % 2:
        return xs[mid]
    return 0.5 * (xs[mid - 1] + xs[mid])


def feature_summary_rows(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groupings = ["overall", "by_room_type", "by_class", "by_room_type_class"]
    buckets: dict[tuple[str, str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in objects:
        if int(row.get("is_trackable_for_distribution") or 0) != 1:
            continue
        for grouping in groupings:
            buckets[group_key(row, grouping)].append(row)

    out: list[dict[str, Any]] = []
    for key, rows in sorted(buckets.items(), key=lambda kv: (kv[0], -len(kv[1]))):
        grouping, role, family, variant, room_type, class_name = key
        x_vals = [as_float(r.get("x_norm")) for r in rows]
        y_vals = [as_float(r.get("y_norm")) for r in rows]
        wall_vals = [as_float(r.get("distance_to_nearest_wall_m")) for r in rows]
        corner_vals = [as_float(r.get("distance_to_nearest_corner_m")) for r in rows]
        x_nums = [float(v) for v in x_vals if v is not None]
        y_nums = [float(v) for v in y_vals if v is not None]
        wall_nums = [float(v) for v in wall_vals if v is not None]
        corner_nums = [float(v) for v in corner_vals if v is not None]
        denom = max(len(rows), 1)
        out.append({
            "grouping": grouping,
            "dataset_role": role,
            "creator_family": family,
            "creator_variant": variant,
            "room_type": room_type,
            "class_name": class_name,
            "n_objects": len(rows),
            "n_rooms": len({as_str(r.get("run_id")) for r in rows}),
            "mean_x_norm": mean(x_nums),
            "mean_y_norm": mean(y_nums),
            "mean_distance_to_wall_m": mean(wall_nums),
            "mean_distance_to_corner_m": mean(corner_nums),
            "near_wall_rate": sum(int(r.get("is_near_wall") or 0) for r in rows) / denom,
            "near_corner_rate": sum(int(r.get("is_near_corner") or 0) for r in rows) / denom,
            "center_zone_rate": sum(int(r.get("is_center_zone") or 0) for r in rows) / denom,
            "small_object_rate": sum(int(r.get("is_small_object") or 0) for r in rows) / denom,
        })
    return out


def room_count_summary_rows(rooms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groupings = ["overall", "by_room_type"]
    buckets: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rooms:
        for grouping in groupings:
            room_type = clean_segment(as_str(row.get("room_type"))) if grouping == "by_room_type" else "__all__"
            key = (
                grouping,
                clean_segment(as_str(row.get("dataset_role"))),
                clean_segment(as_str(row.get("creator_family"))),
                clean_segment(as_str(row.get("creator_variant"))),
                room_type,
            )
            buckets[key].append(row)

    out: list[dict[str, Any]] = []
    for key, rows in sorted(buckets.items(), key=lambda kv: (kv[0], -len(kv[1]))):
        grouping, role, family, variant, room_type = key
        counts = [float(as_float(r.get("n_placements"), 0.0) or 0.0) for r in rows]
        trackable_counts = [float(as_float(r.get("n_trackable_placements"), 0.0) or 0.0) for r in rows]
        out.append({
            "grouping": grouping,
            "dataset_role": role,
            "creator_family": family,
            "creator_variant": variant,
            "room_type": room_type,
            "n_rooms": len(rows),
            "mean_objects_per_room": mean(counts),
            "median_objects_per_room": median(counts),
            "min_objects_per_room": min(counts) if counts else None,
            "max_objects_per_room": max(counts) if counts else None,
            "mean_trackable_objects_per_room": mean(trackable_counts),
        })
    return out


def make_plots(hists: list[dict[str, Any]], plots_dir: Path, max_plots: int) -> int:
    try:
        cache_dir = Path("/private/tmp/cgs_layout_distribution_matplotlib")
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "mpl"))
        os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir / "xdg"))
        import matplotlib.pyplot as plt
    except Exception:
        return 0

    selected = [
        r for r in hists
        if r["grouping"] in {"overall", "by_room_type", "by_class"}
        and int(r["grid_size"]) in {10, 20}
    ]
    groups: dict[tuple[str, int, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        key = (
            row["grouping"],
            int(row["grid_size"]),
            row["dataset_role"],
            row["creator_family"],
            row["room_type"],
            row["class_name"],
        )
        groups[key].append(row)

    count = 0
    for key, rows in sorted(groups.items(), key=lambda kv: (-int(kv[1][0]["n_objects"]), str(kv[0]))):
        if count >= max_plots:
            break
        grouping, grid, role, family, room_type, class_name = key
        vec = vec_from_hist(rows, grid)
        matrix = [vec[i * grid:(i + 1) * grid] for i in range(grid)]
        fig = plt.figure(figsize=(5.5, 5.0))
        plt.imshow(matrix, origin="lower", extent=[0, 1, 0, 1], aspect="equal")
        plt.colorbar(label="probability")
        plt.title(f"{family} | {grouping}\nroom={room_type} class={class_name} n={rows[0]['n_objects']}")
        plt.xlabel("x_norm")
        plt.ylabel("y_norm")
        out_name = "__".join(sanitize_filename(str(x)) for x in key) + ".png"
        out_path = plots_dir / f"grid_{grid}" / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        count += 1
    return count


def sanitize_filename(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9_.-]+", "_", text)
    return text.strip("_") or "all"


def load_role_rules(path: Path | None) -> list[tuple[str, str]]:
    if path is None:
        return []
    data = load_json(path)
    if isinstance(data, dict):
        data = data.get("roles", data.get("roots", []))
    rules: list[tuple[str, str]] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                pattern = item.get("pattern") or item.get("path") or item.get("root")
                role = item.get("role") or item.get("dataset_role")
                if pattern and role:
                    rules.append((str(pattern), str(role)))
    return rules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export distributions from scene.v1 / placement.v1 artifacts.")
    parser.add_argument("--roots", nargs="+", default=["out"], help="Roots to scan recursively.")
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT), help="Output root. A timestamped run dir is created inside it.")
    parser.add_argument("--run-name", default=None, help="Optional new run directory name. Must not already exist.")
    parser.add_argument("--grid-sizes", nargs="+", type=int, default=DEFAULT_GRID_SIZES)
    parser.add_argument("--min-objects-per-group", type=int, default=30)
    parser.add_argument("--role-map-json", default=None, help="Optional JSON with role mapping rules.")
    parser.add_argument("--plots", action="store_true", help="Also render histogram PNGs. CSV/JSON exports do not need this.")
    parser.add_argument("--max-plots", type=int, default=80)
    parser.add_argument("--no-plots", action="store_true", help="Deprecated alias: keep plot rendering disabled.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = [Path(p).expanduser().resolve() for p in args.roots]
    out_root = Path(args.out_root).expanduser().resolve()
    run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out_dir = out_root / run_name
    if out_dir.exists():
        raise SystemExit(f"Output run already exists, refusing to overwrite: {out_dir}")
    out_dir.mkdir(parents=True)

    role_rules = load_role_rules(Path(args.role_map_json).expanduser().resolve() if args.role_map_json else None)
    runs = discover_runs(roots)
    run_rows, room_rows, object_rows, failures = build_rows(runs, role_rules)
    hists = histogram_rows(object_rows, args.grid_sizes, args.min_objects_per_group)
    metrics = metric_rows(hists, args.min_objects_per_group)
    feature_summaries = feature_summary_rows(object_rows)
    room_count_summaries = room_count_summary_rows(room_rows)

    write_csv(out_dir / "runs_all.csv", run_rows, RUN_FIELDS)
    write_csv(out_dir / "rooms_all.csv", room_rows, ROOM_FIELDS)
    write_csv(out_dir / "objects_all.csv", object_rows, OBJECT_FIELDS)
    append_jsonl(out_dir / "objects_all.jsonl", object_rows)
    write_csv(out_dir / "failures.csv", failures, ["run_id", "run_dir", "artifact_variant", "scene_file", "placement_file", "error"])
    write_csv(out_dir / "histograms" / "histograms_all.csv", hists, HIST_FIELDS)
    write_csv(out_dir / "metrics" / "feature_summary.csv", feature_summaries, FEATURE_SUMMARY_FIELDS)
    write_csv(out_dir / "metrics" / "room_object_count_summary.csv", room_count_summaries, ROOM_COUNT_FIELDS)

    metrics_dir = out_dir / "metrics"
    write_csv(metrics_dir / "distribution_metrics_all.csv", metrics, METRIC_FIELDS)
    for grid in args.grid_sizes:
        write_csv(
            metrics_dir / f"distribution_metrics_grid_{grid}.csv",
            [r for r in metrics if int(r["grid_size"]) == int(grid)],
            METRIC_FIELDS,
        )
        write_csv(
            out_dir / "histograms" / f"histograms_grid_{grid}.csv",
            [r for r in hists if int(r["grid_size"]) == int(grid)],
            HIST_FIELDS,
        )

    creator_counts = Counter((r["dataset_role"], r["creator_family"], r["creator_variant"]) for r in object_rows)
    creator_rows = [
        {
            "dataset_role": k[0],
            "creator_family": k[1],
            "creator_variant": k[2],
            "n_objects": v,
        }
        for k, v in sorted(creator_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    write_csv(out_dir / "creator_stats.csv", creator_rows, ["dataset_role", "creator_family", "creator_variant", "n_objects"])

    n_plots = 0
    if args.plots and not args.no_plots:
        n_plots = make_plots(hists, out_dir / "plots", args.max_plots)

    config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "cwd": os.getcwd(),
        "git_commit": git_commit(),
        "input_roots": [str(p) for p in roots],
        "out_dir": str(out_dir),
        "grid_sizes": args.grid_sizes,
        "min_objects_per_group": args.min_objects_per_group,
        "role_rules": role_rules,
        "n_runs_found": len(runs),
        "n_run_rows": len(run_rows),
        "n_room_rows": len(room_rows),
        "n_object_rows": len(object_rows),
        "n_failures": len(failures),
        "n_histogram_rows": len(hists),
        "n_metric_rows": len(metrics),
        "n_feature_summary_rows": len(feature_summaries),
        "n_room_count_summary_rows": len(room_count_summaries),
        "n_plots": n_plots,
        "outputs": {
            "runs_all_csv": str(out_dir / "runs_all.csv"),
            "rooms_all_csv": str(out_dir / "rooms_all.csv"),
            "objects_all_csv": str(out_dir / "objects_all.csv"),
            "objects_all_jsonl": str(out_dir / "objects_all.jsonl"),
            "histograms_all_csv": str(out_dir / "histograms" / "histograms_all.csv"),
            "distribution_metrics_all_csv": str(metrics_dir / "distribution_metrics_all.csv"),
        },
    }
    write_json(out_dir / "config.json", config)
    print(json.dumps(config, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
