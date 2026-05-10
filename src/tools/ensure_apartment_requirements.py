#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SANITARY_REQUIRED = ("toilet", "sink", "bath_or_shower")
SCENE_CANDIDATES = (
    "pipeline/optimal/scene_supplier.optimal.v1.flooring.v1.wall_material.v1.curtains.v1.json",
    "pipeline/optimal/scene_supplier.optimal.v1.flooring.v1.wall_material.v1.json",
    "pipeline/optimal/scene_supplier.optimal.v1.flooring.v1.json",
    "pipeline/optimal/scene_supplier.optimal.v1.json",
    "pipeline/optimal/scene.v1.flooring.v1.wall_material.v1.curtains.v1.json",
    "pipeline/optimal/scene.v1.flooring.v1.wall_material.v1.json",
    "pipeline/optimal/scene.v1.json",
)
SUPPORTED_MESH_SUFFIXES = {".fbx", ".obj", ".glb", ".gltf"}
SUPPLIER_CATALOG_PATH = Path("data/sourse/suppliers/supplier_catalog_canonical.json")
DEFAULT_FLOORING_MATERIALS = REPO_ROOT / "data/floor_materials"
DEFAULT_FLOORING_STYLE_RULES = REPO_ROOT / "config/flooring_style_rules.json"
DEFAULT_WALL_MATERIALS = REPO_ROOT / "data/floor_materials"
LOCAL_TABLE_ASSET_ROOT = Path("data/sourse/imodern")
LOCAL_CHAIR_ASSET_ROOT = Path("data/sourse/suppliers/site_assets_imodern_clean/imodern")
_CATALOG_CACHE: dict[tuple[str, ...], list[dict[str, Any]]] = {}

ROLE_CATEGORY = {
    "toilet": "ToiletFactory",
    "sink": "StandingSinkFactory",
    "shower": "ShowerFactory",
    "bath": "BathtubFactory",
    "bed": "BedFactory",
    "table": "SimpleDeskFactory",
    "chair": "ChairFactory",
    "flat_ceiling_light": "CeilingLightFactory",
}
ROLE_SEMANTIC_GROUP = {
    "toilet": "toilet",
    "sink": "bathroom_sink",
    "shower": "shower",
    "bath": "bathtub",
    "bed": "bed",
    "table": "dining_table",
    "chair": "chair",
    "flat_ceiling_light": "lamp_ceiling",
}
ROLE_CATEGORY_NORMS = {
    "toilet": {"toilet", "toilet_bidet"},
    "sink": {"bathroom_sink", "washbasin"},
    "shower": {"shower", "shower_cabin", "shower_system"},
    "bath": {"bathtub", "bath"},
    "bed": {"bed"},
    "table": {"dining_table", "desk", "table"},
    "chair": {"chair", "dining_chair", "office_chair", "armchair", "stool"},
}

DISCOURAGED_SUPPLIER_KEY_TOKENS = {
    # This model contains several shower-cabin variants in one asset and looks
    # like three cabins placed together after fitting into a small bathroom.
    "shower": {
        "ag01090",
        "ag0407",
        "ag04070",
        "schwarzer_diamant",
        "schwarzer diamant",
        "sonnenstrand",
        "dushevaia-stoika",
        "душевая_стойка",
    },
}


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, data: Any) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def write_json_if_changed(path: str | Path, data: Any) -> Path:
    out = Path(path)
    text = json.dumps(data, ensure_ascii=False, indent=2)
    if out.is_file():
        try:
            if out.read_text(encoding="utf-8") == text:
                return out
        except Exception:
            pass
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return out


def norm(value: Any) -> str:
    return str(value or "").replace("ё", "е").lower()


def item_text(item: dict[str, Any]) -> str:
    source = item.get("source") if isinstance(item.get("source"), dict) else {}
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    supplier = meta.get("supplier_candidate") if isinstance(meta.get("supplier_candidate"), dict) else {}
    return norm(
        " ".join(
            str(x or "")
            for x in (
                item.get("id"),
                item.get("name"),
                item.get("category"),
                item.get("semantic_group"),
                source.get("supplier_unique_key"),
                supplier.get("title"),
                supplier.get("category_norm"),
                supplier.get("category_raw"),
            )
        )
    )


def classify_item(item: dict[str, Any]) -> set[str]:
    text = item_text(item)
    out: set[str] = set()
    if any(x in text for x in ("toilet", "унитаз", "wc", "watercloset")):
        out.add("toilet")
    if any(x in text for x in ("standing sink", "bathroom_sink", "washbasin", "basin", "sink", "раковин", "умывальник")):
        out.add("sink")
    if any(x in text for x in ("bathtub", "bath tub", "bathfactory", "ванн")):
        out.add("bath")
        out.add("bath_or_shower")
    if any(x in text for x in ("shower", "душ", "душев")):
        out.add("shower")
        out.add("bath_or_shower")
    if any(x in text for x in ("bedfactory", " bed", "кровать")):
        out.add("bed")
    is_lamp = any(x in text for x in ("lamp", "light", "люстр", "светиль", "ламп"))
    if (not is_lamp) and any(
        x in text
        for x in (
            "tablefactory",
            "simpledeskfactory",
            "deskfactory",
            "dining_table",
            "coffee_table",
            "side_table",
            "стол",
            "desk",
            "table",
        )
    ):
        out.add("table")
    if (not is_lamp) and any(
        x in text
        for x in (
            "chairfactory",
            "armchairfactory",
            "dining_chair",
            "office_chair",
            "стул",
            "кресл",
            " chair",
            "chair ",
        )
    ):
        out.add("chair")
    return out


def room_items(scene: dict[str, Any]) -> list[dict[str, Any]]:
    items = scene.get("placements") or scene.get("items") or []
    return [x for x in items if isinstance(x, dict)]


def room_bounds(room: dict[str, Any]) -> tuple[float, float]:
    width = float(room.get("width_m") or 0.0)
    depth = float(room.get("depth_m") or 0.0)
    if width > 0 and depth > 0:
        return width, depth
    poly = room.get("floor_polygon") or []
    xs = [float(p.get("x", 0.0)) for p in poly if isinstance(p, dict)]
    ys = [float(p.get("y", p.get("z", 0.0))) for p in poly if isinstance(p, dict)]
    return (max(xs) - min(xs), max(ys) - min(ys)) if xs and ys else (3.0, 3.0)


def _room_polygon_xy(room: dict[str, Any]) -> list[tuple[float, float]]:
    poly = room.get("floor_polygon") or room.get("floor_polygon_xz") or []
    out: list[tuple[float, float]] = []
    if isinstance(poly, list):
        for p in poly:
            if not isinstance(p, dict):
                continue
            try:
                out.append((float(p.get("x", 0.0)), float(p.get("y", p.get("z", 0.0)))))
            except Exception:
                continue
    if len(out) >= 3:
        return out
    width, depth = room_bounds(room)
    return [(0.0, 0.0), (width, 0.0), (width, depth), (0.0, depth)]


def _polygon_bbox(poly: list[tuple[float, float]]) -> dict[str, float]:
    return {
        "x_min": min(p[0] for p in poly),
        "x_max": max(p[0] for p in poly),
        "y_min": min(p[1] for p in poly),
        "y_max": max(p[1] for p in poly),
        "z_min": 0.0,
        "z_max": 0.0,
    }


def _polygon_centroid(poly: list[tuple[float, float]]) -> tuple[float, float]:
    if len(poly) < 3:
        box = _polygon_bbox(poly)
        return _aabb_center_xy(box)
    area2 = 0.0
    cx = 0.0
    cy = 0.0
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        cross = x1 * y2 - x2 * y1
        area2 += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    if abs(area2) < 1e-8:
        box = _polygon_bbox(poly)
        return _aabb_center_xy(box)
    return (cx / (3.0 * area2), cy / (3.0 * area2))


def _polygon_area(poly: list[tuple[float, float]]) -> float:
    if len(poly) < 3:
        box = _polygon_bbox(poly)
        return max(0.0, box["x_max"] - box["x_min"]) * max(0.0, box["y_max"] - box["y_min"])
    area2 = 0.0
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        area2 += x1 * y2 - x2 * y1
    return abs(area2) * 0.5


def _point_segment_distance_xy(point: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    px, py = point
    ax, ay = a
    bx, by = b
    vx, vy = bx - ax, by - ay
    denom = vx * vx + vy * vy
    if denom <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / denom))
    qx, qy = ax + vx * t, ay + vy * t
    return math.hypot(px - qx, py - qy)


def _point_in_polygon_xy(point: tuple[float, float], poly: list[tuple[float, float]], eps: float = 1e-6) -> bool:
    x, y = point
    inside = False
    for a, b in zip(poly, poly[1:] + poly[:1]):
        if _point_segment_distance_xy(point, a, b) <= eps:
            return True
        x1, y1 = a
        x2, y2 = b
        if (y1 > y) == (y2 > y):
            continue
        x_cross = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
        if x_cross >= x - eps:
            inside = not inside
    return inside


def _aabb_inside_room_polygon(aabb: dict[str, float], room: dict[str, Any], margin: float = 0.02) -> bool:
    poly = _room_polygon_xy(room)
    corners = [
        (aabb["x_min"], aabb["y_min"]),
        (aabb["x_min"], aabb["y_max"]),
        (aabb["x_max"], aabb["y_min"]),
        (aabb["x_max"], aabb["y_max"]),
        _aabb_center_xy(aabb),
    ]
    return all(_point_in_polygon_xy(point, poly, eps=margin) for point in corners)


def _yaw_from_vector_xy(vec: tuple[float, float]) -> float:
    vx, vy = vec
    return (math.degrees(math.atan2(vx, vy)) + 360.0) % 360.0


def _room_wall_segments(room: dict[str, Any], min_len: float = 0.18) -> list[dict[str, Any]]:
    poly = _room_polygon_xy(room)
    out: list[dict[str, Any]] = []
    for idx, (p1, p2) in enumerate(zip(poly, poly[1:] + poly[:1])):
        vx, vy = p2[0] - p1[0], p2[1] - p1[1]
        length = math.hypot(vx, vy)
        if length < min_len:
            continue
        tangent = (vx / length, vy / length)
        normal = (-tangent[1], tangent[0])
        mid = ((p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5)
        probe = (mid[0] + normal[0] * 0.08, mid[1] + normal[1] * 0.08)
        if not _point_in_polygon_xy(probe, poly, eps=0.02):
            normal = (-normal[0], -normal[1])
        out.append({"index": idx, "p1": p1, "p2": p2, "length": length, "tangent": tangent, "normal": normal, "mid": mid})
    return out


def _segment_projection_point(point: tuple[float, float], p1: tuple[float, float], p2: tuple[float, float]) -> tuple[float, tuple[float, float]]:
    vx, vy = p2[0] - p1[0], p2[1] - p1[1]
    denom = vx * vx + vy * vy
    if denom <= 1e-12:
        return 0.0, p1
    t = max(0.0, min(1.0, ((point[0] - p1[0]) * vx + (point[1] - p1[1]) * vy) / denom))
    return t, (p1[0] + vx * t, p1[1] + vy * t)


def _aabb_distance_xy(a: dict[str, float], b: dict[str, float]) -> float:
    dx = max(b["x_min"] - a["x_max"], a["x_min"] - b["x_max"], 0.0)
    dy = max(b["y_min"] - a["y_max"], a["y_min"] - b["y_max"], 0.0)
    return math.hypot(dx, dy)


def _opening_clearance_zones(
    room: dict[str, Any],
    groups: tuple[str, ...] = ("doors",),
    *,
    reach: float = 0.72,
    pad: float = 0.18,
) -> list[dict[str, float]]:
    zones: list[dict[str, float]] = []
    poly = _room_polygon_xy(room)
    for group in groups:
        openings = room.get(group) if isinstance(room.get(group), list) else []
        for opening in openings:
            if not isinstance(opening, dict):
                continue
            seg = opening.get("segment") if isinstance(opening.get("segment"), dict) else {}
            try:
                p1 = (float(seg.get("x1")), float(seg.get("y1")))
                p2 = (float(seg.get("x2")), float(seg.get("y2")))
            except Exception:
                continue
            vx, vy = p2[0] - p1[0], p2[1] - p1[1]
            length = math.hypot(vx, vy)
            if length <= 1e-6:
                continue
            tangent = (vx / length, vy / length)
            normal = (-tangent[1], tangent[0])
            mid = ((p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5)
            if not _point_in_polygon_xy((mid[0] + normal[0] * 0.08, mid[1] + normal[1] * 0.08), poly, eps=0.04):
                normal = (-normal[0], -normal[1])
            pts = [
                (p1[0] - tangent[0] * pad, p1[1] - tangent[1] * pad),
                (p2[0] + tangent[0] * pad, p2[1] + tangent[1] * pad),
                (p1[0] - tangent[0] * pad + normal[0] * reach, p1[1] - tangent[1] * pad + normal[1] * reach),
                (p2[0] + tangent[0] * pad + normal[0] * reach, p2[1] + tangent[1] * pad + normal[1] * reach),
            ]
            zones.append(
                {
                    "x_min": min(p[0] for p in pts),
                    "x_max": max(p[0] for p in pts),
                    "y_min": min(p[1] for p in pts),
                    "y_max": max(p[1] for p in pts),
                    "z_min": 0.0,
                    "z_max": 2.4,
                    "opening_group": group,
                }
            )
    return zones


def _opening_interval_on_wall(wall: dict[str, Any], opening: dict[str, Any], pad: float) -> tuple[float, float] | None:
    seg = opening.get("segment") if isinstance(opening.get("segment"), dict) else {}
    try:
        op1 = (float(seg.get("x1")), float(seg.get("y1")))
        op2 = (float(seg.get("x2")), float(seg.get("y2")))
    except Exception:
        return None
    wall_id = str(opening.get("wall_id") or "")
    wall_index = int(wall.get("index") or 0)
    p1 = wall["p1"]
    p2 = wall["p2"]
    tangent = wall["tangent"]
    length = float(wall["length"])
    close_to_wall = (
        wall_id == f"w{wall_index}"
        or (
            _point_segment_distance_xy(op1, p1, p2) <= 0.16
            and _point_segment_distance_xy(op2, p1, p2) <= 0.16
        )
    )
    if not close_to_wall:
        return None
    vals = [(point[0] - p1[0]) * tangent[0] + (point[1] - p1[1]) * tangent[1] for point in (op1, op2)]
    lo = max(0.0, min(vals) - pad)
    hi = min(length, max(vals) + pad)
    return (lo, hi) if hi > lo else None


def _wall_blocked_intervals(room: dict[str, Any], wall: dict[str, Any], *, include_windows: bool = True, include_doors: bool = True) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    groups: list[tuple[str, float]] = []
    if include_windows:
        groups.append(("windows", 0.24))
    if include_doors:
        groups.extend((("doors", 0.34), ("openings", 0.28)))
    for group, pad in groups:
        openings = room.get(group) if isinstance(room.get(group), list) else []
        for opening in openings:
            if not isinstance(opening, dict):
                continue
            interval = _opening_interval_on_wall(wall, opening, pad)
            if interval is not None:
                intervals.append(interval)
    intervals.sort()
    merged: list[tuple[float, float]] = []
    for lo, hi in intervals:
        if not merged or lo > merged[-1][1]:
            merged.append((lo, hi))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
    return merged


def _subtract_wall_intervals(length: float, blocked: list[tuple[float, float]]) -> list[tuple[float, float]]:
    free = [(0.0, length)]
    for blo, bhi in blocked:
        next_free: list[tuple[float, float]] = []
        for flo, fhi in free:
            if bhi <= flo or blo >= fhi:
                next_free.append((flo, fhi))
                continue
            if blo > flo:
                next_free.append((flo, max(flo, blo)))
            if bhi < fhi:
                next_free.append((min(fhi, bhi), fhi))
        free = [(lo, hi) for lo, hi in next_free if hi - lo > 0.12]
    return free


def _rotated_rect_aabb(
    origin: tuple[float, float],
    tangent: tuple[float, float],
    normal: tuple[float, float],
    width: float,
    depth: float,
    z_min: float,
    height: float,
) -> dict[str, float]:
    ox, oy = origin
    points = [
        (ox, oy),
        (ox + tangent[0] * width, oy + tangent[1] * width),
        (ox + normal[0] * depth, oy + normal[1] * depth),
        (ox + tangent[0] * width + normal[0] * depth, oy + tangent[1] * width + normal[1] * depth),
    ]
    return {
        "x_min": round(min(p[0] for p in points), 4),
        "x_max": round(max(p[0] for p in points), 4),
        "y_min": round(min(p[1] for p in points), 4),
        "y_max": round(max(p[1] for p in points), 4),
        "z_min": round(z_min, 4),
        "z_max": round(z_min + height, 4),
    }


def _choose_window_safe_kitchen_wall(room: dict[str, Any], width: float, depth: float, height: float) -> dict[str, Any] | None:
    walls = _room_wall_segments(room, min_len=1.65)
    if not walls:
        return None
    min_target = min(max(2.15, width * 0.58), width)
    candidates: list[dict[str, Any]] = []
    for wall in walls:
        if _wall_blocked_intervals(room, wall, include_windows=True, include_doors=False):
            continue
        blocked = _wall_blocked_intervals(room, wall, include_windows=True, include_doors=True)
        for start, end in _subtract_wall_intervals(float(wall["length"]), blocked):
            free_len = end - start
            if free_len < min_target:
                continue
            target_width = min(width, max(min_target, free_len - 0.06))
            start_along = start + max(0.02, (free_len - target_width) * 0.5)
            tangent = wall["tangent"]
            normal = wall["normal"]
            origin = (
                wall["p1"][0] + tangent[0] * start_along + normal[0] * 0.018,
                wall["p1"][1] + tangent[1] * start_along + normal[1] * 0.018,
            )
            aabb = _rotated_rect_aabb(origin, tangent, normal, target_width, depth, 0.0, height)
            if not _aabb_inside_room_polygon(aabb, room, margin=0.08):
                continue
            door_penalty = sum(
                _aabb_xy_overlap_area(aabb, zone) * 8.0
                for zone in _opening_clearance_zones(room, ("doors", "openings"), reach=0.72, pad=0.2)
            )
            score = -target_width + door_penalty
            candidates.append(
                {
                    "score": score,
                    "wall": wall,
                    "origin": origin,
                    "target_width": target_width,
                    "depth": depth,
                    "height": height,
                    "aabb": aabb,
                    "yaw_deg": (math.degrees(math.atan2(tangent[1], tangent[0])) + 360.0) % 360.0,
                    "free_interval": (start, end),
                }
            )
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x["score"], -x["target_width"]))
    return candidates[0]


def _room_floor_obstacles(
    scene: dict[str, Any],
    *,
    exclude_ids: set[str] | None = None,
    max_z_min: float = 1.25,
) -> list[dict[str, float]]:
    exclude_ids = exclude_ids or set()
    obstacles: list[dict[str, float]] = []
    for item in room_items(scene):
        if str(item.get("id") or "") in exclude_ids or _is_ceiling_light_item(item):
            continue
        aabb = _item_aabb(item)
        if not aabb or float(aabb.get("z_min", 0.0)) > max_z_min:
            continue
        obstacles.append(aabb)
    return obstacles


def _valid_floor_aabb(
    aabb: dict[str, float],
    room: dict[str, Any],
    *,
    obstacles: list[dict[str, float]] = (),
    door_zones: list[dict[str, float]] = (),
    window_zones: list[dict[str, float]] = (),
    margin: float = 0.02,
    obstacle_margin: float = 0.02,
) -> bool:
    if not _aabb_inside_room_polygon(aabb, room, margin=margin):
        return False
    for zone in door_zones:
        if _aabb_xy_intersects(aabb, zone):
            return False
    for zone in window_zones:
        if _aabb_xy_intersects(aabb, zone):
            return False
    for obstacle in obstacles:
        if _aabb_xy_intersects(aabb, obstacle, margin=obstacle_margin):
            return False
    return True


def _wall_mount_candidates(
    room: dict[str, Any],
    size_xy: tuple[float, float],
    *,
    margin: float = 0.12,
    min_wall_len: float | None = None,
) -> list[dict[str, Any]]:
    sx, sy = size_xy
    min_len = min_wall_len if min_wall_len is not None else max(0.35, min(sx, sy) * 0.75)
    candidates: list[dict[str, Any]] = []
    for wall in _room_wall_segments(room, min_len=min_len):
        p1, p2 = wall["p1"], wall["p2"]
        length = float(wall["length"])
        tangent = wall["tangent"]
        normal = wall["normal"]
        offset = min(max(sx, sy) * 0.5 + margin, max(0.18, min(sx, sy) * 0.5 + margin + 0.12))
        slots = max(1, min(5, int(length // max(min_len, 0.35)) + 1))
        for idx in range(slots):
            t = (idx + 1) / (slots + 1)
            wall_point = (p1[0] + tangent[0] * length * t, p1[1] + tangent[1] * length * t)
            cx = wall_point[0] + normal[0] * offset
            cy = wall_point[1] + normal[1] * offset
            yaw = _yaw_from_vector_xy(normal)
            aabb = _candidate_floor_aabb(cx, cy, (sx, sy, 0.1))
            candidates.append({"center": (cx, cy), "yaw": yaw, "aabb": aabb, "wall": wall, "wall_t": t})
    return candidates


def _choose_best_floor_candidate(
    room: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    obstacles: list[dict[str, float]] = (),
    door_zones: list[dict[str, float]] = (),
    window_zones: list[dict[str, float]] = (),
    prefer_xy: tuple[float, float] | None = None,
    keep_near_xy: tuple[float, float] | None = None,
) -> dict[str, Any] | None:
    if not candidates:
        return None
    poly = _room_polygon_xy(room)
    prefer_xy = prefer_xy or _polygon_centroid(poly)
    best: dict[str, Any] | None = None
    best_score = float("inf")
    for cand in candidates:
        aabb = cand["aabb"]
        if not _valid_floor_aabb(aabb, room, obstacles=obstacles, door_zones=door_zones, window_zones=window_zones):
            continue
        cx, cy = _aabb_center_xy(aabb)
        score = math.hypot(cx - prefer_xy[0], cy - prefer_xy[1]) * 0.2
        if keep_near_xy is not None:
            score += math.hypot(cx - keep_near_xy[0], cy - keep_near_xy[1]) * 0.35
        for zone in door_zones:
            score += max(0.0, 1.0 - _aabb_distance_xy(aabb, zone)) * 2.0
        for zone in window_zones:
            score += max(0.0, 0.65 - _aabb_distance_xy(aabb, zone)) * 0.8
        for obstacle in obstacles:
            score += max(0.0, 0.45 - _aabb_distance_xy(aabb, obstacle)) * 0.5
        if score < best_score:
            best_score = score
            best = cand
    return best


def _supported_mesh(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_MESH_SUFFIXES


def _find_preferred_mesh(root: Path) -> Path | None:
    if not root.exists():
        return None
    files = [p for p in root.rglob("*") if _supported_mesh(p)]
    if not files:
        return None
    order = {".fbx": 0, ".obj": 1, ".glb": 2, ".gltf": 3}
    files.sort(key=lambda p: (order.get(p.suffix.lower(), 99), len(p.parts), str(p).lower()))
    return files[0]


def _normalize_catalog_candidate(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    dims = out.get("dimensions_cm") if isinstance(out.get("dimensions_cm"), dict) else {}
    for src_key, dst_key in (("width", "width_cm"), ("depth", "depth_cm"), ("height", "height_cm")):
        if out.get(dst_key) is None and dims.get(src_key) is not None:
            out[dst_key] = dims.get(src_key)
    mesh_path = str(out.get("asset_local_path") or out.get("mesh_local_path") or "").strip()
    if mesh_path and Path(mesh_path).expanduser().is_file():
        out["asset_local_path"] = str(Path(mesh_path).expanduser().resolve())
        out["asset_format"] = out.get("asset_format") or Path(mesh_path).suffix.lstrip(".").lower()
        out["asset_status"] = out.get("asset_status") or "local_supplier_asset"
    category_norm = norm(out.get("category_norm"))
    title_text = norm(" ".join(str(out.get(key) or "") for key in ("title", "category_raw", "description")))
    if not str(out.get("semantic_group") or "").strip():
        if category_norm in {"toilet", "toilet_bidet"}:
            out["semantic_group"] = "toilet"
        elif category_norm in {"bathroom_sink", "washbasin"}:
            out["semantic_group"] = "bathroom_sink"
        elif category_norm == "bathroom_furniture" and any(token in title_text for token in ("раковин", "умываль", "sink", "basin", "vanity")):
            out["semantic_group"] = "bathroom_sink"
        elif category_norm in {"shower", "shower_cabin", "shower_system"}:
            out["semantic_group"] = "shower"
        elif category_norm in {"bathtub", "bath"}:
            out["semantic_group"] = "bathtub"
        elif category_norm in {"dining_table", "desk", "table"}:
            out["semantic_group"] = "dining_table" if category_norm != "desk" else "desk"
        elif category_norm in {"chair", "dining_chair", "office_chair", "armchair", "stool"}:
            out["semantic_group"] = "chair" if category_norm != "armchair" else "armchair"
        elif category_norm == "bed":
            out["semantic_group"] = "bed"
    return out


def _candidate_has_local_mesh(candidate: dict[str, Any]) -> bool:
    mesh_path = str(candidate.get("asset_local_path") or candidate.get("mesh_local_path") or "").strip()
    return bool(mesh_path and _supported_mesh(Path(mesh_path).expanduser()))


def _metadata_candidates_from_roots(search_roots: tuple[Path, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in search_roots:
        if not root.exists() or root.name == "imodern":
            continue
        for meta_path in root.rglob("*.metadata.json"):
            if "supplier_assets" not in str(meta_path):
                continue
            try:
                row = read_json(meta_path)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            mesh = _find_preferred_mesh(meta_path.parent)
            if mesh is None:
                continue
            row = _normalize_catalog_candidate(row)
            row["asset_local_path"] = str(mesh.resolve())
            row["asset_format"] = mesh.suffix.lstrip(".").lower()
            row["asset_status"] = row.get("asset_status") or "local_supplier_asset_cache"
            rows.append(row)
    return rows


def _color_from_title(text: str) -> str | None:
    tokens = []
    low = norm(text)
    for color in (
        "белый",
        "белая",
        "серый",
        "серая",
        "черный",
        "черная",
        "коричневый",
        "орех",
        "бежевый",
        "бронза",
        "светлая",
        "темная",
    ):
        if color in low:
            tokens.append(color)
    return " ".join(tokens) or None


def _local_table_candidates() -> list[dict[str, Any]]:
    root = LOCAL_TABLE_ASSET_ROOT
    if not root.exists():
        return []
    rows: list[dict[str, Any]] = []
    for folder in root.iterdir():
        if not folder.is_dir():
            continue
        text = norm(folder.name)
        if ("стол" not in text and "table" not in text) or any(x in text for x in ("лампа", "lamp")):
            continue
        mesh = _find_preferred_mesh(folder)
        if mesh is None:
            continue
        if "журн" in text or "coffee" in text:
            category_norm = "coffee_table"
            semantic_group = "coffee_table"
        elif "рабоч" in text or "письмен" in text or "desk" in text:
            category_norm = "desk"
            semantic_group = "desk"
        else:
            category_norm = "dining_table"
            semantic_group = "dining_table"
        if category_norm == "coffee_table":
            continue
        nums = [float(x) for x in re.findall(r"(?<!\d)(\d{2,3})(?!\d)", folder.name)]
        width_cm = nums[0] if nums else 140.0
        depth_cm = nums[1] if len(nums) > 1 and nums[1] <= 120.0 else 80.0
        height_cm = 76.0
        title = folder.name.replace("_", " ")
        rows.append(
            {
                "unique_key": f"local_imodern::{folder.name}",
                "source_site": "imodern_local",
                "title": title,
                "category_raw": "Столы",
                "category_norm": category_norm,
                "semantic_group": semantic_group,
                "asset_status": "local_supplier_asset",
                "asset_format": mesh.suffix.lstrip(".").lower(),
                "asset_local_path": str(mesh.resolve()),
                "style": "современный",
                "color": _color_from_title(title),
                "width_cm": width_cm,
                "depth_cm": depth_cm,
                "height_cm": height_cm,
                "description": title,
            }
        )
    return rows


def _local_chair_candidates() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in (LOCAL_CHAIR_ASSET_ROOT, LOCAL_TABLE_ASSET_ROOT):
        if not root.exists():
            continue
        for folder in root.iterdir():
            if not folder.is_dir():
                continue
            text = norm(folder.name)
            if ("стул" not in text and "chair" not in text) or any(x in text for x in ("барн", "полубар", "bar_")):
                continue
            clean_glb = folder / "built" / "model.glb"
            mesh = clean_glb if clean_glb.is_file() else _find_preferred_mesh(folder)
            if mesh is None:
                continue
            key = str(mesh.resolve())
            if key in seen:
                continue
            seen.add(key)
            meta_files = sorted(folder.glob("*.metadata.json"))
            row: dict[str, Any] = {}
            if meta_files:
                try:
                    loaded = read_json(meta_files[0])
                    if isinstance(loaded, dict):
                        row = _normalize_catalog_candidate(loaded)
                except Exception:
                    row = {}
            title = str(row.get("title") or folder.name.replace("_", " "))
            row.update(
                {
                    "unique_key": row.get("unique_key") or f"local_imodern_chair::{folder.name}",
                    "source_site": row.get("source_site") or "imodern_local",
                    "title": title,
                    "category_raw": row.get("category_raw") or "Стулья",
                    "category_norm": row.get("category_norm") or "chair",
                    "semantic_group": row.get("semantic_group") or "chair",
                    "asset_status": "local_supplier_asset",
                    "asset_format": mesh.suffix.lstrip(".").lower(),
                    "asset_local_path": str(mesh.resolve()),
                    "style": row.get("style") or "современный",
                    "color": row.get("color") or _color_from_title(title),
                    "width_cm": row.get("width_cm") or 50.0,
                    "depth_cm": row.get("depth_cm") or 56.0,
                    "height_cm": row.get("height_cm") or 82.0,
                    "description": row.get("description") or title,
                }
            )
            rows.append(row)
    return rows


def _local_bed_candidates() -> list[dict[str, Any]]:
    root = LOCAL_TABLE_ASSET_ROOT
    if not root.exists():
        return []
    rows: list[dict[str, Any]] = []
    for folder in root.iterdir():
        if not folder.is_dir():
            continue
        text = norm(folder.name)
        if "кровать" not in text and "bed" not in text:
            continue
        # Sofa-beds and armchair-beds are useful furniture, but they should not
        # satisfy the apartment-level requirement for a real sleeping zone.
        if "диван" in text or "sofa" in text or "кресло" in text:
            continue
        mesh = _find_preferred_mesh(folder)
        if mesh is None:
            continue
        nums = [float(x) for x in re.findall(r"(?<!\d)(\d{2,3})(?!\d)", folder.name)]
        width_cm = 160.0
        depth_cm = 200.0
        for idx, value in enumerate(nums):
            if 80.0 <= value <= 220.0:
                width_cm = value
                if idx + 1 < len(nums) and 120.0 <= nums[idx + 1] <= 240.0:
                    depth_cm = nums[idx + 1]
                break
        title = folder.name.replace("_", " ")
        rows.append(
            {
                "unique_key": f"local_imodern_bed::{folder.name}",
                "source_site": "imodern_local",
                "title": title,
                "category_raw": "Кровати",
                "category_norm": "bed",
                "semantic_group": "bed",
                "asset_status": "local_supplier_asset",
                "asset_format": mesh.suffix.lstrip(".").lower(),
                "asset_local_path": str(mesh.resolve()),
                "style": "современный",
                "color": _color_from_title(title),
                "width_cm": width_cm,
                "depth_cm": depth_cm,
                "height_cm": 90.0,
                "description": title,
            }
        )
    return rows


def load_catalog_candidates(search_roots: tuple[Path, ...]) -> list[dict[str, Any]]:
    key = tuple(str(p.expanduser().resolve()) for p in search_roots)
    cached = _CATALOG_CACHE.get(key)
    if cached is not None:
        return cached
    rows: list[dict[str, Any]] = []
    if SUPPLIER_CATALOG_PATH.is_file():
        try:
            payload = read_json(SUPPLIER_CATALOG_PATH)
            items = payload.get("items") if isinstance(payload, dict) else []
            rows.extend(_normalize_catalog_candidate(x) for x in items if isinstance(x, dict))
        except Exception:
            pass
    rows.extend(_metadata_candidates_from_roots(search_roots))
    rows.extend(_local_table_candidates())
    rows.extend(_local_chair_candidates())
    rows.extend(_local_bed_candidates())
    _CATALOG_CACHE[key] = rows
    return rows


def _candidate_dims_m(candidate: dict[str, Any], fallback: tuple[float, float, float]) -> tuple[float, float, float]:
    try:
        width = float(candidate.get("width_cm") or 0.0) / 100.0
        depth = float(candidate.get("depth_cm") or 0.0) / 100.0
        height = float(candidate.get("height_cm") or 0.0) / 100.0
    except Exception:
        width = depth = height = 0.0
    fw, fd, fh = fallback
    return (width if width > 0 else fw, depth if depth > 0 else fd, height if height > 0 else fh)


def _text_tokens(value: Any) -> set[str]:
    return {x for x in re.split(r"[^0-9a-zа-я]+", norm(value)) if len(x) > 2}


def _scene_style_tokens(scene: dict[str, Any], prompt_room_type: str | None = None) -> set[str]:
    parts: list[str] = [prompt_room_type or ""]
    room = scene.get("room") if isinstance(scene.get("room"), dict) else {}
    meta = scene.get("meta") if isinstance(scene.get("meta"), dict) else {}
    room_meta = room.get("meta") if isinstance(room.get("meta"), dict) else {}
    for source in (room, meta, room_meta):
        for key in ("style", "color", "materials", "description", "prompt", "prompt_text", "source"):
            value = source.get(key)
            if isinstance(value, (str, int, float)):
                parts.append(str(value))
    return _text_tokens(" ".join(parts))


def _candidate_role_match(candidate: dict[str, Any], role: str) -> bool:
    category_norm = norm(candidate.get("category_norm"))
    semantic_group = norm(candidate.get("semantic_group"))
    accepted = ROLE_CATEGORY_NORMS.get(role, {role})
    if category_norm in accepted or semantic_group in accepted:
        return True
    if role == "toilet" and "toilet" in semantic_group:
        return True
    if role == "sink" and semantic_group == "bathroom_sink":
        return True
    if role == "sink" and category_norm == "bathroom_furniture":
        identity = _candidate_identity_text(candidate)
        return any(token in identity for token in ("раковин", "умываль", "sink", "basin", "vanity"))
    if role == "shower" and semantic_group == "shower":
        return True
    if role == "table" and semantic_group in {"dining_table", "desk"}:
        return True
    if role == "chair" and semantic_group in {"chair", "dining_chair", "office_chair", "armchair"}:
        return True
    return False


def _candidate_identity_text(candidate: dict[str, Any]) -> str:
    return norm(
        " ".join(
            str(candidate.get(key) or "")
            for key in (
                "unique_key",
                "title",
                "product_url",
                "model_page_url",
                "model_download_url",
                "asset_local_path",
                "description",
            )
        )
    )


def _candidate_discouraged_for_role(candidate: dict[str, Any], role: str) -> bool:
    tokens = DISCOURAGED_SUPPLIER_KEY_TOKENS.get(role) or set()
    if not tokens:
        return False
    text = _candidate_identity_text(candidate)
    return any(token in text for token in tokens)


def select_catalog_candidate(
    role: str,
    target_size: tuple[float, float, float],
    scene: dict[str, Any],
    prompt_room_type: str | None,
    search_roots: tuple[Path, ...],
) -> dict[str, Any] | None:
    scene_tokens = _scene_style_tokens(scene, prompt_room_type=prompt_room_type)
    best: tuple[float, dict[str, Any]] | None = None
    for candidate in load_catalog_candidates(search_roots):
        if not _candidate_role_match(candidate, role):
            continue
        if _candidate_discouraged_for_role(candidate, role):
            continue
        if not _candidate_has_local_mesh(candidate):
            continue
        category_norm = norm(candidate.get("category_norm"))
        if role == "table" and category_norm not in {"dining_table", "desk", "table"}:
            continue
        if role == "chair" and category_norm not in {"chair", "dining_chair", "office_chair", "armchair", "stool"}:
            continue
        cw, cd, ch = _candidate_dims_m(candidate, target_size)
        tw, td, th = target_size
        normal = abs(math.log(max(cw, 0.02) / max(tw, 0.02))) + abs(math.log(max(cd, 0.02) / max(td, 0.02))) + 0.55 * abs(math.log(max(ch, 0.02) / max(th, 0.02)))
        swapped = abs(math.log(max(cd, 0.02) / max(tw, 0.02))) + abs(math.log(max(cw, 0.02) / max(td, 0.02))) + 0.55 * abs(math.log(max(ch, 0.02) / max(th, 0.02)))
        size_score = min(normal, swapped)
        candidate_tokens = _text_tokens(
            " ".join(
                str(candidate.get(key) or "")
                for key in ("title", "category_raw", "category_norm", "semantic_group", "style", "color", "materials", "description")
            )
        )
        style_overlap = len(scene_tokens & candidate_tokens)
        category_bonus = -0.55 if norm(candidate.get("semantic_group")) == ROLE_SEMANTIC_GROUP.get(role) else 0.0
        if role == "table" and category_norm == "dining_table":
            category_bonus -= 0.35
        if role == "chair":
            if category_norm in {"chair", "dining_chair"}:
                category_bonus -= 0.45
            elif category_norm == "armchair":
                category_bonus += 0.45
            elif category_norm == "stool":
                category_bonus += 0.65
            identity = _candidate_identity_text(candidate)
            if "site_assets_imodern_clean" in identity or "стул_alessa" in identity or "alessa" in identity:
                category_bonus -= 1.05
        if role == "shower":
            if category_norm == "shower_system":
                category_bonus += 1.15
            elif category_norm == "shower_cabin":
                category_bonus -= 0.85
        if role == "bath" and category_norm in {"bathtub", "bath"}:
            category_bonus -= 0.75
        ready_bonus = -0.15 if str(candidate.get("asset_status") or "").startswith("local") else 0.0
        score = size_score + category_bonus + ready_bonus - min(style_overlap, 5) * 0.04
        candidate = dict(candidate)
        candidate["requirement_match_score"] = round(score, 6)
        candidate["requirement_size_score"] = round(size_score, 6)
        candidate["requirement_style_overlap"] = sorted(scene_tokens & candidate_tokens)
        if best is None or score < best[0]:
            best = (score, candidate)
    return best[1] if best is not None else None


def _compact_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "unique_key",
        "source_site",
        "title",
        "brand",
        "collection",
        "category_raw",
        "category_norm",
        "semantic_group",
        "product_url",
        "model_page_url",
        "model_download_url",
        "model_download_landing_url",
        "model_vendor_url",
        "asset_status",
        "asset_format",
        "asset_local_path",
        "price_value",
        "price_currency",
        "style",
        "color",
        "materials",
        "width_cm",
        "depth_cm",
        "height_cm",
        "description",
        "requirement_match_score",
        "requirement_size_score",
        "requirement_style_overlap",
    ]
    return {k: deepcopy(candidate.get(k)) for k in keys if k in candidate}


def _fit_catalog_size_to_room(role: str, candidate: dict[str, Any], target_size: tuple[float, float, float], room_size: tuple[float, float]) -> tuple[float, float, float]:
    sx, sy, sz = _candidate_dims_m(candidate, target_size)
    width, depth = room_size
    category_norm = norm(candidate.get("category_norm"))
    if role == "shower" and category_norm == "shower_system":
        sx, sy, sz = max(sx, 0.70), max(sy, 0.70), max(sz, 1.75)
    max_w = max(0.25, width - 0.24)
    max_d = max(0.25, depth - 0.24)
    if role == "shower" and min(width, depth) < 1.65:
        max_w = min(max_w, max(0.55, width * 0.58))
        max_d = min(max_d, max(0.55, depth * 0.58))
    if role == "bath":
        max_w = min(max_w, max(1.15, width - 0.24))
        max_d = min(max_d, max(0.62, depth - 0.24))
    if role == "chair":
        max_w = min(max_w, 0.72)
        max_d = min(max_d, 0.78)
    if role == "sink":
        if min(width, depth) < 1.6 or width * depth < 2.3:
            max_w = min(max_w, 0.48)
            max_d = min(max_d, 0.34)
        else:
            max_w = min(max_w, 0.82)
            max_d = min(max_d, 0.52)
    if role == "bed":
        max_w = min(max_w, max(0.95, width - 0.28))
        max_d = min(max_d, max(1.35, depth - 0.28))
    scale = min(1.0, max_w / max(sx, 1e-6), max_d / max(sy, 1e-6))
    sx, sy, sz = sx * scale, sy * scale, sz * scale
    if role == "sink" and sz < 0.24:
        sz = max(0.12, sz)
    if role == "chair":
        return max(0.38, sx), max(0.42, sy), max(0.72, min(sz, 0.95))
    if role == "shower":
        return max(0.62, sx), max(0.62, sy), max(1.65, min(sz, 2.15))
    if role == "bath":
        return max(1.10, sx), max(0.58, sy), max(0.42, min(sz, 0.78))
    if role == "bed":
        return max(0.90, sx), max(1.35, sy), max(0.35, min(sz, 1.10))
    return max(0.12, sx), max(0.12, sy), max(0.08, sz)


def _sink_candidate_is_vanity(candidate: dict[str, Any]) -> bool:
    if norm(candidate.get("category_norm")) == "bathroom_furniture":
        return True
    identity = _candidate_identity_text(candidate)
    return any(token in identity for token in ("тумба", "vanity", "cabinet"))


def _sink_item_is_vanity(item: dict[str, Any]) -> bool:
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    candidate = meta.get("supplier_candidate") if isinstance(meta.get("supplier_candidate"), dict) else {}
    return _sink_candidate_is_vanity(candidate) or any(token in item_text(item) for token in ("тумба", "vanity", "cabinet"))


def _sink_size_and_z_for_room(room: dict[str, Any], item: dict[str, Any] | None = None) -> tuple[tuple[float, float, float], float]:
    width, depth = room_bounds(room)
    small = min(width, depth) < 1.6 or width * depth < 2.3
    is_vanity = _sink_item_is_vanity(item or {})
    if small:
        return (0.42, 0.30, 0.40 if is_vanity else 0.22), 0.0 if is_vanity else 0.72
    if is_vanity:
        return (min(0.80, max(0.56, width * 0.42)), min(0.46, max(0.34, depth * 0.26)), 0.55), 0.0
    return (0.40, min(0.72, max(0.48, depth * 0.36)), 0.22), 0.72


def _required_item_constraints(role: str, z_min: float) -> dict[str, Any]:
    if role == "flat_ceiling_light":
        return {"mount_type": "ceiling", "under_ceiling": True}
    if role == "sink" and z_min > 0.05:
        return {"mount_type": "wall"}
    return {"mount_type": "floor", "touch_floor": {"side": "bottom"}}


def make_supplier_required_item(
    *,
    room_id: str,
    role: str,
    index: int,
    center_xy: tuple[float, float],
    size: tuple[float, float, float],
    yaw_deg: float,
    z_min: float,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    sx, sy, sz = size
    cx, cy = center_xy
    item_id = f"req_{role}_{index:02d}"
    mesh_path = str(candidate.get("asset_local_path") or "").strip()
    semantic_group = ROLE_SEMANTIC_GROUP.get(role, role)
    return {
        "id": item_id,
        "name": str(candidate.get("title") or f"supplier {role}"),
        "category": ROLE_CATEGORY.get(role, "SupplierObject"),
        "semantic_group": semantic_group,
        "position_m": [round(cx, 4), round(cy, 4), round(z_min + sz / 2.0, 4)],
        "size_m": [round(sx, 4), round(sy, 4), round(sz, 4)],
        "yaw_deg": round(yaw_deg, 4),
        "rotation_deg": round(yaw_deg, 4),
        "yaw_rad": round(math.radians(yaw_deg), 8),
        "aabb": {
            "x_min": round(cx - sx / 2.0, 4),
            "x_max": round(cx + sx / 2.0, 4),
            "y_min": round(cy - sy / 2.0, 4),
            "y_max": round(cy + sy / 2.0, 4),
            "z_min": round(z_min, 4),
            "z_max": round(z_min + sz, 4),
        },
        "constraints": _required_item_constraints(role, z_min),
        "asset": {"mesh_path": mesh_path, "mesh_fit_mode": "uniform"},
        "source": {
            "placement_source": "requirement_postprocess",
            "asset_source": "supplier_catalog_local_asset",
            "supplier_replaced": True,
            "supplier_target_id": item_id,
            "supplier_unique_key": candidate.get("unique_key"),
            "supplier_source_site": candidate.get("source_site"),
            "supplier_product_url": candidate.get("product_url") or candidate.get("model_page_url"),
            "supplier_model_url": candidate.get("model_download_url") or candidate.get("model_download_landing_url"),
            "placeholder_bbox": False,
            "room_id": room_id,
        },
        "meta": {
            "placeholder_bbox": False,
            "supplier_binding_applied": True,
            "supplier_requirement_added": True,
            "required_role": role,
            "room_id": room_id,
            "supplier_candidate": _compact_candidate(candidate),
            "supplier_candidate_pool": [_compact_candidate(candidate)],
        },
    }


def _record_missing_catalog_asset(scene: dict[str, Any], role: str) -> None:
    meta = scene.setdefault("meta", {})
    req = meta.setdefault("requirement_postprocess", {})
    req.setdefault("missing_catalog_asset", []).append(role)


def _clamp_center(center: tuple[float, float], size: tuple[float, float], room_size: tuple[float, float], margin: float) -> tuple[float, float]:
    sx, sy = size
    width, depth = room_size
    min_x = margin + sx / 2.0
    max_x = max(min_x, width - margin - sx / 2.0)
    min_y = margin + sy / 2.0
    max_y = max(min_y, depth - margin - sy / 2.0)
    return (
        min(max(center[0], min_x), max_x),
        min(max(center[1], min_y), max_y),
    )


def _primary_door_side(room: dict[str, Any], room_size: tuple[float, float]) -> str | None:
    width, depth = room_size
    doors = room.get("doors") if isinstance(room.get("doors"), list) else []
    if not doors:
        return None
    seg = (doors[0] or {}).get("segment") if isinstance(doors[0], dict) else {}
    if not isinstance(seg, dict):
        return None
    try:
        x1, x2 = float(seg.get("x1")), float(seg.get("x2"))
        y1, y2 = float(seg.get("y1")), float(seg.get("y2"))
    except Exception:
        return None
    if abs(x1 - x2) < abs(y1 - y2):
        return "left" if (x1 + x2) * 0.5 < width * 0.5 else "right"
    return "bottom" if (y1 + y2) * 0.5 < depth * 0.5 else "top"


def _sanitary_layout(
    role: str,
    size: tuple[float, float],
    room: dict[str, Any],
    margin: float,
    *,
    obstacles: list[dict[str, float]] | None = None,
) -> tuple[tuple[float, float], float]:
    sx, sy = size
    room_size = room_bounds(room)
    width, depth = room_size
    door_zones = _opening_clearance_zones(room, ("doors",), reach=0.78, pad=0.22)
    window_zones = _opening_clearance_zones(room, ("windows",), reach=0.42, pad=0.18)
    wall_candidates = _wall_mount_candidates(
        room,
        size,
        margin=margin,
        min_wall_len=max(0.32, min(max(sx, sy), min(width, depth)) * 0.65),
    )
    prefer_by_role = {
        "toilet": _polygon_centroid(_room_polygon_xy(room)),
        "sink": _polygon_centroid(_room_polygon_xy(room)),
        "shower": (width * 0.5, depth * 0.5),
        "bath": (width * 0.5, depth * 0.5),
    }
    best = _choose_best_floor_candidate(
        room,
        wall_candidates,
        obstacles=obstacles or [],
        door_zones=door_zones,
        window_zones=window_zones,
        prefer_xy=prefer_by_role.get(role),
    )
    if best is not None:
        return best["center"], float(best["yaw"])

    door_side = _primary_door_side(room, room_size)
    fallback_candidates: list[tuple[tuple[float, float], float]] = []
    if role == "toilet":
        if door_side == "bottom":
            fallback_candidates.extend([((width * 0.5, depth - margin - sy / 2.0), 0.0), ((margin + sx / 2.0, depth * 0.55), 90.0), ((width - margin - sx / 2.0, depth * 0.55), 270.0)])
        elif door_side == "top":
            fallback_candidates.extend([((width * 0.5, margin + sy / 2.0), 180.0), ((margin + sx / 2.0, depth * 0.45), 90.0), ((width - margin - sx / 2.0, depth * 0.45), 270.0)])
        elif door_side == "right":
            fallback_candidates.extend([((margin + sx / 2.0, depth * 0.5), 90.0), ((width * 0.5, margin + sy / 2.0), 0.0), ((width * 0.5, depth - margin - sy / 2.0), 180.0)])
        elif door_side == "left":
            fallback_candidates.extend([((width - margin - sx / 2.0, depth * 0.5), 270.0), ((width * 0.5, margin + sy / 2.0), 0.0), ((width * 0.5, depth - margin - sy / 2.0), 180.0)])
    elif role == "sink":
        if door_side == "bottom":
            fallback_candidates.extend(
                [
                    ((width - margin - sx / 2.0, margin + sy / 2.0), 270.0),
                    ((margin + sx / 2.0, margin + sy / 2.0), 90.0),
                    ((margin + sx / 2.0, depth * 0.5), 90.0),
                    ((width - margin - sx / 2.0, depth * 0.5), 270.0),
                    ((width * 0.5, depth - margin - sy / 2.0), 180.0),
                ]
            )
        elif door_side == "top":
            fallback_candidates.extend(
                [
                    ((width - margin - sx / 2.0, depth - margin - sy / 2.0), 270.0),
                    ((margin + sx / 2.0, depth - margin - sy / 2.0), 90.0),
                    ((margin + sx / 2.0, depth * 0.5), 90.0),
                    ((width - margin - sx / 2.0, depth * 0.5), 270.0),
                    ((width * 0.5, margin + sy / 2.0), 0.0),
                ]
            )
        elif door_side == "right":
            fallback_candidates.extend(
                [
                    ((margin + sx / 2.0, margin + sy / 2.0), 90.0),
                    ((margin + sx / 2.0, depth - margin - sy / 2.0), 90.0),
                    ((width * 0.5, margin + sy / 2.0), 0.0),
                    ((width * 0.5, depth - margin - sy / 2.0), 180.0),
                ]
            )
        elif door_side == "left":
            fallback_candidates.extend(
                [
                    ((width - margin - sx / 2.0, margin + sy / 2.0), 270.0),
                    ((width - margin - sx / 2.0, depth - margin - sy / 2.0), 270.0),
                    ((width * 0.5, margin + sy / 2.0), 0.0),
                    ((width * 0.5, depth - margin - sy / 2.0), 180.0),
                ]
            )
        else:
            fallback_candidates.extend([((margin + sx / 2.0, depth * 0.5), 90.0), ((width - margin - sx / 2.0, depth * 0.5), 270.0)])
    elif role in {"shower", "bath"}:
        fallback_candidates.extend(
            [
                ((margin + sx / 2.0, depth - margin - sy / 2.0), 180.0),
                ((width - margin - sx / 2.0, depth - margin - sy / 2.0), 180.0),
                ((margin + sx / 2.0, margin + sy / 2.0), 0.0),
                ((width - margin - sx / 2.0, margin + sy / 2.0), 0.0),
            ]
        )
    scored_fallbacks: list[tuple[float, tuple[float, float], float]] = []
    for center, yaw in fallback_candidates:
        center = _clamp_center(center, size, room_size, margin)
        aabb = _candidate_floor_aabb(center[0], center[1], (sx, sy, 0.1))
        if not _aabb_inside_room_polygon(aabb, room, margin=0.03):
            continue
        score = sum(_aabb_xy_overlap_area(aabb, zone) * 8.0 + max(0.0, 0.35 - _aabb_distance_xy(aabb, zone)) for zone in door_zones)
        score += sum(_aabb_xy_overlap_area(aabb, obstacle) * 80.0 for obstacle in (obstacles or []))
        scored_fallbacks.append((score, center, yaw))
    if scored_fallbacks:
        scored_fallbacks.sort(key=lambda x: x[0])
        return scored_fallbacks[0][1], scored_fallbacks[0][2]

    defaults = {
        "toilet": (min(width - 0.33, max(0.33, width * 0.28)), margin + 0.34),
        "sink": (max(0.34, width * 0.50), min(depth - 0.21, max(0.35, depth * 0.50))),
        "shower": (max(0.53, width - 0.53), max(0.53, depth - 0.53)),
        "bath": (max(0.65, width * 0.5), max(0.45, depth - 0.45)),
    }
    fallback_yaw = {"bottom": 0.0, "top": 180.0, "left": 90.0, "right": 270.0}.get(str(door_side or ""), 0.0)
    return _clamp_center(defaults.get(role, (width * 0.5, depth * 0.5)), size, room_size, margin), fallback_yaw


def _set_item_geometry(
    item: dict[str, Any],
    *,
    center_xy: tuple[float, float],
    size: tuple[float, float, float],
    yaw_deg: float,
    z_min: float,
    role: str | None = None,
) -> None:
    sx, sy, sz = size
    cx, cy = center_xy
    item["position_m"] = [round(cx, 4), round(cy, 4), round(z_min + sz / 2.0, 4)]
    item["size_m"] = [round(sx, 4), round(sy, 4), round(sz, 4)]
    item["yaw_deg"] = round(yaw_deg, 4)
    item["rotation_deg"] = round(yaw_deg, 4)
    item["yaw_rad"] = round(math.radians(yaw_deg), 8)
    item["aabb"] = {
        "x_min": round(cx - sx / 2.0, 4),
        "x_max": round(cx + sx / 2.0, 4),
        "y_min": round(cy - sy / 2.0, 4),
        "y_max": round(cy + sy / 2.0, 4),
        "z_min": round(z_min, 4),
        "z_max": round(z_min + sz, 4),
    }
    if role:
        item["semantic_group"] = ROLE_SEMANTIC_GROUP.get(role, item.get("semantic_group") or role)
        item["constraints"] = _required_item_constraints(role, z_min)
        meta = item.setdefault("meta", {})
        meta["sanitary_layout_repaired"] = True
        meta["required_role"] = role
        source = item.setdefault("source", {})
        source["placeholder_bbox"] = False


def _remove_items(scene: dict[str, Any], predicate) -> list[dict[str, Any]]:
    placements = scene.get("placements")
    if not isinstance(placements, list):
        return []
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    for item in placements:
        if isinstance(item, dict) and predicate(item):
            removed.append(item)
        else:
            kept.append(item)
    scene["placements"] = kept
    return removed


def _is_ceiling_light_item(item: dict[str, Any]) -> bool:
    text = item_text(item)
    category = norm(item.get("category"))
    semantic = norm(item.get("semantic_group"))
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    candidate = meta.get("supplier_candidate") if isinstance(meta.get("supplier_candidate"), dict) else {}
    cand_semantic = norm(candidate.get("semantic_group"))
    cand_category = norm(candidate.get("category_norm"))
    return (
        semantic == "lamp_ceiling"
        or cand_semantic == "lamp_ceiling"
        or "ceilinglightfactory" in category
        or "ceiling light" in text
        or "chandelier" in text
        or "люстр" in text
        or ("потолоч" in text and ("светиль" in text or "light" in text or "lamp" in text))
        or cand_category in {"chandelier", "ceiling_lamp", "pendant_lamp", "recessed_spot_track_light"}
    )


def _is_small_sanitary_clutter_item(item: dict[str, Any]) -> bool:
    if classify_item(item) & {"toilet", "sink", "bath", "shower", "bath_or_shower"}:
        return False
    if _is_ceiling_light_item(item):
        return False
    text = item_text(item)
    return any(
        token in text
        for token in (
            "bookstack",
            "bookcase",
            "bookshelf",
            "largeshelffactory",
            "simplebookcasefactory",
            "natureshelftrinkets",
            "trinket",
            "bowlfactory",
            "beziercurve",
            "béziercurve",
            "стеллаж",
            "книж",
            "декор",
        )
    )


def _room_ceiling_height(room: dict[str, Any]) -> float:
    try:
        return float(room.get("ceiling_height_m") or room.get("ceiling_height") or 2.8)
    except Exception:
        return 2.8


def _ceiling_coverage_centers(room: dict[str, Any], count: int) -> list[tuple[float, float]]:
    width, depth = room_bounds(room)
    poly = _room_polygon_xy(room)
    box = _polygon_bbox(poly)
    centroid = _polygon_centroid(poly)
    count = max(1, int(count))
    margin = min(0.55, max(0.18, min(width, depth) * 0.22))
    if count == 1:
        return [centroid]
    centers: list[tuple[float, float]] = []
    if width >= depth:
        usable = max(0.01, (box["x_max"] - box["x_min"]) - margin * 2.0)
        for idx in range(count):
            t = (idx + 1) / (count + 1)
            point = (box["x_min"] + margin + usable * t, centroid[1])
            centers.append(point if _point_in_polygon_xy(point, poly, eps=0.05) else centroid)
    else:
        usable = max(0.01, (box["y_max"] - box["y_min"]) - margin * 2.0)
        for idx in range(count):
            t = (idx + 1) / (count + 1)
            point = (centroid[0], box["y_min"] + margin + usable * t)
            centers.append(point if _point_in_polygon_xy(point, poly, eps=0.05) else centroid)
    return centers


def _max_ceiling_light_count(room: dict[str, Any], *, is_sanitary: bool) -> int:
    if is_sanitary:
        return 1
    width, depth = room_bounds(room)
    area = max(_polygon_area(_room_polygon_xy(room)), max(0.0, width * depth) * 0.6)
    if area <= 8.0:
        return 1
    if area <= 25.0:
        return 2
    if area <= 45.0:
        return 3
    return 4


def _set_ceiling_light_geometry(item: dict[str, Any], room: dict[str, Any], center_xy: tuple[float, float], size_xy: tuple[float, float] | None = None) -> None:
    aabb = item.get("aabb") if isinstance(item.get("aabb"), dict) else {}
    if size_xy is None:
        sx = max(0.12, float(aabb.get("x_max", 0.0)) - float(aabb.get("x_min", 0.0)) if aabb else 0.28)
        sy = max(0.12, float(aabb.get("y_max", 0.0)) - float(aabb.get("y_min", 0.0)) if aabb else 0.28)
    else:
        sx, sy = size_xy
    sz = max(0.035, min(0.16, float(aabb.get("z_max", 0.0)) - float(aabb.get("z_min", 0.0)) if aabb else 0.055))
    z_max = _room_ceiling_height(room) - 0.01
    _set_item_geometry(
        item,
        center_xy=center_xy,
        size=(sx, sy, sz),
        yaw_deg=0.0,
        z_min=z_max - sz,
        role="flat_ceiling_light",
    )
    item["category"] = "CeilingLightFactory"
    item["semantic_group"] = "lamp_ceiling"
    item["constraints"] = {"mount_type": "ceiling", "under_ceiling": True}
    item.setdefault("meta", {})["ceiling_light_position_repaired"] = True


def make_flat_ceiling_light_item(room_id: str, index: int, room: dict[str, Any], center_xy: tuple[float, float]) -> dict[str, Any]:
    width, depth = room_bounds(room)
    diameter = min(0.32, max(0.20, min(width, depth) * 0.18))
    height = 0.045
    z_max = _room_ceiling_height(room) - 0.012
    z_min = z_max - height
    cx, cy = center_xy
    return {
        "id": f"req_flat_ceiling_light_{index:02d}",
        "name": "Плоский потолочный светильник",
        "category": "CeilingLightFactory",
        "semantic_group": "lamp_ceiling",
        "position_m": [round(cx, 4), round(cy, 4), round(z_min + height / 2.0, 4)],
        "size_m": [round(diameter, 4), round(diameter, 4), round(height, 4)],
        "yaw_deg": 0.0,
        "rotation_deg": 0.0,
        "yaw_rad": 0.0,
        "aabb": {
            "x_min": round(cx - diameter / 2.0, 4),
            "x_max": round(cx + diameter / 2.0, 4),
            "y_min": round(cy - diameter / 2.0, 4),
            "y_max": round(cy + diameter / 2.0, 4),
            "z_min": round(z_min, 4),
            "z_max": round(z_max, 4),
        },
        "constraints": {"mount_type": "ceiling", "under_ceiling": True},
        "asset": {"kind": "procedural_flat_ceiling_light", "mesh_fit_mode": "exact"},
        "source": {
            "placement_source": "requirement_postprocess",
            "asset_source": "procedural_lighting",
            "supplier_replaced": False,
            "supplier_target_id": f"req_flat_ceiling_light_{index:02d}",
            "placeholder_bbox": False,
            "room_id": room_id,
        },
        "meta": {
            "placeholder_bbox": False,
            "procedural_lighting": True,
            "required_role": "flat_ceiling_light",
            "room_id": room_id,
            "ceiling_light_position_repaired": True,
            "sanitary_flat_light": True,
        },
    }


def repair_ceiling_lighting_layouts(scene_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    repairs: list[dict[str, Any]] = []
    for entry in scene_entries:
        scene = entry.get("scene") if isinstance(entry.get("scene"), dict) else {}
        if not scene:
            continue
        room = scene.get("room") if isinstance(scene.get("room"), dict) else {}
        room_id = str(room.get("id") or entry.get("room_id") or "room")
        text = _room_type_text(scene, _sanitary_entry_prompt(entry))
        is_sanitary = any(token in text for token in ("bathroom", "toilet", "сануз", "ванн", "туалет"))
        ceiling_lights = [item for item in room_items(scene) if _is_ceiling_light_item(item)]
        if is_sanitary:
            removed = _remove_items(scene, lambda item: _is_ceiling_light_item(item))
            if removed or not any(_is_ceiling_light_item(item) for item in room_items(scene)):
                center = _ceiling_coverage_centers(room, 1)[0]
                item = make_flat_ceiling_light_item(room_id, len(room_items(scene)) + 1, room, center)
                scene.setdefault("placements", []).append(item)
                entry.setdefault("added", []).append(item)
                repairs.append(
                    {
                        "room_id": room_id,
                        "action": "replaced_sanitary_chandelier_with_flat_light",
                        "removed_ids": [str(x.get("id") or "") for x in removed],
                        "added_id": item["id"],
                        "center_xy": [round(center[0], 4), round(center[1], 4)],
                    }
                )
            continue

        if not ceiling_lights:
            continue
        max_count = _max_ceiling_light_count(room, is_sanitary=False)
        if len(ceiling_lights) > max_count:
            keep_ids = {str(item.get("id") or "") for item in ceiling_lights[:max_count]}
            removed = _remove_items(scene, lambda item: _is_ceiling_light_item(item) and str(item.get("id") or "") not in keep_ids)
            repairs.append(
                {
                    "room_id": room_id,
                    "action": "removed_excess_ceiling_lights",
                    "removed_ids": [str(x.get("id") or "") for x in removed],
                    "kept_count": max_count,
                }
            )
            ceiling_lights = [item for item in room_items(scene) if _is_ceiling_light_item(item)]
        centers = _ceiling_coverage_centers(room, len(ceiling_lights))
        for item, center in zip(ceiling_lights, centers):
            old_pos = item.get("position_m")
            _set_ceiling_light_geometry(item, room, center)
            repairs.append(
                {
                    "room_id": room_id,
                    "action": "normalized_ceiling_light_position",
                    "id": item.get("id"),
                    "old_position_m": old_pos,
                    "new_position_m": item.get("position_m"),
                }
            )
        scene.setdefault("meta", {}).setdefault("requirement_postprocess", {})["ceiling_lighting_repaired"] = True
    return repairs


def _is_table_lamp_item(item: dict[str, Any]) -> bool:
    text = item_text(item)
    semantic = norm(item.get("semantic_group"))
    category = norm(item.get("category"))
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    candidate = meta.get("supplier_candidate") if isinstance(meta.get("supplier_candidate"), dict) else {}
    return (
        semantic == "lamp_table"
        or norm(candidate.get("semantic_group")) == "lamp_table"
        or "desklampfactory" in category
        or "lamp_table" in text
        or "desk lamp" in text
        or "table lamp" in text
        or "настоль" in text
    )


def _is_support_surface_item(item: dict[str, Any]) -> bool:
    if _is_ceiling_light_item(item) or _is_table_lamp_item(item) or _is_desktop_device_item(item):
        return False
    text = item_text(item)
    roles = classify_item(item)
    return "table" in roles or any(
        token in text
        for token in (
            "side_table",
            "nightstand",
            "bookcase",
            "bookshelf",
            "largeshelffactory",
            "simplebookcasefactory",
            "shelf",
            "console",
            "cabinet",
            "dresser",
            "стеллаж",
            "полк",
            "комод",
            "тумб",
            "шкаф",
        )
    )


def _support_surfaces(scene: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, float]]]:
    surfaces: list[tuple[dict[str, Any], dict[str, float]]] = []
    for item in room_items(scene):
        if not _is_support_surface_item(item):
            continue
        aabb = _item_aabb(item)
        if not aabb:
            continue
        width = aabb["x_max"] - aabb["x_min"]
        depth = aabb["y_max"] - aabb["y_min"]
        if width < 0.22 or depth < 0.18:
            continue
        if aabb["z_max"] < 0.22 or aabb["z_max"] > 1.75:
            continue
        surfaces.append((item, aabb))
    return surfaces


def _center_on_support(
    support: dict[str, float],
    item_size: tuple[float, float],
    old_center: tuple[float, float],
    *,
    keep_old_when_inside: bool = True,
) -> tuple[float, float]:
    sx, sy = item_size
    margin = 0.04
    min_x = support["x_min"] + sx / 2.0 + margin
    max_x = support["x_max"] - sx / 2.0 - margin
    min_y = support["y_min"] + sy / 2.0 + margin
    max_y = support["y_max"] - sy / 2.0 - margin
    center_x = (support["x_min"] + support["x_max"]) * 0.5
    center_y = (support["y_min"] + support["y_max"]) * 0.5
    if min_x > max_x:
        min_x = max_x = center_x
    if min_y > max_y:
        min_y = max_y = center_y
    if keep_old_when_inside and min_x <= old_center[0] <= max_x and min_y <= old_center[1] <= max_y:
        return old_center
    return (min(max(center_x, min_x), max_x), min(max(center_y, min_y), max_y))


def _nearest_support_for_item(
    scene: dict[str, Any],
    item: dict[str, Any],
    aabb: dict[str, float],
    *,
    prefer_tables: bool = False,
) -> dict[str, float] | None:
    cx, cy = _aabb_center_xy(aabb)
    best: tuple[float, dict[str, float]] | None = None
    for support_item, support in _support_surfaces(scene):
        if support_item is item:
            continue
        overlap_x = support["x_min"] - 0.18 <= cx <= support["x_max"] + 0.18
        overlap_y = support["y_min"] - 0.18 <= cy <= support["y_max"] + 0.18
        dx = 0.0 if overlap_x else min(abs(cx - support["x_min"]), abs(cx - support["x_max"]))
        dy = 0.0 if overlap_y else min(abs(cy - support["y_min"]), abs(cy - support["y_max"]))
        xy_dist = math.hypot(dx, dy)
        if xy_dist > 1.10:
            continue
        z_gap = abs(aabb["z_min"] - support["z_max"])
        score = xy_dist + z_gap * 0.55
        if prefer_tables and "table" in classify_item(support_item):
            score -= 0.30
        if best is None or score < best[0]:
            best = (score, support)
    return best[1] if best is not None else None


def repair_table_lamp_sizes(scene_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    repairs: list[dict[str, Any]] = []
    for entry in scene_entries:
        scene = entry.get("scene") if isinstance(entry.get("scene"), dict) else {}
        if not scene:
            continue
        room = scene.get("room") if isinstance(scene.get("room"), dict) else {}
        room_id = str(room.get("id") or entry.get("room_id") or "room")
        room_repaired = False
        for item in room_items(scene):
            if not _is_table_lamp_item(item):
                continue
            aabb = _item_aabb(item)
            if not aabb:
                continue
            sx = aabb["x_max"] - aabb["x_min"]
            sy = aabb["y_max"] - aabb["y_min"]
            sz = aabb["z_max"] - aabb["z_min"]
            new_sx = min(max(sx, 0.34), 0.46)
            new_sy = min(max(sy, 0.34), 0.46)
            new_sz = min(max(sz, 0.52), 0.68)
            cx, cy = _aabb_center_xy(aabb)
            support = _nearest_support_for_item(scene, item, aabb, prefer_tables=True)
            target_center = (cx, cy)
            target_z = aabb["z_min"]
            if support is not None:
                target_center = _center_on_support(support, (new_sx, new_sy), (cx, cy), keep_old_when_inside=True)
                target_z = support["z_max"] + 0.004
            changed = (
                abs(new_sx - sx) >= 1e-4
                or abs(new_sy - sy) >= 1e-4
                or abs(new_sz - sz) >= 1e-4
                or abs(target_center[0] - cx) >= 1e-4
                or abs(target_center[1] - cy) >= 1e-4
                or abs(target_z - aabb["z_min"]) >= 0.015
            )
            if not changed:
                continue
            old_size = item.get("size_m")
            old_pos = item.get("position_m")
            _set_item_geometry(
                item,
                center_xy=target_center,
                size=(new_sx, new_sy, new_sz),
                yaw_deg=float(item.get("yaw_deg", item.get("rotation_deg", 0.0)) or 0.0),
                z_min=target_z,
                role=None,
            )
            item["semantic_group"] = "lamp_table"
            item.setdefault("meta", {})["table_lamp_size_repaired"] = True
            if support is not None:
                item.setdefault("meta", {})["support_surface_reanchored"] = True
            room_repaired = True
            repairs.append(
                {
                    "room_id": room_id,
                    "action": "repaired_table_lamp_on_support",
                    "id": item.get("id"),
                    "old_size_m": old_size,
                    "new_size_m": item.get("size_m"),
                    "old_position_m": old_pos,
                    "new_position_m": item.get("position_m"),
                }
            )
        if room_repaired:
            scene.setdefault("meta", {}).setdefault("requirement_postprocess", {})["table_lamp_sizes_repaired"] = True
    return repairs


def _is_desktop_support_item(item: dict[str, Any]) -> bool:
    if "table" not in classify_item(item):
        return False
    text = item_text(item)
    return any(token in text for token in ("desk", "simpledeskfactory", "dining_table", "стол", "table"))


def _is_desktop_device_item(item: dict[str, Any]) -> bool:
    text = item_text(item)
    return any(token in text for token in ("imac", "monitor", "keyboard", "mouse", "computer", "laptop", "клавиат", "мыш", "монитор", "компьют"))


def repair_desktop_support_items(scene_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    repairs: list[dict[str, Any]] = []
    for entry in scene_entries:
        scene = entry.get("scene") if isinstance(entry.get("scene"), dict) else {}
        if not scene:
            continue
        room = scene.get("room") if isinstance(scene.get("room"), dict) else {}
        room_id = str(room.get("id") or entry.get("room_id") or "room")
        supports = [(item, _item_aabb(item)) for item in room_items(scene) if _is_desktop_support_item(item)]
        supports = [(item, aabb) for item, aabb in supports if aabb]
        for item in room_items(scene):
            if not _is_desktop_device_item(item):
                continue
            aabb = _item_aabb(item)
            if not aabb:
                continue
            cx, cy = _aabb_center_xy(aabb)
            support = None
            for support_item, support_aabb in supports:
                if support_item is item:
                    continue
                if support_aabb["x_min"] - 0.12 <= cx <= support_aabb["x_max"] + 0.12 and support_aabb["y_min"] - 0.12 <= cy <= support_aabb["y_max"] + 0.12:
                    support = support_aabb
                    break
            if support is None:
                continue
            sx = aabb["x_max"] - aabb["x_min"]
            sy = aabb["y_max"] - aabb["y_min"]
            sz = aabb["z_max"] - aabb["z_min"]
            target_center = _center_on_support(
                support,
                (sx, sy),
                (cx, cy),
                keep_old_when_inside=False,
            )
            target_z = support["z_max"] + 0.004
            if (
                abs(aabb["z_min"] - target_z) < 0.015
                and abs(target_center[0] - cx) < 0.025
                and abs(target_center[1] - cy) < 0.025
            ):
                continue
            old_pos = item.get("position_m")
            _set_item_geometry(
                item,
                center_xy=target_center,
                size=(sx, sy, sz),
                yaw_deg=float(item.get("yaw_deg", item.get("rotation_deg", 0.0)) or 0.0),
                z_min=target_z,
                role=None,
            )
            item.setdefault("meta", {})["desktop_support_reanchored"] = True
            repairs.append(
                {
                    "room_id": room_id,
                    "action": "reanchored_desktop_device_to_support",
                    "id": item.get("id"),
                    "old_position_m": old_pos,
                    "new_position_m": item.get("position_m"),
                    "old_z_min": aabb["z_min"],
                    "new_z_min": target_z,
                }
            )
    return repairs


def _is_support_decor_item(item: dict[str, Any]) -> bool:
    if _is_table_lamp_item(item) or _is_desktop_device_item(item) or _is_ceiling_light_item(item):
        return False
    if classify_item(item) & {"toilet", "sink", "bath", "shower", "bath_or_shower", "bed", "table", "chair"}:
        return False
    text = item_text(item)
    return any(
        token in text
        for token in (
            "plant",
            "plantcontainerfactory",
            "vase",
            "trinket",
            "decor",
            "горш",
            "растен",
            "ваза",
            "декор",
        )
    )


def repair_support_decor_items(scene_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    repairs: list[dict[str, Any]] = []
    for entry in scene_entries:
        scene = entry.get("scene") if isinstance(entry.get("scene"), dict) else {}
        if not scene:
            continue
        room = scene.get("room") if isinstance(scene.get("room"), dict) else {}
        room_id = str(room.get("id") or entry.get("room_id") or "room")
        for item in room_items(scene):
            if not _is_support_decor_item(item):
                continue
            aabb = _item_aabb(item)
            if not aabb:
                continue
            sx = aabb["x_max"] - aabb["x_min"]
            sy = aabb["y_max"] - aabb["y_min"]
            sz = aabb["z_max"] - aabb["z_min"]
            text = item_text(item)
            large_floor_decor = (
                any(token in text for token in ("plant", "plantcontainerfactory", "растен", "горш"))
                and (max(sx, sy) > 0.45 or sz > 0.75)
            )
            if large_floor_decor:
                if aabb["z_min"] <= 0.06:
                    continue
                old_pos = item.get("position_m")
                _set_item_geometry(
                    item,
                    center_xy=_aabb_center_xy(aabb),
                    size=(sx, sy, sz),
                    yaw_deg=float(item.get("yaw_deg", item.get("rotation_deg", 0.0)) or 0.0),
                    z_min=0.0,
                    role=None,
                )
                item.setdefault("meta", {})["support_surface_reanchored"] = "large_floor_decor"
                repairs.append(
                    {
                        "room_id": room_id,
                        "action": "snapped_large_decor_to_floor",
                        "id": item.get("id"),
                        "old_position_m": old_pos,
                        "new_position_m": item.get("position_m"),
                    }
                )
                continue
            support = _nearest_support_for_item(scene, item, aabb, prefer_tables=False)
            if support is None:
                if aabb["z_min"] <= 0.08:
                    continue
                old_pos = item.get("position_m")
                _set_item_geometry(
                    item,
                    center_xy=_aabb_center_xy(aabb),
                    size=(sx, sy, sz),
                    yaw_deg=float(item.get("yaw_deg", item.get("rotation_deg", 0.0)) or 0.0),
                    z_min=0.0,
                    role=None,
                )
                item.setdefault("meta", {})["support_surface_reanchored"] = "floor_fallback"
                repairs.append(
                    {
                        "room_id": room_id,
                        "action": "snapped_decor_to_floor",
                        "id": item.get("id"),
                        "old_position_m": old_pos,
                        "new_position_m": item.get("position_m"),
                    }
                )
                continue
            target_center = _center_on_support(support, (sx, sy), _aabb_center_xy(aabb), keep_old_when_inside=True)
            target_z = support["z_max"] + 0.004
            if (
                abs(target_z - aabb["z_min"]) < 0.012
                and abs(target_center[0] - _aabb_center_xy(aabb)[0]) < 0.02
                and abs(target_center[1] - _aabb_center_xy(aabb)[1]) < 0.02
            ):
                continue
            old_pos = item.get("position_m")
            _set_item_geometry(
                item,
                center_xy=target_center,
                size=(sx, sy, sz),
                yaw_deg=float(item.get("yaw_deg", item.get("rotation_deg", 0.0)) or 0.0),
                z_min=target_z,
                role=None,
            )
            item.setdefault("meta", {})["support_surface_reanchored"] = True
            repairs.append(
                {
                    "room_id": room_id,
                    "action": "reanchored_decor_to_support",
                    "id": item.get("id"),
                    "old_position_m": old_pos,
                    "new_position_m": item.get("position_m"),
                }
            )
    return repairs


def _is_bed_item(item: dict[str, Any]) -> bool:
    return "bed" in classify_item(item)


def repair_bed_layouts(scene_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    repairs: list[dict[str, Any]] = []
    for entry in scene_entries:
        scene = entry.get("scene") if isinstance(entry.get("scene"), dict) else {}
        if not scene:
            continue
        room = scene.get("room") if isinstance(scene.get("room"), dict) else {}
        room_text = _room_type_text(scene, _sanitary_entry_prompt(entry))
        if not any(token in room_text for token in ("bedroom", "спаль", "studio", "студ")):
            continue
        room_id = str(room.get("id") or entry.get("room_id") or "room")
        width, depth = room_bounds(room)
        for item in room_items(scene):
            if not _is_bed_item(item):
                continue
            aabb = _item_aabb(item)
            if not aabb:
                continue
            sx = aabb["x_max"] - aabb["x_min"]
            sy = aabb["y_max"] - aabb["y_min"]
            sz = aabb["z_max"] - aabb["z_min"]
            margin = 0.12
            old_center = _aabb_center_xy(aabb)
            door_zones = _opening_clearance_zones(room, ("doors",), reach=0.85, pad=0.22)
            window_zones = _opening_clearance_zones(room, ("windows",), reach=0.55, pad=0.22)
            wall_candidates = _wall_mount_candidates(
                room,
                (sx, sy),
                margin=margin,
                min_wall_len=max(0.8, min(sx, sy) * 0.65),
            )
            best = _choose_best_floor_candidate(
                room,
                wall_candidates,
                door_zones=door_zones,
                window_zones=window_zones,
                prefer_xy=_polygon_centroid(_room_polygon_xy(room)),
                keep_near_xy=old_center,
            )
            if best is not None:
                cx, cy = best["center"]
                normal = best["wall"]["normal"]
                yaw = (math.degrees(math.atan2(-float(normal[1]), -float(normal[0]))) + 360.0) % 360.0
                side = f"wall_{best['wall']['index']}"
            else:
                distances = {
                    "left": aabb["x_min"],
                    "right": width - aabb["x_max"],
                    "bottom": aabb["y_min"],
                    "top": depth - aabb["y_max"],
                }
                side = min(distances, key=distances.get)
                cx, cy = old_center
                if side == "bottom":
                    cy = margin + sy / 2.0
                    yaw = 270.0
                elif side == "top":
                    cy = depth - margin - sy / 2.0
                    yaw = 90.0
                elif side == "left":
                    cx = margin + sx / 2.0
                    yaw = 180.0
                else:
                    cx = width - margin - sx / 2.0
                    yaw = 0.0
                cx, cy = _clamp_center((cx, cy), (sx, sy), (width, depth), margin)
            old_pos = item.get("position_m")
            old_yaw = item.get("yaw_deg")
            if abs(float(old_yaw or 0.0) - yaw) < 1e-4 and old_pos and abs(float(old_pos[0]) - cx) < 1e-4 and abs(float(old_pos[1]) - cy) < 1e-4:
                continue
            _set_item_geometry(item, center_xy=(cx, cy), size=(sx, sy, sz), yaw_deg=yaw, z_min=aabb["z_min"], role=None)
            item["semantic_group"] = "bed"
            item.setdefault("meta", {})["bed_headboard_repaired"] = True
            item.setdefault("meta", {})["bed_anchor_wall"] = side
            repairs.append(
                {
                    "room_id": room_id,
                    "action": "reoriented_bed_headboard_to_wall",
                    "id": item.get("id"),
                    "anchor_wall": side,
                    "old_position_m": old_pos,
                    "new_position_m": item.get("position_m"),
                    "old_yaw_deg": old_yaw,
                    "new_yaw_deg": yaw,
                }
            )
    return repairs


def repair_sanitary_layouts(
    scene_entries: list[dict[str, Any]],
    asset_search_roots: tuple[Path, ...] = (),
) -> list[dict[str, Any]]:
    repairs: list[dict[str, Any]] = []
    for entry in scene_entries:
        scene = entry.get("scene") if isinstance(entry.get("scene"), dict) else {}
        if not scene or not _is_sanitary_scene(scene, _sanitary_entry_prompt(entry)):
            continue
        room = scene.get("room") if isinstance(scene.get("room"), dict) else {}
        room_id = str(room.get("id") or entry.get("room_id") or "room")
        text = _room_type_text(scene, _sanitary_entry_prompt(entry))
        stretch_info = _ensure_min_sanitary_room_extent(scene, _sanitary_entry_prompt(entry))
        if stretch_info is not None:
            repairs.append({"room_id": room_id, "action": "stretched_small_sanitary_room", **stretch_info})
        width, depth = room_bounds(room)
        margin = 0.12

        clutter_removed = _remove_items(scene, _is_small_sanitary_clutter_item)
        for item in clutter_removed:
            repairs.append({"room_id": room_id, "action": "removed_small_sanitary_clutter", "id": item.get("id"), "name": item.get("name")})

        removed = _remove_items(
            scene,
            lambda item: (
                ("toilet" in text or "туалет" in text)
                and ("toilet" in classify_item(item))
                and bool((item.get("meta") or {}).get("supplier_requirement_added"))
            )
            or (
                ("shower" in classify_item(item))
                and (
                    bool((item.get("meta") or {}).get("supplier_requirement_added"))
                    or _candidate_discouraged_for_role((item.get("meta") or {}).get("supplier_candidate") or {}, "shower")
                    or "ag01090" in item_text(item)
                )
            ),
        )
        for item in removed:
            repairs.append({"room_id": room_id, "action": "removed_bad_sanitary_item", "id": item.get("id"), "name": item.get("name")})

        is_toilet_only_room = "toilet" in text or "туалет" in text
        is_bathroom_room = ("bathroom" in text or "ванн" in text or "сануз" in text) and not is_toilet_only_room

        if is_toilet_only_room:
            target_size = (0.48, 0.72, 0.8)
            candidate = select_catalog_candidate("toilet", target_size, scene, _sanitary_entry_prompt(entry), asset_search_roots)
            if candidate is not None and "toilet" not in _sanitary_roles_present(scene):
                sx, sy, sz = _fit_catalog_size_to_room("toilet", candidate, target_size, (width, depth))
                sx, sy, sz = max(0.44, sx), max(0.68, sy), max(0.76, sz)
                center, yaw = _sanitary_layout("toilet", (sx, sy), room, margin, obstacles=_room_floor_obstacles(scene, max_z_min=1.4))
                item = make_supplier_required_item(
                    room_id=room_id,
                    role="toilet",
                    index=len(room_items(scene)) + 1,
                    center_xy=center,
                    size=(sx, sy, sz),
                    yaw_deg=yaw,
                    z_min=0.0,
                    candidate=candidate,
                )
                scene.setdefault("placements", []).append(item)
                entry.setdefault("added", []).append(item)
                repairs.append({"room_id": room_id, "action": "replaced_toilet_layout", "id": item["id"], "center_xy": list(center), "yaw_deg": yaw})

        sink_items = [item for item in room_items(scene) if "sink" in classify_item(item)]
        if sink_items:
            sink = sink_items[0]
            sink_size, sink_z_min = _sink_size_and_z_for_room(room, sink)
            center, yaw = _sanitary_layout(
                "sink",
                (sink_size[0], sink_size[1]),
                room,
                margin,
                obstacles=_room_floor_obstacles(scene, exclude_ids={str(sink.get("id") or "")}, max_z_min=1.4),
            )
            _set_item_geometry(
                sink,
                center_xy=center,
                size=sink_size,
                yaw_deg=yaw,
                z_min=sink_z_min,
                role="sink",
            )
            repairs.append({"room_id": room_id, "action": "reanchored_sink_to_wall", "id": sink.get("id"), "center_xy": list(center), "yaw_deg": yaw})

        if is_bathroom_room:
            if not ({"bath", "shower", "bath_or_shower"} & _sanitary_roles_present(scene)):
                target_role = _actual_sanitary_role("bath_or_shower", room, _sanitary_entry_prompt(entry))
                target_size = (1.55, 0.74, 0.58) if target_role == "bath" else (0.86, 0.86, 1.95)
                candidate = select_catalog_candidate(target_role, target_size, scene, _sanitary_entry_prompt(entry), asset_search_roots)
                if candidate is None and target_role == "bath":
                    target_role = "shower"
                    target_size = (0.86, 0.86, 1.95)
                    candidate = select_catalog_candidate(target_role, target_size, scene, _sanitary_entry_prompt(entry), asset_search_roots)
                if candidate is not None:
                    sx, sy, sz = _fit_catalog_size_to_room(target_role, candidate, target_size, (width, depth))
                    center, yaw = _sanitary_layout(target_role, (sx, sy), room, margin, obstacles=_room_floor_obstacles(scene, max_z_min=1.4))
                    item = make_supplier_required_item(
                        room_id=room_id,
                        role=target_role,
                        index=len(room_items(scene)) + 1,
                        center_xy=center,
                        size=(sx, sy, sz),
                        yaw_deg=yaw,
                        z_min=0.0,
                        candidate=candidate,
                    )
                    scene.setdefault("placements", []).append(item)
                    entry.setdefault("added", []).append(item)
                    repairs.append({"room_id": room_id, "action": "replaced_bath_or_shower_with_catalog_asset", "role": target_role, "id": item["id"], "center_xy": list(center), "yaw_deg": yaw, "candidate": candidate.get("unique_key")})

        if repairs:
            scene.setdefault("meta", {}).setdefault("requirement_postprocess", {})["sanitary_layout_repaired"] = True
    return repairs


def _room_type_text(scene: dict[str, Any], prompt_room_type: str | None = None) -> str:
    room = scene.get("room") if isinstance(scene.get("room"), dict) else {}
    return norm(" ".join([room.get("room_type") or "", room.get("source_room_type") or "", prompt_room_type or ""]))


def _is_sanitary_scene(scene: dict[str, Any], prompt_room_type: str | None = None) -> bool:
    return any(x in _room_type_text(scene, prompt_room_type) for x in ("bathroom", "toilet", "сануз", "ванн", "туалет"))


def _ensure_min_sanitary_room_extent(scene: dict[str, Any], prompt_room_type: str | None = None) -> dict[str, Any] | None:
    room = scene.get("room") if isinstance(scene.get("room"), dict) else {}
    text = _room_type_text(scene, prompt_room_type)
    if not room or not _is_sanitary_scene(scene, prompt_room_type):
        return None
    toilet_only = ("toilet" in text or "туалет" in text) and not any(token in text for token in ("bathroom", "ванн", "сануз"))
    if toilet_only:
        return None
    width, depth = room_bounds(room)
    if width <= 0 or depth <= 0:
        return None
    min_short = 1.45
    min_long = 2.05
    min_area = 2.85
    if min(width, depth) >= min_short and max(width, depth) >= min_long and width * depth >= min_area:
        return None

    if width <= depth:
        new_width = max(width, min_short)
        new_depth = max(depth, min_long, min_area / max(new_width, 1e-6))
    else:
        new_depth = max(depth, min_short)
        new_width = max(width, min_long, min_area / max(new_depth, 1e-6))

    poly = _room_polygon_xy(room)
    bbox = _polygon_bbox(poly)
    old_w = max(bbox["x_max"] - bbox["x_min"], width, 1e-6)
    old_d = max(bbox["y_max"] - bbox["y_min"], depth, 1e-6)
    sx = new_width / old_w
    sy = new_depth / old_d
    raw_poly = room.get("floor_polygon")
    if isinstance(raw_poly, list) and raw_poly:
        for point in raw_poly:
            if not isinstance(point, dict):
                continue
            try:
                x = float(point.get("x", 0.0))
                y_key = "y" if "y" in point else "z"
                y = float(point.get(y_key, 0.0))
            except Exception:
                continue
            point["x"] = round(bbox["x_min"] + (x - bbox["x_min"]) * sx, 6)
            point[y_key] = round(bbox["y_min"] + (y - bbox["y_min"]) * sy, 6)
    room["width_m"] = round(new_width, 4)
    room["depth_m"] = round(new_depth, 4)
    room["area_m2"] = round(new_width * new_depth, 4)
    info = {
        "old_size_m": [round(width, 4), round(depth, 4)],
        "new_size_m": [round(new_width, 4), round(new_depth, 4)],
        "reason": "sanitary_room_too_small_for_toilet_sink_shower",
    }
    room.setdefault("meta", {}).setdefault("requirement_postprocess", {})["sanitary_room_stretched"] = info
    scene.setdefault("meta", {}).setdefault("requirement_postprocess", {})["sanitary_room_stretched"] = info
    return info


def _sanitary_roles_present(scene: dict[str, Any]) -> set[str]:
    present: set[str] = set()
    for item in room_items(scene):
        present |= classify_item(item)
    return present


def _actual_sanitary_role(role: str, room: dict[str, Any] | None = None, prompt_room_type: str | None = None) -> str:
    if role != "bath_or_shower":
        return role
    room = room if isinstance(room, dict) else {}
    width, depth = room_bounds(room)
    text = _room_type_text({"room": room}, prompt_room_type)
    if ("bathroom" in text or "ванн" in text or "сануз" in text) and min(width, depth) >= 1.55 and max(width, depth) >= 1.65:
        return "bath"
    return "shower"


def _sanitary_role_fits_room(role: str, room: dict[str, Any], prompt_room_type: str | None = None) -> bool:
    actual_role = _actual_sanitary_role(role, room, prompt_room_type)
    if actual_role not in {"bath", "shower"}:
        return True
    width, depth = room_bounds(room)
    area = max(0.0, width * depth)
    text = _room_type_text({"room": room}, prompt_room_type)
    toilet_only = ("toilet" in text or "туалет" in text) and not ("bathroom" in text or "ванн" in text or "сануз" in text)
    if toilet_only and area < 2.15:
        return False
    if actual_role == "bath":
        return min(width, depth) >= 1.05 and max(width, depth) >= 1.55 and area >= 2.6
    return min(width, depth) >= 1.05 and area >= 2.15


def add_sanitary_roles_to_room(
    scene: dict[str, Any],
    roles: list[str],
    prompt_room_type: str | None = None,
    asset_search_roots: tuple[Path, ...] = (),
) -> list[dict[str, Any]]:
    room = scene.get("room") if isinstance(scene.get("room"), dict) else {}
    room_id = str(room.get("id") or "room")
    if not roles or not _is_sanitary_scene(scene, prompt_room_type):
        return []

    width, depth = room_bounds(room)
    margin = 0.12
    specs = {
        "toilet": ((0.48, 0.72, 0.8), 0.0, 0.0),
        "sink": ((0.40, 0.62, 0.22), 180.0, 0.72),
        "shower": ((0.86, 0.86, 1.95), 0.0, 0.0),
        "bath": ((1.55, 0.74, 0.58), 0.0, 0.0),
    }
    added: list[dict[str, Any]] = []
    for role in roles:
        actual_role = _actual_sanitary_role(role, room, prompt_room_type)
        if not _sanitary_role_fits_room(role, room, prompt_room_type):
            scene.setdefault("meta", {}).setdefault("requirement_postprocess", {}).setdefault("skipped_sanitary_roles", []).append(
                {"role": role, "actual_role": actual_role, "reason": "room_too_small_for_collision_free_fixture"}
            )
            continue
        target_size, yaw, z_min = specs[actual_role]
        if actual_role == "sink":
            target_size, z_min = _sink_size_and_z_for_room(room)
        candidate = select_catalog_candidate(actual_role, target_size, scene, prompt_room_type, asset_search_roots)
        if candidate is None:
            _record_missing_catalog_asset(scene, actual_role)
            continue
        if actual_role == "sink" and _sink_candidate_is_vanity(candidate):
            z_min = 0.0
        sx, sy, sz = _fit_catalog_size_to_room(actual_role, candidate, target_size, (width, depth))
        placed_ids = {str(item.get("id") or "") for item in added}
        obstacles = _room_floor_obstacles(scene, exclude_ids=placed_ids, max_z_min=1.4)
        obstacles.extend([_item_aabb(item) for item in added if _item_aabb(item)])
        (cx, cy), layout_yaw = _sanitary_layout(actual_role, (sx, sy), room, margin, obstacles=obstacles)
        added.append(
            make_supplier_required_item(
                room_id=room_id,
                role=actual_role,
                index=len(room_items(scene)) + len(added) + 1,
                center_xy=(cx, cy),
                size=(sx, sy, sz),
                yaw_deg=layout_yaw if layout_yaw is not None else yaw,
                z_min=z_min,
                candidate=candidate,
            )
        )
    scene.setdefault("placements", []).extend(added)
    meta = scene.setdefault("meta", {})
    req = meta.setdefault("requirement_postprocess", {})
    req.setdefault("added_sanitary", []).extend(x["id"] for x in added)
    return added


def add_missing_sanitary(
    scene: dict[str, Any],
    prompt_room_type: str | None = None,
    asset_search_roots: tuple[Path, ...] = (),
) -> list[dict[str, Any]]:
    if not _is_sanitary_scene(scene, prompt_room_type):
        return []
    present = _sanitary_roles_present(scene)
    missing = [role for role in SANITARY_REQUIRED if role not in present]
    return add_sanitary_roles_to_room(scene, missing, prompt_room_type, asset_search_roots)


def add_missing_sanitary_per_room(
    scene_entries: list[dict[str, Any]],
    asset_search_roots: tuple[Path, ...] = (),
) -> list[dict[str, Any]]:
    added: list[dict[str, Any]] = []
    for entry in scene_entries:
        scene = entry.get("scene") if isinstance(entry.get("scene"), dict) else {}
        if not scene or not _is_sanitary_scene(scene, _sanitary_entry_prompt(entry)):
            continue
        item_added = add_missing_sanitary(
            scene,
            prompt_room_type=_sanitary_entry_prompt(entry),
            asset_search_roots=asset_search_roots,
        )
        if not item_added:
            continue
        entry.setdefault("added", []).extend(item_added)
        added.extend(item_added)
        meta = scene.setdefault("meta", {})
        meta.setdefault("requirement_postprocess", {})["sanitary_scope"] = "each_sanitary_room"
    return added


def _sanitary_entry_prompt(entry: dict[str, Any]) -> str | None:
    room_meta = entry.get("room_meta") if isinstance(entry.get("room_meta"), dict) else {}
    value = room_meta.get("prompt_room_type")
    return str(value) if value is not None else None


def _sanitary_target_score(entry: dict[str, Any], role: str) -> tuple[float, float]:
    scene = entry.get("scene") if isinstance(entry.get("scene"), dict) else {}
    room = scene.get("room") if isinstance(scene.get("room"), dict) else {}
    text = _room_type_text(scene, _sanitary_entry_prompt(entry))
    width, depth = room_bounds(room)
    area = float(room.get("area_m2") or (width * depth))
    present = _sanitary_roles_present(scene)
    actual_role = _actual_sanitary_role(role, room, _sanitary_entry_prompt(entry))
    has_bath_or_shower = bool({"bath", "shower", "bath_or_shower"} & present)
    if actual_role == "shower":
        preference = 0.0 if ("bathroom" in text or "ванн" in text) else 2.0
        if "toilet" in text or "туалет" in text:
            preference += 4.0
        if has_bath_or_shower:
            preference += 3.0
        return preference, -area
    if actual_role == "toilet":
        preference = 0.0 if ("toilet" in text or "туалет" in text or "сануз" in text) else 1.0
        if has_bath_or_shower:
            preference += 1.5
        return preference, area
    if actual_role == "sink":
        preference = 0.0 if "toilet" in present else 1.0
        if "sink" in present:
            preference += 4.0
        return preference, area
    return 1.0, -area


def _select_sanitary_target_entry(entries: list[dict[str, Any]], role: str) -> dict[str, Any] | None:
    candidates = [
        entry
        for entry in entries
        if isinstance(entry.get("scene"), dict) and _is_sanitary_scene(entry["scene"], _sanitary_entry_prompt(entry))
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda entry: _sanitary_target_score(entry, role))[0]


def add_missing_sanitary_apartment(
    scene_entries: list[dict[str, Any]],
    asset_search_roots: tuple[Path, ...] = (),
) -> list[dict[str, Any]]:
    sanitary_entries = [
        entry
        for entry in scene_entries
        if isinstance(entry.get("scene"), dict) and _is_sanitary_scene(entry["scene"], _sanitary_entry_prompt(entry))
    ]
    if not sanitary_entries:
        return []
    present: set[str] = set()
    for entry in sanitary_entries:
        present |= _sanitary_roles_present(entry["scene"])
    missing = [role for role in SANITARY_REQUIRED if role not in present]
    added: list[dict[str, Any]] = []
    for role in missing:
        target = _select_sanitary_target_entry(sanitary_entries, role)
        if target is None:
            continue
        item_added = add_sanitary_roles_to_room(
            target["scene"],
            [role],
            prompt_room_type=_sanitary_entry_prompt(target),
            asset_search_roots=asset_search_roots,
        )
        target.setdefault("added", []).extend(item_added)
        added.extend(item_added)
    for entry in sanitary_entries:
        scene = entry["scene"]
        meta = scene.setdefault("meta", {})
        meta.setdefault("requirement_postprocess", {})["sanitary_scope"] = "apartment"
    return added


def _item_aabb(item: dict[str, Any]) -> dict[str, float] | None:
    aabb = item.get("aabb") if isinstance(item.get("aabb"), dict) else {}
    try:
        if {"x_min", "x_max", "y_min", "y_max", "z_min", "z_max"} <= set(aabb):
            return {k: float(aabb[k]) for k in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")}
    except Exception:
        pass
    pos = item.get("position_m") if isinstance(item.get("position_m"), list) else None
    size = item.get("size_m") if isinstance(item.get("size_m"), list) else None
    if not pos or not size or len(pos) < 3 or len(size) < 3:
        return None
    try:
        cx, cy, cz = float(pos[0]), float(pos[1]), float(pos[2])
        sx, sy, sz = float(size[0]), float(size[1]), float(size[2])
    except Exception:
        return None
    return {
        "x_min": cx - sx / 2.0,
        "x_max": cx + sx / 2.0,
        "y_min": cy - sy / 2.0,
        "y_max": cy + sy / 2.0,
        "z_min": cz - sz / 2.0,
        "z_max": cz + sz / 2.0,
    }


def _aabb_center_xy(aabb: dict[str, float]) -> tuple[float, float]:
    return (0.5 * (aabb["x_min"] + aabb["x_max"]), 0.5 * (aabb["y_min"] + aabb["y_max"]))


def _aabb_xy_area(aabb: dict[str, float]) -> float:
    return max(0.0, aabb["x_max"] - aabb["x_min"]) * max(0.0, aabb["y_max"] - aabb["y_min"])


def _aabb_xy_overlap_area(a: dict[str, float], b: dict[str, float], margin: float = 0.0) -> float:
    ix = max(0.0, min(a["x_max"], b["x_max"] + margin) - max(a["x_min"], b["x_min"] - margin))
    iy = max(0.0, min(a["y_max"], b["y_max"] + margin) - max(a["y_min"], b["y_min"] - margin))
    return ix * iy


def _aabb_xy_intersects(a: dict[str, float], b: dict[str, float], margin: float = 0.0) -> bool:
    return _aabb_xy_overlap_area(a, b, margin=margin) > 1e-6


def _candidate_floor_aabb(cx: float, cy: float, size: tuple[float, float, float]) -> dict[str, float]:
    sx, sy, sz = size
    return {
        "x_min": cx - sx / 2.0,
        "x_max": cx + sx / 2.0,
        "y_min": cy - sy / 2.0,
        "y_max": cy + sy / 2.0,
        "z_min": 0.0,
        "z_max": sz,
    }


def _room_door_clearance_zones(room: dict[str, Any], width: float, depth: float) -> list[dict[str, float]]:
    return _opening_clearance_zones(room, ("doors",), reach=0.72, pad=0.18)


def _is_kitchen_scene(scene: dict[str, Any], prompt_room_type: str | None = None) -> bool:
    return any(token in _room_type_text(scene, prompt_room_type) for token in ("kitchen", "кух"))


def _is_table_item(item: dict[str, Any]) -> bool:
    text = item_text(item)
    return "table" in classify_item(item) and "coffee_table" not in text and "side_table" not in text


def _is_chair_item(item: dict[str, Any]) -> bool:
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    return "chair" in classify_item(item) or str(meta.get("affordance") or "") == "table_chair"


def _yaw_towards_point(cx: float, cy: float, tx: float, ty: float) -> float:
    return (math.degrees(math.atan2(tx - cx, ty - cy)) + 360.0) % 360.0


def _chair_yaw_facing_table(cx: float, cy: float, tx: float, ty: float) -> float:
    # In our procedural chair and most imported dining-chair assets local +Y is
    # the back side. Offset by 180 deg so the seating/front side faces the table.
    return (_yaw_towards_point(cx, cy, tx, ty) + 180.0) % 360.0


def _set_item_yaw(item: dict[str, Any], yaw_deg: float) -> None:
    item["yaw_deg"] = round(yaw_deg, 4)
    item["rotation_deg"] = round(yaw_deg, 4)
    item["yaw_rad"] = round(math.radians(yaw_deg), 8)


def _is_procedural_kitchen_item(item: dict[str, Any]) -> bool:
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    asset = item.get("asset") if isinstance(item.get("asset"), dict) else {}
    text = norm(
        " ".join(
            str(x or "")
            for x in (
                item.get("category"),
                item.get("type"),
                item.get("assembly_type"),
                meta.get("procedural_assembly"),
                meta.get("assembly_type"),
                asset.get("kind"),
                asset.get("assembly_type"),
            )
        )
    )
    return "kitchen_set" in text or "procedural_kitchen" in text


def _module_key(module: dict[str, Any]) -> str:
    return norm(" ".join(str(module.get(k) or "") for k in ("id", "type", "role", "appliance")))


def _ensure_kitchen_microwave_decor(assembly: dict[str, Any]) -> bool:
    design = assembly.get("design_spec") if isinstance(assembly.get("design_spec"), dict) else {}
    functional = design.get("functional_requirements") if isinstance(design.get("functional_requirements"), dict) else {}
    if functional and not bool(functional.get("microwave")):
        return False

    decor = [x for x in (assembly.get("decor_items") or []) if isinstance(x, dict)]
    upper_modules = [x for x in (assembly.get("upper_modules") or []) if isinstance(x, dict)]
    upper_by_id = {str(x.get("id") or ""): x for x in upper_modules}
    microwaves = [x for x in decor if str(x.get("type") or "") == "microwave"]

    preferred = next((m for m in upper_modules if str(m.get("type") or "") == "microwave_open_shelf"), None)
    if preferred is None:
        preferred = next((m for m in upper_modules if "hood" not in _module_key(m)), None)

    changed = False
    if preferred is not None:
        target_id = str(preferred.get("id") or "")
        module_x = float(preferred.get("x_m") or 0.0)
        module_y = float(preferred.get("y_m") or 0.0)
        module_z = float(preferred.get("z_m") or 1.45)
        module_w = float(preferred.get("width_m") or 0.55)
        module_d = float(preferred.get("depth_m") or 0.32)
        orientation = str(preferred.get("orientation") or "x")
        target = microwaves[0] if microwaves else {"id": "decor_microwave_001", "type": "microwave"}
        before = deepcopy(target)
        target.update(
            {
                "type": "microwave",
                "x_m": round(module_x + module_w * 0.5, 4),
                "y_m": round(module_y + module_d * 0.5, 4),
                "z_m": round(module_z + 0.045, 4),
                "orientation": orientation,
                "upper_module_id": target_id,
                "placement": "upper_open_shelf",
                "shelf_width_m": round(module_w, 4),
                "shelf_depth_m": round(module_d, 4),
            }
        )
        if before != target:
            changed = True
        if not microwaves:
            decor.append(target)
            changed = True
    else:
        target = microwaves[0] if microwaves else {"id": "decor_microwave_001", "type": "microwave"}
        before = deepcopy(target)
        target.update(
            {
                "type": "microwave",
                "x_m": round(float(target.get("x_m") or 0.72), 4),
                "y_m": round(float(target.get("y_m") or 0.25), 4),
                "z_m": round(float(target.get("z_m") or 0.86), 4),
                "orientation": str(target.get("orientation") or "x"),
                "placement": "countertop",
            }
        )
        if "upper_module_id" in target:
            target.pop("upper_module_id", None)
        if before != target:
            changed = True
        if not microwaves:
            decor.append(target)
            changed = True

    valid_upper_ids = set(upper_by_id)
    filtered = []
    for item in decor:
        if str(item.get("type") or "") == "microwave":
            filtered.append(item)
            continue
        upper_id = str(item.get("upper_module_id") or "")
        if not upper_id or upper_id in valid_upper_ids:
            filtered.append(item)
        else:
            changed = True
    assembly["decor_items"] = filtered
    if changed:
        assembly.setdefault("warnings", []).append("postprocess:microwave_decor_preserved")
    return changed


def _compact_straight_kitchen_assembly(assembly: dict[str, Any], target_width: float, target_depth: float) -> None:
    base_modules = [deepcopy(x) for x in (assembly.get("base_modules") or []) if isinstance(x, dict)]
    if not base_modules:
        return

    def pick(predicate) -> dict[str, Any] | None:
        return next((m for m in base_modules if predicate(_module_key(m))), None)

    desired = [
        pick(lambda t: "fridge" in t),
        pick(lambda t: "sink" in t),
        pick(lambda t: "dishwasher" in t or "water_appliance" in t),
        pick(lambda t: "drawer" in t or "base_cabinet" in t),
        pick(lambda t: "oven" in t or "cooking" in t),
    ]
    kept = [m for m in desired if m is not None]
    if target_width < 2.82 and len(kept) > 4:
        kept = [m for m in kept if "dishwasher" not in _module_key(m) and "water_appliance" not in _module_key(m)]
    if not kept:
        kept = [m for m in base_modules if "filler" not in _module_key(m)][: max(1, min(5, len(base_modules)))]

    module_w = max(0.42, target_width / max(1, len(kept)))
    used_width = module_w * len(kept)
    for idx, module in enumerate(kept):
        module["x_m"] = round(idx * module_w, 4)
        module["y_m"] = 0.0
        module["width_m"] = round(module_w, 4)
        module["depth_m"] = round(min(max(target_depth, 0.54), 0.65), 4)
        module["orientation"] = "x"
        if "filler" in _module_key(module):
            module["width_m"] = round(max(0.08, module_w), 4)

    kept_ids = {str(m.get("id") or "") for m in kept}
    assembly["base_modules"] = kept
    assembly.setdefault("dimensions", {})["width_m"] = round(used_width, 4)
    assembly.setdefault("dimensions", {})["depth_m"] = round(target_depth, 4)

    upper_modules = []
    source_upper = {str(m.get("above_base_module_id") or ""): deepcopy(m) for m in (assembly.get("upper_modules") or []) if isinstance(m, dict)}
    for module in kept:
        module_id = str(module.get("id") or "")
        if module_id not in kept_ids or not module.get("has_upper_cabinet", True):
            continue
        key = _module_key(module)
        upper = source_upper.get(module_id) or {
            "id": f"upper_{len(upper_modules) + 1:03d}",
            "type": "wall_cabinet",
            "above_base_module_id": module_id,
            "above": module.get("role"),
        }
        if "oven" in key or "cooking" in key:
            upper["type"] = "hood_wall_mounted"
            upper["height_m"] = min(float(upper.get("height_m") or 0.36), 0.42)
        upper["x_m"] = module["x_m"]
        upper["y_m"] = 0.0
        upper["z_m"] = float(upper.get("z_m") or 1.458)
        upper["width_m"] = module["width_m"]
        upper["depth_m"] = min(float(upper.get("depth_m") or 0.32), 0.34)
        upper["orientation"] = "x"
        upper_modules.append(upper)
    assembly["upper_modules"] = upper_modules

    counter_bases = [m for m in kept if bool(m.get("has_countertop", True))]
    if counter_bases:
        start_x = min(float(m.get("x_m") or 0.0) for m in counter_bases)
        end_x = max(float(m.get("x_m") or 0.0) + float(m.get("width_m") or module_w) for m in counter_bases)
        cutouts = []
        for module in counter_bases:
            key = _module_key(module)
            rel_x = float(module.get("x_m") or 0.0) - start_x
            mw = float(module.get("width_m") or module_w)
            if "sink" in key:
                cutouts.append({"type": "sink", "module_id": module.get("id"), "shape": "rounded_rect", "x_m": round(rel_x + mw * 0.08, 4), "y_m": 0.14, "width_m": round(min(0.48, mw * 0.82), 4), "depth_m": 0.38})
            if "oven" in key or "cooking" in key:
                cutouts.append({"type": "cooktop", "module_id": module.get("id"), "shape": "rect", "x_m": round(rel_x + mw * 0.06, 4), "y_m": 0.055, "width_m": round(min(0.52, mw * 0.88), 4), "depth_m": 0.46})
        assembly["countertop_segments"] = [
            {
                "id": "countertop_001",
                "x_m": round(start_x, 4),
                "y_m": 0.0,
                "z_m": 0.82,
                "width_m": round(max(0.1, end_x - start_x), 4),
                "depth_m": min(0.6, target_depth),
                "thickness_m": 0.038,
                "orientation": "x",
                "cutouts": cutouts,
            }
        ]
        assembly["backsplash_segments"] = [
            {
                "id": "backsplash_001",
                "x_m": round(start_x, 4),
                "y_m": 0.0,
                "z_m": 0.858,
                "width_m": round(max(0.1, end_x - start_x), 4),
                "height_m": 0.6,
                "thickness_m": 0.004,
                "orientation": "x",
            }
        ]
    assembly["decor_items"] = [
        item
        for item in (assembly.get("decor_items") or [])
        if isinstance(item, dict) and (not item.get("upper_module_id") or str(item.get("upper_module_id") or "") in {str(m.get("id") or "") for m in upper_modules})
    ]
    _ensure_kitchen_microwave_decor(assembly)


def repair_kitchen_layout_for_scene(scene: dict[str, Any], prompt_room_type: str | None = None) -> list[dict[str, Any]]:
    if not _is_kitchen_scene(scene, prompt_room_type):
        return []
    room = scene.get("room") if isinstance(scene.get("room"), dict) else {}
    repairs: list[dict[str, Any]] = []
    for item in room_items(scene):
        if not _is_procedural_kitchen_item(item):
            continue
        aabb = _item_aabb(item)
        if not aabb:
            continue
        width = max(0.1, float((item.get("meta") or {}).get("dimensions", {}).get("width_m") or (aabb["x_max"] - aabb["x_min"])))
        depth = max(0.45, min(0.68, float((item.get("meta") or {}).get("dimensions", {}).get("depth_m") or (aabb["y_max"] - aabb["y_min"]))))
        height = max(1.8, min(2.35, float((item.get("meta") or {}).get("dimensions", {}).get("height_m") or (aabb["z_max"] - aabb["z_min"]))))
        choice = _choose_window_safe_kitchen_wall(room, width, depth, height)
        if choice is None:
            continue
        old_aabb = deepcopy(item.get("aabb"))
        old_yaw = item.get("yaw_deg")
        target_width = float(choice["target_width"])
        meta = item.setdefault("meta", {})
        _compact_straight_kitchen_assembly(meta, target_width, depth)
        used_width = float((meta.get("dimensions") or {}).get("width_m") or target_width)
        used_depth = float((meta.get("dimensions") or {}).get("depth_m") or depth)
        new_aabb = _rotated_rect_aabb(choice["origin"], choice["wall"]["tangent"], choice["wall"]["normal"], used_width, used_depth, 0.0, height)
        item["position"] = [round(choice["origin"][0], 4), round(choice["origin"][1], 4), 0.0]
        item["position_m"] = [
            round((new_aabb["x_min"] + new_aabb["x_max"]) * 0.5, 4),
            round((new_aabb["y_min"] + new_aabb["y_max"]) * 0.5, 4),
            round(height * 0.5, 4),
        ]
        item["aabb"] = new_aabb
        item["size_m"] = [
            round(new_aabb["x_max"] - new_aabb["x_min"], 4),
            round(new_aabb["y_max"] - new_aabb["y_min"], 4),
            round(height, 4),
        ]
        _set_item_yaw(item, float(choice["yaw_deg"]))
        meta["kitchen_window_safe_wall_repaired"] = True
        meta["kitchen_anchor_wall_index"] = int(choice["wall"].get("index") or 0)
        meta["position"] = [0.0, 0.0, 0.0]
        source = item.setdefault("source", {})
        source["kitchen_window_safe_wall_repaired"] = True
        removed = _remove_items(
            scene,
            lambda other: (
                isinstance(other.get("meta"), dict)
                and other["meta"].get("required_role") in {"table", "chair"}
                and (other.get("source") or {}).get("placement_source") == "requirement_postprocess"
            ),
        )
        repairs.append(
            {
                "room_id": room.get("id"),
                "action": "moved_kitchen_to_window_safe_wall",
                "id": item.get("id"),
                "old_aabb": old_aabb,
                "new_aabb": item.get("aabb"),
                "old_yaw_deg": old_yaw,
                "new_yaw_deg": item.get("yaw_deg"),
                "removed_dependent_requirement_items": [str(x.get("id") or "") for x in removed],
            }
        )
    if repairs:
        scene.setdefault("meta", {}).setdefault("requirement_postprocess", {})["kitchen_layout_repairs"] = repairs
    return repairs


def _valid_kitchen_chair_aabb(
    aabb: dict[str, float],
    *,
    room: dict[str, Any],
    room_size: tuple[float, float],
    obstacles: list[dict[str, float]],
    occupied: list[dict[str, float]],
    door_zones: list[dict[str, float]],
    window_zones: list[dict[str, float]],
) -> bool:
    width, depth = room_size
    if aabb["x_min"] < 0.06 or aabb["y_min"] < 0.06 or aabb["x_max"] > width - 0.06 or aabb["y_max"] > depth - 0.06:
        return False
    if not _aabb_inside_room_polygon(aabb, room, margin=0.025):
        return False
    for zone in [*door_zones, *window_zones]:
        if _aabb_xy_intersects(aabb, zone):
            return False
    for obstacle in obstacles:
        if _aabb_xy_intersects(aabb, obstacle, margin=0.02):
            return False
    for other in occupied:
        if _aabb_xy_intersects(aabb, other, margin=0.04):
            return False
    return True


def add_kitchen_table_seating(
    scene: dict[str, Any],
    prompt_room_type: str | None = None,
    asset_search_roots: tuple[Path, ...] = (),
) -> list[dict[str, Any]]:
    if not _is_kitchen_scene(scene, prompt_room_type):
        return []
    room = scene.get("room") if isinstance(scene.get("room"), dict) else {}
    room_id = str(room.get("id") or "room")
    width, depth = room_bounds(room)
    tables = [item for item in room_items(scene) if _is_table_item(item)]
    if not tables:
        return []
    table = max(tables, key=lambda item: _aabb_xy_area(_item_aabb(item) or {"x_min": 0.0, "x_max": 0.0, "y_min": 0.0, "y_max": 0.0, "z_min": 0.0, "z_max": 0.0}))
    table_aabb = _item_aabb(table)
    if not table_aabb:
        return []
    tx, ty = _aabb_center_xy(table_aabb)

    existing_chairs = [item for item in room_items(scene) if _is_chair_item(item)]
    for chair in existing_chairs:
        chair_aabb = _item_aabb(chair)
        if not chair_aabb:
            continue
        cx, cy = _aabb_center_xy(chair_aabb)
        _set_item_yaw(chair, _chair_yaw_facing_table(cx, cy, tx, ty))
        chair.setdefault("meta", {})["chair_orientation_repaired"] = True

    target_count = 2
    if len(existing_chairs) >= target_count:
        scene.setdefault("meta", {}).setdefault("requirement_postprocess", {})["oriented_kitchen_table_chairs"] = [str(x.get("id") or "") for x in existing_chairs]
        return []

    candidate = select_catalog_candidate("chair", (0.52, 0.58, 0.82), scene, prompt_room_type, asset_search_roots)
    if candidate is None:
        _record_missing_catalog_asset(scene, "chair")
        return []
    sx, sy, sz = _fit_catalog_size_to_room("chair", candidate, (0.52, 0.58, 0.82), (width, depth))
    obstacles: list[dict[str, float]] = []
    for item in room_items(scene):
        if item is table or _is_chair_item(item) or _is_ceiling_light_item(item):
            continue
        aabb = _item_aabb(item)
        if not aabb or aabb.get("z_min", 0.0) > 1.25:
            continue
        obstacles.append(aabb)
    door_zones = _room_door_clearance_zones(room, width, depth)
    window_zones = _opening_clearance_zones(room, ("windows",), reach=0.45, pad=0.18)
    occupied = [_item_aabb(item) for item in existing_chairs if _item_aabb(item)]

    gap = 0.14
    candidates = [
        (tx, table_aabb["y_min"] - sy / 2.0 - gap),
        (table_aabb["x_max"] + sx / 2.0 + gap, ty),
        (table_aabb["x_min"] - sx / 2.0 - gap, ty),
        (tx, table_aabb["y_max"] + sy / 2.0 + gap),
    ]
    radial_offset = max(sx, sy) * 0.75 + max(table_aabb["x_max"] - table_aabb["x_min"], table_aabb["y_max"] - table_aabb["y_min"]) * 0.55
    for angle in (45.0, 135.0, 225.0, 315.0):
        rad = math.radians(angle)
        candidates.append((tx + math.cos(rad) * radial_offset, ty + math.sin(rad) * radial_offset))
    added: list[dict[str, Any]] = []
    for cx, cy in candidates:
        if len(existing_chairs) + len(added) >= target_count:
            break
        yaw = _chair_yaw_facing_table(cx, cy, tx, ty)
        aabb = _candidate_floor_aabb(cx, cy, (sx, sy, sz))
        if not _valid_kitchen_chair_aabb(
            aabb,
            room=room,
            room_size=(width, depth),
            obstacles=obstacles,
            occupied=occupied,
            door_zones=door_zones,
            window_zones=window_zones,
        ):
            continue
        item = make_supplier_required_item(
            room_id=room_id,
            role="chair",
            index=len(room_items(scene)) + len(added) + 1,
            center_xy=(cx, cy),
            size=(sx, sy, sz),
            yaw_deg=yaw,
            z_min=0.0,
            candidate=candidate,
        )
        meta = item.setdefault("meta", {})
        meta.update(
            {
                "affordance": "table_chair",
                "target_table_id": table.get("id"),
                "support_group": "kitchen_dining",
                "chair_orientation_repaired": True,
                "chair_faces_table_center_xy": [round(tx, 4), round(ty, 4)],
            }
        )
        scene.setdefault("placements", []).append(item)
        added.append(item)
        occupied.append(aabb)

    if added:
        scene.setdefault("meta", {}).setdefault("requirement_postprocess", {})["added_kitchen_table_chairs"] = [x["id"] for x in added]
    return added


def add_missing_kitchen_table(
    scene: dict[str, Any],
    prompt_room_type: str | None = None,
    asset_search_roots: tuple[Path, ...] = (),
) -> list[dict[str, Any]]:
    room = scene.get("room") if isinstance(scene.get("room"), dict) else {}
    room_id = str(room.get("id") or "room")
    room_type_text = norm(" ".join([room.get("room_type") or "", room.get("source_room_type") or "", prompt_room_type or ""]))
    if "kitchen" not in room_type_text and "кух" not in room_type_text:
        return []

    present: set[str] = set()
    for item in room_items(scene):
        present |= classify_item(item)
    if "table" in present:
        return []

    width, depth = room_bounds(room)
    target_size = (min(1.25, max(0.75, width * 0.36)), min(0.78, max(0.55, depth * 0.24)), 0.76)
    candidate = select_catalog_candidate("table", target_size, scene, prompt_room_type, asset_search_roots)
    if candidate is None:
        _record_missing_catalog_asset(scene, "table")
        return []
    sx, sy, sz = _fit_catalog_size_to_room("table", candidate, target_size, (width, depth))
    obstacles = _room_floor_obstacles(scene, max_z_min=1.25)
    door_zones = _opening_clearance_zones(room, ("doors",), reach=0.85, pad=0.22)
    window_zones = _opening_clearance_zones(room, ("windows",), reach=0.35, pad=0.18)
    poly = _room_polygon_xy(room)
    box = _polygon_bbox(poly)
    centroid = _polygon_centroid(poly)
    grid_candidates: list[dict[str, Any]] = []
    nx = max(3, min(7, int(width / 0.55) + 2))
    ny = max(3, min(7, int(depth / 0.55) + 2))
    min_x = box["x_min"] + sx / 2.0 + 0.18
    max_x = box["x_max"] - sx / 2.0 - 0.18
    min_y = box["y_min"] + sy / 2.0 + 0.18
    max_y = box["y_max"] - sy / 2.0 - 0.18
    if max_x >= min_x and max_y >= min_y:
        for ix in range(nx):
            for iy in range(ny):
                x = min_x + (max_x - min_x) * (ix + 0.5) / nx
                y = min_y + (max_y - min_y) * (iy + 0.5) / ny
                aabb = _candidate_floor_aabb(x, y, (sx, sy, sz))
                grid_candidates.append({"center": (x, y), "yaw": 0.0, "aabb": aabb})
    best = _choose_best_floor_candidate(
        room,
        grid_candidates,
        obstacles=obstacles,
        door_zones=door_zones,
        window_zones=window_zones,
        prefer_xy=centroid,
    )
    if best is not None:
        cx, cy = best["center"]
        yaw = float(best["yaw"])
    else:
        cx = min(max(width * 0.62, sx / 2.0 + 0.18), max(sx / 2.0 + 0.18, width - sx / 2.0 - 0.18))
        cy = min(max(depth * 0.66, sy / 2.0 + 0.18), max(sy / 2.0 + 0.18, depth - sy / 2.0 - 0.18))
        yaw = 0.0
    item = make_supplier_required_item(
        room_id=room_id,
        role="table",
        index=len(room_items(scene)) + 1,
        center_xy=(cx, cy),
        size=(sx, sy, sz),
        yaw_deg=yaw,
        z_min=0.0,
        candidate=candidate,
    )
    scene.setdefault("placements", []).append(item)
    meta = scene.setdefault("meta", {})
    meta.setdefault("requirement_postprocess", {})["added_kitchen_table"] = [item["id"]]
    return [item]


def add_apartment_required_objects(
    apartment_scenes: list[dict[str, Any]],
    asset_search_roots: tuple[Path, ...] = (),
) -> list[dict[str, Any]]:
    present: set[str] = set()
    for scene in apartment_scenes:
        for item in room_items(scene):
            present |= classify_item(item)
    added: list[dict[str, Any]] = []
    if "bed" not in present:
        def bed_target_score(scene: dict[str, Any]) -> tuple[int, float]:
            room = scene.get("room") if isinstance(scene.get("room"), dict) else {}
            text = _room_type_text(scene, None)
            priority = 0 if any(token in text for token in ("bedroom", "спаль", "studio", "студ")) else 1
            return priority, -float(room.get("area_m2") or 0.0)

        target = min(apartment_scenes, key=bed_target_score)
        room = target.get("room") or {}
        width, depth = room_bounds(room)
        target_size = (
            min(1.8, max(1.2, width - 0.4)),
            min(2.1, max(1.6, depth - 0.4)),
            0.65,
        )
        candidate = select_catalog_candidate("bed", target_size, target, None, asset_search_roots)
        if candidate is None:
            _record_missing_catalog_asset(target, "bed")
        else:
            sx, sy, sz = _fit_catalog_size_to_room("bed", candidate, target_size, (width, depth))
            wall_candidates = _wall_mount_candidates(room, (sx, sy), margin=0.12, min_wall_len=max(0.8, min(sx, sy) * 0.65))
            best = _choose_best_floor_candidate(
                room,
                wall_candidates,
                obstacles=_room_floor_obstacles(target, max_z_min=1.4),
                door_zones=_opening_clearance_zones(room, ("doors",), reach=0.85, pad=0.22),
                window_zones=_opening_clearance_zones(room, ("windows",), reach=0.55, pad=0.22),
                prefer_xy=_polygon_centroid(_room_polygon_xy(room)),
            )
            if best is not None:
                cx, cy = best["center"]
                normal = best["wall"]["normal"]
                yaw = (math.degrees(math.atan2(-float(normal[1]), -float(normal[0]))) + 360.0) % 360.0
            else:
                cx, cy = (min(width - 1.0, max(1.0, width * 0.5)), min(depth - 0.75, max(0.75, depth * 0.5)))
                yaw = 0.0
            item = make_supplier_required_item(
                room_id=str(room.get("id") or "room"),
                role="bed",
                index=len(room_items(target)) + 1,
                center_xy=(cx, cy),
                size=(sx, sy, sz),
                yaw_deg=yaw,
                z_min=0.0,
                candidate=candidate,
            )
            item.setdefault("meta", {})["bed_headboard_repaired"] = True
            target.setdefault("placements", []).append(item)
            added.append(item)
    if "table" not in present:
        candidates = sorted(
            apartment_scenes,
            key=lambda s: 0
            if norm((s.get("room") or {}).get("room_type")) in {"kitchen", "living_room", "bedroom"}
            else 1,
        )
        target = candidates[0]
        room = target.get("room") or {}
        width, depth = room_bounds(room)
        target_size = (1.2, 0.7, 0.75)
        candidate = select_catalog_candidate("table", target_size, target, None, asset_search_roots)
        if candidate is None:
            _record_missing_catalog_asset(target, "table")
        else:
            sx, sy, sz = _fit_catalog_size_to_room("table", candidate, target_size, (width, depth))
            poly = _room_polygon_xy(room)
            box = _polygon_bbox(poly)
            grid_candidates: list[dict[str, Any]] = []
            for ix in range(5):
                for iy in range(5):
                    cx = box["x_min"] + (box["x_max"] - box["x_min"]) * (ix + 0.5) / 5.0
                    cy = box["y_min"] + (box["y_max"] - box["y_min"]) * (iy + 0.5) / 5.0
                    grid_candidates.append({"center": (cx, cy), "yaw": 0.0, "aabb": _candidate_floor_aabb(cx, cy, (sx, sy, sz))})
            best = _choose_best_floor_candidate(
                room,
                grid_candidates,
                obstacles=_room_floor_obstacles(target, max_z_min=1.25),
                door_zones=_opening_clearance_zones(room, ("doors",), reach=0.85, pad=0.22),
                window_zones=_opening_clearance_zones(room, ("windows",), reach=0.35, pad=0.18),
                prefer_xy=_polygon_centroid(poly),
            )
            center_xy = best["center"] if best is not None else (max(0.6, width * 0.5), max(0.45, depth * 0.5))
            item = make_supplier_required_item(
                room_id=str(room.get("id") or "room"),
                role="table",
                index=len(room_items(target)) + 1,
                center_xy=center_xy,
                size=(sx, sy, sz),
                yaw_deg=0.0,
                z_min=0.0,
                candidate=candidate,
            )
            target.setdefault("placements", []).append(item)
            added.append(item)
    return added


def inverse_room_frame(point: tuple[float, float], frame: dict[str, Any]) -> tuple[float, float]:
    off = frame.get("offset_xy") or [0.0, 0.0]
    origin = frame.get("origin_xy") or [0.0, 0.0]
    angle = float(frame.get("rotation_rad") or 0.0)
    x = point[0] - float(off[0])
    y = point[1] - float(off[1])
    return (
        x * math.cos(angle) - y * math.sin(angle) + float(origin[0]),
        x * math.sin(angle) + y * math.cos(angle) + float(origin[1]),
    )


def estimate_apartment_min(apartment: dict[str, Any], room_jsons: dict[str, Path]) -> tuple[float, float]:
    door_graph = (((apartment.get("room") or {}).get("meta") or {}).get("door_graph") or {})
    graph_doors = door_graph.get("doors") or []
    estimates: list[tuple[float, float]] = []
    for door in graph_doors:
        room_id = str(door.get("to") or "")
        center = door.get("center_xy")
        if room_id not in room_jsons or not isinstance(center, list) or len(center) < 2:
            continue
        room = read_json(room_jsons[room_id]).get("room") or {}
        frame = ((room.get("meta") or {}).get("coordinate_frame") or {})
        doors = room.get("doors") or []
        if not frame or not doors:
            continue
        seg = (doors[0] or {}).get("segment") or {}
        if not {"x1", "x2", "y1", "y2"} <= set(seg):
            continue
        local_center = ((float(seg["x1"]) + float(seg["x2"])) / 2.0, (float(seg["y1"]) + float(seg["y2"])) / 2.0)
        gx, gy = inverse_room_frame(local_center, frame)
        estimates.append((gx - float(center[0]), gy - float(center[1])))
    if estimates:
        return (
            sorted(x for x, _ in estimates)[len(estimates) // 2],
            sorted(y for _, y in estimates)[len(estimates) // 2],
        )
    poly = (apartment.get("room") or {}).get("floor_polygon") or []
    return (min(float(p.get("x", 0.0)) for p in poly), min(float(p.get("y", 0.0)) for p in poly)) if poly else (0.0, 0.0)


def transform_item_to_apartment(item: dict[str, Any], frame: dict[str, Any], apt_min: tuple[float, float], room_id: str) -> dict[str, Any]:
    out = deepcopy(item)
    prefix = f"{room_id}__"
    out["id"] = prefix + str(out.get("id") or "item")
    source = out.setdefault("source", {})
    source["source_room_id"] = room_id
    meta = out.setdefault("meta", {})
    meta["source_room_id"] = room_id
    angle_deg = float(frame.get("rotation_deg") or math.degrees(float(frame.get("rotation_rad") or 0.0)))

    aabb = item.get("aabb") or {}
    corners = [
        (float(aabb.get("x_min", 0.0)), float(aabb.get("y_min", 0.0))),
        (float(aabb.get("x_min", 0.0)), float(aabb.get("y_max", 0.0))),
        (float(aabb.get("x_max", 0.0)), float(aabb.get("y_min", 0.0))),
        (float(aabb.get("x_max", 0.0)), float(aabb.get("y_max", 0.0))),
    ]
    apt_pts = []
    for pt in corners:
        gx, gy = inverse_room_frame(pt, frame)
        apt_pts.append((gx - apt_min[0], gy - apt_min[1]))
    xs = [p[0] for p in apt_pts]
    ys = [p[1] for p in apt_pts]
    out["aabb"] = {
        "x_min": round(min(xs), 4),
        "x_max": round(max(xs), 4),
        "y_min": round(min(ys), 4),
        "y_max": round(max(ys), 4),
        "z_min": float(aabb.get("z_min", 0.0)),
        "z_max": float(aabb.get("z_max", 0.0)),
    }
    pos = item.get("position_m") or [
        (float(aabb.get("x_min", 0.0)) + float(aabb.get("x_max", 0.0))) / 2.0,
        (float(aabb.get("y_min", 0.0)) + float(aabb.get("y_max", 0.0))) / 2.0,
        (float(aabb.get("z_min", 0.0)) + float(aabb.get("z_max", 0.0))) / 2.0,
    ]
    gx, gy = inverse_room_frame((float(pos[0]), float(pos[1])), frame)
    out["position_m"] = [round(gx - apt_min[0], 4), round(gy - apt_min[1], 4), float(pos[2])]
    raw_root_pos = item.get("position")
    if isinstance(raw_root_pos, list) and len(raw_root_pos) >= 2:
        root_gx, root_gy = inverse_room_frame((float(raw_root_pos[0]), float(raw_root_pos[1])), frame)
        root_z = float(raw_root_pos[2]) if len(raw_root_pos) >= 3 else 0.0
        out["position"] = [round(root_gx - apt_min[0], 4), round(root_gy - apt_min[1], 4), root_z]
    out["size_m"] = [
        round(out["aabb"]["x_max"] - out["aabb"]["x_min"], 4),
        round(out["aabb"]["y_max"] - out["aabb"]["y_min"], 4),
        round(out["aabb"]["z_max"] - out["aabb"]["z_min"], 4),
    ]
    yaw = float(item.get("yaw_deg", item.get("rotation_deg", 0.0)) or 0.0) + angle_deg
    out["yaw_deg"] = round(yaw, 4)
    out["rotation_deg"] = round(yaw, 4)
    return out


def should_skip_apartment_item(item: dict[str, Any]) -> bool:
    name = norm(item.get("name"))
    return name.startswith("room_floor_supplieroverlay") or name.startswith("room_wallpaper_supplieroverlay")


def _storage_role_from_item(item: dict[str, Any]) -> str | None:
    text = item_text(item)
    category = norm(item.get("category"))
    if any(token in text for token in ("bookcase", "bookshelf", "стеллаж", "полк")) or "shelffactory" in category or "bookcasefactory" in category:
        return "shelf"
    if any(token in text for token in ("wardrobe", "closet", "шкаф", "гардероб")):
        return "wardrobe"
    if any(token in text for token in ("dresser", "sideboard", "комод", "тумб", "cabinet")) or "singlecabinetfactory" in category:
        return "dresser"
    return None


def _local_storage_candidate_role(path: Path) -> str | None:
    text = norm(" ".join(path.parts[-7:]))
    if any(token in text for token in ("стеллаж", "bookcase", "shelf")):
        return "shelf"
    if any(token in text for token in ("шкаф", "wardrobe", "closet", "гардероб")):
        return "wardrobe"
    if any(token in text for token in ("комод", "тумб", "dresser", "sideboard", "cabinet")):
        return "dresser"
    return None


def _model_family_key(path: Path) -> str:
    stem = norm(path.stem)
    match = re.search(r"\d{4,}", stem)
    if match:
        return match.group(0)
    stem = re.sub(r"(?i)(?:model|corona|vray|v-ray|export|fbx|obj|gltf|glb|_)+", " ", stem)
    return re.sub(r"\s+", "_", stem).strip("_") or norm(path.stem)


def _candidate_from_local_storage_asset(path: Path, role: str) -> dict[str, Any]:
    family = _model_family_key(path)
    title_map = {"shelf": "стеллаж", "wardrobe": "шкаф", "dresser": "комод"}
    category_map = {"shelf": "bookcase", "wardrobe": "wardrobe", "dresser": "sideboard"}
    return {
        "unique_key": f"local_apt_storage::{role}::{family}",
        "source_site": "local_apt_supplier_assets",
        "title": f"{title_map.get(role, role)} {family}",
        "category_raw": title_map.get(role, role),
        "category_norm": category_map.get(role, role),
        "semantic_group": role,
        "asset_status": "local_supplier_asset",
        "asset_format": path.suffix.lower().lstrip("."),
        "asset_local_path": str(path.resolve()),
        "description": f"Local apartment supplier asset discovered for {role}: {path.name}",
    }


def _discover_local_storage_candidates(apt_dir: Path) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {"shelf": [], "wardrobe": [], "dresser": []}
    seen: set[str] = set()
    for path in sorted(apt_dir.glob("rooms/*/pipeline/*/supplier_assets/**/*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_MESH_SUFFIXES:
            continue
        role = _local_storage_candidate_role(path)
        if role is None:
            continue
        key = f"{role}:{_model_family_key(path)}"
        # Prefer a single import file per visual model; OBJ is usually lighter,
        # FBX is kept when it is the only available representation.
        if key in seen and path.suffix.lower() != ".obj":
            continue
        if key in seen and path.suffix.lower() == ".obj":
            out[role] = [c for c in out[role] if c.get("unique_key") != f"local_apt_storage::{role}::{_model_family_key(path)}"]
        seen.add(key)
        out[role].append(_candidate_from_local_storage_asset(path, role))
    return out


def repair_storage_supplier_candidate_pools(scene_entries: list[dict[str, Any]], apt_dir: Path) -> list[dict[str, Any]]:
    local_candidates = _discover_local_storage_candidates(apt_dir)
    reports: list[dict[str, Any]] = []
    duplicate_groups: dict[str, list[tuple[dict[str, Any], str]]] = {}
    for entry in scene_entries:
        scene = entry.get("scene") if isinstance(entry.get("scene"), dict) else {}
        for item in room_items(scene):
            role = _storage_role_from_item(item)
            if role is None:
                continue
            meta = item.setdefault("meta", {})
            candidate = meta.get("supplier_candidate") if isinstance(meta.get("supplier_candidate"), dict) else {}
            key = str(candidate.get("unique_key") or "")
            if key:
                duplicate_groups.setdefault(key, []).append((item, role))

    item_offsets: dict[str, int] = {}
    for group_items in duplicate_groups.values():
        if len(group_items) <= 1:
            continue
        for idx, (item, _role) in enumerate(group_items):
            item_offsets[str(item.get("id") or "")] = idx

    for entry in scene_entries:
        scene = entry.get("scene") if isinstance(entry.get("scene"), dict) else {}
        room_id = str((scene.get("room") or {}).get("id") or entry.get("room_id") or "")
        for item in room_items(scene):
            role = _storage_role_from_item(item)
            if role is None:
                continue
            meta = item.setdefault("meta", {})
            primary = meta.get("supplier_candidate") if isinstance(meta.get("supplier_candidate"), dict) else None
            if not primary:
                continue
            existing_pool = [c for c in (meta.get("supplier_candidate_pool") or []) if isinstance(c, dict)]
            existing_keys = {str(c.get("unique_key") or "") for c in existing_pool}
            primary_path = Path(str(primary.get("asset_local_path") or "")) if primary.get("asset_local_path") else None
            primary_family = _model_family_key(primary_path) if primary_path is not None else ""
            alternatives = [
                deepcopy(c)
                for c in local_candidates.get(role, [])
                if str(c.get("unique_key") or "") not in existing_keys
                and (not primary_family or not str(c.get("unique_key") or "").endswith(f"::{primary_family}"))
            ]
            if alternatives:
                offset = item_offsets.get(str(item.get("id") or ""), 0) % len(alternatives)
                alternatives = alternatives[offset:] + alternatives[:offset]
                meta["supplier_candidate_pool"] = existing_pool + [_compact_candidate(c) for c in alternatives]
                reports.append(
                    {
                        "room_id": room_id,
                        "id": item.get("id"),
                        "role": role,
                        "action": "augmented_storage_supplier_pool",
                        "added_candidates": [c.get("unique_key") for c in alternatives],
                    }
                )
            asset = item.setdefault("asset", {})
            if not isinstance(asset, dict):
                asset = {}
                item["asset"] = asset
            if not asset.get("mesh_path"):
                asset["mesh_path"] = primary.get("asset_local_path")
            # Storage furniture should remain visually like the supplier model.
            # Stretching wardrobes/shelves into arbitrary bbox proportions is
            # exactly what makes repeated replacements look wrong.
            asset["mesh_fit_mode"] = "uniform"
    return reports


def find_room_scene(room_dir: Path) -> Path | None:
    for rel in SCENE_CANDIDATES:
        path = room_dir / rel
        if path.is_file():
            return path
    return None


def _kitchen_dining_scene_item(room_id: str, dining_item: dict[str, Any], index: int) -> dict[str, Any] | None:
    item_type = norm(dining_item.get("type"))
    role = "table" if "table" in item_type else "chair" if "chair" in item_type or "стул" in item_type else ""
    if role not in {"table", "chair"}:
        return None
    candidate = dining_item.get("supplier_candidate") if isinstance(dining_item.get("supplier_candidate"), dict) else {}
    try:
        cx = float(dining_item.get("x_m"))
        cy = float(dining_item.get("y_m"))
        sx = float(dining_item.get("width_m"))
        sy = float(dining_item.get("depth_m"))
        sz = float(dining_item.get("height_m"))
        z_min = float(dining_item.get("z_m") or 0.0)
        yaw = float(dining_item.get("yaw_deg") or 0.0)
    except Exception:
        return None
    item_id = str(dining_item.get("id") or f"kitchen_dining_{role}_{index:02d}")
    mesh_path = str(candidate.get("asset_local_path") or "").strip()
    return {
        "id": item_id,
        "name": str(candidate.get("title") or dining_item.get("type") or role),
        "category": ROLE_CATEGORY.get(role, "SupplierObject"),
        "semantic_group": ROLE_SEMANTIC_GROUP.get(role, role),
        "position_m": [round(cx, 4), round(cy, 4), round(z_min + sz / 2.0, 4)],
        "size_m": [round(sx, 4), round(sy, 4), round(sz, 4)],
        "yaw_deg": round(yaw, 4),
        "rotation_deg": round(yaw, 4),
        "yaw_rad": round(math.radians(yaw), 8),
        "aabb": {
            "x_min": round(cx - sx / 2.0, 4),
            "x_max": round(cx + sx / 2.0, 4),
            "y_min": round(cy - sy / 2.0, 4),
            "y_max": round(cy + sy / 2.0, 4),
            "z_min": round(z_min, 4),
            "z_max": round(z_min + sz, 4),
        },
        "constraints": {"mount_type": "floor", "touch_floor": {"side": "bottom"}},
        "asset": {"mesh_path": mesh_path, "mesh_fit_mode": "uniform"},
        "source": {
            "placement_source": "kitchen_dining_supplier",
            "asset_source": "supplier_catalog_local_asset",
            "supplier_replaced": True,
            "supplier_target_id": item_id,
            "supplier_unique_key": candidate.get("unique_key"),
            "supplier_source_site": candidate.get("source_site"),
            "supplier_product_url": candidate.get("product_url"),
            "placeholder_bbox": False,
            "room_id": room_id,
        },
        "meta": {
            "placeholder_bbox": False,
            "supplier_binding_applied": True,
            "required_role": role,
            "room_id": room_id,
            "support_group": "kitchen_dining",
            "supplier_candidate": candidate,
            "supplier_candidate_pool": [candidate] if candidate else [],
        },
    }


def kitchen_scene_from_assembly(room_dir: Path) -> dict[str, Any] | None:
    kitchen_dir = room_dir / "kitchen"
    json_path = kitchen_dir / f"{room_dir.name}.json"
    if not json_path.is_file():
        candidates: list[Path] = []
        for path in sorted(kitchen_dir.glob("*.json")):
            name = path.name.lower()
            if ".selection." in name or name.endswith(".selection.v1.json"):
                continue
            try:
                data = read_json(path)
            except Exception:
                continue
            if isinstance(data, dict) and (data.get("assembly_type") or data.get("bill_of_materials") or data.get("base_modules")):
                candidates.append(path)
        if not candidates:
            return None
        json_path = candidates[0]
    if not json_path.is_file():
        return None
    assembly = read_json(json_path)
    if isinstance(assembly, dict):
        _ensure_kitchen_microwave_decor(assembly)
    context = assembly.get("room_context") if isinstance(assembly.get("room_context"), dict) else {}
    room = context.get("room") if isinstance(context.get("room"), dict) else (read_json(room_dir / "room.json").get("room") or {})
    dims = assembly.get("dimensions") or {}
    width = float(dims.get("width_m") or room.get("width_m") or 2.4)
    depth = float(dims.get("depth_m") or 0.65)
    height = float(dims.get("height_m") or 2.2)
    item = {
        "id": str(assembly.get("id") or f"{room.get('id')}_kitchen"),
        "name": "procedural kitchen set",
        "category": "kitchen_set",
        "type": "procedural_assembly",
        "assembly_type": "procedural_kitchen",
        "position_m": [width / 2.0, depth / 2.0, height / 2.0],
        "size_m": [width, depth, height],
        "yaw_deg": 0.0,
        "rotation_deg": 0.0,
        "aabb": {"x_min": 0.0, "x_max": width, "y_min": 0.0, "y_max": depth, "z_min": 0.0, "z_max": height},
        "asset": {"kind": "procedural_kitchen", "assembly_type": "procedural_kitchen"},
        "meta": {**assembly, "procedural_assembly": "kitchen"},
        "source": {"asset_source": "procedural_kitchen", "source_room_id": room.get("id")},
    }
    placements = [item]
    for idx, dining_item in enumerate(context.get("dining_items") or [], start=1):
        if not isinstance(dining_item, dict):
            continue
        dining_scene_item = _kitchen_dining_scene_item(str(room.get("id") or room_dir.name), dining_item, idx)
        if dining_scene_item is not None:
            placements.append(dining_scene_item)
    return {"schema": "scene.v1", "room": room, "placements": placements, "meta": {"source": str(json_path)}}


def _kitchen_material_prompt(scene: dict[str, Any], room_dir: Path, prompt_room_type: str | None) -> str:
    prompt_path = room_dir / "prompt.txt"
    parts: list[str] = []
    if prompt_path.is_file():
        try:
            parts.append(prompt_path.read_text(encoding="utf-8").strip())
        except Exception:
            pass
    for item in room_items(scene):
        if not _is_procedural_kitchen_item(item):
            continue
        meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
        design = meta.get("design_spec") if isinstance(meta.get("design_spec"), dict) else {}
        if design.get("source_prompt"):
            parts.append(str(design.get("source_prompt")))
        palette = design.get("palette") if isinstance(design.get("palette"), dict) else {}
        if palette:
            parts.append("Kitchen palette: " + json.dumps(palette, ensure_ascii=False))
        break
    parts.append("Room type: kitchen. Choose durable water-resistant floor and calm washable wall material compatible with the kitchen set.")
    if prompt_room_type:
        parts.append(f"Prompt room type: {prompt_room_type}")
    return "\n".join(part for part in parts if part).strip()


def apply_missing_kitchen_surface_materials(
    scene: dict[str, Any],
    room_dir: Path,
    mode: str,
    prompt_room_type: str | None = None,
) -> list[dict[str, Any]]:
    if not _is_kitchen_scene(scene, prompt_room_type):
        return []
    room = scene.get("room") if isinstance(scene.get("room"), dict) else {}
    room_id = str(room.get("id") or room_dir.name)
    repairs: list[dict[str, Any]] = []
    if isinstance(room.get("floor_material"), dict) and isinstance(room.get("wall_material"), dict):
        return repairs

    run_dir = room_dir / "pipeline" / mode
    run_dir.mkdir(parents=True, exist_ok=True)
    prompt = _kitchen_material_prompt(scene, room_dir, prompt_room_type)
    llm_settings = {"provider": "none", "top_n": 5}

    try:
        from src.pipeline.flooring_stage import apply_flooring_to_scene, run_flooring_selection
        from src.pipeline.wall_stage import apply_wall_material_to_scene_with_catalog, run_wall_selection
    except Exception as exc:
        scene.setdefault("meta", {}).setdefault("requirement_postprocess", {}).setdefault("kitchen_surface_material_warnings", []).append(
            f"import_failed:{exc!r}"
        )
        return repairs

    if not isinstance(room.get("floor_material"), dict) and DEFAULT_FLOORING_MATERIALS.exists() and DEFAULT_FLOORING_STYLE_RULES.is_file():
        try:
            selection_path = run_dir / "flooring.selection.requirements.v1.json"
            selection = run_flooring_selection(
                prompt=prompt,
                style="modern",
                room_type="kitchen",
                room_description="kitchen room with procedural kitchen set",
                room_id=room_id,
                materials_path=DEFAULT_FLOORING_MATERIALS,
                style_rules_path=DEFAULT_FLOORING_STYLE_RULES,
                out_path=selection_path,
                top_k=10,
                llm_settings=llm_settings,
            )
            updated = apply_flooring_to_scene(scene, selection)
            scene.clear()
            scene.update(updated)
            repairs.append(
                {
                    "room_id": room_id,
                    "action": "selected_kitchen_floor_material",
                    "selection_json": str(selection_path.resolve()),
                    "selected_sku": (selection.get("selected_material") or {}).get("sku"),
                    "selected_name": (selection.get("selected_material") or {}).get("name"),
                }
            )
        except Exception as exc:
            scene.setdefault("meta", {}).setdefault("requirement_postprocess", {}).setdefault("kitchen_surface_material_warnings", []).append(
                f"flooring_failed:{exc!r}"
            )

    room = scene.get("room") if isinstance(scene.get("room"), dict) else {}
    if not isinstance(room.get("wall_material"), dict) and DEFAULT_WALL_MATERIALS.exists():
        try:
            selection_path = run_dir / "wall_material.selection.requirements.v1.json"
            selection = run_wall_selection(
                prompt=prompt,
                style="modern",
                room_type="kitchen",
                room_description="kitchen room with washable walls",
                room_id=room_id,
                materials_path=DEFAULT_WALL_MATERIALS,
                out_path=selection_path,
                top_k=10,
                llm_settings=llm_settings,
            )
            updated = apply_wall_material_to_scene_with_catalog(scene, selection, materials_path=DEFAULT_WALL_MATERIALS)
            scene.clear()
            scene.update(updated)
            repairs.append(
                {
                    "room_id": room_id,
                    "action": "selected_kitchen_wall_material",
                    "selection_json": str(selection_path.resolve()),
                    "selected_sku": (selection.get("selected_material") or {}).get("sku"),
                    "selected_name": (selection.get("selected_material") or {}).get("name"),
                }
            )
        except Exception as exc:
            scene.setdefault("meta", {}).setdefault("requirement_postprocess", {}).setdefault("kitchen_surface_material_warnings", []).append(
                f"wall_failed:{exc!r}"
            )

    if repairs:
        scene.setdefault("meta", {}).setdefault("requirement_postprocess", {})["kitchen_surface_material_repairs"] = repairs
    return repairs


def _scene_has_curtain_items(scene: dict[str, Any]) -> bool:
    for item in room_items(scene):
        text = item_text(item)
        if any(token in text for token in ("curtain", "shtor", "штор", "занавес")):
            return True
    return False


def apply_missing_curtains_to_scene(
    scene: dict[str, Any],
    room_dir: Path,
    mode: str,
    prompt_room_type: str | None = None,
) -> dict[str, Any] | None:
    room = scene.get("room") if isinstance(scene.get("room"), dict) else {}
    if _scene_has_curtain_items(scene):
        return None
    if not isinstance(room.get("windows"), list) or not room.get("windows"):
        return None
    try:
        from src.pipeline.curtain_stage import (
            discover_curtain_models,
            discover_supplier_curtain_models,
            load_curtain_catalog,
        )
        from src.pipeline.infinigen_scene_improvers import apply_curtains_to_scene
    except Exception as exc:
        scene.setdefault("meta", {}).setdefault("requirement_postprocess", {}).setdefault("curtain_warnings", []).append(
            f"import_failed:{exc!r}"
        )
        return None

    materials_path = REPO_ROOT / "data/floor_materials/shtorystore_curtains"
    if not materials_path.exists():
        return None
    try:
        catalog, catalog_base_dir = load_curtain_catalog(materials_path)
        if not catalog:
            return None
        run_dir = room_dir / "pipeline" / mode
        style_profile = {}
        style_path = run_dir / "style_profile.json"
        if style_path.is_file():
            try:
                style_profile = read_json(style_path)
            except Exception:
                style_profile = {}
        seed_text = str(room.get("id") or room_dir.name)
        seed = sum((idx + 1) * ord(ch) for idx, ch in enumerate(seed_text)) % (2**31)
        updated, info = apply_curtains_to_scene(
            scene,
            catalog=catalog,
            catalog_base_dir=catalog_base_dir,
            curtain_model_paths=discover_curtain_models(REPO_ROOT / "data/sourse/curtains_3d"),
            curtain_models=discover_supplier_curtain_models(SUPPLIER_CATALOG_PATH, REPO_ROOT / "data/sourse/suppliers/manual_assets/3ddd"),
            style_profile=style_profile if isinstance(style_profile, dict) else {},
            seed=seed,
        )
        if int(info.get("added_count") or 0) <= 0:
            return None
        scene.clear()
        scene.update(updated)
        info = dict(info)
        info["room_id"] = str(room.get("id") or room_dir.name)
        info["action"] = "added_shtorystore_curtains"
        info["prompt_room_type"] = prompt_room_type
        return info
    except Exception as exc:
        scene.setdefault("meta", {}).setdefault("requirement_postprocess", {}).setdefault("curtain_warnings", []).append(
            f"apply_failed:{exc!r}"
        )
        return None


def process_apartment(apt_dir: Path, mode: str) -> dict[str, Any]:
    manifest_path = apt_dir / "manifest.json"
    apartment_path = apt_dir / "apartment.json"
    if not manifest_path.is_file() or not apartment_path.is_file():
        raise FileNotFoundError(f"Missing manifest/apartment json in {apt_dir}")
    manifest = read_json(manifest_path)
    apartment = read_json(apartment_path)
    rooms_meta = manifest.get("rooms") or []
    room_jsons: dict[str, Path] = {}
    loaded_scenes: list[dict[str, Any]] = []
    scene_entries: list[dict[str, Any]] = []
    room_reports: list[dict[str, Any]] = []
    asset_search_roots = (apt_dir, LOCAL_TABLE_ASSET_ROOT, LOCAL_CHAIR_ASSET_ROOT)
    kitchen_layout_repairs: list[dict[str, Any]] = []
    kitchen_surface_material_repairs: list[dict[str, Any]] = []
    curtain_repairs: list[dict[str, Any]] = []

    for room_meta in rooms_meta:
        room_id = str(room_meta.get("room_id") or "")
        room_json = Path(str(room_meta.get("room_json") or ""))
        if room_id and room_json.is_file():
            room_jsons[room_id] = room_json

    for room_meta in rooms_meta:
        room_id = str(room_meta.get("room_id") or "")
        room_dir = apt_dir / "rooms" / room_id
        scene_path = find_room_scene(room_dir)
        scene = read_json(scene_path) if scene_path else kitchen_scene_from_assembly(room_dir)
        if not isinstance(scene, dict):
            room_reports.append({"room_id": room_id, "status": "missing_scene"})
            continue
        added: list[dict[str, Any]] = []
        kitchen_surface_material_repairs.extend(
            apply_missing_kitchen_surface_materials(scene, room_dir, mode, prompt_room_type=room_meta.get("prompt_room_type"))
        )
        curtain_info = apply_missing_curtains_to_scene(scene, room_dir, mode, prompt_room_type=room_meta.get("prompt_room_type"))
        if curtain_info is not None:
            curtain_repairs.append(curtain_info)
        kitchen_layout_repairs.extend(repair_kitchen_layout_for_scene(scene, prompt_room_type=room_meta.get("prompt_room_type")))
        added.extend(
            add_missing_kitchen_table(
                scene,
                prompt_room_type=room_meta.get("prompt_room_type"),
                asset_search_roots=asset_search_roots,
            )
        )
        added.extend(
            add_kitchen_table_seating(
                scene,
                prompt_room_type=room_meta.get("prompt_room_type"),
                asset_search_roots=asset_search_roots,
            )
        )
        loaded_scenes.append(scene)
        scene_entries.append(
            {
                "scene": scene,
                "room_meta": room_meta,
                "room_id": room_id,
                "room_dir": room_dir,
                "scene_path": scene_path,
                "added": added,
            }
        )

    storage_supplier_repairs = repair_storage_supplier_candidate_pools(scene_entries, apt_dir)
    sanitary_repairs = repair_sanitary_layouts(scene_entries, asset_search_roots=asset_search_roots)
    lighting_repairs = repair_ceiling_lighting_layouts(scene_entries)
    table_lamp_repairs = repair_table_lamp_sizes(scene_entries)
    desktop_support_repairs = repair_desktop_support_items(scene_entries)
    support_decor_repairs = repair_support_decor_items(scene_entries)
    bed_repairs = repair_bed_layouts(scene_entries)
    sanitary_added_per_room = add_missing_sanitary_per_room(scene_entries, asset_search_roots=asset_search_roots)
    sanitary_added = add_missing_sanitary_apartment(scene_entries, asset_search_roots=asset_search_roots)
    apartment_added = add_apartment_required_objects(loaded_scenes, asset_search_roots=asset_search_roots)

    for entry in scene_entries:
        scene = entry["scene"]
        room_meta = entry["room_meta"]
        room_id = str(entry["room_id"])
        room_dir = Path(entry["room_dir"])
        scene_path = entry.get("scene_path")
        added = entry.get("added") if isinstance(entry.get("added"), list) else []
        patched_path = room_dir / "pipeline" / mode / "scene_requirements.v1.json"
        write_json_if_changed(patched_path, scene)
        room_reports.append(
            {
                "room_id": room_id,
                "room_type": room_meta.get("room_type"),
                "prompt_room_type": room_meta.get("prompt_room_type"),
                "source_scene": str(scene_path) if scene_path else str(room_dir / "kitchen"),
                "requirements_scene": str(patched_path.resolve()),
                "added": [{"id": x["id"], "role": x["meta"]["required_role"]} for x in added],
            }
        )

    apt_min = estimate_apartment_min(apartment, room_jsons)

    placements: list[dict[str, Any]] = []
    for scene in loaded_scenes:
        room = scene.get("room") or {}
        room_id = str(room.get("id") or "")
        frame = ((room.get("meta") or {}).get("coordinate_frame") or {})
        if not room_id or not frame:
            continue
        for item in room_items(scene):
            if should_skip_apartment_item(item):
                continue
            placements.append(transform_item_to_apartment(item, frame, apt_min, room_id))

    out_scene = {
        "schema": "scene.v1",
        "room": apartment.get("room") or {},
        "placements": placements,
        "meta": {
            "source": "ensure_apartment_requirements",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "apartment_dir": str(apt_dir.resolve()),
            "mode": mode,
            "apartment_global_min_xy": [round(apt_min[0], 6), round(apt_min[1], 6)],
            "room_reports": room_reports,
            "sanitary_added": [
                {"id": x["id"], "role": x.get("meta", {}).get("required_role"), "room_id": x.get("meta", {}).get("room_id")}
                for x in sanitary_added
            ],
            "sanitary_added_per_room": [
                {"id": x["id"], "role": x.get("meta", {}).get("required_role"), "room_id": x.get("meta", {}).get("room_id")}
                for x in sanitary_added_per_room
            ],
            "sanitary_repairs": sanitary_repairs,
            "kitchen_layout_repairs": kitchen_layout_repairs,
            "kitchen_surface_material_repairs": kitchen_surface_material_repairs,
            "curtain_repairs": curtain_repairs,
            "storage_supplier_repairs": storage_supplier_repairs,
            "lighting_repairs": lighting_repairs,
            "table_lamp_repairs": table_lamp_repairs,
            "desktop_support_repairs": desktop_support_repairs,
            "support_decor_repairs": support_decor_repairs,
            "bed_repairs": bed_repairs,
            "apartment_added": [
                {"id": x["id"], "role": x.get("meta", {}).get("required_role"), "room_id": x.get("meta", {}).get("room_id")}
                for x in apartment_added
            ],
            "requirements": {
                "sanitary_each_bathroom_or_toilet": list(SANITARY_REQUIRED),
                "sanitary_apartment": list(SANITARY_REQUIRED),
                "apartment": ["bed", "table"],
            },
        },
    }
    out_dir = apt_dir / "apartment_pipeline" / mode
    out_path = write_json(out_dir / "scene_apartment.requirements.v1.json", out_scene)
    report_path = write_json(
        out_dir / "requirements_report.json",
        {
            "apartment_dir": str(apt_dir.resolve()),
            "scene_json": str(out_path.resolve()),
            "room_reports": room_reports,
            "sanitary_added": out_scene["meta"]["sanitary_added"],
            "sanitary_added_per_room": out_scene["meta"]["sanitary_added_per_room"],
            "sanitary_repairs": sanitary_repairs,
            "kitchen_layout_repairs": kitchen_layout_repairs,
            "kitchen_surface_material_repairs": kitchen_surface_material_repairs,
            "curtain_repairs": curtain_repairs,
            "storage_supplier_repairs": storage_supplier_repairs,
            "lighting_repairs": lighting_repairs,
            "table_lamp_repairs": table_lamp_repairs,
            "desktop_support_repairs": desktop_support_repairs,
            "support_decor_repairs": support_decor_repairs,
            "bed_repairs": bed_repairs,
            "apartment_added": out_scene["meta"]["apartment_added"],
            "placement_count": len(placements),
        },
    )
    return {"apartment_dir": str(apt_dir), "scene_json": str(out_path), "report_json": str(report_path), "placement_count": len(placements)}


def iter_apartments(root: Path) -> list[Path]:
    if (root / "manifest.json").is_file() and (root / "apartment.json").is_file():
        return [root]
    return sorted(p for p in root.glob("*/*") if (p / "manifest.json").is_file() and (p / "apartment.json").is_file())


def build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Ensure apartment-level required objects and assemble room scenes into one apartment scene.")
    ap.add_argument("root", help="Apartment dir or root containing project/apartment dirs.")
    ap.add_argument("--mode", default="optimal")
    ap.add_argument("--out-summary", default=None)
    return ap


def main() -> None:
    args = build_cli().parse_args()
    root = Path(args.root).expanduser().resolve()
    results = [process_apartment(apt_dir, args.mode) for apt_dir in iter_apartments(root)]
    summary_path = Path(args.out_summary).expanduser().resolve() if args.out_summary else root / "apartment_requirements_summary.json"
    write_json(summary_path, {"root": str(root), "count": len(results), "results": results})
    print(f"processed_apartments = {len(results)}")
    print(f"summary = {summary_path}")
    for result in results:
        print(f"{result['apartment_dir']} -> {result['scene_json']} ({result['placement_count']} placements)")


if __name__ == "__main__":
    main()
