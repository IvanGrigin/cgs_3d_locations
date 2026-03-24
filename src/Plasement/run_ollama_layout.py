#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/Plasement/run_ollama_layout.py

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from LLMModule.ollama_client import chat_json
from LLMModule.retry_llm_json import ValidationResult, run_retry_loop


# ============================================================
# IO
# ============================================================

def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def short_json(obj: Any, max_len: int = 4000) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        s = repr(obj)
    if len(s) <= max_len:
        return s
    return s[:max_len] + "\n...<truncated>..."


def short_text(text: str, max_len: int = 4000) -> str:
    s = (text or "").strip()
    if len(s) <= max_len:
        return s
    return s[:max_len] + "\n...<truncated>..."


# ============================================================
# Geometry
# ============================================================

def quantize_rot_0_90_180_270(deg: float) -> float:
    a = float(deg or 0.0) % 360.0
    allowed = (0.0, 90.0, 180.0, 270.0)
    return min(
        allowed,
        key=lambda t: (abs(((a - t + 180.0) % 360.0) - 180.0), t),
    )


def point_in_polygon(x: float, y: float, polygon_xy: List[Tuple[float, float]]) -> bool:
    inside = False
    n = len(polygon_xy)
    if n < 3:
        return False

    j = n - 1
    for i in range(n):
        xi, yi = polygon_xy[i]
        xj, yj = polygon_xy[j]
        intersect = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / max((yj - yi), 1e-12) + xi
        )
        if intersect:
            inside = not inside
        j = i
    return inside


def polygon_bbox(poly_xy: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in poly_xy]
    ys = [p[1] for p in poly_xy]
    return min(xs), max(xs), min(ys), max(ys)


def polygon_centroid(poly_xy: List[Tuple[float, float]]) -> Tuple[float, float]:
    if not poly_xy:
        return 0.0, 0.0
    x_sum = sum(x for x, _ in poly_xy)
    y_sum = sum(y for _, y in poly_xy)
    return x_sum / len(poly_xy), y_sum / len(poly_xy)


def aabb_from_center_size_rotation(
    cx: float,
    cy: float,
    sx: float,
    sy: float,
    sz: float,
    yaw_deg: float,
    z_floor_m: float = 0.0,
) -> Dict[str, float]:
    rot = quantize_rot_0_90_180_270(yaw_deg)
    if rot in (90.0, 270.0):
        sx, sy = sy, sx

    return {
        "x_min": cx - sx / 2.0,
        "x_max": cx + sx / 2.0,
        "y_min": cy - sy / 2.0,
        "y_max": cy + sy / 2.0,
        "z_min": z_floor_m,
        "z_max": z_floor_m + sz,
    }


def rect_inside_polygon(
    cx: float,
    cy: float,
    sx: float,
    sy: float,
    yaw_deg: float,
    polygon_xy: List[Tuple[float, float]],
) -> bool:
    rot = quantize_rot_0_90_180_270(yaw_deg)
    if rot in (90.0, 270.0):
        sx, sy = sy, sx

    hx = sx / 2.0
    hy = sy / 2.0
    corners = [
        (cx - hx, cy - hy),
        (cx - hx, cy + hy),
        (cx + hx, cy - hy),
        (cx + hx, cy + hy),
    ]
    return all(point_in_polygon(x, y, polygon_xy) for x, y in corners)


def rects_overlap_2d(a: Dict[str, float], b: Dict[str, float], eps: float = 1e-6) -> bool:
    return not (
        a["x_max"] <= b["x_min"] + eps
        or a["x_min"] >= b["x_max"] - eps
        or a["y_max"] <= b["y_min"] + eps
        or a["y_min"] >= b["y_max"] - eps
    )


def intersection_area_2d(a: Dict[str, float], b: Dict[str, float]) -> float:
    ix = max(0.0, min(a["x_max"], b["x_max"]) - max(a["x_min"], b["x_min"]))
    iy = max(0.0, min(a["y_max"], b["y_max"]) - max(a["y_min"], b["y_min"]))
    return ix * iy


def aabb_area_2d(a: Dict[str, float]) -> float:
    return max(0.0, a["x_max"] - a["x_min"]) * max(0.0, a["y_max"] - a["y_min"])


def aabb_overlap_ratio(a: Dict[str, float], b: Dict[str, float]) -> float:
    inter = intersection_area_2d(a, b)
    if inter <= 0.0:
        return 0.0
    area_a = max(1e-9, aabb_area_2d(a))
    return inter / area_a


def find_nearest_valid_position(
    target_x: float,
    target_y: float,
    sx: float,
    sy: float,
    sz: float,
    yaw_deg: float,
    polygon_xy: List[Tuple[float, float]],
    occupied: List[Dict[str, float]],
    forbidden: Optional[List[Dict[str, float]]] = None,
    grid_step: float = 0.10,
    max_radius_steps: int = 200,
) -> Tuple[float, float]:
    forbidden = forbidden or []

    cand_aabb = aabb_from_center_size_rotation(target_x, target_y, sx, sy, sz, yaw_deg)
    if rect_inside_polygon(target_x, target_y, sx, sy, yaw_deg, polygon_xy):
        if not any(rects_overlap_2d(cand_aabb, occ) for occ in occupied):
            if not any(rects_overlap_2d(cand_aabb, bad) for bad in forbidden):
                return target_x, target_y

    for r in range(1, max_radius_steps + 1):
        d = r * grid_step

        xs = [target_x - d + i * grid_step for i in range(2 * r + 1)]
        ys = [target_y - d + i * grid_step for i in range(2 * r + 1)]

        border_points: List[Tuple[float, float]] = []
        for x in xs:
            border_points.append((x, target_y - d))
            border_points.append((x, target_y + d))
        for y in ys[1:-1]:
            border_points.append((target_x - d, y))
            border_points.append((target_x + d, y))

        border_points.sort(
            key=lambda p: (
                abs(p[0] - target_x) + abs(p[1] - target_y),
                p[0],
                p[1],
            )
        )

        for cx, cy in border_points:
            if not rect_inside_polygon(cx, cy, sx, sy, yaw_deg, polygon_xy):
                continue
            aabb = aabb_from_center_size_rotation(cx, cy, sx, sy, sz, yaw_deg)
            if any(rects_overlap_2d(aabb, occ) for occ in occupied):
                continue
            if any(rects_overlap_2d(aabb, bad) for bad in forbidden):
                continue
            return cx, cy

    raise ValueError("Не удалось найти допустимую позицию для объекта repair-алгоритмом")


def distance_point_to_segment(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    ab2 = abx * abx + aby * aby
    if ab2 <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab2))
    qx = ax + t * abx
    qy = ay + t * aby
    return math.hypot(px - qx, py - qy)


def classify_rect_wall_contact(
    aabb: Dict[str, float],
    poly_xy: List[Tuple[float, float]],
    tol: float = 0.18,
) -> Dict[str, Any]:
    cx = 0.5 * (aabb["x_min"] + aabb["x_max"])
    cy = 0.5 * (aabb["y_min"] + aabb["y_max"])
    rect_w = aabb["x_max"] - aabb["x_min"]
    rect_h = aabb["y_max"] - aabb["y_min"]

    best_idx = -1
    best_dist = 1e18
    best_p1 = (0.0, 0.0)
    best_p2 = (0.0, 0.0)

    for i in range(len(poly_xy)):
        x1, y1 = poly_xy[i]
        x2, y2 = poly_xy[(i + 1) % len(poly_xy)]
        d = distance_point_to_segment(cx, cy, x1, y1, x2, y2)
        if d < best_dist:
            best_dist = d
            best_idx = i
            best_p1 = (x1, y1)
            best_p2 = (x2, y2)

    wall_vec = (best_p2[0] - best_p1[0], best_p2[1] - best_p1[1])
    wall_len = math.hypot(wall_vec[0], wall_vec[1])
    wall_axis = "horizontal" if abs(wall_vec[0]) >= abs(wall_vec[1]) else "vertical"

    if wall_axis == "horizontal":
        back_to_wall = (
            abs(aabb["y_min"] - best_p1[1]) <= tol or abs(aabb["y_max"] - best_p1[1]) <= tol
        )
    else:
        back_to_wall = (
            abs(aabb["x_min"] - best_p1[0]) <= tol or abs(aabb["x_max"] - best_p1[0]) <= tol
        )

    return {
        "nearest_wall_index": best_idx,
        "nearest_wall_distance": best_dist,
        "nearest_wall_axis": wall_axis,
        "nearest_wall_length": wall_len,
        "rect_w": rect_w,
        "rect_h": rect_h,
        "back_to_wall": bool(back_to_wall),
    }


# ============================================================
# Data extraction
# ============================================================

def extract_room_polygon_xy(room: Dict[str, Any]) -> List[Tuple[float, float]]:
    root = room.get("room") if isinstance(room.get("room"), dict) else room
    poly = root.get("floor_polygon")
    if not isinstance(poly, list) or len(poly) < 3:
        raise ValueError("В room.json не найден корректный room.floor_polygon")

    out = []
    for p in poly:
        out.append((float(p["x"]), float(p["y"])))
    return out


def extract_ceiling_height(room: Dict[str, Any]) -> float:
    root = room.get("room") if isinstance(room.get("room"), dict) else room
    return float(root.get("ceiling_height", 2.8))


def extract_room_root(room: Dict[str, Any]) -> Dict[str, Any]:
    return room.get("room") if isinstance(room.get("room"), dict) else room


def extract_objects(src: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = src.get("objects") or src.get("items") or src.get("placements") or []
    if not isinstance(items, list):
        raise ValueError("objects.json: ожидается objects/items/placements как список")
    return items


def extract_class_name(obj: Dict[str, Any]) -> str:
    for key in ("class_name", "class", "type", "name"):
        v = obj.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()

    asset_meta = obj.get("asset_meta") or {}
    for key in ("category", "super-category", "super_category"):
        v = asset_meta.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()

    return "object"


def extract_size_m(obj: Dict[str, Any]) -> List[float]:
    for key in ("size_m", "bbox_size_m", "size"):
        v = obj.get(key)
        if isinstance(v, list) and len(v) == 3:
            return [float(v[0]), float(v[1]), float(v[2])]

    for key in ("min_size_mm", "max_size_mm"):
        v = obj.get(key)
        if isinstance(v, list) and len(v) == 3:
            return [float(v[0]) / 1000.0, float(v[1]) / 1000.0, float(v[2]) / 1000.0]

    asset_meta = obj.get("asset_meta") or {}
    if all(k in asset_meta for k in ("size_x", "size_y", "size_z")):
        return [
            float(asset_meta["size_x"]),
            float(asset_meta["size_y"]),
            float(asset_meta["size_z"]),
        ]

    raise ValueError(f"Не удалось определить size_m для объекта: {obj}")


def extract_mount_type(obj: Dict[str, Any]) -> str:
    constraints = obj.get("constraints") or {}
    mount_type = constraints.get("mount_type")
    if isinstance(mount_type, str) and mount_type.strip():
        return mount_type.strip().lower()
    return "floor"


def extract_openings(room: Dict[str, Any]) -> Dict[str, Any]:
    root = extract_room_root(room)

    def _clean(items: Any) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not isinstance(items, list):
            return out
        for x in items:
            if not isinstance(x, dict):
                continue
            row: Dict[str, Any] = {}
            for k, v in x.items():
                if isinstance(v, (int, float, str, bool)) or v is None:
                    row[k] = v
                elif isinstance(v, list):
                    if all(isinstance(z, (int, float, str, bool)) or z is None for z in v):
                        row[k] = v
            out.append(row)
        return out

    return {
        "doors": _clean(root.get("doors")),
        "windows": _clean(root.get("windows")),
        "openings": _clean(root.get("openings")),
    }


def build_opening_summary(room: Dict[str, Any], poly_xy: List[Tuple[float, float]]) -> Dict[str, Any]:
    root = extract_room_root(room)
    out: Dict[str, Any] = {
        "doors": [],
        "windows": [],
        "openings": [],
    }

    wall_map = {}
    for idx, w in enumerate(root.get("walls") or []):
        if not isinstance(w, dict):
            continue
        fv = int(w.get("from_vertex", idx))
        tv = int(w.get("to_vertex", (idx + 1) % len(poly_xy)))
        if 0 <= fv < len(poly_xy) and 0 <= tv < len(poly_xy):
            wall_map[str(w.get("id", f"w{idx}"))] = {
                "from_vertex": fv,
                "to_vertex": tv,
                "p1": poly_xy[fv],
                "p2": poly_xy[tv],
            }

    def _summarize(items: Any, kind: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        if not isinstance(items, list):
            return rows
        for i, row in enumerate(items):
            if not isinstance(row, dict):
                continue
            wall_id = str(row.get("wall_id", ""))
            side_info: Dict[str, Any] = {"kind": kind, "id": str(row.get("id", f"{kind}_{i}"))}
            if wall_id and wall_id in wall_map:
                info = wall_map[wall_id]
                p1 = info["p1"]
                p2 = info["p2"]
                side_info["wall_id"] = wall_id
                side_info["wall_from"] = [round(p1[0], 3), round(p1[1], 3)]
                side_info["wall_to"] = [round(p2[0], 3), round(p2[1], 3)]
                side_info["wall_axis"] = "horizontal" if abs(p1[0] - p2[0]) >= abs(p1[1] - p2[1]) else "vertical"
            for k in ("offset", "width", "height", "position", "s", "e", "center"):
                if k in row and isinstance(row[k], (int, float, str)):
                    side_info[k] = row[k]
            rows.append(side_info)
        return rows

    out["doors"] = _summarize(root.get("doors"), "door")
    out["windows"] = _summarize(root.get("windows"), "window")
    out["openings"] = _summarize(root.get("openings"), "opening")
    return out


def build_llm_payload(room: Dict[str, Any], objects_data: Dict[str, Any]) -> Dict[str, Any]:
    poly = extract_room_polygon_xy(room)
    x_min, x_max, y_min, y_max = polygon_bbox(poly)
    cx, cy = polygon_centroid(poly)
    ceiling_height = extract_ceiling_height(room)
    openings = extract_openings(room)
    opening_summary = build_opening_summary(room, poly)

    src_items = extract_objects(objects_data)
    compact_objects = []

    for idx, obj in enumerate(src_items):
        sx, sy, sz = extract_size_m(obj)
        compact_objects.append({
            "index": idx,
            "name": extract_class_name(obj),
            "mount_type": extract_mount_type(obj),
            "size_m": [round(sx, 3), round(sy, 3), round(sz, 3)],
            "footprint_area_m2": round(sx * sy, 3),
        })

    compact_objects.sort(key=lambda x: (-x["footprint_area_m2"], x["index"]))

    return {
        "room": {
            "floor_polygon_xy": [[round(x, 3), round(y, 3)] for x, y in poly],
            "bbox_xy": [round(x_min, 3), round(x_max, 3), round(y_min, 3), round(y_max, 3)],
            "centroid_xy": [round(cx, 3), round(cy, 3)],
            "ceiling_height": round(ceiling_height, 3),
            "openings": openings,
            "opening_summary": opening_summary,
        },
        "objects": compact_objects,
    }


# ============================================================
# Prompts
# ============================================================

def build_plan_schema(n_objects: int) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "zoning": {
                "type": "object",
                "properties": {
                    "sleeping_zone": {"type": "string"},
                    "storage_zone": {"type": "string"},
                    "dressing_zone": {"type": "string"},
                    "circulation": {"type": "string"},
                },
                "required": ["sleeping_zone", "storage_zone", "dressing_zone", "circulation"],
                "additionalProperties": False,
            },
            "placements": {
                "type": "array",
                "minItems": 1,
                "maxItems": n_objects,
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "yaw_deg": {"type": "integer", "enum": [0, 90, 180, 270]},
                        "role": {"type": "string"},
                        "explanation": {"type": "string"},
                    },
                    "required": ["index", "x", "y", "yaw_deg", "role", "explanation"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["reasoning", "zoning", "placements"],
        "additionalProperties": False,
    }


def build_plan_system_prompt() -> str:
    return (
        "You are a senior interior designer and spatial planner.\n"
        "\n"
        "This is the planner stage.\n"
        "You must output exactly one JSON object and nothing else.\n"
        "\n"
        "Primary goals:\n"
        "- realistic bedroom composition;\n"
        "- clear circulation from the door;\n"
        "- free space in front of door and window;\n"
        "- balanced use of the room;\n"
        "- no dense clustering in one corner or one half of the room;\n"
        "- bed as the main focal object;\n"
        "- symmetric nightstands when possible;\n"
        "- wardrobe against a wall with its back side to the wall;\n"
        "- dressing table with accessible standing space;\n"
        "- ceiling lamp near the visual center of the composition.\n"
        "\n"
        "Hard rules:\n"
        "- Use each object index at most once.\n"
        "- Do not invent objects.\n"
        "- yaw_deg must be one of 0, 90, 180, 270.\n"
        "- x and y are center coordinates of the footprint.\n"
        "- Provide as many explicit placements as possible, especially for the main objects.\n"
        "- If some coordinates remain uncertain, still return a valid partial placements list.\n"
    )


def build_plan_user_prompt(
    payload: Dict[str, Any],
    mode: str,
    design_brief: str,
    previous_feedback: str = "",
) -> str:
    return (
        "Plan a high-quality bedroom arrangement.\n"
        "\n"
        "User brief:\n"
        f"{design_brief.strip() if design_brief.strip() else 'No extra brief provided.'}\n"
        "\n"
        "Interpret the room as a real bedroom interior design task.\n"
        "Make the composition ergonomic, balanced, and realistic.\n"
        "\n"
        "Important layout requirements:\n"
        "- bed should be placed under the window if feasible;\n"
        "- two nightstands should be symmetric on both sides of the bed;\n"
        "- wardrobe must stand against a wall with its back side touching the wall;\n"
        "- wardrobe must not act as a divider;\n"
        "- preserve free space in front of the door;\n"
        "- preserve entry path from the door into the room;\n"
        "- preserve access to the dressing table;\n"
        "- ceiling lamp should be near the visual center of the composition;\n"
        "- do not cluster almost all furniture in one half of the room.\n"
        "\n"
        "Previous failed-attempt feedback that must be corrected:\n"
        f"{previous_feedback.strip() if previous_feedback.strip() else 'None.'}\n"
        "\n"
        f"mode={mode}\n"
        "\n"
        "Room and objects data:\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def build_plan_repair_system_prompt() -> str:
    return (
        "You are a strict JSON repair and normalization model.\n"
        "\n"
        "You will receive:\n"
        "- room and object payload,\n"
        "- a raw planner response that may be prose, malformed JSON, or incomplete JSON.\n"
        "\n"
        "Your task is to return exactly one valid JSON object with fields:\n"
        "- reasoning: string\n"
        "- zoning: object with sleeping_zone, storage_zone, dressing_zone, circulation\n"
        "- placements: array of objects with index, x, y, yaw_deg, role, explanation\n"
        "\n"
        "Rules:\n"
        "- Output JSON only.\n"
        "- No markdown.\n"
        "- No prose outside JSON.\n"
        "- Preserve planner intent when possible.\n"
        "- If raw planner text is malformed JSON, repair it.\n"
        "- If raw planner text is prose only, infer a structured plan conservatively.\n"
        "- It is acceptable to return a partial placements list, but it must not be empty.\n"
        "- yaw_deg must be one of 0, 90, 180, 270.\n"
    )


def build_plan_repair_user_prompt(
    payload: Dict[str, Any],
    design_brief: str,
    raw_plan_text: str,
    n_objects: int,
) -> str:
    return (
        "Repair planner output into valid planner JSON.\n"
        "\n"
        f"Expected object index range: 0..{n_objects - 1}\n"
        "\n"
        "Design brief:\n"
        f"{design_brief.strip() if design_brief.strip() else 'No design brief provided.'}\n"
        "\n"
        "ROOM+OBJECTS:\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n\n"
        "RAW_PLANNER_RESPONSE:\n"
        + raw_plan_text
    )


def build_json_system_prompt() -> str:
    return (
        "You are a strict scene builder.\n"
        "\n"
        "You receive:\n"
        "- room geometry,\n"
        "- object sizes,\n"
        "- planner output,\n"
        "- extracted draft placements,\n"
        "- repair feedback from previous failed attempts.\n"
        "\n"
        "Your task is to build strict JSON placements.\n"
        "The planner output is only a draft, not a ground truth.\n"
        "\n"
        "Priority order:\n"
        "1. geometric feasibility,\n"
        "2. user design brief,\n"
        "3. repair feedback,\n"
        "4. planner draft intent.\n"
        "\n"
        "If planner coordinates are geometrically invalid or conflict with the brief,\n"
        "you must correct them.\n"
        "\n"
        "Hard rules:\n"
        "- Output exactly one JSON object.\n"
        "- No prose.\n"
        "- No markdown.\n"
        "- No code fences.\n"
        "- The first character must be '{'.\n"
        "- The last character must be '}'.\n"
        "- Use each index exactly once.\n"
        "- Do not invent objects.\n"
        "- Do not drop objects.\n"
        "- yaw_deg must be one of 0, 90, 180, 270.\n"
        "- Keep bed on the requested window wall when required.\n"
        "- Keep wardrobe back-to-wall when required.\n"
        "- Keep door clearance free.\n"
        "- Keep dressing table accessible.\n"
        "- Do not place wardrobe as a room divider.\n"
    )


def build_json_user_prompt(
    payload: Dict[str, Any],
    mode: str,
    plan_json: Dict[str, Any],
    extracted_hints: Dict[str, Any],
    n_objects: int,
    repair_feedback: str,
) -> str:
    return (
        "Build strict JSON placements.\n"
        "\n"
        "Allowed output format only:\n"
        "{\"placements\":[{\"index\":0,\"x\":1.0,\"y\":2.0,\"yaw_deg\":0}]}\n"
        "\n"
        f"Expected number of objects: {n_objects}\n"
        f"mode={mode}\n"
        "\n"
        "Important:\n"
        "- planner JSON is a draft, not mandatory truth;\n"
        "- extracted hints are draft placements, not mandatory truth;\n"
        "- if draft placements conflict with geometry or repair feedback, correct them;\n"
        "- prefer valid, ergonomic, constraint-satisfying placements over literal copying.\n"
        "\n"
        "Rules:\n"
        "- use all indices exactly once;\n"
        "- keep only index, x, y, yaw_deg;\n"
        "- keep the bed under the window if requested;\n"
        "- keep wardrobe against a wall with back side touching the wall;\n"
        "- do not use wardrobe as divider;\n"
        "- keep free space in front of the door;\n"
        "- keep entry path clear;\n"
        "- keep dressing table accessible;\n"
        "- keep ceiling lamp near composition center.\n"
        "\n"
        "Repair feedback from previous failed attempts:\n"
        f"{repair_feedback.strip() if repair_feedback.strip() else 'None.'}\n"
        "\n"
        "ROOM+OBJECTS:\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n\n"
        "PLANNER_JSON:\n"
        + json.dumps(plan_json, ensure_ascii=False, indent=2)
        + "\n\n"
        "EXTRACTED_HINTS:\n"
        + json.dumps(extracted_hints, ensure_ascii=False, indent=2)
    )


def build_critic_system_prompt() -> str:
    return (
        "You are a strict interior-layout critic.\n"
        "\n"
        "You receive:\n"
        "- room geometry,\n"
        "- object sizes,\n"
        "- the original design brief,\n"
        "- the planner reasoning,\n"
        "- the built JSON placements,\n"
        "- deterministic validation issues.\n"
        "\n"
        "Your task is to decide whether the built scene satisfies the design intent.\n"
        "\n"
        "Return exactly one JSON object with this shape:\n"
        "{\n"
        '  "ok": true,\n'
        '  "hard_fail": false,\n'
        '  "should_replan": false,\n'
        '  "issues": ["..."],\n'
        '  "fix_instructions": ["..."],\n'
        '  "summary": "..." \n'
        "}\n"
        "\n"
        "Critical output rules:\n"
        "- JSON only.\n"
        "- No markdown.\n"
        "- No prose before JSON.\n"
        "- No prose after JSON.\n"
        "- issues must contain at most 8 short unique strings.\n"
        "- fix_instructions must contain at most 6 short unique strings.\n"
        "- Never repeat the same issue twice.\n"
        "- summary must be at most 200 characters.\n"
        "- If deterministic issues already contain the problem, do not duplicate it many times.\n"
        "- hard_fail=true only for major layout violations.\n"
        "- should_replan=true only if the composition is structurally wrong.\n"
    )


def build_critic_user_prompt(
    payload: Dict[str, Any],
    design_brief: str,
    plan_json: Dict[str, Any],
    llm_layout: Dict[str, Any],
    deterministic_issues: List[str],
    deterministic_metrics: Dict[str, Any],
    attempt_index: int,
) -> str:
    return (
        f"Critique scene attempt #{attempt_index}.\n"
        "\n"
        "Original design brief:\n"
        f"{design_brief.strip() if design_brief.strip() else 'No design brief provided.'}\n"
        "\n"
        "Deterministic issues found by Python validator:\n"
        + json.dumps(deterministic_issues, ensure_ascii=False, indent=2)
        + "\n\n"
        "Deterministic metrics:\n"
        + json.dumps(deterministic_metrics, ensure_ascii=False, indent=2)
        + "\n\n"
        "ROOM+OBJECTS:\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n\n"
        "PLANNER_JSON:\n"
        + json.dumps(plan_json, ensure_ascii=False, indent=2)
        + "\n\n"
        "BUILT JSON PLACEMENTS:\n"
        + json.dumps(llm_layout, ensure_ascii=False, indent=2)
    )


def build_output_schema(n_objects: int) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "placements": {
                "type": "array",
                "minItems": n_objects,
                "maxItems": n_objects,
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "yaw_deg": {"type": "integer", "enum": [0, 90, 180, 270]},
                    },
                    "required": ["index", "x", "y", "yaw_deg"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["placements"],
        "additionalProperties": False,
    }


def build_critic_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "ok": {"type": "boolean"},
            "hard_fail": {"type": "boolean"},
            "should_replan": {"type": "boolean"},
            "issues": {"type": "array", "items": {"type": "string"}},
            "fix_instructions": {"type": "array", "items": {"type": "string"}},
            "summary": {"type": "string"},
        },
        "required": ["ok", "hard_fail", "should_replan", "issues", "fix_instructions", "summary"],
        "additionalProperties": False,
    }


# ============================================================
# JSON extraction / normalization
# ============================================================

def _strip_code_fences(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_balanced_json_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_str = False
    esc = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    return None


def extract_first_json_object(text: str) -> Dict[str, Any]:
    text = _strip_code_fences(text)

    if not text:
        raise ValueError("Пустой ответ модели")

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
    if fenced:
        try:
            data = json.loads(fenced.group(1))
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    balanced = _extract_balanced_json_object(text)
    if balanced:
        data = json.loads(balanced)
        if isinstance(data, dict):
            return data

    raise ValueError("В ответе модели не найден JSON-объект")


def normalize_critic_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    def _uniq_short_str_list(value: Any, limit: int) -> List[str]:
        out: List[str] = []
        seen = set()
        if not isinstance(value, list):
            return out
        for item in value:
            if not isinstance(item, str):
                continue
            s = item.strip()
            if not s:
                continue
            if s in seen:
                continue
            seen.add(s)
            out.append(s[:200])
            if len(out) >= limit:
                break
        return out

    summary = data.get("summary", "")
    if not isinstance(summary, str):
        summary = ""
    summary = summary.strip()[:200]

    return {
        "ok": bool(data.get("ok")),
        "hard_fail": bool(data.get("hard_fail")),
        "should_replan": bool(data.get("should_replan")),
        "issues": _uniq_short_str_list(data.get("issues"), 8),
        "fix_instructions": _uniq_short_str_list(data.get("fix_instructions"), 6),
        "summary": summary,
    }


def extract_safe_role_from_index(idx: int) -> str:
    return f"object_{idx}"


def normalize_plan_json_candidate(data: Dict[str, Any], n_objects: int) -> Dict[str, Any]:
    reasoning = data.get("reasoning")
    if not isinstance(reasoning, str) or not reasoning.strip():
        reasoning = "Planner omitted detailed reasoning; structure was repaired automatically."

    zoning = data.get("zoning")
    if not isinstance(zoning, dict):
        zoning = {}
    zoning = {
        "sleeping_zone": str(zoning.get("sleeping_zone", "bed area")),
        "storage_zone": str(zoning.get("storage_zone", "wardrobe area")),
        "dressing_zone": str(zoning.get("dressing_zone", "dressing table area")),
        "circulation": str(zoning.get("circulation", "clear path from door")),
    }

    placements_raw = data.get("placements")
    if not isinstance(placements_raw, list):
        placements_raw = []

    seen = set()
    placements: List[Dict[str, Any]] = []

    for row in placements_raw:
        if not isinstance(row, dict):
            continue
        try:
            idx = int(row["index"])
            x = float(row["x"])
            y = float(row["y"])
            yaw = int(row["yaw_deg"])
        except Exception:
            continue

        if idx < 0 or idx >= n_objects or idx in seen:
            continue

        yaw_q = int(quantize_rot_0_90_180_270(yaw))
        role = str(row.get("role", extract_safe_role_from_index(idx)))
        explanation = str(row.get("explanation", "planner placement"))

        placements.append({
            "index": idx,
            "x": x,
            "y": y,
            "yaw_deg": yaw_q,
            "role": role,
            "explanation": explanation,
        })
        seen.add(idx)

    placements.sort(key=lambda z: z["index"])

    return {
        "reasoning": reasoning,
        "zoning": zoning,
        "placements": placements,
    }


# ============================================================
# Planner normalization / validation
# ============================================================

def validate_plan_structure(raw_text: str, n_objects: int) -> ValidationResult[Dict[str, Any]]:
    try:
        data = extract_first_json_object(raw_text)
    except Exception as e:
        excerpt = short_text(raw_text, 2000)
        return ValidationResult(
            ok=False,
            feedback=f"Planner response is not valid JSON: {e}\nRAW_TEXT_BEGIN\n{excerpt}\nRAW_TEXT_END",
        )

    normalized = normalize_plan_json_candidate(data, n_objects=n_objects)

    if not normalized["placements"]:
        return ValidationResult(ok=False, feedback="Planner JSON contains zero usable placements.")

    return ValidationResult(ok=True, normalized=normalized)


def extract_stage1_hints(plan_json: Dict[str, Any], n_objects: int) -> Dict[str, Any]:
    placements = plan_json.get("placements")
    if not isinstance(placements, list):
        return {"placements": []}

    hints: Dict[int, Dict[str, Any]] = {}

    for row in placements:
        if not isinstance(row, dict):
            continue
        try:
            idx = int(row["index"])
            x = float(row["x"])
            y = float(row["y"])
            yaw = int(row["yaw_deg"])
        except Exception:
            continue

        if 0 <= idx < n_objects:
            hints[idx] = {
                "index": idx,
                "x": x,
                "y": y,
                "yaw_deg": int(quantize_rot_0_90_180_270(yaw)),
                "role": str(row.get("role", "")),
                "explanation": str(row.get("explanation", "")),
            }

    return {"placements": [hints[i] for i in sorted(hints.keys())]}


# ============================================================
# JSON validation
# ============================================================

def validate_structure(raw_text: str, n_objects: int) -> ValidationResult[Dict[str, Any]]:
    try:
        data = extract_first_json_object(raw_text)
    except Exception as e:
        excerpt = (raw_text or "").strip()
        if len(excerpt) > 2000:
            excerpt = excerpt[:2000] + "\n...<truncated>..."
        return ValidationResult(
            ok=False,
            feedback=(
                f"Ответ не является корректным JSON. Ошибка: {e}\n"
                f"RAW_TEXT_BEGIN\n{excerpt}\nRAW_TEXT_END"
            ),
        )

    placements = data.get("placements")
    if not isinstance(placements, list):
        return ValidationResult(ok=False, feedback='В JSON отсутствует поле "placements" как список.')

    if len(placements) != n_objects:
        return ValidationResult(
            ok=False,
            feedback=f'Количество placements неверно: {len(placements)}. Ожидается ровно {n_objects}.',
        )

    seen = set()
    normalized: List[Dict[str, Any]] = []

    for i, p in enumerate(placements):
        if not isinstance(p, dict):
            return ValidationResult(ok=False, feedback=f"Элемент placements[{i}] должен быть объектом.")

        if "index" not in p or "x" not in p or "y" not in p or "yaw_deg" not in p:
            return ValidationResult(
                ok=False,
                feedback=f'Элемент placements[{i}] должен содержать поля "index", "x", "y", "yaw_deg".',
            )

        try:
            idx = int(p["index"])
            x = float(p["x"])
            y = float(p["y"])
            yaw_deg = int(p["yaw_deg"])
        except Exception:
            return ValidationResult(ok=False, feedback=f"Элемент placements[{i}] содержит поля неверного типа.")

        if idx < 0 or idx >= n_objects:
            return ValidationResult(
                ok=False,
                feedback=f"Недопустимый index={idx}. Ожидается от 0 до {n_objects - 1}.",
            )

        if idx in seen:
            return ValidationResult(ok=False, feedback=f"Индекс {idx} встречается более одного раза.")
        seen.add(idx)

        if yaw_deg not in (0, 90, 180, 270):
            return ValidationResult(
                ok=False,
                feedback=f"Для index={idx} yaw_deg={yaw_deg}. Допустимы только 0, 90, 180, 270.",
            )

        normalized.append({"index": idx, "x": x, "y": y, "yaw_deg": yaw_deg})

    normalized.sort(key=lambda z: z["index"])
    return ValidationResult(ok=True, normalized={"placements": normalized})


def validate_critic_response(raw_text: str) -> ValidationResult[Dict[str, Any]]:
    try:
        data = extract_first_json_object(raw_text)
    except Exception as e:
        excerpt = (raw_text or "").strip()
        if len(excerpt) > 2000:
            excerpt = excerpt[:2000] + "\n...<truncated>..."
        return ValidationResult(
            ok=False,
            feedback=f"Critic response is not valid JSON: {e}\nRAW_TEXT_BEGIN\n{excerpt}\nRAW_TEXT_END",
        )

    normalized = normalize_critic_payload(data)
    return ValidationResult(ok=True, normalized=normalized)


# ============================================================
# Ollama helpers
# ============================================================

def extract_text_from_ollama_response(resp: Dict[str, Any]) -> str:
    message = resp.get("message")
    if isinstance(message, dict):
        for key in ("content", "text"):
            value = message.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    for key in ("response", "content", "text", "output"):
        value = resp.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    if isinstance(message, dict):
        for key in ("reasoning_content", "thinking", "reasoning"):
            value = message.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    return ""


def normalize_model_list(primary_model: str, models: Optional[List[str]]) -> List[str]:
    result: List[str] = []

    def add(model_name: Optional[str]) -> None:
        if not model_name:
            return
        m = str(model_name).strip()
        if not m:
            return
        if m not in result:
            result.append(m)

    add(primary_model)
    if models:
        for m in models:
            add(m)

    if not result:
        raise ValueError("Список моделей Ollama пуст")

    return result


def build_placement_map(
    src_items: List[Dict[str, Any]],
    llm_layout: Dict[str, Any],
    poly_xy: List[Tuple[float, float]],
    ceiling_height: float,
) -> Dict[int, Dict[str, Any]]:
    by_index = {int(p["index"]): p for p in llm_layout.get("placements", [])}
    out: Dict[int, Dict[str, Any]] = {}

    for idx, src_obj in enumerate(src_items):
        if idx not in by_index:
            continue

        pred = by_index[idx]
        sx, sy, sz = extract_size_m(src_obj)
        yaw = quantize_rot_0_90_180_270(float(pred["yaw_deg"]))
        mount_type = extract_mount_type(src_obj)

        z_floor = 0.0
        if mount_type == "ceiling":
            z_floor = max(0.0, ceiling_height - sz)

        aabb = aabb_from_center_size_rotation(
            cx=float(pred["x"]),
            cy=float(pred["y"]),
            sx=sx,
            sy=sy,
            sz=sz,
            yaw_deg=yaw,
            z_floor_m=z_floor,
        )

        out[idx] = {
            "index": idx,
            "name": extract_class_name(src_obj),
            "mount_type": mount_type,
            "size_m": [sx, sy, sz],
            "yaw_deg": yaw,
            "x": float(pred["x"]),
            "y": float(pred["y"]),
            "aabb": aabb,
            "wall_contact": classify_rect_wall_contact(aabb, poly_xy) if mount_type != "ceiling" else None,
        }

    return out


# ============================================================
# Room semantics helpers
# ============================================================

def parse_brief_flags(design_brief: str) -> Dict[str, bool]:
    s = (design_brief or "").lower()
    return {
        "bed_under_window": ("under the window" in s) or ("bed must be placed under the window" in s),
        "keep_entry_clear": ("entry path clear" in s) or ("clear walking path" in s) or ("clear path from the door" in s),
        "free_space_in_front_of_door": ("free space in front of the door" in s) or ("preserve free space in front of the door" in s),
        "wardrobe_back_to_wall": ("wardrobe must stand against a wall" in s) or ("wardrobe against a wall" in s) or ("back side touching the wall" in s),
        "wardrobe_not_divider": ("do not place the wardrobe as a divider" in s) or ("do not use the wardrobe as a divider" in s),
        "nightstands_symmetric": ("two symmetric nightstands" in s) or ("nightstands must be symmetric" in s) or ("symmetric on both sides of the bed" in s),
        "lamp_center": ("lamp near the center" in s) or ("ceiling lamp near the center" in s) or ("visual center of the composition" in s),
    }


def find_object_indices_by_name(src_items: List[Dict[str, Any]]) -> Dict[str, List[int]]:
    result: Dict[str, List[int]] = {
        "bed": [],
        "nightstand": [],
        "wardrobe": [],
        "dressing_table": [],
        "ceiling_lamp": [],
    }
    for idx, obj in enumerate(src_items):
        name = extract_class_name(obj).lower()
        if "bed" in name:
            result["bed"].append(idx)
        if "nightstand" in name:
            result["nightstand"].append(idx)
        if "wardrobe" in name:
            result["wardrobe"].append(idx)
        if "dressing table" in name:
            result["dressing_table"].append(idx)
        if "ceiling lamp" in name or "pendant lamp" in name or "lamp" in name:
            result["ceiling_lamp"].append(idx)
    return result


def estimate_window_wall_axis(payload: Dict[str, Any], poly_xy: List[Tuple[float, float]]) -> Optional[str]:
    opening_summary = (((payload.get("room") or {}).get("opening_summary") or {}).get("windows") or [])
    if not opening_summary:
        return None

    row = opening_summary[0]
    axis = row.get("wall_axis")
    if isinstance(axis, str) and axis in ("horizontal", "vertical"):
        return axis

    return None


def get_first_window_wall(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    windows = (((payload.get("room") or {}).get("opening_summary") or {}).get("windows") or [])
    if not windows:
        return None
    row = windows[0]
    wall_from = row.get("wall_from")
    wall_to = row.get("wall_to")
    if not (isinstance(wall_from, list) and isinstance(wall_to, list) and len(wall_from) == 2 and len(wall_to) == 2):
        return None

    x1, y1 = float(wall_from[0]), float(wall_from[1])
    x2, y2 = float(wall_to[0]), float(wall_to[1])

    center_hint = None
    if "s" in row and "width" in row:
        try:
            s = float(row["s"])
            width = float(row["width"])
            if abs(x1 - x2) >= abs(y1 - y2):
                start_x = x1
                end_x = x2
                if start_x >= end_x:
                    center_hint = start_x - s - width / 2.0
                else:
                    center_hint = start_x + s + width / 2.0
            else:
                start_y = y1
                end_y = y2
                if start_y >= end_y:
                    center_hint = start_y - s - width / 2.0
                else:
                    center_hint = start_y + s + width / 2.0
        except Exception:
            center_hint = None

    return {
        "wall_from": [x1, y1],
        "wall_to": [x2, y2],
        "axis": "horizontal" if abs(x1 - x2) >= abs(y1 - y2) else "vertical",
        "center_hint": center_hint,
    }


def door_clearance_box(payload: Dict[str, Any], poly_xy: List[Tuple[float, float]]) -> Optional[Dict[str, float]]:
    doors = (((payload.get("room") or {}).get("opening_summary") or {}).get("doors") or [])
    if not doors:
        return None

    door = doors[0]
    wall_from = door.get("wall_from")
    wall_to = door.get("wall_to")
    if not (
        isinstance(wall_from, list)
        and isinstance(wall_to, list)
        and len(wall_from) == 2
        and len(wall_to) == 2
    ):
        return None

    x1, y1 = float(wall_from[0]), float(wall_from[1])
    x2, y2 = float(wall_to[0]), float(wall_to[1])

    wall_axis = "horizontal" if abs(x1 - x2) >= abs(y1 - y2) else "vertical"
    depth_hint = 1.0

    if wall_axis == "horizontal":
        x_mid = 0.5 * (x1 + x2)
        y = y1
        x_min = x_mid - 0.6
        x_max = x_mid + 0.6
        room_cx, room_cy = polygon_centroid(poly_xy)
        if room_cy >= y:
            return {"x_min": x_min, "x_max": x_max, "y_min": y, "y_max": y + depth_hint}
        return {"x_min": x_min, "x_max": x_max, "y_min": y - depth_hint, "y_max": y}
    else:
        y_mid = 0.5 * (y1 + y2)
        x = x1
        y_min = y_mid - 0.6
        y_max = y_mid + 0.6
        room_cx, room_cy = polygon_centroid(poly_xy)
        if room_cx >= x:
            return {"x_min": x, "x_max": x + depth_hint, "y_min": y_min, "y_max": y_max}
        return {"x_min": x - depth_hint, "x_max": x, "y_min": y_min, "y_max": y_max}


# ============================================================
# Deterministic semantic validation
# ============================================================

def evaluate_layout_semantics(
    payload: Dict[str, Any],
    src_items: List[Dict[str, Any]],
    llm_layout: Dict[str, Any],
    poly_xy: List[Tuple[float, float]],
    ceiling_height: float,
    design_brief: str,
) -> Tuple[List[str], Dict[str, Any]]:
    issues: List[str] = []
    metrics: Dict[str, Any] = {}

    flags = parse_brief_flags(design_brief)
    placement_map = build_placement_map(src_items, llm_layout, poly_xy, ceiling_height)
    name_map = find_object_indices_by_name(src_items)
    x_min, x_max, y_min, y_max = polygon_bbox(poly_xy)
    cx, cy = polygon_centroid(poly_xy)

    floor_rows = [row for row in placement_map.values() if row["mount_type"] != "ceiling"]
    if floor_rows:
        left = sum(1 for r in floor_rows if r["x"] < cx)
        right = sum(1 for r in floor_rows if r["x"] >= cx)
        bottom = sum(1 for r in floor_rows if r["y"] < cy)
        top = sum(1 for r in floor_rows if r["y"] >= cy)
        metrics["spread_counts"] = {"left": left, "right": right, "bottom": bottom, "top": top}

        if max(left, right) >= max(4, len(floor_rows)):
            issues.append("composition_overconcentrated_left_right")
        if max(bottom, top) >= max(4, len(floor_rows)):
            issues.append("composition_overconcentrated_bottom_top")

    bed_idxs = name_map["bed"]
    bed_row = placement_map.get(bed_idxs[0]) if bed_idxs else None
    if bed_row is not None:
        bed_aabb = bed_row["aabb"]
        bed_wc = bed_row["wall_contact"]
        metrics["bed_wall_contact"] = bed_wc

        if flags["bed_under_window"]:
            window_axis = estimate_window_wall_axis(payload, poly_xy)
            metrics["window_wall_axis"] = window_axis
            if window_axis == "horizontal":
                if not (abs(bed_aabb["y_max"] - y_max) <= 0.30 or abs(bed_aabb["y_min"] - y_min) <= 0.30):
                    issues.append("bed_not_on_window_wall")
            elif window_axis == "vertical":
                if not (abs(bed_aabb["x_max"] - x_max) <= 0.30 or abs(bed_aabb["x_min"] - x_min) <= 0.30):
                    issues.append("bed_not_on_window_wall")

        side_clearances = []
        for axis in ("left", "right", "bottom", "top"):
            min_gap = 1e9
            for other in floor_rows:
                if other["index"] == bed_row["index"]:
                    continue
                oa = other["aabb"]
                if axis == "left":
                    overlaps_y = not (oa["y_max"] <= bed_aabb["y_min"] or oa["y_min"] >= bed_aabb["y_max"])
                    if overlaps_y and oa["x_max"] <= bed_aabb["x_min"]:
                        min_gap = min(min_gap, bed_aabb["x_min"] - oa["x_max"])
                elif axis == "right":
                    overlaps_y = not (oa["y_max"] <= bed_aabb["y_min"] or oa["y_min"] >= bed_aabb["y_max"])
                    if overlaps_y and oa["x_min"] >= bed_aabb["x_max"]:
                        min_gap = min(min_gap, oa["x_min"] - bed_aabb["x_max"])
                elif axis == "bottom":
                    overlaps_x = not (oa["x_max"] <= bed_aabb["x_min"] or oa["x_min"] >= bed_aabb["x_max"])
                    if overlaps_x and oa["y_max"] <= bed_aabb["y_min"]:
                        min_gap = min(min_gap, bed_aabb["y_min"] - oa["y_max"])
                else:
                    overlaps_x = not (oa["x_max"] <= bed_aabb["x_min"] or oa["x_min"] >= bed_aabb["x_max"])
                    if overlaps_x and oa["y_min"] >= bed_aabb["y_max"]:
                        min_gap = min(min_gap, oa["y_min"] - bed_aabb["y_max"])

            if min_gap >= 1e8:
                if axis == "left":
                    min_gap = bed_aabb["x_min"] - x_min
                elif axis == "right":
                    min_gap = x_max - bed_aabb["x_max"]
                elif axis == "bottom":
                    min_gap = bed_aabb["y_min"] - y_min
                else:
                    min_gap = y_max - bed_aabb["y_max"]

            side_clearances.append((axis, round(min_gap, 3)))
        metrics["bed_side_clearances"] = side_clearances
        if max(g for _, g in side_clearances) < 0.55:
            issues.append("bed_has_poor_side_access")

    night_idxs = name_map["nightstand"]
    if flags["nightstands_symmetric"] and bed_row is not None and len(night_idxs) >= 2:
        n1 = placement_map.get(night_idxs[0])
        n2 = placement_map.get(night_idxs[1])
        if n1 is not None and n2 is not None:
            dx1 = n1["x"] - bed_row["x"]
            dx2 = n2["x"] - bed_row["x"]
            dy1 = n1["y"] - bed_row["y"]
            dy2 = n2["y"] - bed_row["y"]
            symmetry_error = min(
                abs(dx1 + dx2) + abs(dy1 - dy2),
                abs(dx1 - dx2) + abs(dy1 + dy2),
            )
            metrics["nightstand_symmetry_error"] = round(symmetry_error, 3)
            if symmetry_error > 0.80:
                issues.append("nightstands_not_symmetric")

    wardrobe_idxs = name_map["wardrobe"]
    if wardrobe_idxs:
        wr = placement_map.get(wardrobe_idxs[0])
        if wr is not None and wr["wall_contact"] is not None:
            metrics["wardrobe_wall_contact"] = wr["wall_contact"]

            if flags["wardrobe_back_to_wall"] and not bool(wr["wall_contact"]["back_to_wall"]):
                issues.append("wardrobe_not_back_to_wall")

            if flags["wardrobe_not_divider"]:
                wa = wr["aabb"]
                room_w = x_max - x_min
                room_h = y_max - y_min
                central_band_x = (cx - 0.15 * room_w, cx + 0.15 * room_w)
                central_band_y = (cy - 0.15 * room_h, cy + 0.15 * room_h)
                overlap_x = not (wa["x_max"] <= central_band_x[0] or wa["x_min"] >= central_band_x[1])
                overlap_y = not (wa["y_max"] <= central_band_y[0] or wa["y_min"] >= central_band_y[1])
                metrics["wardrobe_central_overlap"] = {"x_band": overlap_x, "y_band": overlap_y}
                if overlap_x or overlap_y:
                    issues.append("wardrobe_acts_like_divider")

    dressing_idxs = name_map["dressing_table"]
    if dressing_idxs:
        dr = placement_map.get(dressing_idxs[0])
        if dr is not None:
            da = dr["aabb"]
            forward_box = {
                "x_min": da["x_min"] - 0.25,
                "x_max": da["x_max"] + 0.25,
                "y_min": da["y_min"] - 0.75,
                "y_max": da["y_max"] + 0.75,
            }
            best_overlap = 0.0
            for other in floor_rows:
                if other["index"] == dr["index"]:
                    continue
                best_overlap = max(best_overlap, aabb_overlap_ratio(forward_box, other["aabb"]))
            metrics["dressing_table_access_overlap"] = round(best_overlap, 3)
            if best_overlap > 0.25:
                issues.append("dressing_table_front_access_poor")

    lamp_idxs = name_map["ceiling_lamp"]
    if flags["lamp_center"] and lamp_idxs:
        lamp = placement_map.get(lamp_idxs[0])
        if lamp is not None:
            dist_center = math.hypot(lamp["x"] - cx, lamp["y"] - cy)
            metrics["lamp_distance_to_room_center"] = round(dist_center, 3)
            if dist_center > 1.30:
                issues.append("ceiling_lamp_too_far_from_center")

    door_box = door_clearance_box(payload, poly_xy)
    if door_box is not None:
        metrics["door_clearance_box"] = door_box
        bad = False
        for row in floor_rows:
            if rects_overlap_2d(row["aabb"], door_box, eps=1e-6):
                bad = True
                break
        if bad and (flags["keep_entry_clear"] or flags["free_space_in_front_of_door"]):
            issues.append("door_clearance_blocked")

    if floor_rows:
        xs = [r["x"] for r in floor_rows]
        ys = [r["y"] for r in floor_rows]
        spread_x = max(xs) - min(xs)
        spread_y = max(ys) - min(ys)
        metrics["object_spread_xy"] = [round(spread_x, 3), round(spread_y, 3)]
        if spread_x < 1.7:
            issues.append("composition_spread_x_too_small")
        if spread_y < 1.2:
            issues.append("composition_spread_y_too_small")

    return sorted(set(issues)), metrics


# ============================================================
# Deterministic repair
# ============================================================

def _occupied_except(layout: Dict[str, Any], src_items: List[Dict[str, Any]], skip_indices: set[int]) -> List[Dict[str, float]]:
    occupied: List[Dict[str, float]] = []
    by_index = {int(p["index"]): p for p in layout.get("placements", [])}
    for idx, obj in enumerate(src_items):
        if idx in skip_indices or idx not in by_index:
            continue
        p = by_index[idx]
        sx, sy, sz = extract_size_m(obj)
        occupied.append(
            aabb_from_center_size_rotation(
                cx=float(p["x"]),
                cy=float(p["y"]),
                sx=sx,
                sy=sy,
                sz=sz,
                yaw_deg=float(p["yaw_deg"]),
                z_floor_m=0.0,
            )
        )
    return occupied


def _set_layout_row(layout_map: Dict[int, Dict[str, Any]], idx: int, x: float, y: float, yaw_deg: int) -> None:
    layout_map[idx] = {
        "index": int(idx),
        "x": float(x),
        "y": float(y),
        "yaw_deg": int(quantize_rot_0_90_180_270(yaw_deg)),
    }


def _find_valid_on_wall(
    sx: float,
    sy: float,
    sz: float,
    wall_axis: str,
    wall_value: float,
    along_candidates: List[float],
    polygon_xy: List[Tuple[float, float]],
    occupied: List[Dict[str, float]],
    forbidden: List[Dict[str, float]],
) -> Optional[Tuple[float, float, int]]:
    yaw_options = [0, 90]
    for yaw in yaw_options:
        rot = quantize_rot_0_90_180_270(yaw)
        use_sx, use_sy = sx, sy
        if rot in (90.0, 270.0):
            use_sx, use_sy = sy, sx

        if wall_axis == "horizontal":
            for side in ("min", "max"):
                if side == "min":
                    cy = wall_value + use_sy / 2.0
                else:
                    cy = wall_value - use_sy / 2.0
                for cx in along_candidates:
                    if not rect_inside_polygon(cx, cy, sx, sy, yaw, polygon_xy):
                        continue
                    aabb = aabb_from_center_size_rotation(cx, cy, sx, sy, sz, yaw)
                    if any(rects_overlap_2d(aabb, occ) for occ in occupied):
                        continue
                    if any(rects_overlap_2d(aabb, bad) for bad in forbidden):
                        continue
                    return cx, cy, int(rot)
        else:
            for side in ("min", "max"):
                if side == "min":
                    cx = wall_value + use_sx / 2.0
                else:
                    cx = wall_value - use_sx / 2.0
                for cy in along_candidates:
                    if not rect_inside_polygon(cx, cy, sx, sy, yaw, polygon_xy):
                        continue
                    aabb = aabb_from_center_size_rotation(cx, cy, sx, sy, sz, yaw)
                    if any(rects_overlap_2d(aabb, occ) for occ in occupied):
                        continue
                    if any(rects_overlap_2d(aabb, bad) for bad in forbidden):
                        continue
                    return cx, cy, int(rot)
    return None


def repair_layout_deterministically(
    room: Dict[str, Any],
    payload: Dict[str, Any],
    src_items: List[Dict[str, Any]],
    llm_layout: Dict[str, Any],
    design_brief: str,
) -> Dict[str, Any]:
    poly_xy = extract_room_polygon_xy(room)
    x_min, x_max, y_min, y_max = polygon_bbox(poly_xy)
    cx, cy = polygon_centroid(poly_xy)
    flags = parse_brief_flags(design_brief)
    name_map = find_object_indices_by_name(src_items)
    door_box = door_clearance_box(payload, poly_xy)
    forbidden = [door_box] if door_box is not None else []

    layout_map: Dict[int, Dict[str, Any]] = {
        int(p["index"]): {
            "index": int(p["index"]),
            "x": float(p["x"]),
            "y": float(p["y"]),
            "yaw_deg": int(quantize_rot_0_90_180_270(float(p["yaw_deg"]))),
        }
        for p in llm_layout.get("placements", [])
    }

    for idx, obj in enumerate(src_items):
        if idx not in layout_map:
            layout_map[idx] = {"index": idx, "x": cx, "y": cy, "yaw_deg": 0}

    bed_idxs = name_map["bed"]
    if bed_idxs:
        bed_idx = bed_idxs[0]
        bed_obj = src_items[bed_idx]
        bsx, bsy, bsz = extract_size_m(bed_obj)

        occupied = _occupied_except(layout_map_to_layout(layout_map), src_items, {bed_idx})
        window_wall = get_first_window_wall(payload)

        if flags["bed_under_window"] and window_wall is not None:
            if window_wall["axis"] == "horizontal":
                target_x = float(window_wall["center_hint"]) if window_wall["center_hint"] is not None else cx
                target_x = min(max(target_x, x_min + 0.2), x_max - 0.2)

                candidates = [target_x]
                for d in [0.0, 0.2, -0.2, 0.4, -0.4, 0.6, -0.6]:
                    candidates.append(target_x + d)

                res = _find_valid_on_wall(
                    sx=bsx,
                    sy=bsy,
                    sz=bsz,
                    wall_axis="horizontal",
                    wall_value=y_max if abs(window_wall["wall_from"][1] - y_max) < abs(window_wall["wall_from"][1] - y_min) else y_min,
                    along_candidates=candidates,
                    polygon_xy=poly_xy,
                    occupied=occupied,
                    forbidden=forbidden,
                )
                if res is not None:
                    bx, by, byaw = res
                    _set_layout_row(layout_map, bed_idx, bx, by, byaw)
            else:
                target_y = float(window_wall["center_hint"]) if window_wall["center_hint"] is not None else cy
                target_y = min(max(target_y, y_min + 0.2), y_max - 0.2)

                candidates = [target_y]
                for d in [0.0, 0.2, -0.2, 0.4, -0.4, 0.6, -0.6]:
                    candidates.append(target_y + d)

                res = _find_valid_on_wall(
                    sx=bsx,
                    sy=bsy,
                    sz=bsz,
                    wall_axis="vertical",
                    wall_value=x_max if abs(window_wall["wall_from"][0] - x_max) < abs(window_wall["wall_from"][0] - x_min) else x_min,
                    along_candidates=candidates,
                    polygon_xy=poly_xy,
                    occupied=occupied,
                    forbidden=forbidden,
                )
                if res is not None:
                    bx, by, byaw = res
                    _set_layout_row(layout_map, bed_idx, bx, by, byaw)

    if bed_idxs and len(name_map["nightstand"]) >= 2:
        bed_idx = bed_idxs[0]
        bed = layout_map[bed_idx]
        bsx, bsy, _ = extract_size_m(src_items[bed_idx])
        bed_rot = quantize_rot_0_90_180_270(int(bed["yaw_deg"]))

        ns1, ns2 = name_map["nightstand"][:2]
        n1sx, n1sy, _ = extract_size_m(src_items[ns1])
        n2sx, n2sy, _ = extract_size_m(src_items[ns2])

        offset_x = max(0.25, bsx / 2.0 + max(n1sx, n1sy, n2sx, n2sy) / 2.0 + 0.08)
        offset_y = max(0.25, bsy / 2.0 + max(n1sx, n1sy, n2sx, n2sy) / 2.0 + 0.08)

        occupied_base = _occupied_except(layout_map_to_layout(layout_map), src_items, {ns1, ns2})

        if bed_rot in (0.0, 180.0):
            cands = [
                (float(bed["x"]) - offset_x, float(bed["y"]), 0),
                (float(bed["x"]) + offset_x, float(bed["y"]), 0),
            ]
        else:
            cands = [
                (float(bed["x"]), float(bed["y"]) - offset_y, 90),
                (float(bed["x"]), float(bed["y"]) + offset_y, 90),
            ]

        rows = []
        for idx, (tx, ty, tyaw) in zip([ns1, ns2], cands):
            sx, sy, sz = extract_size_m(src_items[idx])
            try:
                fx, fy = find_nearest_valid_position(
                    target_x=tx,
                    target_y=ty,
                    sx=sx,
                    sy=sy,
                    sz=sz,
                    yaw_deg=tyaw,
                    polygon_xy=poly_xy,
                    occupied=occupied_base,
                    forbidden=forbidden,
                    grid_step=0.05,
                    max_radius_steps=60,
                )
                aabb = aabb_from_center_size_rotation(fx, fy, sx, sy, sz, tyaw)
                occupied_base.append(aabb)
                rows.append((idx, fx, fy, tyaw))
            except Exception:
                pass

        if len(rows) == 2:
            for idx, fx, fy, tyaw in rows:
                _set_layout_row(layout_map, idx, fx, fy, tyaw)

    wardrobe_idxs = name_map["wardrobe"]
    if wardrobe_idxs:
        wr_idx = wardrobe_idxs[0]
        wsx, wsy, wsz = extract_size_m(src_items[wr_idx])

        occupied = _occupied_except(layout_map_to_layout(layout_map), src_items, {wr_idx})

        along_y = [
            y_min + 0.6,
            y_max - 0.6,
            cy - 0.8,
            cy + 0.8,
            cy - 1.2,
            cy + 1.2,
            cy,
        ]
        along_y = [min(max(v, y_min + 0.2), y_max - 0.2) for v in along_y]

        res = _find_valid_on_wall(
            sx=wsx,
            sy=wsy,
            sz=wsz,
            wall_axis="vertical",
            wall_value=x_min,
            along_candidates=along_y,
            polygon_xy=poly_xy,
            occupied=occupied,
            forbidden=forbidden,
        )
        if res is None:
            res = _find_valid_on_wall(
                sx=wsx,
                sy=wsy,
                sz=wsz,
                wall_axis="vertical",
                wall_value=x_max,
                along_candidates=along_y,
                polygon_xy=poly_xy,
                occupied=occupied,
                forbidden=forbidden,
            )
        if res is None:
            along_x = [
                x_min + 0.7,
                x_max - 0.7,
                cx - 1.2,
                cx + 1.2,
                cx - 0.8,
                cx + 0.8,
            ]
            along_x = [min(max(v, x_min + 0.2), x_max - 0.2) for v in along_x]
            res = _find_valid_on_wall(
                sx=wsx,
                sy=wsy,
                sz=wsz,
                wall_axis="horizontal",
                wall_value=y_min,
                along_candidates=along_x,
                polygon_xy=poly_xy,
                occupied=occupied,
                forbidden=forbidden,
            )
        if res is None:
            along_x = [
                x_min + 0.7,
                x_max - 0.7,
                cx - 1.2,
                cx + 1.2,
                cx - 0.8,
                cx + 0.8,
            ]
            along_x = [min(max(v, x_min + 0.2), x_max - 0.2) for v in along_x]
            res = _find_valid_on_wall(
                sx=wsx,
                sy=wsy,
                sz=wsz,
                wall_axis="horizontal",
                wall_value=y_max,
                along_candidates=along_x,
                polygon_xy=poly_xy,
                occupied=occupied,
                forbidden=forbidden,
            )

        if res is not None:
            wx, wy, wyaw = res
            _set_layout_row(layout_map, wr_idx, wx, wy, wyaw)

    dressing_idxs = name_map["dressing_table"]
    if dressing_idxs:
        dr_idx = dressing_idxs[0]
        dsx, dsy, dsz = extract_size_m(src_items[dr_idx])

        occupied = _occupied_except(layout_map_to_layout(layout_map), src_items, {dr_idx})

        candidate_points = [
            (x_max - 0.8, y_min + 0.9, 90),
            (x_max - 0.8, y_max - 0.9, 90),
            (x_min + 0.8, y_min + 0.9, 90),
            (x_min + 0.8, y_max - 0.9, 90),
            (x_max - 1.0, cy, 90),
            (x_min + 1.0, cy, 90),
        ]

        for tx, ty, tyaw in candidate_points:
            try:
                fx, fy = find_nearest_valid_position(
                    target_x=tx,
                    target_y=ty,
                    sx=dsx,
                    sy=dsy,
                    sz=dsz,
                    yaw_deg=tyaw,
                    polygon_xy=poly_xy,
                    occupied=occupied,
                    forbidden=forbidden,
                    grid_step=0.05,
                    max_radius_steps=80,
                )
                _set_layout_row(layout_map, dr_idx, fx, fy, tyaw)
                break
            except Exception:
                continue

    lamp_idxs = name_map["ceiling_lamp"]
    if lamp_idxs:
        lamp_idx = lamp_idxs[0]
        lsx, lsy, _ = extract_size_m(src_items[lamp_idx])
        tx, ty = cx, cy
        if not point_in_polygon(tx, ty, poly_xy):
            tx = min(max(tx, x_min + lsx / 2.0), x_max - lsx / 2.0)
            ty = min(max(ty, y_min + lsy / 2.0), y_max - lsy / 2.0)
        _set_layout_row(layout_map, lamp_idx, tx, ty, 0)

    floor_positions = []
    for idx, obj in enumerate(src_items):
        if extract_mount_type(obj) != "ceiling":
            row = layout_map[idx]
            floor_positions.append((idx, float(row["x"]), float(row["y"])))

    if floor_positions:
        xs = [x for _, x, _ in floor_positions]
        ys = [y for _, _, y in floor_positions]
        spread_x = max(xs) - min(xs)
        spread_y = max(ys) - min(ys)

        if spread_x < 1.5 or spread_y < 1.0:
            movable = [idx for idx, _, _ in floor_positions if idx not in set(bed_idxs + wardrobe_idxs)]
            push_targets = [
                (x_min + 0.9, y_min + 1.0),
                (x_max - 0.9, y_min + 1.0),
                (x_min + 0.9, y_max - 1.0),
                (x_max - 0.9, y_max - 1.0),
            ]
            for idx, (tx, ty) in zip(movable, push_targets):
                sx, sy, sz = extract_size_m(src_items[idx])
                occupied = _occupied_except(layout_map_to_layout(layout_map), src_items, {idx})
                try:
                    fx, fy = find_nearest_valid_position(
                        target_x=tx,
                        target_y=ty,
                        sx=sx,
                        sy=sy,
                        sz=sz,
                        yaw_deg=layout_map[idx]["yaw_deg"],
                        polygon_xy=poly_xy,
                        occupied=occupied,
                        forbidden=forbidden,
                        grid_step=0.05,
                        max_radius_steps=80,
                    )
                    _set_layout_row(layout_map, idx, fx, fy, int(layout_map[idx]["yaw_deg"]))
                except Exception:
                    continue

    repaired = {"placements": [layout_map[i] for i in sorted(layout_map.keys())]}
    return repaired


def layout_map_to_layout(layout_map: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    return {"placements": [layout_map[i] for i in sorted(layout_map.keys())]}


# ============================================================
# Hard-constraint projection before critic
# ============================================================

def oriented_size_xy(size_m: List[float], yaw_deg: float) -> Tuple[float, float]:
    sx, sy, _ = size_m
    yaw_q = int(quantize_rot_0_90_180_270(yaw_deg))
    if yaw_q in (90, 270):
        return sy, sx
    return sx, sy


def get_window_wall_side(payload: Dict[str, Any]) -> Optional[str]:
    windows = (((payload.get("room") or {}).get("opening_summary") or {}).get("windows") or [])
    if not windows:
        return None

    row = windows[0]
    wall_from = row.get("wall_from")
    wall_to = row.get("wall_to")
    if not (
        isinstance(wall_from, list) and len(wall_from) == 2 and
        isinstance(wall_to, list) and len(wall_to) == 2
    ):
        return None

    x1, y1 = float(wall_from[0]), float(wall_from[1])
    x2, y2 = float(wall_to[0]), float(wall_to[1])

    if abs(y1 - y2) < 1e-6:
        return "top" if y1 > 0 else "bottom"
    if abs(x1 - x2) < 1e-6:
        return "right" if x1 > 0 else "left"
    return None


def get_window_center(payload: Dict[str, Any], poly_xy: List[Tuple[float, float]]) -> Tuple[float, float]:
    windows = (((payload.get("room") or {}).get("opening_summary") or {}).get("windows") or [])
    if not windows:
        return polygon_centroid(poly_xy)

    row = windows[0]
    wall_from = row.get("wall_from")
    wall_to = row.get("wall_to")
    if not (
        isinstance(wall_from, list) and len(wall_from) == 2 and
        isinstance(wall_to, list) and len(wall_to) == 2
    ):
        return polygon_centroid(poly_xy)

    x1, y1 = float(wall_from[0]), float(wall_from[1])
    x2, y2 = float(wall_to[0]), float(wall_to[1])

    s = row.get("s")
    width = row.get("width")
    if isinstance(s, (int, float)) and isinstance(width, (int, float)):
        wall_len = math.hypot(x2 - x1, y2 - y1)
        if wall_len > 1e-9:
            ux = (x2 - x1) / wall_len
            uy = (y2 - y1) / wall_len
            cx = x1 + ux * (float(s) + float(width) / 2.0)
            cy = y1 + uy * (float(s) + float(width) / 2.0)
            return cx, cy

    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def place_bed_on_window_wall(
    payload: Dict[str, Any],
    src_items: List[Dict[str, Any]],
    llm_layout: Dict[str, Any],
    poly_xy: List[Tuple[float, float]],
) -> None:
    name_map = find_object_indices_by_name(src_items)
    if not name_map["bed"]:
        return

    bed_idx = name_map["bed"][0]
    placements = {int(p["index"]): p for p in llm_layout["placements"]}
    if bed_idx not in placements:
        return

    row = placements[bed_idx]
    size_m = extract_size_m(src_items[bed_idx])

    window_side = get_window_wall_side(payload)
    wx, wy = get_window_center(payload, poly_xy)
    x_min, x_max, y_min, y_max = polygon_bbox(poly_xy)

    yaw = int(quantize_rot_0_90_180_270(row.get("yaw_deg", 0)))

    if window_side in {"top", "bottom"}:
        yaw = 0
    elif window_side in {"left", "right"}:
        yaw = 90

    sx, sy = oriented_size_xy(size_m, yaw)

    if window_side == "top":
        row["x"] = wx
        row["y"] = y_max - sy / 2.0
    elif window_side == "bottom":
        row["x"] = wx
        row["y"] = y_min + sy / 2.0
    elif window_side == "left":
        row["x"] = x_min + sx / 2.0
        row["y"] = wy
    elif window_side == "right":
        row["x"] = x_max - sx / 2.0
        row["y"] = wy

    row["yaw_deg"] = yaw


def symmetrize_nightstands_around_bed(
    src_items: List[Dict[str, Any]],
    llm_layout: Dict[str, Any],
) -> None:
    name_map = find_object_indices_by_name(src_items)
    if len(name_map["bed"]) < 1 or len(name_map["nightstand"]) < 2:
        return

    placements = {int(p["index"]): p for p in llm_layout["placements"]}
    bed_idx = name_map["bed"][0]
    n1_idx, n2_idx = name_map["nightstand"][:2]

    if bed_idx not in placements or n1_idx not in placements or n2_idx not in placements:
        return

    bed = placements[bed_idx]
    n1 = placements[n1_idx]
    n2 = placements[n2_idx]

    bed_sx, bed_sy = oriented_size_xy(extract_size_m(src_items[bed_idx]), bed["yaw_deg"])
    n1_sx, n1_sy = oriented_size_xy(extract_size_m(src_items[n1_idx]), bed["yaw_deg"])
    n2_sx, n2_sy = oriented_size_xy(extract_size_m(src_items[n2_idx]), bed["yaw_deg"])

    gap = 0.10

    if int(quantize_rot_0_90_180_270(bed["yaw_deg"])) in (0, 180):
        offset = bed_sx / 2.0 + max(n1_sx, n2_sx) / 2.0 + gap
        y_same = bed["y"]
        n1["x"] = bed["x"] - offset
        n2["x"] = bed["x"] + offset
        n1["y"] = y_same
        n2["y"] = y_same
        n1["yaw_deg"] = bed["yaw_deg"]
        n2["yaw_deg"] = bed["yaw_deg"]
    else:
        offset = bed_sy / 2.0 + max(n1_sy, n2_sy) / 2.0 + gap
        x_same = bed["x"]
        n1["x"] = x_same
        n2["x"] = x_same
        n1["y"] = bed["y"] - offset
        n2["y"] = bed["y"] + offset
        n1["yaw_deg"] = bed["yaw_deg"]
        n2["yaw_deg"] = bed["yaw_deg"]


def place_lamp_in_center(
    payload: Dict[str, Any],
    src_items: List[Dict[str, Any]],
    llm_layout: Dict[str, Any],
    poly_xy: List[Tuple[float, float]],
) -> None:
    name_map = find_object_indices_by_name(src_items)
    if not name_map["ceiling_lamp"]:
        return

    lamp_idx = name_map["ceiling_lamp"][0]
    placements = {int(p["index"]): p for p in llm_layout["placements"]}
    if lamp_idx not in placements:
        return

    cx, cy = polygon_centroid(poly_xy)
    placements[lamp_idx]["x"] = cx
    placements[lamp_idx]["y"] = cy
    placements[lamp_idx]["yaw_deg"] = 0


def place_wardrobe_on_side_wall(
    payload: Dict[str, Any],
    src_items: List[Dict[str, Any]],
    llm_layout: Dict[str, Any],
    poly_xy: List[Tuple[float, float]],
) -> None:
    name_map = find_object_indices_by_name(src_items)
    if not name_map["wardrobe"]:
        return

    wardrobe_idx = name_map["wardrobe"][0]
    placements = {int(p["index"]): p for p in llm_layout["placements"]}
    if wardrobe_idx not in placements:
        return

    row = placements[wardrobe_idx]
    x_min, x_max, _, _ = polygon_bbox(poly_xy)
    _, cy = polygon_centroid(poly_xy)

    yaw = 90
    sx, _ = oriented_size_xy(extract_size_m(src_items[wardrobe_idx]), yaw)

    row["x"] = x_max - sx / 2.0
    row["y"] = cy
    row["yaw_deg"] = yaw


def apply_hard_constraints_before_critic(
    payload: Dict[str, Any],
    src_items: List[Dict[str, Any]],
    llm_layout: Dict[str, Any],
    poly_xy: List[Tuple[float, float]],
) -> Dict[str, Any]:
    out = {
        "placements": [
            {
                "index": int(p["index"]),
                "x": float(p["x"]),
                "y": float(p["y"]),
                "yaw_deg": int(quantize_rot_0_90_180_270(p["yaw_deg"])),
            }
            for p in llm_layout.get("placements", [])
        ]
    }

    place_bed_on_window_wall(payload, src_items, out, poly_xy)
    symmetrize_nightstands_around_bed(src_items, out)
    place_wardrobe_on_side_wall(payload, src_items, out, poly_xy)
    place_lamp_in_center(payload, src_items, out, poly_xy)

    return out


# ============================================================
# Stage runners
# ============================================================

def run_plan_stage_raw(
    model_name: str,
    base_url: str,
    system_prompt: str,
    user_prompt: str,
    output_schema: Dict[str, Any],
    timeout_sec: int,
    temperature: float,
    think_mode: Optional[str],
    debug_dir: Path,
) -> str:
    debug_dir.mkdir(parents=True, exist_ok=True)
    save_text(debug_dir / "system_prompt.txt", system_prompt)
    save_text(debug_dir / "user_prompt.txt", user_prompt)
    save_json(debug_dir / "output_schema.json", output_schema)

    resp = chat_json(
        base_url=base_url,
        model=model_name,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        json_schema=output_schema,
        timeout_sec=timeout_sec,
        temperature=temperature,
        think=think_mode,
    )

    save_json(debug_dir / "raw_response.json", resp)

    content = extract_text_from_ollama_response(resp)
    if not content:
        raise ValueError(
            "Пустой текстовый ответ модели на этапе plan. "
            f"Raw Ollama response:\n{short_json(resp)}"
        )

    save_text(debug_dir / "raw_text.txt", content)
    return content


def run_plan_repair_stage(
    model_name: str,
    base_url: str,
    payload: Dict[str, Any],
    design_brief: str,
    raw_plan_text: str,
    output_schema: Dict[str, Any],
    timeout_sec: int,
    temperature: float,
    think_mode: Optional[str],
    debug_dir: Path,
    n_objects: int,
) -> Dict[str, Any]:
    debug_dir.mkdir(parents=True, exist_ok=True)

    system_prompt = build_plan_repair_system_prompt()
    user_prompt = build_plan_repair_user_prompt(
        payload=payload,
        design_brief=design_brief,
        raw_plan_text=raw_plan_text,
        n_objects=n_objects,
    )

    save_text(debug_dir / "system_prompt.txt", system_prompt)
    save_text(debug_dir / "user_prompt.txt", user_prompt)
    save_json(debug_dir / "output_schema.json", output_schema)

    resp = chat_json(
        base_url=base_url,
        model=model_name,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        json_schema=output_schema,
        timeout_sec=timeout_sec,
        temperature=temperature,
        think=think_mode,
    )
    save_json(debug_dir / "raw_response.json", resp)

    content = extract_text_from_ollama_response(resp)
    if not content:
        raise ValueError(
            "Пустой текстовый ответ модели на этапе plan_repair. "
            f"Raw Ollama response:\n{short_json(resp)}"
        )

    save_text(debug_dir / "raw_text.txt", content)

    validation = validate_plan_structure(content, n_objects=n_objects)
    if not validation.ok or validation.normalized is None:
        raise ValueError(validation.feedback or "Plan repair failed validation.")

    save_json(debug_dir / "normalized_plan.json", validation.normalized)
    return validation.normalized


def run_json_stage_with_retry(
    model_name: str,
    base_url: str,
    system_prompt: str,
    user_prompt: str,
    output_schema: Dict[str, Any],
    timeout_sec: int,
    temperature: float,
    max_llm_attempts: int,
    debug_dir: Path,
    n_objects: int,
    think_mode: Optional[str],
) -> Tuple[Dict[str, Any], int]:
    raw_counter = {"value": 0}

    debug_dir.mkdir(parents=True, exist_ok=True)
    save_text(debug_dir / "system_prompt.txt", system_prompt)
    save_text(debug_dir / "user_prompt.txt", user_prompt)
    save_json(debug_dir / "output_schema.json", output_schema)

    def _generate(_: str) -> str:
        raw_counter["value"] += 1
        call_id = raw_counter["value"]

        resp = chat_json(
            base_url=base_url,
            model=model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_schema=output_schema,
            timeout_sec=timeout_sec,
            temperature=temperature,
            think=think_mode,
        )

        save_json(debug_dir / f"raw_response_{call_id:02d}.json", resp)

        content = extract_text_from_ollama_response(resp)
        if content:
            save_text(debug_dir / f"raw_text_{call_id:02d}.txt", content)
            return content

        raise ValueError(
            "Пустой текстовый ответ модели на этапе json. "
            f"Raw Ollama response:\n{short_json(resp)}"
        )

    def _validate(raw_text: str) -> ValidationResult[Dict[str, Any]]:
        return validate_structure(raw_text=raw_text, n_objects=n_objects)

    retry_result = run_retry_loop(
        generate_fn=_generate,
        validate_fn=_validate,
        initial_prompt=user_prompt,
        max_attempts=max_llm_attempts,
        debug_dir=str(debug_dir),
    )

    return retry_result.normalized, retry_result.attempts_used


def run_critic_stage(
    model_name: str,
    base_url: str,
    system_prompt: str,
    user_prompt: str,
    schema: Dict[str, Any],
    timeout_sec: int,
    temperature: float,
    think_mode: Optional[str],
    debug_dir: Path,
) -> Dict[str, Any]:
    debug_dir.mkdir(parents=True, exist_ok=True)
    save_text(debug_dir / "system_prompt.txt", system_prompt)
    save_text(debug_dir / "user_prompt.txt", user_prompt)
    save_json(debug_dir / "schema.json", schema)

    resp = chat_json(
        base_url=base_url,
        model=model_name,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        json_schema=schema,
        timeout_sec=timeout_sec,
        temperature=temperature,
        think=think_mode,
    )

    save_json(debug_dir / "raw_response.json", resp)
    content = extract_text_from_ollama_response(resp)
    if not content:
        raise ValueError(
            "Пустой ответ модели на этапе critic. "
            f"Raw Ollama response:\n{short_json(resp)}"
        )
    save_text(debug_dir / "raw_text.txt", content)

    result = validate_critic_response(content)
    if not result.ok or result.normalized is None:
        raise ValueError(result.feedback or "Invalid critic response.")

    normalized = normalize_critic_payload(result.normalized)
    save_json(debug_dir / "normalized_critic.json", normalized)
    return normalized


def choose_plan_model(
    models_to_try: List[str],
    base_url: str,
    system_prompt: str,
    user_prompt: str,
    output_schema: Dict[str, Any],
    timeout_sec: int,
    temperature: float,
    think_mode: Optional[str],
    base_debug_dir: Path,
    n_objects: int,
    payload: Dict[str, Any],
    design_brief: str,
) -> Tuple[str, Dict[str, Any], Dict[str, Any], List[Dict[str, str]]]:
    errors: List[Dict[str, str]] = []

    for model_name in models_to_try:
        safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name)
        debug_dir = base_debug_dir / safe_model

        print(f"INFO: trying plan model -> {model_name}", flush=True)

        try:
            raw_plan_text = run_plan_stage_raw(
                model_name=model_name,
                base_url=base_url,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                output_schema=output_schema,
                timeout_sec=timeout_sec,
                temperature=temperature,
                think_mode=think_mode,
                debug_dir=debug_dir / "raw_plan",
            )

            validation = validate_plan_structure(raw_plan_text, n_objects=n_objects)
            if validation.ok and validation.normalized is not None:
                plan_json = validation.normalized
            else:
                print(
                    f"INFO: planner raw output invalid for {model_name}, trying repair stage",
                    flush=True,
                )
                plan_json = run_plan_repair_stage(
                    model_name=model_name,
                    base_url=base_url,
                    payload=payload,
                    design_brief=design_brief,
                    raw_plan_text=raw_plan_text,
                    output_schema=output_schema,
                    timeout_sec=timeout_sec,
                    temperature=temperature,
                    think_mode=think_mode,
                    debug_dir=debug_dir / "repair_plan",
                    n_objects=n_objects,
                )

            reasoning = str(plan_json.get("reasoning", "")).strip()
            if len(reasoning) < 10:
                plan_json["reasoning"] = "Planner reasoning was minimal; normalized automatically."

            extracted_hints = extract_stage1_hints(plan_json, n_objects=n_objects)
            hint_count = len(extracted_hints.get("placements", []))
            save_json(debug_dir / "final_plan.json", plan_json)
            save_json(debug_dir / "extracted_hints.json", extracted_hints)

            if hint_count == 0:
                raise ValueError("Planner did not provide any parseable placement hints.")

            return model_name, plan_json, extracted_hints, errors
        except Exception as e:
            err = str(e)
            errors.append({"model": model_name, "error": err})
            print(f"WARNING: plan model failed -> {model_name}: {err}", flush=True)

    raise RuntimeError(
        "Все plan-модели Ollama завершились неудачно: "
        + json.dumps(errors, ensure_ascii=False)
    )


def choose_json_model(
    models_to_try: List[str],
    base_url: str,
    system_prompt: str,
    user_prompt: str,
    output_schema: Dict[str, Any],
    timeout_sec: int,
    temperature: float,
    max_llm_attempts: int,
    base_debug_dir: Path,
    n_objects: int,
    think_mode: Optional[str],
) -> Tuple[str, Dict[str, Any], int, List[Dict[str, str]]]:
    errors: List[Dict[str, str]] = []

    for model_name in models_to_try:
        safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name)
        debug_dir = base_debug_dir / safe_model

        print(f"INFO: trying json model -> {model_name}", flush=True)

        try:
            normalized_layout, attempts_used = run_json_stage_with_retry(
                model_name=model_name,
                base_url=base_url,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                output_schema=output_schema,
                timeout_sec=timeout_sec,
                temperature=temperature,
                max_llm_attempts=max_llm_attempts,
                debug_dir=debug_dir,
                n_objects=n_objects,
                think_mode=think_mode,
            )
            return model_name, normalized_layout, attempts_used, errors
        except Exception as e:
            err = str(e)
            errors.append({"model": model_name, "error": err})
            print(f"WARNING: json model failed -> {model_name}: {err}", flush=True)

    raise RuntimeError(
        "Все json-модели Ollama завершились неудачно: "
        + json.dumps(errors, ensure_ascii=False)
    )


def choose_critic_model(
    models_to_try: List[str],
    base_url: str,
    system_prompt: str,
    user_prompt: str,
    schema: Dict[str, Any],
    timeout_sec: int,
    temperature: float,
    think_mode: Optional[str],
    base_debug_dir: Path,
) -> Tuple[str, Dict[str, Any], List[Dict[str, str]]]:
    errors: List[Dict[str, str]] = []

    for model_name in models_to_try:
        safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name)
        debug_dir = base_debug_dir / safe_model

        print(f"INFO: trying critic model -> {model_name}", flush=True)

        try:
            verdict = run_critic_stage(
                model_name=model_name,
                base_url=base_url,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=schema,
                timeout_sec=timeout_sec,
                temperature=temperature,
                think_mode=think_mode,
                debug_dir=debug_dir,
            )
            return model_name, verdict, errors
        except Exception as e:
            err = str(e)
            errors.append({"model": model_name, "error": err})
            print(f"WARNING: critic model failed -> {model_name}: {err}", flush=True)

    raise RuntimeError(
        "Все critic-модели Ollama завершились неудачно: "
        + json.dumps(errors, ensure_ascii=False)
    )


# ============================================================
# Stage coherence checks
# ============================================================

def check_plan_json_consistency(
    extracted_hints: Dict[str, Any],
    llm_layout: Dict[str, Any],
    n_objects: int,
) -> None:
    placements = llm_layout.get("placements")
    if not isinstance(placements, list):
        raise ValueError("JSON stage returned invalid placements structure.")

    json_indices = sorted(int(p["index"]) for p in placements)
    expected = list(range(n_objects))
    if json_indices != expected:
        raise ValueError(
            f"JSON stage lost or corrupted indices. Expected {expected}, got {json_indices}."
        )

    hint_rows = extracted_hints.get("placements") if isinstance(extracted_hints, dict) else None
    if not isinstance(hint_rows, list) or not hint_rows:
        return

    json_map = {int(p["index"]): p for p in placements}

    for row in hint_rows:
        if not isinstance(row, dict):
            continue

        try:
            idx = int(row["index"])
            hint_x = float(row["x"])
            hint_y = float(row["y"])
            hint_yaw = int(row["yaw_deg"])
        except Exception:
            continue

        if idx not in json_map:
            raise ValueError(f"JSON stage lost hinted index={idx} from stage-1 output.")

        built = json_map[idx]
        built_x = float(built["x"])
        built_y = float(built["y"])
        built_yaw = int(built["yaw_deg"])

        if built_yaw != hint_yaw:
            continue

        dx = abs(built_x - hint_x)
        dy = abs(built_y - hint_y)

        if dx <= 0.60 and dy <= 0.60:
            continue

        continue


# ============================================================
# Normalize and repair
# ============================================================

def normalize_and_repair_layout(
    room: Dict[str, Any],
    objects_data: Dict[str, Any],
    llm_layout: Dict[str, Any],
    llm_attempts_used: int,
    llm_model_used: str,
    llm_models_tried: List[str],
    llm_plan_json: Dict[str, Any],
    llm_extracted_hints: Dict[str, Any],
    llm_plan_model_used: str,
    llm_plan_models_tried: List[str],
    llm_critic_text: Dict[str, Any],
    llm_critic_model_used: str,
    llm_critic_models_tried: List[str],
    attempt_index: int,
) -> Dict[str, Any]:
    poly = extract_room_polygon_xy(room)
    ceiling_height = extract_ceiling_height(room)
    src_items = extract_objects(objects_data)
    placements = llm_layout["placements"]

    occupied_floor: List[Dict[str, float]] = []
    tmp_result: Dict[int, Dict[str, Any]] = {}

    floor_indices: List[int] = []
    ceiling_indices: List[int] = []

    for idx, obj in enumerate(src_items):
        if extract_mount_type(obj) == "ceiling":
            ceiling_indices.append(idx)
        else:
            floor_indices.append(idx)

    floor_indices.sort(key=lambda i: -(extract_size_m(src_items[i])[0] * extract_size_m(src_items[i])[1]))

    by_index = {int(p["index"]): p for p in placements}

    for idx in floor_indices:
        src_obj = src_items[idx]
        pred = by_index[idx]

        sx, sy, sz = extract_size_m(src_obj)
        yaw_deg = quantize_rot_0_90_180_270(float(pred["yaw_deg"]))
        target_x = float(pred["x"])
        target_y = float(pred["y"])

        fixed_x, fixed_y = find_nearest_valid_position(
            target_x=target_x,
            target_y=target_y,
            sx=sx,
            sy=sy,
            sz=sz,
            yaw_deg=yaw_deg,
            polygon_xy=poly,
            occupied=occupied_floor,
        )

        aabb = aabb_from_center_size_rotation(
            cx=fixed_x,
            cy=fixed_y,
            sx=sx,
            sy=sy,
            sz=sz,
            yaw_deg=yaw_deg,
            z_floor_m=0.0,
        )

        item = dict(src_obj)
        item["placement_source"] = "ollama_llm"
        item["rotation"] = yaw_deg
        item["yaw_deg"] = yaw_deg
        item["yaw_rad"] = math.radians(yaw_deg)
        item["position_room_xy_m"] = [fixed_x, fixed_y]
        item["z_floor_m"] = 0.0
        item["size_m"] = [sx, sy, sz]
        item["aabb"] = aabb
        item["bbox"] = dict(aabb)
        item["llm_target_position_room_xy_m"] = [target_x, target_y]
        item["llm_target_yaw_deg"] = float(pred["yaw_deg"])
        item["llm_attempts_used"] = llm_attempts_used
        item["llm_model_used"] = llm_model_used
        item["llm_plan_model_used"] = llm_plan_model_used
        item["llm_critic_model_used"] = llm_critic_model_used
        item["scene_attempt_index"] = attempt_index

        occupied_floor.append(aabb)
        tmp_result[idx] = item

    for idx in ceiling_indices:
        src_obj = src_items[idx]
        pred = by_index[idx]

        sx, sy, sz = extract_size_m(src_obj)
        yaw_deg = quantize_rot_0_90_180_270(float(pred["yaw_deg"]))
        x = float(pred["x"])
        y = float(pred["y"])

        if not point_in_polygon(x, y, poly):
            x_min, x_max, y_min, y_max = polygon_bbox(poly)
            x = min(max(x, x_min), x_max)
            y = min(max(y, y_min), y_max)

        z_floor_m = max(0.0, ceiling_height - sz)

        aabb = aabb_from_center_size_rotation(
            cx=x,
            cy=y,
            sx=sx,
            sy=sy,
            sz=sz,
            yaw_deg=yaw_deg,
            z_floor_m=z_floor_m,
        )

        item = dict(src_obj)
        item["placement_source"] = "ollama_llm"
        item["rotation"] = yaw_deg
        item["yaw_deg"] = yaw_deg
        item["yaw_rad"] = math.radians(yaw_deg)
        item["position_room_xy_m"] = [x, y]
        item["z_floor_m"] = z_floor_m
        item["size_m"] = [sx, sy, sz]
        item["aabb"] = aabb
        item["bbox"] = dict(aabb)
        item["llm_target_position_room_xy_m"] = [float(pred["x"]), float(pred["y"])]
        item["llm_target_yaw_deg"] = float(pred["yaw_deg"])
        item["llm_attempts_used"] = llm_attempts_used
        item["llm_model_used"] = llm_model_used
        item["llm_plan_model_used"] = llm_plan_model_used
        item["llm_critic_model_used"] = llm_critic_model_used
        item["scene_attempt_index"] = attempt_index

        tmp_result[idx] = item

    out_items = [tmp_result[i] for i in range(len(src_items))]

    return {
        "placer": "ollama_llm",
        "placements": out_items,
        "llm_raw": llm_layout,
        "llm_attempts_used": llm_attempts_used,
        "llm_model_used": llm_model_used,
        "llm_models_tried": llm_models_tried,
        "llm_plan_json": llm_plan_json,
        "llm_extracted_hints": llm_extracted_hints,
        "llm_plan_model_used": llm_plan_model_used,
        "llm_plan_models_tried": llm_plan_models_tried,
        "llm_critic_verdict": llm_critic_text,
        "llm_critic_model_used": llm_critic_model_used,
        "llm_critic_models_tried": llm_critic_models_tried,
        "scene_attempt_index": attempt_index,
    }


# ============================================================
# Main
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="LLM/Ollama placer: planner -> builder -> deterministic-repair -> critic"
    )

    ap.add_argument("--room", required=True, help="Путь к room.json")
    ap.add_argument("--objects", required=True, help="Путь к objects.json")
    ap.add_argument("--out", required=True, help="Путь к итоговому placement_result.json")
    ap.add_argument("--mode", default="llm", help="Режим расстановки")
    ap.add_argument("--design-brief", default="", help="Полный текст интерьерного запроса пользователя")

    ap.add_argument("--ollama-url", default="http://127.0.0.1:11434", help="База Ollama API")

    ap.add_argument("--ollama-model", default="gpt-oss:20b", help="Основная модель builder/json-stage")
    ap.add_argument("--ollama-models", nargs="*", default=None, help="Список fallback-моделей builder/json-stage")

    ap.add_argument("--plan-model", default=None, help="Основная модель planner-stage")
    ap.add_argument("--plan-models", nargs="*", default=None, help="Список fallback-моделей planner-stage")

    ap.add_argument("--critic-model", default=None, help="Основная модель critic-stage")
    ap.add_argument("--critic-models", nargs="*", default=None, help="Список fallback-моделей critic-stage")

    ap.add_argument("--timeout", type=int, default=300, help="Таймаут HTTP в секундах")
    ap.add_argument("--temperature", type=float, default=0.0, help="temperature для builder/json-stage")
    ap.add_argument("--plan-temperature", type=float, default=0.0, help="temperature для planner-stage")
    ap.add_argument("--critic-temperature", type=float, default=0.0, help="temperature для critic-stage")

    ap.add_argument("--max-llm-attempts", type=int, default=3, help="Максимум попыток перепосылки в builder/json-stage")
    ap.add_argument("--max-scene-attempts", type=int, default=3, help="Максимум полных planner->builder->repair->critic циклов")

    ap.add_argument("--llm-think", choices=["none", "low"], default="none", help="Режим think для builder/json-stage")
    ap.add_argument("--plan-think", choices=["none", "low"], default="low", help="Режим think для planner-stage")
    ap.add_argument("--critic-think", choices=["none", "low"], default="low", help="Режим think для critic-stage")
    args = ap.parse_args()

    room_path = Path(args.room).expanduser().resolve()
    objects_path = Path(args.objects).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    if not room_path.is_file():
        raise FileNotFoundError(room_path)
    if not objects_path.is_file():
        raise FileNotFoundError(objects_path)

    room = load_json(room_path)
    objects_data = load_json(objects_path)
    src_items = extract_objects(objects_data)

    payload = build_llm_payload(room, objects_data)
    n_objects = len(src_items)
    output_schema = build_output_schema(n_objects)
    critic_schema = build_critic_schema()
    plan_schema = build_plan_schema(n_objects)

    json_models_to_try = normalize_model_list(args.ollama_model, args.ollama_models)

    plan_primary = args.plan_model if args.plan_model else args.ollama_model
    plan_fallbacks = args.plan_models if args.plan_models is not None else args.ollama_models
    plan_models_to_try = normalize_model_list(plan_primary, plan_fallbacks)

    critic_primary = args.critic_model if args.critic_model else plan_primary
    critic_fallbacks = args.critic_models if args.critic_models is not None else plan_fallbacks
    critic_models_to_try = normalize_model_list(critic_primary, critic_fallbacks)

    json_think_mode = None if args.llm_think == "none" else args.llm_think
    plan_think_mode = None if args.plan_think == "none" else args.plan_think
    critic_think_mode = None if args.critic_think == "none" else args.critic_think

    plan_system_prompt = build_plan_system_prompt()
    json_system_prompt = build_json_system_prompt()
    critic_system_prompt = build_critic_system_prompt()

    last_errors: List[str] = []
    final_result: Optional[Dict[str, Any]] = None
    final_plan_model_used = ""
    final_json_model_used = ""
    final_critic_model_used = ""
    final_json_attempts_used = 0
    final_plan_errors: List[Dict[str, str]] = []
    final_json_errors: List[Dict[str, str]] = []
    final_critic_errors: List[Dict[str, str]] = []
    final_extracted_hints: Dict[str, Any] = {"placements": []}
    final_plan_json: Dict[str, Any] = {}
    final_critic_verdict: Dict[str, Any] = {}
    final_attempt_index = 0
    final_repaired_layout: Dict[str, Any] = {}

    poly_xy = extract_room_polygon_xy(room)
    ceiling_height = extract_ceiling_height(room)

    for scene_attempt in range(1, int(args.max_scene_attempts) + 1):
        final_attempt_index = scene_attempt
        print(f"INFO: scene attempt {scene_attempt}/{args.max_scene_attempts}", flush=True)

        previous_feedback = "\n".join(f"- {x}" for x in last_errors) if last_errors else ""

        plan_user_prompt = build_plan_user_prompt(
            payload=payload,
            mode=args.mode,
            design_brief=args.design_brief,
            previous_feedback=previous_feedback,
        )

        plan_model_used, plan_json, extracted_hints, plan_model_errors = choose_plan_model(
            models_to_try=plan_models_to_try,
            base_url=args.ollama_url,
            system_prompt=plan_system_prompt,
            user_prompt=plan_user_prompt,
            output_schema=plan_schema,
            timeout_sec=int(args.timeout),
            temperature=float(args.plan_temperature),
            think_mode=plan_think_mode,
            base_debug_dir=out_path.parent / "ollama_debug" / f"attempt_{scene_attempt:02d}" / "stage1_plan",
            n_objects=n_objects,
            payload=payload,
            design_brief=args.design_brief,
        )
        final_plan_model_used = plan_model_used
        final_plan_errors = plan_model_errors
        final_plan_json = plan_json
        final_extracted_hints = extracted_hints

        repair_feedback = "\n".join(f"- {x}" for x in last_errors) if last_errors else ""

        json_user_prompt = build_json_user_prompt(
            payload=payload,
            mode=args.mode,
            plan_json=plan_json,
            extracted_hints=extracted_hints,
            n_objects=n_objects,
            repair_feedback=repair_feedback,
        )

        json_model_used, llm_layout_raw, llm_attempts_used, json_model_errors = choose_json_model(
            models_to_try=json_models_to_try,
            base_url=args.ollama_url,
            system_prompt=json_system_prompt,
            user_prompt=json_user_prompt,
            output_schema=output_schema,
            timeout_sec=int(args.timeout),
            temperature=float(args.temperature),
            max_llm_attempts=int(args.max_llm_attempts),
            base_debug_dir=out_path.parent / "ollama_debug" / f"attempt_{scene_attempt:02d}" / "stage2_json",
            n_objects=n_objects,
            think_mode=json_think_mode,
        )
        final_json_model_used = json_model_used
        final_json_attempts_used = llm_attempts_used
        final_json_errors = json_model_errors

        check_plan_json_consistency(
            extracted_hints=extracted_hints,
            llm_layout=llm_layout_raw,
            n_objects=n_objects,
        )

        # Сначала твой более сильный детерминированный repair.
        llm_layout = repair_layout_deterministically(
            room=room,
            payload=payload,
            src_items=src_items,
            llm_layout=llm_layout_raw,
            design_brief=args.design_brief,
        )

        # Потом жёсткая проекция ключевых ограничений перед critic.
        llm_layout = apply_hard_constraints_before_critic(
            payload=payload,
            src_items=src_items,
            llm_layout=llm_layout,
            poly_xy=poly_xy,
        )

        final_repaired_layout = llm_layout

        save_json(
            out_path.parent / "ollama_debug" / f"attempt_{scene_attempt:02d}" / "repaired_layout.json",
            llm_layout,
        )

        deterministic_issues, deterministic_metrics = evaluate_layout_semantics(
            payload=payload,
            src_items=src_items,
            llm_layout=llm_layout,
            poly_xy=poly_xy,
            ceiling_height=ceiling_height,
            design_brief=args.design_brief,
        )

        critic_user_prompt = build_critic_user_prompt(
            payload=payload,
            design_brief=args.design_brief,
            plan_json=plan_json,
            llm_layout=llm_layout,
            deterministic_issues=deterministic_issues,
            deterministic_metrics=deterministic_metrics,
            attempt_index=scene_attempt,
        )

        critic_model_used, critic_verdict, critic_model_errors = choose_critic_model(
            models_to_try=critic_models_to_try,
            base_url=args.ollama_url,
            system_prompt=critic_system_prompt,
            user_prompt=critic_user_prompt,
            schema=critic_schema,
            timeout_sec=int(args.timeout),
            temperature=float(args.critic_temperature),
            think_mode=critic_think_mode,
            base_debug_dir=out_path.parent / "ollama_debug" / f"attempt_{scene_attempt:02d}" / "stage3_critic",
        )
        final_critic_model_used = critic_model_used
        final_critic_errors = critic_model_errors
        final_critic_verdict = critic_verdict

        critic_ok = bool(critic_verdict.get("ok"))
        hard_fail = bool(critic_verdict.get("hard_fail"))
        critic_issues = [str(x) for x in critic_verdict.get("issues", [])]
        critic_fixes = [str(x) for x in critic_verdict.get("fix_instructions", [])]

        hard_deterministic = {
            "bed_not_on_window_wall",
            "door_clearance_blocked",
            "wardrobe_not_back_to_wall",
            "wardrobe_acts_like_divider",
        }
        has_hard_deterministic = any(x in hard_deterministic for x in deterministic_issues)

        if not has_hard_deterministic and not hard_fail:
            final_result = normalize_and_repair_layout(
                room=room,
                objects_data=objects_data,
                llm_layout=llm_layout,
                llm_attempts_used=llm_attempts_used,
                llm_model_used=json_model_used,
                llm_models_tried=json_models_to_try,
                llm_plan_json=plan_json,
                llm_extracted_hints=extracted_hints,
                llm_plan_model_used=plan_model_used,
                llm_plan_models_tried=plan_models_to_try,
                llm_critic_text=critic_verdict,
                llm_critic_model_used=critic_model_used,
                llm_critic_models_tried=critic_models_to_try,
                attempt_index=scene_attempt,
            )
            final_result["llm_layout_before_repair"] = llm_layout_raw
            final_result["llm_layout_after_repair"] = llm_layout
            break

        merged_errors: List[str] = []
        merged_errors.extend(deterministic_issues)
        merged_errors.extend(critic_issues)
        merged_errors.extend(critic_fixes)
        merged_errors = [x for x in merged_errors if x]
        last_errors = merged_errors or ["critic_rejected_scene_without_explicit_reason"]

        print(
            "WARNING: scene attempt rejected -> "
            + json.dumps(
                {
                    "attempt": scene_attempt,
                    "deterministic_issues": deterministic_issues,
                    "has_hard_deterministic": has_hard_deterministic,
                    "critic_ok": critic_ok,
                    "critic_hard_fail": hard_fail,
                    "critic_issues": critic_issues,
                    "critic_fix_instructions": critic_fixes,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    if final_result is None:
        raise RuntimeError(
            "Не удалось собрать удовлетворительную scene после "
            f"{args.max_scene_attempts} полных попыток.\n"
            f"Последние ошибки: {json.dumps(last_errors, ensure_ascii=False)}"
        )

    final_result["llm_plan_model_errors"] = final_plan_errors
    final_result["llm_json_model_errors"] = final_json_errors
    final_result["llm_critic_model_errors"] = final_critic_errors
    final_result["design_brief"] = args.design_brief
    final_result["final_scene_attempt_index"] = final_attempt_index
    final_result["final_repaired_layout"] = final_repaired_layout

    save_json(out_path, final_result)

    print(f"OK: saved placement -> {out_path}")
    print(f"OK: plan model used -> {final_plan_model_used}")
    print(f"OK: json model used -> {final_json_model_used}")
    print(f"OK: critic model used -> {final_critic_model_used}")
    print(f"OK: json attempts used -> {final_json_attempts_used}")
    print(f"OK: extracted stage1 hints -> {len(final_extracted_hints.get('placements', []))}")
    print(f"OK: scene attempt used -> {final_attempt_index}")
    print(f"OK: plan models tried -> {', '.join(plan_models_to_try)}")
    print(f"OK: json models tried -> {', '.join(json_models_to_try)}")
    print(f"OK: critic models tried -> {', '.join(critic_models_to_try)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise