from __future__ import annotations

import math
from typing import Any

from .schemas import as_float, point


def _extract_polygon(data: dict[str, Any]) -> list[dict[str, float]]:
    room = data.get("room") if isinstance(data.get("room"), dict) else {}
    raw = room.get("floor_polygon") or data.get("floor_polygon") or data.get("polygon")
    if raw is None and isinstance(data.get("scene"), dict):
        raw = data["scene"].get("floor_polygon")
    if raw is None:
        raise ValueError("room floor polygon is missing")
    pts: list[dict[str, float]] = []
    for p in raw:
        if isinstance(p, dict):
            pts.append(point(p.get("x"), p.get("y", p.get("z"))))
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            pts.append(point(p[0], p[1]))
    if len(pts) < 3:
        raise ValueError("room floor polygon must contain at least 3 points")
    return pts


def normalize_room_input(data: dict[str, Any], prompt: str | None = None) -> dict[str, Any]:
    room_in = data.get("room") if isinstance(data.get("room"), dict) else {}
    polygon = _extract_polygon(data)
    openings = room_in.get("openings") if isinstance(room_in.get("openings"), dict) else {}
    room = {
        "id": str(room_in.get("id") or data.get("room_id") or "room_001"),
        "type_hint": str(room_in.get("type_hint") or data.get("room_type") or ""),
        "height_m": as_float(room_in.get("height_m", data.get("height_m", 2.8)), 2.8),
        "floor_polygon": polygon,
        "openings": {
            "doors": list(openings.get("doors") or room_in.get("doors") or []),
            "windows": list(openings.get("windows") or room_in.get("windows") or []),
        },
    }
    return {
        "schema": "room_input/v1",
        "room": room,
        "prompt": str(prompt if prompt is not None else data.get("prompt") or ""),
        "source_schema": data.get("schema"),
        "source_scene": data if data.get("schema") in {"scene.v1", "placement.v1"} else None,
    }


def _signed_area(points: list[dict[str, float]]) -> float:
    total = 0.0
    for i, p in enumerate(points):
        q = points[(i + 1) % len(points)]
        total += p["x"] * q["y"] - q["x"] * p["y"]
    return total / 2.0


def _centroid(points: list[dict[str, float]]) -> dict[str, float]:
    signed = _signed_area(points)
    if abs(signed) < 1e-9:
        return {"x": sum(p["x"] for p in points) / len(points), "y": sum(p["y"] for p in points) / len(points)}
    cx = 0.0
    cy = 0.0
    for i, p in enumerate(points):
        q = points[(i + 1) % len(points)]
        cross = p["x"] * q["y"] - q["x"] * p["y"]
        cx += (p["x"] + q["x"]) * cross
        cy += (p["y"] + q["y"]) * cross
    return {"x": cx / (6.0 * signed), "y": cy / (6.0 * signed)}


def analyze_room_geometry(input_json: dict[str, Any]) -> dict[str, Any]:
    normalized = input_json if input_json.get("schema") == "room_input/v1" else normalize_room_input(input_json)
    room = normalized["room"]
    pts = room["floor_polygon"]
    xs = [p["x"] for p in pts]
    ys = [p["y"] for p in pts]
    area = abs(_signed_area(pts))
    center = _centroid(pts)
    ccw = _signed_area(pts) > 0
    walls = []
    for idx, p in enumerate(pts):
        q = pts[(idx + 1) % len(pts)]
        dx = q["x"] - p["x"]
        dy = q["y"] - p["y"]
        length = math.hypot(dx, dy)
        if length <= 1e-9:
            nx = ny = 0.0
        elif ccw:
            nx, ny = -dy / length, dx / length
        else:
            nx, ny = dy / length, -dx / length
        walls.append({
            "id": f"wall_{idx}",
            "from": dict(p),
            "to": dict(q),
            "length_m": round(length, 4),
            "normal_to_inside": {"x": round(nx, 6), "y": round(ny, 6)},
        })
    lengths = sorted((w["length_m"] for w in walls), reverse=True)
    long_cut = lengths[min(1, len(lengths) - 1)] if lengths else 0.0
    short_cut = sorted(lengths)[min(1, len(lengths) - 1)] if lengths else 0.0
    openings = room.get("openings") or {}
    doors = list(openings.get("doors") or [])
    windows = list(openings.get("windows") or [])
    assumptions: list[str] = []
    if not doors:
        assumptions.append("Door position is unknown.")
    if not windows:
        assumptions.append("Window position is unknown.")
        assumptions.append("Desk cannot be reliably placed near natural light.")
    return {
        "schema": "room_geometry/v1",
        "area_m2": round(area, 4),
        "bbox": {"width_m": round(max(xs) - min(xs), 4), "depth_m": round(max(ys) - min(ys), 4), "x_min": min(xs), "x_max": max(xs), "y_min": min(ys), "y_max": max(ys)},
        "center": {"x": round(center["x"], 4), "y": round(center["y"], 4)},
        "walls": walls,
        "long_walls": [w["id"] for w in walls if w["length_m"] >= long_cut],
        "short_walls": [w["id"] for w in walls if w["length_m"] <= short_cut],
        "has_known_door": bool(doors),
        "has_known_window": bool(windows),
        "openings": {"doors": doors, "windows": windows},
        "assumptions": assumptions,
    }
