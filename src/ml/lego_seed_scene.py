#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import math
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple


ALLOWED_ROTATIONS = (0, 90, 180, 270)


def load_json(path: str | Path) -> Any:
    p = Path(path).expanduser().resolve()
    return json.loads(p.read_text(encoding="utf-8"))


def save_json(path: str | Path, data: Any) -> None:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def polygon_area(poly: List[Tuple[float, float]]) -> float:
    s = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(0.5 * s)


def room_polygon_xy(room_json: Dict[str, Any]) -> List[Tuple[float, float]]:
    room = room_json.get("room", room_json)
    floor_polygon = room.get("floor_polygon") or []
    out: List[Tuple[float, float]] = []
    for p in floor_polygon:
        if isinstance(p, dict):
            out.append((float(p["x"]), float(p["y"])))
        else:
            out.append((float(p[0]), float(p[1])))
    if len(out) < 3:
        raise RuntimeError("В room.json нет корректного floor_polygon")
    return out


def point_in_polygon_xy(x: float, y: float, poly: List[Tuple[float, float]]) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cross = (y1 > y) != (y2 > y)
        if cross:
            denom = (y2 - y1)
            if abs(denom) < 1e-12:
                continue
            x_cross = x1 + (y - y1) * (x2 - x1) / denom
            if x <= x_cross:
                inside = not inside
    return inside


def sample_point_in_polygon(poly: List[Tuple[float, float]], max_tries: int = 1000) -> Tuple[float, float]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    for _ in range(max_tries):
        x = random.uniform(xmin, xmax)
        y = random.uniform(ymin, ymax)
        if point_in_polygon_xy(x, y, poly):
            return x, y

    return (0.5 * (xmin + xmax), 0.5 * (ymin + ymax))


def build_aabb_from_center_size(position_m: List[float], size_m: List[float]) -> Dict[str, float]:
    cx, cy, cz = position_m
    sx, sy, sz = size_m
    return {
        "x_min": cx - sx / 2.0,
        "x_max": cx + sx / 2.0,
        "y_min": cy - sy / 2.0,
        "y_max": cy + sy / 2.0,
        "z_min": cz - sz / 2.0,
        "z_max": cz + sz / 2.0,
    }


def room_area_m2(room_json: Dict[str, Any]) -> float:
    return polygon_area(room_polygon_xy(room_json))


def object_footprint_m2(obj: Dict[str, Any]) -> float:
    size_m = obj.get("size_m") or [0.0, 0.0, 0.0]
    if not isinstance(size_m, list) or len(size_m) != 3:
        return 0.0
    return max(0.0, float(size_m[0])) * max(0.0, float(size_m[1]))


def total_objects_footprint_m2(objects_v1: Dict[str, Any]) -> float:
    objs = objects_v1.get("objects") or []
    total = 0.0
    for obj in objs:
        if not isinstance(obj, dict):
            continue
        constraints = obj.get("constraints") or {}
        mount_type = constraints.get("mount_type")
        if mount_type == "ceiling":
            continue
        total += object_footprint_m2(obj)
    return total


def object_priority(obj: Dict[str, Any]) -> Tuple[int, float]:
    category = str(obj.get("category", obj.get("name", ""))).lower()
    area = object_footprint_m2(obj)

    if "bed" in category or "кровать" in category:
        pr = 0
    elif "wardrobe" in category or "шкаф" in category:
        pr = 1
    elif "sofa" in category or "диван" in category:
        pr = 2
    elif "desk" in category or "table" in category or "стол" in category:
        pr = 3
    elif "chair" in category or "стул" in category or "кресло" in category:
        pr = 4
    elif "nightstand" in category or "тумб" in category:
        pr = 5
    elif "lamp" in category or "light" in category or "ламп" in category or "свет" in category:
        pr = 6
    else:
        pr = 7

    return pr, -area


def sort_objects_for_generation(objects_v1: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(objects_v1)
    objs = out.get("objects") or []
    objs = [o for o in objs if isinstance(o, dict)]
    objs.sort(key=object_priority)
    out["objects"] = objs
    return out


def crop_last_object(objects_v1: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(objects_v1)
    objs = out.get("objects") or []
    if objs:
        out["objects"] = objs[:-1]
    return out


def build_seed_scene_and_placement(
    room_json: Dict[str, Any],
    objects_v1: Dict[str, Any],
    seed: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    random.seed(int(seed))

    room_block = deepcopy(room_json.get("room", room_json))
    ceiling_h = float(room_block.get("ceiling_height", room_block.get("ceiling_height_m", 2.8)))

    poly = room_polygon_xy(room_json)
    placements: List[Dict[str, Any]] = []

    for obj in objects_v1.get("objects", []):
        size_m = deepcopy(obj.get("size_m") or [0.0, 0.0, 0.0])
        constraints = deepcopy(obj.get("constraints") or {})
        asset = deepcopy(obj.get("asset") or {})
        meta = deepcopy(obj.get("meta") or {})

        mount_type = constraints.get("mount_type")
        if mount_type is None:
            mount_type = obj.get("mount_type")

        rotation_deg = random.choice(ALLOWED_ROTATIONS)

        if mount_type == "ceiling":
            x, y = sample_point_in_polygon(poly)
            z = ceiling_h - float(size_m[2]) / 2.0
        else:
            x, y = sample_point_in_polygon(poly)
            z = float(size_m[2]) / 2.0

        position_m = [float(x), float(y), float(z)]

        placement = {
            "id": obj.get("id"),
            "name": obj.get("name"),
            "category": obj.get("category", obj.get("name")),
            "position_m": position_m,
            "size_m": size_m,
            "rotation_deg": int(rotation_deg),
            "yaw_deg": float(rotation_deg),
            "yaw_rad": math.radians(float(rotation_deg)),
            "aabb": build_aabb_from_center_size(position_m, size_m),
            "mount_type": mount_type,
            "wall_contact_side": None,
            "constraints": constraints,
            "asset": asset,
            "source": {
                "placement_source": "lego_seed_random_full",
            },
            "meta": meta,
            "color": deepcopy(obj.get("color", [0.7, 0.7, 0.7])),
        }
        placements.append(placement)

    placement_v1 = {
        "schema": "placement.v1",
        "placer": "lego_gen",
        "mode": "random_full",
        "placements": placements,
        "meta": {
            "seed": int(seed),
            "generated_from": "objects.v1",
        },
    }

    scene_v1 = {
        "schema": "scene.v1",
        "room": room_block,
        "placements": placements,
        "meta": {
            "placer": "lego_gen",
            "mode": "random_full",
            "seed": int(seed),
        },
    }

    return scene_v1, placement_v1