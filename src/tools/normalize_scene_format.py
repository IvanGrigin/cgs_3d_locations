#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
normalize_scene_format.py

Один универсальный конвертер JSON-форматов пайплайна расстановки мебели.

Что умеет:
1. Принимать старые форматы:
   - chooser objects:
       {"seed": ..., "items": [...]}
   - cube placement:
       {"room": {...}, "items": [...]}
   - diffuscene / ollama placement:
       {"placer": "...", "placements": [...]}
   - scene-подобные:
       {"room": {...}, "placements": [...]}
       {"room": {...}, "items": [...]}

2. Принимать уже новые форматы:
   - objects.v1
   - placement.v1
   - scene.v1

3. Конвертировать всё в канонический формат:
   - objects.v1
   - placement.v1
   - scene.v1

4. Не терять данные:
   - нестандартные поля складываются в meta
   - служебные source-поля сохраняются в source
   - asset_meta переносится в meta/asset

5. Уметь собирать scene.v1 из room.json + placement.json

Примеры:
    python normalize_scene_format.py --input old.json --output new.json
    python normalize_scene_format.py --input old.json --output placement.json --target placement
    python normalize_scene_format.py --input old.json --output objects.json --target objects
    python normalize_scene_format.py --input old.json --output scene.json --target scene
    python normalize_scene_format.py --room room.json --placement placement.json --output scene.json --target scene

Если --target auto:
- objects_like -> objects.v1
- placement_like -> placement.v1
- scene_like / cube_or_old_scene -> scene.v1
"""

from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# IO
# ============================================================

def load_json(path: str | Path) -> Any:
    p = Path(path).expanduser().resolve()
    return json.loads(p.read_text(encoding="utf-8"))


def save_json(path: str | Path, data: Any) -> None:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ============================================================
# Базовые утилиты
# ============================================================

def is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def as_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def as_str(x: Any, default: str = "") -> str:
    if x is None:
        return default
    return str(x)


def ensure_list3(v: Any) -> Optional[List[float]]:
    if isinstance(v, list) and len(v) == 3:
        return [as_float(v[0]), as_float(v[1]), as_float(v[2])]
    if isinstance(v, tuple) and len(v) == 3:
        return [as_float(v[0]), as_float(v[1]), as_float(v[2])]
    return None


def as_list3(value: Any, default: Optional[List[float]] = None) -> List[float]:
    out = ensure_list3(value)
    if out is not None:
        return out
    return list(default or [0.0, 0.0, 0.0])


def radians_from_deg(deg: float) -> float:
    return math.radians(float(deg))


def degrees_from_rad(rad: float) -> float:
    return math.degrees(float(rad))


def quantize_rot_0_90_180_270(deg: float) -> int:
    a = float(deg) % 360.0
    allowed = (0.0, 90.0, 180.0, 270.0)
    best = min(allowed, key=lambda t: (abs(((a - t + 180.0) % 360.0) - 180.0), t))
    return int(best)


def first_non_none(*values: Any) -> Any:
    for v in values:
        if v is not None:
            return v
    return None


def deep_copy_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return deepcopy(obj)
    return {}


def mm3_to_m3(v: Any) -> Optional[List[float]]:
    arr = ensure_list3(v)
    if arr is None:
        return None
    return [x / 1000.0 for x in arr]


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


def center_from_aabb(aabb: Dict[str, Any]) -> List[float]:
    x_min = as_float(aabb.get("x_min"))
    x_max = as_float(aabb.get("x_max"))
    y_min = as_float(aabb.get("y_min"))
    y_max = as_float(aabb.get("y_max"))
    z_min = as_float(aabb.get("z_min"))
    z_max = as_float(aabb.get("z_max"))
    return [
        0.5 * (x_min + x_max),
        0.5 * (y_min + y_max),
        0.5 * (z_min + z_max),
    ]


def size_from_aabb(aabb: Dict[str, Any]) -> List[float]:
    return [
        max(0.0, as_float(aabb.get("x_max")) - as_float(aabb.get("x_min"))),
        max(0.0, as_float(aabb.get("y_max")) - as_float(aabb.get("y_min"))),
        max(0.0, as_float(aabb.get("z_max")) - as_float(aabb.get("z_min"))),
    ]


def build_center_from_xy_floor_and_size(pos_xy: List[float], z_floor_m: float, size_m: List[float]) -> List[float]:
    return [float(pos_xy[0]), float(pos_xy[1]), float(z_floor_m) + float(size_m[2]) / 2.0]


# ============================================================
# Определение формата входа
# ============================================================

def detect_input_kind(data: Any) -> str:
    if not isinstance(data, dict):
        raise ValueError("Корневой JSON должен быть объектом")

    schema = data.get("schema")
    if schema == "objects.v1":
        return "objects_v1"
    if schema == "placement.v1":
        return "placement_v1"
    if schema == "scene.v1":
        return "scene_v1"

    has_room = isinstance(data.get("room"), dict)
    has_items = isinstance(data.get("items"), list)
    has_objects = isinstance(data.get("objects"), list)
    has_placements = isinstance(data.get("placements"), list)

    if has_objects and not has_room and not has_placements:
        return "objects_like"

    if has_items and "seed" in data and not has_room:
        return "objects_like"

    if has_room and has_items:
        sample = data["items"][0] if data["items"] else None
        if isinstance(sample, dict):
            if any(k in sample for k in ("center", "rotation", "aabb", "yaw_deg", "position_room_xy_m", "position_m")):
                return "cube_or_old_scene"
        return "cube_or_old_scene"

    if has_room and has_placements:
        return "scene_like"

    if has_placements:
        return "placement_like"

    if has_items:
        sample = data["items"][0] if data["items"] else None
        if isinstance(sample, dict):
            if any(k in sample for k in ("center", "rotation", "aabb", "yaw_deg", "position_room_xy_m", "position_m")):
                return "placement_like"
        return "items_like"

    raise ValueError("Не удалось определить тип входного JSON")


def auto_target_from_input_kind(kind: str) -> str:
    if kind in {"objects_v1", "objects_like", "items_like"}:
        return "objects"
    if kind in {"placement_v1", "placement_like"}:
        return "placement"
    if kind in {"scene_v1", "scene_like", "cube_or_old_scene"}:
        return "scene"
    raise ValueError(f"Не удалось вывести target из kind={kind}")


# ============================================================
# Room
# ============================================================

def normalize_room(room: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(room)

    if isinstance(out.get("room"), dict):
        root = deepcopy(out["room"])
        for k, v in out.items():
            if k != "room":
                root.setdefault(k, v)
        out = root

    if "units" not in out:
        out["units"] = "m"

    return out


def normalize_room_dict(room: Dict[str, Any]) -> Dict[str, Any]:
    return normalize_room(room)


# ============================================================
# Общие extractors для objects/placements
# ============================================================

def make_object_id(index: int, obj: Dict[str, Any]) -> str:
    for key in ("id", "object_id", "uid"):
        v = obj.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return f"obj_{index + 1:04d}"


def extract_name(obj: Dict[str, Any], default: str = "object") -> str:
    for key in ("name", "class_name", "class", "type"):
        v = obj.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()

    asset_meta = obj.get("asset_meta") or {}
    for key in ("category", "super_category", "super-category"):
        v = asset_meta.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()

    return default


def extract_category(obj: Dict[str, Any], fallback_name: Optional[str] = None) -> str:
    for key in ("category",):
        v = obj.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()

    asset_meta = obj.get("asset_meta") or {}
    for key in ("category", "super_category", "super-category"):
        v = asset_meta.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()

    return fallback_name or extract_name(obj)


def extract_constraints(obj: Dict[str, Any]) -> Dict[str, Any]:
    v = obj.get("constraints")
    if isinstance(v, dict):
        return dict(v)
    return {}


def extract_color(obj: Dict[str, Any]) -> List[float]:
    color = obj.get("color")
    if isinstance(color, list):
        vals = []
        for x in color[:4]:
            try:
                vals.append(float(x))
            except Exception:
                vals.append(0.7)
        return vals
    return [0.7, 0.7, 0.7]


def extract_asset_block(obj: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(obj.get("asset"), dict):
        asset = dict(obj["asset"])
        if "mesh_path" in obj and "mesh_path" not in asset:
            asset["mesh_path"] = obj["mesh_path"]
        if "mesh_fit_mode" in obj and "mesh_fit_mode" not in asset:
            asset["mesh_fit_mode"] = obj["mesh_fit_mode"]
        if "mesh_texture_dirs" in obj and "mesh_texture_dirs" not in asset:
            asset["mesh_texture_dirs"] = obj["mesh_texture_dirs"]
        return asset

    asset_meta = obj.get("asset_meta") or {}
    asset: Dict[str, Any] = {
        "source": obj.get("asset_source"),
        "model_id": asset_meta.get("model_id") or obj.get("jid") or obj.get("model_jid") or obj.get("future_jid"),
        "mesh_path": obj.get("mesh_path"),
        "mesh_fit_mode": obj.get("mesh_fit_mode"),
        "mesh_texture_dirs": obj.get("mesh_texture_dirs"),
    }

    return {k: v for k, v in asset.items() if v is not None}


def extract_meta_block_from_object(obj: Dict[str, Any]) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    if isinstance(obj.get("meta"), dict):
        meta.update(obj["meta"])

    asset_meta = obj.get("asset_meta") or {}
    for key_src, key_dst in (
        ("super_category", "super_category"),
        ("super-category", "super_category"),
        ("style", "style"),
        ("theme", "theme"),
        ("material", "material"),
        ("dir", "dir"),
    ):
        if key_src in asset_meta and key_dst not in meta:
            meta[key_dst] = asset_meta[key_src]

    return meta


def extract_size_m_from_object_like(obj: Dict[str, Any]) -> List[float]:
    if isinstance(obj.get("size_m"), list) and len(obj["size_m"]) == 3:
        return as_list3(obj["size_m"])

    if isinstance(obj.get("size"), list) and len(obj["size"]) == 3:
        return as_list3(obj["size"])

    min_mm = obj.get("min_size_mm")
    max_mm = obj.get("max_size_mm")
    if isinstance(min_mm, list) and len(min_mm) == 3 and isinstance(max_mm, list) and len(max_mm) == 3:
        return [
            0.5 * (float(min_mm[0]) + float(max_mm[0])) / 1000.0,
            0.5 * (float(min_mm[1]) + float(max_mm[1])) / 1000.0,
            0.5 * (float(min_mm[2]) + float(max_mm[2])) / 1000.0,
        ]

    asset_meta = obj.get("asset_meta") or {}
    if all(k in asset_meta for k in ("size_x", "size_y", "size_z")):
        return [
            float(asset_meta["size_x"]),
            float(asset_meta["size_y"]),
            float(asset_meta["size_z"]),
        ]

    return [0.0, 0.0, 0.0]


def extract_size_bounds_m(obj: Dict[str, Any]) -> Tuple[List[float], List[float], List[float]]:
    if "size_m" in obj:
        s = as_list3(obj["size_m"])
        return s, list(s), list(s)

    min_mm = obj.get("min_size_mm")
    max_mm = obj.get("max_size_mm")
    if isinstance(min_mm, list) and len(min_mm) == 3 and isinstance(max_mm, list) and len(max_mm) == 3:
        size_min = [float(min_mm[0]) / 1000.0, float(min_mm[1]) / 1000.0, float(min_mm[2]) / 1000.0]
        size_max = [float(max_mm[0]) / 1000.0, float(max_mm[1]) / 1000.0, float(max_mm[2]) / 1000.0]
        size_mid = [
            0.5 * (size_min[0] + size_max[0]),
            0.5 * (size_min[1] + size_max[1]),
            0.5 * (size_min[2] + size_max[2]),
        ]
        return size_mid, size_min, size_max

    s = extract_size_m_from_object_like(obj)
    return s, list(s), list(s)


# ============================================================
# Objects -> objects.v1
# ============================================================

def normalize_one_object(obj: Dict[str, Any], index: int) -> Dict[str, Any]:
    src = deepcopy(obj)

    size_m, size_min_m, size_max_m = extract_size_bounds_m(src)
    name = extract_name(src)
    category = extract_category(src, fallback_name=name)

    out = {
        "id": make_object_id(index, src),
        "name": name,
        "category": category,
        "size_m": size_m,
        "size_min_m": size_min_m,
        "size_max_m": size_max_m,
        "color": extract_color(src),
        "constraints": extract_constraints(src),
        "asset": extract_asset_block(src),
        "meta": extract_meta_block_from_object(src),
    }

    # Сохраняем неизвестные поля, чтобы не терять данные
    known_keys = {
        "id", "object_id", "uid",
        "name", "class_name", "class", "type", "category",
        "size_m", "size_min_m", "size_max_m", "size",
        "min_size_mm", "max_size_mm",
        "color", "constraints",
        "asset", "asset_source", "asset_meta",
        "mesh_path", "mesh_fit_mode", "mesh_texture_dirs",
        "jid", "model_jid", "future_jid",
        "meta",
    }
    for k, v in src.items():
        if k not in known_keys:
            out["meta"][k] = deepcopy(v)

    return out


def convert_to_objects_v1(data: Dict[str, Any]) -> Dict[str, Any]:
    src_items = data.get("objects")
    if not isinstance(src_items, list):
        src_items = data.get("items")
    if not isinstance(src_items, list):
        src_items = data.get("placements")
    if not isinstance(src_items, list):
        raise ValueError("Для objects.v1 ожидается список objects/items/placements")

    meta: Dict[str, Any] = {}
    if isinstance(data.get("meta"), dict):
        meta.update(deepcopy(data["meta"]))

    for k, v in data.items():
        if k not in {"schema", "seed", "objects", "items", "placements", "meta"}:
            meta[k] = deepcopy(v)

    out = {
        "schema": "objects.v1",
        "seed": as_int(data.get("seed"), 0),
        "objects": [normalize_one_object(obj, i) for i, obj in enumerate(src_items) if isinstance(obj, dict)],
        "meta": meta,
    }

    return out


def normalize_objects_data(data: Dict[str, Any]) -> Dict[str, Any]:
    return convert_to_objects_v1(data)


# ============================================================
# Placement -> placement.v1
# ============================================================

def extract_position_m(item: Dict[str, Any], size_m: Optional[List[float]], aabb: Optional[Dict[str, Any]]) -> Optional[List[float]]:
    pos = ensure_list3(item.get("position_m"))
    if pos is not None:
        return pos

    center = ensure_list3(item.get("center"))
    if center is not None:
        return center

    pos_xy = item.get("position_room_xy_m")
    if isinstance(pos_xy, list) and len(pos_xy) == 2 and size_m is not None:
        x = as_float(pos_xy[0])
        y = as_float(pos_xy[1])
        z_floor = as_float(item.get("z_floor_m"), 0.0)
        return [x, y, z_floor + size_m[2] / 2.0]

    translation_m = ensure_list3(item.get("translation_m"))
    if translation_m is not None:
        return translation_m

    position = ensure_list3(item.get("position"))
    if position is not None:
        return position

    if aabb is not None:
        return center_from_aabb(aabb)

    return None


def extract_position_m_from_placement_item(obj: Dict[str, Any], size_m: List[float]) -> List[float]:
    pos = extract_position_m(obj, size_m, obj.get("aabb") if isinstance(obj.get("aabb"), dict) else None)
    if pos is not None:
        return pos

    bbox = obj.get("bbox")
    if isinstance(bbox, dict):
        return center_from_aabb(bbox)

    return [0.0, 0.0, size_m[2] / 2.0 if len(size_m) == 3 else 0.0]


def extract_placement_size_m(item: Dict[str, Any], aabb: Optional[Dict[str, Any]]) -> Optional[List[float]]:
    size_m = ensure_list3(item.get("size_m"))
    if size_m is not None:
        return size_m

    size = ensure_list3(item.get("size"))
    if size is not None:
        return size

    bbox_size = ensure_list3(item.get("bbox_size_m"))
    if bbox_size is not None:
        return bbox_size

    mm_min = mm3_to_m3(item.get("min_size_mm"))
    mm_max = mm3_to_m3(item.get("max_size_mm"))
    if mm_min and mm_max:
        if all(abs(a - b) <= 1e-9 for a, b in zip(mm_min, mm_max)):
            return mm_min
        return [(a + b) * 0.5 for a, b in zip(mm_min, mm_max)]

    if aabb is not None:
        return size_from_aabb(aabb)

    asset_meta = deep_copy_dict(item.get("asset_meta"))
    if all(k in asset_meta for k in ("size_x", "size_y", "size_z")):
        return [
            as_float(asset_meta["size_x"]),
            as_float(asset_meta["size_y"]),
            as_float(asset_meta["size_z"]),
        ]

    return None


def extract_rotation_info(item: Dict[str, Any]) -> Tuple[int, float]:
    if is_number(item.get("rotation_deg")):
        yaw_deg = as_float(item.get("rotation_deg"))
    elif is_number(item.get("rotation")):
        yaw_deg = as_float(item.get("rotation"))
    elif is_number(item.get("yaw_deg")):
        yaw_deg = as_float(item.get("yaw_deg"))
    elif is_number(item.get("yaw_rad")):
        yaw_deg = degrees_from_rad(as_float(item.get("yaw_rad")))
    else:
        yaw_deg = 0.0

    rotation_deg = quantize_rot_0_90_180_270(yaw_deg)
    yaw_rad = radians_from_deg(rotation_deg)
    return rotation_deg, yaw_rad


def extract_rotation_block(obj: Dict[str, Any]) -> Tuple[int, int, float]:
    rotation_deg, yaw_rad = extract_rotation_info(obj)
    return rotation_deg, rotation_deg, yaw_rad


def extract_mount_type(obj: Dict[str, Any]) -> Optional[str]:
    if isinstance(obj.get("mount_type"), str) and obj["mount_type"].strip():
        return obj["mount_type"].strip()

    constraints = obj.get("constraints") or {}
    if isinstance(constraints, dict):
        v = constraints.get("mount_type")
        if isinstance(v, str) and v.strip():
            return v.strip()

    return None


def extract_source_block_for_placement(obj: Dict[str, Any], default_placer: Optional[str]) -> Dict[str, Any]:
    source: Dict[str, Any] = {}
    if isinstance(obj.get("source"), dict):
        source.update(obj["source"])

    placement_source = obj.get("placement_source") or default_placer
    if placement_source is not None:
        source["placement_source"] = placement_source

    for key in ("server_class_name", "server_index"):
        if key in obj:
            source[key] = obj[key]

    return source


def extract_meta_block_for_placement(obj: Dict[str, Any]) -> Dict[str, Any]:
    meta = extract_meta_block_from_object(obj)

    for key in (
        "llm_target_position_room_xy_m",
        "llm_target_yaw_deg",
        "llm_attempts_used",
    ):
        if key in obj:
            meta[key] = deepcopy(obj[key])

    forward = obj.get("forward")
    if isinstance(forward, list):
        meta["forward"] = deepcopy(forward)

    return meta


def normalize_one_placement(obj: Dict[str, Any], index: int, default_placer: Optional[str]) -> Dict[str, Any]:
    src = deepcopy(obj)

    name = extract_name(src)
    category = extract_category(src, fallback_name=name)

    aabb = src.get("aabb")
    if not isinstance(aabb, dict):
        bbox = src.get("bbox")
        if isinstance(bbox, dict):
            aabb = deepcopy(bbox)
        else:
            aabb = None
    else:
        aabb = deepcopy(aabb)

    size_m = extract_placement_size_m(src, aabb)
    if size_m is None:
        size_m = [0.0, 0.0, 0.0]

    position_m = extract_position_m(src, size_m, aabb)
    if position_m is None:
        if aabb is not None:
            position_m = center_from_aabb(aabb)
        else:
            position_m = [0.0, 0.0, size_m[2] / 2.0]

    rotation_deg, yaw_deg, yaw_rad = extract_rotation_block(src)

    if aabb is None:
        aabb = build_aabb_from_center_size(position_m, size_m)

    out = {
        "id": make_object_id(index, src),
        "name": name,
        "category": category,
        "position_m": position_m,
        "size_m": size_m,
        "rotation_deg": rotation_deg,
        "yaw_deg": yaw_deg,
        "yaw_rad": yaw_rad,
        "aabb": aabb,
        "mount_type": extract_mount_type(src),
        "wall_contact_side": src.get("wall_contact_side"),
        "constraints": extract_constraints(src),
        "asset": extract_asset_block(src),
        "source": extract_source_block_for_placement(src, default_placer),
        "meta": extract_meta_block_for_placement(src),
    }

    if ensure_list3(src.get("color")) is not None:
        out["color"] = ensure_list3(src.get("color"))

    known_keys = {
        "id", "object_id", "uid",
        "name", "class_name", "class", "type", "category",
        "position_m", "center", "position_room_xy_m", "z_floor_m", "translation_m", "position",
        "size_m", "size", "bbox_size_m", "min_size_mm", "max_size_mm",
        "rotation_deg", "rotation", "yaw_deg", "yaw_rad",
        "aabb", "bbox",
        "mount_type", "wall_contact_side", "constraints",
        "asset", "asset_source", "asset_meta",
        "mesh_path", "mesh_fit_mode", "mesh_texture_dirs",
        "jid", "model_jid", "future_jid",
        "placement_source", "server_class_name", "server_index",
        "llm_target_position_room_xy_m", "llm_target_yaw_deg", "llm_attempts_used",
        "forward", "meta",
        "color",
    }
    for k, v in src.items():
        if k not in known_keys:
            out["meta"][k] = deepcopy(v)

    return {k: v for k, v in out.items() if v is not None}


def convert_to_placement_v1(data: Dict[str, Any]) -> Dict[str, Any]:
    placer = data.get("placer")
    if not isinstance(placer, str) or not placer.strip():
        if "room" in data and isinstance(data.get("items"), list):
            placer = "cube"
        else:
            placer = "unknown"

    src_items = data.get("placements")
    if not isinstance(src_items, list):
        src_items = data.get("items")
    if not isinstance(src_items, list):
        src_items = data.get("objects")
    if not isinstance(src_items, list):
        raise ValueError("Для placement.v1 ожидается список placements/items/objects")

    meta: Dict[str, Any] = {}
    if isinstance(data.get("meta"), dict):
        meta.update(deepcopy(data["meta"]))

    for key in ("server_raw", "llm_raw", "llm_attempts_used"):
        if key in data:
            meta[key] = deepcopy(data[key])

    for k, v in data.items():
        if k not in {
            "schema", "placer", "mode", "placements", "items", "objects", "room", "meta",
            "server_raw", "llm_raw", "llm_attempts_used",
        }:
            if k not in meta:
                meta[k] = deepcopy(v)

    return {
        "schema": "placement.v1",
        "placer": placer,
        "mode": data.get("mode"),
        "placements": [
            normalize_one_placement(obj, i, default_placer=placer)
            for i, obj in enumerate(src_items)
            if isinstance(obj, dict)
        ],
        "meta": meta,
    }


def normalize_placement_data(data: Dict[str, Any]) -> Dict[str, Any]:
    return convert_to_placement_v1(data)


# ============================================================
# Scene -> scene.v1
# ============================================================

def convert_to_scene_v1(data: Dict[str, Any]) -> Dict[str, Any]:
    room = data.get("room")
    if not isinstance(room, dict):
        raise ValueError("Для scene.v1 требуется поле room")

    placements_raw = data.get("placements")
    if not isinstance(placements_raw, list):
        placements_raw = data.get("items")
    if not isinstance(placements_raw, list):
        raise ValueError("Для scene.v1 требуется список placements/items")

    placer = as_str(data.get("placer"), "unknown")
    mode = data.get("mode")

    normalized_placement = convert_to_placement_v1({
        "placer": placer,
        "mode": mode,
        "placements": placements_raw,
        "meta": data.get("meta", {}),
    })

    meta: Dict[str, Any] = {}
    if isinstance(data.get("meta"), dict):
        meta.update(deepcopy(data["meta"]))

    if "placer" in data:
        meta.setdefault("placer", data["placer"])
    if "mode" in data:
        meta.setdefault("mode", data["mode"])

    for k, v in data.items():
        if k not in {"schema", "room", "placements", "items", "placer", "mode", "meta"}:
            meta[k] = deepcopy(v)

    return {
        "schema": "scene.v1",
        "room": normalize_room_dict(room),
        "placements": normalized_placement["placements"],
        "meta": meta,
    }


def normalize_scene_data(data: Dict[str, Any]) -> Dict[str, Any]:
    return convert_to_scene_v1(data)


def build_scene_from_room_and_placement(room_data: Dict[str, Any], placement_data: Dict[str, Any]) -> Dict[str, Any]:
    normalized_placement = convert_to_placement_v1(placement_data)

    meta: Dict[str, Any] = {
        "placer": normalized_placement.get("placer"),
        "mode": normalized_placement.get("mode"),
    }
    if isinstance(normalized_placement.get("meta"), dict) and normalized_placement["meta"]:
        meta["placement_meta"] = normalized_placement["meta"]

    return {
        "schema": "scene.v1",
        "room": normalize_room_dict(room_data),
        "placements": normalized_placement["placements"],
        "meta": meta,
    }


# ============================================================
# Универсальный маршрут
# ============================================================

def convert_json(data: Dict[str, Any], target: str, input_kind: Optional[str] = None) -> Dict[str, Any]:
    kind = input_kind or detect_input_kind(data)

    if target == "auto":
        target = auto_target_from_input_kind(kind)

    if target == "objects":
        return convert_to_objects_v1(data)

    if target == "placement":
        return convert_to_placement_v1(data)

    if target == "scene":
        return convert_to_scene_v1(data)

    raise ValueError(f"Неизвестный target: {target}")


# ============================================================
# CLI
# ============================================================

def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Один универсальный конвертер старых и новых JSON форматов pipeline"
    )

    p.add_argument("--input", default=None, help="Входной JSON")
    p.add_argument("--output", required=True, help="Выходной JSON")

    p.add_argument(
        "--target",
        choices=["auto", "objects", "placement", "scene"],
        default="auto",
        help="Целевой канонический формат",
    )

    p.add_argument(
        "--kind",
        choices=[
            "auto",
            "objects_v1",
            "placement_v1",
            "scene_v1",
            "objects_like",
            "placement_like",
            "scene_like",
            "cube_or_old_scene",
            "items_like",
        ],
        default="auto",
        help="Явно задать тип входа, если autodetect ошибается",
    )

    p.add_argument("--room", default=None, help="room.json для сборки scene")
    p.add_argument("--placement", default=None, help="placement.json для сборки scene")
    p.add_argument("--print-kind", action="store_true", help="Напечатать определённый тип входного JSON")

    return p


def main() -> None:
    args = build_cli().parse_args()

    # Спец-режим: scene из room + placement
    if args.target == "scene" and args.room and args.placement:
        room_data = load_json(args.room)
        placement_data = load_json(args.placement)
        out = build_scene_from_room_and_placement(room_data, placement_data)
        save_json(args.output, out)
        print(f"OK: scene.v1 saved -> {Path(args.output).expanduser().resolve()}")
        return

    if not args.input:
        raise RuntimeError("Нужно передать --input, либо для target=scene передать --room и --placement")

    data = load_json(args.input)

    detected_kind = detect_input_kind(data) if args.kind == "auto" else args.kind
    if args.print_kind:
        print(f"input_kind = {detected_kind}")

    out = convert_json(
        data=data,
        target=args.target,
        input_kind=detected_kind,
    )

    save_json(args.output, out)

    print(f"OK: input_kind={detected_kind}")
    print(f"OK: target_schema={out.get('schema')}")
    print(f"OK: saved -> {Path(args.output).expanduser().resolve()}")


if __name__ == "__main__":
    main()