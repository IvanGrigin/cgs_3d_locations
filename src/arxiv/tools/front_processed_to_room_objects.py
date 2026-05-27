#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
front_processed_to_room_objects.py

Пакетный конвертер из 3D-FRONT-processed/*.json в подготовленные данные.

Что делает:
1. Проходит по одному JSON-файлу или по всей папке с 3D-FRONT-processed.
2. Для КАЖДОЙ комнаты сохраняет:
   - room.json
   - objects.v1.json
   - scene_gt.v1.json (если задан --write-scene-gt)
   - conversion_report.json
3. Не выкидывает нераспознанные объекты.
4. Восстанавливает размеры по каскаду источников:
   - prepared_model_info.json
   - bbox модели из 3D-FUTURE normalized_model.obj / raw_model.obj
   - полная сцена 3D-FRONT/<uid>.json по ref -> furniture.uid
   - сырой bbox/size из processed JSON
5. Восстанавливает семантические метки по каскаду:
   - prepared.category
   - raw_front.category
   - raw_front.title
   - sourceCategoryId -> label (если передан индекс)
   - processed.category
   - prepared.super_category
   - fallback
6. Комнаты без корректного polygon по умолчанию пропускаются.

Типовой запуск:
python src/tools/front_processed_to_room_objects.py \
  --input data/sourse/3D-FRONT/3D-FRONT-processed \
  --front-root data/sourse/3D-FRONT/3D-FRONT \
  --prepared-info data/sourse/3D-FRONT/prepared_model_info.json \
  --future-root data/sourse/3D-FRONT/3D-FUTURE-model \
  --source-category-label-index data/sourse/3D-FRONT/source_category_label_index/source_category_id_to_label.json \
  --out-dir data/sourse/3D-FRONT/3D-FRONT-prepared-for-estimation \
  --write-scene-gt
"""

from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ============================================================
# Глобальные кэши
# ============================================================

OBJ_SIZE_CACHE: Dict[str, Optional[Tuple[float, float, float]]] = {}
RAW_FRONT_CACHE: Dict[str, Optional[Dict[str, Any]]] = {}
RAW_FRONT_INDEX_CACHE: Dict[str, Dict[str, Dict[str, Any]]] = {}
SOURCE_CATEGORY_LABEL_INDEX_CACHE: Optional[Dict[str, str]] = None


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


def ensure_dir(path: str | Path) -> Path:
    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


# ============================================================
# Базовые утилиты
# ============================================================

def as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def as_str(x: Any, default: str = "") -> str:
    if x is None:
        return default
    return str(x)


def round6(x: float) -> float:
    return round(float(x), 6)


def unique_preserve_order(values: Iterable[Any]) -> List[Any]:
    out: List[Any] = []
    seen = set()
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def normalize_name_for_fs(s: str) -> str:
    bad = '<>:"/\\|?*'
    out = s
    for ch in bad:
        out = out.replace(ch, "_")
    out = out.replace(" ", "_")
    while "__" in out:
        out = out.replace("__", "_")
    out = out.strip("_")
    return out or "room"


def safe_relpath(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path.resolve())


def polygon_area(poly: List[Tuple[float, float]]) -> float:
    if len(poly) < 3:
        return 0.0
    s = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) * 0.5


def polygon_bounds(poly: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def translate_polygon(poly: List[Tuple[float, float]], dx: float, dy: float) -> List[Tuple[float, float]]:
    return [(x + dx, y + dy) for x, y in poly]


def degrees_to_radians(deg: float) -> float:
    return math.radians(float(deg))


def quantize_rot_0_90_180_270(deg: float) -> int:
    a = float(deg) % 360.0
    allowed = (0.0, 90.0, 180.0, 270.0)
    best = min(allowed, key=lambda t: (abs(((a - t + 180.0) % 360.0) - 180.0), t))
    return int(best)


def round_size_triplet(size_xyz: Tuple[float, float, float]) -> List[float]:
    return [round6(size_xyz[0]), round6(size_xyz[1]), round6(size_xyz[2])]


def has_nonzero_size(size_m: List[float]) -> bool:
    return (
        len(size_m) >= 3
        and max(abs(as_float(size_m[0])), abs(as_float(size_m[1])), abs(as_float(size_m[2]))) > 1e-12
    )


# ============================================================
# sourceCategoryId -> label
# ============================================================

def load_source_category_label_index(path: Optional[Path]) -> Dict[str, str]:
    global SOURCE_CATEGORY_LABEL_INDEX_CACHE

    if path is None:
        return {}

    if SOURCE_CATEGORY_LABEL_INDEX_CACHE is not None:
        return SOURCE_CATEGORY_LABEL_INDEX_CACHE

    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise RuntimeError(f"Файл sourceCategoryId->label не найден: {p}")

    data = load_json(p)
    if not isinstance(data, dict):
        raise RuntimeError("source_category_id_to_label.json должен быть dict")

    out: Dict[str, str] = {}
    for k, v in data.items():
        ks = as_str(k).strip()
        vs = as_str(v).strip()
        if ks and vs:
            out[ks] = vs

    SOURCE_CATEGORY_LABEL_INDEX_CACHE = out
    return out


def extract_label_from_source_category_id(
    raw_front_entry: Optional[Dict[str, Any]],
    source_category_label_index: Dict[str, str],
) -> Optional[str]:
    if not isinstance(raw_front_entry, dict):
        return None

    source_category_id = as_str(raw_front_entry.get("sourceCategoryId")).strip()
    if not source_category_id:
        return None

    label = as_str(source_category_label_index.get(source_category_id)).strip()
    return label or None


# ============================================================
# Работа с model id
# ============================================================

def extract_model_id_candidates_from_obj(obj: Dict[str, Any]) -> List[str]:
    raw_candidates = [
        obj.get("jid"),
        obj.get("model_id"),
        obj.get("modelId"),
        obj.get("model_jid"),
        obj.get("mesh_uid"),
        obj.get("mesh_id"),
        obj.get("uid"),
        obj.get("ref"),
    ]
    out: List[str] = []
    for v in raw_candidates:
        s = as_str(v).strip()
        if s:
            out.append(s)
    return unique_preserve_order(out)


def extract_model_id_candidates_from_prepared_meta(meta: Dict[str, Any]) -> List[str]:
    raw_candidates = [
        meta.get("model_id"),
        meta.get("jid"),
        meta.get("model_jid"),
        meta.get("id"),
        meta.get("uid"),
    ]
    out: List[str] = []
    for v in raw_candidates:
        s = as_str(v).strip()
        if s:
            out.append(s)
    return unique_preserve_order(out)


# ============================================================
# Индекс prepared_model_info
# ============================================================

def build_prepared_index(data: Any) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}

    def add_meta(meta: Dict[str, Any], fallback_key: Optional[str] = None) -> None:
        candidates = extract_model_id_candidates_from_prepared_meta(meta)
        if fallback_key:
            fk = as_str(fallback_key).strip()
            if fk:
                candidates.append(fk)
        for c in unique_preserve_order(candidates):
            out[c] = meta

    if isinstance(data, list):
        for row in data:
            if isinstance(row, dict):
                add_meta(row)
        return out

    if isinstance(data, dict):
        direct_ok = True
        for k, v in data.items():
            if not isinstance(k, str) or not isinstance(v, dict):
                direct_ok = False
                break

        if direct_ok:
            for k, v in data.items():
                add_meta(v, fallback_key=k)
            if out:
                return out

        stack: List[Any] = [data]
        while stack:
            cur = stack.pop()
            if isinstance(cur, dict):
                if any(k in cur for k in ("model_id", "jid", "model_jid")):
                    add_meta(cur)
                for v in cur.values():
                    if isinstance(v, (dict, list)):
                        stack.append(v)
            elif isinstance(cur, list):
                for v in cur:
                    if isinstance(v, (dict, list)):
                        stack.append(v)

    return out


def resolve_prepared_meta_for_obj(
    obj: Dict[str, Any],
    prepared_index: Dict[str, Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    for candidate in extract_model_id_candidates_from_obj(obj):
        meta = prepared_index.get(candidate)
        if isinstance(meta, dict):
            return meta, candidate
    return None, None


# ============================================================
# Работа с полным 3D-FRONT
# ============================================================

def load_raw_front_scene_by_uid(front_root: Path, source_uid: str) -> Optional[Dict[str, Any]]:
    if not source_uid:
        return None

    if source_uid in RAW_FRONT_CACHE:
        return RAW_FRONT_CACHE[source_uid]

    path = front_root / f"{source_uid}.json"
    if not path.is_file():
        RAW_FRONT_CACHE[source_uid] = None
        return None

    try:
        scene = load_json(path)
        if isinstance(scene, dict):
            RAW_FRONT_CACHE[source_uid] = scene
            return scene
    except Exception:
        pass

    RAW_FRONT_CACHE[source_uid] = None
    return None


def build_raw_front_indices(scene: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    scene_uid = as_str(scene.get("uid"))
    if scene_uid in RAW_FRONT_INDEX_CACHE:
        return RAW_FRONT_INDEX_CACHE[scene_uid]

    furniture_by_uid: Dict[str, Dict[str, Any]] = {}
    mesh_by_uid: Dict[str, Dict[str, Any]] = {}

    furniture = scene.get("furniture")
    if isinstance(furniture, list):
        for row in furniture:
            if not isinstance(row, dict):
                continue
            uid = as_str(row.get("uid")).strip()
            if uid:
                furniture_by_uid[uid] = row

    mesh = scene.get("mesh")
    if isinstance(mesh, list):
        for row in mesh:
            if not isinstance(row, dict):
                continue
            uid = as_str(row.get("uid")).strip()
            if uid:
                mesh_by_uid[uid] = row

    out = {
        "furniture_by_uid": furniture_by_uid,
        "mesh_by_uid": mesh_by_uid,
    }
    RAW_FRONT_INDEX_CACHE[scene_uid] = out
    return out


def find_raw_front_entry_for_processed_obj(
    raw_front_scene: Optional[Dict[str, Any]],
    processed_obj: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not isinstance(raw_front_scene, dict):
        return None, None

    ref = as_str(processed_obj.get("ref")).strip()
    if not ref:
        return None, None

    idx = build_raw_front_indices(raw_front_scene)
    furniture_by_uid = idx.get("furniture_by_uid", {})
    mesh_by_uid = idx.get("mesh_by_uid", {})

    if ref in furniture_by_uid:
        return furniture_by_uid[ref], "furniture"

    if ref in mesh_by_uid:
        return mesh_by_uid[ref], "mesh"

    return None, None


def extract_size_full_xyz_from_raw_front_entry(
    entry: Optional[Dict[str, Any]],
) -> Tuple[Optional[Tuple[float, float, float]], Optional[str]]:
    if not isinstance(entry, dict):
        return None, None

    for key in ("size", "bbox"):
        val = entry.get(key)
        t = _triplet_from_any(val)
        if t is not None:
            return (abs(t[0]), abs(t[1]), abs(t[2])), f"raw_front:{key}"

    return None, None


def extract_model_id_from_raw_front_entry(entry: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(entry, dict):
        return None

    for key in ("jid", "model_id", "modelId"):
        s = as_str(entry.get(key)).strip()
        if s:
            return s
    return None


def extract_name_category_from_raw_front_entry(
    entry: Optional[Dict[str, Any]],
) -> Tuple[Optional[str], Optional[str]]:
    if not isinstance(entry, dict):
        return None, None

    category = as_str(entry.get("category")).strip() or None
    title = as_str(entry.get("title")).strip() or None

    if category:
        return category, category
    if title:
        return title, title
    return None, None


# ============================================================
# Room type / title
# ============================================================

def normalize_room_type(raw_type: str) -> str:
    t = (raw_type or "").strip().lower()

    if "masterbedroom" in t or "secondbedroom" in t or "bedroom" in t:
        return "bedroom"
    if "living" in t:
        return "livingroom"
    if "dining" in t:
        return "diningroom"
    if "kids" in t or "children" in t:
        return "kids"
    if "study" in t or "office" in t or "library" in t:
        return "office"
    if "bathroom" in t:
        return "bathroom"
    if "balcony" in t:
        return "balcony"
    if "kitchen" in t:
        return "kitchen"
    return "other"


def room_title_ru(room_type: str) -> str:
    mapping = {
        "bedroom": "Спальня",
        "livingroom": "Гостиная",
        "diningroom": "Столовая",
        "kids": "Детская",
        "office": "Кабинет",
        "bathroom": "Ванная комната",
        "balcony": "Балкон",
        "kitchen": "Кухня",
        "other": "Комната",
    }
    return mapping.get(room_type, "Комната")


# ============================================================
# Извлечение размеров
# ============================================================

def _triplet_from_mapping(d: Dict[str, Any], aliases: List[Tuple[str, str, str]]) -> Optional[Tuple[float, float, float]]:
    for ax, ay, az in aliases:
        if ax in d and ay in d and az in d:
            return (
                as_float(d.get(ax), 0.0),
                as_float(d.get(ay), 0.0),
                as_float(d.get(az), 0.0),
            )
    return None


def _triplet_from_any(v: Any) -> Optional[Tuple[float, float, float]]:
    if isinstance(v, list) and len(v) >= 3 and all(not isinstance(x, (list, tuple, dict)) for x in v[:3]):
        return (as_float(v[0]), as_float(v[1]), as_float(v[2]))

    if isinstance(v, tuple) and len(v) >= 3 and all(not isinstance(x, (list, tuple, dict)) for x in v[:3]):
        return (as_float(v[0]), as_float(v[1]), as_float(v[2]))

    if isinstance(v, dict):
        return _triplet_from_mapping(
            v,
            aliases=[
                ("x", "y", "z"),
                ("width", "height", "depth"),
                ("w", "h", "d"),
                ("size_x", "size_y", "size_z"),
                ("dx", "dy", "dz"),
            ],
        )

    return None


def _size_from_bounds_dict(d: Dict[str, Any]) -> Optional[Tuple[float, float, float]]:
    if all(k in d for k in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")):
        sx = as_float(d["x_max"]) - as_float(d["x_min"])
        sy = as_float(d["y_max"]) - as_float(d["y_min"])
        sz = as_float(d["z_max"]) - as_float(d["z_min"])
        return abs(sx), abs(sy), abs(sz)

    for lo_key, hi_key in (("min", "max"), ("lower", "upper")):
        if lo_key in d and hi_key in d:
            lo = _triplet_from_any(d[lo_key])
            hi = _triplet_from_any(d[hi_key])
            if lo is not None and hi is not None:
                sx = hi[0] - lo[0]
                sy = hi[1] - lo[1]
                sz = hi[2] - lo[2]
                return abs(sx), abs(sy), abs(sz)

    return None


def _size_from_bounds_list(v: Any) -> Optional[Tuple[float, float, float]]:
    if not isinstance(v, list):
        return None

    if len(v) >= 3 and all(not isinstance(x, (list, tuple, dict)) for x in v[:3]):
        return (
            abs(as_float(v[0])),
            abs(as_float(v[1])),
            abs(as_float(v[2])),
        )

    if len(v) == 1 and isinstance(v[0], (list, tuple)) and len(v[0]) >= 3:
        return (
            abs(as_float(v[0][0])),
            abs(as_float(v[0][1])),
            abs(as_float(v[0][2])),
        )

    if len(v) >= 2:
        a = v[0]
        b = v[1]
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) >= 3 and len(b) >= 3:
            return (
                abs(as_float(b[0]) - as_float(a[0])),
                abs(as_float(b[1]) - as_float(a[1])),
                abs(as_float(b[2]) - as_float(a[2])),
            )

    return None


def extract_raw_size_full_xyz_front(obj: Dict[str, Any]) -> Tuple[Optional[Tuple[float, float, float]], Optional[str]]:
    explicit_full_keys = [
        "size",
        "size_xyz",
        "dimensions",
        "dimension",
        "extent",
        "extents",
        "bbox_size",
        "box_size",
    ]
    for key in explicit_full_keys:
        if key in obj:
            t = _triplet_from_any(obj[key])
            if t is not None:
                sx, sy, sz = abs(t[0]), abs(t[1]), abs(t[2])
                return (sx, sy, sz), f"raw:{key}"

    bounds_keys = [
        "bbox",
        "aabb",
        "bounds",
        "bounding_box",
        "box",
        "obb",
        "world_bbox",
    ]
    for key in bounds_keys:
        val = obj.get(key)

        if isinstance(val, dict):
            t = _size_from_bounds_dict(val)
            if t is not None:
                return t, f"raw:{key}"

        if isinstance(val, list):
            t = _size_from_bounds_list(val)
            if t is not None:
                return t, f"raw:{key}"

    half_keys = [
        "half_extents",
        "half_extent",
        "half_size",
        "half_sizes",
    ]
    for key in half_keys:
        if key in obj:
            t = _triplet_from_any(obj[key])
            if t is not None:
                return (abs(t[0]) * 2.0, abs(t[1]) * 2.0, abs(t[2]) * 2.0), f"raw:{key}*2"

    return None, None


def convert_scale(obj: Dict[str, Any]) -> Tuple[float, float, float]:
    scale = obj.get("scale")
    if isinstance(scale, list) and len(scale) >= 3:
        return (
            abs(as_float(scale[0], 1.0)),
            abs(as_float(scale[1], 1.0)),
            abs(as_float(scale[2], 1.0)),
        )
    return 1.0, 1.0, 1.0


def prepared_size_full_xyz(
    prepared_meta: Dict[str, Any],
    assume_half_extents: bool,
) -> Tuple[float, float, float]:
    if not all(k in prepared_meta for k in ("size_x", "size_y", "size_z")):
        raise RuntimeError("В prepared_meta нет size_x/size_y/size_z")

    sx = as_float(prepared_meta["size_x"])
    sy = as_float(prepared_meta["size_y"])
    sz = as_float(prepared_meta["size_z"])

    if assume_half_extents:
        sx *= 2.0
        sy *= 2.0
        sz *= 2.0

    return abs(sx), abs(sy), abs(sz)


def load_obj_size_full_xyz(obj_path: Path) -> Optional[Tuple[float, float, float]]:
    key = str(obj_path.resolve())
    if key in OBJ_SIZE_CACHE:
        return OBJ_SIZE_CACHE[key]

    if not obj_path.is_file():
        OBJ_SIZE_CACHE[key] = None
        return None

    min_x = min_y = min_z = None
    max_x = max_y = max_z = None

    try:
        with obj_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.startswith("v "):
                    continue
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                x = as_float(parts[1], 0.0)
                y = as_float(parts[2], 0.0)
                z = as_float(parts[3], 0.0)

                min_x = x if min_x is None else min(min_x, x)
                min_y = y if min_y is None else min(min_y, y)
                min_z = z if min_z is None else min(min_z, z)

                max_x = x if max_x is None else max(max_x, x)
                max_y = y if max_y is None else max(max_y, y)
                max_z = z if max_z is None else max(max_z, z)

        if None in (min_x, min_y, min_z, max_x, max_y, max_z):
            OBJ_SIZE_CACHE[key] = None
            return None

        size_xyz = (
            abs(max_x - min_x),
            abs(max_y - min_y),
            abs(max_z - min_z),
        )
        OBJ_SIZE_CACHE[key] = size_xyz
        return size_xyz
    except Exception:
        OBJ_SIZE_CACHE[key] = None
        return None


def mesh_path_for_model_id(future_root: Path, model_id: str) -> Tuple[Optional[str], List[str]]:
    if not model_id:
        return None, []

    model_dir = future_root / model_id
    tex_dirs = [str(model_dir.resolve())] if model_dir.exists() else []

    normalized_obj = model_dir / "normalized_model.obj"
    raw_obj = model_dir / "raw_model.obj"

    if normalized_obj.is_file():
        return str(normalized_obj.resolve()), tex_dirs
    if raw_obj.is_file():
        return str(raw_obj.resolve()), tex_dirs
    return None, tex_dirs


def resolve_object_size_and_source(
    obj: Dict[str, Any],
    prepared_meta: Optional[Dict[str, Any]],
    assume_half_extents: bool,
    mesh_size_xyz: Optional[Tuple[float, float, float]],
    raw_front_entry: Optional[Dict[str, Any]],
) -> Tuple[Tuple[float, float, float], str]:
    """
    Возвращает размер в системе objects.v1:
      [x_on_floor, y_on_floor, z_up]
    """
    scx, scy, scz = convert_scale(obj)

    if isinstance(prepared_meta, dict) and all(k in prepared_meta for k in ("size_x", "size_y", "size_z")):
        sx, sy, sz = prepared_size_full_xyz(prepared_meta, assume_half_extents=assume_half_extents)
        return (sx * scx, sz * scz, sy * scy), "prepared+scale"

    if mesh_size_xyz is not None:
        sx, sy, sz = mesh_size_xyz
        return (sx * scx, sz * scz, sy * scy), "mesh_obj_bbox+scale"

    raw_front_xyz, raw_front_source = extract_size_full_xyz_from_raw_front_entry(raw_front_entry)
    if raw_front_xyz is not None:
        sx, sy, sz = raw_front_xyz
        return (sx * scx, sz * scz, sy * scy), raw_front_source or "raw_front"

    raw_xyz_front, raw_source = extract_raw_size_full_xyz_front(obj)
    if raw_xyz_front is not None:
        sx, sy, sz = raw_xyz_front
        return (sx * scx, sz * scz, sy * scy), raw_source or "raw"

    return (0.0, 0.0, 0.0), "missing"


# ============================================================
# Room
# ============================================================

def room_has_valid_polygon(room_obj: Dict[str, Any]) -> bool:
    raw_poly = room_obj.get("polygon")
    if not isinstance(raw_poly, list) or len(raw_poly) < 3:
        return False

    poly_xz_world: List[Tuple[float, float]] = []
    for pt in raw_poly:
        if not isinstance(pt, dict):
            continue
        poly_xz_world.append((as_float(pt.get("x")), as_float(pt.get("z"))))

    return len(poly_xz_world) >= 3


def convert_opening_list(
    items: Any,
    shift_x: float,
    shift_z: float,
    kind: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    if not isinstance(items, list):
        return out

    for i, it in enumerate(items):
        if not isinstance(it, dict):
            continue

        center = it.get("center")
        center_xy = None
        center_h = None

        if isinstance(center, dict):
            cx = as_float(center.get("x"), 0.0) + shift_x
            cz = as_float(center.get("z"), 0.0) + shift_z
            cy = as_float(center.get("y"), 0.0)
            center_xy = [round6(cx), round6(cz)]
            center_h = round6(cy)

        rec = {
            "id": as_str(it.get("uid") or it.get("id"), f"{kind}_{i}"),
            "center_room_xy_m": center_xy,
            "center_height_m": center_h,
            "yaw_deg": round6(as_float(it.get("yaw_deg"), 0.0)),
            "dist_to_room_m": round6(as_float(it.get("dist_to_room"), 0.0)),
            "meta": {},
        }

        for k, v in it.items():
            if k not in {"uid", "id", "center", "yaw_deg", "dist_to_room"}:
                rec["meta"][k] = deepcopy(v)

        out.append(rec)

    return out


def estimate_ceiling_height_m(
    room_obj: Dict[str, Any],
    prepared_index: Dict[str, Dict[str, Any]],
    raw_front_scene: Optional[Dict[str, Any]],
    future_root: Path,
    assume_half_extents: bool,
    default_ceiling_height: float,
) -> float:
    objects = room_obj.get("objects")
    if not isinstance(objects, list):
        return round6(default_ceiling_height)

    def estimate_from_subset(only_valid: bool) -> float:
        best_top = 0.0

        for obj in objects:
            if not isinstance(obj, dict):
                continue
            if only_valid and not obj.get("valid", True):
                continue

            prepared_meta, matched_model_id = resolve_prepared_meta_for_obj(obj, prepared_index)
            raw_front_entry, _ = find_raw_front_entry_for_processed_obj(raw_front_scene, obj)
            raw_front_model_id = extract_model_id_from_raw_front_entry(raw_front_entry)
            raw_jid = as_str(obj.get("jid")).strip()

            model_candidates = unique_preserve_order(
                [x for x in [matched_model_id, raw_front_model_id, raw_jid] if x]
            )

            mesh_size_xyz = None
            for candidate in model_candidates:
                candidate_mesh_path, _ = mesh_path_for_model_id(future_root, candidate)
                if candidate_mesh_path:
                    mesh_size_xyz = load_obj_size_full_xyz(Path(candidate_mesh_path))
                    if mesh_size_xyz is not None:
                        break

            size_m, _ = resolve_object_size_and_source(
                obj=obj,
                prepared_meta=prepared_meta,
                assume_half_extents=assume_half_extents,
                mesh_size_xyz=mesh_size_xyz,
                raw_front_entry=raw_front_entry,
            )

            height = size_m[2]
            if height <= 1e-9:
                continue

            pos = obj.get("pos") or {}
            y_center = as_float(pos.get("y"), 0.0)
            top = y_center + 0.5 * height
            if top > best_top:
                best_top = top

        return best_top

    best_top = estimate_from_subset(only_valid=True)
    if best_top <= 1e-6:
        best_top = estimate_from_subset(only_valid=False)

    if best_top <= 1e-6:
        return round6(default_ceiling_height)

    est = max(default_ceiling_height, best_top + 0.1)
    est = math.ceil(est * 10.0) / 10.0
    return round6(est)


def build_room_json(
    root: Dict[str, Any],
    room_obj: Dict[str, Any],
    prepared_index: Dict[str, Dict[str, Any]],
    raw_front_scene: Optional[Dict[str, Any]],
    future_root: Path,
    assume_half_extents: bool,
    default_ceiling_height: float,
) -> Dict[str, Any]:
    raw_poly = room_obj.get("polygon")
    if not isinstance(raw_poly, list) or len(raw_poly) < 3:
        raise RuntimeError("У комнаты отсутствует корректный polygon")

    poly_xz_world: List[Tuple[float, float]] = []
    for pt in raw_poly:
        if not isinstance(pt, dict):
            continue
        poly_xz_world.append((as_float(pt.get("x")), as_float(pt.get("z"))))

    if len(poly_xz_world) < 3:
        raise RuntimeError("polygon содержит слишком мало корректных вершин")

    xmin, zmin, xmax, zmax = polygon_bounds(poly_xz_world)
    shift_x = -xmin
    shift_z = -zmin

    poly_local = translate_polygon(poly_xz_world, shift_x, shift_z)
    width_m = xmax - xmin
    depth_m = zmax - zmin
    area_m2 = polygon_area(poly_local)

    raw_room_id = as_str(room_obj.get("id"), "room")
    raw_room_type = as_str(room_obj.get("type"), "other")
    room_type = normalize_room_type(raw_room_type)

    ceiling_height_m = estimate_ceiling_height_m(
        room_obj=room_obj,
        prepared_index=prepared_index,
        raw_front_scene=raw_front_scene,
        future_root=future_root,
        assume_half_extents=assume_half_extents,
        default_ceiling_height=default_ceiling_height,
    )

    room_json = {
        "id": raw_room_id,
        "name": raw_room_id,
        "title_ru": room_title_ru(room_type),
        "room_type": room_type,
        "style_hint": raw_room_type,
        "width_m": round6(width_m),
        "depth_m": round6(depth_m),
        "area_m2": round6(area_m2),
        "ceiling_height_m": round6(ceiling_height_m),
        "floor_polygon": [
            {"x": round6(x), "y": round6(z)}
            for x, z in poly_local
        ],
        "floor_polygon_xz": [
            {"x": round6(x), "z": round6(z)}
            for x, z in poly_local
        ],
        "doors": convert_opening_list(room_obj.get("doors"), shift_x, shift_z, "door"),
        "windows": convert_opening_list(room_obj.get("windows"), shift_x, shift_z, "window"),
        "openings": [],
        "notes": {
            "source": "3D-FRONT-processed",
            "source_uid": as_str(root.get("uid")),
            "source_room_type_raw": raw_room_type,
            "polygon_was_shifted_to_local_origin": True,
            "shift_x_m": round6(shift_x),
            "shift_z_m": round6(shift_z),
            "comment": "Комната переведена в локальные координаты: min_x -> 0, min_z -> 0",
        },
        "version": "1.0",
        "units": "m",
    }

    return room_json


# ============================================================
# Objects
# ============================================================

def infer_constraints(category: str) -> Dict[str, Any]:
    c = (category or "").strip().lower()

    if "ceiling lamp" in c or "pendant" in c or "chandelier" in c or "pendant lamp" in c:
        return {"mount_type": "ceiling"}
    if c == "lamp":
        return {"mount_type": "ceiling"}

    floor_basic = {
        "nightstand",
        "chair",
        "corner/side table",
        "corner table",
        "side table",
        "desk",
        "sofa",
        "coffee table",
        "armchair",
        "tea table",
        "tv stand",
        "l-shaped sofa",
        "table",
        "dining chair",
    }
    if c in floor_basic:
        return {"mount_type": "floor"}

    wall_back = {
        "bed",
        "king-size bed",
        "single bed",
        "kids bed",
        "bunk bed",
        "wardrobe",
        "drawer chest / corner cabinet",
        "drawer chest",
        "cabinet",
        "tv stand",
        "dresser",
        "bookcase / jewelry armoire",
        "bookcase",
        "storage unit/armoire",
        "media unit/floor-based media unit",
        "shelf",
        "sideboard / side cabinet / console",
    }
    if c in wall_back:
        return {"touch_wall": {"sides": ["back"]}}

    return {}


def object_id_from_raw(obj: Dict[str, Any], index: int) -> str:
    raw = as_str(
        obj.get("instanceid")
        or obj.get("uid")
        or obj.get("id")
        or obj.get("ref")
        or f"obj_{index + 1:05d}",
        f"obj_{index + 1:05d}",
    )
    raw = raw.replace("/", "_").replace("\\", "_")
    return raw


def build_name_category(
    obj: Dict[str, Any],
    prepared_meta: Optional[Dict[str, Any]],
    raw_front_entry: Optional[Dict[str, Any]],
    source_category_label_index: Dict[str, str],
    index: int,
) -> Tuple[str, str, str, str]:
    """
    Возвращает:
      name, category, super_category, naming_source

    Приоритет:
      prepared.category
      raw_front.category
      raw_front.title
      sourceCategoryId-mapped label
      processed.category
      prepared.super_category
      fallback
    """
    category_prepared = as_str(prepared_meta.get("category")) if isinstance(prepared_meta, dict) else ""
    super_category = ""
    if isinstance(prepared_meta, dict):
        super_category = as_str(prepared_meta.get("super-category") or prepared_meta.get("super_category"))

    raw_front_name, raw_front_category = extract_name_category_from_raw_front_entry(raw_front_entry)
    mapped_label = extract_label_from_source_category_id(raw_front_entry, source_category_label_index)
    category_raw = as_str(obj.get("category")).strip()

    if category_prepared:
        name = category_prepared
        category = category_prepared
        naming_source = "prepared.category"
    elif raw_front_category:
        name = raw_front_category
        category = raw_front_category
        naming_source = "raw_front.category"
    elif raw_front_name:
        name = raw_front_name
        category = raw_front_name
        naming_source = "raw_front.title"
    elif mapped_label:
        name = mapped_label
        category = mapped_label
        naming_source = "sourceCategoryId.label"
    elif category_raw:
        name = category_raw
        category = category_raw
        naming_source = "processed.category"
    elif super_category:
        name = super_category
        category = super_category
        naming_source = "prepared.super_category"
    else:
        name = f"UnrecognizedObject_{index + 1}"
        category = "UnrecognizedObject"
        naming_source = "fallback"

    return name, category, super_category, naming_source


def classify_zero_size_reason(
    *,
    resolved_model_id: Optional[str],
    mesh_path: Optional[str],
    raw_front_entry: Optional[Dict[str, Any]],
    obj: Dict[str, Any],
) -> str:
    raw_front_has_geometry = False
    if isinstance(raw_front_entry, dict):
        raw_front_has_geometry = any(raw_front_entry.get(key) is not None for key in ("size", "bbox"))

    processed_has_geometry = any(obj.get(key) is not None for key in ("size", "bbox", "aabb", "bounds"))

    if resolved_model_id and mesh_path is None and not raw_front_has_geometry and not processed_has_geometry:
        return "known_model_but_no_mesh_and_no_bbox"

    if resolved_model_id and mesh_path is not None and not raw_front_has_geometry and not processed_has_geometry:
        return "mesh_exists_but_obj_bbox_unavailable"

    if resolved_model_id and not raw_front_has_geometry and not processed_has_geometry:
        return "known_model_but_no_size_bbox"

    if not resolved_model_id and (raw_front_has_geometry or processed_has_geometry):
        return "has_geometry_source_but_parse_failed"

    if not resolved_model_id:
        return "no_model_no_size_bbox"

    return "unknown_zero_size_reason"


def build_one_object_v1(
    obj: Dict[str, Any],
    prepared_meta: Optional[Dict[str, Any]],
    matched_model_id: Optional[str],
    raw_front_entry: Optional[Dict[str, Any]],
    raw_front_source_kind: Optional[str],
    source_category_label_index: Dict[str, str],
    future_root: Path,
    assume_half_extents: bool,
    index: int,
) -> Dict[str, Any]:
    name, category, super_category, naming_source = build_name_category(
        obj=obj,
        prepared_meta=prepared_meta,
        raw_front_entry=raw_front_entry,
        source_category_label_index=source_category_label_index,
        index=index,
    )

    raw_front_model_id = extract_model_id_from_raw_front_entry(raw_front_entry)
    raw_jid = as_str(obj.get("jid")).strip()

    model_candidates: List[str] = []
    if matched_model_id:
        model_candidates.append(matched_model_id)
    if raw_front_model_id:
        model_candidates.append(raw_front_model_id)
    if raw_jid:
        model_candidates.append(raw_jid)
    model_candidates = unique_preserve_order([x for x in model_candidates if x])

    mesh_path = None
    tex_dirs: List[str] = []
    mesh_model_id = None
    mesh_size_xyz = None

    for candidate in model_candidates:
        candidate_mesh_path, candidate_tex_dirs = mesh_path_for_model_id(future_root, candidate)
        if candidate_mesh_path:
            mesh_path = candidate_mesh_path
            tex_dirs = candidate_tex_dirs
            mesh_model_id = candidate
            mesh_size_xyz = load_obj_size_full_xyz(Path(candidate_mesh_path))
            if mesh_size_xyz is not None:
                break

    size_m_tuple, size_source = resolve_object_size_and_source(
        obj=obj,
        prepared_meta=prepared_meta,
        assume_half_extents=assume_half_extents,
        mesh_size_xyz=mesh_size_xyz,
        raw_front_entry=raw_front_entry,
    )
    size_m = round_size_triplet(size_m_tuple)

    prepared_has_size = bool(
        isinstance(prepared_meta, dict)
        and all(k in prepared_meta for k in ("size_x", "size_y", "size_z"))
    )
    prepared_recognized = prepared_meta is not None

    resolved_model_id = mesh_model_id or raw_front_model_id or matched_model_id or raw_jid or None

    if prepared_has_size:
        recognition_reason = "matched_prepared_model_info_with_size"
    elif mesh_size_xyz is not None:
        recognition_reason = "mesh_found"
    elif raw_front_entry is not None:
        recognition_reason = "resolved_from_raw_front"
    elif resolved_model_id:
        recognition_reason = "model_id_known_but_no_size"
    else:
        recognition_reason = "model_not_resolved"

    zero_size_reason = None
    if not has_nonzero_size(size_m):
        zero_size_reason = classify_zero_size_reason(
            resolved_model_id=resolved_model_id,
            mesh_path=mesh_path,
            raw_front_entry=raw_front_entry,
            obj=obj,
        )

    return {
        "id": object_id_from_raw(obj, index=index),
        "name": name,
        "category": category,
        "size_m": list(size_m),
        "size_min_m": list(size_m),
        "size_max_m": list(size_m),
        "color": [0.7, 0.7, 0.7],
        "constraints": infer_constraints(category),
        "asset": {
            "source": "3dfuture_prepared" if prepared_recognized else "3dfront_processed_unrecognized",
            "model_id": resolved_model_id,
            "mesh_path": mesh_path,
            "mesh_fit_mode": "uniform",
            "mesh_texture_dirs": tex_dirs,
        },
        "recognition": {
            "prepared_recognized": prepared_recognized,
            "prepared_has_size": prepared_has_size,
            "reason": recognition_reason,
            "size_source": size_source,
            "naming_source": naming_source,
            "zero_size_reason": zero_size_reason,
        },
        "meta": {
            "super_category": super_category or None,
            "style": prepared_meta.get("style") if isinstance(prepared_meta, dict) else None,
            "theme": prepared_meta.get("theme") if isinstance(prepared_meta, dict) else None,
            "material": prepared_meta.get("material") if isinstance(prepared_meta, dict) else None,
            "prepared_category": prepared_meta.get("category") if isinstance(prepared_meta, dict) else None,
            "prepared_super_category": (
                prepared_meta.get("super-category") or prepared_meta.get("super_category")
                if isinstance(prepared_meta, dict)
                else None
            ),
            "matched_model_id": matched_model_id,
            "raw_front_model_id": raw_front_model_id,
            "mesh_model_id": mesh_model_id,
            "raw_front_source_kind": raw_front_source_kind,
            "raw_front_entry": deepcopy(raw_front_entry),
            "raw_instanceid": obj.get("instanceid"),
            "raw_uid": obj.get("uid"),
            "raw_id": obj.get("id"),
            "raw_ref": obj.get("ref"),
            "raw_category": obj.get("category"),
            "raw_yaw_deg": obj.get("yaw_deg"),
            "raw_scale": deepcopy(obj.get("scale")),
            "raw_valid": obj.get("valid", True),
            "raw_object": deepcopy(obj),
        },
    }


def build_objects_v1(
    room_obj: Dict[str, Any],
    prepared_index: Dict[str, Dict[str, Any]],
    raw_front_scene: Optional[Dict[str, Any]],
    source_category_label_index: Dict[str, str],
    future_root: Path,
    assume_half_extents: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    src_objects = room_obj.get("objects")
    if not isinstance(src_objects, list):
        raise RuntimeError("У комнаты нет objects[]")

    objects_out: List[Dict[str, Any]] = []

    stats: Dict[str, Any] = {
        "total_raw_objects": 0,
        "exported_objects": 0,
        "prepared_recognized_objects": 0,
        "prepared_has_size_objects": 0,
        "objects_sized_from_mesh": 0,
        "objects_sized_from_raw_front": 0,
        "objects_sized_from_raw": 0,
        "objects_named_from_raw_front": 0,
        "objects_named_from_source_category_id": 0,
        "objects_named_from_prepared": 0,
        "objects_named_from_processed": 0,
        "objects_named_from_fallback": 0,
        "unresolved_objects": 0,
        "zero_size_known_model_no_bbox": 0,
        "zero_size_no_model_no_bbox": 0,
        "zero_size_mesh_exists_but_bbox_unavailable": 0,
        "zero_size_parse_failed_with_geometry_source": 0,
        "invalid_flagged_objects": 0,
        "objects_with_zero_size": 0,
        "malformed_objects_skipped": 0,
        "size_source_counts": {},
        "naming_source_counts": {},
        "zero_size_reason_counts": {},
        "examples_unresolved": [],
    }

    def bump(counter: Dict[str, Any], key: str) -> None:
        counter[key] = counter.get(key, 0) + 1

    for obj in src_objects:
        stats["total_raw_objects"] += 1

        if not isinstance(obj, dict):
            stats["malformed_objects_skipped"] += 1
            continue

        prepared_meta, matched_model_id = resolve_prepared_meta_for_obj(obj, prepared_index)
        raw_front_entry, raw_front_source_kind = find_raw_front_entry_for_processed_obj(raw_front_scene, obj)

        item = build_one_object_v1(
            obj=obj,
            prepared_meta=prepared_meta,
            matched_model_id=matched_model_id,
            raw_front_entry=raw_front_entry,
            raw_front_source_kind=raw_front_source_kind,
            source_category_label_index=source_category_label_index,
            future_root=future_root,
            assume_half_extents=assume_half_extents,
            index=len(objects_out),
        )
        objects_out.append(item)
        stats["exported_objects"] += 1

        if item["recognition"]["prepared_recognized"]:
            stats["prepared_recognized_objects"] += 1
        if item["recognition"]["prepared_has_size"]:
            stats["prepared_has_size_objects"] += 1

        source = as_str(item["recognition"]["size_source"])
        bump(stats["size_source_counts"], source)

        naming_source = as_str(item["recognition"]["naming_source"])
        bump(stats["naming_source_counts"], naming_source)
        if naming_source.startswith("raw_front."):
            stats["objects_named_from_raw_front"] += 1
        elif naming_source == "sourceCategoryId.label":
            stats["objects_named_from_source_category_id"] += 1
        elif naming_source.startswith("prepared."):
            stats["objects_named_from_prepared"] += 1
        elif naming_source.startswith("processed."):
            stats["objects_named_from_processed"] += 1
        elif naming_source == "fallback":
            stats["objects_named_from_fallback"] += 1

        if source == "mesh_obj_bbox+scale":
            stats["objects_sized_from_mesh"] += 1
        elif source.startswith("raw_front:"):
            stats["objects_sized_from_raw_front"] += 1
        elif source.startswith("raw:"):
            stats["objects_sized_from_raw"] += 1
        elif source == "missing":
            stats["unresolved_objects"] += 1
            if len(stats["examples_unresolved"]) < 20:
                stats["examples_unresolved"].append({
                    "id": item["id"],
                    "model_id": item["asset"]["model_id"],
                    "name": item["name"],
                    "category": item["category"],
                    "reason": item["recognition"]["reason"],
                    "raw_ref": item["meta"]["raw_ref"],
                    "zero_size_reason": item["recognition"]["zero_size_reason"],
                })

        if not bool(obj.get("valid", True)):
            stats["invalid_flagged_objects"] += 1

        size_m = item.get("size_m") or [0.0, 0.0, 0.0]
        if not has_nonzero_size(size_m):
            stats["objects_with_zero_size"] += 1
            zero_reason = as_str(item["recognition"].get("zero_size_reason"), "unknown_zero_size_reason")
            bump(stats["zero_size_reason_counts"], zero_reason)
            if zero_reason == "known_model_but_no_mesh_and_no_bbox":
                stats["zero_size_known_model_no_bbox"] += 1
            elif zero_reason == "no_model_no_size_bbox":
                stats["zero_size_no_model_no_bbox"] += 1
            elif zero_reason == "mesh_exists_but_obj_bbox_unavailable":
                stats["zero_size_mesh_exists_but_bbox_unavailable"] += 1
            elif zero_reason == "has_geometry_source_but_parse_failed":
                stats["zero_size_parse_failed_with_geometry_source"] += 1

    out = {
        "schema": "objects.v1",
        "seed": 0,
        "objects": objects_out,
        "meta": {
            "source": "3D-FRONT-processed",
            "export_policy": "export_all_objects_keep_unrecognized",
        },
    }
    return out, stats


# ============================================================
# GT scene.v1
# ============================================================

def build_scene_gt_v1(
    root: Dict[str, Any],
    room_obj: Dict[str, Any],
    room_json: Dict[str, Any],
    prepared_index: Dict[str, Dict[str, Any]],
    raw_front_scene: Optional[Dict[str, Any]],
    source_category_label_index: Dict[str, str],
    future_root: Path,
    assume_half_extents: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    shift_x = as_float(room_json["notes"]["shift_x_m"])
    shift_z = as_float(room_json["notes"]["shift_z_m"])

    placements: List[Dict[str, Any]] = []
    src_objects = room_obj.get("objects") or []

    stats: Dict[str, Any] = {
        "total_source_objects": 0,
        "placements_exported": 0,
        "placements_skipped_no_pos": 0,
        "placements_prepared_has_size": 0,
        "placements_sized_from_mesh": 0,
        "placements_sized_from_raw_front": 0,
        "placements_sized_from_raw": 0,
        "placements_unresolved": 0,
        "placements_zero_size_known_model_no_bbox": 0,
        "placements_zero_size_no_model_no_bbox": 0,
    }

    for obj in src_objects:
        stats["total_source_objects"] += 1

        if not isinstance(obj, dict):
            continue

        pos = obj.get("pos")
        if not isinstance(pos, dict):
            stats["placements_skipped_no_pos"] += 1
            continue

        prepared_meta, matched_model_id = resolve_prepared_meta_for_obj(obj, prepared_index)
        raw_front_entry, raw_front_source_kind = find_raw_front_entry_for_processed_obj(raw_front_scene, obj)

        base_obj = build_one_object_v1(
            obj=obj,
            prepared_meta=prepared_meta,
            matched_model_id=matched_model_id,
            raw_front_entry=raw_front_entry,
            raw_front_source_kind=raw_front_source_kind,
            source_category_label_index=source_category_label_index,
            future_root=future_root,
            assume_half_extents=assume_half_extents,
            index=len(placements),
        )

        px = as_float(pos.get("x"), 0.0) + shift_x
        py = as_float(pos.get("z"), 0.0) + shift_z
        pz = as_float(pos.get("y"), 0.0)

        size_m = deepcopy(base_obj["size_m"])
        rotation_raw = as_float(obj.get("yaw_deg"), 0.0)
        rotation_deg = quantize_rot_0_90_180_270(rotation_raw)

        aabb = {
            "x_min": round6(px - size_m[0] / 2.0),
            "x_max": round6(px + size_m[0] / 2.0),
            "y_min": round6(py - size_m[1] / 2.0),
            "y_max": round6(py + size_m[1] / 2.0),
            "z_min": round6(pz - size_m[2] / 2.0),
            "z_max": round6(pz + size_m[2] / 2.0),
        }

        placement = {
            "id": base_obj["id"],
            "name": base_obj["name"],
            "category": base_obj["category"],
            "position_m": [round6(px), round6(py), round6(pz)],
            "size_m": size_m,
            "rotation_deg": rotation_deg,
            "yaw_deg": rotation_deg,
            "yaw_rad": round6(degrees_to_radians(rotation_deg)),
            "aabb": aabb,
            "mount_type": base_obj["constraints"].get("mount_type"),
            "constraints": deepcopy(base_obj["constraints"]),
            "asset": deepcopy(base_obj["asset"]),
            "recognition": deepcopy(base_obj["recognition"]),
            "source": {
                "placement_source": "3dfront_processed_gt"
            },
            "meta": deepcopy(base_obj["meta"]),
            "color": deepcopy(base_obj["color"]),
        }

        placements.append(placement)
        stats["placements_exported"] += 1

        src = as_str(base_obj["recognition"]["size_source"])
        if base_obj["recognition"]["prepared_has_size"]:
            stats["placements_prepared_has_size"] += 1
        elif src == "mesh_obj_bbox+scale":
            stats["placements_sized_from_mesh"] += 1
        elif src.startswith("raw_front:"):
            stats["placements_sized_from_raw_front"] += 1
        elif src.startswith("raw:"):
            stats["placements_sized_from_raw"] += 1
        elif src == "missing":
            stats["placements_unresolved"] += 1

        if not has_nonzero_size(size_m):
            zr = as_str(base_obj["recognition"].get("zero_size_reason"))
            if zr == "known_model_but_no_mesh_and_no_bbox":
                stats["placements_zero_size_known_model_no_bbox"] += 1
            elif zr == "no_model_no_size_bbox":
                stats["placements_zero_size_no_model_no_bbox"] += 1

    scene = {
        "schema": "scene.v1",
        "room": deepcopy(room_json),
        "placements": placements,
        "meta": {
            "placer": "3dfront_processed_gt",
            "mode": None,
            "source_uid": root.get("uid"),
            "source_room_id": room_obj.get("id"),
            "source_room_type": room_obj.get("type"),
            "export_policy": "export_all_objects_with_positions_keep_unrecognized",
        },
    }
    return scene, stats


# ============================================================
# Выбор файлов / комнат
# ============================================================

def iterate_input_files(input_path: Path) -> List[Path]:
    p = input_path.expanduser().resolve()
    if p.is_file():
        return [p]
    if p.is_dir():
        files = sorted(x for x in p.rglob("*.json") if x.is_file())
        if not files:
            raise RuntimeError(f"Во входной папке нет *.json: {p}")
        return files
    raise RuntimeError(f"Путь не найден: {p}")


def choose_rooms_from_root(
    root: Dict[str, Any],
    room_id: Optional[str],
    room_index: Optional[int],
) -> List[Tuple[int, Dict[str, Any]]]:
    rooms = root.get("rooms")
    if not isinstance(rooms, list) or not rooms:
        raise RuntimeError("Во входном JSON нет rooms[]")

    if room_id:
        for idx, room in enumerate(rooms):
            if isinstance(room, dict) and as_str(room.get("id")) == room_id:
                return [(idx, room)]
        raise RuntimeError(f"Комната room_id={room_id} не найдена")

    if room_index is not None:
        if room_index < 0 or room_index >= len(rooms):
            raise RuntimeError(f"room_index={room_index} вне диапазона [0, {len(rooms)-1}]")
        room = rooms[room_index]
        if not isinstance(room, dict):
            raise RuntimeError("Выбранный элемент rooms[] не является объектом")
        return [(room_index, room)]

    out: List[Tuple[int, Dict[str, Any]]] = []
    for idx, room in enumerate(rooms):
        if isinstance(room, dict):
            out.append((idx, room))
    if not out:
        raise RuntimeError("Во входном JSON нет корректных объектов комнат")
    return out


def build_room_output_dir(
    out_root: Path,
    input_file: Path,
    input_root: Path,
    root: Dict[str, Any],
    room_obj: Dict[str, Any],
    room_idx: int,
) -> Path:
    rel_parent = input_file.parent.resolve()
    try:
        rel_parent = rel_parent.relative_to(input_root.resolve())
    except Exception:
        rel_parent = Path()

    source_uid = normalize_name_for_fs(as_str(root.get("uid"), input_file.stem))
    room_id = normalize_name_for_fs(as_str(room_obj.get("id"), f"room_{room_idx}"))
    room_type = normalize_name_for_fs(as_str(room_obj.get("type"), "unknown"))
    room_folder = f"{source_uid}__{room_id}__{room_type}"

    return out_root / rel_parent / room_folder


# ============================================================
# Обработка
# ============================================================

def process_one_room(
    *,
    input_file: Path,
    input_root: Path,
    out_root: Path,
    root: Dict[str, Any],
    room_idx: int,
    room_obj: Dict[str, Any],
    prepared_index: Dict[str, Dict[str, Any]],
    raw_front_scene: Optional[Dict[str, Any]],
    source_category_label_index: Dict[str, str],
    future_root: Path,
    assume_half_extents: bool,
    default_ceiling_height: float,
    write_scene_gt: bool,
) -> Dict[str, Any]:
    out_dir = ensure_dir(
        build_room_output_dir(
            out_root=out_root,
            input_file=input_file,
            input_root=input_root,
            root=root,
            room_obj=room_obj,
            room_idx=room_idx,
        )
    )

    room_json = build_room_json(
        root=root,
        room_obj=room_obj,
        prepared_index=prepared_index,
        raw_front_scene=raw_front_scene,
        future_root=future_root,
        assume_half_extents=assume_half_extents,
        default_ceiling_height=default_ceiling_height,
    )

    objects_v1, objects_stats = build_objects_v1(
        room_obj=room_obj,
        prepared_index=prepared_index,
        raw_front_scene=raw_front_scene,
        source_category_label_index=source_category_label_index,
        future_root=future_root,
        assume_half_extents=assume_half_extents,
    )

    room_out = out_dir / "room.json"
    objects_out = out_dir / "objects.v1.json"

    save_json(room_out, room_json)
    save_json(objects_out, objects_v1)

    report: Dict[str, Any] = {
        "status": "ok",
        "input_file": str(input_file.resolve()),
        "input_file_relative": safe_relpath(input_file, input_root),
        "future_root": str(future_root.resolve()),
        "selected_room_id": room_obj.get("id"),
        "selected_room_index": room_idx,
        "selected_room_type_raw": room_obj.get("type"),
        "selected_room_type_norm": room_json.get("room_type"),
        "prepared_sizes_mode": "half_extents" if assume_half_extents else "full_extents",
        "raw_front_scene_found": raw_front_scene is not None,
        "source_category_label_index_size": len(source_category_label_index),
        "stats": {
            "objects": objects_stats,
        },
        "outputs": {
            "room_json": str(room_out.resolve()),
            "objects_v1_json": str(objects_out.resolve()),
        },
    }

    if write_scene_gt:
        scene_gt, scene_stats = build_scene_gt_v1(
            root=root,
            room_obj=room_obj,
            room_json=room_json,
            prepared_index=prepared_index,
            raw_front_scene=raw_front_scene,
            source_category_label_index=source_category_label_index,
            future_root=future_root,
            assume_half_extents=assume_half_extents,
        )
        scene_out = out_dir / "scene_gt.v1.json"
        save_json(scene_out, scene_gt)
        report["outputs"]["scene_gt_v1_json"] = str(scene_out.resolve())
        report["stats"]["scene_gt"] = scene_stats

    report_out = out_dir / "conversion_report.json"
    save_json(report_out, report)

    print(
        f"[OK] {safe_relpath(input_file, input_root)} "
        f"room_id={room_obj.get('id')} -> {safe_relpath(out_dir, out_root)}"
    )
    return report


def accumulate_global_stats(global_stats: Dict[str, Any], report: Dict[str, Any]) -> None:
    global_stats["processed_rooms"] += 1

    obj_stats = report["stats"]["objects"]
    global_stats["total_raw_objects"] += int(obj_stats.get("total_raw_objects", 0))
    global_stats["exported_objects"] += int(obj_stats.get("exported_objects", 0))
    global_stats["prepared_recognized_objects"] += int(obj_stats.get("prepared_recognized_objects", 0))
    global_stats["prepared_has_size_objects"] += int(obj_stats.get("prepared_has_size_objects", 0))
    global_stats["objects_sized_from_mesh"] += int(obj_stats.get("objects_sized_from_mesh", 0))
    global_stats["objects_sized_from_raw_front"] += int(obj_stats.get("objects_sized_from_raw_front", 0))
    global_stats["objects_sized_from_raw"] += int(obj_stats.get("objects_sized_from_raw", 0))
    global_stats["objects_named_from_raw_front"] += int(obj_stats.get("objects_named_from_raw_front", 0))
    global_stats["objects_named_from_source_category_id"] += int(obj_stats.get("objects_named_from_source_category_id", 0))
    global_stats["objects_named_from_prepared"] += int(obj_stats.get("objects_named_from_prepared", 0))
    global_stats["objects_named_from_processed"] += int(obj_stats.get("objects_named_from_processed", 0))
    global_stats["objects_named_from_fallback"] += int(obj_stats.get("objects_named_from_fallback", 0))
    global_stats["unresolved_objects"] += int(obj_stats.get("unresolved_objects", 0))
    global_stats["zero_size_known_model_no_bbox"] += int(obj_stats.get("zero_size_known_model_no_bbox", 0))
    global_stats["zero_size_no_model_no_bbox"] += int(obj_stats.get("zero_size_no_model_no_bbox", 0))
    global_stats["zero_size_mesh_exists_but_bbox_unavailable"] += int(obj_stats.get("zero_size_mesh_exists_but_bbox_unavailable", 0))
    global_stats["zero_size_parse_failed_with_geometry_source"] += int(obj_stats.get("zero_size_parse_failed_with_geometry_source", 0))
    global_stats["invalid_flagged_objects"] += int(obj_stats.get("invalid_flagged_objects", 0))
    global_stats["objects_with_zero_size"] += int(obj_stats.get("objects_with_zero_size", 0))
    global_stats["malformed_objects_skipped"] += int(obj_stats.get("malformed_objects_skipped", 0))

    scene_stats = report["stats"].get("scene_gt")
    if isinstance(scene_stats, dict):
        global_stats["scene_gt_exported"] += int(scene_stats.get("placements_exported", 0))
        global_stats["scene_gt_prepared_has_size"] += int(scene_stats.get("placements_prepared_has_size", 0))
        global_stats["scene_gt_sized_from_mesh"] += int(scene_stats.get("placements_sized_from_mesh", 0))
        global_stats["scene_gt_sized_from_raw_front"] += int(scene_stats.get("placements_sized_from_raw_front", 0))
        global_stats["scene_gt_sized_from_raw"] += int(scene_stats.get("placements_sized_from_raw", 0))
        global_stats["scene_gt_unresolved"] += int(scene_stats.get("placements_unresolved", 0))
        global_stats["scene_gt_zero_size_known_model_no_bbox"] += int(scene_stats.get("placements_zero_size_known_model_no_bbox", 0))
        global_stats["scene_gt_zero_size_no_model_no_bbox"] += int(scene_stats.get("placements_zero_size_no_model_no_bbox", 0))
        global_stats["scene_gt_skipped_no_pos"] += int(scene_stats.get("placements_skipped_no_pos", 0))


def process_input(
    *,
    input_path: Path,
    front_root: Path,
    prepared_info_path: Path,
    future_root: Path,
    out_dir: Path,
    source_category_label_index: Dict[str, str],
    room_id: Optional[str],
    room_index: Optional[int],
    assume_half_extents: bool,
    default_ceiling_height: float,
    write_scene_gt: bool,
    strict_room_polygon: bool,
) -> Dict[str, Any]:
    prepared_raw = load_json(prepared_info_path)
    prepared_index = build_prepared_index(prepared_raw)
    if not prepared_index:
        raise RuntimeError("Не удалось построить индекс prepared_model_info.json")

    input_path = input_path.expanduser().resolve()
    front_root = front_root.expanduser().resolve()
    input_root = input_path if input_path.is_dir() else input_path.parent
    out_root = ensure_dir(out_dir)

    files = iterate_input_files(input_path)

    manifest: Dict[str, Any] = {
        "input": str(input_path),
        "front_root": str(front_root),
        "input_mode": "directory" if input_path.is_dir() else "file",
        "prepared_info": str(prepared_info_path.resolve()),
        "future_root": str(future_root.resolve()),
        "out_dir": str(out_root.resolve()),
        "prepared_index_size": len(prepared_index),
        "source_category_label_index_size": len(source_category_label_index),
        "prepared_sizes_mode": "half_extents" if assume_half_extents else "full_extents",
        "write_scene_gt": bool(write_scene_gt),
        "strict_room_polygon": bool(strict_room_polygon),
        "reports": [],
        "skipped_rooms": [],
        "errors": [],
        "global_stats": {
            "processed_files": 0,
            "processed_rooms": 0,
            "skipped_rooms_no_polygon": 0,
            "total_raw_objects": 0,
            "exported_objects": 0,
            "prepared_recognized_objects": 0,
            "prepared_has_size_objects": 0,
            "objects_sized_from_mesh": 0,
            "objects_sized_from_raw_front": 0,
            "objects_sized_from_raw": 0,
            "objects_named_from_raw_front": 0,
            "objects_named_from_source_category_id": 0,
            "objects_named_from_prepared": 0,
            "objects_named_from_processed": 0,
            "objects_named_from_fallback": 0,
            "unresolved_objects": 0,
            "zero_size_known_model_no_bbox": 0,
            "zero_size_no_model_no_bbox": 0,
            "zero_size_mesh_exists_but_bbox_unavailable": 0,
            "zero_size_parse_failed_with_geometry_source": 0,
            "invalid_flagged_objects": 0,
            "objects_with_zero_size": 0,
            "malformed_objects_skipped": 0,
            "scene_gt_exported": 0,
            "scene_gt_prepared_has_size": 0,
            "scene_gt_sized_from_mesh": 0,
            "scene_gt_sized_from_raw_front": 0,
            "scene_gt_sized_from_raw": 0,
            "scene_gt_unresolved": 0,
            "scene_gt_zero_size_known_model_no_bbox": 0,
            "scene_gt_zero_size_no_model_no_bbox": 0,
            "scene_gt_skipped_no_pos": 0,
            "failed_files": 0,
            "failed_rooms": 0,
            "raw_front_scene_found_for_rooms": 0,
            "raw_front_scene_missing_for_rooms": 0,
        },
    }

    for input_file in files:
        try:
            root = load_json(input_file)
            room_pairs = choose_rooms_from_root(
                root=root,
                room_id=room_id if input_path.is_file() else None,
                room_index=room_index if input_path.is_file() else None,
            )
            manifest["global_stats"]["processed_files"] += 1
        except Exception as exc:
            manifest["global_stats"]["failed_files"] += 1
            manifest["errors"].append({
                "stage": "file_open_or_room_select",
                "input_file": str(input_file.resolve()),
                "error": str(exc),
            })
            print(f"[ERR] {input_file}: {exc}")
            continue

        source_uid = as_str(root.get("uid")).strip()
        raw_front_scene = load_raw_front_scene_by_uid(front_root, source_uid)

        for room_idx, room_obj in room_pairs:
            if raw_front_scene is not None:
                manifest["global_stats"]["raw_front_scene_found_for_rooms"] += 1
            else:
                manifest["global_stats"]["raw_front_scene_missing_for_rooms"] += 1

            if not room_has_valid_polygon(room_obj):
                rec = {
                    "input_file": str(input_file.resolve()),
                    "room_index": room_idx,
                    "room_id": room_obj.get("id") if isinstance(room_obj, dict) else None,
                    "reason": "missing_or_invalid_polygon",
                }
                if strict_room_polygon:
                    manifest["global_stats"]["failed_rooms"] += 1
                    manifest["errors"].append({
                        "stage": "room_process",
                        **rec,
                    })
                    print(f"[ERR] {input_file} room_id={room_obj.get('id')}: У комнаты отсутствует корректный polygon")
                else:
                    manifest["global_stats"]["skipped_rooms_no_polygon"] += 1
                    manifest["skipped_rooms"].append(rec)
                    print(f"[SKIP] {safe_relpath(input_file, input_root)} room_id={room_obj.get('id')}: invalid polygon")
                continue

            try:
                report = process_one_room(
                    input_file=input_file,
                    input_root=input_root,
                    out_root=out_root,
                    root=root,
                    room_idx=room_idx,
                    room_obj=room_obj,
                    prepared_index=prepared_index,
                    raw_front_scene=raw_front_scene,
                    source_category_label_index=source_category_label_index,
                    future_root=future_root,
                    assume_half_extents=assume_half_extents,
                    default_ceiling_height=default_ceiling_height,
                    write_scene_gt=write_scene_gt,
                )
                manifest["reports"].append(report)
                accumulate_global_stats(manifest["global_stats"], report)
            except Exception as exc:
                manifest["global_stats"]["failed_rooms"] += 1
                manifest["errors"].append({
                    "stage": "room_process",
                    "input_file": str(input_file.resolve()),
                    "room_index": room_idx,
                    "room_id": room_obj.get("id") if isinstance(room_obj, dict) else None,
                    "error": str(exc),
                })
                print(f"[ERR] {input_file} room_id={room_obj.get('id')}: {exc}")

    manifest_path = out_root / "manifest.json"
    save_json(manifest_path, manifest)

    print()
    print(f"OK: manifest -> {manifest_path}")
    print(json.dumps(manifest["global_stats"], ensure_ascii=False, indent=2))
    if manifest["errors"]:
        print(f"WARNING: errors={len(manifest['errors'])}")
    if manifest["skipped_rooms"]:
        print(f"INFO: skipped_rooms={len(manifest['skipped_rooms'])}")

    return manifest


# ============================================================
# CLI
# ============================================================

def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Конвертер 3D-FRONT-processed -> room.json + objects.v1.json "
            "для всех комнат во входном файле или папке"
        )
    )

    p.add_argument(
        "--input",
        required=True,
        help="Путь к 3D-FRONT-processed/*.json ИЛИ к папке с такими файлами",
    )
    p.add_argument(
        "--front-root",
        required=True,
        help="Папка с полными JSON 3D-FRONT, например data/sourse/3D-FRONT/3D-FRONT",
    )
    p.add_argument("--prepared-info", required=True, help="Путь к prepared_model_info.json")
    p.add_argument("--future-root", required=True, help="Папка 3D-FUTURE-model")
    p.add_argument(
        "--source-category-label-index",
        default=None,
        help="Путь к source_category_id_to_label.json",
    )
    p.add_argument("--out-dir", required=True, help="Куда сохранить подготовленные папки комнат")

    g = p.add_mutually_exclusive_group()
    g.add_argument("--room-id", default=None, help="ID комнаты внутри rooms[] (только для одиночного файла)")
    g.add_argument("--room-index", type=int, default=None, help="Индекс комнаты внутри rooms[] (только для одиночного файла)")

    p.add_argument(
        "--prepared-sizes",
        choices=["half_extents", "full_extents"],
        default="half_extents",
        help="Как трактовать size_x/size_y/size_z в prepared_model_info.json",
    )

    p.add_argument(
        "--default-ceiling-height",
        type=float,
        default=2.8,
        help="Потолок по умолчанию, если его нельзя оценить по объектам",
    )

    p.add_argument(
        "--write-scene-gt",
        action="store_true",
        help="Дополнительно записать scene_gt.v1.json для каждой комнаты",
    )

    p.add_argument(
        "--strict-room-polygon",
        action="store_true",
        help="Считать комнаты без polygon ошибкой, а не пропуском",
    )

    return p


def main() -> None:
    args = build_cli().parse_args()

    input_path = Path(args.input).expanduser().resolve()
    front_root = Path(args.front_root).expanduser().resolve()
    prepared_info_path = Path(args.prepared_info).expanduser().resolve()
    future_root = Path(args.future_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    source_category_label_index_path = (
        Path(args.source_category_label_index).expanduser().resolve()
        if args.source_category_label_index
        else None
    )
    source_category_label_index = load_source_category_label_index(source_category_label_index_path)

    process_input(
        input_path=input_path,
        front_root=front_root,
        prepared_info_path=prepared_info_path,
        future_root=future_root,
        out_dir=out_dir,
        source_category_label_index=source_category_label_index,
        room_id=args.room_id,
        room_index=args.room_index,
        assume_half_extents=(args.prepared_sizes == "half_extents"),
        default_ceiling_height=float(args.default_ceiling_height),
        write_scene_gt=bool(args.write_scene_gt),
        strict_room_polygon=bool(args.strict_room_polygon),
    )


if __name__ == "__main__":
    main()