#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
src/Plasement/retrieval_knn_scene.py

Retrieval-based placer -> scene.v1

Что делает:
1. Читает предобработанные комнаты из 3D-FRONT-processed-mini.
2. Ищет ближайшие комнаты по:
   - типу комнаты,
   - размерам/площади,
   - набору категорий мебели.
3. Переносит layout из 1-NN / top-k NN в новую комнату.
4. Восстанавливает asset metadata из prepared_model_info.json:
   - size_m
   - model_id
   - mesh_path
   - mesh_texture_dirs
   - style/theme/material
5. Если source_asset_id отсутствует локально в 3D-FUTURE-model,
   автоматически подбирает похожую доступную модель:
   - сначала по category,
   - затем по super_category,
   - с учётом близости размеров и совпадения style/theme/material.
6. После сборки placements выполняет универсальный post-process:
   - делит напольные объекты на левую/правую половины комнаты
   - растягивает группы к левой/правой стенам
   - затем делит на нижнюю/верхнюю половины
   - растягивает группы к нижней/верхней стенам
   Так сохраняется группировка, но освобождается центр.
7. Собирает сразу scene.v1, совместимый с BlenderVisualizePlacement.py.

Пример запуска:
python3 src/Plasement/retrieval_knn_scene.py \
  --dataset-root data/sourse/3D-FRONT/3D-FRONT-processed-mini \
  --target-room data/input/room.json \
  --items "King-size Bed,Nightstand,Nightstand,Wardrobe,Dressing Table,Dining Chair,Pendant Lamp" \
  --room-type bedroom \
  --top-k 3 \
  --prepared-info data/sourse/3D-FRONT/prepared_model_info.json \
  --future-root data/sourse/3D-FRONT/3D-FUTURE-model \
  --out out/retrieval_scene.v1.json \
  --dump-retrieval out/retrieval_layout.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


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
# Small utils
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


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def statistics_median(vals: list[float]) -> float:
    vals = sorted(vals)
    n = len(vals)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return vals[n // 2]
    return 0.5 * (vals[n // 2 - 1] + vals[n // 2])


def quantize_rot_0_90_180_270(deg: float) -> int:
    a = float(deg) % 360.0
    allowed = (0.0, 90.0, 180.0, 270.0)
    best = min(allowed, key=lambda t: (abs(((a - t + 180.0) % 360.0) - 180.0), t))
    return int(best)


def build_aabb_from_center_size(position_m: list[float], size_m: list[float]) -> dict[str, float]:
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


def polygon_area(poly: list[tuple[float, float]]) -> float:
    if len(poly) < 3:
        return 0.0
    s = 0.0
    for i in range(len(poly)):
        x1, z1 = poly[i]
        x2, z2 = poly[(i + 1) % len(poly)]
        s += x1 * z2 - x2 * z1
    return abs(s) * 0.5


def bbox_of_poly(poly: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in poly]
    zs = [p[1] for p in poly]
    return min(xs), max(xs), min(zs), max(zs)


def point_in_polygon(x: float, z: float, poly: list[tuple[float, float]]) -> bool:
    inside = False
    n = len(poly)
    if n < 3:
        return False
    for i in range(n):
        x1, z1 = poly[i]
        x2, z2 = poly[(i + 1) % n]
        cond = ((z1 > z) != (z2 > z))
        if cond:
            xinters = (x2 - x1) * (z - z1) / (z2 - z1 + 1e-12) + x1
            if x < xinters:
                inside = not inside
    return inside


def looks_like_ceiling_light(category: str) -> bool:
    s = (category or "").lower()
    keys = [
        "lamp",
        "light",
        "ceiling",
        "pendant",
        "chandelier",
        "suspension",
        "plafon",
        "люстр",
        "светиль",
        "ламп",
    ]
    return any(k in s for k in keys)


def normalize_text(s: str) -> str:
    return " ".join((s or "").strip().lower().replace("_", " ").replace("-", " ").split())


def prepared_record_size_to_scene_size(rec: Optional[dict[str, Any]]) -> list[float]:
    """
    prepared_model_info:
        size_x = половина размера по X
        size_y = половина размера по ВЕРТИКАЛИ
        size_z = половина размера по глубине

    scene.v1 ожидает:
        [x, y, z] = [ширина по полу, глубина по полу, высота]

    Поэтому нужно:
        [2*size_x, 2*size_z, 2*size_y]
    """
    if rec is None:
        return [0.8, 0.8, 0.8]

    sx = 2.0 * as_float(rec.get("size_x"), 0.0)
    sy = 2.0 * as_float(rec.get("size_y"), 0.0)  # vertical in prepared
    sz = 2.0 * as_float(rec.get("size_z"), 0.0)  # depth in prepared

    if sx <= 0.0 or sy <= 0.0 or sz <= 0.0:
        return [0.8, 0.8, 0.8]

    return [sx, sz, sy]


# ============================================================
# Dataset dataclasses
# ============================================================

@dataclass
class ObjRec:
    asset_id: str
    category: str
    super_category: str
    x: float
    z: float
    yaw_deg: float
    bbox_xy: Optional[list[float]]
    raw_name: str = ""


@dataclass
class RoomRec:
    file_path: str
    room_id: str
    room_type: str
    floor_poly: list[tuple[float, float]]   # (x, z)
    bbox_min_x: float
    bbox_max_x: float
    bbox_min_z: float
    bbox_max_z: float
    width: float
    depth: float
    area: float
    objects: list[ObjRec]

    @property
    def category_counter(self) -> Counter:
        return Counter(o.category for o in self.objects)


# ============================================================
# Category / asset helpers
# ============================================================

def canonical_asset_id(obj: dict[str, Any]) -> str:
    for key in ("jid", "model_id", "ref"):
        v = obj.get(key)
        if v:
            return str(v)
    return ""


def normalize_room_type(raw: str) -> str:
    s = (raw or "").strip().lower()
    if "bedroom" in s or "masterbedroom" in s or "secondbedroom" in s:
        return "bedroom"
    if "living" in s or "dining" in s or "livingdiningroom" in s:
        return "living"
    if "office" in s or "study" in s or "library" in s:
        return "office"
    return s or "unknown"


def infer_room_type_from_room_id(room_id: str) -> str:
    return normalize_room_type(room_id)


def normalize_category(cat: str) -> str:
    return (cat or "").strip()


# ============================================================
# Loading processed-mini dataset
# ============================================================

def load_roomrec_from_mini_file(path: Path) -> list[RoomRec]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rooms = data.get("rooms") or []
    result: list[RoomRec] = []

    for r in rooms:
        room_id = str(r.get("id", path.stem))
        room_type = infer_room_type_from_room_id(room_id)

        poly_raw = r.get("floor_polygon_xz") or []
        if len(poly_raw) < 3:
            continue

        floor_poly = [(float(p["x"]), float(p["z"])) for p in poly_raw]
        x0, x1, z0, z1 = bbox_of_poly(floor_poly)
        width = x1 - x0
        depth = z1 - z0
        area = polygon_area(floor_poly)

        objects: list[ObjRec] = []
        for o in r.get("objects") or []:
            category = normalize_category(o.get("category", "Unknown"))
            super_category = str(o.get("super-category", ""))
            pos = o.get("pos") or {}
            asset_id = canonical_asset_id(o)

            objects.append(
                ObjRec(
                    asset_id=asset_id,
                    category=category,
                    super_category=super_category,
                    x=float(pos.get("x", 0.0)),
                    z=float(pos.get("z", 0.0)),
                    yaw_deg=float(o.get("yaw_deg", 0.0)),
                    bbox_xy=o.get("bbox_world_xy"),
                    raw_name=str(o.get("name", "")),
                )
            )

        result.append(
            RoomRec(
                file_path=str(path.resolve()),
                room_id=room_id,
                room_type=room_type,
                floor_poly=floor_poly,
                bbox_min_x=x0,
                bbox_max_x=x1,
                bbox_min_z=z0,
                bbox_max_z=z1,
                width=width,
                depth=depth,
                area=area,
                objects=objects,
            )
        )

    return result


def load_dataset(dataset_root: Path) -> list[RoomRec]:
    roomrecs: list[RoomRec] = []
    files = sorted(dataset_root.glob("*.json"))
    if not files:
        raise RuntimeError(f"No json files found in {dataset_root}")

    for fp in files:
        try:
            roomrecs.extend(load_roomrec_from_mini_file(fp))
        except Exception:
            continue

    if not roomrecs:
        raise RuntimeError(f"No valid room records loaded from {dataset_root}")
    return roomrecs


def build_asset_to_category_index(roomrecs: list[RoomRec]) -> dict[str, Counter]:
    index: dict[str, Counter] = defaultdict(Counter)
    for room in roomrecs:
        for obj in room.objects:
            if obj.asset_id:
                index[obj.asset_id][obj.category] += 1
    return index


# ============================================================
# Target room parsing
# ============================================================

def load_target_room(path: Path, room_type_override: Optional[str] = None) -> RoomRec:
    data = json.loads(path.read_text(encoding="utf-8"))

    if "room" in data and isinstance(data["room"], dict):
        r = data["room"]
        poly_raw = r.get("floor_polygon") or []
        if len(poly_raw) < 3:
            raise RuntimeError("target room: room.floor_polygon is missing or too short")
        floor_poly = [(float(p["x"]), float(p["y"])) for p in poly_raw]
        room_id = str(r.get("id", path.stem))
        room_type = room_type_override or normalize_room_type(str(r.get("room_type", "")) or room_id)
    elif "rooms" in data and data["rooms"]:
        r = data["rooms"][0]
        poly_raw = r.get("floor_polygon_xz") or []
        if len(poly_raw) < 3:
            raise RuntimeError("target room: rooms[0].floor_polygon_xz is missing or too short")
        floor_poly = [(float(p["x"]), float(p["z"])) for p in poly_raw]
        room_id = str(r.get("id", path.stem))
        room_type = room_type_override or infer_room_type_from_room_id(room_id)
    else:
        raise RuntimeError("Unsupported target room format")

    x0, x1, z0, z1 = bbox_of_poly(floor_poly)
    return RoomRec(
        file_path=str(path.resolve()),
        room_id=room_id,
        room_type=normalize_room_type(room_type),
        floor_poly=floor_poly,
        bbox_min_x=x0,
        bbox_max_x=x1,
        bbox_min_z=z0,
        bbox_max_z=z1,
        width=x1 - x0,
        depth=z1 - z0,
        area=polygon_area(floor_poly),
        objects=[],
    )


def normalize_room_dict(room_data: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(room_data)

    if "room" in out and isinstance(out["room"], dict):
        root = deepcopy(out["room"])
        for k, v in out.items():
            if k != "room":
                root.setdefault(k, v)
        out = root

    out.setdefault("units", "m")
    out.setdefault("doors", [])
    out.setdefault("windows", [])
    out.setdefault("openings", [])

    if not out.get("walls"):
        poly = out.get("floor_polygon") or []
        if isinstance(poly, list) and len(poly) >= 3:
            walls = []
            n = len(poly)
            for i in range(n):
                walls.append({
                    "id": f"w{i}",
                    "from_vertex": i,
                    "to_vertex": (i + 1) % n,
                })
            out["walls"] = walls

    return out


# ============================================================
# Similarity
# ============================================================

def weighted_jaccard_counter(a: Counter, b: Counter) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 1.0
    num = sum(min(a[k], b[k]) for k in keys)
    den = sum(max(a[k], b[k]) for k in keys)
    return num / den if den > 0 else 0.0


def size_similarity(src: RoomRec, tgt: RoomRec) -> float:
    if src.width <= 1e-9 or src.depth <= 1e-9 or src.area <= 1e-9:
        return 0.0

    w_ratio = min(src.width, tgt.width) / max(src.width, tgt.width)
    d_ratio = min(src.depth, tgt.depth) / max(src.depth, tgt.depth)
    a_ratio = min(src.area, tgt.area) / max(src.area, tgt.area)

    src_aspect = src.width / max(src.depth, 1e-9)
    tgt_aspect = tgt.width / max(tgt.depth, 1e-9)
    asp_ratio = min(src_aspect, tgt_aspect) / max(src_aspect, tgt_aspect)

    return 0.30 * w_ratio + 0.30 * d_ratio + 0.20 * a_ratio + 0.20 * asp_ratio


def room_similarity(src: RoomRec, tgt: RoomRec, requested_counter: Counter) -> float:
    if src.room_type != tgt.room_type:
        return -1e9

    src_counter = src.category_counter
    cat_sim = weighted_jaccard_counter(src_counter, requested_counter)
    size_sim = size_similarity(src, tgt)

    n_src = sum(src_counter.values())
    n_req = sum(requested_counter.values())
    if max(n_src, n_req) > 0:
        count_sim = min(n_src, n_req) / max(n_src, n_req)
    else:
        count_sim = 1.0

    empty_penalty = -0.15 if len(src.objects) == 0 else 0.0
    return 0.60 * cat_sim + 0.25 * size_sim + 0.15 * count_sim + empty_penalty


def retrieve_topk_rooms(
    roomrecs: list[RoomRec],
    target_room: RoomRec,
    requested_counter: Counter,
    top_k: int = 3,
) -> list[tuple[float, RoomRec]]:
    scored: list[tuple[float, RoomRec]] = []
    for room in roomrecs:
        s = room_similarity(room, target_room, requested_counter)
        if s < -1e8:
            continue
        scored.append((s, room))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


def choose_reference_layout(neighbors: list[tuple[float, RoomRec]], requested_counter: Counter) -> RoomRec:
    if not neighbors:
        raise RuntimeError("No neighbors found")

    best_room = neighbors[0][1]
    best_score = -1e18

    for sim, room in neighbors:
        overlap = weighted_jaccard_counter(room.category_counter, requested_counter)
        score = 0.8 * sim + 0.2 * overlap
        if score > best_score:
            best_score = score
            best_room = room

    return best_room


# ============================================================
# Layout transfer
# ============================================================

def normalize_pos_in_room(room: RoomRec, x: float, z: float) -> tuple[float, float]:
    nx = (x - room.bbox_min_x) / max(room.width, 1e-9)
    nz = (z - room.bbox_min_z) / max(room.depth, 1e-9)
    return nx, nz


def denormalize_pos_in_room(room: RoomRec, nx: float, nz: float) -> tuple[float, float]:
    x = room.bbox_min_x + nx * room.width
    z = room.bbox_min_z + nz * room.depth
    return x, z


def category_ordered_objects(room: RoomRec) -> dict[str, list[ObjRec]]:
    d: dict[str, list[ObjRec]] = defaultdict(list)
    for obj in room.objects:
        d[obj.category].append(obj)
    for cat in d:
        d[cat].sort(key=lambda o: (o.x, o.z))
    return d


def clamp_point_to_room_bbox(room: RoomRec, x: float, z: float, margin: float = 0.15) -> tuple[float, float]:
    x = clamp(x, room.bbox_min_x + margin, room.bbox_max_x - margin)
    z = clamp(z, room.bbox_min_z + margin, room.bbox_max_z - margin)
    return x, z


def project_inside_polygon_if_needed(room: RoomRec, x: float, z: float) -> tuple[float, float]:
    if point_in_polygon(x, z, room.floor_poly):
        return x, z

    x, z = clamp_point_to_room_bbox(room, x, z)
    if point_in_polygon(x, z, room.floor_poly):
        return x, z

    cx = 0.5 * (room.bbox_min_x + room.bbox_max_x)
    cz = 0.5 * (room.bbox_min_z + room.bbox_max_z)
    for alpha in [0.9, 0.75, 0.5, 0.25, 0.1]:
        xx = alpha * x + (1.0 - alpha) * cx
        zz = alpha * z + (1.0 - alpha) * cz
        if point_in_polygon(xx, zz, room.floor_poly):
            return xx, zz

    return cx, cz


def greedy_category_matching(requested_counter: Counter, ref_room: RoomRec) -> dict[str, list[ObjRec]]:
    ref_by_cat = category_ordered_objects(ref_room)
    res: dict[str, list[ObjRec]] = {}

    for cat, req_count in requested_counter.items():
        ref_objs = ref_by_cat.get(cat, [])
        if not ref_objs:
            res[cat] = []
            continue

        if len(ref_objs) >= req_count:
            res[cat] = ref_objs[:req_count]
        else:
            repeated = []
            for i in range(req_count):
                repeated.append(ref_objs[i % len(ref_objs)])
            res[cat] = repeated

    return res


def fallback_position_for_new_object(target_room: RoomRec, placed: list[dict[str, Any]], category: str) -> tuple[float, float, float]:
    same_cat = [p for p in placed if p["category"] == category]
    if same_cat:
        cx = sum(p["x"] for p in same_cat) / len(same_cat)
        cz = sum(p["z"] for p in same_cat) / len(same_cat)
        x = cx + 0.25
        z = cz + 0.25
    else:
        x = 0.5 * (target_room.bbox_min_x + target_room.bbox_max_x)
        z = 0.5 * (target_room.bbox_min_z + target_room.bbox_max_z)

    x, z = project_inside_polygon_if_needed(target_room, x, z)
    return x, z, 0.0


def transfer_layout_from_reference(
    target_room: RoomRec,
    requested_items: list[str],
    reference_room: RoomRec,
) -> dict[str, Any]:
    requested_counter = Counter(requested_items)
    matched = greedy_category_matching(requested_counter, reference_room)

    placed: list[dict[str, Any]] = []

    for category in requested_items:
        ref_list = matched.get(category, [])
        if ref_list:
            ref_obj = ref_list.pop(0)
            nx, nz = normalize_pos_in_room(reference_room, ref_obj.x, ref_obj.z)
            x, z = denormalize_pos_in_room(target_room, nx, nz)
            x, z = project_inside_polygon_if_needed(target_room, x, z)

            placed.append({
                "category": category,
                "source_category": ref_obj.category,
                "source_room_id": reference_room.room_id,
                "source_asset_id": ref_obj.asset_id,
                "x": x,
                "z": z,
                "yaw_deg": ref_obj.yaw_deg,
                "transfer_mode": "copied_from_reference",
            })
        else:
            x, z, yaw = fallback_position_for_new_object(target_room, placed, category)
            placed.append({
                "category": category,
                "source_category": None,
                "source_room_id": reference_room.room_id,
                "source_asset_id": None,
                "x": x,
                "z": z,
                "yaw_deg": yaw,
                "transfer_mode": "fallback_added",
            })

    return {
        "schema": "retrieval_layout/v1",
        "target_room_id": target_room.room_id,
        "target_room_type": target_room.room_type,
        "reference_room_id": reference_room.room_id,
        "reference_room_file": reference_room.file_path,
        "placements": placed,
        "requested_items": requested_items,
        "requested_counter": dict(requested_counter),
    }


def neighbor_report(neighbors: list[tuple[float, RoomRec]]) -> list[dict[str, Any]]:
    rows = []
    for sim, room in neighbors:
        rows.append({
            "similarity": sim,
            "room_id": room.room_id,
            "room_type": room.room_type,
            "file_path": room.file_path,
            "width": room.width,
            "depth": room.depth,
            "area": room.area,
            "categories": dict(room.category_counter),
        })
    return rows


# ============================================================
# prepared_model_info + auto fallback to similar existing asset
# ============================================================

def load_prepared_info(prepared_info_path: Path) -> list[dict[str, Any]]:
    data = load_json(prepared_info_path)
    if isinstance(data, list):
        return data
    raise RuntimeError("prepared_model_info.json должен быть списком объектов")


def model_dir_exists(model_id: str, future_root: Path) -> bool:
    if not model_id:
        return False
    obj_path = future_root / model_id / "normalized_model.obj"
    return obj_path.exists()


def build_model_index(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for rec in records:
        model_id = as_str(rec.get("model_id")).strip()
        if model_id:
            out[model_id] = rec
    return out


def build_available_candidate_indexes(
    records: list[dict[str, Any]],
    future_root: Path,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
    dict[str, list[float]],
]:
    """
    Возвращает:
    - by_category
    - by_super_category
    - median size by category
    """
    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_super: dict[str, list[dict[str, Any]]] = defaultdict(list)
    size_acc: dict[str, list[list[float]]] = defaultdict(list)

    for rec in records:
        model_id = as_str(rec.get("model_id")).strip()
        if not model_id:
            continue
        if not model_dir_exists(model_id, future_root):
            continue

        cat = as_str(rec.get("category")).strip()
        super_cat = as_str(rec.get("super-category") or rec.get("super_category")).strip()

        size_m = prepared_record_size_to_scene_size(rec)
        sx, sy, sz = size_m
        if sx <= 0 or sy <= 0 or sz <= 0:
            continue

        rec2 = dict(rec)
        rec2["_size_m"] = size_m

        if cat:
            by_cat[cat].append(rec2)
            size_acc[cat].append([sx, sy, sz])
        if super_cat:
            by_super[super_cat].append(rec2)

    cat_median_size: dict[str, list[float]] = {}
    for cat, arrs in size_acc.items():
        xs = [a[0] for a in arrs]
        ys = [a[1] for a in arrs]
        zs = [a[2] for a in arrs]
        cat_median_size[cat] = [
            statistics_median(xs),
            statistics_median(ys),
            statistics_median(zs),
        ]

    return by_cat, by_super, cat_median_size


def size_distance_score(target_size: list[float], cand_size: list[float]) -> float:
    s = 0.0
    for a, b in zip(target_size, cand_size):
        den = max(abs(a), abs(b), 1e-6)
        s += abs(a - b) / den
    return s / 3.0


def choose_similar_existing_model(
    *,
    requested_category: str,
    requested_super_category: Optional[str],
    target_size_m: list[float],
    preferred_style: Optional[str],
    preferred_theme: Optional[str],
    preferred_material: Optional[str],
    by_category: dict[str, list[dict[str, Any]]],
    by_super: dict[str, list[dict[str, Any]]],
) -> Optional[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    if requested_category in by_category:
        candidates.extend(by_category[requested_category])

    if not candidates and requested_super_category and requested_super_category in by_super:
        candidates.extend(by_super[requested_super_category])

    if not candidates:
        norm_cat = normalize_text(requested_category)
        for cat, arr in by_category.items():
            if normalize_text(cat) == norm_cat:
                candidates.extend(arr)
        if not candidates and requested_super_category:
            norm_super = normalize_text(requested_super_category)
            for sc, arr in by_super.items():
                if normalize_text(sc) == norm_super:
                    candidates.extend(arr)

    if not candidates:
        return None

    best = None
    best_score = 1e18

    for cand in candidates:
        cand_size = cand["_size_m"]
        score = size_distance_score(target_size_m, cand_size)

        if preferred_style and as_str(cand.get("style")) == preferred_style:
            score -= 0.08
        if preferred_theme and as_str(cand.get("theme")) == preferred_theme:
            score -= 0.05
        if preferred_material and as_str(cand.get("material")) == preferred_material:
            score -= 0.05

        if as_str(cand.get("category")) == requested_category:
            score -= 0.10

        if score < best_score:
            best_score = score
            best = cand

    return best


def make_asset_block(model_id: str, future_root: Path) -> dict[str, Any]:
    asset_dir = future_root / model_id
    mesh_path = asset_dir / "normalized_model.obj"
    return {
        "source": "3D-FUTURE",
        "model_id": model_id,
        "mesh_path": str(mesh_path.resolve()) if mesh_path.exists() else str(mesh_path),
        "mesh_fit_mode": "uniform",
        "mesh_texture_dirs": [str(asset_dir.resolve())],
    }


# ============================================================
# Group post-process: прижать группы к стенам
# ============================================================

def ensure_aabb_for_placement(p: dict[str, Any]) -> dict[str, float]:
    aabb = p.get("aabb")
    if isinstance(aabb, dict):
        return aabb
    aabb = build_aabb_from_center_size(p["position_m"], p["size_m"])
    p["aabb"] = aabb
    return aabb


def shift_placement_xy(p: dict[str, Any], dx: float, dy: float) -> None:
    p["position_m"][0] += dx
    p["position_m"][1] += dy
    aabb = ensure_aabb_for_placement(p)
    aabb["x_min"] += dx
    aabb["x_max"] += dx
    aabb["y_min"] += dy
    aabb["y_max"] += dy


def room_bbox_xy_from_room(room: dict[str, Any]) -> tuple[float, float, float, float]:
    poly = room.get("floor_polygon") or []
    pts: list[tuple[float, float]] = []
    for p in poly:
        if isinstance(p, dict) and "x" in p and "y" in p:
            pts.append((float(p["x"]), float(p["y"])))
    if len(pts) < 3:
        raise RuntimeError("room.floor_polygon is missing or invalid")
    return bbox_of_poly(pts)


def is_floor_like_placement(p: dict[str, Any]) -> bool:
    return as_str(p.get("mount_type"), "floor") != "ceiling"


def group_spread_to_room_walls(
    room: dict[str, Any],
    placements: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Универсальный post-process:
    1. Берём только напольные объекты.
    2. Делим по X на левую / правую половины.
    3. Левую группу сдвигаем так, чтобы её самый левый край коснулся левой стены.
       Правую группу — чтобы её самый правый край коснулся правой стены.
    4. Затем по Y делим на нижнюю / верхнюю половины.
    5. Нижнюю группу сдвигаем к нижней стене, верхнюю — к верхней.
    6. Внутренняя геометрия групп сохраняется.

    Работает по bbox комнаты, без семантики типа комнаты.
    """
    x0, x1, y0, y1 = room_bbox_xy_from_room(room)
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)

    floor_ids = [i for i, p in enumerate(placements) if is_floor_like_placement(p)]
    if not floor_ids:
        return {
            "enabled": True,
            "num_floor_objects": 0,
            "horizontal": {"dx_left": 0.0, "dx_right": 0.0},
            "vertical": {"dy_bottom": 0.0, "dy_top": 0.0},
        }

    for i in floor_ids:
        ensure_aabb_for_placement(placements[i])

    # ----------------------------
    # Step 1: left / right
    # ----------------------------
    left_ids = [i for i in floor_ids if placements[i]["position_m"][0] <= cx]
    right_ids = [i for i in floor_ids if placements[i]["position_m"][0] > cx]

    dx_left = 0.0
    dx_right = 0.0

    if left_ids:
        leftmost = min(placements[i]["aabb"]["x_min"] for i in left_ids)
        dx_left = x0 - leftmost
        for i in left_ids:
            shift_placement_xy(placements[i], dx_left, 0.0)

    if right_ids:
        rightmost = max(placements[i]["aabb"]["x_max"] for i in right_ids)
        dx_right = x1 - rightmost
        for i in right_ids:
            shift_placement_xy(placements[i], dx_right, 0.0)

    # После горизонтального шага пересчитываем деление по вертикали
    bottom_ids = [i for i in floor_ids if placements[i]["position_m"][1] <= cy]
    top_ids = [i for i in floor_ids if placements[i]["position_m"][1] > cy]

    dy_bottom = 0.0
    dy_top = 0.0

    if bottom_ids:
        bottommost = min(placements[i]["aabb"]["y_min"] for i in bottom_ids)
        dy_bottom = y0 - bottommost
        for i in bottom_ids:
            shift_placement_xy(placements[i], 0.0, dy_bottom)

    if top_ids:
        topmost = max(placements[i]["aabb"]["y_max"] for i in top_ids)
        dy_top = y1 - topmost
        for i in top_ids:
            shift_placement_xy(placements[i], 0.0, dy_top)

    return {
        "enabled": True,
        "num_floor_objects": len(floor_ids),
        "left_group_count": len(left_ids),
        "right_group_count": len(right_ids),
        "bottom_group_count": len(bottom_ids),
        "top_group_count": len(top_ids),
        "horizontal": {
            "dx_left": dx_left,
            "dx_right": dx_right,
        },
        "vertical": {
            "dy_bottom": dy_bottom,
            "dy_top": dy_top,
        },
    }


# ============================================================
# Retrieval layout -> scene.v1
# ============================================================

def convert_retrieval_to_scene(
    retrieval: dict[str, Any],
    room_data: dict[str, Any],
    prepared_records: list[dict[str, Any]],
    future_root: Path,
) -> dict[str, Any]:
    model_index = build_model_index(prepared_records)
    by_cat, by_super, cat_size_fallback = build_available_candidate_indexes(prepared_records, future_root)

    room = normalize_room_dict(room_data)
    ceiling_h = as_float(room.get("ceiling_height"), 2.8)

    placements_out: list[dict[str, Any]] = []

    for i, pl in enumerate(retrieval.get("placements") or []):
        category = as_str(pl.get("category"), "object")
        source_model_id = as_str(pl.get("source_asset_id"), "").strip()

        source_rec = model_index.get(source_model_id)

        if source_rec is not None:
            source_size_m = prepared_record_size_to_scene_size(source_rec)
            source_style = source_rec.get("style")
            source_theme = source_rec.get("theme")
            source_material = source_rec.get("material")
            source_super_category = source_rec.get("super-category") or source_rec.get("super_category")
        else:
            source_size_m = cat_size_fallback.get(category, [0.8, 0.8, 0.8])
            source_style = None
            source_theme = None
            source_material = None
            source_super_category = None

        selected_rec = None
        selected_model_id = None
        selected_via_fallback = False

        if source_model_id and model_dir_exists(source_model_id, future_root) and source_rec is not None:
            selected_model_id = source_model_id
            selected_rec = source_rec
        else:
            fallback_rec = choose_similar_existing_model(
                requested_category=category,
                requested_super_category=source_super_category,
                target_size_m=source_size_m,
                preferred_style=source_style,
                preferred_theme=source_theme,
                preferred_material=source_material,
                by_category=by_cat,
                by_super=by_super,
            )

            if fallback_rec is not None:
                selected_rec = fallback_rec
                selected_model_id = as_str(fallback_rec.get("model_id")).strip()
                selected_via_fallback = True

        if selected_rec is not None:
            size_m = list(selected_rec["_size_m"]) if "_size_m" in selected_rec else prepared_record_size_to_scene_size(selected_rec)
            style = selected_rec.get("style")
            theme = selected_rec.get("theme")
            material = selected_rec.get("material")
            super_category = selected_rec.get("super-category") or selected_rec.get("super_category")
        else:
            size_m = cat_size_fallback.get(category, [0.8, 0.8, 0.8])
            style = source_style
            theme = source_theme
            material = source_material
            super_category = source_super_category

        x = as_float(pl.get("x"))
        y = as_float(pl.get("z"))   # retrieval x,z -> scene x,y
        yaw_deg_raw = as_float(pl.get("yaw_deg"), 0.0)
        rotation_deg = quantize_rot_0_90_180_270(yaw_deg_raw)

        mount_type = "ceiling" if looks_like_ceiling_light(category) else "floor"
        if mount_type == "ceiling":
            z = ceiling_h - size_m[2] / 2.0
        else:
            z = size_m[2] / 2.0

        position_m = [x, y, z]
        aabb = build_aabb_from_center_size(position_m, size_m)

        if selected_model_id:
            asset_block = make_asset_block(selected_model_id, future_root)
        else:
            asset_block = {
                "source": "retrieval_unknown",
                "model_id": None,
                "mesh_fit_mode": "uniform",
            }

        placement = {
            "id": f"obj_{i+1:04d}",
            "name": category,
            "category": category,
            "position_m": position_m,
            "size_m": size_m,
            "rotation_deg": rotation_deg,
            "yaw_deg": yaw_deg_raw,
            "yaw_rad": math.radians(yaw_deg_raw),
            "aabb": aabb,
            "mount_type": mount_type,
            "wall_contact_side": None,
            "constraints": {},
            "asset": asset_block,
            "source": {
                "placement_source": "retrieval_knn",
                "reference_room_id": retrieval.get("reference_room_id"),
                "reference_room_file": retrieval.get("reference_room_file"),
                "source_asset_id": source_model_id or None,
                "selected_model_id": selected_model_id or None,
                "transfer_mode": pl.get("transfer_mode"),
                "selected_via_fallback": selected_via_fallback,
            },
            "meta": {
                "source_category": pl.get("source_category"),
                "target_room_id": retrieval.get("target_room_id"),
                "target_room_type": retrieval.get("target_room_type"),
                "requested_counter": retrieval.get("requested_counter"),
                "asset": {
                    "super_category": super_category,
                    "style": style,
                    "theme": theme,
                    "material": material,
                    "original_source_model_id": source_model_id or None,
                    "selected_model_id": selected_model_id or None,
                    "selected_via_fallback": selected_via_fallback,
                },
            },
        }

        placements_out.append(placement)

    # Универсальный group post-process
    postprocess_info = group_spread_to_room_walls(room, placements_out)

    scene = {
        "schema": "scene.v1",
        "room": room,
        "placements": placements_out,
        "meta": {
            "placer": "retrieval_knn",
            "mode": "retrieval",
            "reference_room_id": retrieval.get("reference_room_id"),
            "reference_room_file": retrieval.get("reference_room_file"),
            "requested_items": retrieval.get("requested_items"),
            "requested_counter": retrieval.get("requested_counter"),
            "neighbors": retrieval.get("neighbors"),
            "group_postprocess": postprocess_info,
        },
    }
    return scene


# ============================================================
# Main pipeline
# ============================================================

def parse_items_arg(items_arg: str) -> list[str]:
    items = [x.strip() for x in items_arg.split(",") if x.strip()]
    if not items:
        raise RuntimeError("--items is empty")
    return items


def build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="1-NN / k-NN retrieval baseline for 3D-FRONT that directly outputs scene.v1"
    )
    ap.add_argument("--dataset-root", required=True, help="Path to 3D-FRONT-processed-mini")
    ap.add_argument("--target-room", required=True, help="Path to target room json")
    ap.add_argument("--items", required=True, help="Comma-separated requested categories")
    ap.add_argument("--room-type", default=None, help="Optional room type override: bedroom/living/office")
    ap.add_argument("--top-k", type=int, default=3)

    ap.add_argument("--prepared-info", required=True, help="prepared_model_info.json")
    ap.add_argument("--future-root", required=True, help="3D-FUTURE-model root")

    ap.add_argument("--out", required=True, help="output scene.v1.json")
    ap.add_argument("--dump-retrieval", default=None, help="Optional path to save retrieval_layout.json")
    ap.add_argument("--dump-neighbors", default=None, help="Optional path to save neighbors json")
    ap.add_argument("--dump-asset-index", default=None, help="Optional path to save jid/model_id -> category map")
    return ap


def main() -> None:
    args = build_cli().parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    target_room_path = Path(args.target_room).expanduser().resolve()
    prepared_info_path = Path(args.prepared_info).expanduser().resolve()
    future_root = Path(args.future_root).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    requested_items = parse_items_arg(args.items)

    roomrecs = load_dataset(dataset_root)
    asset_index = build_asset_to_category_index(roomrecs)

    target_room = load_target_room(target_room_path, room_type_override=args.room_type)
    requested_counter = Counter(requested_items)

    neighbors = retrieve_topk_rooms(
        roomrecs=roomrecs,
        target_room=target_room,
        requested_counter=requested_counter,
        top_k=max(1, int(args.top_k)),
    )
    if not neighbors:
        raise RuntimeError("No suitable neighbors found")

    ref_room = choose_reference_layout(neighbors, requested_counter)

    retrieval_layout = transfer_layout_from_reference(
        target_room=target_room,
        requested_items=requested_items,
        reference_room=ref_room,
    )
    retrieval_layout["neighbors"] = neighbor_report(neighbors)

    if args.dump_retrieval:
        save_json(args.dump_retrieval, retrieval_layout)

    if args.dump_neighbors:
        save_json(args.dump_neighbors, neighbor_report(neighbors))

    if args.dump_asset_index:
        compact_index = {
            asset_id: cnt.most_common(1)[0][0]
            for asset_id, cnt in asset_index.items()
            if cnt
        }
        save_json(args.dump_asset_index, compact_index)

    room_data = load_json(target_room_path)
    prepared_records = load_prepared_info(prepared_info_path)

    scene = convert_retrieval_to_scene(
        retrieval=retrieval_layout,
        room_data=room_data,
        prepared_records=prepared_records,
        future_root=future_root,
    )
    save_json(out_path, scene)

    print(f"Loaded rooms: {len(roomrecs)}")
    print(f"Target room: {target_room.room_id} ({target_room.room_type})")
    print(f"Reference room: {ref_room.room_id}")
    print(f"Similarity top-1: {neighbors[0][0]:.6f}")
    print(f"Placements: {len(scene.get('placements', []))}")
    print(f"Saved scene.v1: {out_path}")


if __name__ == "__main__":
    main()