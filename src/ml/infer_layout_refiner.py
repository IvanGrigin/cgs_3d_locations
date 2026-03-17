#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
infer_layout_refiner.py

Инференс LayoutRefiner на данных pipeline.

Что делает:
1. Читает room.json
2. Читает objects.json / objects.v1
3. Строит начальную расстановку через init-mode:
   - random
   - relaxed
4. Загружает checkpoint LayoutRefiner
5. Делает refinement на cpu / mps / cuda
6. Постобрабатывает результат:
   - проекция внутрь комнаты
   - снап углов
   - разруливание AABB-пересечений
7. Сохраняет placement.v1
8. Опционально сохраняет debug PNG

Замечание по совместимости:
- новый корректный аргумент: --init-mode
- старый аргумент --mode оставлен как алиас для обратной совместимости

Пример:
python src/ml/infer_layout_refiner.py \
  --checkpoint runs/layout_livingroom_mps_v2/best.pt \
  --room data/input/room.json \
  --objects data/output/objects.v1.json \
  --out data/output/placement_layout_refiner.json \
  --device mps \
  --init-mode relaxed \
  --class-map config/layout_refiner_class_map.json \
  --debug-image out/debug_layout.png \
  --print-summary
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn


# ============================================================
# Общие утилиты
# ============================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(requested: str = "auto") -> torch.device:
    requested = str(requested).lower().strip()

    if requested == "cpu":
        return torch.device("cpu")

    if requested == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("MPS недоступен")

    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA недоступна")

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def normalize_vec2(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    norm = torch.sqrt(torch.clamp((x * x).sum(dim=-1, keepdim=True), min=eps))
    return x / norm


def vec2_to_yaw_deg(v: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(v[1], v[0])))


def yaw_deg_to_vec2(deg: float) -> np.ndarray:
    rad = math.radians(float(deg))
    return np.array([math.cos(rad), math.sin(rad)], dtype=np.float32)


def snap_angle_deg(deg: float, step: float) -> float:
    if step <= 0:
        return float(deg)
    return float(round(deg / step) * step)


def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def load_json(path: str | Path) -> Any:
    p = Path(path).expanduser().resolve()
    return json.loads(p.read_text(encoding="utf-8"))


def save_json(path: str | Path, data: Any) -> None:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


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


def extract_list2(x: Any) -> Optional[list[float]]:
    if isinstance(x, (list, tuple)) and len(x) >= 2:
        return [as_float(x[0]), as_float(x[1])]
    return None


def extract_list3(x: Any) -> Optional[list[float]]:
    if isinstance(x, (list, tuple)) and len(x) >= 3:
        return [as_float(x[0]), as_float(x[1]), as_float(x[2])]
    return None


def point_in_polygon_xy(point: np.ndarray, polygon_xy: np.ndarray) -> bool:
    """
    Ray casting. Точка на границе считается допустимой.
    """
    x = float(point[0])
    y = float(point[1])
    inside = False
    n = polygon_xy.shape[0]

    for i in range(n):
        x1, y1 = float(polygon_xy[i, 0]), float(polygon_xy[i, 1])
        x2, y2 = float(polygon_xy[(i + 1) % n, 0]), float(polygon_xy[(i + 1) % n, 1])

        # Проверка попадания на горизонтальные/вертикальные края нам здесь не критична:
        # bbox-клип всё равно ограничит позицию, а polygon-check используется как дополнительный фильтр.
        intersects = ((y1 > y) != (y2 > y))
        if intersects:
            denom = (y2 - y1)
            if abs(denom) < 1e-12:
                continue
            x_cross = x1 + (y - y1) * (x2 - x1) / denom
            if x <= x_cross:
                inside = not inside

    return inside


# ============================================================
# Модель
# ============================================================

class FloorplanEncoder(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, fpoc: torch.Tensor, nfpc: torch.Tensor) -> torch.Tensor:
        """
        fpoc: [B, P, 2]
        nfpc: [B]
        return: [B, d_model]
        """
        batch_size, num_points, _ = fpoc.shape
        device = fpoc.device

        idx = torch.arange(num_points, device=device).unsqueeze(0).expand(batch_size, num_points)
        valid = idx < nfpc.unsqueeze(1)

        h = self.net(fpoc)
        h = h * valid.unsqueeze(-1)

        denom = valid.sum(dim=1, keepdim=True).clamp_min(1)
        pooled = h.sum(dim=1) / denom
        return pooled


class LayoutRefiner(nn.Module):
    def __init__(
        self,
        class_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.class_dim = class_dim
        input_dim = 2 + 2 + 2 + class_dim

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        self.floorplan_encoder = FloorplanEncoder(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 6),
        )

    def forward_once(
        self,
        pos: torch.Tensor,
        ang: torch.Tensor,
        siz: torch.Tensor,
        cla: torch.Tensor,
        valid_mask: torch.Tensor,
        fpoc: torch.Tensor,
        nfpc: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.cat([pos, ang, siz, cla], dim=-1)
        h = self.input_proj(x)

        fp_ctx = self.floorplan_encoder(fpoc, nfpc)
        h = h + fp_ctx.unsqueeze(1)

        padding_mask = ~valid_mask
        h = self.encoder(h, src_key_padding_mask=padding_mask)

        out = self.head(h)
        delta_pos = out[..., 0:2]
        delta_ang = out[..., 2:4]
        delta_siz = out[..., 4:6]

        pred_pos = pos + delta_pos
        pred_ang = normalize_vec2(ang + delta_ang)
        pred_siz = torch.clamp(siz + delta_siz, min=-10.0)

        return pred_pos, pred_ang, pred_siz

    def forward(
        self,
        noisy_pos: torch.Tensor,
        noisy_ang: torch.Tensor,
        noisy_siz: torch.Tensor,
        cla: torch.Tensor,
        valid_mask: torch.Tensor,
        fpoc: torch.Tensor,
        nfpc: torch.Tensor,
        denoise_steps: int = 3,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pos, ang, siz = noisy_pos, noisy_ang, noisy_siz
        for _ in range(denoise_steps):
            pos, ang, siz = self.forward_once(pos, ang, siz, cla, valid_mask, fpoc, nfpc)
        return pos, ang, siz


# ============================================================
# Checkpoint
# ============================================================

@dataclass
class CheckpointBundle:
    model: LayoutRefiner
    class_dim: int
    stats: dict[str, np.ndarray]
    config: dict[str, Any]


def load_checkpoint_bundle(checkpoint_path: str | Path, device: torch.device) -> CheckpointBundle:
    ckpt = torch.load(
        Path(checkpoint_path).expanduser().resolve(),
        map_location="cpu",
        weights_only=False,
    )

    config = ckpt.get("config", {})
    class_dim = int(ckpt["class_dim"])
    stats = ckpt["stats"]

    model = LayoutRefiner(
        class_dim=class_dim,
        d_model=int(config.get("d_model", 256)),
        nhead=int(config.get("nhead", 8)),
        num_layers=int(config.get("num_layers", 6)),
        dim_feedforward=int(config.get("dim_feedforward", 512)),
        dropout=float(config.get("dropout", 0.1)),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    stats_np = {
        "pos_mean": np.asarray(stats["pos_mean"], dtype=np.float32),
        "pos_std": np.asarray(stats["pos_std"], dtype=np.float32),
        "siz_mean": np.asarray(stats["siz_mean"], dtype=np.float32),
        "siz_std": np.asarray(stats["siz_std"], dtype=np.float32),
    }

    return CheckpointBundle(
        model=model,
        class_dim=class_dim,
        stats=stats_np,
        config=config,
    )


# ============================================================
# Каноническое представление сцены
# ============================================================

@dataclass
class CanonicalObject:
    obj_id: str
    name: str
    category: str
    size_m: np.ndarray         # [3]
    constraints: dict[str, Any]
    asset: dict[str, Any]
    meta: dict[str, Any]


@dataclass
class CanonicalRoom:
    room_id: str
    polygon_xy: np.ndarray     # [P, 2]
    z_floor_m: float
    bbox_min_xy: np.ndarray    # [2]
    bbox_max_xy: np.ndarray    # [2]
    raw_room: dict[str, Any]


# ============================================================
# Room adapter
# ============================================================

def unwrap_room_json(data: dict[str, Any]) -> dict[str, Any]:
    if isinstance(data.get("room"), dict):
        room = dict(data["room"])
        for k, v in data.items():
            if k != "room" and k not in room:
                room[k] = v
        return room
    return dict(data)


def polygon_from_room_dict(room: dict[str, Any]) -> Optional[np.ndarray]:
    candidates = [
        room.get("fpoc"),
        room.get("floor_polygon"),
        room.get("polygon"),
        room.get("room_polygon"),
        room.get("contour"),
        room.get("floor_outline"),
        room.get("outline"),
        room.get("vertices"),
    ]

    for cand in candidates:
        if not isinstance(cand, list) or len(cand) < 3:
            continue

        pts: list[list[float]] = []
        ok = True

        for p in cand:
            if isinstance(p, dict):
                if "x" in p and "y" in p:
                    pts.append([as_float(p["x"]), as_float(p["y"])])
                elif "x" in p and "z" in p:
                    pts.append([as_float(p["x"]), as_float(p["z"])])
                else:
                    ok = False
                    break
            elif isinstance(p, (list, tuple)) and len(p) >= 2:
                pts.append([as_float(p[0]), as_float(p[1])])
            else:
                ok = False
                break

        if ok and len(pts) >= 3:
            return np.asarray(pts, dtype=np.float32)

    return None


def bbox_polygon_from_room_dict(room: dict[str, Any]) -> np.ndarray:
    aabb = room.get("aabb") if isinstance(room.get("aabb"), dict) else room.get("bbox")
    if isinstance(aabb, dict):
        x_min = as_float(aabb.get("x_min", aabb.get("xmin", -2.0)))
        x_max = as_float(aabb.get("x_max", aabb.get("xmax", 2.0)))

        if "y_min" in aabb or "y_max" in aabb:
            y_min = as_float(aabb.get("y_min", aabb.get("ymin", -2.0)))
            y_max = as_float(aabb.get("y_max", aabb.get("ymax", 2.0)))
        else:
            y_min = as_float(aabb.get("z_min", aabb.get("zmin", -2.0)))
            y_max = as_float(aabb.get("z_max", aabb.get("zmax", 2.0)))

        return np.asarray(
            [
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_max],
            ],
            dtype=np.float32,
        )

    size_m = extract_list3(room.get("size_m")) or extract_list3(room.get("size"))
    if size_m is not None:
        sx = max(0.1, float(size_m[0]))
        sy = max(0.1, float(size_m[1] if len(size_m) > 1 else size_m[0]))
        return np.asarray(
            [
                [-sx / 2.0, -sy / 2.0],
                [sx / 2.0, -sy / 2.0],
                [sx / 2.0, sy / 2.0],
                [-sx / 2.0, sy / 2.0],
            ],
            dtype=np.float32,
        )

    width = as_float(room.get("width_m", room.get("width", 4.0)), 4.0)
    depth = as_float(room.get("depth_m", room.get("depth", 4.0)), 4.0)

    return np.asarray(
        [
            [-width / 2.0, -depth / 2.0],
            [width / 2.0, -depth / 2.0],
            [width / 2.0, depth / 2.0],
            [-width / 2.0, depth / 2.0],
        ],
        dtype=np.float32,
    )


def parse_room(room_json: dict[str, Any]) -> CanonicalRoom:
    room = unwrap_room_json(room_json)

    polygon_xy = polygon_from_room_dict(room)
    if polygon_xy is None:
        polygon_xy = bbox_polygon_from_room_dict(room)

    bbox_min_xy = polygon_xy.min(axis=0)
    bbox_max_xy = polygon_xy.max(axis=0)

    room_id = as_str(
        room.get("id") or room.get("room_id") or room.get("uid") or room.get("name"),
        "room_001",
    )

    z_floor_m = as_float(
        room.get("z_floor_m", room.get("floor_z_m", room.get("floor_z", 0.0))),
        0.0,
    )

    return CanonicalRoom(
        room_id=room_id,
        polygon_xy=polygon_xy.astype(np.float32),
        z_floor_m=float(z_floor_m),
        bbox_min_xy=bbox_min_xy.astype(np.float32),
        bbox_max_xy=bbox_max_xy.astype(np.float32),
        raw_room=room,
    )


# ============================================================
# Objects adapter
# ============================================================

def parse_objects(objects_json: dict[str, Any]) -> list[CanonicalObject]:
    if isinstance(objects_json.get("objects"), list):
        raw_objects = objects_json["objects"]
    elif isinstance(objects_json.get("items"), list):
        raw_objects = objects_json["items"]
    else:
        raise RuntimeError("objects.json должен содержать список objects или items")

    out: list[CanonicalObject] = []

    for i, obj in enumerate(raw_objects):
        if not isinstance(obj, dict):
            continue

        obj_id = as_str(obj.get("id") or obj.get("object_id") or obj.get("uid"), f"obj_{i + 1:04d}")
        name = as_str(obj.get("name") or obj.get("class_name") or obj.get("class") or obj.get("type"), "object")
        category = as_str(obj.get("category") or name, name)

        size_m = extract_list3(obj.get("size_m"))
        if size_m is None:
            size_min = extract_list3(obj.get("size_min_m"))
            size_max = extract_list3(obj.get("size_max_m"))
            if size_min is not None and size_max is not None:
                size_m = [
                    0.5 * (size_min[0] + size_max[0]),
                    0.5 * (size_min[1] + size_max[1]),
                    0.5 * (size_min[2] + size_max[2]),
                ]
            else:
                size_m = [1.0, 1.0, 1.0]

        constraints = obj.get("constraints") if isinstance(obj.get("constraints"), dict) else {}
        asset = obj.get("asset") if isinstance(obj.get("asset"), dict) else {}
        meta = obj.get("meta") if isinstance(obj.get("meta"), dict) else {}

        out.append(
            CanonicalObject(
                obj_id=obj_id,
                name=name,
                category=category,
                size_m=np.asarray(size_m, dtype=np.float32),
                constraints=dict(constraints),
                asset=dict(asset),
                meta=dict(meta),
            )
        )

    if not out:
        raise RuntimeError("После парсинга список объектов пуст")

    return out


# ============================================================
# Class mapping
# ============================================================

COMMON_CATEGORY_ALIASES = {
    "couch": "sofa",
    "sofa": "sofa",
    "tv stand": "tv_stand",
    "tv_stand": "tv_stand",
    "television stand": "tv_stand",
    "coffee table": "coffee_table",
    "coffee_table": "coffee_table",
    "side table": "side_table",
    "side_table": "side_table",
    "nightstand": "nightstand",
    "wardrobe": "wardrobe",
    "closet": "wardrobe",
    "armchair": "armchair",
    "chair": "chair",
    "table": "table",
    "desk": "desk",
    "bed": "bed",
    "double bed": "bed",
    "single bed": "bed",
    "lamp": "lamp",
    "floor lamp": "lamp",
    "cabinet": "cabinet",
    "dresser": "dresser",
    "bookshelf": "bookshelf",
    "shelf": "shelf",
    "bench": "bench",
    "stool": "stool",
    "plant": "plant",
    "rug": "rug",
    "carpet": "rug",
    "pouf": "ottoman",
    "ottoman": "ottoman",
    "console": "console_table",
    "console_table": "console_table",
}


def normalize_category_name(s: str) -> str:
    x = s.strip().lower().replace("-", " ").replace("_", " ")
    x = " ".join(x.split())
    return COMMON_CATEGORY_ALIASES.get(x, x.replace(" ", "_"))


def load_class_map(path: Optional[str | Path]) -> dict[str, int]:
    if path is None:
        return {}

    data = load_json(path)
    if not isinstance(data, dict):
        raise RuntimeError("--class-map должен быть JSON-объектом вида {category: class_id}")

    out: dict[str, int] = {}
    for k, v in data.items():
        out[normalize_category_name(str(k))] = int(v)
    return out


def fallback_class_id(category: str, class_dim: int) -> int:
    h = hashlib.sha1(category.encode("utf-8")).hexdigest()
    val = int(h[:8], 16)
    return int(val % class_dim)


def category_to_onehot(category: str, class_dim: int, class_map: dict[str, int]) -> np.ndarray:
    key = normalize_category_name(category)

    if key in class_map:
        idx = int(class_map[key])
    else:
        idx = fallback_class_id(key, class_dim)

    idx = max(0, min(class_dim - 1, idx))
    vec = np.zeros(class_dim, dtype=np.float32)
    vec[idx] = 1.0
    return vec


# ============================================================
# Initial layout
# ============================================================

def sample_position_inside_bbox(
    bbox_min_xy: np.ndarray,
    bbox_max_xy: np.ndarray,
    size_xy: np.ndarray,
    margin: float = 0.05,
) -> np.ndarray:
    x_lo = float(bbox_min_xy[0] + size_xy[0] / 2.0 + margin)
    x_hi = float(bbox_max_xy[0] - size_xy[0] / 2.0 - margin)
    y_lo = float(bbox_min_xy[1] + size_xy[1] / 2.0 + margin)
    y_hi = float(bbox_max_xy[1] - size_xy[1] / 2.0 - margin)

    if x_lo > x_hi:
        xc = 0.5 * float(bbox_min_xy[0] + bbox_max_xy[0])
    else:
        xc = random.uniform(x_lo, x_hi)

    if y_lo > y_hi:
        yc = 0.5 * float(bbox_min_xy[1] + bbox_max_xy[1])
    else:
        yc = random.uniform(y_lo, y_hi)

    return np.asarray([xc, yc], dtype=np.float32)


def aabb_overlap_area_xy(
    pos_a: np.ndarray,
    siz_a: np.ndarray,
    pos_b: np.ndarray,
    siz_b: np.ndarray,
) -> float:
    ax0 = pos_a[0] - siz_a[0] / 2.0
    ax1 = pos_a[0] + siz_a[0] / 2.0
    ay0 = pos_a[1] - siz_a[1] / 2.0
    ay1 = pos_a[1] + siz_a[1] / 2.0

    bx0 = pos_b[0] - siz_b[0] / 2.0
    bx1 = pos_b[0] + siz_b[0] / 2.0
    by0 = pos_b[1] - siz_b[1] / 2.0
    by1 = pos_b[1] + siz_b[1] / 2.0

    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    return float(ix * iy)


def project_inside_bbox(
    pos_xy: np.ndarray,
    size_xy: np.ndarray,
    bbox_min_xy: np.ndarray,
    bbox_max_xy: np.ndarray,
) -> np.ndarray:
    x = clamp(
        float(pos_xy[0]),
        float(bbox_min_xy[0] + size_xy[0] / 2.0),
        float(bbox_max_xy[0] - size_xy[0] / 2.0),
    )
    y = clamp(
        float(pos_xy[1]),
        float(bbox_min_xy[1] + size_xy[1] / 2.0),
        float(bbox_max_xy[1] - size_xy[1] / 2.0),
    )
    return np.asarray([x, y], dtype=np.float32)


def project_inside_room(
    pos_xy: np.ndarray,
    size_xy: np.ndarray,
    room: CanonicalRoom,
    max_iters: int = 40,
) -> np.ndarray:
    """
    Сначала жёстко клипим в bbox, затем при необходимости мягко тянем к центру комнаты,
    пока точка центра не войдёт в polygon.
    """
    p = project_inside_bbox(pos_xy, size_xy, room.bbox_min_xy, room.bbox_max_xy)

    if point_in_polygon_xy(p, room.polygon_xy):
        return p

    center = 0.5 * (room.bbox_min_xy + room.bbox_max_xy)
    cur = p.copy()

    for _ in range(max_iters):
        cur = 0.85 * cur + 0.15 * center
        cur = project_inside_bbox(cur, size_xy, room.bbox_min_xy, room.bbox_max_xy)
        if point_in_polygon_xy(cur, room.polygon_xy):
            return cur

    return project_inside_bbox(center, size_xy, room.bbox_min_xy, room.bbox_max_xy)


def build_initial_layout_random(
    room: CanonicalRoom,
    objects: list[CanonicalObject],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(objects)
    pos = np.zeros((n, 2), dtype=np.float32)
    ang = np.zeros((n, 2), dtype=np.float32)
    siz = np.zeros((n, 2), dtype=np.float32)

    angles = [0.0, 90.0, 180.0, 270.0]
    order = sorted(range(n), key=lambda i: float(objects[i].size_m[0] * objects[i].size_m[1]), reverse=True)

    for idx in order:
        obj = objects[idx]
        size_xy = np.asarray([max(0.05, obj.size_m[0]), max(0.05, obj.size_m[1])], dtype=np.float32)
        siz[idx] = size_xy

        best_pos = None
        best_penalty = float("inf")
        best_ang = 0.0

        for _ in range(80):
            candidate_pos = sample_position_inside_bbox(room.bbox_min_xy, room.bbox_max_xy, size_xy)
            candidate_pos = project_inside_room(candidate_pos, size_xy, room)

            candidate_ang = random.choice(angles)
            penalty = 0.0

            for j in range(n):
                if j == idx:
                    continue
                if siz[j].sum() <= 0:
                    continue
                penalty += aabb_overlap_area_xy(candidate_pos, size_xy, pos[j], siz[j])

            if penalty < best_penalty:
                best_penalty = penalty
                best_pos = candidate_pos
                best_ang = candidate_ang

        if best_pos is None:
            best_pos = project_inside_room(
                sample_position_inside_bbox(room.bbox_min_xy, room.bbox_max_xy, size_xy),
                size_xy,
                room,
            )
            best_ang = 0.0

        pos[idx] = best_pos
        ang[idx] = yaw_deg_to_vec2(best_ang)

    return pos, ang, siz


def greedy_relax_layout(
    room: CanonicalRoom,
    pos: np.ndarray,
    ang: np.ndarray,
    siz: np.ndarray,
    steps: int = 120,
    push_scale: float = 0.03,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = pos.shape[0]

    for _ in range(steps):
        moved = False

        for i in range(n):
            pos[i] = project_inside_room(pos[i], siz[i], room)

        for i in range(n):
            for j in range(i + 1, n):
                overlap = aabb_overlap_area_xy(pos[i], siz[i], pos[j], siz[j])
                if overlap <= 1e-8:
                    continue

                d = pos[j] - pos[i]
                norm = np.linalg.norm(d)
                if norm < 1e-6:
                    d = np.random.randn(2).astype(np.float32)
                    norm = np.linalg.norm(d)
                d = d / max(norm, 1e-6)

                shift = d * push_scale
                pos[i] -= shift
                pos[j] += shift
                moved = True

        # крупные объекты слегка прижимаем к ближайшей стороне bbox.
        # это не строгая логика стен из cube, а только инициализация.
        areas = siz[:, 0] * siz[:, 1]
        order = np.argsort(-areas)
        top_k = min(4, n)

        for k in range(top_k):
            i = int(order[k])
            size_xy = siz[i]

            left_gap = abs(pos[i, 0] - (room.bbox_min_xy[0] + size_xy[0] / 2.0))
            right_gap = abs((room.bbox_max_xy[0] - size_xy[0] / 2.0) - pos[i, 0])
            bottom_gap = abs(pos[i, 1] - (room.bbox_min_xy[1] + size_xy[1] / 2.0))
            top_gap = abs((room.bbox_max_xy[1] - size_xy[1] / 2.0) - pos[i, 1])

            best_side = min(
                [
                    ("left", left_gap),
                    ("right", right_gap),
                    ("bottom", bottom_gap),
                    ("top", top_gap),
                ],
                key=lambda x: x[1],
            )[0]

            if best_side == "left":
                pos[i, 0] = room.bbox_min_xy[0] + size_xy[0] / 2.0
                ang[i] = yaw_deg_to_vec2(0.0)
            elif best_side == "right":
                pos[i, 0] = room.bbox_max_xy[0] - size_xy[0] / 2.0
                ang[i] = yaw_deg_to_vec2(180.0)
            elif best_side == "bottom":
                pos[i, 1] = room.bbox_min_xy[1] + size_xy[1] / 2.0
                ang[i] = yaw_deg_to_vec2(90.0)
            else:
                pos[i, 1] = room.bbox_max_xy[1] - size_xy[1] / 2.0
                ang[i] = yaw_deg_to_vec2(270.0)

        for i in range(n):
            pos[i] = project_inside_room(pos[i], siz[i], room)

        if not moved:
            break

    return pos, ang, siz


def build_initial_layout(
    room: CanonicalRoom,
    objects: list[CanonicalObject],
    init_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos, ang, siz = build_initial_layout_random(room, objects)

    if init_mode == "relaxed":
        pos, ang, siz = greedy_relax_layout(room, pos, ang, siz, steps=160, push_scale=0.04)

    return pos, ang, siz


# ============================================================
# Tensorization
# ============================================================

def normalize_pos_np(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean[None, :]) / std[None, :]


def normalize_siz_np(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean[None, :]) / std[None, :]


def denorm_pos_t(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return x * std.view(1, 1, 2) + mean.view(1, 1, 2)


def denorm_siz_t(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return x * std.view(1, 1, 2) + mean.view(1, 1, 2)


def build_model_inputs(
    room: CanonicalRoom,
    objects: list[CanonicalObject],
    class_dim: int,
    class_map: dict[str, int],
    pos_xy: np.ndarray,
    ang_xy: np.ndarray,
    siz_xy: np.ndarray,
    stats: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    n = len(objects)

    cla = np.zeros((n, class_dim), dtype=np.float32)
    for i, obj in enumerate(objects):
        cla[i] = category_to_onehot(obj.category, class_dim, class_map)

    fpoc = room.polygon_xy.astype(np.float32)
    nfpc = np.asarray([fpoc.shape[0]], dtype=np.int64)

    pos_n = normalize_pos_np(pos_xy, stats["pos_mean"], stats["pos_std"]).astype(np.float32)
    siz_n = normalize_siz_np(siz_xy, stats["siz_mean"], stats["siz_std"]).astype(np.float32)

    valid_mask = np.ones((n,), dtype=bool)

    return {
        "pos": pos_n[None, :, :],
        "ang": ang_xy[None, :, :].astype(np.float32),
        "siz": siz_n[None, :, :].astype(np.float32),
        "cla": cla[None, :, :].astype(np.float32),
        "valid_mask": valid_mask[None, :],
        "fpoc": fpoc[None, :, :].astype(np.float32),
        "nfpc": nfpc,
    }


# ============================================================
# Postprocess
# ============================================================

def resolve_collisions_greedy(
    pos_xy: np.ndarray,
    siz_xy: np.ndarray,
    room: CanonicalRoom,
    iterations: int = 120,
    shift_scale: float = 0.02,
) -> np.ndarray:
    pos = pos_xy.copy()
    n = pos.shape[0]

    for _ in range(iterations):
        moved = False

        for i in range(n):
            pos[i] = project_inside_room(pos[i], siz_xy[i], room)

        for i in range(n):
            for j in range(i + 1, n):
                overlap = aabb_overlap_area_xy(pos[i], siz_xy[i], pos[j], siz_xy[j])
                if overlap <= 1e-8:
                    continue

                d = pos[j] - pos[i]
                norm = np.linalg.norm(d)
                if norm < 1e-6:
                    d = np.random.randn(2).astype(np.float32)
                    norm = np.linalg.norm(d)
                d = d / max(norm, 1e-6)

                shift = d * shift_scale
                pos[i] -= shift
                pos[j] += shift
                moved = True

        if not moved:
            break

    for i in range(n):
        pos[i] = project_inside_room(pos[i], siz_xy[i], room)

    return pos


# ============================================================
# Placement export
# ============================================================

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


def export_placement_v1(
    room: CanonicalRoom,
    objects: list[CanonicalObject],
    pred_pos_xy: np.ndarray,
    pred_ang_xy: np.ndarray,
    pred_siz_xy: np.ndarray,
    init_mode: str,
    placer_name: str = "layout_refiner",
) -> dict[str, Any]:
    placements: list[dict[str, Any]] = []

    for i, obj in enumerate(objects):
        yaw_deg = vec2_to_yaw_deg(pred_ang_xy[i])

        size_m = [
            float(max(0.02, pred_siz_xy[i, 0])),
            float(max(0.02, pred_siz_xy[i, 1])),
            float(max(0.02, obj.size_m[2])),
        ]

        position_m = [
            float(pred_pos_xy[i, 0]),
            float(pred_pos_xy[i, 1]),
            float(room.z_floor_m + size_m[2] / 2.0),
        ]

        placements.append(
            {
                "id": obj.obj_id,
                "name": obj.name,
                "category": obj.category,
                "position_m": position_m,
                "size_m": size_m,
                "rotation_deg": float(yaw_deg),
                "yaw_deg": float(yaw_deg),
                "yaw_rad": float(math.radians(yaw_deg)),
                "aabb": build_aabb_from_center_size(position_m, size_m),
                "mount_type": obj.constraints.get("mount_type"),
                "wall_contact_side": obj.constraints.get("wall_contact_side"),
                "constraints": obj.constraints,
                "asset": obj.asset,
                "source": {
                    "placement_source": placer_name,
                },
                "meta": {
                    "refined_from_init_mode": init_mode,
                    "object_meta": obj.meta,
                },
            }
        )

    return {
        "schema": "placement.v1",
        "placer": placer_name,
        "mode": init_mode,
        "placements": placements,
        "meta": {
            "room_id": room.room_id,
            "num_objects": len(objects),
        },
    }


# ============================================================
# Debug visualization
# ============================================================

def draw_debug_layout(
    room: CanonicalRoom,
    objects: list[CanonicalObject],
    init_pos_xy: np.ndarray,
    pred_pos_xy: np.ndarray,
    pred_siz_xy: np.ndarray,
    debug_image_path: str | Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))

    poly = room.polygon_xy
    poly_closed = np.vstack([poly, poly[:1]])
    ax.plot(poly_closed[:, 0], poly_closed[:, 1])

    for i, obj in enumerate(objects):
        ax.scatter(init_pos_xy[i, 0], init_pos_xy[i, 1], s=20, alpha=0.5)

        x = pred_pos_xy[i, 0] - pred_siz_xy[i, 0] / 2.0
        y = pred_pos_xy[i, 1] - pred_siz_xy[i, 1] / 2.0
        w = pred_siz_xy[i, 0]
        h = pred_siz_xy[i, 1]

        rect = plt.Rectangle((x, y), w, h, fill=False)
        ax.add_patch(rect)
        ax.text(pred_pos_xy[i, 0], pred_pos_xy[i, 1], obj.category, fontsize=7)

    ax.set_aspect("equal")
    ax.set_title("layout_refiner debug")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    out_path = Path(debug_image_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=160)
    plt.close(fig)


# ============================================================
# Основной inference
# ============================================================

@torch.no_grad()
def run_inference(
    bundle: CheckpointBundle,
    room: CanonicalRoom,
    objects: list[CanonicalObject],
    class_map: dict[str, int],
    init_mode: str,
    device: torch.device,
    denoise_steps_override: Optional[int],
    snap_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    init_pos_xy, init_ang_xy, init_siz_xy = build_initial_layout(room, objects, init_mode=init_mode)

    inputs_np = build_model_inputs(
        room=room,
        objects=objects,
        class_dim=bundle.class_dim,
        class_map=class_map,
        pos_xy=init_pos_xy,
        ang_xy=init_ang_xy,
        siz_xy=init_siz_xy,
        stats=bundle.stats,
    )

    pos = torch.tensor(inputs_np["pos"], dtype=torch.float32, device=device)
    ang = torch.tensor(inputs_np["ang"], dtype=torch.float32, device=device)
    siz = torch.tensor(inputs_np["siz"], dtype=torch.float32, device=device)
    cla = torch.tensor(inputs_np["cla"], dtype=torch.float32, device=device)
    valid_mask = torch.tensor(inputs_np["valid_mask"], dtype=torch.bool, device=device)
    fpoc = torch.tensor(inputs_np["fpoc"], dtype=torch.float32, device=device)
    nfpc = torch.tensor(inputs_np["nfpc"], dtype=torch.long, device=device)

    denoise_steps = int(
        denoise_steps_override
        if denoise_steps_override is not None
        else bundle.config.get("denoise_steps", 3)
    )

    pred_pos_n, pred_ang, pred_siz_n = bundle.model(
        pos, ang, siz, cla, valid_mask, fpoc, nfpc, denoise_steps=denoise_steps
    )

    pos_mean = torch.tensor(bundle.stats["pos_mean"], dtype=torch.float32, device=device)
    pos_std = torch.tensor(bundle.stats["pos_std"], dtype=torch.float32, device=device)
    siz_mean = torch.tensor(bundle.stats["siz_mean"], dtype=torch.float32, device=device)
    siz_std = torch.tensor(bundle.stats["siz_std"], dtype=torch.float32, device=device)

    pred_pos = denorm_pos_t(pred_pos_n, pos_mean, pos_std)[0].cpu().numpy()
    pred_ang = pred_ang[0].cpu().numpy()
    pred_siz = denorm_siz_t(pred_siz_n, siz_mean, siz_std)[0].cpu().numpy()

    pred_siz = np.maximum(pred_siz, 0.02)

    pred_pos = np.asarray(
        [project_inside_room(pred_pos[i], pred_siz[i], room) for i in range(pred_pos.shape[0])],
        dtype=np.float32,
    )

    pred_ang_snapped = np.zeros_like(pred_ang)
    for i in range(pred_ang.shape[0]):
        yaw = vec2_to_yaw_deg(pred_ang[i])
        yaw = snap_angle_deg(yaw, snap_deg)
        pred_ang_snapped[i] = yaw_deg_to_vec2(yaw)

    pred_pos = resolve_collisions_greedy(
        pred_pos_xy=pred_pos,
        siz_xy=pred_siz,
        room=room,
        iterations=120,
        shift_scale=0.02,
    )

    # z-размер модель не предсказывает, поэтому сохраняем исходный вертикальный размер объекта.
    pred_siz_xy = init_siz_xy.copy()

    return init_pos_xy, init_ang_xy, init_siz_xy, pred_pos, pred_ang_snapped, pred_siz_xy


# ============================================================
# CLI
# ============================================================

def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Инференс layout_refiner на реальных данных")

    p.add_argument("--checkpoint", required=True, help="Путь к best.pt")
    p.add_argument("--room", required=True, help="Путь к room.json")
    p.add_argument("--objects", required=True, help="Путь к objects.json / objects.v1.json")
    p.add_argument("--out", required=True, help="Путь к placement JSON")

    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--init-mode", default=None, choices=["random", "relaxed"], help="Режим начальной расстановки")
    p.add_argument("--mode", default=None, choices=["random", "relaxed"], help=argparse.SUPPRESS)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--class-map", default=None, help="JSON {category: class_id}")
    p.add_argument("--denoise-steps", type=int, default=None, help="Переопределить число refine-итераций")
    p.add_argument("--snap-angle-deg", type=float, default=90.0, help="Шаг дискретизации угла, 0 = не снапить")
    p.add_argument("--debug-image", default=None, help="PNG для 2D debug-визуализации")
    p.add_argument("--print-summary", action="store_true")

    return p


def main() -> None:
    args = build_cli().parse_args()

    init_mode = args.init_mode or args.mode or "relaxed"

    seed_everything(int(args.seed))
    device = pick_device(args.device)

    room_json = load_json(args.room)
    objects_json = load_json(args.objects)

    room = parse_room(room_json)
    objects = parse_objects(objects_json)
    class_map = load_class_map(args.class_map)

    bundle = load_checkpoint_bundle(args.checkpoint, device)

    if args.print_summary:
        print(f"torch: {torch.__version__}")
        print(f"device: {device}")
        print(f"checkpoint: {Path(args.checkpoint).expanduser().resolve()}")
        print(f"class_dim: {bundle.class_dim}")
        print(f"room_id: {room.room_id}")
        print(f"num_objects: {len(objects)}")
        print(f"polygon_points: {room.polygon_xy.shape[0]}")
        print(f"bbox_min_xy: {room.bbox_min_xy.tolist()}")
        print(f"bbox_max_xy: {room.bbox_max_xy.tolist()}")
        print(f"class_map_size: {len(class_map)}")
        print(f"init_mode: {init_mode}")

    init_pos_xy, init_ang_xy, init_siz_xy, pred_pos_xy, pred_ang_xy, pred_siz_xy = run_inference(
        bundle=bundle,
        room=room,
        objects=objects,
        class_map=class_map,
        init_mode=init_mode,
        device=device,
        denoise_steps_override=args.denoise_steps,
        snap_deg=float(args.snap_angle_deg),
    )

    placement = export_placement_v1(
        room=room,
        objects=objects,
        pred_pos_xy=pred_pos_xy,
        pred_ang_xy=pred_ang_xy,
        pred_siz_xy=pred_siz_xy,
        init_mode=init_mode,
        placer_name="layout_refiner",
    )
    save_json(args.out, placement)

    if args.debug_image:
        draw_debug_layout(
            room=room,
            objects=objects,
            init_pos_xy=init_pos_xy,
            pred_pos_xy=pred_pos_xy,
            pred_siz_xy=pred_siz_xy,
            debug_image_path=args.debug_image,
        )

    print(f"OK: placement saved -> {Path(args.out).expanduser().resolve()}")
    if args.debug_image:
        print(f"OK: debug image saved -> {Path(args.debug_image).expanduser().resolve()}")


if __name__ == "__main__":
    main()