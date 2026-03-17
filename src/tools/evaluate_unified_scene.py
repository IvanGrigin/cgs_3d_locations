#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_unified_scene.py

Оценка унифицированной сцены scene.v1 по метрикам проекта.

Поддерживаемые метрики:
1. Процент коллизий между объектами
2. Все ли предметы внутри комнаты
3. Свободны ли двери и окна с учётом зоны доступа 1 метр:
   - для двери: вглубь комнаты на 1 м ничего не должно быть
   - для окна: вглубь комнаты на 1 м ничего не должно быть,
     но объект под окном допустим, если его верх не выше нижней кромки окна
4. Есть ли путь от двери:
   - ко всем объектам (BFS, A*)
   - ко всем окнам (BFS, A*)
   Путь строится по осям XY, как "змейка".
   Человек моделируется квадратом 0.6 м x 0.6 м.
5. Выполнение условий из промпта
   - ПОКА ЗАГЛУШКА: TODO
6. Совпадение стиля помещения со стилем из промпта
7. Полное время генерации
8. Максимальный пустой квадрат на полу
9. Максимальный пустой прямоугольник на полу
10. Итоговая оценка

Поддерживаемые варианты room:
A) Развёрнутый:
    {
      "x_min": ..., "x_max": ...,
      "y_min": ..., "y_max": ...,
      "z_min": ..., "z_max": ...,
      "openings": [{"kind": "door|window", "aabb": {...}}, ...]
    }

B) Геометрический:
    {
      "ceiling_height": 2.8,
      "floor_polygon": [{"x":0,"y":0}, ...] или [[x,y], ...],
      "walls": [{"id":"w0","from_vertex":0,"to_vertex":1}, ...],
      "doors": [{"wall_id":"w0","s":..., "width":..., "z0":..., "height":...}, ...],
      "windows": [{"wall_id":"w2","s":..., "width":..., "z0":..., "height":...}, ...]
    }
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


EPS = 1e-9
UUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")


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

def as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def as_str(x: Any, default: str = "") -> str:
    if x is None:
        return default
    return str(x)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    if abs(b) < EPS:
        return default
    return a / b


def normalize_score(x: float) -> float:
    return clamp(x, 0.0, 1.0)


def interval_overlap_len(a1: float, a2: float, b1: float, b2: float) -> float:
    return max(0.0, min(a2, b2) - max(a1, b1))


def center_from_aabb(aabb: Dict[str, Any]) -> Tuple[float, float, float]:
    return (
        0.5 * (as_float(aabb["x_min"]) + as_float(aabb["x_max"])),
        0.5 * (as_float(aabb["y_min"]) + as_float(aabb["y_max"])),
        0.5 * (as_float(aabb["z_min"]) + as_float(aabb["z_max"])),
    )


def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def extract_uuid_from_path(path: Optional[str]) -> Optional[str]:
    if not isinstance(path, str) or not path:
        return None
    m = UUID_RE.search(path)
    return m.group(0) if m else None


def rects_intersect_xy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
    return (
        interval_overlap_len(a[0], a[1], b[0], b[1]) > 1e-9 and
        interval_overlap_len(a[2], a[3], b[2], b[3]) > 1e-9
    )


# ============================================================
# Данные сцены
# ============================================================

@dataclass
class Opening:
    kind: str
    name: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float = 0.0
    z_max: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def rect_xy(self) -> Tuple[float, float, float, float]:
        return (self.x_min, self.x_max, self.y_min, self.y_max)

    @property
    def center_xy(self) -> Tuple[float, float]:
        return ((self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0)


@dataclass
class Room:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    floor_polygon: Optional[List[Tuple[float, float]]] = None
    openings: List[Opening] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def center_xy(self) -> Tuple[float, float]:
        return ((self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0)


@dataclass
class Placement:
    idx: int
    obj_id: str
    name: str
    category: str
    class_name: str
    position_m: Tuple[float, float, float]
    size_m: Tuple[float, float, float]
    rotation_deg: float
    yaw_deg: float
    yaw_rad: float
    aabb: Dict[str, float]
    mount_type: Optional[str]
    wall_contact_side: Optional[str]
    constraints: Dict[str, Any]
    asset: Dict[str, Any]
    source: Dict[str, Any]
    meta: Dict[str, Any]
    color: List[float]

    @property
    def x_min(self) -> float:
        return self.aabb["x_min"]

    @property
    def x_max(self) -> float:
        return self.aabb["x_max"]

    @property
    def y_min(self) -> float:
        return self.aabb["y_min"]

    @property
    def y_max(self) -> float:
        return self.aabb["y_max"]

    @property
    def z_min(self) -> float:
        return self.aabb["z_min"]

    @property
    def z_max(self) -> float:
        return self.aabb["z_max"]

    @property
    def center_xy(self) -> Tuple[float, float]:
        return ((self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0)

    @property
    def area_xy(self) -> float:
        return max(0.0, self.x_max - self.x_min) * max(0.0, self.y_max - self.y_min)

    def is_ceiling_object(self) -> bool:
        if self.mount_type == "ceiling":
            return True
        return self.class_name in {"ceiling_lamp", "pendant_lamp", "ceiling lamp"}

    def is_floor_object(self) -> bool:
        if self.mount_type == "floor":
            return True
        if self.mount_type == "ceiling":
            return False
        return not self.is_ceiling_object()

    def style(self) -> Optional[str]:
        v = self.meta.get("style")
        if isinstance(v, str) and v.strip():
            return v.strip()
        return None

    def model_id(self) -> Optional[str]:
        v = self.asset.get("model_id")
        if isinstance(v, str) and v.strip():
            return v.strip()
        return extract_uuid_from_path(self.asset.get("mesh_path"))


# ============================================================
# Parsing scene.v1
# ============================================================

def infer_class_name(name: str, category: str, source: Dict[str, Any]) -> str:
    if isinstance(source.get("server_class_name"), str) and source["server_class_name"].strip():
        return source["server_class_name"].strip()

    s = f"{name} {category}".lower()
    if "bed" in s:
        return "double_bed"
    if "nightstand" in s or "side table" in s:
        return "nightstand"
    if "wardrobe" in s:
        return "wardrobe"
    if "cabinet" in s or "drawer chest" in s:
        return "cabinet"
    if "lamp" in s:
        return "ceiling_lamp"
    if "desk" in s:
        return "desk"
    if "chair" in s or "armchair" in s:
        return "chair"
    if "sofa" in s:
        return "sofa"
    return category.lower().replace(" ", "_")


def parse_floor_polygon(room_raw: Dict[str, Any]) -> Optional[List[Tuple[float, float]]]:
    raw = room_raw.get("floor_polygon")
    if not isinstance(raw, list):
        return None

    pts: List[Tuple[float, float]] = []
    for pt in raw:
        if isinstance(pt, dict) and "x" in pt and "y" in pt:
            pts.append((as_float(pt["x"]), as_float(pt["y"])))
        elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
            pts.append((as_float(pt[0]), as_float(pt[1])))

    return pts or None


def parse_bounds_from_polygon(floor_polygon: Optional[List[Tuple[float, float]]]) -> Optional[Tuple[float, float, float, float]]:
    if not floor_polygon:
        return None
    xs = [p[0] for p in floor_polygon]
    ys = [p[1] for p in floor_polygon]
    return (min(xs), max(xs), min(ys), max(ys))


def build_wall_map(room_raw: Dict[str, Any], floor_polygon: Optional[List[Tuple[float, float]]]) -> Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]]:
    out: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]] = {}
    walls = room_raw.get("walls")
    if not isinstance(walls, list) or not floor_polygon:
        return out

    for wall in walls:
        if not isinstance(wall, dict):
            continue
        wall_id = as_str(wall.get("id"))
        i0 = wall.get("from_vertex")
        i1 = wall.get("to_vertex")
        if not wall_id:
            continue
        if not isinstance(i0, int) or not isinstance(i1, int):
            continue
        if not (0 <= i0 < len(floor_polygon) and 0 <= i1 < len(floor_polygon)):
            continue
        out[wall_id] = (floor_polygon[i0], floor_polygon[i1])

    return out


def opening_rect_from_wall_segment(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    s: float,
    width: float,
    thickness: float = 0.12,
) -> Tuple[float, float, float, float]:
    vx = p1[0] - p0[0]
    vy = p1[1] - p0[1]
    wall_len = math.hypot(vx, vy)
    if wall_len < EPS:
        return (p0[0], p0[0], p0[1], p0[1])

    ux = vx / wall_len
    uy = vy / wall_len
    nx = -uy
    ny = ux

    q0x = p0[0] + ux * s
    q0y = p0[1] + uy * s
    q1x = p0[0] + ux * (s + width)
    q1y = p0[1] + uy * (s + width)

    pts = [
        (q0x + nx * thickness / 2.0, q0y + ny * thickness / 2.0),
        (q0x - nx * thickness / 2.0, q0y - ny * thickness / 2.0),
        (q1x + nx * thickness / 2.0, q1y + ny * thickness / 2.0),
        (q1x - nx * thickness / 2.0, q1y - ny * thickness / 2.0),
    ]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), max(xs), min(ys), max(ys))


def parse_openings(room_raw: Dict[str, Any], floor_polygon: Optional[List[Tuple[float, float]]]) -> List[Opening]:
    out: List[Opening] = []

    openings = room_raw.get("openings")
    if isinstance(openings, list):
        for i, obj in enumerate(openings):
            if not isinstance(obj, dict):
                continue
            kind = as_str(obj.get("kind"), "unknown").lower()
            name = as_str(obj.get("name"), f"{kind}_{i+1}")
            aabb = obj.get("aabb")
            if isinstance(aabb, dict):
                out.append(
                    Opening(
                        kind=kind,
                        name=name,
                        x_min=as_float(aabb.get("x_min")),
                        x_max=as_float(aabb.get("x_max")),
                        y_min=as_float(aabb.get("y_min")),
                        y_max=as_float(aabb.get("y_max")),
                        z_min=as_float(aabb.get("z_min")),
                        z_max=as_float(aabb.get("z_max")),
                        meta={k: v for k, v in obj.items() if k not in {"kind", "name", "aabb"}},
                    )
                )

    wall_map = build_wall_map(room_raw, floor_polygon)

    for key, kind in [("doors", "door"), ("windows", "window")]:
        arr = room_raw.get(key)
        if not isinstance(arr, list):
            continue

        for i, obj in enumerate(arr):
            if not isinstance(obj, dict):
                continue

            wall_id = as_str(obj.get("wall_id"))
            wall_seg = wall_map.get(wall_id)
            if wall_seg is None:
                continue

            s = as_float(obj.get("s"))
            width = as_float(obj.get("width"))
            z0 = as_float(obj.get("z0"))
            height = as_float(obj.get("height"))

            x_min, x_max, y_min, y_max = opening_rect_from_wall_segment(
                wall_seg[0], wall_seg[1], s=s, width=width
            )

            name = as_str(obj.get("id"), f"{kind}_{i+1}")
            out.append(
                Opening(
                    kind=kind,
                    name=name,
                    x_min=x_min,
                    x_max=x_max,
                    y_min=y_min,
                    y_max=y_max,
                    z_min=z0,
                    z_max=z0 + height,
                    meta={k: v for k, v in obj.items() if k not in {"id", "wall_id", "s", "width", "z0", "height"}},
                )
            )

    return out


def parse_room(scene: Dict[str, Any]) -> Room:
    room_raw = scene.get("room")
    if not isinstance(room_raw, dict):
        raise ValueError("scene.v1 должен содержать объект room")

    floor_polygon = parse_floor_polygon(room_raw)

    x_min = room_raw.get("x_min")
    x_max = room_raw.get("x_max")
    y_min = room_raw.get("y_min")
    y_max = room_raw.get("y_max")

    if any(v is None for v in (x_min, x_max, y_min, y_max)):
        bounds = parse_bounds_from_polygon(floor_polygon)
        if bounds is None:
            x_min = y_min = 0.0
            x_max = y_max = 0.0
        else:
            x_min, x_max, y_min, y_max = bounds

    z_min = room_raw.get("z_min")
    z_max = room_raw.get("z_max")
    ceiling_height = room_raw.get("ceiling_height")

    if z_min is None:
        z_min = 0.0
    if z_max is None:
        if ceiling_height is not None:
            z_max = as_float(z_min) + as_float(ceiling_height)
        else:
            z_max = 0.0

    openings = parse_openings(room_raw, floor_polygon)

    return Room(
        x_min=as_float(x_min),
        x_max=as_float(x_max),
        y_min=as_float(y_min),
        y_max=as_float(y_max),
        z_min=as_float(z_min),
        z_max=as_float(z_max),
        floor_polygon=floor_polygon,
        openings=openings,
        meta={k: v for k, v in room_raw.items() if k not in {
            "x_min", "x_max", "y_min", "y_max", "z_min", "z_max",
            "ceiling_height", "floor_polygon", "openings", "doors", "windows", "walls"
        }},
    )


def parse_placements(scene: Dict[str, Any]) -> List[Placement]:
    placements_raw = scene.get("placements")
    if not isinstance(placements_raw, list):
        raise ValueError("scene.v1 должен содержать список placements")

    out: List[Placement] = []
    for i, obj in enumerate(placements_raw):
        if not isinstance(obj, dict):
            continue

        aabb = obj.get("aabb")
        if not isinstance(aabb, dict):
            continue

        source = obj.get("source") if isinstance(obj.get("source"), dict) else {}
        meta = obj.get("meta") if isinstance(obj.get("meta"), dict) else {}
        asset = obj.get("asset") if isinstance(obj.get("asset"), dict) else {}
        constraints = obj.get("constraints") if isinstance(obj.get("constraints"), dict) else {}
        color = obj.get("color") if isinstance(obj.get("color"), list) else [0.7, 0.7, 0.7]

        name = as_str(obj.get("name"), f"object_{i+1}")
        category = as_str(obj.get("category"), name)
        class_name = infer_class_name(name, category, source)

        if "model_id" not in asset or not asset.get("model_id"):
            model_id = extract_uuid_from_path(asset.get("mesh_path"))
            if model_id:
                asset = dict(asset)
                asset["model_id"] = model_id

        out.append(
            Placement(
                idx=i,
                obj_id=as_str(obj.get("id"), f"obj_{i+1:04d}"),
                name=name,
                category=category,
                class_name=class_name,
                position_m=tuple(obj.get("position_m", center_from_aabb(aabb))),
                size_m=tuple(obj.get("size_m", [
                    as_float(aabb.get("x_max")) - as_float(aabb.get("x_min")),
                    as_float(aabb.get("y_max")) - as_float(aabb.get("y_min")),
                    as_float(aabb.get("z_max")) - as_float(aabb.get("z_min")),
                ])),
                rotation_deg=as_float(obj.get("rotation_deg")),
                yaw_deg=as_float(obj.get("yaw_deg")),
                yaw_rad=as_float(obj.get("yaw_rad")),
                aabb={
                    "x_min": as_float(aabb.get("x_min")),
                    "x_max": as_float(aabb.get("x_max")),
                    "y_min": as_float(aabb.get("y_min")),
                    "y_max": as_float(aabb.get("y_max")),
                    "z_min": as_float(aabb.get("z_min")),
                    "z_max": as_float(aabb.get("z_max")),
                },
                mount_type=obj.get("mount_type"),
                wall_contact_side=obj.get("wall_contact_side"),
                constraints=constraints,
                asset=asset,
                source=source,
                meta=meta,
                color=[as_float(x, 0.7) for x in color[:4]],
            )
        )
    return out


# ============================================================
# model_info styles
# ============================================================

def load_model_info(model_info_path: str | Path) -> Dict[str, Dict[str, Any]]:
    data = load_json(model_info_path)
    if not isinstance(data, list):
        raise ValueError("model_info.json должен содержать список объектов")

    out: Dict[str, Dict[str, Any]] = {}
    for row in data:
        if not isinstance(row, dict):
            continue
        model_id = row.get("model_id")
        if isinstance(model_id, str) and model_id.strip():
            out[model_id.strip()] = {
                "style": row.get("style"),
                "super_category": row.get("super-category") or row.get("super_category"),
                "category": row.get("category"),
                "theme": row.get("theme"),
                "material": row.get("material"),
            }
    return out


def resolve_object_style(p: Placement, model_info: Dict[str, Dict[str, Any]]) -> str:
    local_style = p.style()
    if local_style:
        return local_style

    model_id = p.model_id()
    if model_id and model_id in model_info:
        style = model_info[model_id].get("style")
        if isinstance(style, str) and style.strip():
            return style.strip()

    return "unknown"


def object_style_weight(p: Placement) -> float:
    c = p.class_name
    if c in {"double_bed", "single_bed", "bed", "sofa", "wardrobe", "desk"}:
        return 3.0
    if c in {"cabinet", "nightstand", "table"}:
        return 2.0
    if c in {"ceiling_lamp", "chair"}:
        return 1.0
    return 1.5


# ============================================================
# Геометрия / коллизии
# ============================================================

def rect_intersection_area_xy(a: Placement, b: Placement) -> float:
    ox = interval_overlap_len(a.x_min, a.x_max, b.x_min, b.x_max)
    oy = interval_overlap_len(a.y_min, a.y_max, b.y_min, b.y_max)
    return ox * oy


def box_intersection_volume(a: Placement, b: Placement) -> float:
    ox = interval_overlap_len(a.x_min, a.x_max, b.x_min, b.x_max)
    oy = interval_overlap_len(a.y_min, a.y_max, b.y_min, b.y_max)
    oz = interval_overlap_len(a.z_min, a.z_max, b.z_min, b.z_max)
    return ox * oy * oz


def should_ignore_collision(a: Placement, b: Placement) -> bool:
    if (a.is_ceiling_object() and b.is_floor_object()) or (b.is_ceiling_object() and a.is_floor_object()):
        return box_intersection_volume(a, b) <= 1e-6
    return False


# ============================================================
# Навигационная сетка
# ============================================================

class Grid:
    def __init__(self, room: Room, cell: float):
        self.room = room
        self.cell = cell
        self.nx = max(1, int(math.ceil(max(room.width, EPS) / cell)))
        self.ny = max(1, int(math.ceil(max(room.height, EPS) / cell)))
        self.valid = [[True for _ in range(self.ny)] for _ in range(self.nx)]

    def cell_center(self, ix: int, iy: int) -> Tuple[float, float]:
        x = self.room.x_min + (ix + 0.5) * self.cell
        y = self.room.y_min + (iy + 0.5) * self.cell
        return x, y

    def point_to_cell(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        if not (self.room.x_min <= x <= self.room.x_max and self.room.y_min <= y <= self.room.y_max):
            return None
        ix = int((x - self.room.x_min) / self.cell)
        iy = int((y - self.room.y_min) / self.cell)
        ix = min(max(ix, 0), self.nx - 1)
        iy = min(max(iy, 0), self.ny - 1)
        return ix, iy

    def is_valid(self, ix: int, iy: int) -> bool:
        return 0 <= ix < self.nx and 0 <= iy < self.ny and self.valid[ix][iy]


def build_navigation_grid(room: Room, placements: List[Placement], person_size: float, cell: float) -> Grid:
    """
    Клетка валидна, если квадрат person_size x person_size с центром в центре клетки:
    1) полностью внутри комнаты
    2) не пересекает ни один floor object
    """
    grid = Grid(room, cell)
    half = person_size / 2.0
    floor_rects = [(p.x_min, p.x_max, p.y_min, p.y_max) for p in placements if p.is_floor_object()]

    for ix in range(grid.nx):
        for iy in range(grid.ny):
            cx, cy = grid.cell_center(ix, iy)
            person_rect = (cx - half, cx + half, cy - half, cy + half)

            inside_room = (
                person_rect[0] >= room.x_min - EPS and
                person_rect[1] <= room.x_max + EPS and
                person_rect[2] >= room.y_min - EPS and
                person_rect[3] <= room.y_max + EPS
            )
            if not inside_room:
                grid.valid[ix][iy] = False
                continue

            ok = True
            for rect in floor_rects:
                if rects_intersect_xy(person_rect, rect):
                    ok = False
                    break
            grid.valid[ix][iy] = ok

    return grid


def bfs_reachable(grid: Grid, start: Tuple[int, int]) -> List[List[bool]]:
    visited = [[False for _ in range(grid.ny)] for _ in range(grid.nx)]
    sx, sy = start
    if not grid.is_valid(sx, sy):
        return visited

    q = [(sx, sy)]
    visited[sx][sy] = True
    head = 0
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    while head < len(q):
        x, y = q[head]
        head += 1
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid.nx and 0 <= ny < grid.ny and not visited[nx][ny] and grid.is_valid(nx, ny):
                visited[nx][ny] = True
                q.append((nx, ny))
    return visited


def astar_exists(grid: Grid, start: Tuple[int, int], goal: Tuple[int, int]) -> bool:
    if not grid.is_valid(*start) or not grid.is_valid(*goal):
        return False

    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    open_heap: List[Tuple[float, float, Tuple[int, int]]] = []
    heapq.heappush(open_heap, (heuristic(start, goal), 0.0, start))
    best_g = {start: 0.0}

    while open_heap:
        _, g, cur = heapq.heappop(open_heap)
        if cur == goal:
            return True

        if g > best_g.get(cur, float("inf")) + EPS:
            continue

        x, y = cur
        for dx, dy in dirs:
            nxt = (x + dx, y + dy)
            if not (0 <= nxt[0] < grid.nx and 0 <= nxt[1] < grid.ny):
                continue
            if not grid.is_valid(*nxt):
                continue

            ng = g + 1.0
            if ng + EPS < best_g.get(nxt, float("inf")):
                best_g[nxt] = ng
                f = ng + heuristic(nxt, goal)
                heapq.heappush(open_heap, (f, ng, nxt))

    return False


# ============================================================
# Доступ к объектам / окнам
# ============================================================

def get_door_openings(room: Room) -> List[Opening]:
    return [o for o in room.openings if o.kind == "door"]


def get_window_openings(room: Room) -> List[Opening]:
    return [o for o in room.openings if o.kind == "window"]


def opening_wall_side(room: Room, opening: Opening, tol: float = 0.2) -> Optional[str]:
    if abs(opening.y_min - room.y_min) <= tol:
        return "bottom"
    if abs(opening.y_max - room.y_max) <= tol:
        return "top"
    if abs(opening.x_min - room.x_min) <= tol:
        return "left"
    if abs(opening.x_max - room.x_max) <= tol:
        return "right"
    return None


def default_entry_points(room: Room, person_size: float = 0.6, gap: float = 0.05) -> List[Tuple[float, float]]:
    doors = get_door_openings(room)
    if not doors:
        return [(room.x_min + person_size / 2.0 + gap, (room.y_min + room.y_max) / 2.0)]

    pts: List[Tuple[float, float]] = []
    shift = person_size / 2.0 + gap
    for d in doors:
        cx, cy = d.center_xy
        side = opening_wall_side(room, d)

        if side == "bottom":
            pts.append((cx, room.y_min + shift))
        elif side == "top":
            pts.append((cx, room.y_max - shift))
        elif side == "left":
            pts.append((room.x_min + shift, cy))
        elif side == "right":
            pts.append((room.x_max - shift, cy))
        else:
            pts.extend([
                (cx, cy + shift),
                (cx, cy - shift),
                (cx + shift, cy),
                (cx - shift, cy),
            ])

    return pts


def object_access_points(p: Placement, person_size: float = 0.6, gap: float = 0.01) -> List[Tuple[float, float]]:
    half = person_size / 2.0
    cx = (p.x_min + p.x_max) / 2.0
    cy = (p.y_min + p.y_max) / 2.0
    return [
        (p.x_min - half - gap, cy),
        (p.x_max + half + gap, cy),
        (cx, p.y_min - half - gap),
        (cx, p.y_max + half + gap),
    ]


def window_access_points(w: Opening, room: Room, person_size: float = 0.6, gap: float = 0.01) -> List[Tuple[float, float]]:
    half = person_size / 2.0
    cx, cy = w.center_xy
    side = opening_wall_side(room, w)

    if side == "bottom":
        return [(cx, room.y_min + half + gap)]
    if side == "top":
        return [(cx, room.y_max - half - gap)]
    if side == "left":
        return [(room.x_min + half + gap, cy)]
    if side == "right":
        return [(room.x_max - half - gap, cy)]

    return [
        (cx, w.y_min - half - gap),
        (cx, w.y_max + half + gap),
        (w.x_min - half - gap, cy),
        (w.x_max + half + gap, cy),
    ]


def first_valid_start_cell(grid: Grid, start_points: List[Tuple[float, float]]) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[float, float]]]:
    for pt in start_points:
        cell = grid.point_to_cell(*pt)
        if cell is not None and grid.is_valid(*cell):
            return cell, pt
    return None, None


# ============================================================
# Метрики
# ============================================================

def metric_collisions(room: Room, placements: List[Placement]) -> Dict[str, Any]:
    n = len(placements)
    total_pairs = n * (n - 1) // 2
    collision_pairs = []
    overlap_area_total = 0.0
    overlap_volume_total = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            a = placements[i]
            b = placements[j]
            if should_ignore_collision(a, b):
                continue

            volume = box_intersection_volume(a, b)
            if volume > 1e-6:
                area = rect_intersection_area_xy(a, b)
                collision_pairs.append({
                    "i": a.idx,
                    "j": b.idx,
                    "id_i": a.obj_id,
                    "id_j": b.obj_id,
                    "name_i": a.name,
                    "name_j": b.name,
                    "intersection_area_xy": area,
                    "intersection_volume": volume,
                })
                overlap_area_total += area
                overlap_volume_total += volume

    pair_rate = safe_div(len(collision_pairs), total_pairs, 0.0)
    overlap_area_norm = safe_div(overlap_area_total, room.area, 0.0)

    return {
        "scene_collision_free": len(collision_pairs) == 0,
        "collision_pair_count": len(collision_pairs),
        "collision_pair_rate": pair_rate,
        "overlap_area_total": overlap_area_total,
        "overlap_area_norm_by_room": overlap_area_norm,
        "overlap_volume_total": overlap_volume_total,
        "pairs": collision_pairs,
        "score": normalize_score(1.0 - pair_rate),
    }


def metric_in_room(room: Room, placements: List[Placement], tol: float = 0.0) -> Dict[str, Any]:
    inside = 0
    violations = []

    for p in placements:
        ok = (
            p.x_min >= room.x_min - tol and
            p.x_max <= room.x_max + tol and
            p.y_min >= room.y_min - tol and
            p.y_max <= room.y_max + tol and
            p.z_min >= room.z_min - tol and
            p.z_max <= room.z_max + tol
        )
        if ok:
            inside += 1
        else:
            violations.append({
                "idx": p.idx,
                "id": p.obj_id,
                "name": p.name,
                "aabb": p.aabb,
            })

    rate = safe_div(inside, len(placements), 1.0)
    return {
        "all_inside_room": inside == len(placements),
        "inside_count": inside,
        "total_count": len(placements),
        "inside_rate": rate,
        "violations": violations,
        "score": rate,
    }


def opening_clearance_zone(room: Room, opening: Opening, clearance: float = 1.0) -> Optional[Tuple[float, float, float, float]]:
    side = opening_wall_side(room, opening)
    if side == "bottom":
        return (
            max(room.x_min, opening.x_min),
            min(room.x_max, opening.x_max),
            room.y_min,
            min(room.y_max, room.y_min + clearance),
        )
    if side == "top":
        return (
            max(room.x_min, opening.x_min),
            min(room.x_max, opening.x_max),
            max(room.y_min, room.y_max - clearance),
            room.y_max,
        )
    if side == "left":
        return (
            room.x_min,
            min(room.x_max, room.x_min + clearance),
            max(room.y_min, opening.y_min),
            min(room.y_max, opening.y_max),
        )
    if side == "right":
        return (
            max(room.x_min, room.x_max - clearance),
            room.x_max,
            max(room.y_min, opening.y_min),
            min(room.y_max, opening.y_max),
        )
    return None


def metric_openings_block(room: Room, placements: List[Placement], clearance_distance: float = 1.0) -> Dict[str, Any]:
    door_violations = []
    window_violations = []

    floor_objects = [p for p in placements if p.is_floor_object()]

    for opening in room.openings:
        zone = opening_clearance_zone(room, opening, clearance=clearance_distance)
        if zone is None:
            continue

        for p in floor_objects:
            if not rects_intersect_xy(zone, (p.x_min, p.x_max, p.y_min, p.y_max)):
                continue

            if opening.kind == "window" and p.z_max <= opening.z_min + 1e-6:
                # Под окном допустимо
                continue

            rec = {
                "opening_name": opening.name,
                "opening_kind": opening.kind,
                "clearance_distance_m": clearance_distance,
                "clearance_zone_xy": {
                    "x_min": zone[0],
                    "x_max": zone[1],
                    "y_min": zone[2],
                    "y_max": zone[3],
                },
                "object_idx": p.idx,
                "object_id": p.obj_id,
                "object_name": p.name,
                "object_z_max": p.z_max,
            }

            if opening.kind == "door":
                door_violations.append(rec)
            elif opening.kind == "window":
                window_violations.append(rec)

    doors = get_door_openings(room)
    windows = get_window_openings(room)

    door_free = (len(door_violations) == 0) if doors else True
    window_free = (len(window_violations) == 0) if windows else True

    door_score = 1.0 if door_free else 0.0
    window_score = 1.0 if window_free else 0.0
    combined = 0.6 * door_score + 0.4 * window_score

    return {
        "clearance_distance_m": clearance_distance,
        "door_free": door_free,
        "window_free": window_free,
        "door_violations_count": len(door_violations),
        "window_violations_count": len(window_violations),
        "door_violations": door_violations,
        "window_violations": window_violations,
        "score": combined,
    }


def metric_accessibility_objects_bfs(room: Room, placements: List[Placement], person_size: float = 0.6, cell: float = 0.08) -> Dict[str, Any]:
    grid = build_navigation_grid(room, placements, person_size=person_size, cell=cell)
    start_points = default_entry_points(room, person_size=person_size)
    start_cell, start_point = first_valid_start_cell(grid, start_points)

    if start_cell is None:
        return {
            "algorithm": "bfs_grid_objects",
            "person_size": person_size,
            "start_point": None,
            "start_candidates": start_points,
            "reachable_rate": 0.0,
            "targets": [],
            "score": 0.0,
            "error": "no_valid_start_cell_near_door",
        }

    visited = bfs_reachable(grid, start_cell)
    targets = []
    reachable_weight = 0.0
    total_weight = 0.0

    for p in placements:
        if p.is_ceiling_object():
            continue

        points = object_access_points(p, person_size=person_size)
        reachable = False
        for qx, qy in points:
            cell_id = grid.point_to_cell(qx, qy)
            if cell_id is None:
                continue
            if visited[cell_id[0]][cell_id[1]]:
                reachable = True
                break

        weight = 1.0
        if p.class_name in {"double_bed", "single_bed", "bed"}:
            weight = 2.0
        elif p.class_name in {"wardrobe", "cabinet", "desk"}:
            weight = 1.5

        total_weight += weight
        if reachable:
            reachable_weight += weight

        targets.append({
            "idx": p.idx,
            "id": p.obj_id,
            "name": p.name,
            "class_name": p.class_name,
            "reachable": reachable,
            "weight": weight,
        })

    rate = safe_div(reachable_weight, total_weight, 1.0)
    return {
        "algorithm": "bfs_grid_objects",
        "person_size": person_size,
        "start_point": start_point,
        "start_candidates": start_points,
        "reachable_rate": rate,
        "targets": targets,
        "score": rate,
    }


def metric_accessibility_objects_astar(room: Room, placements: List[Placement], person_size: float = 0.6, cell: float = 0.08) -> Dict[str, Any]:
    grid = build_navigation_grid(room, placements, person_size=person_size, cell=cell)
    start_points = default_entry_points(room, person_size=person_size)
    start_cell, start_point = first_valid_start_cell(grid, start_points)

    if start_cell is None:
        return {
            "algorithm": "astar_grid_objects",
            "person_size": person_size,
            "start_point": None,
            "start_candidates": start_points,
            "reachable_rate": 0.0,
            "targets": [],
            "score": 0.0,
            "error": "no_valid_start_cell_near_door",
        }

    targets = []
    reachable_weight = 0.0
    total_weight = 0.0

    for p in placements:
        if p.is_ceiling_object():
            continue

        points = object_access_points(p, person_size=person_size)
        reachable = False
        chosen_goal = None

        for qx, qy in points:
            goal = grid.point_to_cell(qx, qy)
            if goal is None:
                continue
            if astar_exists(grid, start_cell, goal):
                reachable = True
                chosen_goal = goal
                break

        weight = 1.0
        if p.class_name in {"double_bed", "single_bed", "bed"}:
            weight = 2.0
        elif p.class_name in {"wardrobe", "cabinet", "desk"}:
            weight = 1.5

        total_weight += weight
        if reachable:
            reachable_weight += weight

        targets.append({
            "idx": p.idx,
            "id": p.obj_id,
            "name": p.name,
            "class_name": p.class_name,
            "reachable": reachable,
            "weight": weight,
            "goal_cell": chosen_goal,
        })

    rate = safe_div(reachable_weight, total_weight, 1.0)
    return {
        "algorithm": "astar_grid_objects",
        "person_size": person_size,
        "start_point": start_point,
        "start_candidates": start_points,
        "reachable_rate": rate,
        "targets": targets,
        "score": rate,
    }


def metric_accessibility_windows_bfs(room: Room, placements: List[Placement], person_size: float = 0.6, cell: float = 0.08) -> Dict[str, Any]:
    windows = get_window_openings(room)
    grid = build_navigation_grid(room, placements, person_size=person_size, cell=cell)
    start_points = default_entry_points(room, person_size=person_size)
    start_cell, start_point = first_valid_start_cell(grid, start_points)

    if start_cell is None:
        return {
            "algorithm": "bfs_grid_windows",
            "person_size": person_size,
            "start_point": None,
            "start_candidates": start_points,
            "reachable_rate": 0.0,
            "targets": [],
            "score": 0.0,
            "error": "no_valid_start_cell_near_door",
        }

    visited = bfs_reachable(grid, start_cell)
    targets = []
    reachable = 0

    for w in windows:
        pts = window_access_points(w, room, person_size=person_size)
        ok = False
        for qx, qy in pts:
            cell_id = grid.point_to_cell(qx, qy)
            if cell_id is None:
                continue
            if visited[cell_id[0]][cell_id[1]]:
                ok = True
                break
        if ok:
            reachable += 1
        targets.append({
            "name": w.name,
            "reachable": ok,
        })

    rate = safe_div(reachable, len(windows), 1.0 if not windows else 0.0)
    return {
        "algorithm": "bfs_grid_windows",
        "person_size": person_size,
        "start_point": start_point,
        "start_candidates": start_points,
        "reachable_rate": rate,
        "targets": targets,
        "score": rate,
    }


def metric_accessibility_windows_astar(room: Room, placements: List[Placement], person_size: float = 0.6, cell: float = 0.08) -> Dict[str, Any]:
    windows = get_window_openings(room)
    grid = build_navigation_grid(room, placements, person_size=person_size, cell=cell)
    start_points = default_entry_points(room, person_size=person_size)
    start_cell, start_point = first_valid_start_cell(grid, start_points)

    if start_cell is None:
        return {
            "algorithm": "astar_grid_windows",
            "person_size": person_size,
            "start_point": None,
            "start_candidates": start_points,
            "reachable_rate": 0.0,
            "targets": [],
            "score": 0.0,
            "error": "no_valid_start_cell_near_door",
        }

    targets = []
    reachable = 0

    for w in windows:
        pts = window_access_points(w, room, person_size=person_size)
        ok = False
        chosen_goal = None
        for qx, qy in pts:
            goal = grid.point_to_cell(qx, qy)
            if goal is None:
                continue
            if astar_exists(grid, start_cell, goal):
                ok = True
                chosen_goal = goal
                break
        if ok:
            reachable += 1
        targets.append({
            "name": w.name,
            "reachable": ok,
            "goal_cell": chosen_goal,
        })

    rate = safe_div(reachable, len(windows), 1.0 if not windows else 0.0)
    return {
        "algorithm": "astar_grid_windows",
        "person_size": person_size,
        "start_point": start_point,
        "start_candidates": start_points,
        "reachable_rate": rate,
        "targets": targets,
        "score": rate,
    }


def metric_prompt_constraints_stub() -> Dict[str, Any]:
    return {
        "status": "TODO",
        "score": None,
        "details": "Проверка текстовых условий из промпта пока не реализована.",
    }


def metric_style_match(
    placements: List[Placement],
    model_info: Dict[str, Dict[str, Any]],
    prompt_style: Optional[str],
) -> Dict[str, Any]:
    style_weights: Dict[str, float] = {}
    objects_info = []
    total_weight = 0.0

    for p in placements:
        style = resolve_object_style(p, model_info)
        w = object_style_weight(p)
        style_weights[style] = style_weights.get(style, 0.0) + w
        total_weight += w

        objects_info.append({
            "idx": p.idx,
            "id": p.obj_id,
            "name": p.name,
            "class_name": p.class_name,
            "model_id": p.model_id(),
            "style": style,
            "weight": w,
        })

    dominant_style = None
    dominant_weight = 0.0
    if style_weights:
        dominant_style, dominant_weight = max(style_weights.items(), key=lambda kv: kv[1])

    prompt_style_weight = None
    score = None
    if prompt_style:
        prompt_style_weight = style_weights.get(prompt_style, 0.0)
        score = safe_div(prompt_style_weight, total_weight, 0.0)

    return {
        "prompt_style": prompt_style,
        "dominant_scene_style": dominant_style,
        "dominant_scene_style_weight": dominant_weight,
        "style_weight_distribution": style_weights,
        "style_match_score": score,
        "objects": objects_info,
        "score": score,
    }


def extract_generation_time_sec(scene_meta: Dict[str, Any], override: Optional[float]) -> Dict[str, Any]:
    if override is not None:
        return {
            "generation_time_sec": override,
            "source": "cli_override",
            "score": None,
        }

    candidates = [
        ("generation_time_sec", scene_meta.get("generation_time_sec")),
        ("total_time_sec", scene_meta.get("total_time_sec")),
    ]

    timing = scene_meta.get("timing")
    if isinstance(timing, dict):
        candidates.append(("timing.total_sec", timing.get("total_sec")))
        candidates.append(("timing.generation_sec", timing.get("generation_sec")))

    placement_meta = scene_meta.get("placement_meta")
    if isinstance(placement_meta, dict):
        candidates.append(("placement_meta.generation_time_sec", placement_meta.get("generation_time_sec")))
        candidates.append(("placement_meta.total_time_sec", placement_meta.get("total_time_sec")))
        pm_timing = placement_meta.get("timing")
        if isinstance(pm_timing, dict):
            candidates.append(("placement_meta.timing.total_sec", pm_timing.get("total_sec")))
            candidates.append(("placement_meta.timing.generation_sec", pm_timing.get("generation_sec")))

    for src, value in candidates:
        if value is not None:
            try:
                return {
                    "generation_time_sec": float(value),
                    "source": src,
                    "score": None,
                }
            except Exception:
                pass

    return {
        "generation_time_sec": None,
        "source": "not_found",
        "score": None,
    }


# ============================================================
# Пустое пространство на полу
# ============================================================

def build_floor_occupancy_grid(room: Room, placements: List[Placement], cell: float = 0.05) -> Tuple[List[List[bool]], int, int]:
    """
    True = клетка свободна
    False = занята мебелью
    """
    nx = max(1, int(math.ceil(room.width / cell)))
    ny = max(1, int(math.ceil(room.height / cell)))
    free = [[True for _ in range(ny)] for _ in range(nx)]

    floor_rects = [(p.x_min, p.x_max, p.y_min, p.y_max) for p in placements if p.is_floor_object()]

    for ix in range(nx):
        for iy in range(ny):
            cx = room.x_min + (ix + 0.5) * cell
            cy = room.y_min + (iy + 0.5) * cell
            for rect in floor_rects:
                if rect[0] <= cx <= rect[1] and rect[2] <= cy <= rect[3]:
                    free[ix][iy] = False
                    break

    return free, nx, ny


def metric_max_empty_square(room: Room, placements: List[Placement], cell: float = 0.05) -> Dict[str, Any]:
    free, nx, ny = build_floor_occupancy_grid(room, placements, cell=cell)
    dp = [[0 for _ in range(ny)] for _ in range(nx)]

    best_side_cells = 0
    best_ix = -1
    best_iy = -1

    for ix in range(nx):
        for iy in range(ny):
            if not free[ix][iy]:
                dp[ix][iy] = 0
                continue

            if ix == 0 or iy == 0:
                dp[ix][iy] = 1
            else:
                dp[ix][iy] = 1 + min(
                    dp[ix - 1][iy],
                    dp[ix][iy - 1],
                    dp[ix - 1][iy - 1],
                )

            if dp[ix][iy] > best_side_cells:
                best_side_cells = dp[ix][iy]
                best_ix = ix
                best_iy = iy

    side_m = best_side_cells * cell
    area_m2 = side_m * side_m

    if best_side_cells > 0:
        x_max = room.x_min + (best_ix + 1) * cell
        y_max = room.y_min + (best_iy + 1) * cell
        x_min = x_max - side_m
        y_min = y_max - side_m
    else:
        x_min = x_max = y_min = y_max = 0.0

    return {
        "cell_size_m": cell,
        "side_m": side_m,
        "area_m2": area_m2,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
    }


def largest_rectangle_in_histogram(heights: List[int]) -> Tuple[int, int, int]:
    """
    Возвращает:
    area_cells, left_idx, right_idx_exclusive, height_cells
    """
    stack: List[int] = []
    best_area = 0
    best_left = 0
    best_right = 0
    best_height = 0

    extended = heights + [0]
    for i, h in enumerate(extended):
        while stack and extended[stack[-1]] > h:
            top = stack.pop()
            height = extended[top]
            left = stack[-1] + 1 if stack else 0
            right = i
            area = height * (right - left)
            if area > best_area:
                best_area = area
                best_left = left
                best_right = right
                best_height = height
        stack.append(i)

    return best_area, best_left, best_right, best_height


def metric_max_empty_rectangle(room: Room, placements: List[Placement], cell: float = 0.05) -> Dict[str, Any]:
    free, nx, ny = build_floor_occupancy_grid(room, placements, cell=cell)

    heights = [0] * nx
    best_area_cells = 0
    best_left = 0
    best_right = 0
    best_top_row = 0
    best_height_cells = 0

    for iy in range(ny):
        for ix in range(nx):
            heights[ix] = heights[ix] + 1 if free[ix][iy] else 0

        area_cells, left, right, height_cells = largest_rectangle_in_histogram(heights)
        if area_cells > best_area_cells:
            best_area_cells = area_cells
            best_left = left
            best_right = right
            best_top_row = iy
            best_height_cells = height_cells

    width_cells = best_right - best_left
    width_m = width_cells * cell
    height_m = best_height_cells * cell
    area_m2 = width_m * height_m

    if best_area_cells > 0:
        x_min = room.x_min + best_left * cell
        x_max = room.x_min + best_right * cell
        y_max = room.y_min + (best_top_row + 1) * cell
        y_min = y_max - height_m
    else:
        x_min = x_max = y_min = y_max = 0.0

    return {
        "cell_size_m": cell,
        "width_m": width_m,
        "height_m": height_m,
        "area_m2": area_m2,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
    }


# ============================================================
# Агрегация
# ============================================================

def aggregate_total_score(
    collisions: Dict[str, Any],
    in_room: Dict[str, Any],
    openings: Dict[str, Any],
    access_objects_bfs: Dict[str, Any],
    access_objects_astar: Dict[str, Any],
    access_windows_bfs: Dict[str, Any],
    access_windows_astar: Dict[str, Any],
    prompt_constraints: Dict[str, Any],
    style_match: Dict[str, Any],
    generation_time: Dict[str, Any],
    target_time_sec: float = 60.0,
) -> Dict[str, Any]:
    access_objects_score = min(access_objects_bfs["score"], access_objects_astar["score"])
    access_windows_score = min(access_windows_bfs["score"], access_windows_astar["score"])
    access_score = 0.7 * access_objects_score + 0.3 * access_windows_score

    q_hard = (
        0.30 * (1.0 - collisions["collision_pair_rate"]) +
        0.20 * in_room["score"] +
        0.20 * openings["score"] +
        0.30 * access_score
    )

    style_score = style_match["score"]
    q_semantic = style_score if style_score is not None else None

    t = generation_time["generation_time_sec"]
    q_time = min(1.0, target_time_sec / max(t, EPS)) if t is not None else None

    if q_semantic is None and q_time is None:
        q_total = q_hard
    elif q_semantic is None:
        q_total = 0.95 * q_hard + 0.05 * q_time
    elif q_time is None:
        q_total = 0.80 * q_hard + 0.20 * q_semantic
    else:
        q_total = 0.75 * q_hard + 0.20 * q_semantic + 0.05 * q_time

    if q_total < 0.35:
        verdict = "poor"
    elif q_total < 0.55:
        verdict = "weak"
    elif q_total < 0.72:
        verdict = "acceptable"
    elif q_total < 0.86:
        verdict = "good"
    else:
        verdict = "excellent"

    return {
        "hard_score": q_hard,
        "semantic_score": q_semantic,
        "time_score": q_time,
        "total_score": q_total,
        "verdict": verdict,
    }


# ============================================================
# Главная функция
# ============================================================

def evaluate_scene(
    scene: Dict[str, Any],
    model_info: Dict[str, Dict[str, Any]],
    prompt_style: Optional[str],
    generation_time_override: Optional[float],
) -> Dict[str, Any]:
    if scene.get("schema") != "scene.v1":
        raise ValueError("Скрипт ожидает на вход именно scene.v1")

    room = parse_room(scene)
    placements = parse_placements(scene)
    scene_meta = scene.get("meta") if isinstance(scene.get("meta"), dict) else {}

    collisions = metric_collisions(room, placements)
    in_room = metric_in_room(room, placements)
    openings = metric_openings_block(room, placements, clearance_distance=1.0)

    access_objects_bfs = metric_accessibility_objects_bfs(room, placements, person_size=0.6)
    access_objects_astar = metric_accessibility_objects_astar(room, placements, person_size=0.6)
    access_windows_bfs = metric_accessibility_windows_bfs(room, placements, person_size=0.6)
    access_windows_astar = metric_accessibility_windows_astar(room, placements, person_size=0.6)

    prompt_constraints = metric_prompt_constraints_stub()
    style_match = metric_style_match(placements, model_info, prompt_style)
    generation_time = extract_generation_time_sec(scene_meta, generation_time_override)

    max_empty_square = metric_max_empty_square(room, placements, cell=0.05)
    max_empty_rectangle = metric_max_empty_rectangle(room, placements, cell=0.05)

    aggregate = aggregate_total_score(
        collisions=collisions,
        in_room=in_room,
        openings=openings,
        access_objects_bfs=access_objects_bfs,
        access_objects_astar=access_objects_astar,
        access_windows_bfs=access_windows_bfs,
        access_windows_astar=access_windows_astar,
        prompt_constraints=prompt_constraints,
        style_match=style_match,
        generation_time=generation_time,
    )

    valid_scene_flag = (
        collisions["scene_collision_free"] and
        in_room["all_inside_room"] and
        openings["door_free"] and
        openings["window_free"] and
        access_objects_bfs["score"] >= 0.999 and
        access_objects_astar["score"] >= 0.999 and
        access_windows_bfs["score"] >= 0.999 and
        access_windows_astar["score"] >= 0.999
    )

    return {
        "scene_schema": scene.get("schema"),
        "room_summary": {
            "x_min": room.x_min,
            "x_max": room.x_max,
            "y_min": room.y_min,
            "y_max": room.y_max,
            "z_min": room.z_min,
            "z_max": room.z_max,
            "width": room.width,
            "height": room.height,
            "area": room.area,
            "openings_count": len(room.openings),
            "doors_count": len(get_door_openings(room)),
            "windows_count": len(get_window_openings(room)),
        },
        "placements_count": len(placements),
        "metrics": {
            "collisions": collisions,
            "in_room": in_room,
            "openings": openings,
            "accessibility_objects_bfs": access_objects_bfs,
            "accessibility_objects_astar": access_objects_astar,
            "accessibility_windows_bfs": access_windows_bfs,
            "accessibility_windows_astar": access_windows_astar,
            "max_empty_square": max_empty_square,
            "max_empty_rectangle": max_empty_rectangle,
            "prompt_constraints": prompt_constraints,
            "style_match": style_match,
            "generation_time": generation_time,
        },
        "valid_scene": valid_scene_flag,
        "aggregate": aggregate,
    }


# ============================================================
# CLI
# ============================================================

def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Оценка scene.v1 по метрикам проекта")
    p.add_argument("--scene", required=True, help="Путь к scene.v1")
    p.add_argument("--model-info", required=True, help="Путь к data/sourse/3D-FRONT/3D-FUTURE-model/model_info.json")
    p.add_argument("--prompt-style", default=None, help="Стиль из промпта, например Modern")
    p.add_argument("--generation-time-sec", type=float, default=None, help="Явно задать полное время генерации")
    p.add_argument("--output", default=None, help="Куда сохранить evaluation.json")
    p.add_argument("--pretty", action="store_true", help="Печатать JSON красиво")
    return p


def main() -> None:
    args = build_cli().parse_args()

    scene = load_json(args.scene)
    model_info = load_model_info(args.model_info)

    result = evaluate_scene(
        scene=scene,
        model_info=model_info,
        prompt_style=args.prompt_style,
        generation_time_override=args.generation_time_sec,
    )

    if args.output:
        save_json(args.output, result)

    if args.pretty:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(result, ensure_ascii=False))

    if args.output:
        print(f"OK: saved -> {Path(args.output).expanduser().resolve()}")


if __name__ == "__main__":
    main()