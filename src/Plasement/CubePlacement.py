#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/Plasement/CubePlacement.py
#
# Расстановка мебели в комнате (GLB или room.json).
# - Разрешены повороты 0/90/180/270 градусов.
# - Поддержка touch_wall (прижатие к стене) + ориентация "спинкой к стене".
# - Поддержка clearance (свободные зоны рядом с предметом) с эвристиками по типу + override из objects.json.
# - Диагностика причин, почему предмет "не влезает".
# - Для roomspec (floor_polygon) touch_wall работает по РЕАЛЬНЫМ ребрам полигона,
#   а не по bbox-комнаты (иначе L-образные комнаты ломают init/relax).
# - Units всегда "m" (нормализация отключена).
#
# Требования:
#   - glb_parser.load_room_from_glb, glb_parser.Room
#   - placement_checks: rotated_size, aabb_intersect, inside_room, is_side_touching_wall, check_human_access_astar

import json
import random
import math
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

from glb_parser import load_room_from_glb, Room
from placement_checks import (
    rotated_size,
    aabb_intersect,
    inside_room,
    is_side_touching_wall,
    check_human_access_astar,
)

DEFAULT_GLB = "data/input/room.glb"
DEFAULT_JSON = "data/input/objects.json"
OUTPUT_JSON = "data/output/placement_result.json"

EPS_Z = 1e-5
WALL_MARGIN = 0.01

# Разрешаем 0 тоже, иначе "спинкой к стене" для некоторых стен будет недостижимо.
ALLOWED_ROTATIONS = (0, 90, 180, 270)

# ============================================================
# ROOMSPEC JSON -> Room (units всегда "m", нормализация отключена)
# ============================================================

def _as_xy_point(pt: Any) -> Tuple[float, float]:
    if isinstance(pt, dict):
        return float(pt["x"]), float(pt["y"])
    if isinstance(pt, (list, tuple)) and len(pt) >= 2:
        return float(pt[0]), float(pt[1])
    raise RuntimeError(f"room.json: неверная точка floor_polygon: {pt!r}")


def _load_room_from_roomspec_json(room_json_path: str) -> Room:
    p = Path(room_json_path).expanduser().resolve()
    data = json.loads(p.read_text(encoding="utf-8"))

    units = str(data.get("units", "m") or "m").strip().lower()
    if units != "m":
        raise RuntimeError(f"room.json: ожидались метры (units='m'), но получено: {units!r}")

    room = data.get("room")
    if not isinstance(room, dict):
        raise RuntimeError("room.json: нет поля 'room' (dict)")

    fp = room.get("floor_polygon")
    if not isinstance(fp, list) or len(fp) < 3:
        raise RuntimeError("room.json: room.floor_polygon должен быть списком точек (>=3)")

    poly = [_as_xy_point(pt) for pt in fp]

    xs = [x for (x, _) in poly]
    ys = [y for (_, y) in poly]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # Поддержка: ceiling_height (в метрах)
    if room.get("ceiling_height") is not None:
        h = float(room["ceiling_height"])
    else:
        h = float(room.get("ceiling_height_m", room.get("height_m", 2.7)))

    if h <= 0.0 or h > 50.0:
        raise RuntimeError(f"room.json: подозрительная высота потолка: {h}")

    r = Room(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, z_min=0.0, z_max=h)
    r.floor_polygon = poly
    r.poly_source = "roomspec"  # inside_room() использует полигон ТОЛЬКО при этом маркере
    return r


def load_room_auto(path_str: str) -> Room:
    p = str(path_str).strip()
    if p.lower().endswith(".json"):
        room = _load_room_from_roomspec_json(p)
        span_x = room.x_max - room.x_min
        span_y = room.y_max - room.y_min
        h = room.z_max - room.z_min
        print(f"[room] roomspec: span_x={span_x:.3f}m span_y={span_y:.3f}m h={h:.3f}m")
        return room

    room = load_room_from_glb(p)
    setattr(room, "poly_source", "glb")  # для GLB не включаем polygon-check (если inside_room так устроен)

    span_x = room.x_max - room.x_min
    span_y = room.y_max - room.y_min
    h = room.z_max - room.z_min
    print(f"[room] glb bbox: x={span_x:.3f}m y={span_y:.3f}m z={h:.3f}m (no normalization)")
    return room


# ============================================================
# МОДЕЛИ
# ============================================================

class Item:
    def __init__(
        self,
        name: str,
        min_size_mm: List[float],
        max_size_mm: List[float],
        color: Any,
        extra: Dict[str, Any],
        mesh_path: Optional[str] = None,
        mesh_fit_mode: str = "stretch",
    ):
        self.name = name
        # размеры предметов в objects.json в мм -> метры
        self.sx = random.uniform(min_size_mm[0], max_size_mm[0]) / 1000.0
        self.sy = random.uniform(min_size_mm[1], max_size_mm[1]) / 1000.0
        self.sz = random.uniform(min_size_mm[2], max_size_mm[2]) / 1000.0
        self.color = color
        self.extra: Dict[str, Any] = extra or {}
        self.mesh_path = mesh_path
        self.mesh_fit_mode = mesh_fit_mode  # "stretch" | "uniform"


class PlacedItem:
    def __init__(
        self,
        item: Item,
        center: Tuple[float, float, float],
        rotation_deg: float,
        wall_contact_side: Optional[str] = None,
    ):
        self.item = item
        self.cx, self.cy, self.cz = center
        self.rotation = int(rotation_deg) % 360
        self.wall_contact_side = wall_contact_side  # "front/back/left/right" | None

        # Реальный размер AABB в XY после поворота
        self.rx, self.ry = rotated_size(item.sx, item.sy, self.rotation)

    def aabb(self) -> Dict[str, float]:
        return _aabb_from_center(self.cx, self.cy, self.cz, self.rx, self.ry, self.item.sz)

    def forward_vector(self) -> Tuple[float, float, float]:
        """
        forward = локальная +Y в мировых координатах при вращении вокруг Z.
        rotation=0   -> (0,+1)
        rotation=90  -> (+1,0)
        rotation=180 -> (0,-1)
        rotation=270 -> (-1,0)
        """
        a = math.radians(self.rotation)
        dx = math.sin(a)
        dy = math.cos(a)
        return dx, dy, 0.0


# ============================================================
# БАЗОВЫЕ УТИЛИТЫ AABB/ГЕОМЕТРИИ
# ============================================================

def _aabb_from_center(cx: float, cy: float, cz: float, rx: float, ry: float, sz: float) -> Dict[str, float]:
    return {
        "x_min": cx - rx / 2,
        "x_max": cx + rx / 2,
        "y_min": cy - ry / 2,
        "y_max": cy + ry / 2,
        "z_min": cz - sz / 2,
        "z_max": cz + sz / 2,
    }


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def clamp_inside_room_xy(room: Room, cx: float, cy: float, rx: float, ry: float) -> Tuple[float, float]:
    half_rx = rx / 2.0
    half_ry = ry / 2.0
    new_cx = max(room.x_min + half_rx, min(room.x_max - half_rx, cx))
    new_cy = max(room.y_min + half_ry, min(room.y_max - half_ry, cy))
    return new_cx, new_cy


def _norm_deg90(rot: float) -> int:
    r = int(round(rot)) % 360
    return min(ALLOWED_ROTATIONS, key=lambda x: abs(x - r))

def _item_sort_key(item: Item) -> Tuple[int, float, float, float]:
    """
    Сначала ставим самые важные и самые большие предметы.
    Приоритет:
      0 -> wardrobe / bed
      1 -> sofa / table / dresser
      2 -> nightstand / chair / armchair
      3 -> lighting / прочее

    Затем сортировка по площади основания, объёму, максимальному габариту.
    """
    kind = _detect_kind(item.name)

    if kind in {"wardrobe", "bed"}:
        priority = 0
    elif kind in {"sofa", "table", "dresser"}:
        priority = 1
    elif kind in {"nightstand", "armchair"}:
        priority = 2
    else:
        priority = 3

    footprint = item.sx * item.sy
    volume = item.sx * item.sy * item.sz
    max_dim = max(item.sx, item.sy, item.sz)

    return (priority, -footprint, -volume, -max_dim)


def _candidate_wall_sides_for_item(item: Item, extra: Dict[str, Any]) -> List[str]:
    """
    Порядок перебора сторон стены.
    Для больших предметов задаём разумный приоритет.
    """
    kind = _detect_kind(item.name)

    tw = (extra or {}).get("touch_wall")
    if isinstance(tw, dict):
        if tw.get("side"):
            return [str(tw["side"]).lower().strip()]
        if isinstance(tw.get("sides"), list) and tw["sides"]:
            return [str(x).lower().strip() for x in tw["sides"] if str(x).strip()]

    sides = (extra or {}).get("touch_wall_sides")
    if isinstance(sides, list) and sides:
        return [str(x).lower().strip() for x in sides if str(x).strip()]

    if kind == "wardrobe":
        return ["back", "left", "right", "front"]
    if kind == "bed":
        # кровать чаще всего к дальней или боковой стене
        return ["back", "left", "right", "front"]

    return ["back", "left", "right", "front"]


def _bed_rotations_for_wall_side(wall_side: str, forward_axis: str = "+Y") -> List[Tuple[int, str]]:
    """
    Для кровати разрешаем два типа постановки:
    1) изголовьем/спинкой к стене
    2) боком к стене

    Возвращаем список пар:
      (rotation, local_back_side)

    local_back_side — какая ЛОКАЛЬНАЯ сторона предмета считается "спинкой",
    которая должна касаться стены после данного rotation.
    """
    out: List[Tuple[int, str]] = []

    # Основной сценарий: локальная back прижата к выбранной стене
    main_rot = rotation_for_back_to_wall(wall_side, forward_axis=forward_axis)
    out.append((main_rot, "back"))

    # Дополнительный сценарий: кровать боком к стене
    # Если кровать стоит боком, то к стене может смотреть либо левый, либо правый бок.
    if wall_side in ("front", "back"):
        side_rots = [
            (_norm_deg90((90 + _axis_to_rot_offset(forward_axis)) % 360), "left"),
            (_norm_deg90((270 + _axis_to_rot_offset(forward_axis)) % 360), "right"),
        ]
    else:
        side_rots = [
            (_norm_deg90((0 + _axis_to_rot_offset(forward_axis)) % 360), "left"),
            (_norm_deg90((180 + _axis_to_rot_offset(forward_axis)) % 360), "right"),
        ]

    for rot, local_back_side in side_rots:
        if (rot, local_back_side) not in out:
            out.append((rot, local_back_side))

    return out


def _candidate_rotations_for_item(
    item: Item,
    wall_side: Optional[str],
    extra: Dict[str, Any],
) -> List[Tuple[int, str]]:
    """
    Возвращает список кандидатов в формате:
      (rotation, local_back_side)

    local_back_side:
      - какая локальная сторона предмета считается той самой "спинкой",
        которую мы хотим прижать к wall_side
      - для обычного случая это "back"
      - для кровати боком это может быть "left" или "right"
    """
    forward_axis = str((extra or {}).get("forward_axis", "+Y"))
    kind = _detect_kind(item.name)

    if wall_side is not None:
        if kind == "bed":
            return _bed_rotations_for_wall_side(wall_side, forward_axis=forward_axis)
        return [(rotation_for_back_to_wall(wall_side, forward_axis=forward_axis), "back")]

    return [(rot, "back") for rot in ALLOWED_ROTATIONS]


def _try_place_candidate(
    room: Room,
    item: Item,
    placed: List[PlacedItem],
    wall_side: Optional[str],
    rotation: int,
    base_cx: float,
    base_cy: float,
) -> Optional[PlacedItem]:
    """
    Единая попытка поставить один предмет.

    Важно:
    - для потолочных светильников позиция выбирается специальным образом:
      максимально ровно по потолку, подальше от стен и других ламп;
    - для остальной мебели логика прежняя.
    """
    rx, ry = rotated_size(item.sx, item.sy, rotation)
    cz = resolve_vertical_center(room, item)

    # ------------------------------------------------------------
    # Спец-логика для потолочных светильников
    # ------------------------------------------------------------
    if _is_ceiling_light(item):
        best_pos = _best_ceiling_position(
            room=room,
            item=item,
            placed=placed,
            rotation=rotation,
        )
        if best_pos is None:
            return None

        cx, cy = best_pos
        candidate = PlacedItem(item, (cx, cy, cz), rotation, None)
        box = candidate.aabb()

        if not inside_room(box, room):
            return None

        if any(aabb_intersect(box, other.aabb()) for other in placed):
            return None

        if not validate_clearance(candidate, room, placed):
            return None

        if item.extra.get("mount_type") == "ceiling" or item.extra.get("under_ceiling"):
            if abs(box["z_max"] - room.z_max) > 2 * EPS_Z:
                return None

        return candidate

    # ------------------------------------------------------------
    # Обычная логика для мебели
    # ------------------------------------------------------------
    cx, cy = clamp_inside_room_xy(room, base_cx, base_cy, rx, ry)

    if wall_side is not None:
        if getattr(room, "poly_source", "") == "roomspec" and getattr(room, "floor_polygon", None):
            pos = _sample_touch_wall_position_poly(
                room, rx, ry, cz, item.sz, rotation, wall_side, wall_margin=WALL_MARGIN
            )
            if pos is None:
                return None
            cx, cy = pos
        else:
            cx, cy = _apply_touch_wall_lock(room, cx, cy, rx, ry, wall_side, wall_margin=WALL_MARGIN)

    cx, cy = clamp_inside_room_xy(room, cx, cy, rx, ry)

    candidate = PlacedItem(item, (cx, cy, cz), rotation, wall_side)
    box = candidate.aabb()

    if not inside_room(box, room):
        return None

    if any(aabb_intersect(box, other.aabb()) for other in placed):
        return None

    if not validate_clearance(candidate, room, placed):
        return None

    extra = item.extra or {}
    if wall_side is not None and getattr(room, "poly_source", "") != "roomspec":
        if extra.get("touch_wall") and not is_side_touching_wall(box, wall_side, room):
            return None

    if (
        extra.get("mount_type") == "floor"
        or (isinstance(extra.get("touch_floor"), dict) and extra["touch_floor"].get("side") == "bottom")
    ):
        if abs(box["z_min"] - room.z_min) > 2 * EPS_Z:
            return None

    return candidate

def _distance_point_to_segment(px: float, py: float, a: Tuple[float, float], b: Tuple[float, float]) -> float:
    qx, qy = _closest_point_on_segment(px, py, a, b)
    return math.hypot(px - qx, py - qy)


def _nearest_wall_side_bbox(room: Room, cx: float, cy: float) -> str:
    """
    Для bbox-комнаты возвращает ближайшую стену:
      front -> y_min
      back  -> y_max
      left  -> x_min
      right -> x_max
    """
    dists = {
        "front": abs(cy - room.y_min),
        "back": abs(room.y_max - cy),
        "left": abs(cx - room.x_min),
        "right": abs(room.x_max - cx),
    }
    return min(dists, key=dists.get)


def _nearest_wall_side_poly(room: Room, cx: float, cy: float) -> str:
    """
    Для roomspec-полигона:
    1) находим ближайшее ребро полигона,
    2) классифицируем его как front/back/left/right по направлению inward normal.
    """
    poly = getattr(room, "floor_polygon", None)
    if not poly:
        return _nearest_wall_side_bbox(room, cx, cy)

    best_edge = None
    best_dist = float("inf")

    for a, b in _poly_edges(poly):
        d = _distance_point_to_segment(cx, cy, a, b)
        if d < best_dist:
            best_dist = d
            best_edge = (a, b)

    if best_edge is None:
        return _nearest_wall_side_bbox(room, cx, cy)

    a, b = best_edge
    nx, ny = _edge_inward_normal(poly, a, b)

    # inward normal показывает, куда должен смотреть "перед" предмета
    # классифицируем по доминирующей компоненте
    if abs(nx) >= abs(ny):
        return "left" if nx > 0 else "right"
    else:
        return "front" if ny > 0 else "back"


def choose_nearest_wall_side(room: Room, cx: float, cy: float) -> str:
    """
    Единая функция выбора ближайшей стены комнаты.
    Потом её легко править отдельно.
    """
    if getattr(room, "poly_source", "") == "roomspec" and getattr(room, "floor_polygon", None):
        return _nearest_wall_side_poly(room, cx, cy)
    return _nearest_wall_side_bbox(room, cx, cy)


def choose_rotation_opposite_to_nearest_wall(
    room: Room,
    cx: float,
    cy: float,
    forward_axis: str = "+Y",
) -> int:
    """
    Передняя часть предмета направляется ПРОТИВОПОЛОЖНО ближайшей стене,
    то есть внутрь комнаты.

    Если ближайшая стена:
      front -> forward +Y -> rot 0
      back  -> forward -Y -> rot 180
      left  -> forward +X -> rot 90
      right -> forward -X -> rot 270
    """
    nearest_side = choose_nearest_wall_side(room, cx, cy)
    return rotation_for_back_to_wall(nearest_side, forward_axis=forward_axis)


def choose_wall_contact_side(extra: Dict[str, Any]) -> Optional[str]:
    """
    Отдельная функция выбора стороны для touch_wall.
    Сейчас логика простая:
      - если задан side -> берём его
      - если задан список sides -> берём первый
      - иначе берём первый из touch_wall_sides
    Сделано детерминированно, без random, чтобы потом было проще подправлять.
    """
    tw = (extra or {}).get("touch_wall")
    if not tw:
        return None

    if isinstance(tw, dict):
        if tw.get("side"):
            return str(tw["side"]).lower().strip()

        if isinstance(tw.get("sides"), list) and tw["sides"]:
            return str(tw["sides"][0]).lower().strip()

    sides = (extra or {}).get("touch_wall_sides", ["front", "back", "left", "right"])
    if isinstance(sides, list) and sides:
        return str(sides[0]).lower().strip()

    return None

# ============================================================
# ПОЛИГОН-СТЕНЫ (для roomspec floor_polygon)
# ============================================================

def _poly_edges(poly: List[Tuple[float, float]]) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    return list(zip(poly, poly[1:] + poly[:1]))


def _poly_signed_area2(poly: List[Tuple[float, float]]) -> float:
    s = 0.0
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        s += (x1 * y2 - x2 * y1)
    return s


def _edge_unit_dir(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float, float]:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    n = math.hypot(dx, dy)
    if n < 1e-12:
        return 0.0, 0.0, 0.0
    return dx / n, dy / n, n


def _edge_inward_normal(poly: List[Tuple[float, float]], a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    """
    Нормаль, направленная ВНУТРЬ полигона.
    Для CCW: inward = (-dy, dx) для направления a->b.
    Для CW:  inward = ( dy,-dx).
    """
    ux, uy, _ = _edge_unit_dir(a, b)
    if abs(ux) < 1e-12 and abs(uy) < 1e-12:
        return 0.0, 0.0
    area2 = _poly_signed_area2(poly)
    if area2 > 0:      # CCW
        return -uy, ux
    else:              # CW
        return uy, -ux


def _pick_wall_edge(poly: List[Tuple[float, float]], side: str) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Приближение "стороны" для произвольного полигона:
      left  -> ребро с минимальным средним x
      right -> ребро с максимальным средним x
      front -> ребро с минимальным средним y
      back  -> ребро с максимальным средним y
    """
    side = str(side).lower().strip()
    edges = _poly_edges(poly)

    def mid(e):
        (x1, y1), (x2, y2) = e
        return (x1 + x2) * 0.5, (y1 + y2) * 0.5

    if side == "left":
        return min(edges, key=lambda e: mid(e)[0])
    if side == "right":
        return max(edges, key=lambda e: mid(e)[0])
    if side == "front":
        return min(edges, key=lambda e: mid(e)[1])
    if side == "back":
        return max(edges, key=lambda e: mid(e)[1])

    return random.choice(edges)


def _closest_point_on_segment(px: float, py: float, a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    ax, ay = a
    bx, by = b
    vx, vy = (bx - ax), (by - ay)
    denom = vx * vx + vy * vy
    if denom < 1e-12:
        return ax, ay
    t = ((px - ax) * vx + (py - ay) * vy) / denom
    t = _clamp(t, 0.0, 1.0)
    return ax + t * vx, ay + t * vy

def _support_distance_along_normal(nx: float, ny: float, rx: float, ry: float) -> float:
    """
    Полуразмер AABB вдоль направления нормали (nx, ny).

    Для осевых нормалей:
      - вертикальная стена -> ry / 2
      - горизонтальная стена -> rx / 2

    В общем случае для AABB:
      extent = 0.5 * (|nx| * rx + |ny| * ry)
    """
    return 0.5 * (abs(nx) * rx + abs(ny) * ry)


def _project_to_wall_poly(
    room: Room,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    wall_side: str,
    wall_margin: float = WALL_MARGIN,
) -> Tuple[float, float]:
    """
    Проекция точки (cx, cy) на выбранное ребро "стены" и сдвиг внутрь.

    Важно:
    раньше offset считался как 0.5 * max(rx, ry), что неверно для длинных предметов,
    если стена касается короткой стороны AABB. Из-за этого шкафы могли оставаться
    далеко от стены.

    Теперь берём точный полуразмер AABB вдоль inward-normal стены.
    """
    poly = getattr(room, "floor_polygon", None)
    if not poly:
        return cx, cy

    a, b = _pick_wall_edge(poly, wall_side)
    nx, ny = _edge_inward_normal(poly, a, b)
    if abs(nx) < 1e-12 and abs(ny) < 1e-12:
        return cx, cy

    qx, qy = _closest_point_on_segment(cx, cy, a, b)

    offset = _support_distance_along_normal(nx, ny, rx, ry) + wall_margin
    ncx = qx + nx * offset
    ncy = qy + ny * offset

    ncx, ncy = clamp_inside_room_xy(room, ncx, ncy, rx, ry)
    return ncx, ncy

def _poly_wall_contact_gap(
    room: Room,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    wall_side: str,
) -> Optional[float]:
    """
    Возвращает зазор между AABB предмета и выбранной polygon-стеной.
    Чем ближе к 0, тем лучше контакт.

    Если полигон недоступен, возвращает None.
    """
    poly = getattr(room, "floor_polygon", None)
    if not poly:
        return None

    a, b = _pick_wall_edge(poly, wall_side)
    nx, ny = _edge_inward_normal(poly, a, b)
    if abs(nx) < 1e-12 and abs(ny) < 1e-12:
        return None

    qx, qy = _closest_point_on_segment(cx, cy, a, b)
    center_dist = math.hypot(cx - qx, cy - qy)
    support = _support_distance_along_normal(nx, ny, rx, ry)
    gap = center_dist - support
    return gap

def _sample_touch_wall_position_poly(
    room: Room,
    rx: float,
    ry: float,
    cz: float,
    sz: float,
    rotation: int,
    wall_side: str,
    tries: int = 400,
    wall_margin: float = WALL_MARGIN,
) -> Optional[Tuple[float, float]]:
    """
    Семплим позицию прижатия к "стене" (ребро полигона) и проверяем inside_room(AABB).

    Важно:
    смещение внутрь берём по реальному полуразмеру AABB вдоль нормали стены,
    а не по max(rx, ry), иначе длинные предметы получают искусственный зазор.
    """
    poly = getattr(room, "floor_polygon", None)
    if not poly:
        return None

    a, b = _pick_wall_edge(poly, wall_side)
    nx, ny = _edge_inward_normal(poly, a, b)
    if abs(nx) < 1e-12 and abs(ny) < 1e-12:
        return None

    offset = _support_distance_along_normal(nx, ny, rx, ry) + wall_margin

    for _ in range(tries):
        t = random.random()
        px = a[0] * (1.0 - t) + b[0] * t
        py = a[1] * (1.0 - t) + b[1] * t

        cx = px + nx * offset
        cy = py + ny * offset

        cx, cy = clamp_inside_room_xy(room, cx, cy, rx, ry)

        box = _aabb_from_center(cx, cy, cz, rx, ry, sz)
        if inside_room(box, room):
            return cx, cy

    return None

# ============================================================
# ОРИЕНТАЦИЯ "СПИНКОЙ К СТЕНЕ" (+ поддержка forward_axis)
# ============================================================

def _axis_to_rot_offset(forward_axis: str) -> int:
    """
    forward_axis задаёт, куда смотрит "forward" модели в её локальных координатах
    относительно нашей конвенции (+Y).
      "+Y" -> 0
      "-Y" -> 180
      "+X" -> 90   (т.е. чтобы совместить +X модели с +Y нашей системы надо повернуть -90,
                   но мы здесь используем "добавку к rotation", поэтому +90 даёт обратное;
                   для простоты придерживаемся практической схемы ниже)
    На практике чаще всего нужно только "+Y" или "-Y".
    """
    ax = str(forward_axis or "+Y").strip().upper()
    if ax == "+Y":
        return 0
    if ax == "-Y":
        return 180
    if ax == "+X":
        return 270
    if ax == "-X":
        return 90
    return 0


def rotation_for_back_to_wall(wall_side: str, forward_axis: str = "+Y") -> int:
    """
    Базовая конвенция стен по bbox (и по "сторонам" для полигона):
      - front: y = y_min
      - back:  y = y_max
      - left:  x = x_min
      - right: x = x_max

    "Спинкой к стене" => FORWARD должен смотреть ВНУТРЬ комнаты:
      front wall: forward +Y => rot 0
      back  wall: forward -Y => rot 180
      left  wall: forward +X => rot 90
      right wall: forward -X => rot 270

    Если модель фактически "смотрит" -Y (часто бывает), добавляем +180.
    """
    side = str(wall_side).lower().strip()
    mapping = {"front": 0, "back": 180, "left": 90, "right": 270}
    if side not in mapping:
        raise ValueError(f"Unknown wall side: {wall_side}")

    rot = mapping[side]
    rot = (180 + rot + _axis_to_rot_offset(forward_axis)) % 360 # +180 позволяет правильно риентировать предметы, там они стоят верно - спиной к стене
    return _norm_deg90(rot)


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def random_center_xy(room: Room, rx: float, ry: float) -> Tuple[float, float]:
    return (
        random.uniform(room.x_min + rx / 2, room.x_max - rx / 2),
        random.uniform(room.y_min + ry / 2, room.y_max - ry / 2),
    )

def _is_ceiling_light(item: Item) -> bool:
    extra = item.extra or {}
    kind = _detect_kind(item.name)

    return (
        extra.get("mount_type") == "ceiling"
        or extra.get("under_ceiling")
        or kind == "lighting"
    )


def _distance_to_bbox_walls(room: Room, cx: float, cy: float, rx: float, ry: float) -> float:
    return min(
        cx - (room.x_min + rx / 2.0),
        (room.x_max - rx / 2.0) - cx,
        cy - (room.y_min + ry / 2.0),
        (room.y_max - ry / 2.0) - cy,
    )


def _distance_to_poly_boundary(room: Room, cx: float, cy: float) -> float:
    poly = getattr(room, "floor_polygon", None)
    if not poly:
        return float("inf")

    best = float("inf")
    for a, b in _poly_edges(poly):
        d = _distance_point_to_segment(cx, cy, a, b)
        if d < best:
            best = d
    return best


def _distance_to_room_walls(room: Room, cx: float, cy: float, rx: float, ry: float) -> float:
    """
    Минимальное расстояние от AABB предмета до стен комнаты.
    Для roomspec используем floor_polygon, иначе bbox.
    """
    if getattr(room, "poly_source", "") == "roomspec" and getattr(room, "floor_polygon", None):
        d_center = _distance_to_poly_boundary(room, cx, cy)
        return d_center - 0.5 * math.hypot(rx, ry)

    return _distance_to_bbox_walls(room, cx, cy, rx, ry)


def _placed_ceiling_items(placed: List[PlacedItem]) -> List[PlacedItem]:
    return [p for p in placed if _is_ceiling_light(p.item)]


def _candidate_ceiling_positions(
    room: Room,
    rx: float,
    ry: float,
    cz: float,
    sz: float,
    placed: List[PlacedItem],
    grid_nx: int = 11,
    grid_ny: int = 9,
) -> List[Tuple[float, float]]:
    """
    Генерирует разумные кандидатные точки для потолочного светильника.
    Мы не хотим углы — поэтому берём внутреннюю сетку, исключая крайние точки.
    """
    x_lo = room.x_min + rx / 2.0
    x_hi = room.x_max - rx / 2.0
    y_lo = room.y_min + ry / 2.0
    y_hi = room.y_max - ry / 2.0

    if x_hi < x_lo or y_hi < y_lo:
        return []

    xs: List[float] = []
    ys: List[float] = []

    if grid_nx <= 1:
        xs = [(x_lo + x_hi) * 0.5]
    else:
        for i in range(grid_nx):
            t = (i + 1) / (grid_nx + 1)   # без крайних точек
            xs.append(x_lo * (1.0 - t) + x_hi * t)

    if grid_ny <= 1:
        ys = [(y_lo + y_hi) * 0.5]
    else:
        for i in range(grid_ny):
            t = (i + 1) / (grid_ny + 1)
            ys.append(y_lo * (1.0 - t) + y_hi * t)

    out: List[Tuple[float, float]] = []
    seen = set()

    # Сначала центр — для одного светильника это почти всегда лучший ответ
    cx0 = 0.5 * (x_lo + x_hi)
    cy0 = 0.5 * (y_lo + y_hi)
    center_key = (round(cx0, 6), round(cy0, 6))
    seen.add(center_key)
    out.append((cx0, cy0))

    for cx in xs:
        for cy in ys:
            key = (round(cx, 6), round(cy, 6))
            if key in seen:
                continue

            box = _aabb_from_center(cx, cy, cz, rx, ry, sz)
            if not inside_room(box, room):
                continue

            out.append((cx, cy))
            seen.add(key)

    return out


def _score_ceiling_position(
    room: Room,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    ceiling_items: List[PlacedItem],
) -> float:
    """
    Чем больше score, тем лучше позиция.

    Идея:
    1) хотим быть далеко от стен;
    2) хотим быть далеко от уже поставленных ламп;
    3) при равенстве слегка предпочитаем более центральные точки.
    """
    dist_walls = _distance_to_room_walls(room, cx, cy, rx, ry)

    if ceiling_items:
        dist_lamps = min(math.hypot(cx - p.cx, cy - p.cy) for p in ceiling_items)
    else:
        dist_lamps = float("inf")

    cx_room = 0.5 * (room.x_min + room.x_max)
    cy_room = 0.5 * (room.y_min + room.y_max)
    dist_center = math.hypot(cx - cx_room, cy - cy_room)

    # maximin: главным делаем минимум из "до стен" и "до других ламп"
    primary = min(dist_walls, dist_lamps)

    # маленький бонус за центральность, чтобы первая лампа не уезжала к краям
    return primary - 0.08 * dist_center


def _best_ceiling_position(
    room: Room,
    item: Item,
    placed: List[PlacedItem],
    rotation: int,
) -> Optional[Tuple[float, float]]:
    rx, ry = rotated_size(item.sx, item.sy, rotation)
    cz = resolve_vertical_center(room, item)

    candidates = _candidate_ceiling_positions(
        room=room,
        rx=rx,
        ry=ry,
        cz=cz,
        sz=item.sz,
        placed=placed,
    )
    if not candidates:
        return None

    ceiling_items = _placed_ceiling_items(placed)

    best_pos: Optional[Tuple[float, float]] = None
    best_score = -1e18

    for cx, cy in candidates:
        box = _aabb_from_center(cx, cy, cz, rx, ry, item.sz)

        if not inside_room(box, room):
            continue

        if any(aabb_intersect(box, other.aabb()) for other in placed):
            continue

        candidate = PlacedItem(item, (cx, cy, cz), rotation, None)
        if not validate_clearance(candidate, room, placed):
            continue

        score = _score_ceiling_position(room, cx, cy, rx, ry, ceiling_items)
        if score > best_score:
            best_score = score
            best_pos = (cx, cy)

    return best_pos

def resolve_vertical_center(room: Room, item: Item) -> float:
    """
    Вычисляет вертикальный центр предмета.

    Правила:
    - floor/mount_type=floor -> стоит на полу
    - ceiling/mount_type=ceiling -> верх AABB касается потолка
    - для подвесных ламп можно задать drop_from_ceiling_m
    - mount_height_m остаётся как override
    """
    extra = item.extra or {}
    kind = _detect_kind(item.name)

    tf = extra.get("touch_floor")
    if (
        (isinstance(tf, dict) and tf.get("side") == "bottom")
        or extra.get("mount_type") == "floor"
    ):
        return room.z_min + item.sz / 2.0

    is_ceiling = (
        extra.get("mount_type") == "ceiling"
        or extra.get("under_ceiling")
        or kind == "lighting"
        or "lamp" in (item.name or "").lower()
        or "light" in (item.name or "").lower()
    )

    if is_ceiling:
        drop = float(extra.get("drop_from_ceiling_m", 0.0) or 0.0)
        cz = room.z_max - item.sz / 2.0 - drop
        cz = max(room.z_min + item.sz / 2.0, min(room.z_max - item.sz / 2.0, cz))
        return cz

    if "mount_height_m" in extra and extra["mount_height_m"] is not None:
        h = float(extra["mount_height_m"])
        anchor = str(extra.get("mount_anchor", "center"))
        if anchor == "bottom":
            cz = room.z_min + h + item.sz / 2.0
        elif anchor == "top":
            cz = room.z_min + h - item.sz / 2.0
        else:
            cz = room.z_min + h
        return max(room.z_min + item.sz / 2.0, min(room.z_max - item.sz / 2.0, cz))

    return room.z_min + item.sz / 2.0


def enforce_vertical_constraints(pi: PlacedItem, room: Room) -> None:
    pi.cz = resolve_vertical_center(room, pi.item)


def _apply_touch_wall_lock(
    room: Room,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    wall_side: str,
    wall_margin: float = WALL_MARGIN,
) -> Tuple[float, float]:
    """
    Применяем "замок" wall contact:
      - для roomspec: проекция на ребро полигона + сдвиг внутрь
      - иначе: прижатие к bbox-стене
    """
    if not wall_side:
        return cx, cy

    if getattr(room, "poly_source", "") == "roomspec" and getattr(room, "floor_polygon", None):
        return _project_to_wall_poly(room, cx, cy, rx, ry, wall_side, wall_margin=wall_margin)

    # bbox
    if wall_side == "back":
        cy = room.y_max - ry / 2 - wall_margin
    elif wall_side == "front":
        cy = room.y_min + ry / 2 + wall_margin
    elif wall_side == "left":
        cx = room.x_min + rx / 2 + wall_margin
    elif wall_side == "right":
        cx = room.x_max - rx / 2 - wall_margin

    cx, cy = clamp_inside_room_xy(room, cx, cy, rx, ry)
    return cx, cy


# ============================================================
# CLEARANCE
# ============================================================

def _detect_kind(name: str) -> str:
    n = (name or "").strip().lower()

    if "кровать" in n:
        return "bed"
    if "шкаф" in n or "гардероб" in n or "купе" in n:
        return "wardrobe"
    if "тумб" in n:
        return "nightstand"
    if "комод" in n:
        return "dresser"
    if "диван" in n:
        return "sofa"
    if "кресл" in n:
        return "armchair"
    if "стол" in n:
        return "table"
    if (
        "lamp" in n
        or "light" in n
        or "люстр" in n
        or "светиль" in n
        or "ламп" in n
        or "бра" in n
        or "plafon" in n
    ):
        return "lighting"

    return "generic"


def _rot2(a_deg: float, vx: float, vy: float) -> Tuple[int, int]:
    """
    Поворот вектора (vx,vy) на a_deg, результат округляем к -1/0/+1,
    так как a_deg ∈ {0,90,180,270}.
    """
    a = math.radians(int(a_deg) % 360)
    wx = vx * math.cos(a) - vy * math.sin(a)
    wy = vx * math.sin(a) + vy * math.cos(a)
    return int(round(wx)), int(round(wy))


def _local_to_world_axis_and_sign(local_side: str, rot_deg: float) -> Tuple[str, int]:
    """
    Возвращает ('x'|'y', sign) — куда смотрит локальная сторона в мире.
    Локальные стороны:
      front = +Y, back = -Y, right = +X, left = -X
    """
    s = str(local_side).lower().strip()
    if s == "front":
        vx, vy = 0, 1
    elif s == "back":
        vx, vy = 0, -1
    elif s == "right":
        vx, vy = 1, 0
    elif s == "left":
        vx, vy = -1, 0
    else:
        raise ValueError(f"Unknown local_side: {local_side!r}")

    wx, wy = _rot2(rot_deg, vx, vy)
    if abs(wx) == 1 and wy == 0:
        return "x", wx
    if abs(wy) == 1 and wx == 0:
        return "y", wy

    if abs(wx) >= abs(wy):
        return "x", 1 if wx >= 0 else -1
    return "y", 1 if wy >= 0 else -1


def _clearance_aabb_for_side(box: Dict[str, float], axis: str, sign: int, dist: float) -> Dict[str, float]:
    dist = float(dist)
    if dist <= 0:
        raise ValueError("clearance dist must be > 0")

    c = dict(box)
    if axis == "y":
        if sign > 0:
            c["y_min"] = box["y_max"]
            c["y_max"] = box["y_max"] + dist
        else:
            c["y_min"] = box["y_min"] - dist
            c["y_max"] = box["y_min"]
    elif axis == "x":
        if sign > 0:
            c["x_min"] = box["x_max"]
            c["x_max"] = box["x_max"] + dist
        else:
            c["x_min"] = box["x_min"] - dist
            c["x_max"] = box["x_min"]
    else:
        raise ValueError(f"bad axis: {axis}")
    return c


def _default_clearance_rule(item: Item) -> Optional[Dict[str, Any]]:
    extra = item.extra or {}

    user = extra.get("clearance")
    if isinstance(user, dict):
        mode = str(user.get("mode", "all")).lower()
        sides = user.get("sides")
        if isinstance(sides, dict) and sides:
            rule = {"mode": ("any" if mode == "any" else "all"), "sides": {}, "disallow": []}
            for k, v in sides.items():
                rule["sides"][str(k).lower()] = float(v)
            dis = user.get("disallow", [])
            if isinstance(dis, list):
                rule["disallow"] = [str(x).lower() for x in dis]
            return rule

    kind = _detect_kind(item.name)

    if kind == "wardrobe":
        return {"mode": "all", "sides": {"front": 0.50}, "disallow": []}

    if kind == "nightstand":
        base = 0.6 * min(item.sx, item.sy)
        dist = max(0.25, min(0.55, base))
        return {"mode": "all", "sides": {"front": dist}, "disallow": []}

    if kind == "dresser":
        base = 0.7 * min(item.sx, item.sy)
        dist = max(0.30, min(0.70, base))
        return {"mode": "all", "sides": {"front": dist}, "disallow": []}

    if kind == "bed":
        return {"mode": "any", "sides": {"front": 0.60, "left": 0.60, "right": 0.60}, "disallow": ["back"]}

    if kind == "sofa":
        return {"mode": "all", "sides": {"front": 0.50}, "disallow": []}

    return None


def validate_clearance(candidate: PlacedItem, room: Room, others: List[PlacedItem]) -> bool:
    rule = _default_clearance_rule(candidate.item)
    if not rule:
        return True

    mode = str(rule.get("mode", "all")).lower()
    sides: Dict[str, float] = dict(rule.get("sides") or {})
    disallow = set([str(x).lower() for x in (rule.get("disallow") or [])])

    for s in list(sides.keys()):
        if s in disallow:
            sides.pop(s, None)

    if not sides:
        return True

    cand_box = candidate.aabb()
    other_boxes = [p.aabb() for p in others]

    def side_ok(local_side: str, dist: float) -> bool:
        axis, sign = _local_to_world_axis_and_sign(local_side, candidate.rotation)
        clear_box = _clearance_aabb_for_side(cand_box, axis, sign, dist)

        if not inside_room(clear_box, room):
            return False

        for ob in other_boxes:
            if aabb_intersect(clear_box, ob):
                return False

        return True

    if mode == "any":
        return any(side_ok(s, float(dist)) for s, dist in sides.items())

    return all(side_ok(s, float(dist)) for s, dist in sides.items())


# ============================================================
# RANDOM
# ============================================================

def place_all_random(room: Room, items: List[Item]) -> List[PlacedItem]:
    MAX_GLOBAL_TRIES = 60
    MAX_ITEM_TRIES = 400

    items_sorted = sorted(items, key=_item_sort_key)

    cx_room = (room.x_min + room.x_max) / 2.0
    cy_room = (room.y_min + room.y_max) / 2.0

    for global_try in range(MAX_GLOBAL_TRIES):
        placed: List[PlacedItem] = []
        failed = False

        for item in items_sorted:
            extra = item.extra or {}
            success = False
            reasons = {"inside": 0, "collide": 0, "clearance": 0, "touchwall": 0, "floor": 0, "all": 0}

            wall_side_candidates = _candidate_wall_sides_for_item(item, extra) if extra.get("touch_wall") else [None]

            for wall_side in wall_side_candidates:
                rotations = _candidate_rotations_for_item(item, wall_side, extra)

                for rotation, _local_back_side in rotations:
                    for _ in range(MAX_ITEM_TRIES):
                        if wall_side is not None:
                            base_cx = cx_room + random.uniform(-0.4, 0.4)
                            base_cy = cy_room + random.uniform(-0.4, 0.4)
                        else:
                            tmp_rx, tmp_ry = rotated_size(item.sx, item.sy, rotation)
                            base_cx, base_cy = random_center_xy(room, tmp_rx, tmp_ry)

                        candidate = _try_place_candidate(
                            room=room,
                            item=item,
                            placed=placed,
                            wall_side=wall_side,
                            rotation=rotation,
                            base_cx=base_cx,
                            base_cy=base_cy,
                        )

                        if candidate is not None:
                            placed.append(candidate)
                            success = True
                            break

                        reasons["all"] += 1

                    if success:
                        break
                if success:
                    break

            if not success:
                print(f"⚠️ random: не влез: {item.name} reasons={reasons}")
                failed = True
                break

        if not failed:
            for pi in placed:
                enforce_vertical_constraints(pi, room)
            return placed

        if global_try == 0 or (global_try + 1) % 10 == 0:
            print(f"⚠️ random: попытка {global_try+1}/{MAX_GLOBAL_TRIES} провалилась, пробуем заново")

    raise RuntimeError("❌ Не удалось расставить предметы (random)")


# ============================================================
# RELAXED
# ============================================================

def relax_layout(
    room: Room,
    placed: List[PlacedItem],
    iterations: int = 260,
    max_step: float = 0.15,
    wall_margin: float = WALL_MARGIN,
) -> None:
    cx_room = (room.x_min + room.x_max) / 2.0
    cy_room = (room.y_min + room.y_max) / 2.0

    for _ in range(iterations):
        n = len(placed)
        displacements = [(0.0, 0.0) for _ in range(n)]

        # 1) Раздвигаем пересечения AABB
        for i in range(n):
            a = placed[i].aabb()
            for j in range(i + 1, n):
                b = placed[j].aabb()
                if not aabb_intersect(a, b):
                    continue

                overlap_x = min(a["x_max"], b["x_max"]) - max(a["x_min"], b["x_min"])
                overlap_y = min(a["y_max"], b["y_max"]) - max(a["y_min"], b["y_min"])
                if overlap_x <= 0 or overlap_y <= 0:
                    continue

                if overlap_x < overlap_y:
                    dir_sign = 1.0 if placed[i].cx >= placed[j].cx else -1.0
                    move = overlap_x * 0.6
                    dx_i = dir_sign * move * 0.5
                    dx_j = -dir_sign * move * 0.5
                    dix, diy = displacements[i]
                    djx, djy = displacements[j]
                    displacements[i] = (dix + dx_i, diy)
                    displacements[j] = (djx + dx_j, djy)
                else:
                    dir_sign = 1.0 if placed[i].cy >= placed[j].cy else -1.0
                    move = overlap_y * 0.6
                    dy_i = dir_sign * move * 0.5
                    dy_j = -dir_sign * move * 0.5
                    dix, diy = displacements[i]
                    djx, djy = displacements[j]
                    displacements[i] = (dix, diy + dy_i)
                    displacements[j] = (djx, djy + dy_j)

        # 2) Лёгкое отталкивание от центра
        k_out = 0.05
        for i, pi in enumerate(placed):
            dix, diy = displacements[i]
            vx = pi.cx - cx_room
            vy = pi.cy - cy_room
            dist = math.hypot(vx, vy)
            if dist > 1e-3:
                dix += k_out * vx / dist
                diy += k_out * vy / dist
            displacements[i] = (dix, diy)

        # 3) Применяем сдвиги с backtracking
        for i, pi in enumerate(placed):
            mx, my = displacements[i]
            step_len = math.hypot(mx, my)
            if step_len > max_step:
                scale = max_step / step_len
                mx *= scale
                my *= scale

            old_cx, old_cy = pi.cx, pi.cy
            side = pi.wall_contact_side
            extra = pi.item.extra or {}

            mx_try, my_try = mx, my
            moved = False

            for _bt in range(10):
                new_cx = old_cx + mx_try
                new_cy = old_cy + my_try
                new_cx, new_cy = clamp_inside_room_xy(room, new_cx, new_cy, pi.rx, pi.ry)

                # Жёстко возвращаем wall-contact предмет обратно к стене
                if extra.get("touch_wall") and side is not None:
                    new_cx, new_cy = _apply_touch_wall_lock(
                        room,
                        new_cx,
                        new_cy,
                        pi.rx,
                        pi.ry,
                        side,
                        wall_margin=wall_margin,
                    )

                tmp_box = _aabb_from_center(new_cx, new_cy, pi.cz, pi.rx, pi.ry, pi.item.sz)

                if inside_room(tmp_box, room):
                    pi.cx, pi.cy = new_cx, new_cy
                    enforce_vertical_constraints(pi, room)
                    moved = True
                    break

                mx_try *= 0.5
                my_try *= 0.5

            if not moved:
                pi.cx, pi.cy = old_cx, old_cy
                if extra.get("touch_wall") and side is not None:
                    pi.cx, pi.cy = _apply_touch_wall_lock(
                        room,
                        pi.cx,
                        pi.cy,
                        pi.rx,
                        pi.ry,
                        side,
                        wall_margin=wall_margin,
                    )
                enforce_vertical_constraints(pi, room)

    # 4) Финальный жёсткий repin всех wall-contact предметов
    for pi in placed:
        extra = pi.item.extra or {}
        if extra.get("touch_wall") and pi.wall_contact_side is not None:
            pi.cx, pi.cy = _apply_touch_wall_lock(
                room,
                pi.cx,
                pi.cy,
                pi.rx,
                pi.ry,
                pi.wall_contact_side,
                wall_margin=wall_margin,
            )
            enforce_vertical_constraints(pi, room)


def place_all_relaxed(room: Room, items: List[Item]) -> List[PlacedItem]:
    cx_room = (room.x_min + room.x_max) / 2.0
    cy_room = (room.y_min + room.y_max) / 2.0

    MAX_GLOBAL_TRIES = 55
    items_sorted = sorted(items, key=_item_sort_key)

    for global_try in range(MAX_GLOBAL_TRIES):
        placed: List[PlacedItem] = []
        failed_init = False

        # 1) Инициализация: сначала крупные предметы
        for item in items_sorted:
            extra = item.extra or {}
            success = False

            wall_side_candidates = _candidate_wall_sides_for_item(item, extra) if extra.get("touch_wall") else [None]

            for wall_side in wall_side_candidates:
                rotations = _candidate_rotations_for_item(item, wall_side, extra)

                for rotation, _local_back_side in rotations:
                    for _attempt in range(160):
                        if wall_side is not None:
                            base_cx = cx_room + random.uniform(-0.35, 0.35)
                            base_cy = cy_room + random.uniform(-0.35, 0.35)
                        else:
                            tmp_rx, tmp_ry = rotated_size(item.sx, item.sy, rotation)
                            base_cx, base_cy = random_center_xy(room, tmp_rx, tmp_ry)

                        candidate = _try_place_candidate(
                            room=room,
                            item=item,
                            placed=placed,
                            wall_side=wall_side,
                            rotation=rotation,
                            base_cx=base_cx,
                            base_cy=base_cy,
                        )
                        if candidate is not None:
                            placed.append(candidate)
                            success = True
                            break

                    if success:
                        break
                if success:
                    break

            if not success:
                print(f"⚠️ init: не удалось стартово поставить предмет: {item.name}")
                failed_init = True
                break

        if failed_init:
            continue

        # 2) Релаксация
        relax_layout(room, placed)

        # 3) Свободные предметы можно довернуть в комнату
        for pi in placed:
            if pi.wall_contact_side is not None:
                continue

            forward_axis = str((pi.item.extra or {}).get("forward_axis", "+Y"))
            new_rotation = choose_rotation_opposite_to_nearest_wall(
                room, pi.cx, pi.cy, forward_axis=forward_axis
            )
            pi.rotation = new_rotation
            pi.rx, pi.ry = rotated_size(pi.item.sx, pi.item.sy, pi.rotation)
            pi.cx, pi.cy = clamp_inside_room_xy(room, pi.cx, pi.cy, pi.rx, pi.ry)
            enforce_vertical_constraints(pi, room)

        # 4) Валидация
        valid = True
        reasons = {"inside": 0, "collide": 0, "clearance": 0, "touchwall": 0, "floor": 0}

        for i, pi in enumerate(placed):
            enforce_vertical_constraints(pi, room)
            box_i = pi.aabb()

            if not inside_room(box_i, room):
                reasons["inside"] += 1
                valid = False
                break

            for j in range(i + 1, len(placed)):
                if aabb_intersect(box_i, placed[j].aabb()):
                    reasons["collide"] += 1
                    valid = False
                    break
            if not valid:
                break

            if pi.item.extra.get("touch_wall") and pi.wall_contact_side is not None:
                if getattr(room, "poly_source", "") == "roomspec":
                    gap = _poly_wall_contact_gap(room, pi.cx, pi.cy, pi.rx, pi.ry, pi.wall_contact_side)
                    if gap is None or gap > (WALL_MARGIN + 0.02):
                        reasons["touchwall"] += 1
                        valid = False
                        break
                else:
                    if not is_side_touching_wall(box_i, pi.wall_contact_side, room):
                        reasons["touchwall"] += 1
                        valid = False
                        break

            others = [p for j, p in enumerate(placed) if j != i]
            if not validate_clearance(pi, room, others):
                reasons["clearance"] += 1
                valid = False
                break

            if (
                pi.item.extra.get("mount_type") == "floor"
                or (
                    isinstance(pi.item.extra.get("touch_floor"), dict)
                    and pi.item.extra["touch_floor"].get("side") == "bottom"
                )
            ):
                if abs(box_i["z_min"] - room.z_min) > 2 * EPS_Z:
                    reasons["floor"] += 1
                    valid = False
                    break

        if valid:
            return placed

        if global_try == 0 or (global_try + 1) % 10 == 0:
            print(f"⚠️ relaxed: попытка {global_try+1}/{MAX_GLOBAL_TRIES} провал, reasons={reasons}")

    raise RuntimeError("❌ Не удалось расставить предметы (relaxed)")

# ============================================================
# ВЫБОР РЕЖИМА
# ============================================================

def place_all(room: Room, items: List[Item], mode: str = "relaxed") -> List[PlacedItem]:
    mode = str(mode).lower().strip()
    if mode == "random":
        return place_all_random(room, items)
    if mode == "relaxed":
        return place_all_relaxed(room, items)
    raise ValueError(f"Неизвестный режим расстановки: {mode!r}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=== РАССТАНОВКА ОБЪЕКТОВ ===")
    print("Режимы: [relaxed] — размывание от центра, [random] — перебор")
    print(f"Разрешённые повороты: {ALLOWED_ROTATIONS}")

    room_path = input(f"GLB комнаты [{DEFAULT_GLB}] (или room.json): ").strip() or DEFAULT_GLB
    json_path = input(f"JSON объектов [{DEFAULT_JSON}]: ").strip() or DEFAULT_JSON
    mode = input("Режим расстановки [relaxed/random] (по умолчанию relaxed): ").strip() or "relaxed"

    room = load_room_auto(room_path)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "seed" in data and data["seed"] is not None:
        try:
            rnd_seed = int(data["seed"])
            random.seed(rnd_seed)
            print(f"[place] RNG seed = {rnd_seed}")
        except Exception:
            pass

    items: List[Item] = []
    for obj in data["items"]:
        items.append(
            Item(
                name=obj["name"],
                min_size_mm=obj["min_size_mm"],
                max_size_mm=obj["max_size_mm"],
                color=obj.get("color", [1, 1, 1]),
                extra=obj.get("constraints", {}),
                mesh_path=obj.get("mesh_path"),
                mesh_fit_mode=obj.get("mesh_fit_mode", "stretch"),
            )
        )

    placed = place_all(room, items, mode=mode)

    if not check_human_access_astar(room, placed):
        raise RuntimeError("❌ ЧЕЛОВЕК НЕ МОЖЕТ ПОДОЙТИ КО ВСЕМ НУЖНЫМ ОБЪЕКТАМ")

    result = {"room": vars(room), "items": []}

    for p in placed:
        fx, fy, fz = p.forward_vector()
        result["items"].append(
            {
                "name": p.item.name,
                "center": [p.cx, p.cy, p.cz],
                "size": [p.item.sx, p.item.sy, p.item.sz],
                "rotation": p.rotation,
                "aabb": p.aabb(),
                "color": p.item.color,
                "forward": [fx, fy, fz],
                "wall_contact_side": p.wall_contact_side,
                "constraints": p.item.extra,
                "mesh_path": p.item.mesh_path,
                "mesh_fit_mode": p.item.mesh_fit_mode,
            }
        )

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n✅ ГОТОВО! placement_result.json создан (режим: {mode})\n")
    for item in result["items"]:
        print(item["name"], "→ центр", item["center"], "rot:", item["rotation"], "wall_side:", item["wall_contact_side"])


if __name__ == "__main__":
    main()
