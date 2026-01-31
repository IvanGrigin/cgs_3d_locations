# src/Plasement/placement_checks.py

import math
from typing import List, Tuple, Dict, Any, Optional

from glb_parser import Room
from pathfinding_astar import find_path_to_object


# ============================================================
# ГЕОМЕТРИЯ И КОЛЛИЗИИ
# ============================================================

def rotated_size(sx: float, sy: float, angle_deg: float) -> Tuple[float, float]:
    """
    Корректный размер AABB в плоскости XY при произвольном повороте.
    """
    a = math.radians(angle_deg)
    cos_a = abs(math.cos(a))
    sin_a = abs(math.sin(a))

    rx = sx * cos_a + sy * sin_a
    ry = sx * sin_a + sy * cos_a
    return rx, ry


def aabb_intersect(a: Dict[str, float], b: Dict[str, float]) -> bool:
    """
    Пересечение двух AABB в 3D.
    """
    return not (
        a["x_max"] <= b["x_min"] or
        a["x_min"] >= b["x_max"] or
        a["y_max"] <= b["y_min"] or
        a["y_min"] >= b["y_max"] or
        a["z_max"] <= b["z_min"] or
        a["z_min"] >= b["z_max"]
    )


def _point_in_poly(x: float, y: float, poly: List[Tuple[float, float]]) -> bool:
    """
    Ray casting: point-in-polygon для простого полигона.
    """
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        # пересечение горизонтального луча вправо
        if (y1 > y) != (y2 > y):
            # избегаем деления на 0
            denom = (y2 - y1)
            if abs(denom) < 1e-12:
                continue
            x_int = x1 + (y - y1) * (x2 - x1) / denom
            if x_int > x:
                inside = not inside
    return inside


def _aabb_inside_poly(aabb: Dict[str, float], poly: List[Tuple[float, float]], margin: float = 1e-6) -> bool:
    """
    Проверяем, что 9 опорных точек AABB в XY лежат внутри полигона:
      - 4 угла
      - 4 середины ребер
      - центр
    """
    x0 = aabb["x_min"] + margin
    x1 = aabb["x_max"] - margin
    y0 = aabb["y_min"] + margin
    y1 = aabb["y_max"] - margin
    xm = 0.5 * (x0 + x1)
    ym = 0.5 * (y0 + y1)

    samples = [
        (x0, y0), (x0, y1), (x1, y0), (x1, y1),
        (xm, y0), (xm, y1), (x0, ym), (x1, ym),
        (xm, ym),
    ]
    return all(_point_in_poly(x, y, poly) for x, y in samples)


def inside_room(aabb: Dict[str, float], room: Room) -> bool:
    # Z всегда bbox
    if aabb["z_min"] < room.z_min - 1e-6:
        return False
    if aabb["z_max"] > room.z_max + 1e-6:
        return False

    # ВАЖНО: полигон используем ТОЛЬКО если он пришёл из room.json (room-spec)
    if getattr(room, "poly_source", "") == "roomspec":
        poly = getattr(room, "floor_polygon", None)
        if isinstance(poly, list) and len(poly) >= 3:
            return _aabb_inside_poly(aabb, poly, margin=1e-6)

    # fallback: bbox
    return (
        aabb["x_min"] >= room.x_min - 1e-6 and
        aabb["x_max"] <= room.x_max + 1e-6 and
        aabb["y_min"] >= room.y_min - 1e-6 and
        aabb["y_max"] <= room.y_max + 1e-6
    )



def is_side_touching_wall(
    box: Dict[str, float],
    side: str,
    room: Room,
    epsilon: float = 0.02
) -> bool:
    """
    Проверка, что нужная сторона AABB касается стены комнаты.
    side: 'front'/'back'/'left'/'right'

    ВНИМАНИЕ: для room-spec L-комнаты это проверка по bbox-граням.
    Для осевых комнат этого достаточно. Для общих полигонов делается отдельная версия
    (касание по сегментам полигона), но пока держим минимально.
    """
    side = str(side).lower()
    if side == "front":
        return abs(box["y_min"] - room.y_min) < epsilon
    if side == "back":
        return abs(box["y_max"] - room.y_max) < epsilon
    if side == "left":
        return abs(box["x_min"] - room.x_min) < epsilon
    if side == "right":
        return abs(box["x_max"] - room.x_max) < epsilon
    return False


# ============================================================
# ПРОВЕРКА ДОСТУПА ЧЕЛОВЕКА (A*)
# ============================================================

def check_human_access_astar(room: Room, placed: List[Any]) -> bool:
    """
    Проверяем, что человек может подойти к предметам, для которых
    constraints.human_approach = True, и которые НЕ висят:
      - не under_ceiling
      - нет mount_height_m

    placed — список объектов, у которых есть:
      - .item.extra
      - .item.name
      - метод .aabb()
    """
    room_dict: Dict[str, Any] = vars(room)

    items_dicts = [
        {
            "name": p.item.name,
            "aabb": p.aabb(),
            "extra": p.item.extra,
        }
        for p in placed
    ]

    for p, obj in zip(placed, items_dicts):
        extra = p.item.extra

        if not extra.get("human_approach", False):
            continue

        # висячие / под потолком не проверяем
        if extra.get("under_ceiling") or extra.get("mount_height_m") is not None:
            continue

        allowed_sides = ["front", "back", "left", "right"]
        if "free_side_named" in extra:
            allowed_sides = [extra["free_side_named"]["side"]]

        box = obj["aabb"]
        offset = 0.6  # м – расстояние от предмета до точки подхода

        side_targets = {
            "front": ((box["x_min"] + box["x_max"]) / 2, box["y_min"] - offset),
            "back":  ((box["x_min"] + box["x_max"]) / 2, box["y_max"] + offset),
            "left":  (box["x_min"] - offset, (box["y_min"] + box["y_max"]) / 2),
            "right": (box["x_max"] + offset, (box["y_min"] + box["y_max"]) / 2),
        }

        path_found = False

        for side in allowed_sides:
            tx, ty = side_targets[side]

            path = find_path_to_object(
                room_dict,
                items_dicts,
                {
                    "name": p.item.name,
                    "aabb": box,
                    "target_override": (tx, ty),
                },
            )

            if path is not None:
                path_found = True
                break

        if not path_found:
            print(f"❌ Нет подхода к объекту: {p.item.name}")
            return False

    return True
