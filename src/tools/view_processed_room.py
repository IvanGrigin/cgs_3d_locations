#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/tools/view_processed_room.py
"""
CLI-визуализация "processed" JSON (top-down XZ).

Рисует:
- контур пола каждой комнаты (floor.outline_xz)
- bbox комнаты (floor.bbox_xz)
- объекты:
  * если есть size или bbox_local -> прямоугольник в XZ с поворотом yaw
  * иначе -> точка + стрелка направления по yaw

Оси:
- X вправо
- Z вверх на графике
- Y считается up и не визуализируется (top-down)

Примеры:
  python3 src/tools/view_processed_room.py --in processed.json --show
  python3 src/tools/view_processed_room.py --in processed.json --room "Library-76415" --out out/library.png
  python3 src/tools/view_processed_room.py --in processed.json --room-idx 0 --out out/room0.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


def _load_json(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _pick_room(data: Dict[str, Any], room_instanceid: Optional[str], room_idx: Optional[int]) -> List[Dict[str, Any]]:
    rooms = data.get("rooms", [])
    if not isinstance(rooms, list):
        raise ValueError("Ожидалось поле 'rooms' как список.")

    if room_instanceid is None and room_idx is None:
        return rooms

    if room_instanceid is not None:
        for r in rooms:
            if r.get("room_instanceid") == room_instanceid:
                return [r]
        raise ValueError(f"Комната с room_instanceid='{room_instanceid}' не найдена.")

    assert room_idx is not None
    if room_idx < 0 or room_idx >= len(rooms):
        raise ValueError(f"--room-idx вне диапазона: {room_idx}, всего комнат: {len(rooms)}")
    return [rooms[room_idx]]


def _rotate_xz(x: float, z: float, yaw: float) -> Tuple[float, float]:
    """
    Поворот в плоскости XZ на yaw вокруг +Y.
    Модель: yaw вращает ось +X в сторону +Z (правило правой руки).
    """
    c = math.cos(yaw)
    s = math.sin(yaw)
    xr = x * c - z * s
    zr = x * s + z * c
    return xr, zr


def _rect_corners_xz(cx: float, cz: float, sx: float, sz: float, yaw: float) -> List[Tuple[float, float]]:
    hx = 0.5 * sx
    hz = 0.5 * sz
    local = [(-hx, -hz), (hx, -hz), (hx, hz), (-hx, hz)]
    out = []
    for lx, lz in local:
        rx, rz = _rotate_xz(lx, lz, yaw)
        out.append((cx + rx, cz + rz))
    out.append(out[0])
    return out


def _poly_close(poly: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not poly:
        return poly
    if poly[0] != poly[-1]:
        return poly + [poly[0]]
    return poly


def _safe_size(obj: Dict[str, Any]) -> Optional[Tuple[float, float, float]]:
    """
    Возвращает (sx, sy, sz) если доступны размеры.
    Приоритет:
    - size (ожидается [x,y,z])
    - bbox_local:
        * если список длины 3 -> [x,y,z]
        * если [[x,y,z]] -> берём первый
    """
    s = obj.get("size")
    if isinstance(s, list) and len(s) == 3 and all(isinstance(v, (int, float)) for v in s):
        return float(s[0]), float(s[1]), float(s[2])

    b = obj.get("bbox_local")
    if isinstance(b, list):
        if len(b) == 3 and all(isinstance(v, (int, float)) for v in b):
            return float(b[0]), float(b[1]), float(b[2])
        if len(b) == 1 and isinstance(b[0], list) and len(b[0]) == 3 and all(isinstance(v, (int, float)) for v in b[0]):
            return float(b[0][0]), float(b[0][1]), float(b[0][2])

    return None


def _draw_room(ax, room: Dict[str, Any], draw_bbox: bool, draw_objects: bool, draw_labels: bool) -> None:
    room_id = room.get("room_instanceid", "room")
    room_type = room.get("room_type", "")
    floor = room.get("floor", {}) if isinstance(room.get("floor"), dict) else {}

    # Контур пола (outline_xz)
    outline = floor.get("outline_xz")
    if isinstance(outline, list) and all(isinstance(p, list) and len(p) == 2 for p in outline):
        poly = [(float(p[0]), float(p[1])) for p in outline]
        poly = _poly_close(poly)
        xs = [p[0] for p in poly]
        zs = [p[1] for p in poly]
        ax.plot(xs, zs, linewidth=2, label=f"{room_type}:{room_id}")

    # BBox комнаты (bbox_xz: [minx, minz, maxx, maxz])
    if draw_bbox:
        bb = floor.get("bbox_xz")
        if isinstance(bb, list) and len(bb) == 4 and all(isinstance(v, (int, float)) for v in bb):
            minx, minz, maxx, maxz = map(float, bb)
            rect = [(minx, minz), (maxx, minz), (maxx, maxz), (minx, maxz), (minx, minz)]
            ax.plot([p[0] for p in rect], [p[1] for p in rect], linestyle="--", linewidth=1)

    # Объекты
    if not draw_objects:
        return

    objects = room.get("objects", [])
    if not isinstance(objects, list):
        return

    for obj in objects:
        if not isinstance(obj, dict):
            continue

        pos = obj.get("pos")
        if not (isinstance(pos, list) and len(pos) == 3 and all(isinstance(v, (int, float)) for v in pos)):
            continue
        cx, cy, cz = float(pos[0]), float(pos[1]), float(pos[2])

        yaw = obj.get("yaw", 0.0)
        if not isinstance(yaw, (int, float)):
            yaw = 0.0
        yaw = float(yaw)

        inst = obj.get("instanceid", "")
        title = obj.get("title", "")

        size = _safe_size(obj)

        if size is not None:
            sx, sy, sz = size
            corners = _rect_corners_xz(cx, cz, sx, sz, yaw)
            ax.plot([p[0] for p in corners], [p[1] for p in corners], linewidth=1.5)
        else:
            ax.scatter([cx], [cz], s=18)

        # Направление (стрелка) всегда, чтобы визуально проверить yaw
        # Вектор направления в XZ: (cos(yaw), sin(yaw))
        # Длина подбирается относительно типичного масштаба комнаты
        dx = math.cos(yaw)
        dz = math.sin(yaw)
        ax.arrow(cx, cz, 0.35 * dx, 0.35 * dz, head_width=0.12, head_length=0.15, length_includes_head=True)

        if draw_labels and inst:
            label = inst
            if title:
                label = f"{inst} | {title}"
            ax.text(cx, cz, label, fontsize=7)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Путь к processed JSON")
    ap.add_argument("--out", dest="out", default=None, help="Сохранить картинку (png/pdf/svg). Если не задано — только show/консоль.")
    ap.add_argument("--show", action="store_true", help="Открыть окно matplotlib")
    ap.add_argument("--room", dest="room_instanceid", default=None, help="Показать только одну комнату по room_instanceid")
    ap.add_argument("--room-idx", dest="room_idx", type=int, default=None, help="Показать только одну комнату по индексу (0..)")
    ap.add_argument("--no-bbox", action="store_true", help="Не рисовать bbox комнаты")
    ap.add_argument("--no-objects", action="store_true", help="Не рисовать объекты")
    ap.add_argument("--labels", action="store_true", help="Подписи объектов (instanceid/title)")
    args = ap.parse_args()

    inp = Path(args.inp)
    if not inp.exists():
        raise FileNotFoundError(str(inp))

    data = _load_json(inp)

    meta = data.get("meta", {})
    if isinstance(meta, dict):
        axis_up = meta.get("axis_up")
        if axis_up not in (None, "Y"):
            # Визуализация top-down предполагает Y-up; если у вас когда-то появится Z-up — нужно менять плоскость.
            raise ValueError(f"meta.axis_up='{axis_up}' не поддержан этим viewer-ом (ожидается 'Y' или отсутствует).")

    rooms = _pick_room(data, args.room_instanceid, args.room_idx)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    for room in rooms:
        _draw_room(
            ax=ax,
            room=room,
            draw_bbox=not args.no_bbox,
            draw_objects=not args.no_objects,
            draw_labels=args.labels,
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Z (meters)")
    ax.grid(True)

    # легенда может быть большой; включаем только если комнат <= 10
    if len(rooms) <= 10:
        ax.legend(loc="best", fontsize=8)

    # Автоподбор границ
    ax.relim()
    ax.autoscale_view()

    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outp, dpi=180, bbox_inches="tight")

    if args.show or not args.out:
        plt.show()


if __name__ == "__main__":
    main()