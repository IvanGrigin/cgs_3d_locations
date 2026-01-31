#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _as_size_xz(obj: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """
    Возвращает (sx, sz) в метрах для футпринта на плоскости XZ.
    Приоритет: size -> bbox_local.
    size ожидается как [sx, sy, sz].
    bbox_local может быть [sx, sy, sz] или [[sx, sy, sz]].
    """
    size = obj.get("size", None)
    if isinstance(size, list) and len(size) >= 3 and all(isinstance(v, (int, float)) for v in size[:3]):
        sx = float(size[0])
        sz = float(size[2])
        if sx > 0 and sz > 0:
            return sx, sz

    bbox = obj.get("bbox_local", None)
    if isinstance(bbox, list):
        # вариант [[sx, sy, sz]]
        if len(bbox) == 1 and isinstance(bbox[0], list) and len(bbox[0]) >= 3:
            bb = bbox[0]
            if all(isinstance(v, (int, float)) for v in bb[:3]):
                sx = float(bb[0])
                sz = float(bb[2])
                if sx > 0 and sz > 0:
                    return sx, sz
        # вариант [sx, sy, sz]
        if len(bbox) >= 3 and all(isinstance(v, (int, float)) for v in bbox[:3]):
            sx = float(bbox[0])
            sz = float(bbox[2])
            if sx > 0 and sz > 0:
                return sx, sz

    return None


def _rot2d(yaw: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return ((c, -s), (s, c))


def _add_oriented_rect(ax, cx: float, cz: float, sx: float, sz: float, yaw: float, **kwargs):
    """
    Рисует прямоугольник sx x sz, центр (cx, cz), поворот yaw вокруг Y (т.е. в плоскости XZ).
    """
    hx = 0.5 * sx
    hz = 0.5 * sz
    local = [(-hx, -hz), (hx, -hz), (hx, hz), (-hx, hz), (-hx, -hz)]

    R = _rot2d(yaw)
    xs, zs = [], []
    for lx, lz in local:
        x = cx + R[0][0] * lx + R[0][1] * lz
        z = cz + R[1][0] * lx + R[1][1] * lz
        xs.append(x)
        zs.append(z)

    ax.plot(xs, zs, **kwargs)

    # стрелка направления "forward" (ось +X локальная после yaw)
    fx = cx + R[0][0] * (0.6 * hx)
    fz = cz + R[1][0] * (0.6 * hx)
    ax.annotate(
        "",
        xy=(fx, fz),
        xytext=(cx, cz),
        arrowprops=dict(arrowstyle="->", linewidth=1),
    )


def _plot_room(ax, room: Dict[str, Any], title: str, draw_labels: bool = False):
    floor = room.get("floor", {}) or {}
    outline = floor.get("outline_xz", None)

    # Контур пола
    if isinstance(outline, list) and len(outline) >= 3:
        xs = [p[0] for p in outline if isinstance(p, list) and len(p) == 2]
        zs = [p[1] for p in outline if isinstance(p, list) and len(p) == 2]
        if len(xs) >= 3:
            xs.append(xs[0])
            zs.append(zs[0])
            ax.plot(xs, zs, linewidth=2)
    else:
        # fallback: bbox_xz = [minx, minz, maxx, maxz]
        bbox = floor.get("bbox_xz", None)
        if isinstance(bbox, list) and len(bbox) == 4:
            minx, minz, maxx, maxz = map(float, bbox)
            xs = [minx, maxx, maxx, minx, minx]
            zs = [minz, minz, maxz, maxz, minz]
            ax.plot(xs, zs, linewidth=2)

    # Объекты комнаты
    objs = room.get("objects", []) or []
    for obj in objs:
        pos = obj.get("pos", None)
        if not (isinstance(pos, list) and len(pos) >= 3):
            continue
        cx = float(pos[0])
        cz = float(pos[2])

        yaw = obj.get("yaw", 0.0)
        try:
            yaw = float(yaw)
        except Exception:
            yaw = 0.0

        sz = _as_size_xz(obj)
        if sz is None:
            # нет размера — просто точка
            ax.scatter([cx], [cz], s=20)
            if draw_labels:
                ax.text(cx, cz, str(obj.get("instanceid", "")), fontsize=7)
            continue

        sx, szz = sz
        _add_oriented_rect(ax, cx, cz, sx, szz, yaw, linewidth=1.5)

        if draw_labels:
            label = obj.get("title") or obj.get("instanceid") or ""
            if label:
                ax.text(cx, cz, label, fontsize=7)

    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)


def _select_rooms(data: Dict[str, Any], room_id: Optional[str], index: Optional[int], all_rooms: bool) -> List[Dict[str, Any]]:
    rooms = data.get("rooms", []) or []
    if not isinstance(rooms, list):
        return []

    if all_rooms:
        return rooms

    if room_id:
        for r in rooms:
            if str(r.get("room_instanceid", "")) == room_id:
                return [r]
        # если не нашли — вернём пусто
        return []

    if index is not None:
        if 0 <= index < len(rooms):
            return [rooms[index]]
        return []

    # default: первая комната
    return rooms[:1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="processed JSON path")
    ap.add_argument("--room", dest="room_id", default=None, help="room_instanceid (exact match)")
    ap.add_argument("--index", dest="index", type=int, default=None, help="room index in rooms[] (0-based)")
    ap.add_argument("--all", dest="all_rooms", action="store_true", help="render all rooms separately")
    ap.add_argument("--outdir", dest="outdir", default=None, help="save PNGs to this directory")
    ap.add_argument("--show", dest="show", action="store_true", help="show interactive window")
    ap.add_argument("--labels", dest="labels", action="store_true", help="draw text labels for objects")
    ap.add_argument("--dpi", dest="dpi", type=int, default=150)
    args = ap.parse_args()

    inp = Path(args.inp)
    if not inp.exists():
        raise FileNotFoundError(str(inp))

    data = _load_json(inp)
    sel = _select_rooms(data, args.room_id, args.index, args.all_rooms)
    if not sel:
        raise ValueError("Комнаты не выбраны. Проверьте --room/--index или наличие rooms[] в JSON.")

    outdir = Path(args.outdir) if args.outdir else None
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)

    for i, room in enumerate(sel):
        rid = str(room.get("room_instanceid", f"room_{i}"))
        rtype = str(room.get("room_type", ""))
        title = f"{rid}" + (f" | {rtype}" if rtype else "")

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

        _plot_room(ax, room, title=title, draw_labels=args.labels)

        if outdir:
            out_path = outdir / f"{rid}.png"
            fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")

        if args.show:
            plt.show()

        plt.close(fig)


if __name__ == "__main__":
    main()