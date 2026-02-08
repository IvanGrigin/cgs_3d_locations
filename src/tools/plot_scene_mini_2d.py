#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# src/tools/plot_scene_mini_2d.py
#
# Рисует 2D-план (вид сверху) из json-mini, используя ТОЛЬКО прямоугольники:
# - пол: floor_polygon_xz (x,z) рисуем как полигон
# - объекты: bbox_world_xy = [minx, maxx, miny, maxy] (Blender WORLD XY)
#
# Если у какого-то объекта нет bbox_world_xy — это ошибка (скрипт падает),
# потому что вы попросили "используем прямоугольники".
#
# Пример:
# python3 src/tools/plot_scene_mini_2d.py \
#   --mini_json data/output_mini/ffb48a54__LivingDiningRoom-18560.mini.json \
#   --out_png  data/output_mini/ffb48a54__LivingDiningRoom-18560.top.png \
#   --room_id  LivingDiningRoom-18560 \
#   --labels 1
#

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def norm_str(x: Any) -> str:
    return ("" if x is None else str(x)).strip()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_room(data: Dict[str, Any], room_id: Optional[str]) -> Dict[str, Any]:
    rooms = data.get("rooms", []) or []
    if not rooms:
        raise ValueError("rooms пуст в mini.json")
    if room_id is None:
        return rooms[0]
    for r in rooms:
        if norm_str(r.get("id")) == room_id:
            return r
    raise ValueError(f"room_id не найден в mini.json: {room_id}")


def get_floor_poly_xz(room: Dict[str, Any]) -> List[Tuple[float, float]]:
    poly = room.get("floor_polygon_xz", []) or []
    out = []
    for p in poly:
        if "x" in p and "z" in p:
            out.append((float(p["x"]), float(p["z"])))
    return out


def require_bbox_world_xy(obj: Dict[str, Any]) -> Tuple[float, float, float, float]:
    v = obj.get("bbox_world_xy")
    if not (isinstance(v, list) and len(v) == 4):
        name = norm_str(obj.get("name")) or norm_str(obj.get("category")) or norm_str(obj.get("model_id")) or "obj"
        raise ValueError(
            f"У объекта нет bbox_world_xy (нужны прямоугольники): name={name}, keys={list(obj.keys())}"
        )
    minx, maxx, miny, maxy = float(v[0]), float(v[1]), float(v[2]), float(v[3])
    if not (minx <= maxx and miny <= maxy):
        raise ValueError(f"Некорректный bbox_world_xy: {v}")
    return minx, maxx, miny, maxy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mini_json", required=True)
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--room_id", default=None)
    ap.add_argument("--labels", type=int, default=1)
    ap.add_argument("--dpi", type=int, default=220)
    args = ap.parse_args()

    mini_path = Path(args.mini_json).resolve()
    out_png = Path(args.out_png).resolve()
    out_png.parent.mkdir(parents=True, exist_ok=True)

    data = load_json(mini_path)
    room = find_room(data, args.room_id)

    poly = get_floor_poly_xz(room)
    objs = room.get("objects", []) or []

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Пол (polygon_xz): для топ-вида используем (x, z).
    # Важно: bbox_world_xy у объектов — в Blender XY.
    # В вашей схеме to_blender_loc: (x,y,z)->(x,z,y)
    # => Blender Y соответствует исходному z. Поэтому на графике:
    #   ось X = x
    #   ось Y = z  (это Blender Y)
    # => floor_polygon_xz ложится в те же координаты, что bbox_world_xy.
    if poly:
        xs = [p[0] for p in poly] + [poly[0][0]]
        ys = [p[1] for p in poly] + [poly[0][1]]
        ax.plot(xs, ys, linewidth=2)

    # Объекты как прямоугольники (ТОЛЬКО bbox)
    for o in objs:
        name = norm_str(o.get("name")) or norm_str(o.get("category")) or norm_str(o.get("model_id")) or "obj"
        minx, maxx, miny, maxy = require_bbox_world_xy(o)
        w = maxx - minx
        h = maxy - miny
        ax.add_patch(Rectangle((minx, miny), w, h, fill=False, linewidth=1))
        if int(args.labels) == 1:
            ax.text(minx, miny, name, fontsize=6)

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Top view (rectangles): {norm_str(room.get('id'))}")
    ax.set_xlabel("x")
    ax.set_ylabel("z")

    fig.savefig(out_png, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
