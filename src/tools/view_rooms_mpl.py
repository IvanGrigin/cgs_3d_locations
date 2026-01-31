#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Визуализация room-файлов (rooms_processed/*.json) в matplotlib.

Ожидаемый формат входного файла — тот, который генерирует export_rooms_dataset.py:
  {
    "room": {"type": "...", "id": "..."},
    "geometry": {"polygon_xz": [[x,z], ...], "floor_y": ..., "ceiling_y": ...},
    "objects": [{"id": "...", "title": "...", ...}, ...],
    "sizes": {"obj_id": {"dx":..., "dy":..., "dz":...}, ...},
    "placements": {"obj_id": {"center": {"x":..., "y":..., "z":...}, "yaw":..., "on_floor":..., ...}, ...}
  }

Использование:
  # один файл, показать
  python3 src/tools/view_rooms_mpl.py --in data/.../rooms_processed/Library-76415.json --show

  # папка с room-файлами, сохранить картинки
  python3 src/tools/view_rooms_mpl.py --in data/.../rooms_processed --save-dir out/rooms_png

Опции:
  --labels: подписывать объекты (id)
  --title-mode: id | type | both
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


def _load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _ensure_closed(poly: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not poly:
        return poly
    if poly[0] != poly[-1]:
        return poly + [poly[0]]
    return poly


def _rotate2(x: float, z: float, yaw: float) -> Tuple[float, float]:
    """Поворот точки (x,z) вокруг (0,0) на yaw (рад), ось up=Y."""
    c = math.cos(yaw)
    s = math.sin(yaw)
    xr = c * x - s * z
    zr = s * x + c * z
    return xr, zr


def _rect_corners_xz(cx: float, cz: float, dx: float, dz: float, yaw: float) -> List[Tuple[float, float]]:
    """
    Ориентированный прямоугольник в плоскости XZ.
    dx — размер вдоль локальной X, dz — вдоль локальной Z.
    """
    hx = 0.5 * dx
    hz = 0.5 * dz
    local = [(-hx, -hz), (hx, -hz), (hx, hz), (-hx, hz)]
    world = []
    for lx, lz in local:
        rx, rz = _rotate2(lx, lz, yaw)
        world.append((cx + rx, cz + rz))
    world.append(world[0])
    return world


def _obj_title_map(room: Dict[str, Any]) -> Dict[str, str]:
    m: Dict[str, str] = {}
    for o in room.get("objects") or []:
        if isinstance(o, dict):
            oid = str(o.get("id") or "")
            title = str(o.get("title") or "")
            if oid:
                m[oid] = title
    return m


def plot_room(room: Dict[str, Any], ax: Any, labels: bool = False, title_mode: str = "both") -> None:
    r = room.get("room") or {}
    rid = str(r.get("id") or "unknown")
    rtype = str(r.get("type") or "Room")

    geom = room.get("geometry") or {}
    poly_in = geom.get("polygon_xz") or []
    poly: List[Tuple[float, float]] = []
    for p in poly_in:
        if isinstance(p, list) and len(p) == 2:
            poly.append((float(p[0]), float(p[1])))
    poly = _ensure_closed(poly)

    # Заголовок
    if title_mode == "id":
        ax.set_title(rid)
    elif title_mode == "type":
        ax.set_title(rtype)
    else:
        ax.set_title(f"{rid} | {rtype}")

    # Полигон комнаты
    if poly:
        xs = [p[0] for p in poly]
        zs = [p[1] for p in poly]
        ax.plot(xs, zs, linewidth=2)

    # Объекты
    sizes: Dict[str, Any] = room.get("sizes") or {}
    placements: Dict[str, Any] = room.get("placements") or {}
    title_map = _obj_title_map(room)

    # Сначала прямоугольники (если есть размеры), затем точки
    for oid, pl in placements.items():
        if not isinstance(pl, dict):
            continue
        c = (pl.get("center") or {})
        cx = float(c.get("x", 0.0))
        cz = float(c.get("z", 0.0))
        yaw = float(pl.get("yaw", 0.0))

        s = sizes.get(oid)
        if isinstance(s, dict) and ("dx" in s) and ("dz" in s):
            dx = float(s["dx"])
            dz = float(s["dz"])
            corners = _rect_corners_xz(cx, cz, dx, dz, yaw)
            xs = [p[0] for p in corners]
            zs = [p[1] for p in corners]
            ax.plot(xs, zs, linestyle="--", linewidth=1)
        else:
            ax.scatter([cx], [cz], s=40)

        if labels:
            # подпись: id (и короткий title, если есть)
            t = title_map.get(str(oid), "")
            short = t.split("/")[-1] if t else ""
            label = f"{oid}" + (f" ({short})" if short else "")
            ax.text(cx, cz, label, fontsize=7)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")


def render_one(inp_path: Path, show: bool, save_path: Optional[Path], labels: bool, title_mode: str) -> None:
    room = _load_json(inp_path)

    fig = plt.figure(figsize=(6, 6), dpi=150)
    ax = fig.add_subplot(1, 1, 1)
    plot_room(room, ax=ax, labels=labels, title_mode=title_mode)

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="room json file or directory with room json files")
    ap.add_argument("--show", action="store_true", help="show matplotlib window")
    ap.add_argument("--save-dir", default=None, help="directory to save PNGs (if omitted, no saving)")
    ap.add_argument("--labels", action="store_true", help="draw text labels for objects")
    ap.add_argument("--title-mode", default="both", choices=["id", "type", "both"], help="plot title mode")
    args = ap.parse_args()

    inp = Path(args.inp)
    if not inp.exists():
        raise FileNotFoundError(str(inp))

    save_dir = Path(args.save_dir) if args.save_dir else None

    if inp.is_file():
        out_png = (save_dir / (inp.stem + ".png")) if save_dir else None
        render_one(inp, show=args.show, save_path=out_png, labels=args.labels, title_mode=args.title_mode)
        return

    # directory
    files = sorted([p for p in inp.glob("*.json") if p.is_file()])
    if not files:
        raise RuntimeError(f"В каталоге нет .json: {inp}")

    for p in files:
        out_png = (save_dir / (p.stem + ".png")) if save_dir else None
        # Для пакетного режима обычно show выключают, иначе окна будут всплывать по одному.
        render_one(p, show=args.show, save_path=out_png, labels=args.labels, title_mode=args.title_mode)

    if save_dir:
        print(f"[ok] saved {len(files)} PNG into: {save_dir}")
    else:
        print(f"[ok] rendered {len(files)} rooms (no saving)")


if __name__ == "__main__":
    main()