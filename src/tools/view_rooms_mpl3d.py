#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from collections import Counter

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


@dataclass(frozen=True)
class RoomKey:
    house_id: str
    room_name: str
    scene_glb: str


@dataclass
class ObjRow:
    object_id: str
    category: str
    uuid: str
    instance: str

    pos_x: float
    pos_y: float
    pos_z: float
    size_x: float
    size_y: float
    size_z: float

    obb_cx: float = float("nan")
    obb_cy: float = float("nan")
    obb_cz: float = float("nan")
    obb_hx: float = float("nan")
    obb_hy: float = float("nan")
    obb_hz: float = float("nan")

    right_x: float = float("nan")
    right_y: float = float("nan")
    right_z: float = float("nan")

    up_x: float = float("nan")
    up_y: float = float("nan")
    up_z: float = float("nan")

    fwd_x: float = float("nan")
    fwd_y: float = float("nan")
    fwd_z: float = float("nan")

    yaw_deg: float = float("nan")


# world (X,Y,Z) -> plot (X,Z,Y)
def world_to_plot(x: float, y: float, z: float) -> Tuple[float, float, float]:
    return (x, z, y)


def world_vec_to_plot(vx: float, vy: float, vz: float) -> Tuple[float, float, float]:
    return (vx, vz, vy)


def _safe_float(row: Dict[str, str], key: str, default: float = float("nan")) -> float:
    if key not in row:
        return default
    s = row.get(key, "")
    if s is None or s == "":
        return default
    try:
        v = float(s)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def read_front_csv(csv_path: Path, *, drop_none: bool) -> Dict[RoomKey, List[ObjRow]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    rooms: Dict[RoomKey, List[ObjRow]] = {}

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])

        required = {
            "house_id", "room_name", "scene_glb",
            "object_id", "category", "uuid", "instance",
            "pos_x", "pos_y", "pos_z", "size_x", "size_y", "size_z",
        }
        missing = required - fieldnames
        if missing:
            raise ValueError(f"CSV missing columns: {sorted(missing)}")

        for row in reader:
            if drop_none:
                if (row.get("category") or "").strip() in ("None.obj", "") and (row.get("object_id") or "").strip() in ("None.obj", ""):
                    continue

            key = RoomKey(row["house_id"], row["room_name"], row["scene_glb"])
            obj = ObjRow(
                object_id=row["object_id"],
                category=row["category"],
                uuid=row.get("uuid", ""),
                instance=row.get("instance", ""),

                pos_x=_safe_float(row, "pos_x"),
                pos_y=_safe_float(row, "pos_y"),
                pos_z=_safe_float(row, "pos_z"),
                size_x=_safe_float(row, "size_x"),
                size_y=_safe_float(row, "size_y"),
                size_z=_safe_float(row, "size_z"),

                obb_cx=_safe_float(row, "obb_cx"),
                obb_cy=_safe_float(row, "obb_cy"),
                obb_cz=_safe_float(row, "obb_cz"),
                obb_hx=_safe_float(row, "obb_hx"),
                obb_hy=_safe_float(row, "obb_hy"),
                obb_hz=_safe_float(row, "obb_hz"),

                right_x=_safe_float(row, "right_x"),
                right_y=_safe_float(row, "right_y"),
                right_z=_safe_float(row, "right_z"),

                up_x=_safe_float(row, "up_x"),
                up_y=_safe_float(row, "up_y"),
                up_z=_safe_float(row, "up_z"),

                fwd_x=_safe_float(row, "fwd_x"),
                fwd_y=_safe_float(row, "fwd_y"),
                fwd_z=_safe_float(row, "fwd_z"),

                yaw_deg=_safe_float(row, "yaw_deg"),
            )
            rooms.setdefault(key, []).append(obj)

    return rooms


def list_rooms(rooms: Dict[RoomKey, List[ObjRow]], limit: int) -> List[RoomKey]:
    keys = sorted(rooms.keys(), key=lambda k: (k.house_id, k.room_name, k.scene_glb))
    print("INDEX | objects | house_id | room_name | scene_glb")
    print("-" * 120)
    for i, k in enumerate(keys[:limit]):
        print(f"{i:5d} | {len(rooms[k]):7d} | {k.house_id} | {k.room_name} | {k.scene_glb}")
    if len(keys) > limit:
        print(f"... ({len(keys) - limit} more)")
    return keys


def _category_color_map(categories: Iterable[str]) -> Dict[str, Tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab20")
    cats = sorted(set(categories))
    out: Dict[str, Tuple[float, float, float, float]] = {}
    for i, c in enumerate(cats):
        out[c] = cmap(i % cmap.N)
    return out


def _is_finite(*vals: float) -> bool:
    return all(math.isfinite(v) for v in vals)


def _has_obb(o: ObjRow) -> bool:
    return _is_finite(
        o.obb_cx, o.obb_cy, o.obb_cz,
        o.obb_hx, o.obb_hy, o.obb_hz,
        o.right_x, o.right_y, o.right_z,
        o.up_x, o.up_y, o.up_z,
        o.fwd_x, o.fwd_y, o.fwd_z,
    )


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _lighten(rgb: Tuple[float, float, float], amount: float) -> Tuple[float, float, float]:
    r, g, b = rgb
    return (_clamp01(r + (1.0 - r) * amount),
            _clamp01(g + (1.0 - g) * amount),
            _clamp01(b + (1.0 - b) * amount))


def _darken(rgb: Tuple[float, float, float], amount: float) -> Tuple[float, float, float]:
    r, g, b = rgb
    return (_clamp01(r * (1.0 - amount)),
            _clamp01(g * (1.0 - amount)),
            _clamp01(b * (1.0 - amount)))


def aabb_faces_plot_from_world_center_size(cx: float, cy: float, cz: float,
                                          sx: float, sy: float, sz: float) -> List[List[Tuple[float, float, float]]]:
    hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
    corners_world = [
        (cx - hx, cy - hy, cz - hz), (cx - hx, cy - hy, cz + hz),
        (cx - hx, cy + hy, cz - hz), (cx - hx, cy + hy, cz + hz),
        (cx + hx, cy - hy, cz - hz), (cx + hx, cy - hy, cz + hz),
        (cx + hx, cy + hy, cz - hz), (cx + hx, cy + hy, cz + hz),
    ]
    v = [world_to_plot(x, y, z) for (x, y, z) in corners_world]
    return [
        [v[0], v[1], v[3], v[2]],  # -X
        [v[4], v[5], v[7], v[6]],  # +X
        [v[0], v[1], v[5], v[4]],  # -Y
        [v[2], v[3], v[7], v[6]],  # +Y
        [v[0], v[2], v[6], v[4]],  # -Z
        [v[1], v[3], v[7], v[5]],  # +Z
    ]


def obb_faces_plot_world(center: Tuple[float, float, float],
                         R: Tuple[float, float, float],
                         U: Tuple[float, float, float],
                         F: Tuple[float, float, float],
                         half: Tuple[float, float, float]) -> Dict[str, List[Tuple[float, float, float]]]:
    cx, cy, cz = center
    rx, ry, rz = R
    ux, uy, uz = U
    fx, fy, fz = F
    hx, hy, hz = half

    def corner(sr: float, su: float, sf: float) -> Tuple[float, float, float]:
        x = cx + sr * hx * rx + su * hy * ux + sf * hz * fx
        y = cy + sr * hx * ry + su * hy * uy + sf * hz * fy
        z = cz + sr * hx * rz + su * hy * uz + sf * hz * fz
        return world_to_plot(x, y, z)

    front = [corner(-1, -1, +1), corner(+1, -1, +1), corner(+1, +1, +1), corner(-1, +1, +1)]
    back  = [corner(-1, -1, -1), corner(+1, -1, -1), corner(+1, +1, -1), corner(-1, +1, -1)]
    right = [corner(+1, -1, -1), corner(+1, -1, +1), corner(+1, +1, +1), corner(+1, +1, -1)]
    left  = [corner(-1, -1, -1), corner(-1, -1, +1), corner(-1, +1, +1), corner(-1, +1, -1)]
    top   = [corner(-1, +1, -1), corner(+1, +1, -1), corner(+1, +1, +1), corner(-1, +1, +1)]
    bottom= [corner(-1, -1, -1), corner(+1, -1, -1), corner(+1, -1, +1), corner(-1, -1, +1)]

    return {"front": front, "back": back, "right": right, "left": left, "top": top, "bottom": bottom}


def set_axes_equal_3d(ax) -> None:
    x0, x1 = ax.get_xlim3d()
    y0, y1 = ax.get_ylim3d()
    z0, z1 = ax.get_zlim3d()
    xr, yr, zr = abs(x1 - x0), abs(y1 - y0), abs(z1 - z0)
    xm, ym, zm = (x0 + x1) / 2.0, (y0 + y1) / 2.0, (z0 + z1) / 2.0
    r = 0.5 * max(xr, yr, zr)
    ax.set_xlim3d([xm - r, xm + r])
    ax.set_ylim3d([ym - r, ym + r])
    ax.set_zlim3d([zm - r, zm + r])
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((1, 1, 1))


def _attach_key_controls(fig, ax, *, default_elev: float, default_azim: float) -> None:
    def _on_key(event):
        step = 5.0
        key = (event.key or "").lower()
        if key == "left":
            ax.view_init(elev=ax.elev, azim=ax.azim - step); fig.canvas.draw_idle()
        elif key == "right":
            ax.view_init(elev=ax.elev, azim=ax.azim + step); fig.canvas.draw_idle()
        elif key == "up":
            ax.view_init(elev=ax.elev + step, azim=ax.azim); fig.canvas.draw_idle()
        elif key == "down":
            ax.view_init(elev=ax.elev - step, azim=ax.azim); fig.canvas.draw_idle()
        elif key == "r":
            ax.view_init(elev=default_elev, azim=default_azim); fig.canvas.draw_idle()
        elif key in ("q", "escape"):
            plt.close(fig)
    fig.canvas.mpl_connect("key_press_event", _on_key)


def _draw_floor(ax, xmin, xmax, ymin, ymax, z_floor: float, *, grid_step: float) -> None:
    # Пол: заметная плита + сетка, но уже на авто-уровне z_floor
    floor = [(xmin, ymin, z_floor), (xmin, ymax, z_floor), (xmax, ymax, z_floor), (xmax, ymin, z_floor)]
    poly = Poly3DCollection(
        [floor],
        facecolors=(0.72, 0.72, 0.72, 0.28),
        edgecolors=(0.15, 0.15, 0.15, 0.75),
        linewidths=1.2,
    )
    ax.add_collection3d(poly)

    if grid_step > 0:
        x = xmin
        while x <= xmax:
            ax.plot([x, x], [ymin, ymax], [z_floor, z_floor], linewidth=0.7, alpha=0.35)
            x += grid_step
        y = ymin
        while y <= ymax:
            ax.plot([xmin, xmax], [y, y], [z_floor, z_floor], linewidth=0.7, alpha=0.35)
            y += grid_step


def plot_room_3d(
    key: RoomKey,
    objects: List[ObjRow],
    *,
    alpha: float,
    pad: float,
    use_obb: bool,
    highlight_front: bool,
    draw_front_arrow: bool,
    front_boost: float,
    back_darken: float,
    show_floor: bool,
    floor_grid_step: float,
    labels: bool,
    label_mode: str,
    label_every: int,
    elev: float,
    azim: float,
) -> None:
    cats = [o.category for o in objects]
    cat2color = _category_color_map(cats)

    # Консоль: используемые объекты/категории
    cat_counts = Counter([o.category for o in objects])
    obj_ids = [o.object_id for o in objects]
    print(f"\nROOM: {key.room_name} | house={key.house_id}")
    print("CATEGORIES (count):")
    for c, n in cat_counts.most_common(30):
        print(f"  {c}: {n}")
    if len(cat_counts) > 30:
        print(f"  ... ({len(cat_counts)-30} more categories)")
    print("OBJECT_IDS (first 60):")
    for s in obj_ids[:60]:
        print(f"  {s}")
    if len(obj_ids) > 60:
        print(f"  ... ({len(obj_ids)-60} more objects)\n")

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    xmin, xmax = float("inf"), float("-inf")
    ymin, ymax = float("inf"), float("-inf")
    zmin_obj, zmax_obj = float("inf"), float("-inf")  # именно по объектам (для пола)

    def update_extents(points_plot: Iterable[Tuple[float, float, float]]):
        nonlocal xmin, xmax, ymin, ymax, zmin_obj, zmax_obj
        for x, y, z in points_plot:
            xmin = min(xmin, x); xmax = max(xmax, x)
            ymin = min(ymin, y); ymax = max(ymax, y)
            zmin_obj = min(zmin_obj, z); zmax_obj = max(zmax_obj, z)

    n_drawn = 0
    label_i = 0

    for o in objects:
        base = cat2color.get(o.category, (0, 0, 0, 1))
        base_rgb = (base[0], base[1], base[2])

        label_text = o.object_id if label_mode == "object_id" else o.category

        if use_obb and _has_obb(o):
            faces_named = obb_faces_plot_world(
                center=(o.obb_cx, o.obb_cy, o.obb_cz),
                R=(o.right_x, o.right_y, o.right_z),
                U=(o.up_x, o.up_y, o.up_z),
                F=(o.fwd_x, o.fwd_y, o.fwd_z),
                half=(max(o.obb_hx, 1e-6), max(o.obb_hy, 1e-6), max(o.obb_hz, 1e-6)),
            )
            order = ["left", "right", "bottom", "top", "back", "front"]
            faces = [faces_named[name] for name in order]

            if highlight_front:
                rgb_front = _lighten(base_rgb, front_boost)
                rgb_back = _darken(base_rgb, back_darken)
                facecolors, edgecolors = [], []
                for name in order:
                    if name == "front":
                        facecolors.append((rgb_front[0], rgb_front[1], rgb_front[2], alpha))
                        edgecolors.append((rgb_front[0], rgb_front[1], rgb_front[2], 1.0))
                    elif name == "back":
                        facecolors.append((rgb_back[0], rgb_back[1], rgb_back[2], alpha))
                        edgecolors.append((rgb_back[0], rgb_back[1], rgb_back[2], 1.0))
                    else:
                        facecolors.append((base_rgb[0], base_rgb[1], base_rgb[2], alpha))
                        edgecolors.append((base_rgb[0], base_rgb[1], base_rgb[2], 1.0))
            else:
                facecolors = [(base_rgb[0], base_rgb[1], base_rgb[2], alpha)] * len(faces)
                edgecolors = [(base_rgb[0], base_rgb[1], base_rgb[2], 1.0)] * len(faces)

            poly = Poly3DCollection(faces, facecolors=facecolors, edgecolors=edgecolors, linewidths=0.7)
            ax.add_collection3d(poly)
            flat = [p for face in faces for p in face]
            update_extents(flat)
            n_drawn += 1

            if draw_front_arrow:
                cpx, cpy, cpz = world_to_plot(o.obb_cx, o.obb_cy, o.obb_cz)
                fx, fy, fz = world_vec_to_plot(o.fwd_x, o.fwd_y, o.fwd_z)
                L = max(0.15, 2.0 * max(o.obb_hz, 0.05))
                ax.quiver(cpx, cpy, cpz, fx, fy, fz, length=L, normalize=True, linewidth=1.0, alpha=0.8)

            # Подпись: над верхней точкой OBB
            if labels and (label_every <= 1 or (label_i % label_every == 0)):
                label_i += 1
                cz_top = max(p[2] for p in flat)
                cx_mid = sum(p[0] for p in flat) / len(flat)
                cy_mid = sum(p[1] for p in flat) / len(flat)
                ax.text(cx_mid, cy_mid, cz_top + 0.03, label_text, fontsize=7)

        else:
            if not _is_finite(o.pos_x, o.pos_y, o.pos_z, o.size_x, o.size_y, o.size_z):
                continue
            sx = max(o.size_x, 1e-6); sy = max(o.size_y, 1e-6); sz = max(o.size_z, 1e-6)
            faces = aabb_faces_plot_from_world_center_size(o.pos_x, o.pos_y, o.pos_z, sx, sy, sz)
            poly = Poly3DCollection(
                faces,
                facecolors=(base_rgb[0], base_rgb[1], base_rgb[2], alpha),
                edgecolors=(base_rgb[0], base_rgb[1], base_rgb[2], 1.0),
                linewidths=0.6,
            )
            ax.add_collection3d(poly)
            flat = [p for face in faces for p in face]
            update_extents(flat)
            n_drawn += 1

            if labels and (label_every <= 1 or (label_i % label_every == 0)):
                label_i += 1
                cz_top = max(p[2] for p in flat)
                cx_mid = sum(p[0] for p in flat) / len(flat)
                cy_mid = sum(p[1] for p in flat) / len(flat)
                ax.text(cx_mid, cy_mid, cz_top + 0.03, label_text, fontsize=7)

    if n_drawn == 0 or not math.isfinite(xmin) or not math.isfinite(zmin_obj):
        raise ValueError("No finite objects to plot (check CSV values).")

    # Пол: по нижней границе объектов (до pad)
    floor_z = zmin_obj

    # Оси показываем с pad
    xmin_p = xmin - pad; xmax_p = xmax + pad
    ymin_p = ymin - pad; ymax_p = ymax + pad
    zmin_p = zmin_obj - pad; zmax_p = zmax_obj + pad

    ax.set_xlim(xmin_p, xmax_p)
    ax.set_ylim(ymin_p, ymax_p)
    ax.set_zlim(zmin_p, zmax_p)

    if show_floor:
        _draw_floor(ax, xmin_p, xmax_p, ymin_p, ymax_p, floor_z, grid_step=float(floor_grid_step))

    ax.set_xlabel("X (floor)")
    ax.set_ylabel("Z (floor)")
    ax.set_zlabel("Y (up)")

    ax.set_title(f"{key.room_name} | house={key.house_id} | objects={len(objects)}")

    ax.view_init(elev=elev, azim=azim)
    set_axes_equal_3d(ax)
    _attach_key_controls(fig, ax, default_elev=elev, default_azim=azim)
    plt.tight_layout()


def _select_room_key(keys_sorted: List[RoomKey], *,
                     index: Optional[int],
                     house_id: Optional[str],
                     room_name: Optional[str],
                     scene_glb: Optional[str]) -> RoomKey:
    if index is not None:
        if index < 0 or index >= len(keys_sorted):
            raise IndexError(f"--index out of range: {index} (0..{len(keys_sorted)-1})")
        return keys_sorted[index]

    candidates = keys_sorted
    if house_id is not None:
        candidates = [k for k in candidates if k.house_id == house_id]
    if room_name is not None:
        candidates = [k for k in candidates if k.room_name == room_name]
    if scene_glb is not None:
        candidates = [k for k in candidates if k.scene_glb == scene_glb]

    if len(candidates) == 0:
        raise ValueError("No rooms matched your filters. Use --list to see available rooms.")
    if len(candidates) > 1:
        print("Multiple rooms matched. Add more filters or use --index. Matches:")
        for k in candidates[:50]:
            print(f"  - house_id={k.house_id} room_name={k.room_name} scene_glb={k.scene_glb}")
        raise SystemExit(2)

    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="3D-Front rooms viewer (matplotlib 3D).")
    parser.add_argument("--csv", type=str, default="src/data/datasets/3D-Front/front_test_layout_obb.csv")

    parser.add_argument("--list", action="store_true")
    parser.add_argument("--limit", type=int, default=30)

    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--house_id", type=str, default=None)
    parser.add_argument("--room_name", type=str, default=None)
    parser.add_argument("--scene_glb", type=str, default=None)

    parser.add_argument("--alpha", type=float, default=0.20)
    parser.add_argument("--pad", type=float, default=0.25)

    parser.add_argument("--obb", action="store_true")
    parser.add_argument("--front", action="store_true")
    parser.add_argument("--front_arrow", action="store_true")
    parser.add_argument("--front_boost", type=float, default=0.45)
    parser.add_argument("--back_darken", type=float, default=0.35)

    # Пол теперь авто по самому низу объектов; параметр floor_y не нужен
    parser.add_argument("--floor", action="store_true")
    parser.add_argument("--floor_grid", type=float, default=0.25)

    parser.add_argument("--labels", action="store_true")
    parser.add_argument("--label_mode", choices=["object_id", "category"], default="object_id")
    parser.add_argument("--label_every", type=int, default=1, help="Label every N-th object (reduce clutter).")

    parser.add_argument("--keep_none", action="store_true")

    parser.add_argument("--elev", type=float, default=25.0)
    parser.add_argument("--azim", type=float, default=-55.0)

    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--dpi", type=int, default=160)

    args = parser.parse_args()

    csv_path = Path(args.csv)
    rooms = read_front_csv(csv_path, drop_none=not args.keep_none)
    keys_sorted = sorted(rooms.keys(), key=lambda k: (k.house_id, k.room_name, k.scene_glb))

    if args.list:
        list_rooms(rooms, limit=args.limit)
        return

    key = _select_room_key(keys_sorted,
                           index=args.index,
                           house_id=args.house_id,
                           room_name=args.room_name,
                           scene_glb=args.scene_glb)

    plot_room_3d(
        key,
        rooms[key],
        alpha=float(args.alpha),
        pad=float(args.pad),
        use_obb=bool(args.obb),
        highlight_front=bool(args.front),
        draw_front_arrow=bool(args.front_arrow),
        front_boost=float(args.front_boost),
        back_darken=float(args.back_darken),
        show_floor=bool(args.floor),
        floor_grid_step=float(args.floor_grid),
        labels=bool(args.labels),
        label_mode=str(args.label_mode),
        label_every=int(max(1, args.label_every)),
        elev=float(args.elev),
        azim=float(args.azim),
    )

    if args.save:
        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=int(args.dpi))
        print(f"Saved: {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
