#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/tools/room_generator.py

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


# -----------------------------
# Geometry helpers (XY plane)
# -----------------------------

@dataclass(frozen=True)
class Pt:
    x: float
    y: float

def polygon_area_xy(poly: List[Pt]) -> float:
    """Signed area; return abs for area."""
    s = 0.0
    n = len(poly)
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        s += a.x * b.y - b.x * a.y
    return 0.5 * s

def dist(a: Pt, b: Pt) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    return math.hypot(dx, dy)

def scale_polygon(poly: List[Pt], scale: float) -> List[Pt]:
    return [Pt(p.x * scale, p.y * scale) for p in poly]

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# -----------------------------
# Room shape generation
# -----------------------------

def gen_square(target_area: float, rng: random.Random) -> List[Pt]:
    # near-square aspect ratio (but still square by definition)
    side = math.sqrt(target_area)
    # Keep some mild rounding stability
    side = float(side)
    return [
        Pt(0.0, 0.0),
        Pt(side, 0.0),
        Pt(side, side),
        Pt(0.0, side),
    ]

def gen_rectangle(target_area: float, rng: random.Random) -> List[Pt]:
    # Choose aspect ratio in [1.3, 2.5] (clearly rectangular)
    ar = rng.uniform(1.3, 2.5)
    w = math.sqrt(target_area * ar)
    h = target_area / w
    return [
        Pt(0.0, 0.0),
        Pt(w, 0.0),
        Pt(w, h),
        Pt(0.0, h),
    ]

def gen_L_shape(target_area: float, rng: random.Random) -> List[Pt]:
    """
    Make an L-shape as rectangle W x H minus cutout a x b at top-right corner.
    Polygon (CCW):
      (0,0)->(W,0)->(W,H-b)->(W-a,H-b)->(W-a,H)->(0,H)->back
    """
    # Start by picking outer rectangle dims with reasonable aspect
    ar = rng.uniform(1.0, 2.2)
    W = math.sqrt(target_area * ar)
    H = target_area / W

    # Choose cutout fraction so that resulting area matches target_area
    # We'll choose cutout area = outer_area - target_area, i.e. we need outer_area > target_area.
    # So we scale outer rect up a bit randomly, then cut out.
    inflate = rng.uniform(1.15, 1.55)
    W *= inflate
    H *= inflate

    outer_area = W * H
    cut_area = outer_area - target_area

    # Pick cutout sides a, b with constraints
    # Ensure corridors not too thin: remaining legs at least 0.9m.
    min_leg = 0.9
    max_a = max(min_leg, W - min_leg)
    max_b = max(min_leg, H - min_leg)

    # If too small due to random, fallback to rectangle
    if max_a <= min_leg or max_b <= min_leg or cut_area <= 0.0:
        return gen_rectangle(target_area, rng)

    # Choose a and b such that a*b ≈ cut_area
    # Strategy: sample a then compute b, clamp to feasible.
    for _ in range(60):
        a = rng.uniform(min_leg, max_a)
        b = cut_area / a
        if min_leg <= b <= max_b:
            # Good
            break
    else:
        # If can't fit, adjust by scaling polygon to target later
        a = clamp(rng.uniform(min_leg, max_a), min_leg, max_a)
        b = clamp(cut_area / max(a, 1e-6), min_leg, max_b)

    # Build polygon
    poly = [
        Pt(0.0, 0.0),
        Pt(W, 0.0),
        Pt(W, H - b),
        Pt(W - a, H - b),
        Pt(W - a, H),
        Pt(0.0, H),
    ]

    # Due to clamps, area might drift. Fix with uniform scaling to target_area.
    A = abs(polygon_area_xy(poly))
    if A <= 1e-6:
        return gen_rectangle(target_area, rng)
    s = math.sqrt(target_area / A)
    return scale_polygon(poly, s)


# -----------------------------
# Openings placement
# -----------------------------

@dataclass
class WallSeg:
    wall_id: str
    i0: int
    i1: int
    p0: Pt
    p1: Pt
    length: float

def build_walls(poly: List[Pt]) -> List[WallSeg]:
    walls: List[WallSeg] = []
    n = len(poly)
    for i in range(n):
        j = (i + 1) % n
        p0, p1 = poly[i], poly[j]
        w_id = f"w{i}"
        walls.append(WallSeg(wall_id=w_id, i0=i, i1=j, p0=p0, p1=p1, length=dist(p0, p1)))
    return walls

def pick_non_overlapping_interval(
    rng: random.Random,
    L: float,
    width: float,
    margin: float,
    occupied: List[Tuple[float, float]],
    max_tries: int = 200
) -> Optional[float]:
    """
    occupied intervals: list of (s0, s1) along wall.
    Return s for new interval [s, s+width] not overlapping, respecting margins.
    """
    lo = margin
    hi = L - width - margin
    if hi <= lo:
        return None

    for _ in range(max_tries):
        s = rng.uniform(lo, hi)
        s0, s1 = s, s + width
        ok = True
        for a0, a1 in occupied:
            if not (s1 <= a0 or s0 >= a1):
                ok = False
                break
        if ok:
            return s
    return None

def windows_count_by_area(area: float) -> int:
    # Simple heuristic "в зависимости от комнаты"
    if area < 12.0:
        return 1
    if area < 20.0:
        return 2
    return 3


# -----------------------------
# Main generation
# -----------------------------

def weighted_choice(rng: random.Random, items: List[Tuple[str, float]]) -> str:
    s = sum(w for _, w in items)
    if s <= 0:
        return items[0][0]
    r = rng.random() * s
    acc = 0.0
    for name, w in items:
        acc += w
        if r <= acc:
            return name
    return items[-1][0]

def generate_room_dict(seed: int, room_id: str = "room_001") -> Dict:
    rng = random.Random(seed)

    # Target area in [8, 30]
    target_area = rng.uniform(8.0, 30.0)

    # Requested weights: L 0.10, rect 0.40, square 0.40 (sum=0.90) -> normalize
    shape = weighted_choice(rng, [
        ("L", 0.10),
        ("rect", 0.40),
        ("square", 0.40),
    ])

    if shape == "square":
        poly = gen_square(target_area, rng)
    elif shape == "rect":
        poly = gen_rectangle(target_area, rng)
    else:
        poly = gen_L_shape(target_area, rng)

    area = abs(polygon_area_xy(poly))

    # Ceiling height: typical residential range
    ceiling_h = rng.uniform(2.5, 3.1)

    walls = build_walls(poly)

    # Door params
    door_width = rng.uniform(0.80, 0.95)
    door_height = rng.uniform(2.00, min(2.10, ceiling_h - 0.2))
    door_z0 = 0.0

    # Windows: same z0 and height for all windows in room
    win_count = windows_count_by_area(area)
    win_z0 = rng.uniform(0.85, 1.05)
    win_h = rng.uniform(1.10, 1.35)
    # Make sure window fits below ceiling
    if win_z0 + win_h > ceiling_h - 0.15:
        win_h = max(0.8, (ceiling_h - 0.15) - win_z0)

    # Candidate walls for door/windows (need enough length)
    # Provide margins so openings don't touch corners
    margin = 0.25

    long_walls_for_door = [w for w in walls if w.length >= door_width + 2 * margin]
    if not long_walls_for_door:
        # fallback: scale up slightly
        scale = math.sqrt((door_width + 2 * margin) / max(min(w.length for w in walls), 1e-6))
        poly = scale_polygon(poly, scale)
        walls = build_walls(poly)
        long_walls_for_door = [w for w in walls if w.length >= door_width + 2 * margin]

    door_wall = rng.choice(long_walls_for_door)

    # Keep occupied intervals per wall_id to avoid overlaps
    occupied: Dict[str, List[Tuple[float, float]]] = {w.wall_id: [] for w in walls}

    door_s = pick_non_overlapping_interval(
        rng=rng, L=door_wall.length, width=door_width, margin=margin, occupied=occupied[door_wall.wall_id]
    )
    if door_s is None:
        # As last resort, put at margin
        door_s = margin
    occupied[door_wall.wall_id].append((door_s, door_s + door_width))

    # Windows params (vary per-window width a bit, but keep z0/height consistent)
    windows_list = []
    candidate_walls_for_windows = [w for w in walls if w.length >= (1.0 + 2 * margin)]
    if not candidate_walls_for_windows:
        candidate_walls_for_windows = walls[:]  # fallback (then pick smaller windows)

    # Prefer not putting windows on same wall as door, but allow if necessary
    preferred = [w for w in candidate_walls_for_windows if w.wall_id != door_wall.wall_id]
    if preferred:
        candidate_walls_for_windows = preferred

    # Place windows
    for wi in range(win_count):
        win_width = rng.uniform(1.10, 1.80)
        # Ensure fits somewhere; if not, shrink
        # Choose wall; try a few times
        placed = False
        for _ in range(50):
            w = rng.choice(candidate_walls_for_windows)
            if w.length < win_width + 2 * margin:
                # try shrink to fit this wall
                max_w = max(0.7, w.length - 2 * margin)
                if max_w < 0.7:
                    continue
                ww = clamp(win_width, 0.7, max_w)
            else:
                ww = win_width

            s = pick_non_overlapping_interval(
                rng=rng, L=w.length, width=ww, margin=margin, occupied=occupied[w.wall_id]
            )
            if s is None:
                continue

            occupied[w.wall_id].append((s, s + ww))
            windows_list.append({
                "id": f"win_{wi}",
                "wall_id": w.wall_id,
                "s": round(s, 3),
                "width": round(ww, 3),
                "z0": round(win_z0, 3),
                "height": round(win_h, 3),
                "glazing": "double",
            })
            placed = True
            break

        if not placed:
            # fallback: place on any wall with minimal width
            w = max(walls, key=lambda x: x.length)
            ww = clamp(win_width, 0.7, max(0.7, w.length - 2 * margin))
            s = clamp(margin, margin, max(margin, w.length - ww - margin))
            windows_list.append({
                "id": f"win_{wi}",
                "wall_id": w.wall_id,
                "s": round(s, 3),
                "width": round(ww, 3),
                "z0": round(win_z0, 3),
                "height": round(win_h, 3),
                "glazing": "double",
            })

    room = {
        "version": "1.0",
        "units": "m",
        "coordinate_system": {
            "floor_plane": "XY",
            "up": "Z",
            "right_handed": True
        },
        "room": {
            "id": room_id,
            "name": "Generated room",
            "ceiling_height": round(ceiling_h, 3),
            "floor_polygon": [{"x": round(p.x, 3), "y": round(p.y, 3)} for p in poly],
            "walls": [{"id": w.wall_id, "from_vertex": w.i0, "to_vertex": w.i1} for w in walls],
            "doors": [
                {
                    "id": "door_0",
                    "wall_id": door_wall.wall_id,
                    "s": round(door_s, 3),
                    "width": round(door_width, 3),
                    "z0": round(door_z0, 3),
                    "height": round(door_height, 3),
                    "swing": {
                        "hinge": rng.choice(["left", "right"]),
                        "direction": rng.choice(["in", "out"])
                    }
                }
            ],
            "windows": windows_list,
            "meta": {
                "shape": shape,
                "area_m2": round(area, 3),
                "seed": seed
            }
        }
    }
    return room


# -----------------------------
# IO / CLI
# -----------------------------

def save_room_json(room_dict: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(room_dict, f, ensure_ascii=False, indent=2)

def main() -> int:
    ap = argparse.ArgumentParser(description="Generate room.json files (XY floor, Z up).")
    ap.add_argument("--out-dir", default="src/data/input/rooms", help="Output directory for generated rooms.")
    ap.add_argument("--n", type=int, default=10, help="How many rooms to generate.")
    ap.add_argument("--seed", type=int, default=42, help="Base seed.")
    ap.add_argument("--prefix", default="room_", help="File prefix, e.g. room_ -> room_0001.json")
    args = ap.parse_args()

    for i in range(args.n):
        seed = args.seed + i
        room_id = f"{args.prefix}{i:04d}"
        room = generate_room_dict(seed=seed, room_id=room_id)
        out_path = os.path.join(args.out_dir, f"{room_id}.json")
        save_room_json(room, out_path)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())