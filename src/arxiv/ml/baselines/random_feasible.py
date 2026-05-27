# src/ml/baselines/random_feasible.py
from __future__ import annotations

from typing import Dict, Tuple, Optional, List
import numpy as np


def _norm_half_extents(size_room_xyz: np.ndarray, room_h_xyz: np.ndarray) -> Tuple[float, float]:
    sx = float(size_room_xyz[0])
    sz = float(size_room_xyz[2])
    rhx = float(max(room_h_xyz[0], 1e-8))
    rhz = float(max(room_h_xyz[2], 1e-8))
    hx = 0.5 * sx / rhx
    hz = 0.5 * sz / rhz
    return hx, hz


def _rects_overlap(a_x: float, a_z: float, a_hx: float, a_hz: float,
                   b_x: float, b_z: float, b_hx: float, b_hz: float) -> bool:
    if abs(a_x - b_x) >= (a_hx + b_hx):
        return False
    if abs(a_z - b_z) >= (a_hz + b_hz):
        return False
    return True


def random_feasible_layout(
    room_h: np.ndarray,          # (3,) room half-sizes in world units
    size_room: np.ndarray,       # (N,3) object sizes in world units
    mask: np.ndarray,            # (N,)
    *,
    rng: Optional[np.random.Generator] = None,
    max_trials_per_obj: int = 200,
) -> Tuple[np.ndarray, Dict]:
    """
    Рандомная раскладка в нормализованных координатах [-1,1] с rejection по коллизиям.
    Если объект не влезает по bounds (hx>=1 или hz>=1) — unplaceable -> ставим в центр.
    Если trials исчерпаны — fallback_center.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    room_h = np.asarray(room_h, dtype=np.float32)
    size_room = np.asarray(size_room, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)

    N = size_room.shape[0]
    out = np.zeros((N, 2), dtype=np.float32)

    info = {
        "unplaceable": 0,
        "fallback_center": 0,
        "placed": 0,
        "unplaceable_idx": [],
    }

    placed: List[Tuple[float, float, float, float]] = []  # (x,z,hx,hz)

    for j in range(N):
        if mask[j] <= 0.5:
            continue

        hx, hz = _norm_half_extents(size_room[j], room_h)

        # если объект физически не помещается в [-1,1]
        if hx >= 1.0 or hz >= 1.0:
            out[j] = (0.0, 0.0)
            info["unplaceable"] += 1
            info["unplaceable_idx"].append(j)
            placed.append((0.0, 0.0, hx, hz))
            continue

        lo_x, hi_x = -1.0 + hx, 1.0 - hx
        lo_z, hi_z = -1.0 + hz, 1.0 - hz

        # защитимся от схлопнутых диапазонов
        if hi_x < lo_x or hi_z < lo_z:
            out[j] = (0.0, 0.0)
            info["unplaceable"] += 1
            info["unplaceable_idx"].append(j)
            placed.append((0.0, 0.0, hx, hz))
            continue

        ok = False
        for _ in range(max_trials_per_obj):
            x = float(rng.uniform(lo_x, hi_x))
            z = float(rng.uniform(lo_z, hi_z))

            collision = False
            for (px, pz, phx, phz) in placed:
                if _rects_overlap(x, z, hx, hz, px, pz, phx, phz):
                    collision = True
                    break
            if not collision:
                out[j] = (x, z)
                placed.append((x, z, hx, hz))
                info["placed"] += 1
                ok = True
                break

        if not ok:
            out[j] = (0.0, 0.0)
            placed.append((0.0, 0.0, hx, hz))
            info["fallback_center"] += 1

    return out, info
