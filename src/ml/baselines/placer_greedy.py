# src/ml/baselines/placer_greedy.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np


def _norm_half_extents(size_room_xyz: np.ndarray, room_h_xyz: np.ndarray) -> Tuple[float, float]:
    """
    size_room_xyz: (3,) in world units (sx, sy, sz)
    room_h_xyz   : (3,) half-sizes of room in world units (hx, hy, hz)
    returns (hx_norm, hz_norm) where coords are in [-1,1]
    """
    sx = float(size_room_xyz[0])
    sz = float(size_room_xyz[2])
    rhx = float(max(room_h_xyz[0], 1e-8))
    rhz = float(max(room_h_xyz[2], 1e-8))
    hx = 0.5 * sx / rhx
    hz = 0.5 * sz / rhz
    return hx, hz


def _clamp_to_bounds(x: float, z: float, hx: float, hz: float) -> Tuple[float, float]:
    """
    Valid bounds in normalized coords:
      x in [-1+hx, 1-hx]
      z in [-1+hz, 1-hz]
    """
    lo_x, hi_x = -1.0 + hx, 1.0 - hx
    lo_z, hi_z = -1.0 + hz, 1.0 - hz

    # если объект почти не влезает — диапазон может схлопнуться
    if hi_x < lo_x:
        x = 0.0
    else:
        x = float(np.clip(x, lo_x, hi_x))

    if hi_z < lo_z:
        z = 0.0
    else:
        z = float(np.clip(z, lo_z, hi_z))
    return x, z


def _rects_overlap(a_x: float, a_z: float, a_hx: float, a_hz: float,
                   b_x: float, b_z: float, b_hx: float, b_hz: float) -> bool:
    """
    AABB overlap in XZ normalized space.
    """
    if abs(a_x - b_x) >= (a_hx + b_hx):
        return False
    if abs(a_z - b_z) >= (a_hz + b_hz):
        return False
    return True


def greedy_place(
    pred_xz: np.ndarray,         # (N,2) normalized [-1,1]
    size_room: np.ndarray,       # (N,3) world units
    mask: np.ndarray,            # (N,) 0/1
    room_h: Optional[np.ndarray] = None,   # (3,) world half-sizes; if None -> assume 1,1,1
    *,
    rng: Optional[np.random.Generator] = None,
    step: float = 0.06,
    rings: int = 12,
    jitter: float = 0.015,
) -> Tuple[np.ndarray, Dict]:
    """
    Делает корректную (по границам) и более-менее неколлидирующую раскладку.

    Поиск кандидатов: вокруг таргет-точки (x0,z0) по квадратным "кольцам"
    + небольшой джиттер (если rng задан), чтобы не залипать в сетку.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    pred_xz = np.asarray(pred_xz, dtype=np.float32)
    size_room = np.asarray(size_room, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)

    N = pred_xz.shape[0]
    out = pred_xz.copy()

    if room_h is None:
        room_h = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    room_h = np.asarray(room_h, dtype=np.float32)

    active: List[int] = [i for i in range(N) if mask[i] > 0.5]

    # сортируем по площади основания (sx*sz) по убыванию
    areas = []
    for i in active:
        sx = float(size_room[i, 0])
        sz = float(size_room[i, 2])
        areas.append((sx * sz, i))
    areas.sort(reverse=True)
    order = [i for _, i in areas]

    placed = []  # список уже зафиксированных объектов: (x,z,hx,hz, idx)
    info = {
        "placed": 0,
        "fallback": 0,
        "unplaceable": 0,
        "unplaceable_idx": [],
    }

    for i in order:
        hx, hz = _norm_half_extents(size_room[i], room_h)

        # если объект физически не влезает в нормализованные пределы
        if hx >= 1.0 or hz >= 1.0:
            out[i, 0] = 0.0
            out[i, 1] = 0.0
            info["unplaceable"] += 1
            info["unplaceable_idx"].append(i)
            placed.append((0.0, 0.0, hx, hz, i))
            continue

        x0 = float(out[i, 0])
        z0 = float(out[i, 1])
        x0, z0 = _clamp_to_bounds(x0, z0, hx, hz)

        # генерим кандидатов: центр + кольца
        best = None

        # стартовый кандидат
        cand_list = [(x0, z0)]

        for r in range(1, rings + 1):
            d = step * r
            # периметр квадрата (8r точек по сути), но проще собрать сетку границы
            xs = [x0 - d, x0, x0 + d]
            zs = [z0 - d, z0, z0 + d]
            # верх/низ
            for x in np.linspace(x0 - d, x0 + d, num=2 * r + 1):
                cand_list.append((float(x), z0 - d))
                cand_list.append((float(x), z0 + d))
            # лево/право (без углов, чтобы не дублировать)
            for z in np.linspace(z0 - d, z0 + d, num=2 * r + 1)[1:-1]:
                cand_list.append((x0 - d, float(z)))
                cand_list.append((x0 + d, float(z)))

        # проверяем кандидатов
        for (x, z) in cand_list:
            # джиттер (чуть помогает, когда всё упирается в одинаковые сеточные точки)
            if jitter > 0:
                x = x + float(rng.uniform(-jitter, jitter))
                z = z + float(rng.uniform(-jitter, jitter))

            x, z = _clamp_to_bounds(x, z, hx, hz)

            ok = True
            for (px, pz, phx, phz, _) in placed:
                if _rects_overlap(x, z, hx, hz, px, pz, phx, phz):
                    ok = False
                    break
            if ok:
                best = (x, z)
                break

        if best is None:
            # не нашли — оставляем clamped таргет (будет коллизия, но хоть в границах)
            out[i, 0] = x0
            out[i, 1] = z0
            info["fallback"] += 1
            placed.append((x0, z0, hx, hz, i))
        else:
            out[i, 0] = best[0]
            out[i, 1] = best[1]
            placed.append((best[0], best[1], hx, hz, i))
            info["placed"] += 1

    return out, info
