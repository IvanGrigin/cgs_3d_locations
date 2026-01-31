from __future__ import annotations
import numpy as np


def _half_norm_xz(room_h: np.ndarray, size_room: np.ndarray, j: int) -> tuple[float, float]:
    """
    half sizes в canonical coords [-1,1] по x,z (как в random_feasible и метриках).
    size_room хранит (sx,sy,sz) в нормализованной системе датасета.
    room_h хранит (hx,hy,hz) — полуразмеры комнаты в той же “канонической” системе.
    """
    hx = (0.5 * float(size_room[j, 0])) / max(float(room_h[0]), 1e-6)
    hz = (0.5 * float(size_room[j, 2])) / max(float(room_h[2]), 1e-6)
    return hx, hz


def _clamp_room(x: float, z: float, hx: float, hz: float) -> tuple[float, float]:
    x = float(np.clip(x, -1.0 + hx, 1.0 - hx))
    z = float(np.clip(z, -1.0 + hz, 1.0 - hz))
    return x, z


def relaxed_cube_layout(
    room_h: np.ndarray,
    size_room: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    iters: int = 260,
    max_step: float = 0.15,
    k_out: float = 0.05,
) -> tuple[np.ndarray, dict]:
    """
    Relaxed placement “на кубах” в canonical coords.
    Выход: pred[N,2] (x,z) + info stats.

    Логика:
      - init около центра
      - итеративное раздвижение по overlap AABB (x,z)
      - лёгкий push out от центра
      - clamp в комнату каждый шаг
    """
    N = int(size_room.shape[0])
    pred = np.zeros((N, 2), dtype=np.float32)

    active = np.where(mask > 0.5)[0]
    info = {
        "placed": 0,
        "fallback_center": 0,  # оставлено для унификации (здесь не используем)
        "unplaceable": 0,
        "total_masked": int(active.size),
        "unplaceable_idx": [],
    }

    # ---- init near center
    for j in active:
        hx, hz = _half_norm_xz(room_h, size_room, int(j))
        if (1.0 - hx) <= (-1.0 + hx) or (1.0 - hz) <= (-1.0 + hz):
            pred[j] = 0.0
            info["unplaceable"] += 1
            info["unplaceable_idx"].append(int(j))
            continue

        x = float(rng.uniform(-0.05, 0.05))
        z = float(rng.uniform(-0.05, 0.05))
        x, z = _clamp_room(x, z, hx, hz)
        pred[j] = (x, z)
        info["placed"] += 1

    # ---- relax iterations
    for _ in range(iters):
        disp = np.zeros_like(pred)

        # pairwise repel on AABB overlap (in x,z)
        for ai in range(active.size):
            i = int(active[ai])
            hi_x, hi_z = _half_norm_xz(room_h, size_room, i)
            xi, zi = float(pred[i, 0]), float(pred[i, 1])

            for aj in range(ai + 1, active.size):
                j = int(active[aj])
                hj_x, hj_z = _half_norm_xz(room_h, size_room, j)
                xj, zj = float(pred[j, 0]), float(pred[j, 1])

                ov_x = (hi_x + hj_x) - abs(xi - xj)
                ov_z = (hi_z + hj_z) - abs(zi - zj)
                if ov_x <= 0.0 or ov_z <= 0.0:
                    continue

                if ov_x < ov_z:
                    s = 1.0 if xi >= xj else -1.0
                    push = 0.6 * ov_x
                    disp[i, 0] += s * 0.5 * push
                    disp[j, 0] -= s * 0.5 * push
                else:
                    s = 1.0 if zi >= zj else -1.0
                    push = 0.6 * ov_z
                    disp[i, 1] += s * 0.5 * push
                    disp[j, 1] -= s * 0.5 * push

        # mild push away from center (0,0)
        for j in active:
            disp[int(j), 0] += k_out * float(pred[int(j), 0])
            disp[int(j), 1] += k_out * float(pred[int(j), 1])

        # apply with max_step + clamp
        for j in active:
            j = int(j)
            dx, dz = float(disp[j, 0]), float(disp[j, 1])
            norm = (dx * dx + dz * dz) ** 0.5
            if norm > max_step and norm > 1e-9:
                s = max_step / norm
                dx *= s
                dz *= s

            hx, hz = _half_norm_xz(room_h, size_room, j)
            x = float(pred[j, 0]) + dx
            z = float(pred[j, 1]) + dz
            x, z = _clamp_room(x, z, hx, hz)
            pred[j] = (x, z)

    return pred, info