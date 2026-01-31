# src/ml/metrics/layout_metrics.py
from __future__ import annotations

import numpy as np


def rmse_xz(pred_xz: np.ndarray, gt_xz: np.ndarray, mask: np.ndarray) -> float:
    """
    pred_xz, gt_xz: [B,N,2]
    mask: [B,N] float {0,1}
    """
    diff2 = (pred_xz - gt_xz) ** 2
    diff2 = diff2.sum(axis=-1)  # [B,N]
    w = mask
    denom = np.maximum(w.sum(), 1e-6)
    return float(np.sqrt((diff2 * w).sum() / denom))


def mae_xz(pred_xz: np.ndarray, gt_xz: np.ndarray, mask: np.ndarray) -> float:
    diff = np.abs(pred_xz - gt_xz).sum(axis=-1)  # L1 over x,z
    w = mask
    denom = np.maximum(w.sum(), 1e-6)
    return float((diff * w).sum() / denom)


def boundary_violation_rate(
    pred_xz: np.ndarray,
    size_room_aabb: np.ndarray,
    room_h: np.ndarray,
    mask: np.ndarray,
) -> float:
    """
    pred_xz: [B,N,2] in normalized coords
    size_room_aabb: [B,N,3] full sizes in room axes (meters)
    room_h: [B,3] (hx,hy,hz) meters
    """
    hx_room = np.maximum(room_h[:, 0:1], 1e-6)  # [B,1]
    hz_room = np.maximum(room_h[:, 2:3], 1e-6)  # [B,1]

    half_x = (size_room_aabb[..., 0] * 0.5) / hx_room  # [B,N]
    half_z = (size_room_aabb[..., 2] * 0.5) / hz_room

    x = pred_xz[..., 0]
    z = pred_xz[..., 1]

    ex = np.maximum(np.abs(x) + half_x - 1.0, 0.0)
    ez = np.maximum(np.abs(z) + half_z - 1.0, 0.0)

    viol = ((ex + ez) > 0).astype(np.float32) * mask
    denom = np.maximum(mask.sum(), 1e-6)
    return float(viol.sum() / denom)


def collision_pair_rate(
    pred_xz: np.ndarray,
    size_room_aabb: np.ndarray,
    room_h: np.ndarray,
    mask: np.ndarray,
) -> float:
    """
    Возвращает долю пересекающихся пар среди валидных объектов (по маске).
    Считаем 2D AABB на полу в норм. координатах.
    """
    B, N, _ = pred_xz.shape
    hx_room = np.maximum(room_h[:, 0], 1e-6)  # [B]
    hz_room = np.maximum(room_h[:, 2], 1e-6)

    half_x = (size_room_aabb[..., 0] * 0.5) / hx_room[:, None]  # [B,N]
    half_z = (size_room_aabb[..., 2] * 0.5) / hz_room[:, None]

    x = pred_xz[..., 0]
    z = pred_xz[..., 1]

    total_pairs = 0.0
    coll_pairs = 0.0

    for b in range(B):
        idx = np.where(mask[b] > 0.5)[0]
        m = len(idx)
        if m < 2:
            continue
        total_pairs += m * (m - 1) / 2

        xb = x[b, idx]
        zb = z[b, idx]
        hxb = half_x[b, idx]
        hzb = half_z[b, idx]

        # pairwise overlap check
        for i in range(m):
            for j in range(i + 1, m):
                if (abs(xb[i] - xb[j]) < (hxb[i] + hxb[j])) and (abs(zb[i] - zb[j]) < (hzb[i] + hzb[j])):
                    coll_pairs += 1

    if total_pairs < 1e-6:
        return 0.0
    return float(coll_pairs / total_pairs)
