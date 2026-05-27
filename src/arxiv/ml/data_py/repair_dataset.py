from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.ml.data_py.dataset_front import FrontCanonicalDataset


def _safe_cat(cat_id: np.ndarray, idx: int) -> int:
    return int(cat_id[idx]) if 0 <= idx < cat_id.shape[0] else 0


def _rect_from_center_size(pos_xz: np.ndarray, size_xz: np.ndarray) -> Tuple[float, float, float, float]:
    hx = 0.5 * float(size_xz[0])
    hz = 0.5 * float(size_xz[1])
    return (
        float(pos_xz[0] - hx),
        float(pos_xz[0] + hx),
        float(pos_xz[1] - hz),
        float(pos_xz[1] + hz),
    )


def _rect_intersects(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0] or a[3] <= b[2] or b[3] <= a[2])


def _sample_outside_position(
    pos_gt: np.ndarray,
    size_xz: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    hx = 0.5 * float(size_xz[0])
    hz = 0.5 * float(size_xz[1])
    margin_x = rng.uniform(0.02, 0.25)
    margin_z = rng.uniform(0.02, 0.25)
    side = int(rng.integers(0, 4))

    out = pos_gt.copy()
    if side == 0:
        out[0] = float(1.0 + hx + margin_x)
    elif side == 1:
        out[0] = float(-1.0 - hx - margin_x)
    elif side == 2:
        out[1] = float(1.0 + hz + margin_z)
    else:
        out[1] = float(-1.0 - hz - margin_z)

    axis_jitter = rng.normal(0.0, 0.10, size=2).astype(np.float32)
    out = out + axis_jitter
    return out.astype(np.float32)


def _sample_colliding_position(
    pos_gt: np.ndarray,
    size_xz: np.ndarray,
    ctx_pos: np.ndarray,
    ctx_size_xz: np.ndarray,
    ctx_mask: np.ndarray,
    target_idx: int,
    rng: np.random.Generator,
) -> np.ndarray:
    valid = np.where(ctx_mask > 0.5)[0]
    valid = valid[valid != target_idx]
    if valid.size == 0:
        return pos_gt.copy()

    j = int(valid[int(rng.integers(0, valid.size))])
    other_pos = ctx_pos[j].astype(np.float32)
    other_size = ctx_size_xz[j].astype(np.float32)

    half_sum = 0.5 * (size_xz + other_size)
    push = rng.uniform(-0.45, 0.45, size=2).astype(np.float32) * np.maximum(half_sum, 1e-3)
    out = other_pos + push
    return out.astype(np.float32)


def _sample_mixed_position(
    pos_gt: np.ndarray,
    size_xz: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    scale = np.maximum(size_xz * rng.uniform(0.4, 1.8), 0.05).astype(np.float32)
    noise = rng.normal(0.0, 1.0, size=2).astype(np.float32) * scale
    return (pos_gt + noise).astype(np.float32)


def _classify_corruption(
    corrupted_pos: np.ndarray,
    target_size_xz: np.ndarray,
    ctx_pos: np.ndarray,
    ctx_size_xz: np.ndarray,
    ctx_mask: np.ndarray,
    target_idx: int,
) -> np.ndarray:
    rect = _rect_from_center_size(corrupted_pos, target_size_xz)
    hx = 0.5 * float(target_size_xz[0])
    hz = 0.5 * float(target_size_xz[1])

    outside = float(
        corrupted_pos[0] - hx < -1.0 or
        corrupted_pos[0] + hx > 1.0 or
        corrupted_pos[1] - hz < -1.0 or
        corrupted_pos[1] + hz > 1.0
    )

    collision = 0.0
    valid = np.where(ctx_mask > 0.5)[0]
    for j in valid:
        if int(j) == int(target_idx):
            continue
        other_rect = _rect_from_center_size(ctx_pos[j], ctx_size_xz[j])
        if _rect_intersects(rect, other_rect):
            collision = 1.0
            break

    noisy = 1.0 if (outside < 0.5 and collision < 0.5) else 0.0
    return np.array([outside, collision, noisy], dtype=np.float32)


@dataclass
class RepairBatch:
    x0_target: torch.Tensor
    corrupted_target: torch.Tensor
    context_pos: torch.Tensor
    context_size: torch.Tensor
    context_cat: torch.Tensor
    context_mask: torch.Tensor
    target_index: torch.Tensor
    target_cat: torch.Tensor
    target_size: torch.Tensor
    corruption_type: torch.Tensor
    room_h_world: torch.Tensor


class FrontRepairDataset(Dataset):
    """
    Строит supervised-пары для задачи repair одного объекта:
    - x0_target: истинная позиция target-объекта
    - corrupted_target: испорченная позиция target-объекта
    - context_*: остальные объекты фиксированы
    """

    def __init__(
        self,
        npz_path: str,
        splits_path: str,
        split: str = "train",
        samples_per_scene: int = 4,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.base = FrontCanonicalDataset(npz_path=npz_path, splits_path=splits_path, split=split)
        self.samples_per_scene = int(max(1, samples_per_scene))
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.base) * self.samples_per_scene

    def _pick_target_idx(self, mask: np.ndarray, rng: np.random.Generator) -> int:
        valid = np.where(mask > 0.5)[0]
        if valid.size == 0:
            return 0
        return int(valid[int(rng.integers(0, valid.size))])

    def _sample_corrupted(
        self,
        pos_gt: np.ndarray,
        size_xz: np.ndarray,
        ctx_pos: np.ndarray,
        ctx_size_xz: np.ndarray,
        ctx_mask: np.ndarray,
        target_idx: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        mode = int(rng.integers(0, 3))
        if mode == 0:
            return _sample_outside_position(pos_gt, size_xz, rng)
        if mode == 1:
            return _sample_colliding_position(pos_gt, size_xz, ctx_pos, ctx_size_xz, ctx_mask, target_idx, rng)
        return _sample_mixed_position(pos_gt, size_xz, rng)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        base_idx = idx // self.samples_per_scene
        local_seed = self.seed + idx * 10007 + base_idx * 7919
        rng = np.random.default_rng(local_seed)

        item = self.base[base_idx]
        pos = item["pos_gt_xz"].astype(np.float32)
        size_room = item["size_room"].astype(np.float32)
        cat_id = item["cat_id"].astype(np.int64)
        mask = item["mask"].astype(np.float32)
        room_h_world = item["room_h_world"].astype(np.float32)

        target_idx = self._pick_target_idx(mask, rng)
        target_pos = pos[target_idx].copy()
        target_size = size_room[target_idx, [0, 2]].copy()

        corrupted_pos = self._sample_corrupted(
            pos_gt=target_pos,
            size_xz=target_size,
            ctx_pos=pos,
            ctx_size_xz=size_room[:, [0, 2]],
            ctx_mask=mask,
            target_idx=target_idx,
            rng=rng,
        )

        corruption_type = _classify_corruption(
            corrupted_pos=corrupted_pos,
            target_size_xz=target_size,
            ctx_pos=pos,
            ctx_size_xz=size_room[:, [0, 2]],
            ctx_mask=mask,
            target_idx=target_idx,
        )

        context_pos = pos.copy()
        context_pos[target_idx] = corrupted_pos

        return {
            "x0_target": target_pos.astype(np.float32),
            "corrupted_target": corrupted_pos.astype(np.float32),
            "context_pos": context_pos.astype(np.float32),
            "context_size": size_room[:, [0, 2]].astype(np.float32),
            "context_cat": cat_id.astype(np.int64),
            "context_mask": mask.astype(np.float32),
            "target_index": np.int64(target_idx),
            "target_cat": np.int64(_safe_cat(cat_id, target_idx)),
            "target_size": target_size.astype(np.float32),
            "corruption_type": corruption_type.astype(np.float32),
            "room_h_world": room_h_world.astype(np.float32),
        }


def collate_repair(batch: List[Dict[str, np.ndarray]]) -> RepairBatch:
    return RepairBatch(
        x0_target=torch.from_numpy(np.stack([b["x0_target"] for b in batch], axis=0)),
        corrupted_target=torch.from_numpy(np.stack([b["corrupted_target"] for b in batch], axis=0)),
        context_pos=torch.from_numpy(np.stack([b["context_pos"] for b in batch], axis=0)),
        context_size=torch.from_numpy(np.stack([b["context_size"] for b in batch], axis=0)),
        context_cat=torch.from_numpy(np.stack([b["context_cat"] for b in batch], axis=0)),
        context_mask=torch.from_numpy(np.stack([b["context_mask"] for b in batch], axis=0)),
        target_index=torch.from_numpy(np.stack([b["target_index"] for b in batch], axis=0)),
        target_cat=torch.from_numpy(np.stack([b["target_cat"] for b in batch], axis=0)),
        target_size=torch.from_numpy(np.stack([b["target_size"] for b in batch], axis=0)),
        corruption_type=torch.from_numpy(np.stack([b["corruption_type"] for b in batch], axis=0)),
        room_h_world=torch.from_numpy(np.stack([b["room_h_world"] for b in batch], axis=0)),
    )
