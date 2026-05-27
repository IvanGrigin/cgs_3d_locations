from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class FrontBatch:
    pos_gt_xz: torch.Tensor      # [B,N,2]
    size_room: torch.Tensor      # [B,N,3]  (normalized)
    cat_id: torch.Tensor         # [B,N]
    mask: torch.Tensor           # [B,N]
    room_h: torch.Tensor         # [B,3]    (canonical: ones)
    room_h_world: torch.Tensor   # [B,3]    (meters)
    room_c_world: torch.Tensor   # [B,3]
    room_axes_world: torch.Tensor # [B,9]


class FrontCanonicalDataset(Dataset):
    def __init__(self, npz_path: str, splits_path: str, split: str = "train"):
        self.npz_path = str(npz_path)
        self.splits_path = str(splits_path)
        self.split = split

        z = np.load(self.npz_path)
        self.pos_gt_xz = z["pos_gt_xz"].astype(np.float32)
        self.size_room = z["size_room"].astype(np.float32)
        self.cat_id = z["cat_id"].astype(np.int64)
        self.mask = z["mask"].astype(np.uint8)
        self.room_h = z["room_h"].astype(np.float32)
        self.room_h_world = z["room_h_world"].astype(np.float32)
        self.room_c_world = z["room_c_world"].astype(np.float32)
        self.room_axes_world = z["room_axes_world"].astype(np.float32)

        with open(self.splits_path, "r", encoding="utf-8") as f:
            splits = json.load(f)

        idx = splits.get(split)
        if idx is None:
            raise ValueError(f"Unknown split={split}. Available: {list(splits.keys())}")

        self.indices = np.array(idx, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, i: int) -> Dict[str, np.ndarray]:
        k = int(self.indices[i])
        return {
            "pos_gt_xz": self.pos_gt_xz[k],
            "size_room": self.size_room[k],
            "cat_id": self.cat_id[k],
            "mask": self.mask[k],
            "room_h": self.room_h[k],
            "room_h_world": self.room_h_world[k],
            "room_c_world": self.room_c_world[k],
            "room_axes_world": self.room_axes_world[k],
        }


def collate_front(batch: List[Dict[str, np.ndarray]]) -> FrontBatch:
    pos_gt_xz = torch.from_numpy(np.stack([b["pos_gt_xz"] for b in batch], axis=0))
    size_room = torch.from_numpy(np.stack([b["size_room"] for b in batch], axis=0))
    cat_id = torch.from_numpy(np.stack([b["cat_id"] for b in batch], axis=0))
    mask = torch.from_numpy(np.stack([b["mask"] for b in batch], axis=0)).float()
    room_h = torch.from_numpy(np.stack([b["room_h"] for b in batch], axis=0))
    room_h_world = torch.from_numpy(np.stack([b["room_h_world"] for b in batch], axis=0))
    room_c_world = torch.from_numpy(np.stack([b["room_c_world"] for b in batch], axis=0))
    room_axes_world = torch.from_numpy(np.stack([b["room_axes_world"] for b in batch], axis=0))

    return FrontBatch(
        pos_gt_xz=pos_gt_xz,
        size_room=size_room,
        cat_id=cat_id,
        mask=mask,
        room_h=room_h,
        room_h_world=room_h_world,
        room_c_world=room_c_world,
        room_axes_world=room_axes_world,
    )
