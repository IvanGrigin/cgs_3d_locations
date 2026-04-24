from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def build_feature_vector(candidate: dict) -> np.ndarray:
    feat = [
        float(candidate["center_norm"][0]),
        float(candidate["center_norm"][1]),
        float(candidate["center_norm"][2]),
        float(candidate["size_m"][0]),
        float(candidate["size_m"][1]),
        float(candidate["size_m"][2]),
        float(candidate["yaw_sin"]),
        float(candidate["yaw_cos"]),
        float(candidate["footprint_area_2d"]),
        float(candidate["bbox_volume_3d"]),
        float(candidate["area_ratio_to_room"]),
        float(candidate["collision_pair_count"]),
        float(candidate["collision_area_sum_2d"]),
        float(candidate["collision_volume_sum_3d"]),
        float(candidate["collision_area_ratio_self_sum"]),
        float(candidate["collision_area_ratio_self_max"]),
        float(candidate["collision_volume_ratio_self_sum"]),
        float(candidate["collision_volume_ratio_self_max"]),
        float(candidate["collision_volume_ratio_minvol_max"]),
        float(candidate["corners_inside_count"]) / 4.0,
        float(candidate["corners_inside_ratio"]),
        1.0 if candidate["center_inside_room"] else 0.0,
        1.0 if candidate["outside_room"] else 0.0,
        float(candidate["floor_contact_abs_error_m"]),
        float(candidate["smaller_than_colliders_count"]),
        float(candidate["larger_than_colliders_count"]),
        1.0 if candidate["is_bad_by_bbox"] else 0.0,
        float(candidate["strongest_collider_center_norm"][0]),
        float(candidate["strongest_collider_center_norm"][1]),
        float(candidate["strongest_collider_center_norm"][2]),
        float(candidate["strongest_collider_size_m"][0]),
        float(candidate["strongest_collider_size_m"][1]),
        float(candidate["strongest_collider_size_m"][2]),
        float(candidate["strongest_collider_center_distance_norm"]),
        float(candidate["strongest_intersection_volume_3d"]),
        float(candidate["strongest_intersection_area_2d"]),
        float(candidate["strongest_collision_volume_ratio_self"]),
        float(candidate["strongest_collision_volume_ratio_minvol"]),
        float(candidate["strongest_collision_area_ratio_self"]),
        float(candidate["strongest_collider_area_ratio_other_to_self"]),
        float(candidate["strongest_collider_volume_ratio_other_to_self"]),
        1.0 if candidate.get("important_furniture_candidate") else 0.0,
        1.0 if candidate.get("wall_anchor_expected") else 0.0,
        1.0 if candidate.get("central_furniture_expected") else 0.0,
        float(candidate.get("nearest_wall_distance_norm", 0.0)),
        float(candidate.get("room_center_distance_norm", 0.0)),
        float(candidate.get("isolated_layout_anomaly_score", 0.0)),
        float(candidate.get("node_suspect_score", 0.0)),
        float(candidate.get("component_size", 0.0)),
        float(candidate.get("component_score", 0.0)),
        float(candidate.get("component_rank_in_scene", 0.0)),
        float(candidate.get("suspect_rank_in_component", 0.0)),
        float(candidate.get("global_anomaly_score", 0.0)),
        float(candidate.get("global_suspect_rank", 0.0)),
    ]
    return np.asarray(feat, dtype=np.float32)


@dataclass
class SelectorVocabs:
    category_vocab: Dict[str, int]
    super_vocab: Dict[str, int]
    mount_vocab: Dict[str, int]
    room_type_vocab: Dict[str, int]


@dataclass
class SelectorBatch:
    features: torch.Tensor
    category: torch.Tensor
    super_category: torch.Tensor
    mount_type: torch.Tensor
    room_type: torch.Tensor
    mask: torch.Tensor
    target_index: torch.Tensor


def _one_hot(value: str, vocab: Dict[str, int]) -> int:
    return int(vocab.get(str(value), 0))


def load_selector_rows(jsonl_path: str, split: str) -> List[dict]:
    rows = []
    with Path(jsonl_path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if split and obj.get("split") != split:
                continue
            rows.append(obj)
    if split and not rows:
        raise RuntimeError(f"No selector rows for split={split} in {jsonl_path}")
    return rows


def build_vocabs(rows: List[dict]) -> SelectorVocabs:
    cats = set()
    supers = set()
    mounts = set()
    room_types = set()
    for row in rows:
        room_types.add(str(row["room_type"]))
        for cand in row["candidates"]:
            cats.add(str(cand["category"]))
            supers.add(str(cand["super_category"]))
            mounts.add(str(cand["mount_type"]))
    return SelectorVocabs(
        category_vocab={name: i + 1 for i, name in enumerate(sorted(cats))},
        super_vocab={name: i + 1 for i, name in enumerate(sorted(supers))},
        mount_vocab={name: i + 1 for i, name in enumerate(sorted(mounts))},
        room_type_vocab={name: i + 1 for i, name in enumerate(sorted(room_types))},
    )


class CorruptedObjectSelectorDatasetV1(Dataset):
    def __init__(self, jsonl_path: str, split: str, vocabs: Optional[SelectorVocabs] = None):
        self.rows = load_selector_rows(jsonl_path, split)
        self.vocabs = vocabs or build_vocabs(self.rows)
        self.feature_dim = int(build_feature_vector(self.rows[0]["candidates"][0]).shape[0])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        features = np.stack([build_feature_vector(c) for c in row["candidates"]], axis=0)
        category = np.asarray([_one_hot(c["category"], self.vocabs.category_vocab) for c in row["candidates"]], dtype=np.int64)
        super_category = np.asarray([_one_hot(c["super_category"], self.vocabs.super_vocab) for c in row["candidates"]], dtype=np.int64)
        mount_type = np.asarray([_one_hot(c["mount_type"], self.vocabs.mount_vocab) for c in row["candidates"]], dtype=np.int64)
        return {
            "features": features,
            "category": category,
            "super_category": super_category,
            "mount_type": mount_type,
            "room_type": np.int64(_one_hot(row["room_type"], self.vocabs.room_type_vocab)),
            "target_index": np.int64(int(row["target_candidate_index"])),
        }


def collate_selector(batch: List[dict]) -> SelectorBatch:
    bsz = len(batch)
    max_n = max(int(b["features"].shape[0]) for b in batch)
    feat_dim = int(batch[0]["features"].shape[1])

    features = np.zeros((bsz, max_n, feat_dim), dtype=np.float32)
    category = np.zeros((bsz, max_n), dtype=np.int64)
    super_category = np.zeros((bsz, max_n), dtype=np.int64)
    mount_type = np.zeros((bsz, max_n), dtype=np.int64)
    room_type = np.zeros((bsz,), dtype=np.int64)
    mask = np.zeros((bsz, max_n), dtype=np.float32)
    target_index = np.zeros((bsz,), dtype=np.int64)

    for i, row in enumerate(batch):
        n = int(row["features"].shape[0])
        features[i, :n] = row["features"]
        category[i, :n] = row["category"]
        super_category[i, :n] = row["super_category"]
        mount_type[i, :n] = row["mount_type"]
        room_type[i] = row["room_type"]
        mask[i, :n] = 1.0
        target_index[i] = row["target_index"]

    return SelectorBatch(
        features=torch.from_numpy(features),
        category=torch.from_numpy(category),
        super_category=torch.from_numpy(super_category),
        mount_type=torch.from_numpy(mount_type),
        room_type=torch.from_numpy(room_type),
        mask=torch.from_numpy(mask),
        target_index=torch.from_numpy(target_index),
    )
