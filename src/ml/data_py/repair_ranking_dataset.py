from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def _one_hot(value: str, vocab: Dict[str, int]) -> int:
    return int(vocab.get(str(value), 0))


def build_feature_vector(candidate: dict) -> np.ndarray:
    cm = candidate["candidate_metrics"]
    lbl = candidate["label"]
    feat = [
        float(candidate["heuristic_score"]),
        float(candidate["candidate_norm_xy"][0]),
        float(candidate["candidate_norm_xy"][1]),
        float(candidate["corrupted_norm_xy"][0]),
        float(candidate["corrupted_norm_xy"][1]),
        float(candidate["candidate_delta_from_corrupted_roomnorm"][0]),
        float(candidate["candidate_delta_from_corrupted_roomnorm"][1]),
        float(candidate["candidate_delta_from_corrupted_roomnorm"][2]),
        float(candidate["candidate_dyaw_deg_from_corrupted"]) / 180.0,
        float(candidate["candidate_yaw_sin"]),
        float(candidate["candidate_yaw_cos"]),
        float(candidate["target_size_m"][0]),
        float(candidate["target_size_m"][1]),
        float(candidate["target_size_m"][2]),
        float(cm["collision_pair_count"]),
        float(cm["collision_area_sum_2d"]),
        float(cm["collision_volume_sum_3d"]),
        float(cm["corners_inside_ratio"]),
        1.0 if cm["center_inside_room"] else 0.0,
        1.0 if cm["outside_room"] else 0.0,
        float(cm["floor_contact_abs_error_m"]),
        1.0 if cm["valid"] else 0.0,
    ]
    return np.asarray(feat, dtype=np.float32)


@dataclass
class RankingBatch:
    features: torch.Tensor
    target_cat: torch.Tensor
    target_super: torch.Tensor
    corruption_type: torch.Tensor
    room_type: torch.Tensor
    target_score: torch.Tensor
    target_best: torch.Tensor


@dataclass
class RankingVocabs:
    target_cat_vocab: Dict[str, int]
    target_super_vocab: Dict[str, int]
    corruption_vocab: Dict[str, int]
    room_type_vocab: Dict[str, int]


def load_ranking_rows(jsonl_path: str, split: str) -> List[dict]:
    rows = []
    with Path(jsonl_path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("split") == split:
                rows.append(obj)
    if not rows:
        raise RuntimeError(f"No ranking rows for split={split} in {jsonl_path}")
    return rows


def build_vocabs(rows: List[dict]) -> RankingVocabs:
    target_categories = sorted({r["target_category"] for r in rows})
    target_supers = sorted({r["target_super_category"] for r in rows})
    corruption_types = sorted({r["corruption_type"] for r in rows})
    room_types = sorted({r["room_type"] for r in rows})
    return RankingVocabs(
        target_cat_vocab={name: i + 1 for i, name in enumerate(target_categories)},
        target_super_vocab={name: i + 1 for i, name in enumerate(target_supers)},
        corruption_vocab={name: i + 1 for i, name in enumerate(corruption_types)},
        room_type_vocab={name: i + 1 for i, name in enumerate(room_types)},
    )


class RepairRankingDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        split: str,
        seed: int = 42,
        vocabs: Optional[RankingVocabs] = None,
    ):
        self.jsonl_path = str(jsonl_path)
        self.split = str(split)
        self.seed = int(seed)

        rows = load_ranking_rows(self.jsonl_path, self.split)
        self.rows = rows
        self.vocabs = vocabs or build_vocabs(rows)
        self.target_cat_vocab = self.vocabs.target_cat_vocab
        self.target_super_vocab = self.vocabs.target_super_vocab
        self.corruption_vocab = self.vocabs.corruption_vocab
        self.room_type_vocab = self.vocabs.room_type_vocab

        flat_items = []
        for row in rows:
            best_idx = int(row["best_candidate_index"])
            for idx, cand in enumerate(row["candidates"]):
                flat_items.append(
                    {
                        "sample_id": row["sample_id"],
                        "candidate_index": idx,
                        "target_category": row["target_category"],
                        "target_super_category": row["target_super_category"],
                        "corruption_type": row["corruption_type"],
                        "room_type": row["room_type"],
                        "features": build_feature_vector(cand),
                        "target_score": float(cand["label"]["quality_score"]),
                        "target_best": 1.0 if idx == best_idx else 0.0,
                    }
                )
        self.items = flat_items
        self.feature_dim = int(self.items[0]["features"].shape[0])

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        return {
            "features": item["features"],
            "target_cat": np.int64(_one_hot(item["target_category"], self.target_cat_vocab)),
            "target_super": np.int64(_one_hot(item["target_super_category"], self.target_super_vocab)),
            "corruption_type": np.int64(_one_hot(item["corruption_type"], self.corruption_vocab)),
            "room_type": np.int64(_one_hot(item["room_type"], self.room_type_vocab)),
            "target_score": np.float32(item["target_score"]),
            "target_best": np.float32(item["target_best"]),
        }


def collate_ranking(batch: List[dict]) -> RankingBatch:
    return RankingBatch(
        features=torch.from_numpy(np.stack([b["features"] for b in batch], axis=0)),
        target_cat=torch.from_numpy(np.stack([b["target_cat"] for b in batch], axis=0)),
        target_super=torch.from_numpy(np.stack([b["target_super"] for b in batch], axis=0)),
        corruption_type=torch.from_numpy(np.stack([b["corruption_type"] for b in batch], axis=0)),
        room_type=torch.from_numpy(np.stack([b["room_type"] for b in batch], axis=0)),
        target_score=torch.from_numpy(np.stack([b["target_score"] for b in batch], axis=0)),
        target_best=torch.from_numpy(np.stack([b["target_best"] for b in batch], axis=0)),
    )
