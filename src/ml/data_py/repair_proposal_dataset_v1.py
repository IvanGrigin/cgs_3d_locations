from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def as_str(v: Any, default: str = "") -> str:
    if v is None:
        return default
    return str(v)


def wrap_angle_deg(angle: float) -> float:
    return float((angle + 180.0) % 360.0 - 180.0)


@dataclass
class ProposalVocabs:
    category_vocab: Dict[str, int]
    corruption_vocab: Dict[str, int]
    room_type_vocab: Dict[str, int]


@dataclass
class ProposalBatch:
    clean_pose: torch.Tensor
    corrupted_pose: torch.Tensor
    context_pos: torch.Tensor
    context_size: torch.Tensor
    context_cat: torch.Tensor
    context_mask: torch.Tensor
    target_index: torch.Tensor
    target_cat: torch.Tensor
    target_size: torch.Tensor
    corruption_type: torch.Tensor
    room_type: torch.Tensor
    room_scale: torch.Tensor
    corrupted_flags: torch.Tensor


def normalize_xy(x: float, y: float, room_json: Dict[str, Any]) -> Tuple[float, float]:
    bounds = room_json["bounds_xz"]
    x_min = float(bounds["x_min"])
    x_max = float(bounds["x_max"])
    y_min = float(bounds["z_min"])
    y_max = float(bounds["z_max"])
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    hx = max(0.5 * (x_max - x_min), 1e-6)
    hy = max(0.5 * (y_max - y_min), 1e-6)
    return ((float(x) - cx) / hx, (float(y) - cy) / hy)


def denormalize_xy(nx: float, ny: float, room_json: Dict[str, Any]) -> Tuple[float, float]:
    bounds = room_json["bounds_xz"]
    x_min = float(bounds["x_min"])
    x_max = float(bounds["x_max"])
    y_min = float(bounds["z_min"])
    y_max = float(bounds["z_max"])
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    hx = max(0.5 * (x_max - x_min), 1e-6)
    hy = max(0.5 * (y_max - y_min), 1e-6)
    return (float(cx + nx * hx), float(cy + ny * hy))


def normalize_size_xy(size_x: float, size_y: float, room_json: Dict[str, Any]) -> Tuple[float, float]:
    bounds = room_json["bounds_xz"]
    room_w = max(float(bounds["x_max"]) - float(bounds["x_min"]), 1e-6)
    room_h = max(float(bounds["z_max"]) - float(bounds["z_min"]), 1e-6)
    return (float(size_x / room_w), float(size_y / room_h))


def room_scale_features(room_json: Dict[str, Any], target_size: List[float]) -> np.ndarray:
    bounds = room_json["bounds_xz"]
    room_w = max(float(bounds["x_max"]) - float(bounds["x_min"]), 1e-6)
    room_h = max(float(bounds["z_max"]) - float(bounds["z_min"]), 1e-6)
    return np.asarray([room_w, room_h, float(target_size[2])], dtype=np.float32)


def pose_to_model(position_m: List[float], yaw_deg: float, room_json: Dict[str, Any]) -> np.ndarray:
    nx, ny = normalize_xy(float(position_m[0]), float(position_m[1]), room_json)
    yaw_rad = math.radians(float(yaw_deg))
    return np.asarray([nx, ny, float(position_m[2]), math.sin(yaw_rad), math.cos(yaw_rad)], dtype=np.float32)


def model_pose_to_world(
    pose_model: np.ndarray,
    room_json: Dict[str, Any],
    fallback_z: float,
) -> Tuple[List[float], float, float]:
    nx = float(np.clip(pose_model[0], -1.25, 1.25))
    ny = float(np.clip(pose_model[1], -1.25, 1.25))
    x, y = denormalize_xy(nx, ny, room_json)
    z = float(pose_model[2]) if np.isfinite(float(pose_model[2])) else float(fallback_z)
    yaw_sin = float(pose_model[3])
    yaw_cos = float(pose_model[4])
    norm = math.hypot(yaw_sin, yaw_cos)
    if norm < 1e-6:
        yaw_deg = 0.0
    else:
        yaw_deg = math.degrees(math.atan2(yaw_sin / norm, yaw_cos / norm))
    return [float(x), float(y), float(z)], float(wrap_angle_deg(yaw_deg)), float(math.radians(wrap_angle_deg(yaw_deg)))


def load_sample_rows(jsonl_path: str, split: Optional[str] = None, limit: int = 0) -> List[dict]:
    rows: List[dict] = []
    with Path(jsonl_path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if split and as_str(row.get("split")).strip().lower() != split:
                continue
            rows.append(row)
            if limit > 0 and len(rows) >= int(limit):
                break
    if split and not rows:
        raise RuntimeError(f"No repair proposal rows for split={split} in {jsonl_path}")
    return rows


def reconstruct_corrupted_scene(sample: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    scene_gt = load_json(sample["clean_scene_ref"])
    placements = json.loads(json.dumps(scene_gt["placements"]))
    target_id = as_str(sample["target_object_id"])
    clean_target = None
    corrupted_scene: List[Dict[str, Any]] = []
    for p in placements:
        if as_str(p["id"]) == target_id:
            clean_target = json.loads(json.dumps(p))
            corr = json.loads(json.dumps(p))
            corr_pose = sample["corrupted_pose"]
            corr["position_m"] = list(corr_pose["position_m"])
            corr["yaw_deg"] = float(corr_pose["yaw_deg"])
            corr["yaw_rad"] = float(corr_pose["yaw_rad"])
            corr["rotation_deg"] = int(round(float(corr["yaw_deg"]))) % 360
            corrupted_scene.append(corr)
        else:
            corrupted_scene.append(json.loads(json.dumps(p)))
    if clean_target is None:
        raise RuntimeError(f"target_id={target_id} not found in clean scene")
    return scene_gt, corrupted_scene, clean_target


def build_vocabs(rows: List[dict]) -> ProposalVocabs:
    corruption_types = sorted({as_str(r["corruption_type"]) for r in rows})
    room_types = sorted({as_str(r["room_type"]) for r in rows})
    categories = set(as_str(r["target_category"]) for r in rows)

    unique_scene_refs = sorted({as_str(r["clean_scene_ref"]) for r in rows})
    for scene_ref in unique_scene_refs:
        scene = load_json(scene_ref)
        for p in scene.get("placements", []):
            categories.add(as_str(p.get("category")))

    return ProposalVocabs(
        category_vocab={name: i + 1 for i, name in enumerate(sorted(categories))},
        corruption_vocab={name: i + 1 for i, name in enumerate(corruption_types)},
        room_type_vocab={name: i + 1 for i, name in enumerate(room_types)},
    )


def room_json_from_scene(scene: Dict[str, Any]) -> Dict[str, Any]:
    room = scene.get("room") or {}
    floor_polygon = room.get("floor_polygon")
    polygon_xy: List[List[float]]
    if isinstance(floor_polygon, list) and floor_polygon:
        if isinstance(floor_polygon[0], dict):
            polygon_xy = [[float(p["x"]), float(p["y"])] for p in floor_polygon]
        else:
            polygon_xy = [[float(p[0]), float(p[1])] for p in floor_polygon]
    else:
        width = float(room.get("width_m", room.get("width", 0.0)))
        depth = float(room.get("depth_m", room.get("depth", 0.0)))
        x_min = float(room.get("x_min", 0.0))
        y_min = float(room.get("y_min", 0.0))
        x_max = float(room.get("x_max", x_min + width))
        y_max = float(room.get("y_max", y_min + depth))
        polygon_xy = [
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max],
        ]
    xs = [float(p[0]) for p in polygon_xy]
    ys = [float(p[1]) for p in polygon_xy]
    return {
        "bounds_xz": {
            "x_min": min(xs),
            "x_max": max(xs),
            "z_min": min(ys),
            "z_max": max(ys),
        },
        "floor_polygon_xz": polygon_xy,
    }


def category_id(name: str, vocab: Dict[str, int]) -> int:
    return int(vocab.get(as_str(name), 0))


def sample_corrupted_flags(sample: Dict[str, Any], room_json: Dict[str, Any], target_size: List[float]) -> np.ndarray:
    cm = sample["corrupted_metrics"]
    bounds = room_json["bounds_xz"]
    room_w = max(float(bounds["x_max"]) - float(bounds["x_min"]), 1e-6)
    room_h = max(float(bounds["z_max"]) - float(bounds["z_min"]), 1e-6)
    target_area = max(float(target_size[0]) * float(target_size[1]), 1e-6)
    return np.asarray(
        [
            float(cm["collision_pair_count"]) / 8.0,
            float(cm["collision_area_sum_2d"]) / target_area,
            1.0 if cm["outside_room"] else 0.0,
            float(cm["corners_inside_ratio"]),
            float(cm["floor_contact_abs_error_m"]) / 0.25,
            room_w,
            room_h,
        ],
        dtype=np.float32,
    )


def encode_scene_target(
    target_id: str,
    target_category: str,
    corruption_type: str,
    room_type: str,
    corrupted_metrics: Dict[str, Any],
    room_json: Dict[str, Any],
    corrupted_scene: List[Dict[str, Any]],
    vocabs: ProposalVocabs,
) -> Dict[str, np.ndarray]:
    target_index = -1
    context_pos: List[List[float]] = []
    context_size: List[List[float]] = []
    context_cat: List[int] = []
    for idx, p in enumerate(corrupted_scene):
        if as_str(p["id"]) == target_id:
            target_index = idx
        nx, ny = normalize_xy(float(p["position_m"][0]), float(p["position_m"][1]), room_json)
        sx, sy = normalize_size_xy(float(p["size_m"][0]), float(p["size_m"][1]), room_json)
        context_pos.append([nx, ny, float(p["position_m"][2])])
        context_size.append([sx, sy, float(p["size_m"][2])])
        context_cat.append(category_id(as_str(p.get("category")), vocabs.category_vocab))
    if target_index < 0:
        raise RuntimeError(f"target_id={target_id} not found in corrupted scene")

    corrupted_target = corrupted_scene[target_index]
    target_size = [float(v) for v in corrupted_target["size_m"]]
    return {
        "corrupted_pose": pose_to_model(corrupted_target["position_m"], float(corrupted_target["yaw_deg"]), room_json),
        "context_pos": np.asarray(context_pos, dtype=np.float32),
        "context_size": np.asarray(context_size, dtype=np.float32),
        "context_cat": np.asarray(context_cat, dtype=np.int64),
        "context_mask": np.ones((len(corrupted_scene),), dtype=np.float32),
        "target_index": np.int64(target_index),
        "target_cat": np.int64(category_id(as_str(target_category), vocabs.category_vocab)),
        "target_size": np.asarray(target_size, dtype=np.float32),
        "corruption_type": np.int64(category_id(as_str(corruption_type), vocabs.corruption_vocab)),
        "room_type": np.int64(category_id(as_str(room_type), vocabs.room_type_vocab)),
        "room_scale": room_scale_features(room_json, target_size),
        "corrupted_flags": sample_corrupted_flags({"corrupted_metrics": corrupted_metrics}, room_json, target_size),
    }


def encode_sample(
    sample: Dict[str, Any],
    room_json: Dict[str, Any],
    corrupted_scene: List[Dict[str, Any]],
    clean_target: Dict[str, Any],
    vocabs: ProposalVocabs,
) -> Dict[str, np.ndarray]:
    enc = encode_scene_target(
        target_id=as_str(sample["target_object_id"]),
        target_category=as_str(sample["target_category"]),
        corruption_type=as_str(sample["corruption_type"]),
        room_type=as_str(sample["room_type"]),
        corrupted_metrics=sample["corrupted_metrics"],
        room_json=room_json,
        corrupted_scene=corrupted_scene,
        vocabs=vocabs,
    )
    enc["clean_pose"] = pose_to_model(clean_target["position_m"], float(clean_target["yaw_deg"]), room_json)
    return enc


class RepairProposalDatasetV1(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        split: str,
        vocabs: Optional[ProposalVocabs] = None,
        limit: int = 0,
    ) -> None:
        self.jsonl_path = str(jsonl_path)
        self.split = str(split)
        self.rows = load_sample_rows(self.jsonl_path, split=self.split, limit=limit)
        self.vocabs = vocabs or build_vocabs(self.rows)
        self.room_cache: Dict[str, Dict[str, Any]] = {}
        self.clean_scene_cache: Dict[str, Dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _load_room(self, path: str) -> Dict[str, Any]:
        key = str(Path(path).expanduser().resolve())
        if key not in self.room_cache:
            self.room_cache[key] = load_json(key)
        return self.room_cache[key]

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        row = self.rows[idx]
        room_ref = as_str(row.get("room_ref")).strip()
        if room_ref:
            room_json = self._load_room(room_ref)
        else:
            clean_scene = load_json(row["clean_scene_ref"])
            room_json = room_json_from_scene(clean_scene)
        _, corrupted_scene, clean_target = reconstruct_corrupted_scene(row)
        return encode_sample(row, room_json, corrupted_scene, clean_target, self.vocabs)


def collate_proposal(batch: List[Dict[str, np.ndarray]]) -> ProposalBatch:
    max_n = max(int(b["context_pos"].shape[0]) for b in batch)
    bsz = len(batch)

    context_pos = np.zeros((bsz, max_n, 3), dtype=np.float32)
    context_size = np.zeros((bsz, max_n, 3), dtype=np.float32)
    context_cat = np.zeros((bsz, max_n), dtype=np.int64)
    context_mask = np.zeros((bsz, max_n), dtype=np.float32)
    for i, item in enumerate(batch):
        n = int(item["context_pos"].shape[0])
        context_pos[i, :n] = item["context_pos"]
        context_size[i, :n] = item["context_size"]
        context_cat[i, :n] = item["context_cat"]
        context_mask[i, :n] = item["context_mask"]

    return ProposalBatch(
        clean_pose=torch.from_numpy(np.stack([b["clean_pose"] for b in batch], axis=0)),
        corrupted_pose=torch.from_numpy(np.stack([b["corrupted_pose"] for b in batch], axis=0)),
        context_pos=torch.from_numpy(context_pos),
        context_size=torch.from_numpy(context_size),
        context_cat=torch.from_numpy(context_cat),
        context_mask=torch.from_numpy(context_mask),
        target_index=torch.from_numpy(np.stack([b["target_index"] for b in batch], axis=0)),
        target_cat=torch.from_numpy(np.stack([b["target_cat"] for b in batch], axis=0)),
        target_size=torch.from_numpy(np.stack([b["target_size"] for b in batch], axis=0)),
        corruption_type=torch.from_numpy(np.stack([b["corruption_type"] for b in batch], axis=0)),
        room_type=torch.from_numpy(np.stack([b["room_type"] for b in batch], axis=0)),
        room_scale=torch.from_numpy(np.stack([b["room_scale"] for b in batch], axis=0)),
        corrupted_flags=torch.from_numpy(np.stack([b["corrupted_flags"] for b in batch], axis=0)),
    )
