#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[3]))

try:
    from src.ml.data_py.repair_proposal_dataset_v1 import ProposalVocabs, encode_scene_target, load_json, model_pose_to_world
    from src.ml.models.repair_proposal_v1 import RepairProposalNetV1
    from src.ml.data_py.build_repair_corruptions_v1 import (
        aabb_intersection_metrics,
        collision_metrics_for_target,
        corners_inside_ratio,
        floor_contact_abs_error_m,
        point_in_polygon,
        should_ignore_pair_overlap,
        update_aabb_for_placement,
    )
    from src.ml.data_py.build_corrupted_object_selector_v1 import (
        candidate_record,
        bounds_and_room_area,
        is_important_furniture_candidate,
        select_candidate_records,
        significant_collision_metrics_for_target,
    )
    from src.ml.data_py.corrupted_object_selector_dataset_v1 import build_feature_vector as build_selector_feature_vector
    from src.ml.models.corrupted_object_selector_v1 import CorruptedObjectSelectorV1
except ModuleNotFoundError:
    from repair_proposal_dataset_v1 import ProposalVocabs, encode_scene_target, load_json, model_pose_to_world  # type: ignore
    from repair_proposal_v1 import RepairProposalNetV1  # type: ignore
    from build_repair_corruptions_v1 import aabb_intersection_metrics, collision_metrics_for_target, corners_inside_ratio, floor_contact_abs_error_m, point_in_polygon, should_ignore_pair_overlap, update_aabb_for_placement  # type: ignore
    from build_corrupted_object_selector_v1 import candidate_record, bounds_and_room_area, is_important_furniture_candidate, select_candidate_records, significant_collision_metrics_for_target  # type: ignore
    from corrupted_object_selector_dataset_v1 import build_feature_vector as build_selector_feature_vector  # type: ignore
    from corrupted_object_selector_v1 import CorruptedObjectSelectorV1  # type: ignore


def save_json(path: str | Path, data: Any) -> None:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def as_str(v: Any, default: str = "") -> str:
    if v is None:
        return default
    return str(v)


def round6(v: float) -> float:
    return round(float(v), 6)


def wrap_angle_deg(angle: float) -> float:
    angle = float(angle)
    while angle <= -180.0:
        angle += 360.0
    while angle > 180.0:
        angle -= 360.0
    return angle


def pick_device(requested: str = "auto") -> torch.device:
    requested = str(requested).lower().strip()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA not available")
    if requested == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("MPS not available")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def scene_from_inputs(scene_path: Optional[str], room_path: Optional[str], placement_path: Optional[str]) -> Dict[str, Any]:
    if scene_path:
        scene = load_json(scene_path)
        if scene.get("schema") != "scene.v1":
            raise ValueError("apply_repair_proposal_v1 currently expects scene.v1 when using --scene")
        return scene
    if room_path or placement_path:
        raise ValueError("Bootstrap wrapper supports --scene only. Build scene.v1.json first.")
    raise ValueError("Need --scene")


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
    z_min = float(room.get("z_min", 0.0))
    z_max = float(room.get("z_max", room.get("ceiling_height_m", room.get("ceiling_height", 9999.0))))
    return {
        "bounds_xz": {
            "x_min": min(xs),
            "x_max": max(xs),
            "z_min": min(ys),
            "z_max": max(ys),
        },
        "floor_polygon_xz": polygon_xy,
        "vertical_bounds": {
            "z_min": z_min,
            "z_max": z_max,
        },
    }


def infer_room_type(scene: Dict[str, Any]) -> str:
    room = scene.get("room") or {}
    meta = scene.get("meta") or {}
    return as_str(room.get("room_type")) or as_str(room.get("type")) or as_str(meta.get("room_type")) or "unknown"


def is_floor_object(p: Dict[str, Any]) -> bool:
    mount_type = as_str(p.get("mount_type")).strip().lower()
    if mount_type == "floor":
        return True
    if mount_type == "ceiling":
        return False
    hints = " ".join(
        [
            as_str(p.get("class_name")).strip().lower(),
            as_str(p.get("category")).strip().lower(),
            as_str(p.get("name")).strip().lower(),
        ]
    )
    non_floor_tokens = {
        "ceiling lamp",
        "pendant lamp",
        "ceiling_lamp",
        "ceilinglight",
        "ceilinglightfactory",
        "pendant_lamp",
        "chandelier",
        "wall lamp",
        "wall_lamp",
        "wall light",
        "beziercurve",
        "curve",
        " plane",
    }
    if hints == "plane":
        return False
    return not any(token in hints for token in non_floor_tokens)


def box_intersection_volume(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    ax1 = float(a["aabb"]["x_min"])
    ax2 = float(a["aabb"]["x_max"])
    ay1 = float(a["aabb"]["y_min"])
    ay2 = float(a["aabb"]["y_max"])
    az1 = float(a["aabb"]["z_min"])
    az2 = float(a["aabb"]["z_max"])
    bx1 = float(b["aabb"]["x_min"])
    bx2 = float(b["aabb"]["x_max"])
    by1 = float(b["aabb"]["y_min"])
    by2 = float(b["aabb"]["y_max"])
    bz1 = float(b["aabb"]["z_min"])
    bz2 = float(b["aabb"]["z_max"])
    dx = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    dy = max(0.0, min(ay2, by2) - max(ay1, by1))
    dz = max(0.0, min(az2, bz2) - max(az1, bz1))
    return float(dx * dy * dz)


def box_intersection_area_2d(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    ax1 = float(a["aabb"]["x_min"])
    ax2 = float(a["aabb"]["x_max"])
    ay1 = float(a["aabb"]["y_min"])
    ay2 = float(a["aabb"]["y_max"])
    bx1 = float(b["aabb"]["x_min"])
    bx2 = float(b["aabb"]["x_max"])
    by1 = float(b["aabb"]["y_min"])
    by2 = float(b["aabb"]["y_max"])
    dx = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    dy = max(0.0, min(ay2, by2) - max(ay1, by1))
    return float(dx * dy)


def should_ignore_collision(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return False


def invalidity_metrics(
    target: Dict[str, Any],
    scene_placements: List[Dict[str, Any]],
    room_polygon: List[List[float]],
    important_only: bool = True,
) -> Dict[str, Any]:
    coll_count, coll_area, coll_volume, colliding_ids = (
        significant_collision_metrics_for_target(target, scene_placements, important_only=important_only)
        if important_only
        else collision_metrics_for_target(target, scene_placements)
    )
    inside_count, inside_ratio = corners_inside_ratio(target, room_polygon)
    center_inside = point_in_polygon((float(target["position_m"][0]), float(target["position_m"][1])), room_polygon)
    floor_err = floor_contact_abs_error_m(target)
    outside_room = (not center_inside) or (inside_count == 0)
    target_area = max(float(target["size_m"][0]) * float(target["size_m"][1]), 1e-6)
    proxy_quality = 1.0 - min(coll_area / target_area, 1.0) * 0.6 - (1.0 - inside_ratio) * 0.3 - min(floor_err / 0.15, 1.0) * 0.1
    return {
        "collision_pair_count": int(coll_count),
        "collision_area_sum_2d": round6(coll_area),
        "collision_volume_sum_3d": round6(coll_volume),
        "colliding_with_ids": colliding_ids,
        "corners_inside_count": int(inside_count),
        "corners_inside_ratio": round6(inside_ratio),
        "center_inside_room": bool(center_inside),
        "outside_room": bool(outside_room),
        "floor_contact_abs_error_m": round6(floor_err),
        "valid": bool((coll_count == 0) and (not outside_room) and (floor_err <= 0.08)),
        "proxy_quality": round6(max(0.0, min(proxy_quality, 1.0))),
    }


def is_significant_collision_pair(a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[bool, float, float]:
    inter_area_2d, inter_volume_3d = aabb_intersection_metrics(a, b)
    if should_ignore_pair_overlap(a, b, inter_area_2d, inter_volume_3d):
        return False, inter_area_2d, inter_volume_3d
    own_area = max(placement_footprint_area(a), 1e-6)
    other_area = max(placement_footprint_area(b), 1e-6)
    own_vol = max(float(a["size_m"][0]) * float(a["size_m"][1]) * float(a["size_m"][2]), 1e-6)
    other_vol = max(float(b["size_m"][0]) * float(b["size_m"][1]) * float(b["size_m"][2]), 1e-6)
    rel_area = inter_area_2d / min(own_area, other_area)
    rel_vol = inter_volume_3d / min(own_vol, other_vol)
    if inter_volume_3d <= 1e-3 and rel_area <= 0.03 and rel_vol <= 0.01:
        return False, inter_area_2d, inter_volume_3d
    if inter_volume_3d <= 0.01 and rel_area <= 0.08 and rel_vol <= 0.03:
        return False, inter_area_2d, inter_volume_3d
    return True, inter_area_2d, inter_volume_3d


def infer_corruption_type(metrics: Dict[str, Any]) -> str:
    if metrics["outside_room"] and metrics["collision_pair_count"] > 0:
        return "shift_and_yaw"
    if metrics["outside_room"]:
        return "out_of_room_shift"
    if metrics["collision_pair_count"] > 0:
        return "collision_shift"
    return "displaced_inside_room"


def scene_collision_summary(scene: Dict[str, Any], important_only: bool = True) -> Dict[str, Any]:
    placements = list(scene.get("placements") or [])
    colliding = set()
    pair_count = 0
    area_sum = 0.0
    volume_sum = 0.0
    for i, p in enumerate(placements):
        if not is_floor_object(p):
            continue
        if important_only and not is_important_furniture_candidate(p):
            continue
        for j, other in enumerate(placements):
            if i >= j:
                continue
            if not is_floor_object(other):
                continue
            if important_only and not is_important_furniture_candidate(other):
                continue
            significant, inter_area_2d, inter_volume_3d = is_significant_collision_pair(p, other)
            if not significant:
                continue
            pair_count += 1
            area_sum += float(inter_area_2d)
            volume_sum += float(inter_volume_3d)
            colliding.add(i)
            colliding.add(j)
    return {
        "pair_count": int(pair_count),
        "object_count": len(colliding),
        "indices": sorted(colliding),
        "area_sum_2d": round6(area_sum),
        "volume_sum_3d": round6(volume_sum),
    }


def detect_collision_indices(scene: Dict[str, Any], important_only: bool = True) -> List[int]:
    return list(scene_collision_summary(scene, important_only=important_only)["indices"])


def detect_bad_indices(scene: Dict[str, Any], important_only: bool = True) -> List[int]:
    room_json = room_json_from_scene(scene)
    bounds = room_json["bounds_xz"]
    vertical = room_json["vertical_bounds"]
    placements = list(scene.get("placements") or [])
    bad = set()
    for i, p in enumerate(placements):
        if not is_floor_object(p):
            continue
        if important_only and not is_important_furniture_candidate(p):
            continue
        if (
            float(p["aabb"]["x_min"]) < float(bounds["x_min"]) or
            float(p["aabb"]["x_max"]) > float(bounds["x_max"]) or
            float(p["aabb"]["y_min"]) < float(bounds["z_min"]) or
            float(p["aabb"]["y_max"]) > float(bounds["z_max"]) or
            float(p["aabb"]["z_min"]) < float(vertical["z_min"]) or
            float(p["aabb"]["z_max"]) > float(vertical["z_max"])
        ):
            bad.add(i)
        for j, other in enumerate(placements):
            if i >= j:
                continue
            if not is_floor_object(other):
                continue
            if important_only and not is_important_furniture_candidate(other):
                continue
            if should_ignore_collision(p, other):
                continue
            significant, _, _ = is_significant_collision_pair(p, other)
            if significant:
                bad.add(i)
                bad.add(j)
    return sorted(bad)


def placement_name_blob(p: Dict[str, Any]) -> str:
    return " ".join(
        [
            as_str(p.get("category")).lower(),
            as_str(p.get("name")).lower(),
            as_str(p.get("class_name")).lower(),
            as_str(p.get("mount_type")).lower(),
        ]
    )


def placement_footprint_area(p: Dict[str, Any]) -> float:
    size = [float(v) for v in p.get("size_m", [1.0, 1.0, 1.0])]
    return max(size[0] * size[1], 1e-6)


def placement_move_priority(p: Dict[str, Any]) -> float:
    name_blob = placement_name_blob(p)
    movable_tokens = {
        "lamp",
        "light",
        "floorlamp",
        "tablelamp",
        "chandelier",
        "trinket",
        "book",
        "decor",
        "cube",
        "plant",
        "mirror",
        "wallart",
    }
    heavy_tokens = {
        "bed",
        "wardrobe",
        "cabinet",
        "shelf",
        "bookcase",
        "sofa",
        "desk",
        "tvstand",
        "dresser",
        "nightstand",
        "coffee table",
        "table",
    }
    score = 0.0
    if any(tok in name_blob for tok in movable_tokens):
        score += 1.25
    if any(tok in name_blob for tok in heavy_tokens):
        score -= 1.25
    score -= min(placement_footprint_area(p) / 3.0, 1.5)
    return float(score)


def bbox_target_rank_tuple(scene: Dict[str, Any], idx: int) -> Tuple[float, ...]:
    room_json = room_json_from_scene(scene)
    room_polygon = room_json["floor_polygon_xz"]
    placements = list(scene.get("placements") or [])
    target = placements[idx]
    metrics = invalidity_metrics(target, placements, room_polygon)
    target_area = placement_footprint_area(target)
    coll_area_sum = 0.0
    coll_volume_sum = 0.0
    smaller_than_neighbors = 0.0
    larger_than_neighbors = 0.0
    for j, other in enumerate(placements):
        if j == idx or not is_floor_object(other):
            continue
        if should_ignore_collision(target, other):
            continue
        inter_area = box_intersection_area_2d(target, other)
        inter_vol = box_intersection_volume(target, other)
        if inter_area <= 1e-6 and inter_vol <= 1e-6:
            continue
        coll_area_sum += inter_area
        coll_volume_sum += inter_vol
        other_area = placement_footprint_area(other)
        if target_area < other_area * 0.9:
            smaller_than_neighbors += min(other_area / target_area, 4.0)
        elif target_area > other_area * 1.1:
            larger_than_neighbors += min(target_area / other_area, 4.0)
    move_priority = placement_move_priority(target)
    return (
        float(metrics["collision_pair_count"]),
        float(coll_volume_sum),
        float(coll_area_sum),
        1.0 if bool(metrics["outside_room"]) else 0.0,
        float(smaller_than_neighbors),
        0.2 * move_priority,
        -float(larger_than_neighbors),
        -float(target_area),
    )


def select_candidate_indices(scene: Dict[str, Any], bad_indices: List[int], candidate_limit: int) -> List[int]:
    placements = []
    placement_indices = []
    for i, p in enumerate(scene.get("placements") or []):
        pid = as_str(p.get("id")).strip()
        if not pid:
            continue
        cp = deepcopy(p)
        cp["aabb"] = update_aabb_for_placement(cp)
        placements.append(cp)
        placement_indices.append(i)
    if not placements:
        return []
    room_json = room_json_from_scene(scene)
    room_area = bounds_and_room_area(room_json)[1]
    all_records = [candidate_record(p, placements, room_json, room_area, "__no_target__") for p in placements]
    records, _ = select_candidate_records(
        all_records,
        target_id="",
        mode="furniture_components_v1",
        max_candidates=max(1, int(candidate_limit)),
        component_limit=2,
    )
    id_to_scene_idx = {as_str(scene["placements"][idx].get("id")): idx for idx in placement_indices}
    chosen: List[int] = []
    for rec in records:
        idx = id_to_scene_idx.get(as_str(rec["id"]))
        if idx is None:
            continue
        if idx in bad_indices and idx not in chosen:
            chosen.append(idx)
    if chosen:
        return chosen[:candidate_limit]
    if len(bad_indices) <= candidate_limit:
        return list(bad_indices)
    ranked = sorted(bad_indices, key=lambda idx: bbox_target_rank_tuple(scene, idx), reverse=True)
    return ranked[:candidate_limit]


def load_selector_model(ckpt_path: str, device: torch.device) -> Tuple[CorruptedObjectSelectorV1, Dict[str, Dict[str, int]]]:
    obj = torch.load(ckpt_path, map_location=device)
    cfg = dict(obj["cfg"])
    vocabs = {
        "category_vocab": dict(obj["vocabs"]["category_vocab"]),
        "super_vocab": dict(obj["vocabs"]["super_vocab"]),
        "mount_vocab": dict(obj["vocabs"]["mount_vocab"]),
        "room_type_vocab": dict(obj["vocabs"]["room_type_vocab"]),
    }
    model = CorruptedObjectSelectorV1(
        feature_dim=int(obj.get("feature_dim", cfg.get("feature_dim", 27))),
        num_categories=len(vocabs["category_vocab"]),
        num_supers=len(vocabs["super_vocab"]),
        num_mount_types=len(vocabs["mount_vocab"]),
        num_room_types=len(vocabs["room_type_vocab"]),
        hidden_dim=int(cfg.get("hidden_dim", 192)),
        emb_dim=int(cfg.get("emb_dim", 24)),
        num_layers=int(cfg.get("num_layers", 3)),
        num_heads=int(cfg.get("num_heads", 4)),
        dropout=float(cfg.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(obj["model_state"], strict=True)
    model.eval()
    setattr(model, "expected_feature_dim", int(obj.get("feature_dim", cfg.get("feature_dim", 27))))
    return model, vocabs


def selector_candidate_indices(
    scene: Dict[str, Any],
    selector_model: CorruptedObjectSelectorV1,
    selector_vocabs: Dict[str, Dict[str, int]],
    topk: int,
    candidate_limit: int,
    global_fallback_k: int,
    device: torch.device,
) -> Tuple[List[int], List[Dict[str, Any]]]:
    placements = []
    placement_indices = []
    for i, p in enumerate(scene.get("placements") or []):
        pid = as_str(p.get("id")).strip()
        if not pid:
            continue
        cp = deepcopy(p)
        cp["aabb"] = update_aabb_for_placement(cp)
        placements.append(cp)
        placement_indices.append(i)
    if not placements:
        return [], []
    room_json = room_json_from_scene(scene)
    room_area = bounds_and_room_area(room_json)[1]
    dummy_target = "__no_target__"
    all_records = [candidate_record(p, placements, room_json, room_area, dummy_target) for p in placements]
    records, _ = select_candidate_records(
        all_records,
        target_id="",
        mode="furniture_components_v1",
        max_candidates=max(1, int(candidate_limit)),
        component_limit=2,
        global_fallback_k=max(0, int(global_fallback_k)),
    )
    if not records:
        return [], []
    rec_id_to_slot = {as_str(r["id"]): idx for idx, r in enumerate(all_records)}
    filtered_slots = [rec_id_to_slot[as_str(r["id"])] for r in records]
    placements = [placements[idx] for idx in filtered_slots]
    placement_indices = [placement_indices[idx] for idx in filtered_slots]
    features = np.stack([build_selector_feature_vector(r) for r in records], axis=0)
    expected_feature_dim = int(getattr(selector_model, "expected_feature_dim", features.shape[1]))
    if features.shape[1] > expected_feature_dim:
        features = features[:, :expected_feature_dim]
    elif features.shape[1] < expected_feature_dim:
        pad = np.zeros((features.shape[0], expected_feature_dim - features.shape[1]), dtype=np.float32)
        features = np.concatenate([features, pad], axis=1)
    category = np.asarray([selector_vocabs["category_vocab"].get(as_str(r["category"]), 0) for r in records], dtype=np.int64)
    super_category = np.asarray([selector_vocabs["super_vocab"].get(as_str(r["super_category"]), 0) for r in records], dtype=np.int64)
    mount_type = np.asarray([selector_vocabs["mount_vocab"].get(as_str(r["mount_type"]), 0) for r in records], dtype=np.int64)
    room_type_name = infer_room_type(scene)
    room_type = np.asarray([selector_vocabs["room_type_vocab"].get(room_type_name, 0)] * len(records), dtype=np.int64)
    mask = np.ones((len(records),), dtype=np.float32)
    with torch.no_grad():
        out = selector_model(
            features=torch.from_numpy(features[None, ...]).to(device),
            category=torch.from_numpy(category[None, ...]).to(device),
            super_category=torch.from_numpy(super_category[None, ...]).to(device),
            mount_type=torch.from_numpy(mount_type[None, ...]).to(device),
            room_type=torch.from_numpy(room_type[:1]).to(device),
            mask=torch.from_numpy(mask[None, ...]).to(device),
        )
    logits = out.logits[0].detach().cpu().numpy()
    order = np.argsort(-logits)
    chosen = []
    scored = []
    for idx in order[: max(1, int(topk))]:
        scene_idx = int(placement_indices[int(idx)])
        chosen.append(scene_idx)
        scored.append(
            {
                "index": scene_idx,
                "id": as_str(scene["placements"][scene_idx].get("id")),
                "category": as_str(scene["placements"][scene_idx].get("category")),
                "selector_logit": round6(float(logits[int(idx)])),
            }
        )
    return chosen, scored


def load_model(ckpt_path: str, device: torch.device) -> Tuple[RepairProposalNetV1, ProposalVocabs]:
    obj = torch.load(ckpt_path, map_location=device)
    cfg = dict(obj["cfg"])
    vocabs = ProposalVocabs(
        category_vocab=dict(obj["vocabs"]["category_vocab"]),
        corruption_vocab=dict(obj["vocabs"]["corruption_vocab"]),
        room_type_vocab=dict(obj["vocabs"]["room_type_vocab"]),
    )
    model = RepairProposalNetV1(
        num_categories=len(vocabs.category_vocab),
        num_corruption_types=len(vocabs.corruption_vocab),
        num_room_types=len(vocabs.room_type_vocab),
        dim=int(cfg.get("dim", 192)),
        num_layers=int(cfg.get("num_layers", 4)),
        num_heads=int(cfg.get("num_heads", 8)),
        dropout=float(cfg.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(obj["model_state"], strict=True)
    model.eval()
    return model, vocabs


def encode_current_target(scene: Dict[str, Any], target_index: int, vocabs: ProposalVocabs) -> Tuple[dict, Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    room_json = room_json_from_scene(scene)
    corrupted_scene = deepcopy(scene["placements"])
    target = deepcopy(corrupted_scene[target_index])
    room_polygon = room_json["floor_polygon_xz"]
    current_metrics = invalidity_metrics(target, corrupted_scene, room_polygon)
    encoded = encode_scene_target(
        target_id=as_str(target["id"]),
        target_category=as_str(target.get("category")),
        corruption_type=infer_corruption_type(current_metrics),
        room_type=infer_room_type(scene),
        corrupted_metrics=current_metrics,
        room_json=room_json,
        corrupted_scene=corrupted_scene,
        vocabs=vocabs,
    )
    return encoded, room_json, corrupted_scene, current_metrics


def predict_repair_for_index(
    model: RepairProposalNetV1,
    scene: Dict[str, Any],
    target_index: int,
    vocabs: ProposalVocabs,
    device: torch.device,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    enc, room_json, corrupted_scene, before_metrics = encode_current_target(scene, target_index, vocabs)
    target = deepcopy(corrupted_scene[target_index])
    batch_kwargs = {
        "corrupted_pose": torch.from_numpy(enc["corrupted_pose"][None, ...]).to(device),
        "context_pos": torch.from_numpy(enc["context_pos"][None, ...]).to(device),
        "context_size": torch.from_numpy(enc["context_size"][None, ...]).to(device),
        "context_cat": torch.from_numpy(enc["context_cat"][None, ...]).to(device),
        "context_mask": torch.from_numpy(enc["context_mask"][None, ...]).to(device),
        "target_index": torch.from_numpy(np.asarray([enc["target_index"]], dtype=np.int64)).to(device),
        "target_cat": torch.from_numpy(np.asarray([enc["target_cat"]], dtype=np.int64)).to(device),
        "target_size": torch.from_numpy(enc["target_size"][None, ...]).to(device),
        "corruption_type": torch.from_numpy(np.asarray([enc["corruption_type"]], dtype=np.int64)).to(device),
        "room_type": torch.from_numpy(np.asarray([enc["room_type"]], dtype=np.int64)).to(device),
        "room_scale": torch.from_numpy(enc["room_scale"][None, ...]).to(device),
        "corrupted_flags": torch.from_numpy(enc["corrupted_flags"][None, ...]).to(device),
    }
    with torch.no_grad():
        pred = model(**batch_kwargs).clean_pose[0].detach().cpu().numpy()

    world_pos, yaw_deg, yaw_rad = model_pose_to_world(pred, room_json, fallback_z=float(target["position_m"][2]))
    new_target = deepcopy(target)
    new_target["position_m"] = world_pos
    new_target["yaw_deg"] = float(yaw_deg)
    new_target["yaw_rad"] = float(yaw_rad)
    new_target["rotation_deg"] = int(round(float(yaw_deg))) % 360
    new_target["aabb"] = update_aabb_for_placement(new_target)

    repaired_scene = deepcopy(corrupted_scene)
    repaired_scene[target_index] = deepcopy(new_target)
    after_metrics = invalidity_metrics(new_target, repaired_scene, room_json["floor_polygon_xz"])

    return new_target, before_metrics, after_metrics


def target_move_cost(before_target: Dict[str, Any], after_target: Dict[str, Any]) -> float:
    before_pos = np.asarray(before_target["position_m"], dtype=np.float32)
    after_pos = np.asarray(after_target["position_m"], dtype=np.float32)
    translation = float(np.linalg.norm(after_pos - before_pos))
    yaw_delta = abs(wrap_angle_deg(float(after_target["yaw_deg"]) - float(before_target["yaw_deg"])))
    size = [float(v) for v in before_target.get("size_m", [1.0, 1.0, 1.0])]
    footprint = max(size[0] * size[1], 1e-6)
    mobility_bias = -0.25 * placement_move_priority(before_target)
    size_bias = min(footprint / 2.5, 1.5)
    translation_cost = min(translation / 2.5, 1.5)
    yaw_cost = min(yaw_delta / 180.0, 1.0) * 0.15
    return float(translation_cost + yaw_cost + size_bias + mobility_bias)


def candidate_rank_tuple(
    before_bad_count: int,
    after_bad_count: int,
    before_collision_summary: Dict[str, Any],
    after_collision_summary: Dict[str, Any],
    before_metrics: Dict[str, Any],
    after_metrics: Dict[str, Any],
    before_target: Dict[str, Any],
    after_target: Dict[str, Any],
) -> Tuple[float, ...]:
    before_outside = 1.0 if bool(before_metrics.get("outside_room")) else 0.0
    after_outside = 1.0 if bool(after_metrics.get("outside_room")) else 0.0
    move_cost = target_move_cost(before_target, after_target)
    return (
        float(before_collision_summary["pair_count"] - after_collision_summary["pair_count"]),
        float(before_collision_summary["object_count"] - after_collision_summary["object_count"]),
        float(before_bad_count - after_bad_count),
        float(before_collision_summary["volume_sum_3d"] - after_collision_summary["volume_sum_3d"]),
        float(before_collision_summary["area_sum_2d"] - after_collision_summary["area_sum_2d"]),
        float(before_metrics["collision_pair_count"] - after_metrics["collision_pair_count"]),
        float(before_outside - after_outside),
        float(after_metrics["proxy_quality"] - before_metrics["proxy_quality"]),
        -move_cost,
        -float(after_metrics["floor_contact_abs_error_m"]),
        -float(after_metrics["collision_area_sum_2d"]),
    )


def accept_candidate_rank(rank: Tuple[float, ...]) -> bool:
    return tuple(float(v) for v in rank) > ((0.0,) * len(rank))


def normalize_scene_aabbs(scene: Dict[str, Any]) -> Dict[str, Any]:
    fixed = deepcopy(scene)
    placements = []
    for p in list(fixed.get("placements") or []):
        cp = deepcopy(p)
        cp["aabb"] = update_aabb_for_placement(cp)
        placements.append(cp)
    fixed["placements"] = placements
    return fixed


def repair_scene_with_models(
    scene: Dict[str, Any],
    model: RepairProposalNetV1,
    vocabs: ProposalVocabs,
    device: torch.device,
    *,
    max_passes: int = 3,
    target_id: str = "",
    candidate_limit: int = 4,
    selector_model: Optional[CorruptedObjectSelectorV1] = None,
    selector_vocabs: Optional[Dict[str, Dict[str, int]]] = None,
    selector_topk: int = 3,
    selector_candidate_limit: int = 6,
    selector_global_fallback_k: int = 3,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    scene = normalize_scene_aabbs(scene)
    target_id = as_str(target_id).strip()
    target_index = None
    if target_id:
        for i, p in enumerate(scene.get("placements") or []):
            if as_str(p.get("id")) == target_id:
                target_index = i
                break
        if target_index is None:
            raise ValueError(f"target id not found: {target_id}")

    report = {
        "passes": [],
        "initial_bad_indices": detect_bad_indices(scene),
        "initial_collision_indices": detect_collision_indices(scene),
    }
    for pass_idx in range(1, int(max_passes) + 1):
        if target_index is not None:
            bad_indices = [int(target_index)]
            candidate_indices = list(bad_indices)
            selector_candidates: List[Dict[str, Any]] = []
            collision_summary_before = scene_collision_summary(scene)
            collision_indices = list(collision_summary_before["indices"])
        else:
            collision_summary_before = scene_collision_summary(scene)
            collision_indices = list(collision_summary_before["indices"])
            bad_indices = detect_bad_indices(scene)
            problem_indices = sorted(set(bad_indices) | set(collision_indices))
            if selector_model is not None and selector_vocabs is not None:
                selector_raw_indices, selector_candidates = selector_candidate_indices(
                    scene=scene,
                    selector_model=selector_model,
                    selector_vocabs=selector_vocabs,
                    topk=max(1, int(selector_topk)),
                    candidate_limit=max(1, int(selector_candidate_limit)),
                    global_fallback_k=max(0, int(selector_global_fallback_k)),
                    device=device,
                )
                candidate_indices = []
                merged_groups = [
                    [idx for idx in selector_raw_indices if idx in collision_indices],
                    collision_indices,
                    [idx for idx in selector_raw_indices if idx in problem_indices],
                    problem_indices,
                    list(selector_raw_indices),
                ]
                seen = set()
                merged_limit = max(1, int(candidate_limit), int(selector_topk))
                for group in merged_groups:
                    for idx in group:
                        idx = int(idx)
                        if idx in seen:
                            continue
                        seen.add(idx)
                        candidate_indices.append(idx)
                        if len(candidate_indices) >= merged_limit:
                            break
                    if len(candidate_indices) >= merged_limit:
                        break
                if not candidate_indices and problem_indices:
                    candidate_indices = select_candidate_indices(scene, problem_indices, max(1, int(candidate_limit)))
            else:
                candidate_indices = select_candidate_indices(scene, problem_indices, max(1, int(candidate_limit)))
                selector_candidates = []
        if not bad_indices and not collision_indices and not candidate_indices:
            break
        pass_report = {
            "pass": pass_idx,
            "bad_indices_before": bad_indices,
            "collision_indices_before": collision_indices,
            "collision_summary_before": collision_summary_before,
            "candidate_indices": candidate_indices,
            "selector_candidates": selector_candidates,
            "accepted": [],
        }
        changed = False
        if target_index is not None:
            before_bad_count = len(detect_bad_indices(scene))
            before_collision_summary = collision_summary_before
            for idx in bad_indices:
                pred_target, before_metrics, after_metrics = predict_repair_for_index(model, scene, idx, vocabs, device)
                scene_candidate = deepcopy(scene)
                scene_candidate["placements"][idx] = deepcopy(pred_target)
                after_collision_summary = scene_collision_summary(scene_candidate)
                after_bad_indices = detect_bad_indices(scene_candidate)
                rank = candidate_rank_tuple(
                    before_bad_count=before_bad_count,
                    after_bad_count=len(after_bad_indices),
                    before_collision_summary=before_collision_summary,
                    after_collision_summary=after_collision_summary,
                    before_metrics=before_metrics,
                    after_metrics=after_metrics,
                    before_target=scene["placements"][idx],
                    after_target=pred_target,
                )
                if not accept_candidate_rank(rank):
                    continue
                scene["placements"][idx] = pred_target
                changed = True
                pass_report["accepted"].append(
                    {
                        "index": int(idx),
                        "id": as_str(pred_target["id"]),
                        "category": as_str(pred_target.get("category")),
                        "before": before_metrics,
                        "after": after_metrics,
                        "rank": [round6(v) for v in rank],
                        "after_bad_indices_if_applied": after_bad_indices,
                        "after_collision_summary_if_applied": after_collision_summary,
                    }
                )
        else:
            before_bad_count = len(bad_indices)
            before_collision_summary = collision_summary_before
            best_candidate: Optional[Dict[str, Any]] = None
            best_rank: Optional[Tuple[float, ...]] = None
            for idx in candidate_indices:
                bbox_rank = [round6(v) for v in bbox_target_rank_tuple(scene, idx)]
                pred_target, before_metrics, after_metrics = predict_repair_for_index(model, scene, idx, vocabs, device)
                scene_candidate = deepcopy(scene)
                scene_candidate["placements"][idx] = pred_target
                after_collision_summary = scene_collision_summary(scene_candidate)
                after_bad_indices = detect_bad_indices(scene_candidate)
                rank = candidate_rank_tuple(
                    before_bad_count=before_bad_count,
                    after_bad_count=len(after_bad_indices),
                    before_collision_summary=before_collision_summary,
                    after_collision_summary=after_collision_summary,
                    before_metrics=before_metrics,
                    after_metrics=after_metrics,
                    before_target=scene["placements"][idx],
                    after_target=pred_target,
                )
                if best_rank is None or rank > best_rank:
                    best_rank = rank
                    best_candidate = {
                        "index": int(idx),
                        "target": pred_target,
                        "before": before_metrics,
                        "after": after_metrics,
                        "after_collision_summary": after_collision_summary,
                        "after_bad_indices": after_bad_indices,
                        "bbox_selector_rank": bbox_rank,
                        "rank_raw": rank,
                        "rank": [round6(v) for v in rank],
                    }
            if best_candidate is not None and accept_candidate_rank(tuple(best_candidate["rank_raw"])):
                idx = int(best_candidate["index"])
                pred_target = deepcopy(best_candidate["target"])
                scene["placements"][idx] = pred_target
                changed = True
                pass_report["accepted"].append(
                    {
                        "index": idx,
                        "id": as_str(pred_target["id"]),
                        "category": as_str(pred_target.get("category")),
                        "before": best_candidate["before"],
                        "after": best_candidate["after"],
                        "rank": best_candidate["rank"],
                        "bbox_selector_rank": best_candidate["bbox_selector_rank"],
                        "after_bad_indices_if_applied": best_candidate["after_bad_indices"],
                        "after_collision_summary_if_applied": best_candidate["after_collision_summary"],
                    }
                )
        collision_summary_after = scene_collision_summary(scene)
        pass_report["bad_indices_after"] = detect_bad_indices(scene)
        pass_report["collision_indices_after"] = list(collision_summary_after["indices"])
        pass_report["collision_summary_after"] = collision_summary_after
        report["passes"].append(pass_report)
        if not changed:
            break

    report["final_bad_indices"] = detect_bad_indices(scene)
    report["final_collision_indices"] = detect_collision_indices(scene)
    return scene, report


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply repair proposal model to scene.v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report-json", default="")
    ap.add_argument("--max-passes", type=int, default=3)
    ap.add_argument("--target-id", default="", help="If set, repair only this placement id instead of scanning all bad indices.")
    ap.add_argument("--candidate-limit", type=int, default=4, help="If target-id is not set, try only top-K bbox-suspect objects per pass.")
    ap.add_argument("--selector-model", default="", help="Optional corrupted-object selector checkpoint; if set, use its top-k candidates instead of bbox selector.")
    ap.add_argument("--selector-topk", type=int, default=3, help="How many selector candidates to try per pass.")
    ap.add_argument("--selector-candidate-limit", type=int, default=6, help="How many hard-set candidates the selector sees before choosing top-k.")
    ap.add_argument("--selector-global-fallback-k", type=int, default=3, help="How many room-wide anomaly furniture candidates to merge into selector pool.")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    model, vocabs = load_model(args.model, device)
    selector_model = None
    selector_vocabs = None
    if as_str(args.selector_model).strip():
        selector_model, selector_vocabs = load_selector_model(args.selector_model, device)
    scene = scene_from_inputs(args.scene, None, None)
    scene, report = repair_scene_with_models(
        scene=scene,
        model=model,
        vocabs=vocabs,
        device=device,
        max_passes=int(args.max_passes),
        target_id=as_str(args.target_id),
        candidate_limit=int(args.candidate_limit),
        selector_model=selector_model,
        selector_vocabs=selector_vocabs,
        selector_topk=int(args.selector_topk),
        selector_candidate_limit=int(args.selector_candidate_limit),
        selector_global_fallback_k=int(args.selector_global_fallback_k),
    )
    save_json(args.out, scene)
    if args.report_json:
        save_json(args.report_json, report)
    print(f"[apply_repair_proposal_v1] initial_bad={len(report['initial_bad_indices'])}")
    print(f"[apply_repair_proposal_v1] final_bad={len(report['final_bad_indices'])}")
    print(f"[apply_repair_proposal_v1] wrote scene={Path(args.out).expanduser().resolve()}")
    if args.report_json:
        print(f"[apply_repair_proposal_v1] wrote report={Path(args.report_json).expanduser().resolve()}")


if __name__ == "__main__":
    main()
