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

from src.ml.data_py.repair_proposal_dataset_v1 import (
    ProposalVocabs,
    encode_scene_target,
    load_json,
    model_pose_to_world,
)
from src.ml.models.repair_proposal_v1 import RepairProposalNetV1
from src.ml.data_py.build_repair_corruptions_v1 import (
    collision_metrics_for_target,
    corners_inside_ratio,
    floor_contact_abs_error_m,
    point_in_polygon,
    update_aabb_for_placement,
)
from src.tools.evaluate_unified_scene import box_intersection_volume, parse_placements, parse_room, should_ignore_collision
from src.tools.normalize_scene_format import build_scene_from_room_and_placement, convert_to_scene_v1


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


def pick_device(requested: str = "auto") -> torch.device:
    requested = str(requested).lower().strip()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA недоступна")
    if requested == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("MPS недоступна")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def scene_from_inputs(scene_path: Optional[str], room_path: Optional[str], placement_path: Optional[str]) -> Dict[str, Any]:
    if scene_path:
        return convert_to_scene_v1(load_json(scene_path))
    if room_path and placement_path:
        return build_scene_from_room_and_placement(load_json(room_path), load_json(placement_path))
    raise ValueError("Нужно передать либо --scene, либо пару --room + --placement")


def room_json_from_scene(scene: Dict[str, Any]) -> Dict[str, Any]:
    room = parse_room(scene)
    polygon = room.floor_polygon or [
        (room.x_min, room.y_min),
        (room.x_max, room.y_min),
        (room.x_max, room.y_max),
        (room.x_min, room.y_max),
    ]
    return {
        "bounds_xz": {
            "x_min": float(room.x_min),
            "x_max": float(room.x_max),
            "z_min": float(room.y_min),
            "z_max": float(room.y_max),
        },
        "floor_polygon_xz": [[float(x), float(y)] for x, y in polygon],
    }


def infer_room_type(scene: Dict[str, Any]) -> str:
    room_raw = scene.get("room") if isinstance(scene.get("room"), dict) else {}
    return (
        as_str(room_raw.get("room_type"))
        or as_str(room_raw.get("type"))
        or as_str((scene.get("meta") or {}).get("room_type"))
        or "unknown"
    )


def invalidity_metrics(target: Dict[str, Any], scene_placements: List[Dict[str, Any]], room_polygon: List[List[float]]) -> Dict[str, Any]:
    coll_count, coll_area, coll_volume, colliding_ids = collision_metrics_for_target(target, scene_placements)
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


def infer_corruption_type(metrics: Dict[str, Any]) -> str:
    if metrics["outside_room"] and metrics["collision_pair_count"] > 0:
        return "shift_and_yaw"
    if metrics["outside_room"]:
        return "out_of_room_shift"
    if metrics["collision_pair_count"] > 0:
        return "collision_shift"
    return "displaced_inside_room"


def detect_bad_indices(scene: Dict[str, Any]) -> List[int]:
    room = parse_room(scene)
    placements = parse_placements(scene)
    bad = set()
    for i, p in enumerate(placements):
        if not p.is_floor_object():
            continue
        if (
            p.x_min < room.x_min or p.x_max > room.x_max or
            p.y_min < room.y_min or p.y_max > room.y_max or
            p.z_min < room.z_min or p.z_max > room.z_max
        ):
            bad.add(i)
        for j, other in enumerate(placements):
            if i >= j:
                continue
            if should_ignore_collision(p, other):
                continue
            if box_intersection_volume(p, other) > 1e-6:
                bad.add(i)
                bad.add(j)
    return sorted(bad)


def load_model(ckpt_path: str, device: torch.device) -> Tuple[RepairProposalNetV1, dict, ProposalVocabs]:
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
    return model, cfg, vocabs


def encode_current_target(scene: Dict[str, Any], target_index: int, vocabs: ProposalVocabs) -> Tuple[dict, Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    room_json = room_json_from_scene(scene)
    corrupted_scene = deepcopy(scene["placements"])
    target = deepcopy(corrupted_scene[target_index])
    room_polygon = room_json["floor_polygon_xz"]
    current_metrics = invalidity_metrics(target, corrupted_scene, room_polygon)
    sample_like = encode_scene_target(
        target_id=as_str(target["id"]),
        target_category=as_str(target.get("category")),
        corruption_type=infer_corruption_type(current_metrics),
        room_type=infer_room_type(scene),
        corrupted_metrics=current_metrics,
        room_json=room_json,
        corrupted_scene=corrupted_scene,
        vocabs=vocabs,
    )
    return sample_like, room_json, corrupted_scene, current_metrics


def predict_repair_for_index(
    model: RepairProposalNetV1,
    scene: Dict[str, Any],
    target_index: int,
    vocabs: ProposalVocabs,
    device: torch.device,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
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

    before_score = float(before_metrics["proxy_quality"])
    after_score = float(after_metrics["proxy_quality"])
    accept = (
        (after_metrics["valid"] and not before_metrics["valid"])
        or (after_score > before_score + 1e-4)
        or (
            after_metrics["collision_pair_count"] < before_metrics["collision_pair_count"]
            and not after_metrics["outside_room"]
        )
    )
    return (new_target if accept else None), before_metrics, after_metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Repair invalid scene placements with direct ML proposal model")
    ap.add_argument("--model", required=True)
    ap.add_argument("--scene", default=None)
    ap.add_argument("--room", default=None)
    ap.add_argument("--placement", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report-json", default="")
    ap.add_argument("--max-passes", type=int, default=3)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    model, _, vocabs = load_model(args.model, device)
    scene = scene_from_inputs(args.scene, args.room, args.placement)

    report = {"passes": [], "initial_bad_indices": detect_bad_indices(scene)}
    for pass_idx in range(1, int(args.max_passes) + 1):
        bad_indices = detect_bad_indices(scene)
        if not bad_indices:
            break
        pass_report = {"pass": pass_idx, "bad_indices_before": bad_indices, "accepted": []}
        changed = False
        for idx in bad_indices:
            pred_target, before_metrics, after_metrics = predict_repair_for_index(model, scene, idx, vocabs, device)
            if pred_target is None:
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
                }
            )
        pass_report["bad_indices_after"] = detect_bad_indices(scene)
        report["passes"].append(pass_report)
        if not changed:
            break

    report["final_bad_indices"] = detect_bad_indices(scene)
    save_json(args.out, scene)
    if args.report_json:
        save_json(args.report_json, report)
    print(f"[repair_proposal_v1] initial_bad={len(report['initial_bad_indices'])}")
    print(f"[repair_proposal_v1] final_bad={len(report['final_bad_indices'])}")
    print(f"[repair_proposal_v1] wrote scene={Path(args.out).expanduser().resolve()}")
    if args.report_json:
        print(f"[repair_proposal_v1] wrote report={Path(args.report_json).expanduser().resolve()}")


if __name__ == "__main__":
    main()
