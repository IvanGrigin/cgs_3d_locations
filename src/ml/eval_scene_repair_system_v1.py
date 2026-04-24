#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

try:
    from src.ml.data_py.repair_proposal_dataset_v1 import load_sample_rows, reconstruct_corrupted_scene
    from src.ml.data_py.build_repair_corruptions_v1 import aabb_intersection_metrics, compute_metrics
    from src.ml.infer.apply_repair_proposal_v1 import (
        as_str,
        detect_bad_indices,
        load_model,
        load_selector_model,
        normalize_scene_aabbs,
        pick_device,
        repair_scene_with_models,
        room_json_from_scene,
        scene_collision_summary,
        selector_candidate_indices,
    )
except ModuleNotFoundError:
    from repair_proposal_dataset_v1 import load_sample_rows, reconstruct_corrupted_scene  # type: ignore
    from build_repair_corruptions_v1 import aabb_intersection_metrics, compute_metrics  # type: ignore
    from apply_repair_proposal_v1 import (  # type: ignore
        as_str,
        detect_bad_indices,
        load_model,
        load_selector_model,
        normalize_scene_aabbs,
        pick_device,
        repair_scene_with_models,
        room_json_from_scene,
        scene_collision_summary,
        selector_candidate_indices,
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Batch evaluation for predictor -> replacer -> wrapper scene repair system.")
    ap.add_argument("--samples-jsonl", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--selector-model", default="")
    ap.add_argument("--split", default="test", help="train|val|test or empty for all")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-passes", type=int, default=1)
    ap.add_argument("--candidate-limit", type=int, default=4)
    ap.add_argument("--selector-topk", type=int, default=3)
    ap.add_argument("--selector-candidate-limit", type=int, default=6)
    ap.add_argument("--selector-global-fallback-k", type=int, default=3)
    ap.add_argument("--oracle-target", action="store_true", help="Pass true target_id into wrapper to measure proposal quality without selector errors.")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-jsonl", default="")
    return ap.parse_args()


def save_json(path: str | Path, payload: Any) -> None:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def find_target_index(scene: Dict[str, Any], target_id: str) -> int:
    for i, p in enumerate(scene.get("placements") or []):
        if as_str(p.get("id")) == target_id:
            return int(i)
    raise RuntimeError(f"target_id={target_id} not found in scene")


def find_target(scene: Dict[str, Any], target_id: str) -> Dict[str, Any]:
    return deepcopy(scene["placements"][find_target_index(scene, target_id)])


def pose_delta(before_target: Dict[str, Any], after_target: Dict[str, Any]) -> Tuple[float, float]:
    before_pos = [float(v) for v in before_target["position_m"]]
    after_pos = [float(v) for v in after_target["position_m"]]
    pos_delta = sum((a - b) ** 2 for a, b in zip(before_pos, after_pos)) ** 0.5
    yaw_delta = abs(float(after_target["yaw_deg"]) - float(before_target["yaw_deg"]))
    return float(pos_delta), float(yaw_delta)


def footprint_area(target: Dict[str, Any]) -> float:
    size = [float(v) for v in target.get("size_m", [1.0, 1.0, 1.0])]
    return max(size[0] * size[1], 1e-6)


def clean_overlap_ratio(clean_target: Dict[str, Any], eval_target: Dict[str, Any]) -> float:
    inter_area_2d, _ = aabb_intersection_metrics(clean_target, eval_target)
    return float(inter_area_2d / max(footprint_area(clean_target), 1e-6))


def init_stats() -> Dict[str, Any]:
    return {
        "samples_total": 0,
        "samples_with_initial_collisions": 0,
        "samples_with_initial_bad": 0,
        "samples_with_accepted_move": 0,
        "samples_with_accepted_target": 0,
        "accepted_move_count_total": 0,
        "samples_with_collision_pair_reduction": 0,
        "samples_with_collision_pair_solved": 0,
        "samples_with_collision_object_reduction": 0,
        "samples_with_bad_count_reduction": 0,
        "samples_with_quality_improvement": 0,
        "samples_valid_after_repair": 0,
        "samples_success_after_repair": 0,
        "samples_position_error_improved": 0,
        "samples_yaw_error_improved": 0,
        "samples_target_overlap50": 0,
        "samples_target_overlap80": 0,
        "samples_target_overlap95": 0,
        "samples_selector_top1_hit": 0,
        "samples_selector_topk_hit": 0,
        "samples_wrapper_candidate_hit": 0,
        "sum_initial_bad_count": 0.0,
        "sum_final_bad_count": 0.0,
        "sum_initial_collision_pairs": 0.0,
        "sum_final_collision_pairs": 0.0,
        "sum_initial_collision_objects": 0.0,
        "sum_final_collision_objects": 0.0,
        "sum_initial_collision_area_2d": 0.0,
        "sum_final_collision_area_2d": 0.0,
        "sum_initial_collision_volume_3d": 0.0,
        "sum_final_collision_volume_3d": 0.0,
        "sum_position_l2_before": 0.0,
        "sum_position_l2_after": 0.0,
        "sum_yaw_abs_error_before": 0.0,
        "sum_yaw_abs_error_after": 0.0,
        "sum_quality_before": 0.0,
        "sum_quality_after": 0.0,
        "sum_target_overlap_ratio": 0.0,
    }


def update_stats(stats: Dict[str, Any], sample: Dict[str, Any]) -> None:
    for k, v in sample.items():
        if k in stats and isinstance(stats[k], int) and isinstance(v, bool):
            stats[k] += int(v)
        elif k in stats and isinstance(stats[k], (int, float)) and isinstance(v, (int, float)):
            stats[k] += v


def finalize_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    total = max(int(stats["samples_total"]), 1)
    with_initial_collisions = max(int(stats["samples_with_initial_collisions"]), 1)
    with_selector = total
    with_accepted_move = max(int(stats["samples_with_accepted_move"]), 1)
    return {
        **stats,
        "accepted_move_rate": round(float(stats["samples_with_accepted_move"]) / total, 6),
        "accepted_target_rate": round(float(stats["samples_with_accepted_target"]) / total, 6),
        "accepted_target_given_move_rate": round(float(stats["samples_with_accepted_target"]) / with_accepted_move, 6),
        "selector_top1_hit_rate": round(float(stats["samples_selector_top1_hit"]) / with_selector, 6),
        "selector_topk_hit_rate": round(float(stats["samples_selector_topk_hit"]) / with_selector, 6),
        "wrapper_candidate_hit_rate": round(float(stats["samples_wrapper_candidate_hit"]) / total, 6),
        "collision_pair_reduction_rate": round(float(stats["samples_with_collision_pair_reduction"]) / total, 6),
        "collision_pair_solved_rate_given_initial_collisions": round(float(stats["samples_with_collision_pair_solved"]) / with_initial_collisions, 6),
        "collision_object_reduction_rate": round(float(stats["samples_with_collision_object_reduction"]) / total, 6),
        "bad_count_reduction_rate": round(float(stats["samples_with_bad_count_reduction"]) / total, 6),
        "quality_improved_rate": round(float(stats["samples_with_quality_improvement"]) / total, 6),
        "valid_rate_after_repair": round(float(stats["samples_valid_after_repair"]) / total, 6),
        "success_rate_after_repair": round(float(stats["samples_success_after_repair"]) / total, 6),
        "position_error_improved_rate": round(float(stats["samples_position_error_improved"]) / total, 6),
        "yaw_error_improved_rate": round(float(stats["samples_yaw_error_improved"]) / total, 6),
        "target_overlap50_rate": round(float(stats["samples_target_overlap50"]) / total, 6),
        "target_overlap80_rate": round(float(stats["samples_target_overlap80"]) / total, 6),
        "target_overlap95_rate": round(float(stats["samples_target_overlap95"]) / total, 6),
        "mean_initial_bad_count": round(float(stats["sum_initial_bad_count"]) / total, 6),
        "mean_final_bad_count": round(float(stats["sum_final_bad_count"]) / total, 6),
        "mean_initial_collision_pairs": round(float(stats["sum_initial_collision_pairs"]) / total, 6),
        "mean_final_collision_pairs": round(float(stats["sum_final_collision_pairs"]) / total, 6),
        "mean_initial_collision_objects": round(float(stats["sum_initial_collision_objects"]) / total, 6),
        "mean_final_collision_objects": round(float(stats["sum_final_collision_objects"]) / total, 6),
        "mean_initial_collision_area_2d": round(float(stats["sum_initial_collision_area_2d"]) / total, 6),
        "mean_final_collision_area_2d": round(float(stats["sum_final_collision_area_2d"]) / total, 6),
        "mean_initial_collision_volume_3d": round(float(stats["sum_initial_collision_volume_3d"]) / total, 6),
        "mean_final_collision_volume_3d": round(float(stats["sum_final_collision_volume_3d"]) / total, 6),
        "mean_position_l2_before": round(float(stats["sum_position_l2_before"]) / total, 6),
        "mean_position_l2_after": round(float(stats["sum_position_l2_after"]) / total, 6),
        "mean_yaw_abs_error_before": round(float(stats["sum_yaw_abs_error_before"]) / total, 6),
        "mean_yaw_abs_error_after": round(float(stats["sum_yaw_abs_error_after"]) / total, 6),
        "mean_quality_before": round(float(stats["sum_quality_before"]) / total, 6),
        "mean_quality_after": round(float(stats["sum_quality_after"]) / total, 6),
        "mean_target_overlap_ratio": round(float(stats["sum_target_overlap_ratio"]) / total, 6),
    }


def main() -> None:
    args = parse_args()
    split = as_str(args.split).strip().lower() or None
    rows = load_sample_rows(args.samples_jsonl, split=split, limit=int(args.limit))
    device = pick_device(args.device)
    model, vocabs = load_model(args.model, device)
    selector_model = None
    selector_vocabs = None
    if as_str(args.selector_model).strip():
        selector_model, selector_vocabs = load_selector_model(args.selector_model, device)

    overall = init_stats()
    by_corruption_type: Dict[str, Dict[str, Any]] = defaultdict(init_stats)
    per_sample_rows: List[Dict[str, Any]] = []

    for row in rows:
        sample_id = as_str(row.get("sample_id"))
        corruption_type = as_str(row.get("corruption_type"))
        target_id = as_str(row.get("target_object_id"))
        scene_gt, corrupted_placements, clean_target = reconstruct_corrupted_scene(row)
        corrupted_scene = deepcopy(scene_gt)
        corrupted_scene["placements"] = corrupted_placements
        corrupted_scene = normalize_scene_aabbs(corrupted_scene)
        room_json = room_json_from_scene(scene_gt)

        initial_bad_indices = detect_bad_indices(corrupted_scene)
        initial_collision_summary = scene_collision_summary(corrupted_scene)
        selector_top1_hit = False
        selector_topk_hit = False
        if selector_model is not None and selector_vocabs is not None:
            selector_indices, selector_candidates = selector_candidate_indices(
                scene=corrupted_scene,
                selector_model=selector_model,
                selector_vocabs=selector_vocabs,
                topk=max(1, int(args.selector_topk)),
                candidate_limit=max(1, int(args.selector_candidate_limit)),
                global_fallback_k=max(0, int(args.selector_global_fallback_k)),
                device=device,
            )
            selector_candidate_ids = [as_str(c["id"]) for c in selector_candidates]
            selector_top1_hit = bool(selector_candidate_ids and selector_candidate_ids[0] == target_id)
            selector_topk_hit = bool(target_id in selector_candidate_ids)
        else:
            selector_candidates = []

        repaired_scene, report = repair_scene_with_models(
            scene=corrupted_scene,
            model=model,
            vocabs=vocabs,
            device=device,
            max_passes=int(args.max_passes),
            target_id=target_id if bool(args.oracle_target) else "",
            candidate_limit=int(args.candidate_limit),
            selector_model=selector_model,
            selector_vocabs=selector_vocabs,
            selector_topk=int(args.selector_topk),
            selector_candidate_limit=int(args.selector_candidate_limit),
            selector_global_fallback_k=int(args.selector_global_fallback_k),
        )

        final_collision_summary = scene_collision_summary(repaired_scene)
        final_bad_indices = detect_bad_indices(repaired_scene)
        final_target = find_target(repaired_scene, target_id)
        corrupted_target = find_target(corrupted_scene, target_id)
        repaired_metrics = compute_metrics(
            clean_target=clean_target,
            eval_target=final_target,
            eval_scene_placements=list(repaired_scene.get("placements") or []),
            room_polygon=room_json["floor_polygon_xz"],
        )
        accepted_moves = sum(len((p.get("accepted") or [])) for p in report.get("passes") or [])
        accepted_ids = [as_str(a.get("id")) for p in report.get("passes") or [] for a in (p.get("accepted") or [])]
        wrapper_candidate_indices = list(((report.get("passes") or [{}])[0]).get("candidate_indices") or [])
        wrapper_candidate_hit = bool(find_target_index(corrupted_scene, target_id) in wrapper_candidate_indices)
        target_pos_delta, target_yaw_delta = pose_delta(corrupted_target, final_target)
        target_overlap_ratio = clean_overlap_ratio(clean_target, final_target)

        sample_stats = {
            "samples_total": 1,
            "samples_with_initial_collisions": int(initial_collision_summary["pair_count"] > 0),
            "samples_with_initial_bad": int(len(initial_bad_indices) > 0),
            "samples_with_accepted_move": int(accepted_moves > 0),
            "samples_with_accepted_target": int(target_id in accepted_ids),
            "accepted_move_count_total": int(accepted_moves),
            "samples_with_collision_pair_reduction": int(final_collision_summary["pair_count"] < initial_collision_summary["pair_count"]),
            "samples_with_collision_pair_solved": int(initial_collision_summary["pair_count"] > 0 and final_collision_summary["pair_count"] == 0),
            "samples_with_collision_object_reduction": int(final_collision_summary["object_count"] < initial_collision_summary["object_count"]),
            "samples_with_bad_count_reduction": int(len(final_bad_indices) < len(initial_bad_indices)),
            "samples_with_quality_improvement": int(float(repaired_metrics["quality_score"]) > float(row["corrupted_metrics"]["quality_score"])),
            "samples_valid_after_repair": int(bool(repaired_metrics["valid"])),
            "samples_success_after_repair": int(bool(repaired_metrics["valid"]) and float(repaired_metrics["position_l2_m"]) <= 0.20 and float(repaired_metrics["yaw_abs_error_deg"]) <= 15.0),
            "samples_position_error_improved": int(float(repaired_metrics["position_l2_m"]) < float(row["corrupted_metrics"]["position_l2_m"])),
            "samples_yaw_error_improved": int(float(repaired_metrics["yaw_abs_error_deg"]) < float(row["corrupted_metrics"]["yaw_abs_error_deg"])),
            "samples_target_overlap50": int(target_overlap_ratio >= 0.50),
            "samples_target_overlap80": int(target_overlap_ratio >= 0.80),
            "samples_target_overlap95": int(target_overlap_ratio >= 0.95),
            "samples_selector_top1_hit": int(selector_top1_hit),
            "samples_selector_topk_hit": int(selector_topk_hit),
            "samples_wrapper_candidate_hit": int(wrapper_candidate_hit),
            "sum_initial_bad_count": float(len(initial_bad_indices)),
            "sum_final_bad_count": float(len(final_bad_indices)),
            "sum_initial_collision_pairs": float(initial_collision_summary["pair_count"]),
            "sum_final_collision_pairs": float(final_collision_summary["pair_count"]),
            "sum_initial_collision_objects": float(initial_collision_summary["object_count"]),
            "sum_final_collision_objects": float(final_collision_summary["object_count"]),
            "sum_initial_collision_area_2d": float(initial_collision_summary["area_sum_2d"]),
            "sum_final_collision_area_2d": float(final_collision_summary["area_sum_2d"]),
            "sum_initial_collision_volume_3d": float(initial_collision_summary["volume_sum_3d"]),
            "sum_final_collision_volume_3d": float(final_collision_summary["volume_sum_3d"]),
            "sum_position_l2_before": float(row["corrupted_metrics"]["position_l2_m"]),
            "sum_position_l2_after": float(repaired_metrics["position_l2_m"]),
            "sum_yaw_abs_error_before": float(row["corrupted_metrics"]["yaw_abs_error_deg"]),
            "sum_yaw_abs_error_after": float(repaired_metrics["yaw_abs_error_deg"]),
            "sum_quality_before": float(row["corrupted_metrics"]["quality_score"]),
            "sum_quality_after": float(repaired_metrics["quality_score"]),
            "sum_target_overlap_ratio": float(target_overlap_ratio),
        }
        update_stats(overall, sample_stats)
        update_stats(by_corruption_type[corruption_type], sample_stats)

        per_sample_rows.append(
            {
                "sample_id": sample_id,
                "split": as_str(row.get("split")),
                "room_type": as_str(row.get("room_type")),
                "corruption_type": corruption_type,
                "target_id": target_id,
                "target_category": as_str(row.get("target_category")),
                "selector_candidates": selector_candidates,
                "selector_top1_hit": bool(selector_top1_hit),
                "selector_topk_hit": bool(selector_topk_hit),
                "wrapper_candidate_indices": wrapper_candidate_indices,
                "wrapper_candidate_hit": bool(wrapper_candidate_hit),
                "accepted_move_count": int(accepted_moves),
                "accepted_ids": accepted_ids,
                "accepted_target": bool(target_id in accepted_ids),
                "initial_bad_count": len(initial_bad_indices),
                "final_bad_count": len(final_bad_indices),
                "initial_collision_summary": initial_collision_summary,
                "final_collision_summary": final_collision_summary,
                "corrupted_metrics": row["corrupted_metrics"],
                "repaired_metrics": repaired_metrics,
                "target_overlap_ratio": round(target_overlap_ratio, 6),
                "target_overlap50": bool(target_overlap_ratio >= 0.50),
                "target_overlap80": bool(target_overlap_ratio >= 0.80),
                "target_overlap95": bool(target_overlap_ratio >= 0.95),
                "target_position_delta_m": round(target_pos_delta, 6),
                "target_yaw_delta_deg": round(target_yaw_delta, 6),
            }
        )

    summary = {
        "schema": "scene_repair_eval.v1",
        "samples_jsonl": str(Path(args.samples_jsonl).expanduser().resolve()),
        "model": str(Path(args.model).expanduser().resolve()),
        "selector_model": str(Path(args.selector_model).expanduser().resolve()) if as_str(args.selector_model).strip() else None,
        "split": split or "all",
        "limit": int(args.limit),
        "max_passes": int(args.max_passes),
        "candidate_limit": int(args.candidate_limit),
        "selector_topk": int(args.selector_topk),
        "selector_candidate_limit": int(args.selector_candidate_limit),
        "selector_global_fallback_k": int(args.selector_global_fallback_k),
        "oracle_target": bool(args.oracle_target),
        "overall": finalize_stats(overall),
        "by_corruption_type": {k: finalize_stats(v) for k, v in sorted(by_corruption_type.items())},
    }
    save_json(args.out_json, summary)
    if as_str(args.out_jsonl).strip():
        out_jsonl = Path(args.out_jsonl).expanduser().resolve()
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with out_jsonl.open("w", encoding="utf-8") as f:
            for row in per_sample_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    overall_summary = summary["overall"]
    print(f"[scene_repair_eval_v1] samples_total={overall_summary['samples_total']}")
    print(f"[scene_repair_eval_v1] accepted_move_rate={overall_summary['accepted_move_rate']:.4f}")
    print(f"[scene_repair_eval_v1] accepted_target_rate={overall_summary['accepted_target_rate']:.4f}")
    print(f"[scene_repair_eval_v1] selector_topk_hit_rate={overall_summary['selector_topk_hit_rate']:.4f}")
    print(f"[scene_repair_eval_v1] wrapper_candidate_hit_rate={overall_summary['wrapper_candidate_hit_rate']:.4f}")
    print(f"[scene_repair_eval_v1] collision_pair_reduction_rate={overall_summary['collision_pair_reduction_rate']:.4f}")
    print(f"[scene_repair_eval_v1] collision_pair_solved_rate_given_initial_collisions={overall_summary['collision_pair_solved_rate_given_initial_collisions']:.4f}")
    print(f"[scene_repair_eval_v1] valid_rate_after_repair={overall_summary['valid_rate_after_repair']:.4f}")
    print(f"[scene_repair_eval_v1] success_rate_after_repair={overall_summary['success_rate_after_repair']:.4f}")
    print(f"[scene_repair_eval_v1] target_overlap50_rate={overall_summary['target_overlap50_rate']:.4f}")
    print(f"[scene_repair_eval_v1] target_overlap80_rate={overall_summary['target_overlap80_rate']:.4f}")
    print(f"[scene_repair_eval_v1] target_overlap95_rate={overall_summary['target_overlap95_rate']:.4f}")
    print(f"[scene_repair_eval_v1] mean_target_overlap_ratio={overall_summary['mean_target_overlap_ratio']:.4f}")
    print(f"[scene_repair_eval_v1] wrote_summary={Path(args.out_json).expanduser().resolve()}")
    if as_str(args.out_jsonl).strip():
        print(f"[scene_repair_eval_v1] wrote_samples={Path(args.out_jsonl).expanduser().resolve()}")


if __name__ == "__main__":
    main()
