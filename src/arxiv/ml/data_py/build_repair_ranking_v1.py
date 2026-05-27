from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

try:
    from src.ml.baselines.repair_heuristic_v1 import (
        candidate_score,
        generate_candidates,
        reconstruct_corrupted_scene,
    )
    from src.ml.data_py.build_repair_corruptions_v1 import compute_metrics, copy_scene_with_target
except ModuleNotFoundError:
    from repair_heuristic_v1 import candidate_score, generate_candidates, reconstruct_corrupted_scene
    from build_repair_corruptions_v1 import compute_metrics, copy_scene_with_target


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


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
    return float((angle + 180.0) % 360.0 - 180.0)


def norm_room_xy(pos_xy: List[float], room_json: Dict[str, Any]) -> Tuple[float, float]:
    bounds = room_json["bounds_xz"]
    x_min = float(bounds["x_min"])
    x_max = float(bounds["x_max"])
    y_min = float(bounds["z_min"])
    y_max = float(bounds["z_max"])
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    hx = max(0.5 * (x_max - x_min), 1e-6)
    hy = max(0.5 * (y_max - y_min), 1e-6)
    return ((float(pos_xy[0]) - cx) / hx, (float(pos_xy[1]) - cy) / hy)


def room_size(room_json: Dict[str, Any]) -> Tuple[float, float]:
    bounds = room_json["bounds_xz"]
    return (
        max(float(bounds["x_max"]) - float(bounds["x_min"]), 1e-6),
        max(float(bounds["z_max"]) - float(bounds["z_min"]), 1e-6),
    )


def candidate_feature_dict(
    sample: Dict[str, Any],
    room_json: Dict[str, Any],
    corrupted_target: Dict[str, Any],
    candidate_pose: Dict[str, Any],
    candidate_metrics: Dict[str, Any],
    heuristic_score: float,
) -> Dict[str, Any]:
    room_w, room_h = room_size(room_json)
    corr_pos = np.array(corrupted_target["position_m"], dtype=np.float32)
    cand_pos = np.array(candidate_pose["position_m"], dtype=np.float32)
    dpos = cand_pos - corr_pos
    cand_nx, cand_ny = norm_room_xy(candidate_pose["position_m"][:2], room_json)
    corr_nx, corr_ny = norm_room_xy(corrupted_target["position_m"][:2], room_json)
    dyaw = wrap_angle_deg(float(candidate_pose["yaw_deg"]) - float(corrupted_target["yaw_deg"]))
    yaw_rad = math.radians(float(candidate_pose["yaw_deg"]))

    return {
        "target_category": sample["target_category"],
        "target_super_category": sample["target_super_category"],
        "corruption_type": sample["corruption_type"],
        "heuristic_score": round6(heuristic_score),
        "candidate_norm_xy": [round6(cand_nx), round6(cand_ny)],
        "corrupted_norm_xy": [round6(corr_nx), round6(corr_ny)],
        "candidate_delta_from_corrupted_m": [round6(float(v)) for v in dpos.tolist()],
        "candidate_delta_from_corrupted_roomnorm": [
            round6(float(dpos[0]) / room_w),
            round6(float(dpos[1]) / room_h),
            round6(float(dpos[2]) / max(float(candidate_pose["size_m"][2]), 1e-6)),
        ],
        "candidate_dyaw_deg_from_corrupted": round6(dyaw),
        "candidate_yaw_sin": round6(math.sin(yaw_rad)),
        "candidate_yaw_cos": round6(math.cos(yaw_rad)),
        "target_size_m": [round6(float(v)) for v in candidate_pose["size_m"]],
        "candidate_metrics": {
            "collision_pair_count": int(candidate_metrics["collision_pair_count"]),
            "collision_area_sum_2d": round6(float(candidate_metrics["collision_area_sum_2d"])),
            "collision_volume_sum_3d": round6(float(candidate_metrics["collision_volume_sum_3d"])),
            "corners_inside_ratio": round6(float(candidate_metrics["corners_inside_ratio"])),
            "center_inside_room": bool(candidate_metrics["center_inside_room"]),
            "outside_room": bool(candidate_metrics["outside_room"]),
            "floor_contact_abs_error_m": round6(float(candidate_metrics["floor_contact_abs_error_m"])),
            "valid": bool(candidate_metrics["valid"]),
        },
        "label": {
            "quality_score": round6(float(candidate_metrics["quality_score"])),
            "position_l2_m": round6(float(candidate_metrics["position_l2_m"])),
            "yaw_abs_error_deg": round6(float(candidate_metrics["yaw_abs_error_deg"])),
            "is_valid": bool(candidate_metrics["valid"]),
        },
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build candidate-ranking dataset for repair scorer")
    ap.add_argument("--samples", required=True, help="Path to repair_corruptions samples.jsonl")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-candidates-per-sample", type=int, default=24)
    ap.add_argument("--top-valid", type=int, default=8)
    ap.add_argument("--top-any", type=int, default=8)
    ap.add_argument("--random-rest", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    samples_path = Path(args.samples).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    sample_rows: List[Dict[str, Any]] = []
    with samples_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            sample_rows.append(json.loads(line))
            if args.limit > 0 and len(sample_rows) >= int(args.limit):
                break

    out_jsonl = out_dir / "ranking_samples.jsonl"
    sample_count = 0
    candidate_count = 0
    best_valid_hits = 0
    best_any_valid_hits = 0
    valid_candidates_total = 0

    with out_jsonl.open("w", encoding="utf-8") as f:
        for sample in sample_rows:
            room_json = load_json(sample["room_ref"])
            room_polygon = room_json["floor_polygon_xz"]
            _, corrupted_scene, clean_target = reconstruct_corrupted_scene(sample)
            corrupted_target = next(p for p in corrupted_scene if as_str(p["id"]) == as_str(sample["target_object_id"]))

            rows: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []
            for cand in generate_candidates(corrupted_target, room_polygon):
                cand_scene = copy_scene_with_target(corrupted_scene, as_str(sample["target_object_id"]), cand)
                cand_metrics = compute_metrics(
                    clean_target=clean_target,
                    eval_target=cand,
                    eval_scene_placements=cand_scene,
                    room_polygon=room_polygon,
                )
                hscore = candidate_score(cand, corrupted_target, room_polygon)
                rows.append((hscore, cand, cand_metrics))

            rows.sort(key=lambda x: (float(x[2]["quality_score"]), float(x[0])), reverse=True)
            best_quality = float(rows[0][2]["quality_score"]) if rows else 0.0
            best_valid = next((r for r in rows if r[2]["valid"]), None)
            if best_valid is not None:
                valid_candidates_total += 1

            valid_sorted = [r for r in rows if r[2]["valid"]]
            any_sorted = sorted(rows, key=lambda x: float(x[0]), reverse=True)

            picked: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []
            seen = set()

            def add_rows(src: List[Tuple[float, Dict[str, Any], Dict[str, Any]]], k: int) -> None:
                for row in src[:k]:
                    key = (
                        tuple(round6(float(v)) for v in row[1]["position_m"]),
                        round6(float(row[1]["yaw_deg"])),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    picked.append(row)

            add_rows(valid_sorted, int(args.top_valid))
            add_rows(any_sorted, int(args.top_any))

            remaining = [r for r in rows if (
                tuple(round6(float(v)) for v in r[1]["position_m"]),
                round6(float(r[1]["yaw_deg"])),
            ) not in seen]
            if remaining:
                rng.shuffle(remaining)
                add_rows(remaining, int(args.random_rest))

            picked = picked[: int(args.max_candidates_per_sample)]
            if not picked:
                continue

            candidate_entries: List[Dict[str, Any]] = []
            best_idx = None
            for idx, (hscore, cand_pose, cand_metrics) in enumerate(picked):
                entry = candidate_feature_dict(
                    sample=sample,
                    room_json=room_json,
                    corrupted_target=corrupted_target,
                    candidate_pose=cand_pose,
                    candidate_metrics=cand_metrics,
                    heuristic_score=hscore,
                )
                entry["candidate_pose"] = {
                    "position_m": [round6(float(v)) for v in cand_pose["position_m"]],
                    "yaw_deg": round6(float(cand_pose["yaw_deg"])),
                    "yaw_rad": round6(float(cand_pose["yaw_rad"])),
                }
                candidate_entries.append(entry)
                if float(cand_metrics["quality_score"]) >= best_quality - 1e-8 and best_idx is None:
                    best_idx = idx

            if best_idx is None:
                best_idx = 0

            if candidate_entries[best_idx]["label"]["is_valid"]:
                best_valid_hits += 1
            if any(entry["label"]["is_valid"] for entry in candidate_entries):
                best_any_valid_hits += 1

            payload = {
                "schema": "repair_ranking_sample.v1",
                "sample_id": sample["sample_id"],
                "room_id": sample["room_id"],
                "room_type": sample["room_type"],
                "split": sample["split"],
                "target_object_id": sample["target_object_id"],
                "target_category": sample["target_category"],
                "target_super_category": sample["target_super_category"],
                "corruption_type": sample["corruption_type"],
                "clean_scene_ref": sample["clean_scene_ref"],
                "room_ref": sample["room_ref"],
                "best_candidate_index": int(best_idx),
                "corrupted_metrics": sample["corrupted_metrics"],
                "candidates": candidate_entries,
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            sample_count += 1
            candidate_count += len(candidate_entries)

    stats = {
        "samples_total": sample_count,
        "candidates_total": candidate_count,
        "mean_candidates_per_sample": round6(candidate_count / max(sample_count, 1)),
        "samples_with_valid_best_candidate": int(best_valid_hits),
        "samples_with_any_valid_candidate": int(best_any_valid_hits),
        "valid_candidate_coverage": round6(best_any_valid_hits / max(sample_count, 1)),
    }
    save_json(out_dir / "stats.json", stats)

    print(f"[repair_ranking_v1] samples={sample_count}")
    print(f"[repair_ranking_v1] candidates_total={candidate_count}")
    print(f"[repair_ranking_v1] mean_candidates_per_sample={stats['mean_candidates_per_sample']}")
    print(f"[repair_ranking_v1] valid_candidate_coverage={stats['valid_candidate_coverage']}")
    print(f"[repair_ranking_v1] wrote samples={out_jsonl}")
    print(f"[repair_ranking_v1] wrote stats={out_dir / 'stats.json'}")


if __name__ == "__main__":
    main()
