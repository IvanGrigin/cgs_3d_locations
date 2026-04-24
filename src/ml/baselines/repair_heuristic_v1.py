from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))

try:
    from src.ml.data_py.build_repair_corruptions_v1 import (
        compute_metrics,
        copy_scene_with_target,
        corners_inside_ratio,
        footprint_iou,
        point_in_polygon,
        room_bounds,
        update_aabb_for_placement,
        wrap_angle_deg,
    )
except ModuleNotFoundError:
    from build_repair_corruptions_v1 import (  # type: ignore
        compute_metrics,
        copy_scene_with_target,
        corners_inside_ratio,
        footprint_iou,
        point_in_polygon,
        room_bounds,
        update_aabb_for_placement,
        wrap_angle_deg,
    )


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


def clone_json(x: Any) -> Any:
    return json.loads(json.dumps(x))


def reconstruct_corrupted_scene(sample: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    scene_gt = load_json(sample["clean_scene_ref"])
    placements = clone_json(scene_gt["placements"])
    target_id = as_str(sample["target_object_id"])
    clean_target = None
    corrupted_target = None
    corrupted_scene: List[Dict[str, Any]] = []
    for p in placements:
        if as_str(p["id"]) == target_id:
            clean_target = clone_json(p)
            corr = clone_json(p)
            corr_pose = sample["corrupted_pose"]
            corr["position_m"] = list(corr_pose["position_m"])
            corr["yaw_deg"] = float(corr_pose["yaw_deg"])
            corr["yaw_rad"] = float(corr_pose["yaw_rad"])
            corr["rotation_deg"] = int(round(float(corr["yaw_deg"]))) % 360
            corr["aabb"] = update_aabb_for_placement(corr)
            corrupted_target = corr
            corrupted_scene.append(corr)
        else:
            corrupted_scene.append(clone_json(p))
    if clean_target is None or corrupted_target is None:
        raise RuntimeError(f"target_id={target_id} not found in clean scene")
    return scene_gt, corrupted_scene, clean_target


def candidate_score(candidate: Dict[str, Any], corrupted_target: Dict[str, Any], room_polygon: List[List[float]]) -> float:
    iou, _ = footprint_iou(candidate, corrupted_target)
    cpos = np.array(candidate["position_m"], dtype=np.float32)
    opos = np.array(corrupted_target["position_m"], dtype=np.float32)
    trans_pen = float(np.linalg.norm(cpos - opos))
    yaw_pen = abs(wrap_angle_deg(float(candidate["yaw_deg"]) - float(corrupted_target["yaw_deg"])))
    _, inside_ratio = corners_inside_ratio(candidate, room_polygon)
    center_inside = point_in_polygon((float(candidate["position_m"][0]), float(candidate["position_m"][1])), room_polygon)
    return (
        2.0 * iou
        - 0.75 * trans_pen
        - 0.01 * yaw_pen
        + 0.25 * inside_ratio
        + (0.25 if center_inside else -0.50)
    )


def generate_candidates(
    corrupted_target: Dict[str, Any],
    room_polygon: List[List[float]],
) -> Iterable[Dict[str, Any]]:
    x_min, y_min, x_max, y_max = room_bounds(room_polygon)
    base = clone_json(corrupted_target)
    size_x, size_y, _ = [float(v) for v in base["size_m"]]
    x0, y0, z0 = [float(v) for v in base["position_m"]]
    yaw0 = float(base["yaw_deg"])

    # Always evaluate the current pose too.
    yield clone_json(base)

    trans_steps = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60]
    yaw_steps = [0.0, -15.0, 15.0, -30.0, 30.0, -45.0, 45.0, -90.0, 90.0, 180.0]
    directions = [
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (1.0, 1.0),
        (1.0, -1.0),
        (-1.0, 1.0),
        (-1.0, -1.0),
    ]

    for ds in trans_steps:
        for dx, dy in directions:
            norm = math.sqrt(dx * dx + dy * dy)
            ux, uy = dx / norm, dy / norm
            nx = x0 + ux * ds
            ny = y0 + uy * ds
            nx = float(np.clip(nx, x_min - 0.5 * size_x, x_max + 0.5 * size_x))
            ny = float(np.clip(ny, y_min - 0.5 * size_y, y_max + 0.5 * size_y))
            for dyaw in yaw_steps:
                cand = clone_json(base)
                cand["position_m"] = [round6(nx), round6(ny), round6(z0)]
                cand["yaw_deg"] = round6(yaw0 + dyaw)
                cand["yaw_rad"] = round6(math.radians(float(cand["yaw_deg"])))
                cand["rotation_deg"] = int(round(float(cand["yaw_deg"]))) % 360
                cand["aabb"] = update_aabb_for_placement(cand)
                yield cand


def solve_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    room_json = load_json(sample["room_ref"])
    room_polygon = room_json["floor_polygon_xz"]
    scene_gt, corrupted_scene, clean_target = reconstruct_corrupted_scene(sample)
    corrupted_target = next(p for p in corrupted_scene if as_str(p["id"]) == as_str(sample["target_object_id"]))

    best_valid: Optional[Tuple[float, Dict[str, Any], Dict[str, Any]]] = None
    best_any: Optional[Tuple[float, Dict[str, Any], Dict[str, Any]]] = None

    for cand in generate_candidates(corrupted_target, room_polygon):
        cand_scene = copy_scene_with_target(corrupted_scene, as_str(sample["target_object_id"]), cand)
        metrics = compute_metrics(
            clean_target=clean_target,
            eval_target=cand,
            eval_scene_placements=cand_scene,
            room_polygon=room_polygon,
        )
        score = candidate_score(cand, corrupted_target, room_polygon)
        if best_any is None or score > best_any[0]:
            best_any = (score, cand, metrics)
        if metrics["valid"] and (best_valid is None or score > best_valid[0]):
            best_valid = (score, cand, metrics)

    if best_any is None:
        raise RuntimeError("No candidate generated")

    chosen = best_valid if best_valid is not None else best_any
    chosen_score, chosen_pose, chosen_metrics = chosen

    baseline = {
        "sample_id": sample["sample_id"],
        "room_id": sample["room_id"],
        "target_object_id": sample["target_object_id"],
        "target_category": sample["target_category"],
        "corruption_type": sample["corruption_type"],
        "solver_found_valid": bool(best_valid is not None),
        "chosen_score": round6(chosen_score),
        "corrupted_metrics": sample["corrupted_metrics"],
        "repaired_pose": {
            "position_m": [round6(v) for v in chosen_pose["position_m"]],
            "yaw_deg": round6(float(chosen_pose["yaw_deg"])),
            "yaw_rad": round6(float(chosen_pose["yaw_rad"])),
        },
        "repaired_metrics": chosen_metrics,
    }
    return baseline


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Heuristic single-object repair baseline")
    ap.add_argument("--samples", required=True, help="Path to corruption samples.jsonl")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    samples_path = Path(args.samples).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    samples: List[Dict[str, Any]] = []
    with samples_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            samples.append(json.loads(line))
            if args.limit > 0 and len(samples) >= int(args.limit):
                break

    rows: List[Dict[str, Any]] = []
    by_corruption = Counter()
    success = 0
    valid = 0
    improved = 0
    mean_pos = []
    mean_yaw = []

    out_jsonl = out_dir / "results.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for sample in samples:
            row = solve_sample(sample)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
            by_corruption[row["corruption_type"]] += 1
            repaired = row["repaired_metrics"]
            corrupted = row["corrupted_metrics"]
            if repaired["valid"]:
                valid += 1
            if repaired["position_l2_m"] <= 0.20 and repaired["yaw_abs_error_deg"] <= 15.0 and repaired["valid"]:
                success += 1
            if repaired["quality_score"] > corrupted["quality_score"]:
                improved += 1
            mean_pos.append(float(repaired["position_l2_m"]))
            mean_yaw.append(float(repaired["yaw_abs_error_deg"]))

    total = max(len(rows), 1)
    stats = {
        "samples_total": len(rows),
        "valid_rate_after_repair": round6(valid / total),
        "success_rate": round6(success / total),
        "quality_improved_rate": round6(improved / total),
        "mean_position_l2_m": round6(sum(mean_pos) / total if mean_pos else 0.0),
        "mean_yaw_abs_error_deg": round6(sum(mean_yaw) / total if mean_yaw else 0.0),
        "by_corruption_type": dict(by_corruption),
    }
    save_json(out_dir / "stats.json", stats)

    print(f"[repair_heuristic_v1] samples={len(rows)}")
    print(f"[repair_heuristic_v1] valid_rate_after_repair={stats['valid_rate_after_repair']}")
    print(f"[repair_heuristic_v1] success_rate={stats['success_rate']}")
    print(f"[repair_heuristic_v1] quality_improved_rate={stats['quality_improved_rate']}")
    print(f"[repair_heuristic_v1] wrote results={out_jsonl}")
    print(f"[repair_heuristic_v1] wrote stats={out_dir / 'stats.json'}")


if __name__ == "__main__":
    main()
