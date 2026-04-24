#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[3]))

try:
    from src.ml.data_py.build_corrupted_object_selector_v1 import (
        candidate_record,
        global_anomaly_tuple,
        is_important_furniture_candidate,
        is_important_furniture_metadata,
        safe_room_json,
        bounds_and_room_area,
    )
    from src.ml.data_py.build_repair_corruptions_v1 import as_str, clone_placement, update_aabb_for_placement
    from src.ml.data_py.repair_proposal_dataset_v1 import reconstruct_corrupted_scene, load_sample_rows
except ModuleNotFoundError:
    from build_corrupted_object_selector_v1 import (  # type: ignore
        candidate_record,
        global_anomaly_tuple,
        is_important_furniture_candidate,
        is_important_furniture_metadata,
        safe_room_json,
        bounds_and_room_area,
    )
    from build_repair_corruptions_v1 import as_str, clone_placement, update_aabb_for_placement  # type: ignore
    from repair_proposal_dataset_v1 import reconstruct_corrupted_scene, load_sample_rows  # type: ignore


def save_json(path: str | Path, data: Any) -> None:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def round6(v: float) -> float:
    return round(float(v), 6)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build general furniture node-detector dataset from repair_sample.v1 rows"
    )
    ap.add_argument("--samples-jsonl", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--limit-val", type=int, default=0)
    ap.add_argument("--limit-test", type=int, default=0)
    ap.add_argument("--allowed-corruption-types", default="collision_shift,out_of_room_shift,shift_and_yaw")
    ap.add_argument("--allowed-room-types", default="")
    ap.add_argument("--allowed-target-categories", default="")
    ap.add_argument("--important-targets-only", action="store_true")
    ap.add_argument("--max-candidates", type=int, default=0, help="Optional cap after anomaly sorting; 0 keeps all important furniture nodes.")
    return ap.parse_args()


def build_group_row(sample: Dict[str, Any], max_candidates: int) -> Dict[str, Any]:
    scene_gt, corrupted_scene, _ = reconstruct_corrupted_scene(sample)
    placements = []
    for p in corrupted_scene:
        cp = clone_placement(p)
        cp["aabb"] = update_aabb_for_placement(cp)
        placements.append(cp)

    room_json = safe_room_json(sample, scene_gt)
    _, room_area = bounds_and_room_area(room_json)
    target_id = as_str(sample["target_object_id"])

    unique_candidates: List[Dict[str, Any]] = []
    seen_ids = set()
    for p in placements:
        pid = as_str(p.get("id")).strip()
        if not pid or pid in seen_ids:
            continue
        if is_important_furniture_candidate(p) or pid == target_id:
            seen_ids.add(pid)
            unique_candidates.append(p)

    if not unique_candidates:
        raise RuntimeError(f"No important candidates for sample_id={sample['sample_id']}")

    candidates = [candidate_record(p, placements, room_json, room_area, target_id) for p in unique_candidates]
    candidates = sorted(
        candidates,
        key=lambda rec: (global_anomaly_tuple(rec), as_str(rec["id"])),
        reverse=True,
    )

    if int(max_candidates) > 0 and len(candidates) > int(max_candidates):
        keep = candidates[: int(max_candidates)]
        if not any(c["is_target"] for c in keep):
            target_rec = next((c for c in candidates if c["is_target"]), None)
            if target_rec is not None:
                keep[-1] = target_rec
        dedup = []
        seen = set()
        for rec in keep:
            rid = as_str(rec["id"])
            if not rid or rid in seen:
                continue
            seen.add(rid)
            dedup.append(rec)
        candidates = dedup

    target_candidate_index = next((i for i, c in enumerate(candidates) if c["is_target"]), -1)
    if target_candidate_index < 0:
        raise RuntimeError(f"Target not found among detector candidates for sample_id={sample['sample_id']}")

    return {
        "schema": "furniture_node_detector_sample.v1",
        "sample_id": as_str(sample["sample_id"]),
        "room_id": as_str(sample.get("room_id")),
        "room_type": as_str(sample.get("room_type")),
        "split": as_str(sample.get("split")),
        "corruption_type": as_str(sample.get("corruption_type")),
        "target_object_id": target_id,
        "target_category": as_str(sample.get("target_category")),
        "target_candidate_index": int(target_candidate_index),
        "num_all_candidates": len(candidates),
        "candidate_source": "all_important_furniture_nodes_v1",
        "candidates": candidates,
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "samples.jsonl"

    limits = {
        "train": int(args.limit_train),
        "val": int(args.limit_val),
        "test": int(args.limit_test),
    }
    allowed_corruption_types = {x.strip() for x in as_str(args.allowed_corruption_types).split(",") if x.strip()}
    allowed_room_types = {x.strip() for x in as_str(args.allowed_room_types).split(",") if x.strip()}
    allowed_target_categories = {x.strip() for x in as_str(args.allowed_target_categories).split(",") if x.strip()}

    split_counts: Counter[str] = Counter()
    candidate_cat_counts: Counter[str] = Counter()
    target_cat_counts: Counter[str] = Counter()
    sample_count = 0
    total_candidates = 0
    skipped_missing_target = 0
    skipped_corruption_type = 0
    skipped_room_type = 0
    skipped_target_category = 0
    skipped_unimportant_target = 0

    with out_jsonl.open("w", encoding="utf-8") as out_f:
        for split in ("train", "val", "test"):
            try:
                rows = load_sample_rows(args.samples_jsonl, split=split, limit=limits[split])
            except RuntimeError as e:
                if "No repair proposal rows for split=" in str(e):
                    continue
                raise
            for row in rows:
                if allowed_corruption_types and as_str(row.get("corruption_type")) not in allowed_corruption_types:
                    skipped_corruption_type += 1
                    continue
                if allowed_room_types and as_str(row.get("room_type")) not in allowed_room_types:
                    skipped_room_type += 1
                    continue
                if allowed_target_categories and as_str(row.get("target_category")) not in allowed_target_categories:
                    skipped_target_category += 1
                    continue
                if args.important_targets_only and not is_important_furniture_metadata(
                    category=as_str(row.get("target_category")),
                    super_category=as_str(row.get("target_super_category")),
                    name=as_str(row.get("target_category")),
                    class_name="",
                    mount_type="floor",
                    size_m=None,
                ):
                    skipped_unimportant_target += 1
                    continue
                try:
                    group = build_group_row(row, max_candidates=int(args.max_candidates))
                except RuntimeError as e:
                    if "Target not found among detector candidates" in str(e) or "No important candidates" in str(e):
                        skipped_missing_target += 1
                        continue
                    raise
                out_f.write(json.dumps(group, ensure_ascii=False) + "\n")
                sample_count += 1
                split_counts[split] += 1
                total_candidates += len(group["candidates"])
                target_cat_counts[as_str(group["target_category"])] += 1
                for cand in group["candidates"]:
                    candidate_cat_counts[as_str(cand["category"])] += 1

    stats = {
        "samples_total": sample_count,
        "skipped_missing_target": skipped_missing_target,
        "skipped_corruption_type": skipped_corruption_type,
        "skipped_room_type": skipped_room_type,
        "skipped_target_category": skipped_target_category,
        "skipped_unimportant_target": skipped_unimportant_target,
        "samples_by_split": dict(split_counts),
        "mean_candidates_per_sample": round6(total_candidates / max(sample_count, 1)),
        "allowed_corruption_types": sorted(allowed_corruption_types),
        "allowed_room_types": sorted(allowed_room_types),
        "allowed_target_categories": sorted(allowed_target_categories),
        "important_targets_only": bool(args.important_targets_only),
        "max_candidates": int(args.max_candidates),
        "candidate_categories_top50": dict(candidate_cat_counts.most_common(50)),
        "target_categories_top50": dict(target_cat_counts.most_common(50)),
    }
    save_json(out_dir / "stats.json", stats)
    print(f"[furniture_node_detector_v1] samples={sample_count}")
    print(f"[furniture_node_detector_v1] skipped_missing_target={skipped_missing_target}")
    print(f"[furniture_node_detector_v1] skipped_corruption_type={skipped_corruption_type}")
    print(f"[furniture_node_detector_v1] skipped_room_type={skipped_room_type}")
    print(f"[furniture_node_detector_v1] skipped_target_category={skipped_target_category}")
    print(f"[furniture_node_detector_v1] skipped_unimportant_target={skipped_unimportant_target}")
    print(f"[furniture_node_detector_v1] mean_candidates_per_sample={stats['mean_candidates_per_sample']}")
    print(f"[furniture_node_detector_v1] wrote samples={out_jsonl}")
    print(f"[furniture_node_detector_v1] wrote stats={out_dir / 'stats.json'}")


if __name__ == "__main__":
    main()
