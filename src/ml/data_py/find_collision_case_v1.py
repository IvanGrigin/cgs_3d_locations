#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[3]))

try:
    from src.ml.data_py.repair_proposal_dataset_v1 import load_sample_rows, reconstruct_corrupted_scene
    from src.ml.infer.apply_repair_proposal_v1 import scene_collision_summary
    from src.ml.data_py.build_repair_corruptions_v1 import update_aabb_for_placement
except ModuleNotFoundError:
    from repair_proposal_dataset_v1 import load_sample_rows, reconstruct_corrupted_scene  # type: ignore
    from apply_repair_proposal_v1 import scene_collision_summary  # type: ignore
    from build_repair_corruptions_v1 import update_aabb_for_placement  # type: ignore


def save_json(path: str | Path, data: Any) -> None:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Find first reconstructed repair sample with significant furniture collisions.")
    ap.add_argument("--samples-jsonl", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--corruption-type", default="collision_shift")
    ap.add_argument("--room-type", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-scene", required=True)
    ap.add_argument("--out-meta", required=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_sample_rows(args.samples_jsonl, split=args.split, limit=int(args.limit))
    wanted_room_type = str(args.room_type).strip()
    wanted_corruption = str(args.corruption_type).strip()

    for row in rows:
        if wanted_corruption and str(row.get("corruption_type", "")).strip() != wanted_corruption:
            continue
        if wanted_room_type and str(row.get("room_type", "")).strip() != wanted_room_type:
            continue

        scene_gt, corrupted_scene, _ = reconstruct_corrupted_scene(row)
        scene = {
            "schema": "scene.v1",
            "room": scene_gt.get("room", {}),
            "placements": [],
        }
        for p in corrupted_scene:
            cp = dict(p)
            cp["aabb"] = update_aabb_for_placement(cp)
            scene["placements"].append(cp)

        summary = scene_collision_summary(scene)
        if int(summary["pair_count"]) <= 0:
            continue

        meta: Dict[str, Any] = {
            "sample_id": row.get("sample_id"),
            "room_id": row.get("room_id"),
            "room_type": row.get("room_type"),
            "target_object_id": row.get("target_object_id"),
            "target_category": row.get("target_category"),
            "corruption_type": row.get("corruption_type"),
            "collision_summary": summary,
        }
        save_json(args.out_scene, scene)
        save_json(args.out_meta, meta)
        print(json.dumps(meta, ensure_ascii=False, indent=2))
        print(Path(args.out_scene).expanduser().resolve())
        return

    print("NO_COLLISION_CASE_FOUND")


if __name__ == "__main__":
    main()
