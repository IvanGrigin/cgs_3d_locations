#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[3]))

try:
    from src.ml.data_py.build_repair_corruptions_v1 import (
        as_str,
        clone_placement,
        compute_metrics,
        copy_scene_with_target,
        generate_corrupted_target,
        is_floor_candidate,
        round6,
    )
except ModuleNotFoundError:
    from build_repair_corruptions_v1 import (  # type: ignore
        as_str,
        clone_placement,
        compute_metrics,
        copy_scene_with_target,
        generate_corrupted_target,
        is_floor_candidate,
        round6,
    )


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def save_json(path: str | Path, data: Any) -> None:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


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


def infer_room_type(scene: Dict[str, Any], fallback: str) -> str:
    room = scene.get("room") or {}
    meta = scene.get("meta") or {}
    return (
        as_str(room.get("room_type")).strip().lower()
        or as_str(room.get("type")).strip().lower()
        or as_str(meta.get("room_type")).strip().lower()
        or as_str(fallback).strip().lower()
        or "unknown"
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build single-scene corruption samples for repair fine-tuning")
    ap.add_argument("--scene", required=True, help="Input scene.v1 / supplier scene json")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--target-ids", default="", help="Comma-separated placement ids; if empty, uses lamp-like floor objects")
    ap.add_argument("--room-type", default="bedroom")
    ap.add_argument("--split", default="train")
    ap.add_argument("--num-samples-per-target", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--corruption-types",
        default="collision_shift,out_of_room_shift,yaw_only,displaced_inside_room,shift_and_yaw",
    )
    return ap.parse_args()


def auto_target_ids(scene: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for p in scene.get("placements") or []:
        if not is_floor_candidate(p):
            continue
        text = " ".join(
            [
                as_str(p.get("id")).lower(),
                as_str(p.get("name")).lower(),
                as_str(p.get("category")).lower(),
                as_str(p.get("class_name")).lower(),
            ]
        )
        if "lamp" in text or "light" in text or "торшер" in text or "свет" in text:
            out.append(as_str(p["id"]))
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    scene = load_json(args.scene)
    if not scene.get("schema"):
        scene["schema"] = "scene.v1"
    room_json = room_json_from_scene(scene)
    room_polygon = room_json["floor_polygon_xz"]
    placements = [clone_placement(p) for p in (scene.get("placements") or [])]
    floor_placements = [p for p in placements if is_floor_candidate(p)]

    target_ids = [x.strip() for x in as_str(args.target_ids).split(",") if x.strip()]
    if not target_ids:
        target_ids = auto_target_ids(scene)
    if not target_ids:
        raise RuntimeError("No target ids selected")

    corruption_types = [x.strip() for x in as_str(args.corruption_types).split(",") if x.strip()]
    if not corruption_types:
        raise RuntimeError("No corruption types specified")

    clean_scene_ref = out_dir / "scene.clean.v1.json"
    save_json(clean_scene_ref, scene)
    room_ref = out_dir / "room.json"
    save_json(room_ref, room_json)

    rng = np.random.default_rng(int(args.seed))
    rows: List[Dict[str, Any]] = []
    manifest: List[Dict[str, Any]] = []
    room_type = infer_room_type(scene, args.room_type)
    split = as_str(args.split).strip().lower() or "train"

    for target_id in target_ids:
        clean_target = next((clone_placement(p) for p in placements if as_str(p["id"]) == target_id), None)
        if clean_target is None:
            raise RuntimeError(f"Target not found: {target_id}")

        target_out_dir = out_dir / target_id
        target_out_dir.mkdir(parents=True, exist_ok=True)
        count = int(args.num_samples_per_target)
        for i in range(count):
            corruption_type = corruption_types[i % len(corruption_types)]
            corrupted_target = generate_corrupted_target(
                corruption_type=corruption_type,
                target=clean_target,
                floor_placements=floor_placements,
                room_polygon=room_polygon,
                rng=rng,
            )
            if corrupted_target is None:
                continue

            clean_scene = copy_scene_with_target(placements, target_id, clean_target)
            corrupted_scene = copy_scene_with_target(placements, target_id, corrupted_target)
            clean_metrics = compute_metrics(
                clean_target=clean_target,
                eval_target=clean_target,
                eval_scene_placements=clean_scene,
                room_polygon=room_polygon,
            )
            corrupted_metrics = compute_metrics(
                clean_target=clean_target,
                eval_target=corrupted_target,
                eval_scene_placements=corrupted_scene,
                room_polygon=room_polygon,
            )

            row = {
                "schema": "repair_sample.v1",
                "sample_id": f"{target_id}_{i:05d}",
                "room_id": as_str((scene.get("room") or {}).get("id"), "scene_room"),
                "room_type": room_type,
                "split": split,
                "target_object_id": target_id,
                "target_category": as_str(clean_target.get("category")),
                "target_super_category": as_str(((clean_target.get("meta") or {}).get("super_category"))),
                "corruption_type": corruption_type,
                "context_unchanged": True,
                "clean_scene_ref": str(clean_scene_ref),
                "room_ref": str(room_ref),
                "clean_pose": {
                    "position_m": [round6(v) for v in clean_target["position_m"]],
                    "size_m": [round6(v) for v in clean_target["size_m"]],
                    "yaw_deg": round6(float(clean_target["yaw_deg"])),
                    "yaw_rad": round6(float(clean_target["yaw_rad"])),
                    "mount_type": as_str(clean_target.get("mount_type")),
                },
                "corrupted_pose": {
                    "position_m": [round6(v) for v in corrupted_target["position_m"]],
                    "size_m": [round6(v) for v in corrupted_target["size_m"]],
                    "yaw_deg": round6(float(corrupted_target["yaw_deg"])),
                    "yaw_rad": round6(float(corrupted_target["yaw_rad"])),
                    "mount_type": as_str(corrupted_target.get("mount_type")),
                },
                "delta": {
                    "dx_m": round6(float(corrupted_target["position_m"][0] - clean_target["position_m"][0])),
                    "dy_m": round6(float(corrupted_target["position_m"][1] - clean_target["position_m"][1])),
                    "dz_m": round6(float(corrupted_target["position_m"][2] - clean_target["position_m"][2])),
                    "translation_l2_m": round6(
                        float(np.linalg.norm(np.asarray(corrupted_target["position_m"]) - np.asarray(clean_target["position_m"])))
                    ),
                    "dyaw_deg": round6(float(corrupted_target["yaw_deg"] - clean_target["yaw_deg"])),
                },
                "clean_metrics": clean_metrics,
                "corrupted_metrics": corrupted_metrics,
            }
            rows.append(row)
            manifest.append(
                {
                    "sample_id": row["sample_id"],
                    "target_id": target_id,
                    "corruption_type": corruption_type,
                }
            )

    jsonl_path = out_dir / "samples.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    save_json(out_dir / "manifest.json", manifest)
    save_json(
        out_dir / "stats.json",
        {
            "scene": str(Path(args.scene).expanduser().resolve()),
            "clean_scene_ref": str(clean_scene_ref),
            "target_ids": target_ids,
            "num_rows": len(rows),
            "room_type": room_type,
            "split": split,
            "num_samples_per_target": int(args.num_samples_per_target),
            "corruption_types": corruption_types,
        },
    )
    print(f"[build_scene_repair_corruptions_v1] wrote rows={len(rows)}")
    print(f"[build_scene_repair_corruptions_v1] wrote jsonl={jsonl_path}")


if __name__ == "__main__":
    main()
