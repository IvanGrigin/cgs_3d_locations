# src/ml/train/eval_graph_stat.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from src.ml.baselines.graph_stat_placer import (
    GraphStatPlacer,
    _point_in_poly,
    _get_object_dims,
    _rect_from_center_dims,
    _rect_intersect,
)


def load_json(p: str) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_rooms(mini_json_path: str) -> List[Dict[str, Any]]:
    data = load_json(mini_json_path)
    rooms = data.get("rooms", [])
    if not isinstance(rooms, list):
        return []
    return rooms


def poly_from_room(room: Dict[str, Any]) -> np.ndarray:
    poly_raw = room.get("floor_polygon_xz", [])
    if not isinstance(poly_raw, list) or len(poly_raw) < 3:
        return np.zeros((0, 2), dtype=np.float32)
    return np.array([(float(v["x"]), float(v["z"])) for v in poly_raw], dtype=np.float32)


def rmse_mae(dxz: np.ndarray) -> Tuple[float, float]:
    if dxz.shape[0] == 0:
        return float("nan"), float("nan")
    dist = np.sqrt(np.sum(dxz * dxz, axis=1))
    rmse = float(np.sqrt(np.mean(dist * dist)))
    mae = float(np.mean(dist))
    return rmse, mae


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to runs/graph_stat/model.pkl")
    ap.add_argument("--inputs", required=True, help="Glob for *.mini.json (supports ** with recursive=True)")
    ap.add_argument("--max_files", type=int, default=0, help="0=all files, иначе ограничение")
    ap.add_argument("--out_run_json", required=True, help="Where to write run json (for report_runs.py)")
    ap.add_argument("--postprocess", default="none", choices=["none"], help="Reserved (compat)")
    args = ap.parse_args()

    placer = GraphStatPlacer.load(args.model)

    files = glob.glob(args.inputs, recursive=True)
    files = [f for f in files if f.endswith(".mini.json")]
    files.sort()
    if args.max_files and args.max_files > 0:
        files = files[: args.max_files]

    if not files:
        raise SystemExit(f"[eval_graph_stat] No files by glob: {args.inputs}")

    dxz_all: List[np.ndarray] = []
    boundary_viol = 0
    total_obj = 0

    coll_pairs = 0
    total_pairs = 0

    bad_rooms = 0

    for fp in files:
        for room in iter_rooms(fp):
            objs = room.get("objects", [])
            if not isinstance(objs, list) or len(objs) == 0:
                continue

            poly = poly_from_room(room)
            if poly.shape[0] < 3:
                bad_rooms += 1
                continue

            preds = placer.predict_room(room)
            pred_by_id = {p.get("instanceid"): p for p in preds}

            rects: List[Tuple[float, float, float, float]] = []

            for i, obj in enumerate(objs):
                pos = obj.get("pos")
                if not isinstance(pos, dict):
                    continue

                gx = float(pos.get("x", float("nan")))
                gz = float(pos.get("z", float("nan")))
                if not (math.isfinite(gx) and math.isfinite(gz)):
                    continue

                pid = obj.get("instanceid")
                p = pred_by_id.get(pid)
                if p is None:
                    if i < len(preds):
                        p = preds[i]
                    else:
                        continue

                px = float(p["pred"]["pos"]["x"])
                pz = float(p["pred"]["pos"]["z"])

                if not _point_in_poly((px, pz), poly):
                    boundary_viol += 1

                dxz_all.append(np.array([px - gx, pz - gz], dtype=np.float32))
                total_obj += 1

                w, d = _get_object_dims(obj)
                if w > 0 and d > 0:
                    rects.append(_rect_from_center_dims(px, pz, w, d))

            m = len(rects)
            if m >= 2:
                total_pairs += m * (m - 1) // 2
                for i in range(m):
                    for j in range(i + 1, m):
                        if _rect_intersect(rects[i], rects[j]):
                            coll_pairs += 1

    if total_obj == 0:
        raise SystemExit("[eval_graph_stat] No valid objects for evaluation (total_obj=0).")

    dxz = np.stack(dxz_all, axis=0)
    rmse_xz, mae_xz = rmse_mae(dxz)
    bvr = float(boundary_viol) / float(total_obj) if total_obj else float("nan")
    cpr = float(coll_pairs) / float(total_pairs) if total_pairs else 0.0

    run = {
        "model": "graph_stat",
        "postprocess": args.postprocess,
        "metrics": {
            "RMSE_xz": rmse_xz,
            "MAE_xz": mae_xz,
            "BoundaryViolRate": bvr,
            "CollisionPairRate": cpr,
        },
        "meta": {
            "files": len(files),
            "total_obj": total_obj,
            "boundary_viol": boundary_viol,
            "collision_pairs": coll_pairs,
            "total_pairs": total_pairs,
            "bad_rooms": bad_rooms,
        },
    }

    outp = Path(args.out_run_json)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[eval_graph_stat] saved: {outp}")
    print(f"[eval_graph_stat] RMSE_xz={rmse_xz:.6f} MAE_xz={mae_xz:.6f} BVR={bvr:.6f} CPR={cpr:.6f}")


if __name__ == "__main__":
    main()
