# src/ml/train/eval_forest.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from src.ml.baselines.forest_placer import ForestPlacer, _get_object_dims, _point_in_poly


# -----------------------------
# IO helpers
# -----------------------------

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_rooms(mini_json_path: str) -> List[Dict[str, Any]]:
    data = load_json(mini_json_path)
    rooms = data.get("rooms", [])
    return rooms if isinstance(rooms, list) else []


def poly_from_room(room: Dict[str, Any]) -> np.ndarray:
    poly_raw = room.get("floor_polygon_xz", [])
    if not isinstance(poly_raw, list) or len(poly_raw) < 3:
        return np.zeros((0, 2), dtype=np.float32)
    return np.array([(float(v["x"]), float(v["z"])) for v in poly_raw], dtype=np.float32)


# -----------------------------
# AABB geometry (XZ)
# -----------------------------

Rect = Tuple[float, float, float, float]  # (xmin, xmax, zmin, zmax)


def rect_from_center_dims(x: float, z: float, w: float, d: float) -> Rect:
    hw = 0.5 * float(w)
    hd = 0.5 * float(d)
    return (x - hw, x + hw, z - hd, z + hd)


def rect_intersect(a: Rect, b: Rect) -> bool:
    ax1, ax2, az1, az2 = a
    bx1, bx2, bz1, bz2 = b
    return not (ax2 <= bx1 or bx2 <= ax1 or az2 <= bz1 or bz2 <= az1)


# -----------------------------
# Greedy collision-repair (postprocess)
# -----------------------------

def greedy_repair_position(
    x: float,
    z: float,
    w: float,
    d: float,
    poly: np.ndarray,
    placed: List[Rect],
    step: float,
    max_radius: float,
) -> Tuple[float, float, Rect]:
    """
    Логика:
    - если текущая позиция не даёт пересечений с placed — принимаем
    - иначе ищем ближайшую позицию на "кольцах" радиуса step..max_radius
      по сетке смещений (dx,dz) ∈ {-r,0,+r}^2
    - точка должна быть внутри полигона (по центру)
    - пересечения проверяем по AABB в XZ (w,d)
    """
    base = rect_from_center_dims(x, z, w, d)
    if all(not rect_intersect(base, r) for r in placed):
        return x, z, base

    # если размеров нет — не можем чинить коллизии, считаем точкой
    if w <= 0.0 or d <= 0.0:
        return x, z, base

    # поиск вокруг исходной точки
    r = step
    while r <= max_radius + 1e-9:
        for dx in (-r, 0.0, r):
            for dz in (-r, 0.0, r):
                nx = float(x + dx)
                nz = float(z + dz)
                if poly.shape[0] >= 3 and (not _point_in_poly((nx, nz), poly)):
                    continue
                cand = rect_from_center_dims(nx, nz, w, d)
                if all(not rect_intersect(cand, rr) for rr in placed):
                    return nx, nz, cand
        r += step

    # fallback: не нашли — оставляем как есть
    return x, z, base


# -----------------------------
# Metrics
# -----------------------------

def rmse_mae(dxz: np.ndarray) -> Tuple[float, float]:
    """
    dxz: [N,2] (dx,dz)
    RMSE_xz: sqrt(mean(dist^2)), dist = sqrt(dx^2+dz^2)
    MAE_xz : mean(dist)
    """
    if dxz.shape[0] == 0:
        return float("nan"), float("nan")
    dist = np.sqrt(np.sum(dxz * dxz, axis=1))
    rmse = float(np.sqrt(np.mean(dist * dist)))
    mae = float(np.mean(dist))
    return rmse, mae


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to runs/forest/model.pkl")
    ap.add_argument("--inputs", required=True, help="Glob for *.mini.json (supports ** with recursive=True)")
    ap.add_argument("--max_files", type=int, default=0, help="0 = all files, иначе ограничение для быстрого теста")
    ap.add_argument("--out_run_json", required=True, help="Where to write run json (for report_runs.py)")

    ap.add_argument("--postprocess", default="none", choices=["none", "greedy"], help="Postprocess to reduce collisions")
    ap.add_argument("--pp_step", type=float, default=0.25, help="Greedy search step in meters (only for greedy)")
    ap.add_argument("--pp_max_radius", type=float, default=3.0, help="Greedy max search radius in meters (only for greedy)")
    ap.add_argument("--pp_sort", default="area_desc", choices=["none", "area_desc"], help="Order objects for greedy placement")

    args = ap.parse_args()

    placer = ForestPlacer.load(args.model)

    files = glob.glob(args.inputs, recursive=True)
    files = [f for f in files if f.endswith(".mini.json")]
    files.sort()
    if args.max_files and args.max_files > 0:
        files = files[: args.max_files]

    if not files:
        raise SystemExit(f"[eval_forest] No files by glob: {args.inputs}")

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

            # Для collision метрик считаем по предсказанным AABB в комнате.
            # Для greedy: размещаем последовательно и обновляем placed_rects.
            placed_rects: List[Rect] = []
            rects_for_metric: List[Rect] = []

            # порядок обхода объектов (важно для greedy)
            order = list(range(len(objs)))
            if args.postprocess == "greedy" and args.pp_sort == "area_desc":
                def area(i: int) -> float:
                    w, d = _get_object_dims(objs[i])
                    return float(w * d)

                order.sort(key=area, reverse=True)

            # чтобы корректно считать dxz/viol по всем объектам, идём по выбранному order
            for idx in order:
                obj = objs[idx]

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
                    if idx < len(preds):
                        p = preds[idx]
                    else:
                        continue

                px = float(p["pred"]["pos"]["x"])
                pz = float(p["pred"]["pos"]["z"])

                w, d = _get_object_dims(obj)

                # postprocess: greedy repair
                if args.postprocess == "greedy":
                    px, pz, rect = greedy_repair_position(
                        px, pz, w, d, poly, placed_rects,
                        step=float(args.pp_step),
                        max_radius=float(args.pp_max_radius),
                    )
                    placed_rects.append(rect)
                    if w > 0.0 and d > 0.0:
                        rects_for_metric.append(rect)
                else:
                    if w > 0.0 and d > 0.0:
                        rects_for_metric.append(rect_from_center_dims(px, pz, w, d))

                # boundary violation (по центру)
                if not _point_in_poly((px, pz), poly):
                    boundary_viol += 1

                # dxz
                dxz_all.append(np.array([px - gx, pz - gz], dtype=np.float32))
                total_obj += 1

            # collision pairs по rects_for_metric
            m = len(rects_for_metric)
            if m >= 2:
                total_pairs += m * (m - 1) // 2
                for i in range(m):
                    for j in range(i + 1, m):
                        if rect_intersect(rects_for_metric[i], rects_for_metric[j]):
                            coll_pairs += 1

    if total_obj == 0:
        raise SystemExit("[eval_forest] No valid objects for evaluation (total_obj=0).")

    dxz = np.stack(dxz_all, axis=0)
    rmse_xz, mae_xz = rmse_mae(dxz)
    bvr = float(boundary_viol) / float(total_obj) if total_obj else float("nan")
    cpr = float(coll_pairs) / float(total_pairs) if total_pairs else 0.0

    run = {
        "model": "forest",
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
            "pp_step": float(args.pp_step),
            "pp_max_radius": float(args.pp_max_radius),
            "pp_sort": args.pp_sort,
        },
    }

    outp = Path(args.out_run_json)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[eval_forest] saved: {outp}")
    print(f"[eval_forest] RMSE_xz={rmse_xz:.6f} MAE_xz={mae_xz:.6f} BVR={bvr:.6f} CPR={cpr:.6f}")


if __name__ == "__main__":
    main()
