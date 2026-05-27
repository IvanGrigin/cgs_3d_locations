from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from src.ml.data_py.dataset_front import FrontCanonicalDataset, collate_front
from src.ml.metrics.layout_metrics import (
    rmse_xz, mae_xz, boundary_violation_rate, collision_pair_rate
)

from src.ml.baselines.random_feasible import random_feasible_layout
from src.ml.baselines.relaxed_cube import relaxed_cube_layout

# tree
try:
    import joblib
except Exception:
    joblib = None


def maybe_save_metrics(path: str | None, payload: dict) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def compute_metrics(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray,
                    size_room: np.ndarray, room_h: np.ndarray) -> dict:
    return {
        "RMSE_xz": float(rmse_xz(pred, gt, mask)),
        "MAE_xz": float(mae_xz(pred, gt, mask)),
        "BoundaryViolRate": float(boundary_violation_rate(pred, size_room, room_h, mask)),
        "CollisionPairRate": float(collision_pair_rate(pred, size_room, room_h, mask)),
    }


def eval_random_feasible(npz: str, splits: str, seed: int, batch_size: int,
                         print_unplaceable: int, save_metrics: str | None) -> None:
    ds = FrontCanonicalDataset(npz, splits, split="test")
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_front)
    rng = np.random.default_rng(seed)

    stats_total = {"unplaceable": 0, "fallback_center": 0, "placed": 0, "total_masked": 0}
    shown = 0

    all_pred, all_gt, all_mask, all_size, all_room_h = [], [], [], [], []

    for batch in dl:
        B, N = batch.pos_gt_xz.shape[0], batch.pos_gt_xz.shape[1]
        pred = np.zeros((B, N, 2), dtype=np.float32)

        room_h = batch.room_h.numpy().astype(np.float32)
        size_room = batch.size_room.numpy().astype(np.float32)
        mask = batch.mask.numpy().astype(np.float32)

        for b in range(B):
            pred_b, info_b = random_feasible_layout(room_h[b], size_room[b], mask[b], rng=rng)
            pred[b] = pred_b

            stats_total["unplaceable"] += int(info_b.get("unplaceable", 0))
            stats_total["fallback_center"] += int(info_b.get("fallback_center", 0))
            stats_total["placed"] += int(info_b.get("placed", 0))
            stats_total["total_masked"] += int(mask[b].sum())

            if print_unplaceable > 0 and shown < print_unplaceable:
                idxs = info_b.get("unplaceable_idx", []) or []
                for j in idxs:
                    if shown >= print_unplaceable:
                        break
                    hn = [(size_room[b, j, 0] * 0.5), (size_room[b, j, 2] * 0.5)]
                    print("UNPLACEABLE:",
                          "room_h=", room_h[b].tolist(),
                          "size_norm=", size_room[b, j].tolist(),
                          "half_norm(x,z)=", hn)
                    shown += 1

        all_pred.append(pred)
        all_gt.append(batch.pos_gt_xz.numpy().astype(np.float32))
        all_mask.append(mask)
        all_size.append(size_room)
        all_room_h.append(room_h)

    pred = np.concatenate(all_pred, axis=0)
    gt = np.concatenate(all_gt, axis=0)
    mask = np.concatenate(all_mask, axis=0)
    size_room = np.concatenate(all_size, axis=0)
    room_h = np.concatenate(all_room_h, axis=0)

    metrics = compute_metrics(pred, gt, mask, size_room, room_h)

    print("\n=== BASELINE: random_feasible ===")
    print("\nPlacement stats:")
    for k, v in stats_total.items():
        print(f"  {k:16s}: {v}")
    if stats_total["total_masked"] > 0:
        print(f"  unplaceable_rate : {stats_total['unplaceable']/stats_total['total_masked']:.6f}")
        print(f"  fallback_rate    : {stats_total['fallback_center']/stats_total['total_masked']:.6f}")

    print("\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k:18s}: {v:.6f}")

    maybe_save_metrics(save_metrics, {
        "model": "random_feasible",
        "postprocess": None,
        "seed": seed,
        "metrics": metrics,
        "extra": {"placement_stats": stats_total},
    })


def eval_relaxed_cube(npz: str, splits: str, seed: int, batch_size: int,
                      save_metrics: str | None) -> None:
    ds = FrontCanonicalDataset(npz, splits, split="test")
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_front)
    rng = np.random.default_rng(seed)

    stats_total = {"unplaceable": 0, "placed": 0, "total_masked": 0}

    all_pred, all_gt, all_mask, all_size, all_room_h = [], [], [], [], []

    for batch in dl:
        B, N = batch.pos_gt_xz.shape[0], batch.pos_gt_xz.shape[1]
        pred = np.zeros((B, N, 2), dtype=np.float32)

        room_h = batch.room_h.numpy().astype(np.float32)
        size_room = batch.size_room.numpy().astype(np.float32)
        mask = batch.mask.numpy().astype(np.float32)

        for b in range(B):
            pred_b, info_b = relaxed_cube_layout(room_h[b], size_room[b], mask[b], rng=rng)
            pred[b] = pred_b

            stats_total["unplaceable"] += int(info_b.get("unplaceable", 0))
            stats_total["placed"] += int(info_b.get("placed", 0))
            stats_total["total_masked"] += int(mask[b].sum())

        all_pred.append(pred)
        all_gt.append(batch.pos_gt_xz.numpy().astype(np.float32))
        all_mask.append(mask)
        all_size.append(size_room)
        all_room_h.append(room_h)

    pred = np.concatenate(all_pred, axis=0)
    gt = np.concatenate(all_gt, axis=0)
    mask = np.concatenate(all_mask, axis=0)
    size_room = np.concatenate(all_size, axis=0)
    room_h = np.concatenate(all_room_h, axis=0)

    metrics = compute_metrics(pred, gt, mask, size_room, room_h)

    print("\n=== BASELINE: relaxed_cube ===")
    print("\nPlacement stats:")
    for k, v in stats_total.items():
        print(f"  {k:16s}: {v}")
    if stats_total["total_masked"] > 0:
        print(f"  unplaceable_rate : {stats_total['unplaceable']/stats_total['total_masked']:.6f}")

    print("\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k:18s}: {v:.6f}")

    maybe_save_metrics(save_metrics, {
        "model": "relaxed_cube",
        "postprocess": None,
        "seed": seed,
        "metrics": metrics,
        "extra": {"placement_stats": stats_total},
    })


def eval_tree(npz: str, splits: str, model_path: str, batch_size: int,
              postprocess: str, pp_seed: int, save_metrics: str | None) -> None:
    if joblib is None:
        raise RuntimeError("joblib not installed. Install: pip install joblib")

    model = joblib.load(model_path)  # TreeLayoutModel
    ds = FrontCanonicalDataset(npz, splits, split="test")
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_front)

    from src.ml.baselines.tree_regressor import make_features_for_room, clamp_to_room

    pp_stats = None
    if postprocess == "greedy":
        from src.ml.baselines.placer_greedy import greedy_place
        pp_stats = {"placed": 0, "fallback": 0, "unplaceable": 0, "total_masked": 0}

    rng = np.random.default_rng(pp_seed)

    all_pred, all_gt, all_mask, all_size, all_room_h = [], [], [], [], []

    for batch in dl:
        B, N = batch.pos_gt_xz.shape[0], batch.pos_gt_xz.shape[1]

        gt = batch.pos_gt_xz.numpy().astype(np.float32)
        size_room = batch.size_room.numpy().astype(np.float32)
        cat_id = batch.cat_id.numpy().astype(np.int64)
        mask = batch.mask.numpy().astype(np.float32)
        room_h = batch.room_h.numpy().astype(np.float32)
        room_h_world = batch.room_h_world.numpy().astype(np.float32)

        pred = np.zeros((B, N, 2), dtype=np.float32)

        for b in range(B):
            X, idx = make_features_for_room(room_h_world[b], cat_id[b], size_room[b], mask[b], model.num_cats)
            if idx.size > 0:
                yhat = model.predict(X).astype(np.float32)  # [K,2]
                pred[b, idx] = yhat

            pred[b] = clamp_to_room(pred[b], size_room[b], mask[b])

            if postprocess == "greedy":
                pred_b, info = greedy_place(pred[b], size_room[b], mask[b], rng=rng)
                pred[b] = pred_b
                pp_stats["placed"] += int(info.get("placed", 0))
                pp_stats["fallback"] += int(info.get("fallback_center", 0) or info.get("fallback", 0))
                pp_stats["unplaceable"] += int(info.get("unplaceable", 0))
                pp_stats["total_masked"] += int(mask[b].sum())

        all_pred.append(pred)
        all_gt.append(gt)
        all_mask.append(mask)
        all_size.append(size_room)
        all_room_h.append(room_h)

    pred = np.concatenate(all_pred, axis=0)
    gt = np.concatenate(all_gt, axis=0)
    mask = np.concatenate(all_mask, axis=0)
    size_room = np.concatenate(all_size, axis=0)
    room_h = np.concatenate(all_room_h, axis=0)

    metrics = compute_metrics(pred, gt, mask, size_room, room_h)

    print("\n=== MODEL: tree_regressor ===")
    print(f"Loaded: {model_path}")
    print(f"Postprocess: {postprocess}")

    if postprocess == "greedy":
        print("\nPostprocess stats:")
        for k, v in pp_stats.items():
            print(f"  {k:10s}: {v}")
        if pp_stats["total_masked"] > 0:
            print(f"  fallback_rate : {pp_stats['fallback']/pp_stats['total_masked']:.6f}")
            print(f"  unplaceable_rate : {pp_stats['unplaceable']/pp_stats['total_masked']:.6f}")

    print("\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k:18s}: {v:.6f}")

    maybe_save_metrics(save_metrics, {
        "model": "tree_regressor",
        "postprocess": postprocess,
        "seed": None,
        "metrics": metrics,
        "extra": {"postprocess_stats": pp_stats},
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--batch_size", type=int, default=64)

    ap.add_argument("--mode", choices=["random_feasible", "relaxed_cube", "tree"], default="random_feasible")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--print_unplaceable", type=int, default=0)

    ap.add_argument("--tree_model", type=str, default="out/tree_layout_model.pkl")
    ap.add_argument("--postprocess", choices=["none", "greedy"], default="greedy")
    ap.add_argument("--pp_seed", type=int, default=42)

    ap.add_argument("--save_metrics", type=str, default=None)

    args = ap.parse_args()

    if args.mode == "random_feasible":
        eval_random_feasible(args.npz, args.splits, args.seed, args.batch_size, args.print_unplaceable, args.save_metrics)
    elif args.mode == "relaxed_cube":
        eval_relaxed_cube(args.npz, args.splits, args.seed, args.batch_size, args.save_metrics)
    else:
        eval_tree(args.npz, args.splits, args.tree_model, args.batch_size, args.postprocess, args.pp_seed, args.save_metrics)


if __name__ == "__main__":
    main()