from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import joblib
except Exception as e:
    raise ImportError("joblib is required. Install: pip install joblib") from e

from src.ml.data_py.dataset_front import FrontCanonicalDataset
from src.ml.baselines.tree_regressor import (
    TreeLayoutModel,
    make_features_for_room,
    fit_tree_model,
    clamp_to_room,
)
from src.ml.metrics.layout_metrics import rmse_xz, mae_xz, boundary_violation_rate, collision_pair_rate


def _infer_num_cats(npz_path: str) -> int:
    z = np.load(npz_path)
    cat = z["cat_id"].astype(np.int64)
    return int(cat.max()) + 1  # includes PAD=0


def build_xy_from_dataset(ds: FrontCanonicalDataset, num_cats: int):
    X_all = []
    Y_all = []

    for i in range(len(ds)):
        item = ds[i]
        pos = item["pos_gt_xz"]         # [N,2]
        size = item["size_room"]        # [N,3]
        cat = item["cat_id"]            # [N]
        mask = item["mask"]             # [N]
        room_h_w = item["room_h_world"] # [3]

        X, idx = make_features_for_room(room_h_w, cat, size, mask, num_cats)
        if idx.size == 0:
            continue
        Y = pos[idx].astype(np.float32)

        X_all.append(X)
        Y_all.append(Y)

    X_all = np.concatenate(X_all, axis=0).astype(np.float32)
    Y_all = np.concatenate(Y_all, axis=0).astype(np.float32)
    return X_all, Y_all


def eval_on_split(npz: str, splits: str, split: str, model: TreeLayoutModel):
    ds = FrontCanonicalDataset(npz, splits, split=split)

    preds = []
    gts = []
    masks = []
    sizes = []
    room_h = []  # canonical ones (для метрик можно не нужно, но пусть)
    for i in range(len(ds)):
        it = ds[i]
        pos_gt = it["pos_gt_xz"].astype(np.float32)
        size = it["size_room"].astype(np.float32)
        cat = it["cat_id"].astype(np.int64)
        mask = it["mask"].astype(np.float32)
        room_h_w = it["room_h_world"].astype(np.float32)

        X, idx = make_features_for_room(room_h_w, cat, size, mask, model.num_cats)
        pred = np.zeros_like(pos_gt, dtype=np.float32)
        if idx.size > 0:
            yhat = model.predict(X)  # [K,2]
            pred[idx] = yhat

        pred = clamp_to_room(pred, size, mask)

        preds.append(pred)
        gts.append(pos_gt)
        masks.append(mask)
        sizes.append(size)
        room_h.append(np.ones((3,), dtype=np.float32))

    pred = np.stack(preds, axis=0)
    gt = np.stack(gts, axis=0)
    mask = np.stack(masks, axis=0)
    size = np.stack(sizes, axis=0)
    room_h = np.stack(room_h, axis=0)

    return {
        "RMSE_xz": rmse_xz(pred, gt, mask),
        "MAE_xz": mae_xz(pred, gt, mask),
        "BoundaryViolRate": boundary_violation_rate(pred, size, room_h, mask),
        "CollisionPairRate": collision_pair_rate(pred, size, room_h, mask),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--save", default="out/tree_layout_model.pkl")
    ap.add_argument("--num_trees", type=int, default=400)
    ap.add_argument("--max_depth", type=int, default=0, help="0 means None")
    ap.add_argument("--min_leaf", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    num_cats = _infer_num_cats(args.npz)
    print(f"num_cats (including PAD=0): {num_cats}")

    ds_tr = FrontCanonicalDataset(args.npz, args.splits, split="train")
    ds_va = FrontCanonicalDataset(args.npz, args.splits, split="val")

    Xtr, Ytr = build_xy_from_dataset(ds_tr, num_cats)
    Xva, Yva = build_xy_from_dataset(ds_va, num_cats)

    print(f"Train objects: {Xtr.shape[0]} | Val objects: {Xva.shape[0]}")
    print(f"Feature dim: {Xtr.shape[1]}")

    model = fit_tree_model(
        Xtr,
        Ytr,
        num_trees=args.num_trees,
        max_depth=None if args.max_depth == 0 else args.max_depth,
        min_samples_leaf=args.min_leaf,
        seed=args.seed,
    )
    wrapper = TreeLayoutModel(num_cats=num_cats, model=model)

    m_val = eval_on_split(args.npz, args.splits, "val", wrapper)
    print("\n=== TREE (val) ===")
    for k, v in m_val.items():
        print(f"  {k:18s}: {v:.6f}")

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(wrapper, args.save)
    print(f"\nSaved: {args.save}")


if __name__ == "__main__":
    main()
