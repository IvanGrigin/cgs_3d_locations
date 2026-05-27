from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict, Counter
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.ml.data_py.front_obb_bounds import RoomKey, read_room_obbs


def _sf(row: dict, k: str, default: float = float("nan")) -> float:
    s = row.get(k, "")
    if s is None or s == "":
        return default
    try:
        v = float(s)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _split_by_house(keys: List[RoomKey], seed: int, frac_train: float = 0.78, frac_val: float = 0.10) -> Dict[str, List[int]]:
    # Логика: split по house_id, чтобы комнаты одного дома не утекали между train/test.
    rng = np.random.default_rng(seed)
    house2idx = defaultdict(list)
    for i, k in enumerate(keys):
        house2idx[k.house_id].append(i)
    houses = sorted(house2idx.keys())
    rng.shuffle(houses)

    n = len(houses)
    n_train = int(round(n * frac_train))
    n_val = int(round(n * frac_val))
    train_h = set(houses[:n_train])
    val_h = set(houses[n_train:n_train + n_val])
    test_h = set(houses[n_train + n_val:])

    train_idx, val_idx, test_idx = [], [], []
    for h in houses:
        if h in train_h:
            train_idx.extend(house2idx[h])
        elif h in val_h:
            val_idx.extend(house2idx[h])
        else:
            test_idx.extend(house2idx[h])

    return {"train": sorted(train_idx), "val": sorted(val_idx), "test": sorted(test_idx)}


def _aabb_half_extent_along_axis(axis_world: np.ndarray, half_extents_world: np.ndarray) -> float:
    # Логика: если у нас box axis-aligned в WORLD (AABB), то проекция half-extent на произвольную ось:
    # e_axis = |ax|*ex + |ay|*ey + |az|*ez (support function).
    return float(np.sum(np.abs(axis_world) * half_extents_world))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aabb_csv", required=True)
    ap.add_argument("--obb_csv", required=True)
    ap.add_argument("--nmax", type=int, default=32)
    ap.add_argument("--out_npz", required=True)
    ap.add_argument("--out_meta", required=True)
    ap.add_argument("--out_splits", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--drop_unknown_room", action="store_true")
    args = ap.parse_args()

    aabb_csv = Path(args.aabb_csv)
    obb_csv = Path(args.obb_csv)
    if not aabb_csv.exists():
        raise FileNotFoundError(f"AABB CSV not found: {aabb_csv}")
    if not obb_csv.exists():
        raise FileNotFoundError(f"OBB CSV not found: {obb_csv}")

    room_obbs = read_room_obbs(obb_csv)

    # 1) читаем AABB объекты
    rows_by_room: Dict[RoomKey, List[dict]] = defaultdict(list)
    cats = Counter()

    with aabb_csv.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            k = RoomKey(row["house_id"], row["room_name"], row["scene_glb"])
            rows_by_room[k].append(row)
            cats[(row.get("category") or "").strip()] += 1

    keys_all = sorted(rows_by_room.keys(), key=lambda k: (k.house_id, k.room_name, k.scene_glb))
    usable_keys = [k for k in keys_all if (k in room_obbs) or (not args.drop_unknown_room)]

    # 2) категории -> id (PAD=0)
    cat_list = sorted([c for c in cats.keys() if c != "" and c != "None.obj"])
    cat2id = {c: i + 1 for i, c in enumerate(cat_list)}  # PAD=0

    print(f"Rooms in AABB: {len(keys_all)} | usable: {len(usable_keys)} | room bounds known: {sum(k in room_obbs for k in usable_keys)}")
    print(f"Categories: {len(cat_list)} (+PAD)")

    M = len(usable_keys)
    N = int(args.nmax)

    pos_gt_xz = np.zeros((M, N, 2), dtype=np.float32)
    size_room = np.zeros((M, N, 3), dtype=np.float32)       # НОРМАЛИЗОВАННЫЕ размеры (в координатах комнаты)
    cat_id = np.zeros((M, N), dtype=np.int64)
    mask = np.zeros((M, N), dtype=np.uint8)

    room_h = np.ones((M, 3), dtype=np.float32)              # В КАНОНЕ: [1,1,1]
    room_h_world = np.zeros((M, 3), dtype=np.float32)        # В МЕТРАХ из OBB
    room_c_world = np.zeros((M, 3), dtype=np.float32)        # Центр комнаты в world
    room_axes_world = np.zeros((M, 9), dtype=np.float32)     # R,U,F (world), чтобы можно было вернуть назад

    dropped_cap = 0
    obj_counts = []

    rooms_meta = []

    for i, k in enumerate(usable_keys):
        rr = rows_by_room[k]
        obb = room_obbs.get(k)

        if obb is None:
            if args.drop_unknown_room:
                continue
            # fallback: считаем room_h_world по AABB объектов (грубо)
            xs, ys, zs = [], [], []
            for row in rr:
                px, py, pz = _sf(row, "pos_x"), _sf(row, "pos_y"), _sf(row, "pos_z")
                sx, sy, sz = _sf(row, "size_x"), _sf(row, "size_y"), _sf(row, "size_z")
                if not all(math.isfinite(v) for v in [px, py, pz, sx, sy, sz]):
                    continue
                xs.extend([px - sx * 0.5, px + sx * 0.5])
                ys.extend([py - sy * 0.5, py + sy * 0.5])
                zs.extend([pz - sz * 0.5, pz + sz * 0.5])
            if len(xs) < 2:
                continue
            cx, cy, cz = (min(xs) + max(xs)) * 0.5, (min(ys) + max(ys)) * 0.5, (min(zs) + max(zs)) * 0.5
            hx, hy, hz = (max(xs) - min(xs)) * 0.5, (max(ys) - min(ys)) * 0.5, (max(zs) - min(zs)) * 0.5
            R = (1.0, 0.0, 0.0)
            U = (0.0, 1.0, 0.0)
            F = (0.0, 0.0, 1.0)
        else:
            (cx, cy, cz) = obb.c
            (hx, hy, hz) = obb.h
            R, U, F = obb.R, obb.U, obb.F

        if hx <= 1e-8 or hz <= 1e-8:
            continue

        room_h_world[i] = (hx, hy, hz)
        room_c_world[i] = (cx, cy, cz)
        room_axes_world[i] = np.array([*R, *U, *F], dtype=np.float32)

        # Матрица B = [R U F] (world). Для ортонормального базиса локальные координаты: p_local = B^T (p - C)
        B = np.array([[R[0], U[0], F[0]],
                      [R[1], U[1], F[1]],
                      [R[2], U[2], F[2]]], dtype=np.float32)
        Bt = B.T

        # Сбор объектов комнаты (ограничиваем Nmax)
        objs = []
        for row in rr:
            cat = (row.get("category") or "").strip()
            if cat == "" or cat == "None.obj":
                continue

            px, py, pz = _sf(row, "pos_x"), _sf(row, "pos_y"), _sf(row, "pos_z")
            sx, sy, sz = _sf(row, "size_x"), _sf(row, "size_y"), _sf(row, "size_z")
            if not all(math.isfinite(v) for v in [px, py, pz, sx, sy, sz]):
                continue

            objs.append((row, cat, (px, py, pz), (sx, sy, sz)))

        if len(objs) > N:
            # Логика: пока без умного отбора — просто отрезаем хвост (чтобы быстрее стартануть).
            objs = objs[:N]
            dropped_cap += 1

        obj_counts.append(len(objs))

        for j, (row, cat, p, s) in enumerate(objs):
            # позиция в локал
            p_world = np.array([p[0] - cx, p[1] - cy, p[2] - cz], dtype=np.float32)
            p_local = Bt @ p_world  # (x_local, y_local, z_local)

            # нормализуем X и Z в [-1,1] по room_h_world
            x = float(p_local[0] / hx)
            z = float(p_local[2] / hz)
            pos_gt_xz[i, j, 0] = np.clip(x, -1.25, 1.25)  # оставим небольшую “дышку” для грязных данных
            pos_gt_xz[i, j, 1] = np.clip(z, -1.25, 1.25)

            # нормализуем размер: переводим WORLD AABB -> half extent вдоль осей комнаты (R,U,F)
            half_world = np.array([0.5 * s[0], 0.5 * s[1], 0.5 * s[2]], dtype=np.float32)
            Rv = np.array(R, dtype=np.float32)
            Uv = np.array(U, dtype=np.float32)
            Fv = np.array(F, dtype=np.float32)

            hx_loc = _aabb_half_extent_along_axis(Rv, half_world)
            hy_loc = _aabb_half_extent_along_axis(Uv, half_world)
            hz_loc = _aabb_half_extent_along_axis(Fv, half_world)

            sx_norm = (2.0 * hx_loc) / hx
            sy_norm = (2.0 * hy_loc) / max(hy, 1e-6)
            sz_norm = (2.0 * hz_loc) / hz

            size_room[i, j] = (sx_norm, sy_norm, sz_norm)

            cat_id[i, j] = cat2id.get(cat, 0)
            mask[i, j] = 1

        rooms_meta.append({"house_id": k.house_id, "room_name": k.room_name, "scene_glb": k.scene_glb})

    obj_counts = np.array(obj_counts, dtype=np.int32)
    if len(obj_counts) == 0:
        raise RuntimeError("No usable rooms after preprocessing.")

    print(f"Dropped rooms due to Nmax cap: {dropped_cap}")
    print(f"Objects per room (usable): mean={obj_counts.mean():.2f}, p95={np.quantile(obj_counts, 0.95):.0f}")

    # 3) splits
    splits = _split_by_house(usable_keys, seed=args.seed)
    Path(args.out_splits).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_npz).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_meta).parent.mkdir(parents=True, exist_ok=True)

    with open(args.out_splits, "w", encoding="utf-8") as f:
        json.dump(splits, f, ensure_ascii=False, indent=2)

    # 4) save npz
    np.savez_compressed(
        args.out_npz,
        pos_gt_xz=pos_gt_xz,
        size_room=size_room,
        cat_id=cat_id,
        mask=mask,
        room_h=room_h,
        room_h_world=room_h_world,
        room_c_world=room_c_world,
        room_axes_world=room_axes_world,
    )

    meta = {
        "nmax": N,
        "categories": ["<PAD>"] + cat_list,
        "cat2id": cat2id,
        "rooms": rooms_meta,
        "note": "pos_gt_xz and size_room are in room-canonical normalized coords (X,Z ~ [-1,1]). room_h is [1,1,1]. room_h_world is in meters from OBB.",
    }
    with open(args.out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Splits saved: {args.out_splits} | train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
    print(f"Saved dataset: {args.out_npz}")
    print(f"Saved meta: {args.out_meta}")


if __name__ == "__main__":
    main()
