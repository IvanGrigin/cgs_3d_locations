#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def split_by_house(house_ids: List[str], seed: int, frac_train: float = 0.78, frac_val: float = 0.10) -> Dict[str, List[int]]:
    rng = np.random.default_rng(seed)
    house2idx: Dict[str, List[int]] = defaultdict(list)
    for i, h in enumerate(house_ids):
        house2idx[h].append(i)
    houses = sorted(house2idx.keys())
    rng.shuffle(houses)

    n = len(houses)
    n_train = int(round(n * frac_train))
    n_val = int(round(n * frac_val))
    train_h = set(houses[:n_train])
    val_h = set(houses[n_train:n_train + n_val])

    train_idx, val_idx, test_idx = [], [], []
    for h in houses:
        if h in train_h:
            train_idx.extend(house2idx[h])
        elif h in val_h:
            val_idx.extend(house2idx[h])
        else:
            test_idx.extend(house2idx[h])
    return {"train": sorted(train_idx), "val": sorted(val_idx), "test": sorted(test_idx)}


def parse_room_entry(file_path: Path, keep_lighting: bool) -> Tuple[str, str, List[Tuple[float, float]], List[Dict[str, Any]]]:
    data = load_json(file_path)
    rooms = data.get("rooms") or []
    if not isinstance(rooms, list) or not rooms:
        raise ValueError("mini.json does not contain rooms")

    room = rooms[0]
    poly_raw = room.get("floor_polygon_xz") or []
    poly: List[Tuple[float, float]] = []
    for p in poly_raw:
        if isinstance(p, dict):
            poly.append((safe_float(p.get("x")), safe_float(p.get("z"))))
    if len(poly) < 3:
        raise ValueError("room polygon is invalid")

    objects = []
    for obj in room.get("objects") or []:
        if not isinstance(obj, dict):
            continue
        if (not keep_lighting) and bool(obj.get("is_lighting", False)):
            continue
        bb = obj.get("bbox_world_xy")
        pos = obj.get("pos")
        if not (isinstance(bb, list) and len(bb) >= 4 and isinstance(pos, dict)):
            continue
        xmin, xmax, zmin, zmax = [safe_float(v) for v in bb[:4]]
        if xmax <= xmin or zmax <= zmin:
            continue
        objects.append(obj)

    stem = file_path.stem
    house_id, _, room_name = stem.partition("__")
    return house_id or stem, room_name or stem, poly, objects


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert 3D-FRONT mini.json files to canonical npz/splits")
    ap.add_argument("--inputs", required=True, help="Glob for *.mini.json")
    ap.add_argument("--out-npz", required=True)
    ap.add_argument("--out-splits", required=True)
    ap.add_argument("--out-meta", required=True)
    ap.add_argument("--nmax", type=int, default=32)
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep-lighting", action="store_true")
    args = ap.parse_args()

    files = sorted(glob.glob(args.inputs, recursive=True))
    files = [f for f in files if f.endswith(".mini.json")]
    if args.max_files and args.max_files > 0:
        files = files[: args.max_files]
    if not files:
        raise SystemExit(f"No mini files found by glob: {args.inputs}")

    parsed = []
    categories = set()
    skipped = 0
    for fp in files:
        try:
            house_id, room_name, poly, objects = parse_room_entry(Path(fp), keep_lighting=bool(args.keep_lighting))
        except Exception:
            skipped += 1
            continue
        if not objects:
            skipped += 1
            continue
        parsed.append((fp, house_id, room_name, poly, objects))
        for obj in objects:
            categories.add(str(obj.get("category") or obj.get("label") or obj.get("name") or "UNK"))

    if not parsed:
        raise SystemExit("No usable rooms after parsing")

    cat_list = sorted(categories)
    cat2id = {c: i + 1 for i, c in enumerate(cat_list)}

    M = len(parsed)
    N = int(args.nmax)
    pos_gt_xz = np.zeros((M, N, 2), dtype=np.float32)
    size_room = np.zeros((M, N, 3), dtype=np.float32)
    cat_id = np.zeros((M, N), dtype=np.int64)
    mask = np.zeros((M, N), dtype=np.uint8)
    room_h = np.ones((M, 3), dtype=np.float32)
    room_h_world = np.zeros((M, 3), dtype=np.float32)
    room_c_world = np.zeros((M, 3), dtype=np.float32)
    room_axes_world = np.tile(np.array([[1, 0, 0, 0, 1, 0, 0, 0, 1]], dtype=np.float32), (M, 1))

    meta_rooms = []
    house_ids = []
    clipped = 0

    for i, (fp, house_id, room_name, poly, objects) in enumerate(parsed):
        xs = [p[0] for p in poly]
        zs = [p[1] for p in poly]
        x0, x1 = min(xs), max(xs)
        z0, z1 = min(zs), max(zs)
        cx = 0.5 * (x0 + x1)
        cz = 0.5 * (z0 + z1)
        hx = max(0.5 * (x1 - x0), 1e-6)
        hz = max(0.5 * (z1 - z0), 1e-6)

        room_h_world[i] = [hx, 1.0, hz]
        room_c_world[i] = [cx, 0.0, cz]

        if len(objects) > N:
            objects = objects[:N]
            clipped += 1

        for j, obj in enumerate(objects):
            pos = obj["pos"]
            bb = obj["bbox_world_xy"]
            px = safe_float(pos.get("x"))
            pz = safe_float(pos.get("z"))
            sx = max(0.0, safe_float(bb[1]) - safe_float(bb[0]))
            sz = max(0.0, safe_float(bb[3]) - safe_float(bb[2]))

            pos_gt_xz[i, j, 0] = np.clip((px - cx) / hx, -1.25, 1.25)
            pos_gt_xz[i, j, 1] = np.clip((pz - cz) / hz, -1.25, 1.25)
            size_room[i, j] = [sx / hx, 1.0, sz / hz]
            cat = str(obj.get("category") or obj.get("label") or obj.get("name") or "UNK")
            cat_id[i, j] = int(cat2id.get(cat, 0))
            mask[i, j] = 1

        house_ids.append(house_id)
        meta_rooms.append(
            {
                "source_file": str(Path(fp).expanduser().resolve()),
                "house_id": house_id,
                "room_name": room_name,
                "room_bbox": [x0, x1, z0, z1],
                "object_count": int(len(objects)),
            }
        )

    splits = split_by_house(house_ids, seed=int(args.seed))

    out_npz = Path(args.out_npz).expanduser().resolve()
    out_splits = Path(args.out_splits).expanduser().resolve()
    out_meta = Path(args.out_meta).expanduser().resolve()
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    out_splits.parent.mkdir(parents=True, exist_ok=True)
    out_meta.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_npz,
        pos_gt_xz=pos_gt_xz,
        size_room=size_room,
        cat_id=cat_id,
        mask=mask,
        room_h=room_h,
        room_h_world=room_h_world,
        room_c_world=room_c_world,
        room_axes_world=room_axes_world,
    )
    out_splits.write_text(json.dumps(splits, ensure_ascii=False, indent=2), encoding="utf-8")
    out_meta.write_text(
        json.dumps(
            {
                "nmax": int(N),
                "categories": ["<PAD>"] + cat_list,
                "cat2id": cat2id,
                "rooms": meta_rooms,
                "source_glob": args.inputs,
                "parsed_rooms": int(M),
                "skipped_rooms": int(skipped),
                "clipped_rooms": int(clipped),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"[preprocess_mini_to_canonical] rooms={M} skipped={skipped} clipped={clipped} "
        f"train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}"
    )
    print(f"[preprocess_mini_to_canonical] npz={out_npz}")
    print(f"[preprocess_mini_to_canonical] splits={out_splits}")
    print(f"[preprocess_mini_to_canonical] meta={out_meta}")


if __name__ == "__main__":
    main()
