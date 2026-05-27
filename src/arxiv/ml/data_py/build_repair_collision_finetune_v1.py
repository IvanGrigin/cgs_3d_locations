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
    from src.ml.data_py.build_corrupted_object_selector_v1 import is_important_furniture_metadata
    from src.ml.data_py.repair_proposal_dataset_v1 import load_sample_rows
except ModuleNotFoundError:
    from build_corrupted_object_selector_v1 import is_important_furniture_metadata  # type: ignore
    from repair_proposal_dataset_v1 import load_sample_rows  # type: ignore


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build collision-focused fine-tune JSONL for repair_proposal_v1.")
    ap.add_argument("--samples-jsonl", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--important-targets-only", action="store_true")
    ap.add_argument("--main-corruption-types", default="collision_shift")
    ap.add_argument("--regularizer-corruption-types", default="out_of_room_shift,shift_and_yaw")
    ap.add_argument("--main-base-repeat", type=int, default=3)
    ap.add_argument("--balance-target-per-category", type=int, default=1200)
    ap.add_argument("--max-main-repeat", type=int, default=8)
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--limit-val", type=int, default=0)
    ap.add_argument("--limit-test", type=int, default=0)
    return ap.parse_args()


def as_str(v: Any, default: str = "") -> str:
    if v is None:
        return default
    return str(v)


def save_json(path: str | Path, data: Any) -> None:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def target_is_important(row: Dict[str, Any]) -> bool:
    return is_important_furniture_metadata(
        category=as_str(row.get("target_category")),
        super_category=as_str(row.get("target_super_category")),
        name=as_str(row.get("target_category")),
        class_name="",
        mount_type="floor",
        size_m=None,
    )


def main() -> None:
    args = parse_args()
    out_jsonl = Path(args.out_jsonl).expanduser().resolve()
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    limits = {
        "train": int(args.limit_train),
        "val": int(args.limit_val),
        "test": int(args.limit_test),
    }
    main_corruptions = {x.strip() for x in as_str(args.main_corruption_types).split(",") if x.strip()}
    reg_corruptions = {x.strip() for x in as_str(args.regularizer_corruption_types).split(",") if x.strip()}
    keep_corruptions = set(main_corruptions) | set(reg_corruptions)

    rows_by_split: Dict[str, List[Dict[str, Any]]] = {}
    for split in ("train", "val", "test"):
        rows_by_split[split] = load_sample_rows(args.samples_jsonl, split=split, limit=limits[split])

    train_main_cat_counts: Counter[str] = Counter()
    train_source_counts: Counter[str] = Counter()
    skipped_unimportant = 0
    skipped_corruption = 0

    train_main_rows: List[Dict[str, Any]] = []
    train_regularizer_rows: List[Dict[str, Any]] = []
    val_rows: List[Dict[str, Any]] = []
    test_rows: List[Dict[str, Any]] = []

    for split, rows in rows_by_split.items():
        for row in rows:
            corruption = as_str(row.get("corruption_type"))
            if corruption not in keep_corruptions:
                skipped_corruption += 1
                continue
            if args.important_targets_only and not target_is_important(row):
                skipped_unimportant += 1
                continue
            if split == "train":
                if corruption in main_corruptions:
                    train_main_rows.append(row)
                    train_main_cat_counts[as_str(row.get("target_category"))] += 1
                else:
                    train_regularizer_rows.append(row)
                train_source_counts[corruption] += 1
            elif split == "val":
                val_rows.append(row)
            elif split == "test":
                test_rows.append(row)

    target_per_category = max(int(args.balance_target_per_category), 1)
    base_repeat = max(int(args.main_base_repeat), 1)
    max_repeat = max(int(args.max_main_repeat), base_repeat)

    emitted_train = 0
    emitted_val = 0
    emitted_test = 0
    emitted_counts_by_corruption: Counter[str] = Counter()
    emitted_counts_by_category: Counter[str] = Counter()

    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in train_main_rows:
            category = as_str(row.get("target_category"))
            cat_count = max(int(train_main_cat_counts[category]), 1)
            balance_repeat = (target_per_category + cat_count - 1) // cat_count
            repeat = max(base_repeat, balance_repeat)
            repeat = min(repeat, max_repeat)
            for _ in range(int(repeat)):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                emitted_train += 1
                emitted_counts_by_corruption[as_str(row.get("corruption_type"))] += 1
                emitted_counts_by_category[category] += 1

        for row in train_regularizer_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            emitted_train += 1
            emitted_counts_by_corruption[as_str(row.get("corruption_type"))] += 1
            emitted_counts_by_category[as_str(row.get("target_category"))] += 1

        for row in val_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            emitted_val += 1
            emitted_counts_by_corruption[as_str(row.get("corruption_type"))] += 1
            emitted_counts_by_category[as_str(row.get("target_category"))] += 1

        for row in test_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            emitted_test += 1
            emitted_counts_by_corruption[as_str(row.get("corruption_type"))] += 1
            emitted_counts_by_category[as_str(row.get("target_category"))] += 1

    stats = {
        "source_jsonl": str(Path(args.samples_jsonl).expanduser().resolve()),
        "out_jsonl": str(out_jsonl),
        "main_corruption_types": sorted(main_corruptions),
        "regularizer_corruption_types": sorted(reg_corruptions),
        "important_targets_only": bool(args.important_targets_only),
        "main_base_repeat": int(base_repeat),
        "balance_target_per_category": int(target_per_category),
        "max_main_repeat": int(max_repeat),
        "source_train_main_rows": len(train_main_rows),
        "source_train_regularizer_rows": len(train_regularizer_rows),
        "source_val_rows": len(val_rows),
        "source_test_rows": len(test_rows),
        "emitted_train_rows": emitted_train,
        "emitted_val_rows": emitted_val,
        "emitted_test_rows": emitted_test,
        "skipped_unimportant": skipped_unimportant,
        "skipped_corruption": skipped_corruption,
        "source_train_counts_by_corruption": dict(train_source_counts),
        "source_train_main_counts_by_category_top50": dict(train_main_cat_counts.most_common(50)),
        "emitted_counts_by_corruption": dict(emitted_counts_by_corruption),
        "emitted_counts_by_category_top50": dict(emitted_counts_by_category.most_common(50)),
    }
    save_json(out_jsonl.with_suffix(out_jsonl.suffix + ".stats.json"), stats)

    print(f"[repair_collision_finetune_v1] source_train_main_rows={len(train_main_rows)}")
    print(f"[repair_collision_finetune_v1] source_train_regularizer_rows={len(train_regularizer_rows)}")
    print(f"[repair_collision_finetune_v1] emitted_train_rows={emitted_train}")
    print(f"[repair_collision_finetune_v1] emitted_val_rows={emitted_val}")
    print(f"[repair_collision_finetune_v1] emitted_test_rows={emitted_test}")
    print(f"[repair_collision_finetune_v1] skipped_unimportant={skipped_unimportant}")
    print(f"[repair_collision_finetune_v1] skipped_corruption={skipped_corruption}")
    print(f"[repair_collision_finetune_v1] wrote_jsonl={out_jsonl}")
    print(f"[repair_collision_finetune_v1] wrote_stats={out_jsonl.with_suffix(out_jsonl.suffix + '.stats.json')}")


if __name__ == "__main__":
    main()
