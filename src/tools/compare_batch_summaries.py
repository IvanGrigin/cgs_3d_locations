#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/tools/compare_batch_summaries.py

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def load_summary_csv(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise RuntimeError(f"Не найден summary csv: {path}")

    rows: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric = row["metric"]
            rows[metric] = row
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Compare appraisal_summary.csv for two batches")
    p.add_argument("--random-batch", required=True)
    p.add_argument("--relaxed-batch", required=True)
    p.add_argument("--out", default="out/batch_compare_random_vs_relaxed.csv")
    args = p.parse_args()

    random_batch = Path(args.random_batch).expanduser().resolve()
    relaxed_batch = Path(args.relaxed_batch).expanduser().resolve()
    out_csv = Path(args.out).expanduser().resolve()

    random_summary = load_summary_csv(random_batch / "appraisal_summary.csv")
    relaxed_summary = load_summary_csv(relaxed_batch / "appraisal_summary.csv")

    metrics_order = [
        "generation_success_rate_pct",
        "generation_avg_duration_ok_sec",
        "generation_median_duration_ok_sec",
        "generation_min_duration_ok_sec",
        "generation_max_duration_ok_sec",

        "score_10",
        "geometry_score_10",
        "prompt_match_score_10",
        "constraint_score_10",

        "room_area_m2",
        "raw_sum_item_area_m2",
        "free_area_m2",
        "occupied_union_area_m2",
        "free_union_area_m2",
        "largest_free_rectangle_area_m2",
        "largest_free_rectangle_ratio",
        "floor_coverage_ratio",

        "overlap_area_m2",
        "overlap_ratio",
        "outside_room_area_m2",
        "outside_room_ratio",

        "accessible_objects",
        "accessible_objects_total",
        "accessibility_ratio",

        "no_overlap",
        "no_outside",
        "full_access",
        "strict_valid",
    ]

    label_map = {
        "generation_success_rate_pct": "Generation success rate, %",
        "generation_avg_duration_ok_sec": "Generation avg time, s",
        "generation_median_duration_ok_sec": "Generation median time, s",
        "generation_min_duration_ok_sec": "Generation min time, s",
        "generation_max_duration_ok_sec": "Generation max time, s",

        "score_10": "Final score",
        "geometry_score_10": "Geometry score",
        "prompt_match_score_10": "Prompt match score",
        "constraint_score_10": "Constraint score",

        "room_area_m2": "Room area, m²",
        "raw_sum_item_area_m2": "Raw sum item area, m²",
        "free_area_m2": "Free area, m²",
        "occupied_union_area_m2": "Occupied union area, m²",
        "free_union_area_m2": "Free union area, m²",
        "largest_free_rectangle_area_m2": "Largest free rectangle, m²",
        "largest_free_rectangle_ratio": "Largest free rectangle ratio",
        "floor_coverage_ratio": "Floor coverage ratio",

        "overlap_area_m2": "Overlap area, m²",
        "overlap_ratio": "Overlap ratio",
        "outside_room_area_m2": "Outside room area, m²",
        "outside_room_ratio": "Outside room ratio",

        "accessible_objects": "Accessible objects",
        "accessible_objects_total": "Accessible objects total",
        "accessibility_ratio": "Accessibility ratio",

        "no_overlap": "No overlap share",
        "no_outside": "No outside share",
        "full_access": "Full access share",
        "strict_valid": "Strict valid share",
    }

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric_key", "metric_label", "random_mean", "relaxed_mean", "delta_relaxed_minus_random"])

        for metric in metrics_order:
            r = random_summary.get(metric)
            q = relaxed_summary.get(metric)
            if not r and not q:
                continue

            r_mean = r["mean"] if r else ""
            q_mean = q["mean"] if q else ""

            delta = ""
            try:
                delta = f"{float(q_mean) - float(r_mean):.6f}"
            except Exception:
                delta = ""

            writer.writerow([
                metric,
                label_map.get(metric, metric),
                r_mean,
                q_mean,
                delta,
            ])

    print(f"saved: {out_csv}")


if __name__ == "__main__":
    main()