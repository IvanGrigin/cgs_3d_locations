#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/tools/batch_appraise_and_export_metrics.py

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_APPRAISER = "src/Appraiser/appraiser.py"


PER_RUN_FIELDS = [
    "batch_dir",
    "run_dir",
    "room",
    "scenario",
    "run",
    "prompt_file",
    "scene_file",
    "appraisal_json",
    "generation_status",
    "generation_duration_sec",
    "appraisal_status",
    "appraisal_duration_sec",

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

    "error",
]


SUMMARY_FIELDS = [
    "metric",
    "count",
    "mean",
    "median",
    "min",
    "max",
]


def ensure_csv(path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def append_csv_row(path: Path, fieldnames: list[str], row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def safe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def metric_stats(values: list[float | None]) -> dict[str, Any] | None:
    xs = [float(x) for x in values if x is not None]
    if not xs:
        return None
    return {
        "count": len(xs),
        "mean": sum(xs) / len(xs),
        "median": st.median(xs),
        "min": min(xs),
        "max": max(xs),
    }


def load_generation_rows(batch_dir: Path) -> dict[str, dict[str, Any]]:
    """
    Возвращает словарь:
        key = absolute batch_run_dir
        value = row из batch_results.csv
    """
    csv_path = batch_dir / "batch_results.csv"
    res: dict[str, dict[str, Any]] = {}
    if not csv_path.is_file():
        return res

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = str(Path(row["batch_run_dir"]).expanduser().resolve())
            res[key] = row
    return res


def generation_summary(batch_dir: Path) -> dict[str, Any]:
    csv_path = batch_dir / "batch_results.csv"
    if not csv_path.is_file():
        return {
            "jobs_total": 0,
            "ok_count": 0,
            "failed_count": 0,
            "timeout_count": 0,
            "error_count": 0,
            "success_rate_pct": None,
            "avg_duration_ok_sec": None,
            "median_duration_ok_sec": None,
            "min_duration_ok_sec": None,
            "max_duration_ok_sec": None,
        }

    jobs_total = 0
    ok_count = 0
    failed_count = 0
    timeout_count = 0
    error_count = 0
    durations_ok: list[float] = []

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            jobs_total += 1
            status = row.get("status", "")
            if status == "ok":
                ok_count += 1
                try:
                    durations_ok.append(float(row["duration_sec"]))
                except Exception:
                    pass
            elif status == "failed":
                failed_count += 1
            elif status == "timeout":
                timeout_count += 1
            else:
                error_count += 1

    if jobs_total > 0:
        success_rate_pct = 100.0 * ok_count / jobs_total
    else:
        success_rate_pct = None

    if durations_ok:
        avg_duration_ok_sec = sum(durations_ok) / len(durations_ok)
        median_duration_ok_sec = st.median(durations_ok)
        min_duration_ok_sec = min(durations_ok)
        max_duration_ok_sec = max(durations_ok)
    else:
        avg_duration_ok_sec = None
        median_duration_ok_sec = None
        min_duration_ok_sec = None
        max_duration_ok_sec = None

    return {
        "jobs_total": jobs_total,
        "ok_count": ok_count,
        "failed_count": failed_count,
        "timeout_count": timeout_count,
        "error_count": error_count,
        "success_rate_pct": success_rate_pct,
        "avg_duration_ok_sec": avg_duration_ok_sec,
        "median_duration_ok_sec": median_duration_ok_sec,
        "min_duration_ok_sec": min_duration_ok_sec,
        "max_duration_ok_sec": max_duration_ok_sec,
    }


def run_appraiser(appraiser_script: Path, scene: Path, prompt_file: Path, out_json: Path) -> tuple[str, float, str]:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(appraiser_script.resolve()),
        "--scene", str(scene.resolve()),
        "--prompt-file", str(prompt_file.resolve()),
        "--mode", "code",
        "--out", str(out_json.resolve()),
    ]

    try:
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True, check=False)
        dt = round(time.perf_counter() - t0, 3)

        if proc.returncode != 0:
            return "failed", dt, f"appraiser returncode={proc.returncode}"
        if not out_json.is_file():
            return "failed", dt, "appraiser did not create appraisal.json"
        return "ok", dt, ""
    except Exception as e:
        return "error", -1.0, repr(e)


def extract_appraisal_metrics(data: dict[str, Any]) -> dict[str, Any]:
    cr = data.get("code_result") or {}
    sm = (data.get("summary") or {}).get("code") or {}

    row = {
        "score_10": safe_float(data.get("score_10", sm.get("score_10"))),
        "geometry_score_10": safe_float(cr.get("geometry_score_10", sm.get("geometry_score_10"))),
        "prompt_match_score_10": safe_float(cr.get("prompt_match_score_10", sm.get("prompt_match_score_10"))),
        "constraint_score_10": safe_float(cr.get("constraint_score_10", sm.get("constraint_score_10"))),

        "room_area_m2": safe_float(cr.get("room_area_m2", sm.get("room_area_m2"))),
        "raw_sum_item_area_m2": safe_float(cr.get("raw_sum_item_area_m2")),
        "free_area_m2": safe_float(cr.get("free_area_m2", sm.get("free_area_m2"))),
        "occupied_union_area_m2": safe_float(cr.get("occupied_union_area_m2")),
        "free_union_area_m2": safe_float(cr.get("free_union_area_m2")),
        "largest_free_rectangle_area_m2": safe_float(cr.get("largest_free_rectangle_area_m2", sm.get("largest_free_rectangle_area_m2"))),
        "largest_free_rectangle_ratio": safe_float(cr.get("largest_free_rectangle_ratio", sm.get("largest_free_rectangle_ratio"))),
        "floor_coverage_ratio": safe_float(cr.get("floor_coverage_ratio")),

        "overlap_area_m2": safe_float(cr.get("overlap_area_m2")),
        "overlap_ratio": safe_float(cr.get("overlap_ratio", sm.get("overlap_ratio"))),
        "outside_room_area_m2": safe_float(cr.get("outside_room_area_m2")),
        "outside_room_ratio": safe_float(cr.get("outside_room_ratio", sm.get("outside_room_ratio"))),

        "accessible_objects": safe_float(cr.get("accessible_objects", sm.get("accessible_objects"))),
        "accessible_objects_total": safe_float(cr.get("accessible_objects_total", sm.get("accessible_objects_total"))),
        "accessibility_ratio": safe_float(cr.get("accessibility_ratio", sm.get("accessibility_ratio"))),
    }

    eps = 1e-9
    overlap = row["overlap_area_m2"]
    outside = row["outside_room_area_m2"]
    access = row["accessibility_ratio"]

    row["no_overlap"] = 1 if (overlap is not None and overlap <= eps) else 0
    row["no_outside"] = 1 if (outside is not None and outside <= eps) else 0
    row["full_access"] = 1 if (access is not None and abs(access - 1.0) <= eps) else 0
    row["strict_valid"] = 1 if (row["no_overlap"] and row["no_outside"] and row["full_access"]) else 0

    return row


def batch_room_scenario_run(batch_dir: Path, run_dir: Path) -> tuple[str, str, str]:
    rel = run_dir.resolve().relative_to(batch_dir.resolve())
    parts = list(rel.parts)
    room = parts[0] if len(parts) >= 1 else ""
    scenario = parts[1] if len(parts) >= 2 else ""
    run = parts[2] if len(parts) >= 3 else run_dir.name
    return room, scenario, run


def write_summary_csv(batch_dir: Path, per_run_rows: list[dict[str, Any]]) -> None:
    out_csv = batch_dir / "appraisal_summary.csv"
    ensure_csv(out_csv, SUMMARY_FIELDS)

    metrics_order = [
        ("generation_success_rate_pct", [r.get("generation_success_rate_pct") for r in per_run_rows[:1]]),
        ("generation_avg_duration_ok_sec", [r.get("generation_avg_duration_ok_sec") for r in per_run_rows[:1]]),
        ("generation_median_duration_ok_sec", [r.get("generation_median_duration_ok_sec") for r in per_run_rows[:1]]),
        ("generation_min_duration_ok_sec", [r.get("generation_min_duration_ok_sec") for r in per_run_rows[:1]]),
        ("generation_max_duration_ok_sec", [r.get("generation_max_duration_ok_sec") for r in per_run_rows[:1]]),

        ("appraisal_duration_sec", [safe_float(r.get("appraisal_duration_sec")) for r in per_run_rows]),

        ("score_10", [safe_float(r.get("score_10")) for r in per_run_rows]),
        ("geometry_score_10", [safe_float(r.get("geometry_score_10")) for r in per_run_rows]),
        ("prompt_match_score_10", [safe_float(r.get("prompt_match_score_10")) for r in per_run_rows]),
        ("constraint_score_10", [safe_float(r.get("constraint_score_10")) for r in per_run_rows]),

        ("room_area_m2", [safe_float(r.get("room_area_m2")) for r in per_run_rows]),
        ("raw_sum_item_area_m2", [safe_float(r.get("raw_sum_item_area_m2")) for r in per_run_rows]),
        ("free_area_m2", [safe_float(r.get("free_area_m2")) for r in per_run_rows]),
        ("occupied_union_area_m2", [safe_float(r.get("occupied_union_area_m2")) for r in per_run_rows]),
        ("free_union_area_m2", [safe_float(r.get("free_union_area_m2")) for r in per_run_rows]),
        ("largest_free_rectangle_area_m2", [safe_float(r.get("largest_free_rectangle_area_m2")) for r in per_run_rows]),
        ("largest_free_rectangle_ratio", [safe_float(r.get("largest_free_rectangle_ratio")) for r in per_run_rows]),
        ("floor_coverage_ratio", [safe_float(r.get("floor_coverage_ratio")) for r in per_run_rows]),
        ("overlap_area_m2", [safe_float(r.get("overlap_area_m2")) for r in per_run_rows]),
        ("overlap_ratio", [safe_float(r.get("overlap_ratio")) for r in per_run_rows]),
        ("outside_room_area_m2", [safe_float(r.get("outside_room_area_m2")) for r in per_run_rows]),
        ("outside_room_ratio", [safe_float(r.get("outside_room_ratio")) for r in per_run_rows]),
        ("accessible_objects", [safe_float(r.get("accessible_objects")) for r in per_run_rows]),
        ("accessible_objects_total", [safe_float(r.get("accessible_objects_total")) for r in per_run_rows]),
        ("accessibility_ratio", [safe_float(r.get("accessibility_ratio")) for r in per_run_rows]),

        ("no_overlap", [safe_float(r.get("no_overlap")) for r in per_run_rows]),
        ("no_outside", [safe_float(r.get("no_outside")) for r in per_run_rows]),
        ("full_access", [safe_float(r.get("full_access")) for r in per_run_rows]),
        ("strict_valid", [safe_float(r.get("strict_valid")) for r in per_run_rows]),
    ]

    for metric_name, values in metrics_order:
        s = metric_stats(values)
        if s is None:
            continue
        append_csv_row(
            out_csv,
            SUMMARY_FIELDS,
            {
                "metric": metric_name,
                "count": s["count"],
                "mean": f"{s['mean']:.6f}",
                "median": f"{s['median']:.6f}",
                "min": f"{s['min']:.6f}",
                "max": f"{s['max']:.6f}",
            },
        )


def appraise_batch(batch_dir: Path, appraiser_script: Path, rerun_existing: bool) -> None:
    batch_dir = batch_dir.expanduser().resolve()
    if not batch_dir.is_dir():
        raise RuntimeError(f"Не найден batch dir: {batch_dir}")

    gen_rows = load_generation_rows(batch_dir)
    gen_summary = generation_summary(batch_dir)

    per_run_csv = batch_dir / "appraisal_per_run.csv"
    jsonl_path = batch_dir / "appraisal_results.jsonl"
    ensure_csv(per_run_csv, PER_RUN_FIELDS)
    if jsonl_path.exists():
        jsonl_path.unlink()

    runs = sorted(p.parent for p in batch_dir.rglob("scene.v1.json"))

    per_run_rows: list[dict[str, Any]] = []
    jobs_total = 0
    appraisal_ok = 0
    appraisal_failed = 0
    appraisal_error = 0

    print(f"=== APPRAISE BATCH ===")
    print(f"batch: {batch_dir}")
    print(f"scene count: {len(runs)}")

    for run_dir in runs:
        jobs_total += 1
        scene = run_dir / "scene.v1.json"
        prompt = run_dir / "prompt.txt"
        out_json = run_dir / "appraisal.json"

        room, scenario, run = batch_room_scenario_run(batch_dir, run_dir)

        if not prompt.is_file():
            row = {
                "batch_dir": str(batch_dir),
                "run_dir": str(run_dir),
                "room": room,
                "scenario": scenario,
                "run": run,
                "prompt_file": str(prompt),
                "scene_file": str(scene),
                "appraisal_json": str(out_json),
                "generation_status": gen_rows.get(str(run_dir.resolve()), {}).get("status", ""),
                "generation_duration_sec": gen_rows.get(str(run_dir.resolve()), {}).get("duration_sec", ""),
                "appraisal_status": "failed",
                "appraisal_duration_sec": "",
                "score_10": "",
                "geometry_score_10": "",
                "prompt_match_score_10": "",
                "constraint_score_10": "",
                "room_area_m2": "",
                "raw_sum_item_area_m2": "",
                "free_area_m2": "",
                "occupied_union_area_m2": "",
                "free_union_area_m2": "",
                "largest_free_rectangle_area_m2": "",
                "largest_free_rectangle_ratio": "",
                "floor_coverage_ratio": "",
                "overlap_area_m2": "",
                "overlap_ratio": "",
                "outside_room_area_m2": "",
                "outside_room_ratio": "",
                "accessible_objects": "",
                "accessible_objects_total": "",
                "accessibility_ratio": "",
                "no_overlap": "",
                "no_outside": "",
                "full_access": "",
                "strict_valid": "",
                "error": "нет prompt.txt",
            }
            appraisal_failed += 1
            append_csv_row(per_run_csv, PER_RUN_FIELDS, row)
            append_jsonl(jsonl_path, row)
            per_run_rows.append(row)
            continue

        if out_json.is_file() and not rerun_existing:
            status = "ok"
            dt = 0.0
            err = ""
        else:
            status, dt, err = run_appraiser(appraiser_script, scene, prompt, out_json)

        if status == "ok":
            appraisal_ok += 1
            try:
                data = json.loads(out_json.read_text(encoding="utf-8"))
                metrics = extract_appraisal_metrics(data)
            except Exception as e:
                status = "error"
                appraisal_ok -= 1
                appraisal_error += 1
                metrics = {k: "" for k in PER_RUN_FIELDS}
                err = f"bad appraisal.json: {e!r}"
        elif status == "failed":
            appraisal_failed += 1
            metrics = {}
        else:
            appraisal_error += 1
            metrics = {}

        row = {
            "batch_dir": str(batch_dir),
            "run_dir": str(run_dir),
            "room": room,
            "scenario": scenario,
            "run": run,
            "prompt_file": str(prompt),
            "scene_file": str(scene),
            "appraisal_json": str(out_json),
            "generation_status": gen_rows.get(str(run_dir.resolve()), {}).get("status", ""),
            "generation_duration_sec": gen_rows.get(str(run_dir.resolve()), {}).get("duration_sec", ""),
            "appraisal_status": status,
            "appraisal_duration_sec": dt if dt is not None else "",
            "score_10": metrics.get("score_10", ""),
            "geometry_score_10": metrics.get("geometry_score_10", ""),
            "prompt_match_score_10": metrics.get("prompt_match_score_10", ""),
            "constraint_score_10": metrics.get("constraint_score_10", ""),
            "room_area_m2": metrics.get("room_area_m2", ""),
            "raw_sum_item_area_m2": metrics.get("raw_sum_item_area_m2", ""),
            "free_area_m2": metrics.get("free_area_m2", ""),
            "occupied_union_area_m2": metrics.get("occupied_union_area_m2", ""),
            "free_union_area_m2": metrics.get("free_union_area_m2", ""),
            "largest_free_rectangle_area_m2": metrics.get("largest_free_rectangle_area_m2", ""),
            "largest_free_rectangle_ratio": metrics.get("largest_free_rectangle_ratio", ""),
            "floor_coverage_ratio": metrics.get("floor_coverage_ratio", ""),
            "overlap_area_m2": metrics.get("overlap_area_m2", ""),
            "overlap_ratio": metrics.get("overlap_ratio", ""),
            "outside_room_area_m2": metrics.get("outside_room_area_m2", ""),
            "outside_room_ratio": metrics.get("outside_room_ratio", ""),
            "accessible_objects": metrics.get("accessible_objects", ""),
            "accessible_objects_total": metrics.get("accessible_objects_total", ""),
            "accessibility_ratio": metrics.get("accessibility_ratio", ""),
            "no_overlap": metrics.get("no_overlap", ""),
            "no_outside": metrics.get("no_outside", ""),
            "full_access": metrics.get("full_access", ""),
            "strict_valid": metrics.get("strict_valid", ""),
            "error": err,
        }

        per_run_rows.append(row)
        append_csv_row(per_run_csv, PER_RUN_FIELDS, row)
        append_jsonl(jsonl_path, row)

        print(
            f"[{jobs_total}/{len(runs)}] "
            f"{room}/{scenario}/{run} -> appraiser={status} dt={dt}s"
        )

    # Вставим batch-level generation summary в каждую строку summary через per_run_rows[:1]
    if per_run_rows:
        per_run_rows[0]["generation_success_rate_pct"] = gen_summary["success_rate_pct"]
        per_run_rows[0]["generation_avg_duration_ok_sec"] = gen_summary["avg_duration_ok_sec"]
        per_run_rows[0]["generation_median_duration_ok_sec"] = gen_summary["median_duration_ok_sec"]
        per_run_rows[0]["generation_min_duration_ok_sec"] = gen_summary["min_duration_ok_sec"]
        per_run_rows[0]["generation_max_duration_ok_sec"] = gen_summary["max_duration_ok_sec"]

    write_summary_csv(batch_dir, per_run_rows)

    summary_json = batch_dir / "appraisal_summary.json"
    summary_obj = {
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "batch_dir": str(batch_dir),
        "generation": gen_summary,
        "appraiser": {
            "jobs_total": jobs_total,
            "ok_count": appraisal_ok,
            "failed_count": appraisal_failed,
            "error_count": appraisal_error,
            "success_rate_pct": (100.0 * appraisal_ok / jobs_total) if jobs_total > 0 else None,
        },
        "per_run_csv": str(per_run_csv.resolve()),
        "summary_csv": str((batch_dir / "appraisal_summary.csv").resolve()),
        "jsonl": str(jsonl_path.resolve()),
    }
    summary_json.write_text(json.dumps(summary_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== DONE ===")
    print(json.dumps(summary_obj, ensure_ascii=False, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run Appraiser on batch directories and export full metrics to CSV"
    )
    p.add_argument("--batches", nargs="+", required=True, help="Batch directories to appraise")
    p.add_argument("--appraiser-script", default=DEFAULT_APPRAISER)
    p.add_argument("--rerun-existing", action="store_true")
    args = p.parse_args()

    appraiser_script = Path(args.appraiser_script).expanduser().resolve()
    if not appraiser_script.is_file():
        raise RuntimeError(f"Не найден appraiser script: {appraiser_script}")

    for batch in args.batches:
        appraise_batch(Path(batch), appraiser_script, rerun_existing=bool(args.rerun_existing))


if __name__ == "__main__":
    main()