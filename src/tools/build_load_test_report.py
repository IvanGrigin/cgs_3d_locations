#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a short load/perf report for batch runs.

Supports two sources:
1) full-circle batch summary (out/runs/<run>/full_circle_summary.json)
2) generation_time_by_room.csv from `docs/full_circle_vlm_stage_averages/`

The script is intentionally lightweight and can be used right before defense
to produce reproducible evidence for:
- throughput of finished rooms per wall-clock hour
- stage durations (pipeline/render/vlm)
- timing breakdown (infinigen/request-to-result vs local postprocessing)
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Optional


def parse_ts(value: str | None) -> Optional[datetime]:
    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00").replace("T", " "))
    except Exception:
        try:
            return datetime.fromisoformat(value)
        except Exception:
            return None


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if q <= 0:
        return min(values)
    if q >= 1:
        return max(values)
    s = sorted(values)
    pos = (len(s) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    if lo == hi:
        return s[lo]
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac


@dataclass
class StageSummary:
    n: int
    mean: float
    median_: float
    p50: float
    p95: float
    minimum: float
    maximum: float


def summarize(values: list[float]) -> Optional[StageSummary]:
    if not values:
        return None
    return StageSummary(
        n=len(values),
        mean=mean(values),
        median_=median(values),
        p50=quantile(values, 0.5),
        p95=quantile(values, 0.95),
        minimum=min(values),
        maximum=max(values),
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_generation_csv(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}

    out: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case = str(row.get("case") or "").strip()
            if case:
                out[case] = row
    return out


def float_or_none(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def to_dict(summary: StageSummary | None) -> dict[str, Any] | None:
    if summary is None:
        return None
    return {
        "n": summary.n,
        "mean": summary.mean,
        "median": summary.median_,
        "p50": summary.p50,
        "p95": summary.p95,
        "min": summary.minimum,
        "max": summary.maximum,
    }


def build_full_circle_report(full_circle_root: Path) -> dict[str, Any]:
    summary_path = full_circle_root / "full_circle_summary.json"
    if not summary_path.is_file():
        return {"error": f"not found: {summary_path}"}

    data = read_json(summary_path)
    summary_rows = data.get("summary") if isinstance(data, dict) else None
    if not isinstance(summary_rows, list):
        return {"error": "full_circle_summary.json has no list in key='summary'"}

    # Aggregate whole-run throughput
    run_ok = [row for row in summary_rows if row.get("status") == "ok"]
    run_started = parse_ts(data.get("started_at"))
    run_finished = parse_ts(data.get("finished_at"))

    # Gather per-case durations by stage
    stage_values: dict[str, list[float]] = defaultdict(list)
    case_start_times: list[datetime] = []
    case_end_times: list[datetime] = []

    for row in summary_rows:
        timings = row.get("timings") or {}
        if not isinstance(timings, dict):
            continue

        per_case_starts = []
        per_case_ends = []

        for stage in ("pipeline", "render_views", "vlm_eval"):
            stage_info = timings.get(stage)
            if not isinstance(stage_info, dict):
                continue
            dur = stage_info.get("duration_sec")
            if dur is None and stage_info.get("started_at") and stage_info.get("finished_at"):
                s = parse_ts(stage_info.get("started_at"))
                e = parse_ts(stage_info.get("finished_at"))
                if s is not None and e is not None:
                    dur = (e - s).total_seconds()
            if isinstance(dur, (int, float)):
                stage_values[stage].append(float(dur))

            if stage_info.get("started_at"):
                s = parse_ts(stage_info.get("started_at"))
                if s is not None:
                    per_case_starts.append(s)
            if stage_info.get("finished_at"):
                e = parse_ts(stage_info.get("finished_at"))
                if e is not None:
                    per_case_ends.append(e)

        if per_case_starts:
            case_start_times.append(min(per_case_starts))
        if per_case_ends:
            case_end_times.append(max(per_case_ends))

    if run_started is None and case_start_times:
        run_started = min(case_start_times)
    if run_finished is None and case_end_times:
        run_finished = max(case_end_times)

    throughput_rows = len(summary_rows)
    throughput_ok = len(run_ok)
    wall_hours = None
    if run_started and run_finished and run_finished > run_started:
        wall_hours = (run_finished - run_started).total_seconds() / 3600.0

    stages = {stage: to_dict(summarize(vals)) for stage, vals in stage_values.items()}

    return {
        "source": "full_circle_summary",
        "run_root": str(full_circle_root.resolve()),
        "run_started": run_started.isoformat() if run_started else None,
        "run_finished": run_finished.isoformat() if run_finished else None,
        "run_duration_hours": wall_hours,
        "cases_total": throughput_rows,
        "cases_ok": throughput_ok,
        "throughput_cases_per_hour": throughput_ok / wall_hours if wall_hours and wall_hours > 0 else None,
        "stages_sec": stages,
    }


def build_generation_report(csv_path: Path) -> dict[str, Any]:
    rows = read_generation_csv(csv_path)
    if not rows:
        return {"error": f"no data in {csv_path}"}

    generation_time = [float(v) for v in (float_or_none(r.get("generation_time_min")) for r in rows.values()) if v is not None]
    wall = [float(v) for v in (float_or_none(r.get("wall_elapsed_min")) for r in rows.values()) if v is not None]
    infinigen = [float(v) for v in (float_or_none(r.get("raw_infinigen_main_total_min")) for r in rows.values()) if v is not None]
    post_sup = [float(v) for v in (float_or_none(r.get("local_postprocess_supplier_min")) for r in rows.values()) if v is not None]

    return {
        "source": "generation_time_csv",
        "csv_path": str(csv_path.resolve()),
        "records": len(rows),
        "generation_time_min": to_dict(summarize(generation_time)),
        "wall_elapsed_min": to_dict(summarize(wall)),
        "raw_infinigen_main_total_min": to_dict(summarize(infinigen)),
        "local_postprocess_supplier_min": to_dict(summarize(post_sup)),
    }


def write_report(out_dir: Path, payload: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "load_test_report.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = out_dir / "load_test_report.csv"
    rows: list[dict[str, Any]] = []

    # flat rows for csv
    full_circle = payload.get("full_circle") or {}
    gen = payload.get("generation") or {}

    if "run_duration_hours" in full_circle:
        rows.append({
            "metric": "run_duration_hours",
            "value": full_circle.get("run_duration_hours"),
            "source": "full_circle",
        })
    if "throughput_cases_per_hour" in full_circle:
        rows.append({
            "metric": "throughput_cases_per_hour",
            "value": full_circle.get("throughput_cases_per_hour"),
            "source": "full_circle",
        })

    for stage_name, vals in (full_circle.get("stages_sec") or {}).items():
        if not isinstance(vals, dict):
            continue
        base = f"stage:{stage_name}_sec"
        for key in ("mean", "median", "p50", "p95", "min", "max"):
            rows.append({
                "metric": f"{base}_{key}",
                "value": vals.get(key),
                "source": "full_circle",
            })

    for prefix, section in (("generation", gen), ("wall", gen.get("wall_elapsed_min") or {}),
                           ("infinigen", gen.get("raw_infinigen_main_total_min") or {}),
                           ("postprocess_sup", gen.get("local_postprocess_supplier_min") or {})):
        if not isinstance(section, dict) or prefix in {"wall", "infinigen", "postprocess_sup"}:
            pass
        if not section:
            continue
        if prefix in {"wall", "infinigen", "postprocess_sup"}:
            source_section = section
            name = prefix
        else:
            source_section = section
            name = prefix
        for key in ("mean", "median", "p50", "p95", "min", "max"):
            rows.append({
                "metric": f"generation_{name}_{key}",
                "value": source_section.get(key),
                "source": "generation_csv",
            })

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value", "source"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    md_path = out_dir / "load_test_report.md"
    lines = [
        "# Load test quick report",
        "",
    ]

    if "error" in full_circle:
        lines.append(f"- Full circle: {full_circle['error']}")
    else:
        lines.extend(
            [
                "## Full-circle run",
                f"- Cases total: {full_circle.get('cases_total')}",
                f"- Cases ok: {full_circle.get('cases_ok')}",
                f"- Run duration hours: {full_circle.get('run_duration_hours')}",
                f"- Throughput, rooms/hour: {full_circle.get('throughput_cases_per_hour')}",
                "",
            ]
        )
        for stage in ("pipeline", "render_views", "vlm_eval"):
            section = (full_circle.get("stages_sec") or {}).get(stage)
            if isinstance(section, dict):
                lines.append(
                    f"- {stage}: n={section.get('n')}, mean={section.get('mean')}, "
                    f"p50={section.get('p50')}, p95={section.get('p95')}, min={section.get('min')}, max={section.get('max')}"
                )

    if "error" not in gen:
        lines.extend(
            [
                "",
                "## Generation timing CSV",
                f"- Records: {gen.get('records')}",
                f"- Generation time (min): {gen.get('generation_time_min')}",
                f"- Wall elapsed (min): {gen.get('wall_elapsed_min')}",
                f"- Infinigen raw timing (min): {gen.get('raw_infinigen_main_total_min')}",
                f"- Local postprocess+supplier (min): {gen.get('local_postprocess_supplier_min')}",
            ]
        )

    md_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build quick load/perf report.")
    parser.add_argument("--full-circle-root", default=None, help="Run root with full_circle_summary.json")
    parser.add_argument(
        "--generation-csv",
        default="docs/full_circle_vlm_stage_averages/generation_time_by_room.csv",
        help="Path to generation_time_by_room.csv",
    )
    parser.add_argument("--out-dir", required=True, help="Output dir for report files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()

    payload: dict[str, Any] = {
        "full_circle": {},
        "generation": {},
    }

    if args.full_circle_root:
        payload["full_circle"] = build_full_circle_report(Path(args.full_circle_root).expanduser().resolve())

    payload["generation"] = build_generation_report(Path(args.generation_csv).expanduser().resolve())

    write_report(out_dir, payload)
    print(f"report written: {out_dir / 'load_test_report.json'}")
    print(f"report written: {out_dir / 'load_test_report.csv'}")
    print(f"report written: {out_dir / 'load_test_report.md'}")


if __name__ == "__main__":
    main()

