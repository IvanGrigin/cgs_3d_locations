#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collect coarse and detailed timing rows from a full-circle run directory."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any


TIME_RE = re.compile(r"Time:\s*(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)")


def read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def parse_blender_time(text: str) -> float:
    total = 0.0
    for match in TIME_RE.finditer(text):
        hours = int(match.group(1) or 0)
        minutes = int(match.group(2))
        seconds = float(match.group(3))
        total += hours * 3600.0 + minutes * 60.0 + seconds
    return total


def iter_run_dirs(out_root: Path) -> list[Path]:
    return sorted(p for p in out_root.iterdir() if p.is_dir() and p.name != "analysis")


def add_row(rows: list[dict[str, Any]], **row: Any) -> None:
    rows.append(row)


def collect_rows(out_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in iter_run_dirs(out_root):
        room_key = run_dir.name
        timings = read_json(run_dir / "timings.json")
        stages = timings.get("stages") if isinstance(timings, dict) else None
        if isinstance(stages, dict):
            for stage, info in stages.items():
                if not isinstance(info, dict):
                    continue
                if "duration_sec" in info:
                    add_row(
                        rows,
                        room_key=room_key,
                        source="batch_timings",
                        stage=stage,
                        status=info.get("status"),
                        duration_sec=float(info.get("duration_sec") or 0.0),
                        bytes="",
                        path=str(run_dir / "timings.json"),
                    )

        pipeline_detail = read_json(run_dir / "pipeline_stage_timings.json")
        detail_stages = pipeline_detail.get("stages") if isinstance(pipeline_detail, dict) else None
        if isinstance(detail_stages, list):
            for info in detail_stages:
                if not isinstance(info, dict):
                    continue
                add_row(
                    rows,
                    room_key=room_key,
                    source="pipeline_stage_timings",
                    stage=info.get("stage"),
                    status=info.get("status"),
                    duration_sec=float(info.get("duration_sec") or 0.0),
                    bytes=info.get("bytes", ""),
                    path=str(run_dir / "pipeline_stage_timings.json"),
                )

        remote_detail = read_json(run_dir / "infinigen_remote_timings.json")
        remote_stages = remote_detail.get("stages") if isinstance(remote_detail, dict) else None
        if isinstance(remote_stages, list):
            for info in remote_stages:
                if not isinstance(info, dict):
                    continue
                add_row(
                    rows,
                    room_key=room_key,
                    source="infinigen_remote_timings",
                    stage=info.get("stage"),
                    status=info.get("status"),
                    duration_sec=float(info.get("duration_sec") or 0.0),
                    bytes=info.get("bytes", ""),
                    path=str(run_dir / "infinigen_remote_timings.json"),
                )

        for log_path in sorted((run_dir / "vlm_review_views").glob("*/render_stdout.log")):
            duration = parse_blender_time(log_path.read_text(encoding="utf-8", errors="replace"))
            if duration <= 0:
                continue
            add_row(
                rows,
                room_key=room_key,
                source="render_stdout_sum",
                stage=f"render_views_state:{log_path.parent.name}",
                status="ok",
                duration_sec=round(duration, 3),
                bytes="",
                path=str(log_path),
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        key = (str(row.get("source") or ""), str(row.get("stage") or ""))
        groups.setdefault(key, []).append(float(row.get("duration_sec") or 0.0))
    out: list[dict[str, Any]] = []
    for (source, stage), values in sorted(groups.items()):
        values = [v for v in values if v >= 0]
        if not values:
            continue
        out.append(
            {
                "source": source,
                "stage": stage,
                "n": len(values),
                "total_sec": round(sum(values), 3),
                "mean_sec": round(statistics.mean(values), 3),
                "median_sec": round(statistics.median(values), 3),
                "min_sec": round(min(values), 3),
                "max_sec": round(max(values), 3),
            }
        )
    return out


def room_totals(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    totals: dict[str, dict[str, float]] = {}
    for row in rows:
        room = str(row.get("room_key") or "")
        source = str(row.get("source") or "")
        stage = str(row.get("stage") or "")
        value = float(row.get("duration_sec") or 0.0)
        bucket = totals.setdefault(room, {})
        if source == "batch_timings":
            bucket[f"batch_{stage}_sec"] = bucket.get(f"batch_{stage}_sec", 0.0) + value
        elif source == "infinigen_remote_timings":
            bucket[f"remote_{stage}_sec"] = bucket.get(f"remote_{stage}_sec", 0.0) + value
            if stage.startswith("upload_"):
                bucket["remote_upload_total_sec"] = bucket.get("remote_upload_total_sec", 0.0) + value
            if stage.startswith("download_"):
                bucket["remote_download_total_sec"] = bucket.get("remote_download_total_sec", 0.0) + value
        elif source == "pipeline_stage_timings":
            bucket[f"pipeline_{stage}_sec"] = bucket.get(f"pipeline_{stage}_sec", 0.0) + value
        elif source == "render_stdout_sum":
            bucket["render_stdout_sum_sec"] = bucket.get("render_stdout_sum_sec", 0.0) + value
    out: list[dict[str, Any]] = []
    for room, values in sorted(totals.items()):
        row: dict[str, Any] = {"room_key": room}
        row.update({k: round(v, 3) for k, v in sorted(values.items())})
        out.append(row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--analysis-dir", default=None)
    args = parser.parse_args()

    out_root = Path(args.out_root).expanduser().resolve()
    analysis_dir = Path(args.analysis_dir).expanduser().resolve() if args.analysis_dir else out_root / "analysis"
    rows = collect_rows(out_root)
    write_csv(
        analysis_dir / "timing_breakdown_rows.csv",
        rows,
        ["room_key", "source", "stage", "status", "duration_sec", "bytes", "path"],
    )
    summary = summarize(rows)
    write_csv(
        analysis_dir / "timing_breakdown_summary.csv",
        summary,
        ["source", "stage", "n", "total_sec", "mean_sec", "median_sec", "min_sec", "max_sec"],
    )
    totals = room_totals(rows)
    fields = sorted({key for row in totals for key in row.keys()})
    if "room_key" in fields:
        fields.remove("room_key")
    write_csv(analysis_dir / "timing_room_totals.csv", totals, ["room_key", *fields])
    print(f"rows={len(rows)}")
    print(analysis_dir / "timing_breakdown_rows.csv")
    print(analysis_dir / "timing_breakdown_summary.csv")
    print(analysis_dir / "timing_room_totals.csv")


if __name__ == "__main__":
    main()
