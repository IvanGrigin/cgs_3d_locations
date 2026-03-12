#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/tools/batch_blend_to_orbit_gif.py

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_SEARCH_ROOT = "out/batch_typical_rooms"
DEFAULT_SINGLE_BLEND_SCRIPT = "src/tools/blend_to_orbit_gif.py"


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def collect_blend_files(search_root: Path) -> list[Path]:
    files = sorted(search_root.rglob("*.blend"))
    if not files:
        raise RuntimeError(f"Не найдено ни одного .blend в {search_root}")
    return files


def ensure_csv_header(csv_path: Path) -> None:
    import csv

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists() and csv_path.stat().st_size > 0:
        return

    fieldnames = [
        "started_at",
        "finished_at",
        "status",
        "blend_file",
        "gif_file",
        "returncode",
        "duration_sec",
        "stdout_log",
        "stderr_log",
        "error",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def append_csv_row(csv_path: Path, row: dict[str, Any]) -> None:
    import csv

    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writerow(row)


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def run_one_blend(
    single_script: Path,
    blend_path: Path,
    blender: str | None,
    width: int,
    height: int,
    samples: int,
    yaw_step: float,
    elevations: str,
    duration_ms: int,
    margin: float,
    keep_frames: bool,
) -> dict[str, Any]:
    started = datetime.now()

    gif_path = blend_path.with_suffix(".gif")
    stdout_log = blend_path.with_name(f"{blend_path.stem}_gif_stdout.log")
    stderr_log = blend_path.with_name(f"{blend_path.stem}_gif_stderr.log")
    meta_json = blend_path.with_name(f"{blend_path.stem}_gif_job.json")

    cmd = [
        sys.executable,
        str(single_script.resolve()),
        "--blend", str(blend_path.resolve()),
        "--width", str(int(width)),
        "--height", str(int(height)),
        "--samples", str(int(samples)),
        "--yaw-step", str(float(yaw_step)),
        "--elevations", elevations,
        "--duration-ms", str(int(duration_ms)),
        "--margin", str(float(margin)),
    ]
    if blender:
        cmd += ["--blender", blender]
    if keep_frames:
        cmd += ["--keep-frames"]

    meta = {
        "blend_file": str(blend_path.resolve()),
        "gif_file": str(gif_path.resolve()),
        "command": cmd,
        "started_at": started.isoformat(timespec="seconds"),
    }
    meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        with stdout_log.open("w", encoding="utf-8") as out, stderr_log.open("w", encoding="utf-8") as err:
            t0 = time.perf_counter()
            proc = subprocess.run(cmd, stdout=out, stderr=err, text=True, check=False)
            duration_sec = round(time.perf_counter() - t0, 3)

        finished = datetime.now()
        status = "ok" if proc.returncode == 0 else "failed"
        error = "" if proc.returncode == 0 else f"blend_to_orbit_gif returncode={proc.returncode}"

        return {
            "started_at": started.isoformat(timespec="seconds"),
            "finished_at": finished.isoformat(timespec="seconds"),
            "status": status,
            "blend_file": str(blend_path.resolve()),
            "gif_file": str(gif_path.resolve()),
            "returncode": proc.returncode,
            "duration_sec": duration_sec,
            "stdout_log": str(stdout_log.resolve()),
            "stderr_log": str(stderr_log.resolve()),
            "error": error,
        }
    except Exception as e:
        finished = datetime.now()
        return {
            "started_at": started.isoformat(timespec="seconds"),
            "finished_at": finished.isoformat(timespec="seconds"),
            "status": "error",
            "blend_file": str(blend_path.resolve()),
            "gif_file": str(gif_path.resolve()),
            "returncode": -1,
            "duration_sec": -1,
            "stdout_log": str(stdout_log.resolve()),
            "stderr_log": str(stderr_log.resolve()),
            "error": repr(e),
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Пакетно проходит по всем .blend внутри каталога и создаёт GIF-облёт "
            "рядом с каждым .blend. Ошибки отдельных файлов не валят весь batch."
        )
    )
    parser.add_argument("--search-root", default=DEFAULT_SEARCH_ROOT)
    parser.add_argument("--single-script", default=DEFAULT_SINGLE_BLEND_SCRIPT)
    parser.add_argument("--blender", default=None)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--yaw-step", type=float, default=30.0)
    parser.add_argument("--elevations", default="0,35,72")
    parser.add_argument("--duration-ms", type=int, default=500)
    parser.add_argument("--margin", type=float, default=1.35)
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Пропускать .blend, если рядом уже есть .gif",
    )
    args = parser.parse_args()

    search_root = Path(args.search_root).expanduser().resolve()
    single_script = Path(args.single_script).expanduser().resolve()

    if not single_script.is_file():
        raise RuntimeError(f"Не найден single GIF script: {single_script}")

    blend_files = collect_blend_files(search_root)
    if args.skip_existing:
        blend_files = [p for p in blend_files if not p.with_suffix(".gif").exists()]
    if args.limit is not None:
        blend_files = blend_files[: max(0, int(args.limit))]

    batch_dir = search_root / f"_gif_batch_{now_stamp()}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    csv_log = batch_dir / "gif_batch_results.csv"
    jsonl_log = batch_dir / "gif_batch_results.jsonl"
    summary_json = batch_dir / "summary.json"
    ensure_csv_header(csv_log)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "search_root": str(search_root),
        "single_script": str(single_script),
        "blender": args.blender,
        "width": args.width,
        "height": args.height,
        "samples": args.samples,
        "yaw_step": args.yaw_step,
        "elevations": args.elevations,
        "duration_ms": args.duration_ms,
        "margin": args.margin,
        "keep_frames": bool(args.keep_frames),
        "skip_existing": bool(args.skip_existing),
        "limit": args.limit,
        "blend_files": [str(p) for p in blend_files],
    }
    (batch_dir / "gif_batch_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    jobs_total = 0
    ok_count = 0
    failed_count = 0
    error_count = 0

    for blend_path in blend_files:
        jobs_total += 1
        print(f"[{jobs_total}] blend={blend_path}")

        row = run_one_blend(
            single_script=single_script,
            blend_path=blend_path,
            blender=args.blender,
            width=int(args.width),
            height=int(args.height),
            samples=int(args.samples),
            yaw_step=float(args.yaw_step),
            elevations=str(args.elevations),
            duration_ms=int(args.duration_ms),
            margin=float(args.margin),
            keep_frames=bool(args.keep_frames),
        )

        append_csv_row(csv_log, row)
        append_jsonl(jsonl_log, row)

        if row["status"] == "ok":
            ok_count += 1
        elif row["status"] == "failed":
            failed_count += 1
        else:
            error_count += 1

        summary = {
            "jobs_total": jobs_total,
            "ok_count": ok_count,
            "failed_count": failed_count,
            "error_count": error_count,
            "last_job": row,
        }
        summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    final_summary = {
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "jobs_total": jobs_total,
        "ok_count": ok_count,
        "failed_count": failed_count,
        "error_count": error_count,
        "batch_dir": str(batch_dir.resolve()),
        "csv_log": str(csv_log.resolve()),
        "jsonl_log": str(jsonl_log.resolve()),
    }
    summary_json.write_text(json.dumps(final_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== GIF BATCH FINISHED ===")
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()