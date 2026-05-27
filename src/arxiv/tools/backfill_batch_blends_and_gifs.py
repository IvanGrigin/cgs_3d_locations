#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_BATCH_DIR = "out/batch_typical_rooms/batch_20260314_131530"
DEFAULT_GIF_BATCH_SCRIPT = "src/tools/batch_blend_to_orbit_gif.py"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def find_job_meta_files(batch_dir: Path) -> list[Path]:
    return sorted(batch_dir.rglob("job_meta.json"))


def find_blend_files(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("*.blend"))


def has_any_gif_next_to_blend(run_dir: Path) -> bool:
    for blend in find_blend_files(run_dir):
        if blend.with_suffix(".gif").exists():
            return True
    return False


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def strip_flag_with_optional_value(cmd: list[str], flag: str) -> list[str]:
    """
    Удаляет флаг flag.
    Если после него идёт значение, оно НЕ удаляется автоматически,
    поэтому использовать только для флагов без значения, например --skip-blender, --headless.
    """
    return [x for x in cmd if x != flag]


def replace_cmd_python(cmd: list[str]) -> list[str]:
    if not cmd:
        raise RuntimeError("Пустая command в job_meta.json")
    cmd = cmd[:]
    cmd[0] = sys.executable
    return cmd


def ensure_run_dir_arg(cmd: list[str], run_dir: Path) -> list[str]:
    """
    Если в исходной command почему-то нет --run-dir, добавляем.
    Если есть, оставляем как есть.
    """
    if "--run-dir" in cmd:
        return cmd
    return cmd + ["--run-dir", str(run_dir.resolve())]


def build_backfill_command(meta_json: Path, force_headless: bool) -> list[str]:
    meta = load_json(meta_json)
    cmd = meta.get("command")
    if not isinstance(cmd, list) or not cmd:
        raise RuntimeError(f"Некорректное поле command в {meta_json}")

    run_dir = meta_json.parent

    cmd = replace_cmd_python(cmd)
    cmd = strip_flag_with_optional_value(cmd, "--skip-blender")

    if force_headless and "--headless" not in cmd and "--no-headless" not in cmd:
        cmd.append("--headless")

    cmd = ensure_run_dir_arg(cmd, run_dir)
    return cmd


def run_subprocess_with_logs(
    cmd: list[str],
    stdout_log: Path,
    stderr_log: Path,
) -> tuple[int, float]:
    with stdout_log.open("w", encoding="utf-8") as out, stderr_log.open("w", encoding="utf-8") as err:
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, stdout=out, stderr=err, text=True, check=False)
        dt = round(time.perf_counter() - t0, 3)
    return proc.returncode, dt


def run_backfill_for_run(
    meta_json: Path,
    force: bool,
    force_headless: bool,
) -> dict[str, Any]:
    run_dir = meta_json.parent
    started_at = now_iso()

    existing_blends_before = [str(x.resolve()) for x in find_blend_files(run_dir)]

    if existing_blends_before and not force:
        return {
            "started_at": started_at,
            "finished_at": now_iso(),
            "status": "skipped_existing_blend",
            "run_dir": str(run_dir.resolve()),
            "meta_json": str(meta_json.resolve()),
            "returncode": 0,
            "duration_sec": 0.0,
            "stdout_log": "",
            "stderr_log": "",
            "error": "",
            "blends_before": existing_blends_before,
            "blends_after": existing_blends_before,
            "command": None,
        }

    stdout_log = run_dir / "backfill_blend_stdout.log"
    stderr_log = run_dir / "backfill_blend_stderr.log"
    job_json = run_dir / "backfill_blend_job.json"

    try:
        cmd = build_backfill_command(meta_json, force_headless=force_headless)

        job_json.write_text(
            json.dumps(
                {
                    "created_at": now_iso(),
                    "meta_json": str(meta_json.resolve()),
                    "run_dir": str(run_dir.resolve()),
                    "command": cmd,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        returncode, duration_sec = run_subprocess_with_logs(cmd, stdout_log, stderr_log)

        blends_after = [str(x.resolve()) for x in find_blend_files(run_dir)]
        finished_at = now_iso()

        if returncode == 0 and blends_after:
            status = "ok"
            error = ""
        elif returncode == 0 and not blends_after:
            status = "failed_no_blend_created"
            error = "pipeline finished with code 0, but no .blend appeared in run_dir"
        else:
            status = "failed"
            error = f"pipeline returncode={returncode}"

        return {
            "started_at": started_at,
            "finished_at": finished_at,
            "status": status,
            "run_dir": str(run_dir.resolve()),
            "meta_json": str(meta_json.resolve()),
            "returncode": returncode,
            "duration_sec": duration_sec,
            "stdout_log": str(stdout_log.resolve()),
            "stderr_log": str(stderr_log.resolve()),
            "error": error,
            "blends_before": existing_blends_before,
            "blends_after": blends_after,
            "command": cmd,
        }
    except Exception as e:
        return {
            "started_at": started_at,
            "finished_at": now_iso(),
            "status": "error",
            "run_dir": str(run_dir.resolve()),
            "meta_json": str(meta_json.resolve()),
            "returncode": -1,
            "duration_sec": -1.0,
            "stdout_log": str(stdout_log.resolve()),
            "stderr_log": str(stderr_log.resolve()),
            "error": repr(e),
            "blends_before": existing_blends_before,
            "blends_after": [str(x.resolve()) for x in find_blend_files(run_dir)],
            "command": None,
        }


def save_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_gif_batch(
    gif_batch_script: Path,
    batch_dir: Path,
    blender: str | None,
    width: int,
    height: int,
    samples: int,
    yaw_step: float,
    elevations: str,
    duration_ms: int,
    margin: float,
    keep_frames: bool,
    force_rebuild_gifs: bool,
) -> int:
    cmd = [
        sys.executable,
        str(gif_batch_script.resolve()),
        "--search-root", str(batch_dir.resolve()),
        "--width", str(int(width)),
        "--height", str(int(height)),
        "--samples", str(int(samples)),
        "--yaw-step", str(float(yaw_step)),
        "--elevations", str(elevations),
        "--duration-ms", str(int(duration_ms)),
        "--margin", str(float(margin)),
    ]

    if blender:
        cmd += ["--blender", blender]
    if keep_frames:
        cmd += ["--keep-frames"]
    if not force_rebuild_gifs:
        cmd += ["--skip-existing"]

    print("\n=== RUN GIF BATCH ===")
    print(" ".join(shlex.quote(x) for x in cmd))
    proc = subprocess.run(cmd, text=True, check=False)
    return proc.returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Для существующего batch находит все run_* через job_meta.json, "
            "досоздаёт недостающие .blend, затем пакетно строит .gif."
        )
    )
    parser.add_argument("--batch-dir", default=DEFAULT_BATCH_DIR)
    parser.add_argument("--gif-batch-script", default=DEFAULT_GIF_BATCH_SCRIPT)
    parser.add_argument("--blender", default=None)

    parser.add_argument("--force", action="store_true", help="Перегенерировать .blend даже если он уже есть.")
    parser.add_argument("--force-headless", action="store_true", help="Добавлять --headless при backfill-запуске.")
    parser.add_argument("--skip-gif", action="store_true", help="Не запускать генерацию GIF после backfill.")

    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--yaw-step", type=float, default=30.0)
    parser.add_argument("--elevations", default="0,35,72")
    parser.add_argument("--duration-ms", type=int, default=500)
    parser.add_argument("--margin", type=float, default=1.35)
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument(
        "--force-rebuild-gifs",
        action="store_true",
        help="Перестраивать GIF даже если рядом с .blend уже есть .gif",
    )

    args = parser.parse_args()

    batch_dir = Path(args.batch_dir).expanduser().resolve()
    gif_batch_script = Path(args.gif_batch_script).expanduser().resolve()

    if not batch_dir.is_dir():
        raise RuntimeError(f"Не найден batch-dir: {batch_dir}")
    if not gif_batch_script.is_file():
        raise RuntimeError(f"Не найден gif batch script: {gif_batch_script}")

    meta_files = find_job_meta_files(batch_dir)
    if not meta_files:
        raise RuntimeError(f"В {batch_dir} не найдено ни одного job_meta.json")

    control_dir = batch_dir / f"_blend_backfill_{now_stamp()}"
    control_dir.mkdir(parents=True, exist_ok=True)
    jsonl_log = control_dir / "backfill_results.jsonl"
    summary_json = control_dir / "summary.json"

    jobs_total = 0
    ok_count = 0
    skipped_count = 0
    failed_count = 0
    error_count = 0

    manifest = {
        "created_at": now_iso(),
        "batch_dir": str(batch_dir),
        "gif_batch_script": str(gif_batch_script),
        "blender": args.blender,
        "force": bool(args.force),
        "force_headless": bool(args.force_headless),
        "skip_gif": bool(args.skip_gif),
        "width": args.width,
        "height": args.height,
        "samples": args.samples,
        "yaw_step": args.yaw_step,
        "elevations": args.elevations,
        "duration_ms": args.duration_ms,
        "margin": args.margin,
        "keep_frames": bool(args.keep_frames),
        "force_rebuild_gifs": bool(args.force_rebuild_gifs),
        "job_meta_files": [str(x) for x in meta_files],
    }
    (control_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for i, meta_json in enumerate(meta_files, start=1):
        print(f"[{i}/{len(meta_files)}] backfill: {meta_json.parent}")
        jobs_total += 1

        row = run_backfill_for_run(
            meta_json=meta_json,
            force=bool(args.force),
            force_headless=bool(args.force_headless),
        )
        save_jsonl(jsonl_log, row)

        if row["status"] == "ok":
            ok_count += 1
        elif row["status"] == "skipped_existing_blend":
            skipped_count += 1
        elif row["status"] in {"failed", "failed_no_blend_created"}:
            failed_count += 1
        else:
            error_count += 1

        summary_json.write_text(
            json.dumps(
                {
                    "updated_at": now_iso(),
                    "jobs_total": jobs_total,
                    "ok_count": ok_count,
                    "skipped_count": skipped_count,
                    "failed_count": failed_count,
                    "error_count": error_count,
                    "last_row": row,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    final_summary = {
        "finished_at": now_iso(),
        "jobs_total": jobs_total,
        "ok_count": ok_count,
        "skipped_count": skipped_count,
        "failed_count": failed_count,
        "error_count": error_count,
        "control_dir": str(control_dir.resolve()),
        "jsonl_log": str(jsonl_log.resolve()),
    }

    gif_returncode = None
    if not args.skip_gif:
        gif_returncode = run_gif_batch(
            gif_batch_script=gif_batch_script,
            batch_dir=batch_dir,
            blender=args.blender,
            width=int(args.width),
            height=int(args.height),
            samples=int(args.samples),
            yaw_step=float(args.yaw_step),
            elevations=str(args.elevations),
            duration_ms=int(args.duration_ms),
            margin=float(args.margin),
            keep_frames=bool(args.keep_frames),
            force_rebuild_gifs=bool(args.force_rebuild_gifs),
        )
        final_summary["gif_batch_returncode"] = gif_returncode

    summary_json.write_text(
        json.dumps(final_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== BACKFILL FINISHED ===")
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))

    if gif_returncode not in (None, 0):
        sys.exit(gif_returncode)


if __name__ == "__main__":
    main()