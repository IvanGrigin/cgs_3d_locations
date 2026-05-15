#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PAGE_RE = re.compile(
    r"^housesru__(?P<year>[^_]+)__(?P<slug>.+)__p(?P<page>\d{4})\.(?P<ext>[A-Za-z0-9]+)$"
)


@dataclass
class Job:
    year: str
    slug: str
    files: list[Path]

    @property
    def key(self) -> str:
        return f"{self.year}__{self.slug}"

    @property
    def page_count(self) -> int:
        return len(self.files)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _safe_rmtree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def discover_jobs(pages_dir: Path) -> list[Job]:
    grouped: dict[tuple[str, str], list[Path]] = {}

    for path in sorted(pages_dir.iterdir()):
        if not path.is_file():
            continue

        m = PAGE_RE.match(path.name)
        if not m:
            continue

        year = m.group("year")
        slug = m.group("slug")
        grouped.setdefault((year, slug), []).append(path)

    jobs = [
        Job(year=year, slug=slug, files=sorted(files))
        for (year, slug), files in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1]))
    ]
    return jobs


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if mode == "symlink":
        dst.symlink_to(src.resolve())
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def prepare_job_input(job: Job, jobs_input_root: Path, mode: str, resume: bool) -> Path:
    job_input_dir = jobs_input_root / job.key
    if job_input_dir.exists() and not resume:
        _safe_rmtree(job_input_dir)

    _mkdir(job_input_dir)

    # Если resume=True и папка уже наполнена, повторно не создаём.
    existing_files = list(job_input_dir.iterdir()) if job_input_dir.exists() else []
    if resume and len(existing_files) == len(job.files):
        return job_input_dir

    for src in job.files:
        dst = job_input_dir / src.name
        if dst.exists() or dst.is_symlink():
            continue
        link_or_copy(src, dst, mode)

    return job_input_dir


def build_parser_cmd(
    python_exe: str,
    parser_script: Path,
    input_dir: Path,
    out_dir: Path,
    preset: str,
    min_final_score: float,
    debug: bool,
) -> list[str]:
    cmd = [
        python_exe,
        str(parser_script),
        "--input-dir",
        str(input_dir),
        "--out",
        str(out_dir),
        "--preset",
        preset,
        "--min-final-score",
        str(min_final_score),
    ]
    if debug:
        cmd.append("--debug")
    return cmd


def run_job(
    job: Job,
    jobs_input_root: Path,
    jobs_out_root: Path,
    mode: str,
    resume: bool,
    python_exe: str,
    parser_script: Path,
    preset: str,
    min_final_score: float,
    debug: bool,
    repo_root: Path,
) -> dict:
    t0 = time.time()

    job_input_dir = prepare_job_input(job, jobs_input_root, mode, resume)
    job_out_dir = jobs_out_root / job.key
    if job_out_dir.exists() and not resume:
        _safe_rmtree(job_out_dir)
    _mkdir(job_out_dir)

    job_meta_dir = job_out_dir / "_runner_meta"
    _mkdir(job_meta_dir)

    cmd = build_parser_cmd(
        python_exe=python_exe,
        parser_script=parser_script,
        input_dir=job_input_dir,
        out_dir=job_out_dir,
        preset=preset,
        min_final_score=min_final_score,
        debug=debug,
    )

    stdout_path = job_meta_dir / "stdout.txt"
    stderr_path = job_meta_dir / "stderr.txt"
    cmd_path = job_meta_dir / "cmd.json"

    cmd_path.write_text(json.dumps(cmd, ensure_ascii=False, indent=2), encoding="utf-8")

    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )

    stdout_path.write_text(proc.stdout or "", encoding="utf-8")
    stderr_path.write_text(proc.stderr or "", encoding="utf-8")

    duration_sec = time.time() - t0

    manifest_path = job_out_dir / "manifest.json"
    parser_manifest = None
    if manifest_path.exists():
        try:
            parser_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            parser_manifest = None

    result = {
        "job_key": job.key,
        "year": job.year,
        "slug": job.slug,
        "page_count": job.page_count,
        "returncode": proc.returncode,
        "duration_sec": round(duration_sec, 3),
        "input_dir": str(job_input_dir),
        "out_dir": str(job_out_dir),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "manifest_path": str(manifest_path) if manifest_path.exists() else None,
        "ok": proc.returncode == 0,
        "parser_manifest": parser_manifest,
    }
    return result


def iter_files(path: Path) -> Iterable[Path]:
    if not path.exists():
        return []
    return [p for p in sorted(path.iterdir()) if p.is_file()]


def copy_merge_dir(src_dir: Path, dst_dir: Path) -> int:
    if not src_dir.exists():
        return 0

    _mkdir(dst_dir)
    copied = 0

    for src in iter_files(src_dir):
        dst = dst_dir / src.name
        if dst.exists():
            # Имена должны быть уникальны. Если уже есть файл с тем же именем,
            # пропускаем только если размер совпадает.
            if dst.stat().st_size != src.stat().st_size:
                raise RuntimeError(
                    f"Name collision with different file size: {src} -> {dst}"
                )
            continue
        shutil.copy2(src, dst)
        copied += 1

    return copied


def append_file_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    _mkdir(dst.parent)
    with dst.open("a", encoding="utf-8") as fout:
        text = src.read_text(encoding="utf-8", errors="ignore")
        if text:
            fout.write(text)
            if not text.endswith("\n"):
                fout.write("\n")


def merge_results(out_root: Path, results: list[dict]) -> dict:
    merged_pages_dir = out_root / "pages"
    merged_floorplans_dir = out_root / "floorplans"
    merged_debug_dir = out_root / "debug"
    merged_manifests_dir = out_root / "job_manifests"
    merged_logs_dir = out_root / "logs"

    _mkdir(merged_pages_dir)
    _mkdir(merged_floorplans_dir)
    _mkdir(merged_debug_dir)
    _mkdir(merged_manifests_dir)
    _mkdir(merged_logs_dir)

    total_pages_files = 0
    total_floorplans_files = 0
    total_debug_files = 0

    merged_debug_log = merged_logs_dir / "debug_log.merged.jsonl"

    for res in results:
        if not res.get("ok"):
            continue

        job_out_dir = Path(res["out_dir"])

        total_pages_files += copy_merge_dir(job_out_dir / "pages", merged_pages_dir)
        total_floorplans_files += copy_merge_dir(job_out_dir / "floorplans", merged_floorplans_dir)
        total_debug_files += copy_merge_dir(job_out_dir / "debug", merged_debug_dir)

        manifest_path = job_out_dir / "manifest.json"
        if manifest_path.exists():
            dst_manifest = merged_manifests_dir / f"{res['job_key']}.json"
            shutil.copy2(manifest_path, dst_manifest)

        debug_log_path = job_out_dir / "debug_log.jsonl"
        append_file_if_exists(debug_log_path, merged_debug_log)

    merged = {
        "merged_pages_files": total_pages_files,
        "merged_floorplans_files": total_floorplans_files,
        "merged_debug_files": total_debug_files,
        "merged_debug_log": str(merged_debug_log),
    }
    return merged


def build_summary(
    out_root: Path,
    catalog_dir: Path,
    pages_dir: Path,
    jobs: list[Job],
    results: list[dict],
    merged_info: dict,
    args: argparse.Namespace,
) -> dict:
    ok_jobs = [r for r in results if r.get("ok")]
    bad_jobs = [r for r in results if not r.get("ok")]

    summary = {
        "schema": "housesru_floorplans_parallel_run/v1",
        "catalog_dir": str(catalog_dir),
        "pages_dir": str(pages_dir),
        "out_dir": str(out_root),
        "workers": args.workers,
        "mode": args.mode,
        "preset": args.preset,
        "min_final_score": args.min_final_score,
        "debug": args.debug,
        "job_count": len(jobs),
        "ok_job_count": len(ok_jobs),
        "failed_job_count": len(bad_jobs),
        "jobs": results,
        "merged": merged_info,
    }
    return summary


def print_summary(jobs: list[Job], results: list[dict], merged_info: dict, out_root: Path) -> None:
    ok_jobs = [r for r in results if r.get("ok")]
    bad_jobs = [r for r in results if not r.get("ok")]

    print(f"out: {out_root}")
    print(f"jobs: {len(jobs)}")
    print(f"ok: {len(ok_jobs)}")
    print(f"failed: {len(bad_jobs)}")

    for r in results:
        status = "OK" if r.get("ok") else "FAIL"
        print(
            f"{status:4s} | {r['year']} | {r['slug']} | "
            f"pages={r['page_count']} | sec={r['duration_sec']}"
        )

    print("merged_pages_files:", merged_info.get("merged_pages_files", 0))
    print("merged_floorplans_files:", merged_info.get("merged_floorplans_files", 0))
    print("merged_debug_files:", merged_info.get("merged_debug_files", 0))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Parallel wrapper for parse_flippingbook_floorplans_debug_v4.py"
    )
    ap.add_argument(
        "--catalog-dir",
        type=Path,
        default=Path("data/housesru/all_pages_catalog"),
        help="Root directory of the unified pages catalog.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output root directory for the parallel run.",
    )
    ap.add_argument(
        "--parser-script",
        type=Path,
        default=Path("src/tools/parse_flippingbook_floorplans_debug_v4.py"),
        help="Path to the single-run parser script.",
    )
    ap.add_argument(
        "--python-exe",
        type=str,
        default=sys.executable,
        help="Python executable to run the parser.",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=max(1, min((os.cpu_count() or 4) // 2, 8)),
        help="Number of parallel workers.",
    )
    ap.add_argument(
        "--mode",
        choices=["symlink", "copy"],
        default="symlink",
        help="How to create per-job input directories.",
    )
    ap.add_argument(
        "--preset",
        type=str,
        default="balanced",
        help="Preset forwarded to the parser.",
    )
    ap.add_argument(
        "--min-final-score",
        type=float,
        default=7.0,
        help="Minimum final score forwarded to the parser.",
    )
    ap.add_argument(
        "--debug",
        action="store_true",
        help="Forward --debug to the parser.",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Do not delete existing per-job input/output directories.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = _repo_root()
    catalog_dir = args.catalog_dir.resolve()
    out_root = args.out.resolve()
    parser_script = (repo_root / args.parser_script).resolve() if not args.parser_script.is_absolute() else args.parser_script.resolve()

    pages_dir = catalog_dir / "pages"
    if not pages_dir.exists():
        raise FileNotFoundError(f"Pages directory not found: {pages_dir}")

    if not parser_script.exists():
        raise FileNotFoundError(f"Parser script not found: {parser_script}")

    _mkdir(out_root)

    jobs_input_root = out_root / "_job_inputs"
    jobs_out_root = out_root / "_job_outputs"
    _mkdir(jobs_input_root)
    _mkdir(jobs_out_root)

    jobs = discover_jobs(pages_dir)
    if not jobs:
        raise RuntimeError(f"No pages found in: {pages_dir}")

    print(f"catalog: {catalog_dir}")
    print(f"pages_dir: {pages_dir}")
    print(f"out: {out_root}")
    print(f"jobs: {len(jobs)}")
    print(f"workers: {args.workers}")
    print(f"mode: {args.mode}")
    print(f"parser: {parser_script}")

    results: list[dict] = []

    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [
            ex.submit(
                run_job,
                job,
                jobs_input_root,
                jobs_out_root,
                args.mode,
                args.resume,
                args.python_exe,
                parser_script,
                args.preset,
                args.min_final_score,
                args.debug,
                repo_root,
            )
            for job in jobs
        ]

        for fut in cf.as_completed(futures):
            res = fut.result()
            results.append(res)
            status = "OK" if res.get("ok") else "FAIL"
            print(
                f"{status:4s} | {res['year']} | {res['slug']} | "
                f"pages={res['page_count']} | sec={res['duration_sec']}"
            )

    results.sort(key=lambda x: (x["year"], x["slug"]))

    merged_info = merge_results(out_root, results)

    summary = build_summary(
        out_root=out_root,
        catalog_dir=catalog_dir,
        pages_dir=pages_dir,
        jobs=jobs,
        results=results,
        merged_info=merged_info,
        args=args,
    )

    summary_path = out_root / "parallel_run_manifest.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print_summary(jobs, results, merged_info, out_root)
    print("summary:", summary_path)


if __name__ == "__main__":
    main()