#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate a procedural-only room placement corpus and export distributions.

This tool intentionally runs only the lightweight procedural room stage. It
does not call supplier matching, Trellis, Blender, or any LLM stage.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


ROOM_TEMPLATES: dict[str, list[tuple[str, float, float, float]]] = {
    "toilet": [
        ("tiny", 0.92, 1.28, 2.7),
        ("compact", 1.05, 1.45, 2.7),
        ("standard", 1.25, 1.75, 2.7),
        ("wide", 1.45, 1.85, 2.7),
        ("long", 1.10, 2.15, 2.7),
    ],
    "bathroom": [
        ("tiny_shower", 1.55, 1.95, 2.7),
        ("compact", 1.80, 2.20, 2.7),
        ("standard", 2.40, 2.20, 2.7),
        ("wide", 2.75, 2.60, 2.7),
        ("family", 3.10, 3.00, 2.7),
    ],
    "bedroom": [
        ("compact", 2.95, 3.05, 2.8),
        ("student", 3.10, 3.20, 2.8),
        ("standard", 4.20, 3.10, 2.8),
        ("wide", 4.60, 3.70, 2.8),
        ("large", 5.20, 4.20, 2.8),
    ],
    "living_room": [
        ("compact", 3.20, 3.10, 2.8),
        ("standard", 5.00, 3.80, 2.8),
        ("wide", 5.60, 4.20, 2.8),
        ("long", 6.20, 3.60, 2.8),
        ("large", 6.40, 5.00, 2.8),
    ],
    "corridor": [
        ("narrow_short", 1.05, 3.20, 2.7),
        ("standard", 1.25, 5.20, 2.7),
        ("wide", 1.55, 4.60, 2.7),
        ("long", 1.35, 6.60, 2.7),
        ("entry_hall", 2.10, 3.40, 2.7),
    ],
}

PROMPTS = {
    "toilet": "туалет",
    "bathroom": "ванная",
    "bedroom": "спальня",
    "living_room": "гостиная",
    "corridor": "коридор",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_imports() -> None:
    root = _repo_root()
    src = root / "src"
    for candidate in (root, src):
        value = str(candidate)
        if value not in sys.path:
            sys.path.insert(0, value)


_ensure_imports()

try:
    from src.pipeline.procedural_rooms.procedural_room_stage import apply_procedural_room_stage
except ModuleNotFoundError:
    from pipeline.procedural_rooms.procedural_room_stage import apply_procedural_room_stage


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def rectangle_room(
    *,
    room_id: str,
    room_type: str,
    width: float,
    depth: float,
    height: float,
    seed_index: int,
) -> dict[str, Any]:
    door_width = 0.70 if room_type == "toilet" else 0.80 if room_type in {"bathroom", "bedroom"} else 0.90
    door_s = max(0.10, min(width - door_width - 0.10, 0.18 + 0.11 * (seed_index % 5)))
    windows: list[dict[str, Any]] = []
    if room_type in {"bathroom", "bedroom", "living_room"}:
        window_width = min(width * 0.45, 1.40 if room_type == "living_room" else 1.05)
        window_s = max(0.15, min(width - window_width - 0.15, width * (0.28 + 0.08 * (seed_index % 4))))
        windows.append({"wall_id": "w2", "s": round(window_s, 3), "width": round(window_width, 3)})
    if room_type == "corridor" and seed_index % 2 == 0:
        windows = []

    doors = [{"wall_id": "w0", "s": round(door_s, 3), "width": round(door_width, 3)}]
    if room_type == "corridor":
        doors.append({"wall_id": "w2", "s": round(max(0.10, width - door_width - 0.20), 3), "width": round(door_width, 3)})

    return {
        "id": room_id,
        "type_hint": room_type,
        "height_m": height,
        "floor_polygon": [
            {"x": 0.0, "y": 0.0},
            {"x": round(width, 3), "y": 0.0},
            {"x": round(width, 3), "y": round(depth, 3)},
            {"x": 0.0, "y": round(depth, 3)},
        ],
        "openings": {
            "doors": doors,
            "windows": windows,
        },
    }


def build_scene(room: dict[str, Any], *, corpus_id: str, seed: int, density: str, template_name: str) -> dict[str, Any]:
    return {
        "schema": "scene.v1",
        "room": room,
        "placements": [],
        "items": [],
        "meta": {
            "placer": "procedural_room_stage",
            "creator": "procedural_room_stage",
            "generator": "procedural_room_distribution_corpus",
            "mode": "procedural_room_distribution",
            "procedural_distribution_corpus": {
                "corpus_id": corpus_id,
                "seed": seed,
                "density": density,
                "template_name": template_name,
            },
        },
    }


def iter_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    room_types = args.room_types or list(ROOM_TEMPLATES)
    densities = args.densities or ["normal", "high", "very_high"]
    jobs: list[dict[str, Any]] = []
    for room_type in room_types:
        templates = ROOM_TEMPLATES[room_type]
        for template_index, (template_name, width, depth, height) in enumerate(templates):
            for density_index, density in enumerate(densities):
                for seed_index in range(args.seeds_per_template):
                    seed = args.base_seed + len(jobs) * 17 + seed_index + density_index * 101 + template_index * 1009
                    job_id = f"{room_type}_{template_index:02d}_{template_name}_{density}_seed_{seed_index:03d}"
                    room = rectangle_room(
                        room_id=f"{job_id}_room",
                        room_type=room_type,
                        width=width,
                        depth=depth,
                        height=height,
                        seed_index=seed_index + density_index * 10,
                    )
                    jobs.append(
                        {
                            "id": job_id,
                            "room_type": room_type,
                            "template_name": template_name,
                            "density": density,
                            "seed": seed,
                            "room": room,
                            "prompt": PROMPTS[room_type],
                        }
                    )
                    if args.max_runs and len(jobs) >= args.max_runs:
                        return jobs
    return jobs


def run_analysis(args: argparse.Namespace, corpus_dir: Path, analysis_run_name: str) -> Path:
    analysis_out_root = Path(args.analysis_out_root)
    out_dir = analysis_out_root / analysis_run_name
    if out_dir.exists():
        suffix = datetime.now().strftime("%H%M%S")
        analysis_run_name = f"{analysis_run_name}_{suffix}"
        out_dir = analysis_out_root / analysis_run_name
    cmd = [
        sys.executable,
        str(_repo_root() / "src" / "tools" / "export_scene_placement_distributions.py"),
        "--roots",
        str(corpus_dir),
        "--out-root",
        str(analysis_out_root),
        "--run-name",
        analysis_run_name,
        "--grid-sizes",
        *[str(x) for x in args.grid_sizes],
        "--min-objects-per-group",
        str(args.min_objects_per_group),
    ]
    if args.no_plots:
        cmd.append("--no-plots")
    subprocess.run(cmd, cwd=str(_repo_root()), check=True)
    return out_dir


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate many procedural room layouts and export placement distribution CSV/JSONL files."
    )
    parser.add_argument("--out-dir", default=None, help="Corpus output directory with per-run scene/placement JSON files.")
    parser.add_argument("--analysis-out-root", default="out/layout_distribution_analysis")
    parser.add_argument("--analysis-run-name", default=None)
    parser.add_argument("--room-types", nargs="+", choices=sorted(ROOM_TEMPLATES), default=None)
    parser.add_argument("--densities", nargs="+", choices=["normal", "high", "very_high"], default=None)
    parser.add_argument("--seeds-per-template", type=int, default=8)
    parser.add_argument("--base-seed", type=int, default=20260516)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--replace-existing", action="store_true", default=True)
    parser.add_argument("--policy", default="always", choices=["auto", "always", "never"])
    parser.add_argument("--grid-sizes", nargs="+", type=int, default=[5, 10, 20, 40])
    parser.add_argument("--min-objects-per-group", type=int, default=20)
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--no-plots", action="store_true", default=True)
    return parser


def main() -> None:
    args = build_cli().parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    corpus_dir = Path(args.out_dir or f"out/procedural_room_distribution/procedural_many_{timestamp}")
    corpus_dir.mkdir(parents=True, exist_ok=True)
    analysis_run_name = args.analysis_run_name or f"procedural_room_stage_many_{timestamp}"

    jobs = iter_jobs(args)
    reports: list[dict[str, Any]] = []
    print(f"[procedural_distribution] jobs={len(jobs)} out_dir={corpus_dir}")
    for index, job in enumerate(jobs, start=1):
        job_dir = corpus_dir / job["room_type"] / job["id"]
        scene_path = job_dir / "input_scene_from_room.v1.json"
        scene = build_scene(
            job["room"],
            corpus_id=job["id"],
            seed=int(job["seed"]),
            density=str(job["density"]),
            template_name=str(job["template_name"]),
        )
        write_json(scene_path, scene)
        report = apply_procedural_room_stage(
            scene_json_path=scene_path,
            out_dir=job_dir,
            prompt=str(job["prompt"]),
            policy=str(args.policy),
            density=str(job["density"]),
            replace_existing=bool(args.replace_existing),
            seed=int(job["seed"]),
            tag="distribution",
        )
        report["corpus_job_id"] = job["id"]
        report["template_name"] = job["template_name"]
        report["seed"] = job["seed"]
        reports.append(report)
        if index == 1 or index % 50 == 0 or index == len(jobs):
            print(f"[procedural_distribution] generated {index}/{len(jobs)}")

    counts_by_room_type = Counter(str(report.get("room_type") or "unknown") for report in reports)
    generated_objects = sum(int(report.get("final_count") or report.get("generated_count") or 0) for report in reports)
    analysis_dir = None
    if not args.skip_analysis:
        print("[procedural_distribution] exporting distribution tables")
        analysis_dir = run_analysis(args, corpus_dir, analysis_run_name)

    summary = {
        "schema": "procedural_room_distribution_corpus_report/v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "corpus_dir": str(corpus_dir),
        "analysis_dir": str(analysis_dir) if analysis_dir else None,
        "job_count": len(jobs),
        "object_count_from_reports": generated_objects,
        "counts_by_room_type": dict(sorted(counts_by_room_type.items())),
        "reports": reports,
    }
    report_path = corpus_dir / "procedural_room_distribution_corpus_report.json"
    write_json(report_path, summary)
    print(
        json.dumps(
            {
                "corpus_dir": str(corpus_dir),
                "analysis_dir": str(analysis_dir) if analysis_dir else None,
                "job_count": len(jobs),
                "object_count_from_reports": generated_objects,
                "report_json": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
