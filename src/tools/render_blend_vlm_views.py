#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render a fixed VLM review image set for every .blend in a run directory."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_BLENDER_CANDIDATES = [
    os.environ.get("BLENDER_PATH"),
    "/Applications/Blender.app/Contents/MacOS/Blender",
    "blender",
]


def find_blender(explicit: str | None = None) -> str:
    candidates = ([explicit] if explicit else []) + DEFAULT_BLENDER_CANDIDATES
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path.resolve())
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise FileNotFoundError("Blender binary not found; pass --blender or set BLENDER_PATH")


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_float_list(raw: str) -> list[float]:
    out: list[float] = []
    for part in str(raw or "").replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    if not out:
        raise ValueError(f"Empty float list: {raw!r}")
    return out


def collect_blends(args: argparse.Namespace) -> list[Path]:
    blends: list[Path] = []
    for raw in args.blend or []:
        path = Path(raw).expanduser()
        if path.is_file():
            blends.append(path.resolve())
    for raw in args.blend_list or []:
        list_path = Path(raw).expanduser()
        if not list_path.is_file():
            continue
        for line in list_path.read_text(encoding="utf-8").splitlines():
            item = line.strip()
            if not item or item.startswith("#"):
                continue
            path = Path(item).expanduser()
            if path.is_file():
                blends.append(path.resolve())
    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
        blends.extend(sorted(run_dir.glob(str(args.blend_glob))))

    seen: set[str] = set()
    out: list[Path] = []
    for blend in blends:
        if not blend.is_file():
            continue
        key = str(blend.resolve())
        if key in seen:
            continue
        seen.add(key)
        out.append(blend.resolve())
    if not out:
        raise RuntimeError("No .blend files found")
    return out


def view_specs_for_blend(
    *,
    blend: Path,
    out_dir: Path,
    topview_azimuths: list[float],
    oblique_azimuths: list[float],
    topview_elevation: float,
    oblique_elevation: float,
    topview_radius_mult: float,
    oblique_radius_mult: float,
    topview_lens: float,
    oblique_lens: float,
) -> list[dict[str, Any]]:
    blend_dir = out_dir / blend.stem
    blend_dir.mkdir(parents=True, exist_ok=True)
    specs: list[dict[str, Any]] = []
    for idx, azimuth in enumerate(topview_azimuths):
        specs.append(
            {
                "name": f"topview_{idx:02d}_az{azimuth:g}",
                "out": str((blend_dir / f"topview_{idx:02d}_az{azimuth:g}.png").resolve()),
                "azimuth_deg": float(azimuth),
                "elevation_deg": float(topview_elevation),
                "radius_mult": float(topview_radius_mult),
                "lens": float(topview_lens),
            }
        )
    for azimuth in oblique_azimuths:
        specs.append(
            {
                "name": f"oblique_e{oblique_elevation:g}_az{azimuth:g}",
                "out": str((blend_dir / f"oblique_e{oblique_elevation:g}_az{azimuth:g}.png").resolve()),
                "azimuth_deg": float(azimuth),
                "elevation_deg": float(oblique_elevation),
                "radius_mult": float(oblique_radius_mult),
                "lens": float(oblique_lens),
                "hide_nearest_walls": True,
            }
        )
    return specs


def run_one_blend(
    *,
    blender: str,
    render_script: Path,
    blend: Path,
    specs: list[dict[str, Any]],
    out_dir: Path,
    resolution_x: int,
    resolution_y: int,
    skip_existing: bool,
    render_engine: str,
    per_blend_out_dir: bool,
    gpu_backend: str,
) -> dict[str, Any]:
    base_out_dir = blend.parent / "vlm_review_views" if per_blend_out_dir else out_dir
    blend_out_dir = base_out_dir / blend.stem
    blend_out_dir.mkdir(parents=True, exist_ok=True)
    specs_path = blend_out_dir / "view_specs.json"
    stdout_log = blend_out_dir / "render_stdout.log"
    stderr_log = blend_out_dir / "render_stderr.log"
    specs_path.write_text(json.dumps(specs, ensure_ascii=False, indent=2), encoding="utf-8")

    outputs = [Path(str(spec["out"])) for spec in specs if isinstance(spec, dict) and spec.get("out")]
    if skip_existing and outputs and all(path.is_file() and path.stat().st_size > 0 for path in outputs):
        return {
            "blend": str(blend),
            "status": "skipped_existing",
            "views": specs,
            "stdout_log": str(stdout_log),
            "stderr_log": str(stderr_log),
        }

    cmd = [
        blender,
        str(blend),
        "-b",
        "--python",
        str(render_script),
        "--",
        "--out",
        str(outputs[0] if outputs else (blend_out_dir / "view.png")),
        "--view-specs-json",
        str(specs_path),
        "--resolution-x",
        str(int(resolution_x)),
        "--resolution-y",
        str(int(resolution_y)),
        "--render-engine",
        str(render_engine),
    ]
    if str(gpu_backend or "").strip():
        cmd[1:1] = ["--gpu-backend", str(gpu_backend).strip()]
    started = datetime.now()
    t0 = time.perf_counter()
    # On macOS Blender 4.2 can crash during GPU backend initialization when
    # stdout/stderr are redirected from a parent Python process. Let Blender
    # inherit the terminal streams; per-view outputs and manifest still record
    # the reproducible command.
    stdout_log.write_text("Blender output was streamed to the parent process.\n", encoding="utf-8")
    stderr_log.write_text("Blender output was streamed to the parent process.\n", encoding="utf-8")
    proc = subprocess.run(cmd, text=True, check=False)
    duration_sec = round(time.perf_counter() - t0, 3)
    status = "ok" if proc.returncode == 0 and all(path.is_file() and path.stat().st_size > 0 for path in outputs) else "failed"
    return {
        "blend": str(blend),
        "status": status,
        "returncode": proc.returncode,
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "duration_sec": duration_sec,
        "command": cmd,
        "views": specs,
        "out_dir": str(blend_out_dir.resolve()),
        "stdout_log": str(stdout_log.resolve()),
        "stderr_log": str(stderr_log.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Render VLM review views for .blend files without opening Blender UI.")
    parser.add_argument("--run-dir", default=None, help="Pipeline run dir containing .blend files")
    parser.add_argument("--blend", action="append", default=None, help="Explicit .blend path; can be repeated")
    parser.add_argument("--blend-list", action="append", default=None, help="Text file with one .blend path per line")
    parser.add_argument("--blend-glob", default="*.blend")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--per-blend-out-dir",
        action="store_true",
        help="Write each blend's views to <blend parent>/vlm_review_views/<blend stem>.",
    )
    parser.add_argument("--blender", default=None)
    parser.add_argument("--render-script", default="src/tools/render_saved_blend_top_view.py")
    parser.add_argument("--resolution-x", type=int, default=1400)
    parser.add_argument("--resolution-y", type=int, default=1050)
    parser.add_argument("--topview-azimuths", default="0,72,144,216,288")
    parser.add_argument("--oblique-azimuths", default="45,135,225,315")
    parser.add_argument("--topview-elevation", type=float, default=82.0)
    parser.add_argument("--oblique-elevation", type=float, default=60.0)
    parser.add_argument("--topview-radius-mult", type=float, default=0.35)
    parser.add_argument("--oblique-radius-mult", type=float, default=0.72)
    parser.add_argument("--topview-lens", type=float, default=32.0)
    parser.add_argument("--oblique-lens", type=float, default=28.0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--render-engine", choices=["eevee", "workbench"], default="eevee")
    parser.add_argument("--gpu-backend", default="")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    blender = find_blender(args.blender)
    render_script = Path(args.render_script).expanduser().resolve()
    if not render_script.is_file():
        raise RuntimeError(f"Render script not found: {render_script}")

    blends = collect_blends(args)
    if args.limit is not None:
        blends = blends[: max(0, int(args.limit))]
    if args.per_blend_out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else Path("out/blend_cleanup_render_manifest").resolve()
    elif args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
    elif args.run_dir:
        out_dir = Path(args.run_dir).expanduser().resolve() / "vlm_review_views"
    else:
        out_dir = Path("out/vlm_review_views") / now_stamp()
        out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    topview_azimuths = parse_float_list(args.topview_azimuths)
    oblique_azimuths = parse_float_list(args.oblique_azimuths)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "blender": blender,
        "render_script": str(render_script),
        "out_dir": str(out_dir),
        "settings": {
            "resolution_x": int(args.resolution_x),
            "resolution_y": int(args.resolution_y),
            "topview_azimuths": topview_azimuths,
            "oblique_azimuths": oblique_azimuths,
            "topview_elevation": float(args.topview_elevation),
            "oblique_elevation": float(args.oblique_elevation),
            "topview_radius_mult": float(args.topview_radius_mult),
            "oblique_radius_mult": float(args.oblique_radius_mult),
            "topview_lens": float(args.topview_lens),
            "oblique_lens": float(args.oblique_lens),
            "render_engine": str(args.render_engine),
            "per_blend_out_dir": bool(args.per_blend_out_dir),
            "gpu_backend": str(args.gpu_backend),
        },
        "jobs": [],
    }

    for idx, blend in enumerate(blends, start=1):
        print(f"[{idx}/{len(blends)}] {blend}")
        specs = view_specs_for_blend(
            blend=blend,
            out_dir=out_dir,
            topview_azimuths=topview_azimuths,
            oblique_azimuths=oblique_azimuths,
            topview_elevation=float(args.topview_elevation),
            oblique_elevation=float(args.oblique_elevation),
            topview_radius_mult=float(args.topview_radius_mult),
            oblique_radius_mult=float(args.oblique_radius_mult),
            topview_lens=float(args.topview_lens),
            oblique_lens=float(args.oblique_lens),
        )
        if bool(args.per_blend_out_dir):
            specs = view_specs_for_blend(
                blend=blend,
                out_dir=blend.parent / "vlm_review_views",
                topview_azimuths=topview_azimuths,
                oblique_azimuths=oblique_azimuths,
                topview_elevation=float(args.topview_elevation),
                oblique_elevation=float(args.oblique_elevation),
                topview_radius_mult=float(args.topview_radius_mult),
                oblique_radius_mult=float(args.oblique_radius_mult),
                topview_lens=float(args.topview_lens),
                oblique_lens=float(args.oblique_lens),
            )
        job = run_one_blend(
            blender=blender,
            render_script=render_script,
            blend=blend,
            specs=specs,
            out_dir=out_dir,
            resolution_x=int(args.resolution_x),
            resolution_y=int(args.resolution_y),
            skip_existing=bool(args.skip_existing),
            render_engine=str(args.render_engine),
            per_blend_out_dir=bool(args.per_blend_out_dir),
            gpu_backend=str(args.gpu_backend),
        )
        manifest["jobs"].append(job)
        (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  -> {job['status']}")

    summary = {
        "total": len(manifest["jobs"]),
        "ok": sum(1 for job in manifest["jobs"] if job.get("status") == "ok"),
        "failed": sum(1 for job in manifest["jobs"] if job.get("status") == "failed"),
        "skipped_existing": sum(1 for job in manifest["jobs"] if job.get("status") == "skipped_existing"),
        "manifest": str((out_dir / "manifest.json").resolve()),
    }
    manifest["summary"] = summary
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
