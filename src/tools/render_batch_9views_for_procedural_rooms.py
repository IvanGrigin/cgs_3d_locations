#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_BLENDER_CANDIDATES = [
    os.environ.get("BLENDER_PATH"),
    "/Applications/Blender.app/Contents/MacOS/Blender",
    "blender",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


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
    raise FileNotFoundError("Blender executable not found. Pass --blender or set BLENDER_PATH.")


def parse_int_list(raw: str) -> list[int]:
    out: list[int] = []
    for item in str(raw or "").split(","):
        item = item.strip()
        if not item:
            continue
        out.append(int(item))
    return out


def list_room_dirs(run_dir: Path, room_suffixes: list[int] | None = None) -> list[Path]:
    pattern = re.compile(r"^(bathroom|toilet)_(\d{2})$")
    rooms = []
    for entry in sorted(run_dir.iterdir()):
        if not entry.is_dir():
            continue
        match = pattern.match(entry.name)
        if not match:
            continue
        if room_suffixes is not None and int(match.group(2)) not in room_suffixes:
            continue
        rooms.append(entry)
    return rooms


def find_scene_json(room_dir: Path) -> Path | None:
    candidates = sorted(room_dir.glob("scene_procedural_room*.v1.json"))
    if candidates:
        return candidates[0]
    candidates = sorted(room_dir.glob("placement_procedural_room*.v1.json"))
    if candidates:
        return candidates[0]
    return None


def ensure_room_blend(
    *,
    blender: str,
    room_dir: Path,
    scene_json: Path,
    out_blend: Path,
    build_report: Path,
    force: bool = False,
) -> tuple[bool, str]:
    builder_script = _repo_root() / "src/Plasement/BlenderVisualizePlacement.py"
    if not force and out_blend.is_file():
        return False, f"skip_blend:{out_blend}"

    cmd = [
        sys.executable,
        str(builder_script),
        "--json",
        str(scene_json),
        "--blender",
        blender,
        "--background",
        "--save-blend",
        str(out_blend),
        "--build-report",
        str(build_report),
    ]
    out_blend.parent.mkdir(parents=True, exist_ok=True)
    build_report.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Blender build failed for {room_dir.name} (returncode={proc.returncode}); see {build_report} and Blender logs."
        )
    return True, str(out_blend)


def run_views(
    *,
    blends: list[Path],
    out_dir: Path,
    topview_hide_walls: bool,
    oblique_hide_walls: bool,
    render_script: str,
    resolution_x: int,
    resolution_y: int,
    topview_azimuths: str,
    oblique_azimuths: str,
    topview_elevation: float,
    oblique_elevation: float,
    topview_radius_mult: float,
    oblique_radius_mult: float,
    topview_lens: float,
    oblique_lens: float,
    skip_existing: bool,
    per_blend_out_dir: bool,
) -> int:
    if not blends:
        return 0
    cmd = [
        "python3",
        str(_repo_root() / "src/tools/render_blend_vlm_views.py"),
        "--render-script",
        render_script,
        "--out-dir",
        str(out_dir),
        "--resolution-x",
        str(int(resolution_x)),
        "--resolution-y",
        str(int(resolution_y)),
        "--topview-azimuths",
        topview_azimuths,
        "--oblique-azimuths",
        oblique_azimuths,
        "--topview-elevation",
        str(float(topview_elevation)),
        "--oblique-elevation",
        str(float(oblique_elevation)),
        "--topview-radius-mult",
        str(float(topview_radius_mult)),
        "--oblique-radius-mult",
        str(float(oblique_radius_mult)),
        "--topview-lens",
        str(float(topview_lens)),
        "--oblique-lens",
        str(float(oblique_lens)),
    ]
    if topview_hide_walls:
        cmd.append("--topview-hide-nearest-walls")
    if oblique_hide_walls:
        cmd.append("--oblique-hide-nearest-walls")
    if skip_existing:
        cmd.append("--skip-existing")
    if per_blend_out_dir:
        cmd.append("--per-blend-out-dir")
    for blend in blends:
        cmd.extend(["--blend", str(blend)])

    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"render_blend_vlm_views failed (returncode={proc.returncode})")
    return len(blends)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Render 9-view review images for generated procedural rooms.")
    p.add_argument("--run-dir", required=True, help="Directory with bathroom_XX / toilet_XX subfolders.")
    p.add_argument("--blender", default=None, help="Blender executable.")
    p.add_argument("--suffixes", default="", help="Comma-separated room suffixes to render, e.g. 01,03,05.")
    p.add_argument("--out-dir", default="out/bath_toilet_9views", help="Directory for render manifest.")
    p.add_argument(
        "--build-report-name",
        default="procedural_room_render.build_report.json",
        help="Scene build report filename inside each room folder.",
    )
    p.add_argument(
        "--blend-name",
        default="{room_name}.blend",
        help="Blend filename pattern (room_name placeholder supported).",
    )
    p.add_argument("--build-blend", action="store_true", help="Rebuild .blend files even if already present.")
    p.add_argument("--skip-existing", action="store_true", help="Skip rendering blend if all 9 images already exist.")
    p.add_argument("--topview-hide-nearest-walls", action="store_true", default=True, help="Hide nearest walls on topviews.")
    p.add_argument("--no-topview-hide-nearest-walls", dest="topview_hide_nearest_walls", action="store_false")
    p.add_argument("--oblique-hide-nearest-walls", action="store_true", default=False, help="Hide nearest walls on oblique views.")
    p.add_argument("--per-blend-out-dir", action="store_true", default=True, help="Save each room in <room>/vlm_review_views.")
    p.add_argument("--render-script", default="src/tools/render_saved_blend_top_view.py")
    p.add_argument("--dry-run", action="store_true", help="Print commands, do not execute Blender/render commands.")
    p.add_argument("--resolution-x", type=int, default=1400)
    p.add_argument("--resolution-y", type=int, default=1050)
    p.add_argument(
        "--topview-azimuths",
        default="0,72,144,216,288",
        help="Comma-separated topview azimuth angles.",
    )
    p.add_argument(
        "--oblique-azimuths",
        default="45,135,225,315",
        help="Comma-separated oblique azimuth angles.",
    )
    p.add_argument("--topview-elevation", type=float, default=82.0)
    p.add_argument("--oblique-elevation", type=float, default=60.0)
    p.add_argument("--topview-radius-mult", type=float, default=0.35)
    p.add_argument("--oblique-radius-mult", type=float, default=0.72)
    p.add_argument("--topview-lens", type=float, default=32.0)
    p.add_argument("--oblique-lens", type=float, default=28.0)
    return p


def main() -> None:
    args = build_parser().parse_args()
    blender = find_blender(args.blender)
    run_dir = Path(args.run_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    room_suffixes = parse_int_list(args.suffixes) if str(args.suffixes or "").strip() else None
    room_dirs = list_room_dirs(run_dir, room_suffixes=room_suffixes)
    if not room_dirs:
        raise SystemExit(f"No room folders found in {run_dir}")

    blends_for_render: list[Path] = []
    print(f"Rooms found: {[d.name for d in room_dirs]}")

    for room_dir in room_dirs:
        scene_json = find_scene_json(room_dir)
        if scene_json is None:
            print(f"[WARN] {room_dir.name}: no scene/placement json, skipped.")
            continue
        blend_name = str(args.blend_name).replace("{room_name}", room_dir.name)
        blend_path = room_dir / blend_name
        report_path = room_dir / args.build_report_name

        if args.build_blend or not blend_path.is_file():
            if args.dry_run:
                print(f"[DRY-RUN] build command skipped now: room={room_dir.name}, scene={scene_json.name}, blend={blend_path.name}")
                blends_for_render.append(blend_path)
                continue
            changed, _ = ensure_room_blend(
                blender=blender,
                room_dir=room_dir,
                scene_json=scene_json,
                out_blend=blend_path,
                build_report=report_path,
                force=args.build_blend,
            )
            if changed:
                print(f"[BUILD] {room_dir.name}: {blend_path.name}")
            else:
                print(f"[BUILD] {room_dir.name}: exists")

        if not blend_path.is_file():
            print(f"[WARN] {room_dir.name}: blend missing after build, skipped.")
            continue
        blends_for_render.append(blend_path)

    if not blends_for_render:
        raise SystemExit("No blends to render.")

    if args.dry_run:
        print(f"[DRY-RUN] render command would be executed for {len(blends_for_render)} blends:")
        for blend in blends_for_render:
            print(f"  {blend}")
        return

    run_views(
        blends=blends_for_render,
        out_dir=out_dir,
        topview_hide_walls=bool(args.topview_hide_nearest_walls),
        oblique_hide_walls=bool(args.oblique_hide_nearest_walls),
        render_script=str(
            (Path(args.render_script).expanduser().resolve())
            if args.render_script
            else (_repo_root() / "src/tools/render_saved_blend_top_view.py")
        ),
        resolution_x=int(args.resolution_x),
        resolution_y=int(args.resolution_y),
        topview_azimuths=str(args.topview_azimuths),
        oblique_azimuths=str(args.oblique_azimuths),
        topview_elevation=float(args.topview_elevation),
        oblique_elevation=float(args.oblique_elevation),
        topview_radius_mult=float(args.topview_radius_mult),
        oblique_radius_mult=float(args.oblique_radius_mult),
        topview_lens=float(args.topview_lens),
        oblique_lens=float(args.oblique_lens),
        skip_existing=bool(args.skip_existing),
        per_blend_out_dir=bool(args.per_blend_out_dir),
    )
    print(f"[OK] render jobs scheduled: {len(blends_for_render)}")


if __name__ == "__main__":
    main()
