#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


THIS_DIR = Path(__file__).resolve().parent
PLACEMENT_SCRIPT = THIS_DIR / "BlenderVisualizePlacement.py"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _infer_original_json(supplier_json: Path) -> Path:
    candidates = [
        supplier_json.parent / "scene.v1.json",
        supplier_json.parent / "scene_original.v1.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Не найден original scene json рядом с {supplier_json}")


def _collect_replaced_ids(supplier_data: dict) -> list[str]:
    items = supplier_data.get("placements")
    if not isinstance(items, list):
        items = supplier_data.get("items")
    if not isinstance(items, list):
        return []

    ids: list[str] = []
    for item in items:
        item_id = str(item.get("id") or "").strip()
        if not item_id:
            continue
        meta = item.get("meta") or {}
        source = item.get("source") or {}
        if meta.get("supplier_binding_applied") or source.get("supplier_replaced"):
            ids.append(item_id)
    return ids


def _run_visualizer(
    *,
    json_path: Path,
    blender: str | None,
    background: bool,
    save_blend: Path | None = None,
    highlight_ids: list[str] | None = None,
    hide_room_shell: bool = False,
    turntable_render_dir: Path | None = None,
    turntable_frames: int = 24,
    turntable_elevation_deg: float = 30.0,
    no_pack_assets: bool = False,
) -> None:
    cmd = [sys.executable, str(PLACEMENT_SCRIPT), "--json", str(json_path.resolve()), "--no-bbox-fallback"]
    if blender:
        cmd += ["--blender", blender]
    if background:
        cmd += ["--background"]
    if save_blend:
        cmd += ["--save-blend", str(save_blend.resolve())]
    if highlight_ids:
        cmd += ["--highlight-item-ids", ",".join(highlight_ids)]
    if hide_room_shell:
        cmd += ["--hide-room-shell"]
    if turntable_render_dir:
        cmd += ["--turntable-render-dir", str(turntable_render_dir.resolve())]
        cmd += ["--turntable-frames", str(int(turntable_frames))]
        cmd += ["--turntable-elevation-deg", str(float(turntable_elevation_deg))]
    if no_pack_assets:
        cmd += ["--no-pack-assets"]
    subprocess.run(cmd, check=True)


def _render_gif_from_frames(frame_dir: Path, out_gif: Path, fps: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise FileNotFoundError("ffmpeg не найден в PATH")
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    palette = frame_dir / "palette.png"
    frame_pattern = str((frame_dir / "frame_%03d.png").resolve())

    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-framerate",
            str(int(fps)),
            "-i",
            frame_pattern,
            "-vf",
            "palettegen=stats_mode=diff",
            str(palette.resolve()),
        ],
        check=True,
    )
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-framerate",
            str(int(fps)),
            "-i",
            frame_pattern,
            "-i",
            str(palette.resolve()),
            "-lavfi",
            "paletteuse=dither=bayer:bayer_scale=3",
            str(out_gif.resolve()),
        ],
        check=True,
    )


def _parse_elevations(raw: str) -> list[int]:
    out: list[int] = []
    for chunk in str(raw or "").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        out.append(int(round(float(chunk))))
    return out or [0, 30, 45]


def main() -> int:
    ap = argparse.ArgumentParser(description="Build original/supplier blend scenes and interior GIFs with replaced-item bbox overlay.")
    ap.add_argument("--supplier-json", required=True, help="scene_supplier.v1.json")
    ap.add_argument("--original-json", default=None, help="scene.v1.json; if omitted, inferred from supplier run dir")
    ap.add_argument("--blender", default=None, help="Path to Blender binary")
    ap.add_argument("--background", action="store_true", help="Run Blender in background")
    ap.add_argument("--turntable-frames", type=int, default=24)
    ap.add_argument("--gif-fps", type=int, default=4)
    ap.add_argument("--turntable-elevations", default="0,30,45", help="Comma-separated elevation angles in degrees")
    ap.add_argument("--keep-frame-dirs", action="store_true")
    args = ap.parse_args()

    supplier_json = Path(args.supplier_json).expanduser().resolve()
    if not supplier_json.is_file():
        raise FileNotFoundError(f"Не найден supplier json: {supplier_json}")
    original_json = Path(args.original_json).expanduser().resolve() if args.original_json else _infer_original_json(supplier_json)
    if not original_json.is_file():
        raise FileNotFoundError(f"Не найден original json: {original_json}")

    supplier_data = _read_json(supplier_json)
    replaced_ids = _collect_replaced_ids(supplier_data)
    if not replaced_ids:
        raise RuntimeError("В supplier scene не найдены реальные заменённые item ids")

    out_dir = supplier_json.parent
    original_blend = out_dir / "scene_original.marked.blend"
    supplier_blend = out_dir / "scene_supplier.marked.blend"
    elevations = _parse_elevations(args.turntable_elevations)

    _run_visualizer(
        json_path=original_json,
        blender=args.blender,
        background=bool(args.background),
        save_blend=original_blend,
        highlight_ids=replaced_ids,
        no_pack_assets=True,
    )
    _run_visualizer(
        json_path=supplier_json,
        blender=args.blender,
        background=bool(args.background),
        save_blend=supplier_blend,
        highlight_ids=replaced_ids,
        no_pack_assets=True,
    )

    print(f"original_blend = {original_blend}")
    print(f"supplier_blend = {supplier_blend}")
    for elevation in elevations:
        suffix = f"elev_{int(elevation):02d}"
        original_frames = out_dir / f"_frames_original_interior_{suffix}"
        supplier_frames = out_dir / f"_frames_supplier_interior_{suffix}"
        original_gif = out_dir / f"room_original.interior.{suffix}.gif"
        supplier_gif = out_dir / f"room_supplier.interior.{suffix}.gif"

        _run_visualizer(
            json_path=original_json,
            blender=args.blender,
            background=True,
            highlight_ids=replaced_ids,
            hide_room_shell=True,
            turntable_render_dir=original_frames,
            turntable_frames=int(args.turntable_frames),
            turntable_elevation_deg=float(elevation),
            no_pack_assets=True,
        )
        _run_visualizer(
            json_path=supplier_json,
            blender=args.blender,
            background=True,
            highlight_ids=replaced_ids,
            hide_room_shell=True,
            turntable_render_dir=supplier_frames,
            turntable_frames=int(args.turntable_frames),
            turntable_elevation_deg=float(elevation),
            no_pack_assets=True,
        )

        _render_gif_from_frames(original_frames, original_gif, int(args.gif_fps))
        _render_gif_from_frames(supplier_frames, supplier_gif, int(args.gif_fps))

        if not args.keep_frame_dirs:
            shutil.rmtree(original_frames, ignore_errors=True)
            shutil.rmtree(supplier_frames, ignore_errors=True)

        print(f"original_gif_{suffix} = {original_gif}")
        print(f"supplier_gif_{suffix} = {supplier_gif}")
    print(f"highlighted_ids = {','.join(replaced_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
