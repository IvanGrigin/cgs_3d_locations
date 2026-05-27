#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

try:
    from .pipeline_config import PlacementArtifacts
except ImportError:
    from pipeline_config import PlacementArtifacts


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: str | Path) -> Any:
    p = Path(path).expanduser().resolve()
    return json.loads(p.read_text(encoding="utf-8"))


def merge_room_spec_and_placements(room_json_path: str, placement_json_path: str, out_json_path: str) -> None:
    room_p = Path(room_json_path).expanduser().resolve()
    pl_p = Path(placement_json_path).expanduser().resolve()
    out_p = Path(out_json_path).expanduser().resolve()

    room = json.loads(room_p.read_text(encoding="utf-8"))
    pl = json.loads(pl_p.read_text(encoding="utf-8"))

    placements = pl.get("placements") or pl.get("items") or []
    if not isinstance(placements, list):
        placements = []

    scene = dict(room)
    scene["placements"] = placements

    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(scene, ensure_ascii=False, indent=2), encoding="utf-8")


def sync_objects_to_legacy_input(objects_path: Path, legacy_objects_json: str) -> None:
    dst = Path(legacy_objects_json).expanduser().resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(objects_path, dst)


def blender_outputs_for_mode(
    args: argparse.Namespace,
    run_dir: Path,
    mode: str,
    variant_suffix: str = "",
) -> tuple[Optional[str], Optional[str]]:
    suffix = f"_{variant_suffix}" if variant_suffix else ""

    if args.save_blend:
        p = Path(args.save_blend).expanduser().resolve()
        if p.suffix.lower() == ".blend":
            blend = str(p.with_name(f"{p.stem}_{mode}{suffix}.blend"))
        else:
            blend = str(p)
    else:
        blend = str((run_dir / f"scene_{mode}{suffix}.blend").resolve())

    if args.render:
        p = Path(args.render).expanduser().resolve()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            render = str(p.with_name(f"{p.stem}_{mode}{suffix}{p.suffix}"))
        else:
            render = str(p)
    else:
        render = str((run_dir / f"render_{mode}{suffix}.png").resolve())

    return blend, render


def _bool_arg(args: argparse.Namespace, name: str, default: bool = False) -> bool:
    return bool(getattr(args, name, default))


def _blender_output_mode(args: argparse.Namespace) -> str:
    mode = str(getattr(args, "blender_output", "render") or "render").strip().lower()
    return mode if mode in {"render", "gif", "both"} else "render"


def _should_keep_blend(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "save_blend", None) or _bool_arg(args, "keep_blend", False))


def _render_gif_from_frames(frame_dir: Path, out_gif: Path, fps: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise FileNotFoundError("ffmpeg не найден в PATH")  # pragma: no cover
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


def _render_gif_from_blend_isolated(
    *,
    cfg_runtime: dict[str, Any],
    args: argparse.Namespace,
    blend_path: Path,
    frame_dir: Path,
    gif_path: Path,
    frame_count: int,
    elevation_deg: float,
    fps: int,
) -> None:
    script = Path(__file__).resolve().parent / "tools" / "blend_to_orbit_gif.py"
    if not script.is_file():
        script = Path("src/tools/blend_to_orbit_gif.py").resolve()  # pragma: no cover
    yaw_step = 360.0 / max(int(frame_count), 1)
    duration_ms = max(1, int(round(1000.0 / max(int(fps), 1))))
    cmd = [
        sys.executable,
        str(script.resolve()),
        "--blend",
        str(blend_path.resolve()),
        "--frames-dir",
        str(frame_dir.resolve()),
        "--gif",
        str(gif_path.resolve()),
        "--width",
        str(int(getattr(args, "blender_gif_width", 768) or 768)),
        "--height",
        str(int(getattr(args, "blender_gif_height", 768) or 768)),
        "--samples",
        str(int(getattr(args, "blender_gif_samples", 16) or 16)),
        "--yaw-step",
        f"{yaw_step:.8f}",
        "--elevations",
        str(float(elevation_deg)),
        "--duration-ms",
        str(duration_ms),
        "--isolated-frames",
    ]
    if getattr(args, "blender", None):
        cmd += ["--blender", str(args.blender)]  # pragma: no cover
    if _bool_arg(args, "keep_blender_gif_frames", False):
        cmd.append("--keep-frames")

    print("▶ Изолированный GIF из сохранённого .blend:\n ", " ".join(cmd))
    subprocess.run(cmd, check=True)


def copy_tree_contents(src: Path, dst: Path) -> None:
    if not src.is_dir():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def normalize_json_artifact(
    cfg_runtime: dict[str, str],
    input_path: Path,
    output_path: Path,
    target: str,
) -> None:
    cmd = [
        sys.executable,
        cfg_runtime["NORMALIZE_JSON_SCRIPT"],
        "--input",
        str(input_path.expanduser().resolve()),
        "--output",
        str(output_path.expanduser().resolve()),
        "--target",
        target,
    ]
    print("▶ Нормализация JSON:\n ", " ".join(cmd))
    subprocess.run(cmd, check=True)


def build_normalized_scene_artifact(
    cfg_runtime: dict[str, str],
    room_path: str,
    placement_path: Path,
    output_path: Path,
) -> None:
    cmd = [
        sys.executable,
        cfg_runtime["NORMALIZE_JSON_SCRIPT"],
        "--room",
        str(Path(room_path).expanduser().resolve()),
        "--placement",
        str(placement_path.expanduser().resolve()),
        "--output",
        str(output_path.expanduser().resolve()),
        "--target",
        "scene",
    ]
    print("▶ Сборка канонического scene.v1:\n ", " ".join(cmd))
    subprocess.run(cmd, check=True)


def build_scene_artifacts(
    cfg_runtime: dict[str, Any],
    room_path: str,
    run_dir: Path,
    layout_mode: str,
    placement_out: Path,
    variant_suffix: str = "",
) -> PlacementArtifacts:
    suffix = f"_{variant_suffix}" if variant_suffix else ""

    normalized_placement_path = run_dir / f"placement{suffix}.v1.json"
    normalize_json_artifact(
        cfg_runtime=cfg_runtime,
        input_path=placement_out,
        output_path=normalized_placement_path,
        target="placement",
    )

    scene_v1_path = None
    scene_legacy_path = None

    if room_path.lower().endswith(".json"):
        scene_v1_path = run_dir / f"scene{suffix}.v1.json"
        build_normalized_scene_artifact(
            cfg_runtime=cfg_runtime,
            room_path=room_path,
            placement_path=placement_out,
            output_path=scene_v1_path,
        )

        scene_legacy_path = run_dir / f"scene_{layout_mode}{suffix}.json"
        merge_room_spec_and_placements(room_path, str(placement_out.resolve()), str(scene_legacy_path.resolve()))

    return PlacementArtifacts(
        placement_legacy=placement_out,
        placement_v1=normalized_placement_path,
        scene_v1=scene_v1_path,
        scene_legacy=scene_legacy_path,
    )


def choose_scene_for_render(artifacts: PlacementArtifacts) -> Path:
    if artifacts.scene_v1 and artifacts.scene_v1.is_file():
        return artifacts.scene_v1
    if artifacts.scene_legacy and artifacts.scene_legacy.is_file():
        return artifacts.scene_legacy
    raise RuntimeError("Нет доступного scene-артефакта для рендера")  # pragma: no cover


def run_blender_for_mode(
    cfg_runtime: dict[str, Any],
    args: argparse.Namespace,
    room_path: str,
    run_dir: Path,
    layout_mode: str,
    scene_json_path: Path,
    variant_suffix: str = "",
) -> dict[str, Any]:
    if not scene_json_path.is_file():
        raise RuntimeError(f"Scene JSON not found for Blender: {scene_json_path}")

    blend_out, render_out = blender_outputs_for_mode(args, run_dir, layout_mode, variant_suffix=variant_suffix)
    output_mode = _blender_output_mode(args)
    want_render = output_mode in {"render", "both"}
    want_gif = output_mode in {"gif", "both"}
    keep_blend = _should_keep_blend(args)
    build_report = str(Path(blend_out).with_suffix(".build_report.json").resolve()) if blend_out else None
    gif_path = None
    frame_dir = None
    if want_gif:
        gif_suffix = f"_{variant_suffix}" if variant_suffix else ""
        gif_path = str((run_dir / f"render_{layout_mode}{gif_suffix}.gif").resolve())
        frame_dir = run_dir / f"_frames_render_{layout_mode}{gif_suffix}"
        if frame_dir.exists():
            shutil.rmtree(frame_dir, ignore_errors=True)  # pragma: no cover
    cmd = [
        sys.executable,
        cfg_runtime["BLENDER_VIS_SCRIPT"],
        "--json",
        str(scene_json_path.resolve()),
    ]

    if args.blender:
        cmd += ["--blender", args.blender]
    if args.headless:
        cmd.append("--background")
    if getattr(args, "no_bbox_fallback", False):
        cmd.append("--no-bbox-fallback")
    if blend_out:
        cmd += ["--save-blend", str(Path(blend_out).resolve())]
    if build_report:
        cmd += ["--build-report", build_report]
    if want_render and render_out:
        cmd += ["--render", str(Path(render_out).resolve())]
    # GIF frames are rendered after this process exits, from the saved .blend.
    # Rendering them here keeps the heavy imported scene alive for the whole
    # orbit and can push Blender over system memory limits on large rooms.
    if not keep_blend:
        cmd.append("--no-pack-assets")

    print("▶ Запуск Blender-визуализатора:\n ", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        if want_gif and frame_dir and gif_path:
            blend_path = Path(blend_out).resolve() if blend_out else None
            if not blend_path or not blend_path.is_file():
                raise RuntimeError("GIF requested, but saved .blend was not produced")  # pragma: no cover
            _render_gif_from_blend_isolated(
                cfg_runtime=cfg_runtime,
                args=args,
                blend_path=blend_path,
                frame_dir=frame_dir,
                gif_path=Path(gif_path),
                frame_count=int(getattr(args, "blender_gif_frames", 36) or 36),
                elevation_deg=float(getattr(args, "blender_gif_elevation", 30.0) or 0.0),
                fps=int(getattr(args, "blender_gif_fps", 8) or 8),
            )
    finally:
        if blend_out and not keep_blend:
            try:
                Path(blend_out).unlink(missing_ok=True)
            except Exception:  # pragma: no cover
                pass  # pragma: no cover
    return {
        "blend_path": str(Path(blend_out).resolve()) if blend_out and Path(blend_out).is_file() else None,
        "render_path": str(Path(render_out).resolve()) if want_render and render_out else None,
        "gif_path": gif_path if want_gif else None,
        "build_report": build_report,
        "blender_output": output_mode,
        "keep_blend": keep_blend,
    }
