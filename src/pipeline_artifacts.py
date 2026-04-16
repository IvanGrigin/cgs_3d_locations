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
    raise RuntimeError("Нет доступного scene-артефакта для рендера")


def run_blender_for_mode(
    cfg_runtime: dict[str, Any],
    args: argparse.Namespace,
    room_path: str,
    run_dir: Path,
    layout_mode: str,
    scene_json_path: Path,
    variant_suffix: str = "",
) -> None:
    if not scene_json_path.is_file():
        raise RuntimeError(f"Scene JSON not found for Blender: {scene_json_path}")

    blend_out, render_out = blender_outputs_for_mode(args, run_dir, layout_mode, variant_suffix=variant_suffix)
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
    if render_out:
        cmd += ["--render", str(Path(render_out).resolve())]

    print("▶ Запуск Blender-визуализатора:\n ", " ".join(cmd))
    subprocess.run(cmd, check=True)
