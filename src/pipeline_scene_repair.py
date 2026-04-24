#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional, Tuple


def add_scene_repair_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scene-repair-model", default=None, help="Optional repair proposal checkpoint for scene.v1 postprocess")
    parser.add_argument("--scene-repair-selector-model", default=None, help="Optional corrupted-object selector checkpoint for scene repair")
    parser.add_argument("--scene-repair-device", default="auto", help="Device for scene repair inference: auto/cpu/cuda/mps")
    parser.add_argument("--scene-repair-max-passes", type=int, default=1, help="How many repair passes to run per scene")
    parser.add_argument("--scene-repair-candidate-limit", type=int, default=4, help="BBox candidate limit when selector is not used")
    parser.add_argument("--scene-repair-selector-topk", type=int, default=3, help="How many selector candidates to try per pass")
    parser.add_argument("--scene-repair-selector-candidate-limit", type=int, default=6, help="How many candidates the selector sees before top-k filtering")
    parser.add_argument("--scene-repair-selector-global-fallback-k", type=int, default=3, help="How many room-wide anomaly furniture candidates to merge into selector pool")


def scene_repair_enabled(args: argparse.Namespace) -> bool:
    return bool(str(getattr(args, "scene_repair_model", "") or "").strip())


def _repair_script_path() -> Path:
    return (Path(__file__).resolve().parent / "ml" / "infer" / "apply_repair_proposal_v1.py").resolve()


def _looks_like_scene_v1(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(isinstance(data, dict) and data.get("schema") == "scene.v1")


def _repair_summary(report: dict[str, Any]) -> dict[str, Any]:
    passes = report.get("passes") or []
    accepted_move_count = sum(len((p.get("accepted") or [])) for p in passes if isinstance(p, dict))
    return {
        "initial_bad_count": len(report.get("initial_bad_indices") or []),
        "final_bad_count": len(report.get("final_bad_indices") or []),
        "pass_count": len(passes),
        "accepted_move_count": int(accepted_move_count),
    }


def maybe_repair_scene_json(
    *,
    args: argparse.Namespace,
    scene_json_path: Path,
    run_dir: Path,
    tag: str,
) -> Tuple[Path, Optional[dict[str, Any]]]:
    scene_json_path = scene_json_path.expanduser().resolve()
    if not scene_repair_enabled(args):
        return scene_json_path, None
    if not scene_json_path.is_file():
        raise RuntimeError(f"Scene repair input does not exist: {scene_json_path}")

    info: dict[str, Any] = {
        "tag": str(tag),
        "input_scene_json": str(scene_json_path),
        "model": str(Path(str(args.scene_repair_model)).expanduser().resolve()),
        "selector_model": str(Path(str(args.scene_repair_selector_model)).expanduser().resolve()) if str(getattr(args, "scene_repair_selector_model", "") or "").strip() else None,
        "device": str(getattr(args, "scene_repair_device", "auto") or "auto"),
        "max_passes": int(getattr(args, "scene_repair_max_passes", 1) or 1),
        "candidate_limit": int(getattr(args, "scene_repair_candidate_limit", 4) or 4),
        "selector_topk": int(getattr(args, "scene_repair_selector_topk", 3) or 3),
        "selector_candidate_limit": int(getattr(args, "scene_repair_selector_candidate_limit", 6) or 6),
        "selector_global_fallback_k": int(getattr(args, "scene_repair_selector_global_fallback_k", 3) or 3),
    }
    if not _looks_like_scene_v1(scene_json_path):
        info["skipped_reason"] = "unsupported_schema"
        return scene_json_path, info

    out_scene = (run_dir / f"scene_repaired.{tag}.v1.json").resolve()
    report_json = (run_dir / f"scene_repair.{tag}.report.json").resolve()
    cmd = [
        sys.executable,
        str(_repair_script_path()),
        "--model",
        str(Path(str(args.scene_repair_model)).expanduser().resolve()),
        "--scene",
        str(scene_json_path),
        "--out",
        str(out_scene),
        "--report-json",
        str(report_json),
        "--max-passes",
        str(int(getattr(args, "scene_repair_max_passes", 1) or 1)),
        "--candidate-limit",
        str(int(getattr(args, "scene_repair_candidate_limit", 4) or 4)),
        "--device",
        str(getattr(args, "scene_repair_device", "auto") or "auto"),
    ]
    selector_model = str(getattr(args, "scene_repair_selector_model", "") or "").strip()
    if selector_model:
        cmd += [
            "--selector-model",
            str(Path(selector_model).expanduser().resolve()),
            "--selector-topk",
            str(int(getattr(args, "scene_repair_selector_topk", 3) or 3)),
            "--selector-candidate-limit",
            str(int(getattr(args, "scene_repair_selector_candidate_limit", 6) or 6)),
            "--selector-global-fallback-k",
            str(int(getattr(args, "scene_repair_selector_global_fallback_k", 3) or 3)),
        ]

    print(f"🩹 scene repair [{tag}]: {scene_json_path.name}")
    subprocess.run(cmd, check=True)
    report = json.loads(report_json.read_text(encoding="utf-8"))
    info["output_scene_json"] = str(out_scene)
    info["report_json"] = str(report_json)
    info["summary"] = _repair_summary(report)
    return out_scene, info
