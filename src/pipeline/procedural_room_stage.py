from __future__ import annotations

"""Compatibility wrapper for procedural dense room generation.

Import from this module in `src/run_pipeline.py`:

    from src.pipeline.procedural_room_stage import (
        add_procedural_room_arguments,
        maybe_apply_procedural_room_stage,
    )
"""

import json
from pathlib import Path
from typing import Any

from .procedural_rooms import apply_procedural_room_stage_to_artifacts


def add_procedural_room_arguments(parser: Any) -> None:
    group = parser.add_argument_group("Procedural rooms")
    group.add_argument(
        "--procedural-rooms",
        default="auto",
        choices=["auto", "always", "never"],
        help="Apply dense procedural generation for bedroom/living_room/corridor scenes.",
    )
    group.add_argument(
        "--procedural-room-types",
        default="bedroom,living_room,corridor",
        help="Comma-separated normalized room types for --procedural-rooms auto.",
    )
    group.add_argument(
        "--procedural-density",
        default="very_high",
        choices=["normal", "high", "very_high"],
        help="Object density for procedural room generation.",
    )
    group.add_argument(
        "--procedural-replace-existing",
        action="store_true",
        help="Remove existing furniture/decor placements before adding procedural room contents.",
    )
    group.add_argument(
        "--procedural-seed",
        type=int,
        default=None,
        help="Optional deterministic seed for procedural room generation.",
    )


def _split_types(raw: str | None) -> set[str]:
    return {part.strip().lower() for part in (raw or "").split(",") if part.strip()}


def _write_report_to_manifest(manifest_path: str | Path, report: dict[str, Any]) -> None:
    path = Path(manifest_path)
    if not path.exists():
        return
    try:
        with path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception:
        return
    manifest["procedural_room_stage"] = report
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
    except Exception:
        return


def maybe_apply_procedural_room_stage(
    *,
    args: Any,
    artifacts: Any,
    run_dir: str | Path,
    prompt_text: str,
    manifest_path: str | Path | None = None,
    tag: str = "base",
) -> Any:
    policy = getattr(args, "procedural_rooms", "auto")
    if policy in {"never", "off", "false"}:
        return artifacts

    updated_artifacts, report = apply_procedural_room_stage_to_artifacts(
        artifacts=artifacts,
        run_dir=run_dir,
        prompt=prompt_text,
        policy=policy,
        density=getattr(args, "procedural_density", "very_high"),
        replace_existing=bool(getattr(args, "procedural_replace_existing", False)),
        seed=getattr(args, "procedural_seed", None),
        tag=tag,
        enabled_room_types=_split_types(getattr(args, "procedural_room_types", "")),
    )

    if manifest_path is not None:
        _write_report_to_manifest(manifest_path, report)

    return updated_artifacts
