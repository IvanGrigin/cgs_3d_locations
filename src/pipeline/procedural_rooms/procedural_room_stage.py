from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path
from typing import Any

from .bedroom_generator import generate_bedroom
from .corridor_generator import generate_corridor
from .living_room_generator import generate_living_room
from .object_specs import normalize_density
from .room_context import build_room_context
from .validation import validate_placements


REMOVABLE_GENERATED_CATEGORIES = {
    "bed",
    "headboard",
    "nightstand",
    "wardrobe",
    "wardrobe_module",
    "dresser",
    "desk",
    "chair",
    "bench",
    "rug",
    "runner_rug",
    "table_lamp",
    "floor_lamp",
    "wall_light",
    "ceiling_light",
    "mirror",
    "wall_art",
    "plant",
    "decor_vase",
    "decor_books",
    "decor_box",
    "decor_tray",
    "pillow",
    "blanket",
    "sofa",
    "armchair",
    "coffee_table",
    "side_table",
    "tv_stand",
    "tv",
    "bookshelf",
    "console_table",
    "dining_table",
    "dining_chair",
    "ottoman",
    "cabinet",
    "shelf",
    "shoe_cabinet",
    "coat_rack",
    "wall_hooks",
    "umbrella_stand",
    "entry_bench",
}


SUPPORTED_ROOM_TYPES = {"bedroom", "living_room", "corridor"}


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raw = str(value).strip().lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _extract_placements(scene: dict[str, Any]) -> list[dict[str, Any]]:
    placements = scene.get("placements")
    if isinstance(placements, list):
        return placements
    items = scene.get("items")
    if isinstance(items, list):
        return items
    return []


def _is_removable_item(item: dict[str, Any]) -> bool:
    category = str(item.get("category", "")).strip().lower()
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    source = item.get("source") if isinstance(item.get("source"), dict) else {}

    if meta.get("procedural"):
        return True
    if category in REMOVABLE_GENERATED_CATEGORIES:
        return True
    if str(source.get("placement_source", "")).startswith("procedural_room_stage"):
        return True

    # Keep obvious room shell/opening/material/curtain objects.
    keep_tokens = ("wall", "floor", "ceiling", "door", "window", "curtain", "material", "baseboard")
    if any(token in category for token in keep_tokens):
        return False
    return False


def _filter_existing_placements(placements: list[dict[str, Any]], *, replace_existing: bool) -> tuple[list[dict[str, Any]], int]:
    if not replace_existing:
        return list(placements), 0
    kept: list[dict[str, Any]] = []
    removed = 0
    for item in placements:
        if _is_removable_item(item):
            removed += 1
        else:
            kept.append(item)
    return kept, removed


def _build_placement_v1(scene: dict[str, Any], placements: list[dict[str, Any]], *, mode: str) -> dict[str, Any]:
    meta = copy.deepcopy(scene.get("meta") if isinstance(scene.get("meta"), dict) else {})
    meta.update(
        {
            "placer": meta.get("placer", "procedural_room_stage"),
            "mode": mode,
            "procedural_room_stage": True,
        }
    )
    return {
        "schema": "placement.v1",
        "placer": "procedural_room_stage",
        "mode": mode,
        "placements": placements,
        "meta": meta,
    }


def _dispatch_generator(ctx: Any, density: str, seed: int | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    d = normalize_density(density)
    if ctx.room_type == "bedroom":
        return generate_bedroom(ctx, density=d, seed=seed)
    if ctx.room_type == "living_room":
        return generate_living_room(ctx, density=d, seed=seed)
    if ctx.room_type == "corridor":
        return generate_corridor(ctx, density=d, seed=seed)
    return [], {"status": "unsupported_room_type", "room_type": ctx.room_type}


def _counts_by_field(items: list[dict[str, Any]], field: str, *, meta: bool = False) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        source = item.get("meta") if meta and isinstance(item.get("meta"), dict) else item
        if not isinstance(source, dict):
            continue
        value = str(source.get(field) or "unknown").strip() or "unknown"
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def apply_procedural_room_stage(
    *,
    scene_json_path: str | Path,
    out_dir: str | Path,
    prompt: str = "",
    policy: str = "auto",
    density: str = "very_high",
    replace_existing: bool = True,
    seed: int | None = None,
    tag: str = "base",
    enabled_room_types: set[str] | None = None,
) -> dict[str, Any]:
    """Apply dense procedural generation to a scene JSON.

    Returns a report dictionary. The report always contains `skipped`; if skipped
    is false, it also contains `output_scene_json` and `output_placement_json`.

    This function intentionally has no dependency on the rest of the pipeline.
    It can be called from `run_pipeline.py` or tested as a standalone stage.
    """
    scene_json_path = Path(scene_json_path)
    out_dir = Path(out_dir)
    scene = read_json(scene_json_path)
    ctx = build_room_context(scene, prompt=prompt)
    enabled_room_types = enabled_room_types or SUPPORTED_ROOM_TYPES

    policy_raw = (policy or "auto").strip().lower()
    if policy_raw in {"never", "off", "false", "0", "none"}:
        return {
            "schema": "procedural_room_report/v1",
            "skipped": True,
            "reason": "policy_never",
            "room_type": ctx.room_type,
            "input_scene_json": str(scene_json_path),
        }

    if policy_raw == "auto" and ctx.room_type not in enabled_room_types:
        return {
            "schema": "procedural_room_report/v1",
            "skipped": True,
            "reason": "unsupported_room_type",
            "room_type": ctx.room_type,
            "enabled_room_types": sorted(enabled_room_types),
            "input_scene_json": str(scene_json_path),
        }

    base_placements = _extract_placements(scene)
    kept_placements, removed_existing_count = _filter_existing_placements(
        base_placements,
        replace_existing=replace_existing,
    )

    generated, generator_report = _dispatch_generator(ctx, density=density, seed=seed)
    final_placements = kept_placements + generated

    output_scene = copy.deepcopy(scene)
    output_scene["placements"] = final_placements
    output_scene.setdefault("meta", {})
    if isinstance(output_scene["meta"], dict):
        output_scene["meta"]["procedural_room_stage"] = {
            "enabled": True,
            "room_type": ctx.room_type,
            "density": normalize_density(density),
            "replace_existing": replace_existing,
            "generated_count": len(generated),
        }

    stem = f"procedural_room.{tag}"
    output_scene_json = out_dir / f"scene_{stem}.v1.json"
    output_placement_json = out_dir / f"placement_{stem}.v1.json"
    report_json = out_dir / f"procedural_room_report.{tag}.json"

    placement_v1 = _build_placement_v1(output_scene, final_placements, mode=f"procedural_room_{tag}")

    validation = validate_placements(ctx, final_placements)
    report = {
        "schema": "procedural_room_report/v1",
        "skipped": False,
        "policy": policy_raw,
        "room_type": ctx.room_type,
        "room_id": ctx.room_id,
        "area_m2": ctx.area_m2,
        "size_class": ctx.size_class,
        "density": normalize_density(density),
        "replace_existing": replace_existing,
        "input_scene_json": str(scene_json_path),
        "output_scene_json": str(output_scene_json),
        "output_placement_json": str(output_placement_json),
        "removed_existing_count": removed_existing_count,
        "kept_existing_count": len(kept_placements),
        "generated_count": len(generated),
        "counts_by_layer": _counts_by_field(generated, "density_layer", meta=True),
        "counts_by_category": _counts_by_field(generated, "category"),
        "final_count": len(final_placements),
        "generator": generator_report,
        "validation": validation,
        "warnings": [],
    }

    if not generated:
        report["warnings"].append("No procedural objects were generated.")

    write_json(output_scene_json, output_scene)
    write_json(output_placement_json, placement_v1)
    write_json(report_json, report)
    report["report_json"] = str(report_json)
    return report


def _get_attr(obj: Any, names: list[str]) -> Any:
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    if isinstance(obj, dict):
        for name in names:
            if name in obj:
                return obj[name]
    return None


def _updated_artifacts_copy(artifacts: Any, *, scene_path: str, placement_path: str) -> Any:
    """Return artifacts copy with scene/placement fields updated.

    The existing project may use a dataclass, a mutable object or a dict.
    This helper supports all three without importing project-specific classes.
    """
    scene_field_candidates = ["scene_v1", "scene_v1_json", "scene_json", "scene_path"]
    placement_field_candidates = ["placement_v1", "placement_v1_json", "placement_json", "placement_path"]

    if dataclasses.is_dataclass(artifacts):
        updates: dict[str, Any] = {}
        field_names = {field.name for field in dataclasses.fields(artifacts)}
        for name in scene_field_candidates:
            if name in field_names:
                updates[name] = Path(scene_path)
                break
        for name in placement_field_candidates:
            if name in field_names:
                updates[name] = Path(placement_path)
                break
        if updates:
            return dataclasses.replace(artifacts, **updates)

    if isinstance(artifacts, dict):
        copied = dict(artifacts)
        for name in scene_field_candidates:
            if name in copied:
                copied[name] = scene_path
                break
        for name in placement_field_candidates:
            if name in copied:
                copied[name] = placement_path
                break
        return copied

    copied = copy.copy(artifacts)
    for name in scene_field_candidates:
        if hasattr(copied, name):
            setattr(copied, name, scene_path)
            break
    for name in placement_field_candidates:
        if hasattr(copied, name):
            setattr(copied, name, placement_path)
            break
    return copied


def apply_procedural_room_stage_to_artifacts(
    *,
    artifacts: Any,
    run_dir: str | Path,
    prompt: str = "",
    policy: str = "auto",
    density: str = "very_high",
    replace_existing: bool = True,
    seed: int | None = None,
    tag: str = "base",
    enabled_room_types: set[str] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Pipeline-friendly wrapper.

    It expects an artifact object containing one of:
        - scene_v1
        - scene_v1_json
        - scene_json
        - scene_path
    and updates one of:
        - placement_v1
        - placement_v1_json
        - placement_json
        - placement_path

    It returns `(updated_artifacts, report)`.
    """
    scene_path = _get_attr(artifacts, ["scene_v1", "scene_v1_json", "scene_json", "scene_path"])
    if not scene_path:
        report = {
            "schema": "procedural_room_report/v1",
            "skipped": True,
            "reason": "artifact_has_no_scene_path",
        }
        return artifacts, report

    report = apply_procedural_room_stage(
        scene_json_path=scene_path,
        out_dir=run_dir,
        prompt=prompt,
        policy=policy,
        density=density,
        replace_existing=replace_existing,
        seed=seed,
        tag=tag,
        enabled_room_types=enabled_room_types,
    )

    if report.get("skipped"):
        return artifacts, report

    updated = _updated_artifacts_copy(
        artifacts,
        scene_path=str(report["output_scene_json"]),
        placement_path=str(report["output_placement_json"]),
    )
    return updated, report
