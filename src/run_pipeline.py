#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/run_pipeline.py

from __future__ import annotations

import argparse
import json
import math
import os
import re
import secrets
import shutil
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

try:
    from .acquire_supplier_bindings_assets import acquire_assets_for_bindings_json
    from .apply_supplier_bindings import apply_supplier_bindings_to_json
    from .layout_targets import create_layout_selection_stub_artifacts
    from .supplier_layout_matcher import (
        build_bindings_with_candidates,
        load_supplier_catalog_json,
        read_json as read_supplier_matcher_json,
    )
    from .suppliers.room_design_spec_builder import build_room_design_spec
    from .suppliers.supplier_scene_consistency import apply_supplier_scene_consistency
    from .suppliers.supplier_variant_validator import main as supplier_variant_validator_main
    from .pipeline_artifacts import (
        blender_outputs_for_mode,
        build_scene_artifacts,
        choose_scene_for_render,
        normalize_json_artifact,
        run_blender_for_mode,
    )
    from .pipeline_scene_repair import add_scene_repair_arguments, maybe_repair_scene_json
    from .pipeline.curtain_stage import (
        discover_supplier_curtain_models,
        discover_curtain_models,
        load_curtain_catalog,
        write_json as write_curtain_json,
    )
    from .pipeline.infinigen_scene_improvers import (
        apply_curtains_to_scene,
        normalize_chandelier_positions_in_scene,
        repair_furniture_intersections_in_scene,
    )
    from .pipeline.kitchen_stage import apply_kitchen_stage_to_artifacts
    from .pipeline.flooring_stage import apply_flooring_to_scene, run_flooring_selection, write_json as write_flooring_json
    from .pipeline.wall_stage import apply_wall_material_to_scene_with_catalog, run_wall_selection, write_json as write_wall_json
    from .supplier_replacement_report import write_supplier_replacement_reports
    from .pipeline_config import (
        DEFAULT_LEGO_GENERATION_PRESETS,
        DEFAULT_PATHS_CONFIG,
        ModeOutputs,
        PLACER_SPECS,
        PlacementArtifacts,
        apply_config_defaults,
        build_runtime_paths,
        load_yaml,
        make_mode_run_dir,
        parse_modes,
        project_root_from_config,
        read_prompt_from_args,
        write_json,
    )
    from .pipeline_runners import (
        execute_placer,
        resolve_lego_generation_params,
        run_choose_stage,
        run_lego_generate_from_scratch,
    )
    from .style_profiles import attach_style_hint_to_room_json
    from .style_prompt_analyzer import analyze_prompt_to_style_profile
except ImportError:
    from acquire_supplier_bindings_assets import acquire_assets_for_bindings_json
    from apply_supplier_bindings import apply_supplier_bindings_to_json
    from layout_targets import create_layout_selection_stub_artifacts
    from supplier_layout_matcher import (
        build_bindings_with_candidates,
        load_supplier_catalog_json,
        read_json as read_supplier_matcher_json,
    )
    from suppliers.room_design_spec_builder import build_room_design_spec
    from suppliers.supplier_scene_consistency import apply_supplier_scene_consistency
    from suppliers.supplier_variant_validator import main as supplier_variant_validator_main
    from pipeline_artifacts import (
        blender_outputs_for_mode,
        build_scene_artifacts,
        choose_scene_for_render,
        normalize_json_artifact,
        run_blender_for_mode,
    )
    from pipeline_scene_repair import add_scene_repair_arguments, maybe_repair_scene_json
    from pipeline.curtain_stage import (
        discover_supplier_curtain_models,
        discover_curtain_models,
        load_curtain_catalog,
        write_json as write_curtain_json,
    )
    from pipeline.infinigen_scene_improvers import (
        apply_curtains_to_scene,
        normalize_chandelier_positions_in_scene,
        repair_furniture_intersections_in_scene,
    )
    from pipeline.kitchen_stage import apply_kitchen_stage_to_artifacts
    from pipeline.flooring_stage import apply_flooring_to_scene, run_flooring_selection, write_json as write_flooring_json
    from pipeline.wall_stage import apply_wall_material_to_scene_with_catalog, run_wall_selection, write_json as write_wall_json
    from supplier_replacement_report import write_supplier_replacement_reports
    from pipeline_config import (
        DEFAULT_LEGO_GENERATION_PRESETS,
        DEFAULT_PATHS_CONFIG,
        ModeOutputs,
        PLACER_SPECS,
        PlacementArtifacts,
        apply_config_defaults,
        build_runtime_paths,
        load_yaml,
        make_mode_run_dir,
        parse_modes,
        project_root_from_config,
        read_prompt_from_args,
        write_json,
    )
    from pipeline_runners import (
        execute_placer,
        resolve_lego_generation_params,
        run_choose_stage,
        run_lego_generate_from_scratch,
    )
    from style_profiles import attach_style_hint_to_room_json
    from style_prompt_analyzer import analyze_prompt_to_style_profile


def _build_layout_selection_stub_for_artifacts(
    *,
    artifacts: PlacementArtifacts,
    run_dir: Path,
    prefix: str = "",
) -> dict[str, str]:
    source_json_path = artifacts.scene_v1 if artifacts.scene_v1 and artifacts.scene_v1.is_file() else artifacts.placement_v1
    return create_layout_selection_stub_artifacts(
        source_json_path=source_json_path,
        run_dir=run_dir,
        prefix=prefix,
    )


def _to_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace("\u00a0", " ").replace(",", ".")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _polygon_area(points: list[dict[str, Any]]) -> float | None:
    if len(points) < 3:
        return None
    total = 0.0
    for idx, point in enumerate(points):
        nxt = points[(idx + 1) % len(points)]
        x1 = _to_float(point.get("x"))
        y1 = _to_float(point.get("y", point.get("z")))
        x2 = _to_float(nxt.get("x"))
        y2 = _to_float(nxt.get("y", nxt.get("z")))
        if x1 is None or y1 is None or x2 is None or y2 is None:
            return None
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def _polygon_perimeter(points: list[dict[str, Any]]) -> float | None:
    if len(points) < 2:
        return None
    total = 0.0
    for idx, point in enumerate(points):
        nxt = points[(idx + 1) % len(points)]
        x1 = _to_float(point.get("x"))
        y1 = _to_float(point.get("y", point.get("z")))
        x2 = _to_float(nxt.get("x"))
        y2 = _to_float(nxt.get("y", nxt.get("z")))
        if x1 is None or y1 is None or x2 is None or y2 is None:
            return None
        total += math.hypot(x2 - x1, y2 - y1)
    return total


def _room_surface_metrics(room_path: Path) -> dict[str, Any]:
    data = json.loads(room_path.read_text(encoding="utf-8"))
    room = data.get("room") if isinstance(data, dict) else {}
    if not isinstance(room, dict):
        room = {}

    polygon = room.get("floor_polygon") or room.get("floor_polygon_xz") or []
    if not isinstance(polygon, list):
        polygon = []

    width = _to_float(room.get("width_m"))
    depth = _to_float(room.get("depth_m"))
    floor_area = _to_float(room.get("area_m2"))
    if floor_area is None:
        floor_area = _polygon_area(polygon)
    if floor_area is None and width is not None and depth is not None:
        floor_area = width * depth

    perimeter = _polygon_perimeter(polygon)
    if perimeter is None and width is not None and depth is not None:
        perimeter = 2.0 * (width + depth)

    height = _to_float(room.get("ceiling_height_m")) or _to_float(room.get("ceiling_height")) or 2.7
    gross_wall_area = perimeter * height if perimeter is not None and height is not None else None
    opening_area = 0.0
    for group_name in ("doors", "windows", "openings"):
        group = room.get(group_name) or []
        if not isinstance(group, list):
            continue
        for item in group:
            if not isinstance(item, dict):
                continue
            item_width = _to_float(item.get("width"))
            item_height = _to_float(item.get("height"))
            if item_width is not None and item_height is not None:
                opening_area += max(0.0, item_width * item_height)

    wall_area = gross_wall_area
    if wall_area is not None:
        wall_area = max(0.0, wall_area - opening_area)

    return {
        "room_json": str(room_path.resolve()),
        "floor_area_m2": floor_area,
        "wall_area_m2": wall_area,
        "gross_wall_area_m2": gross_wall_area,
        "opening_area_m2": opening_area,
        "perimeter_m": perimeter,
        "ceiling_height_m": height,
    }


def _raw_property(material: dict[str, Any], keys: tuple[str, ...]) -> Any:
    raw = material.get("raw_properties")
    if not isinstance(raw, dict):
        return None
    for key in keys:
        if key in raw:
            return raw.get(key)
    return None


def _floor_package_area_m2(material: dict[str, Any]) -> float | None:
    return (
        _to_float(material.get("package_area_m2"))
        or _to_float(_raw_property(material, ("Площадь упаковки", "Площадь в упаковке", "Площадь")))
    )


def _wall_roll_area_m2(material: dict[str, Any]) -> float | None:
    area = _to_float(_raw_property(material, ("Площадь рулона",)))
    if area is not None:
        return area
    width = _to_float(material.get("width_m")) or _to_float(material.get("width_cm")) or _to_float(_raw_property(material, ("Ширина рулона",)))
    length = _to_float(material.get("length_m")) or _to_float(_raw_property(material, ("Длина рулона",)))
    if width is None or length is None:
        return None
    if width > 5.0:
        width = width / 100.0
    return width * length


def _surface_pricing_item(
    *,
    target_id: str,
    category: str,
    semantic_group: str,
    material: dict[str, Any],
    coverage_area_m2: float | None,
    package_area_m2: float | None,
    quantity_unit: str,
) -> dict[str, Any] | None:
    if not isinstance(material, dict) or not material:
        return None
    unit_price = _to_float(material.get("price"))
    quantity = None
    if coverage_area_m2 is not None and package_area_m2 is not None and package_area_m2 > 0:
        quantity = int(math.ceil(coverage_area_m2 / package_area_m2))
    total = unit_price * quantity if unit_price is not None and quantity is not None else None
    return {
        "target_id": target_id,
        "category": category,
        "semantic_group": semantic_group,
        "replacement_policy": "surface_material",
        "pricing_bucket": "surface_material",
        "price_status": "estimated" if total is not None else "pending",
        "currency": material.get("price_currency") or "RUB",
        "final_price_value": round(total, 2) if total is not None else None,
        "final_asset_source": material.get("source") or "material_catalog",
        "sku": material.get("sku"),
        "name": material.get("name"),
        "brand": material.get("brand"),
        "product_url": material.get("product_url"),
        "material_type": material.get("material_type"),
        "quantity": quantity,
        "quantity_unit": quantity_unit,
        "unit_price_value": unit_price,
        "package_area_m2": package_area_m2,
        "coverage_area_m2": round(coverage_area_m2, 3) if coverage_area_m2 is not None else None,
    }


def _write_surface_material_pricing(
    *,
    run_dir: Path,
    room_path: Path,
    flooring_info: dict[str, Any] | None,
    wall_info: dict[str, Any] | None,
    pricing_stub_json: str | None,
    suffix: str,
) -> dict[str, Any] | None:
    metrics = _room_surface_metrics(room_path)
    items: list[dict[str, Any]] = []
    sources: dict[str, str] = {}

    if flooring_info and flooring_info.get("selection_json"):
        selection_path = Path(str(flooring_info["selection_json"])).expanduser().resolve()
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        material = selection.get("selected_material") or {}
        item = _surface_pricing_item(
            target_id="surface_floor",
            category="floor_covering",
            semantic_group="flooring",
            material=material,
            coverage_area_m2=_to_float(metrics.get("floor_area_m2")),
            package_area_m2=_floor_package_area_m2(material),
            quantity_unit="package",
        )
        if item is not None:
            items.append(item)
            sources["flooring_selection_json"] = str(selection_path)

    if wall_info and wall_info.get("selection_json"):
        selection_path = Path(str(wall_info["selection_json"])).expanduser().resolve()
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        material = selection.get("selected_material") or {}
        item = _surface_pricing_item(
            target_id="surface_walls",
            category="wall_covering",
            semantic_group="wallpaper",
            material=material,
            coverage_area_m2=_to_float(metrics.get("wall_area_m2")),
            package_area_m2=_wall_roll_area_m2(material),
            quantity_unit="roll",
        )
        if item is not None:
            items.append(item)
            sources["wall_material_selection_json"] = str(selection_path)

    if not items:
        return None

    total = sum(float(item["final_price_value"]) for item in items if item.get("final_price_value") is not None)
    path = run_dir / f"surface_materials.pricing{suffix}.json"
    artifact = {
        "schema": "surface_materials_pricing/v1",
        "room_metrics": metrics,
        "sources": sources,
        "totals": {
            "currency": "RUB",
            "surface_material_total_value": round(total, 2),
            "surface_material_item_count": len(items),
        },
        "items": items,
    }
    write_json(path, artifact)

    if pricing_stub_json:
        _merge_surface_materials_into_pricing_stub(Path(pricing_stub_json), artifact, path)

    return {
        "pricing_json": str(path.resolve()),
        "surface_material_total_value": artifact["totals"]["surface_material_total_value"],
        "surface_material_item_count": len(items),
    }


def _merge_surface_materials_into_pricing_stub(
    pricing_stub_path: Path,
    surface_pricing: dict[str, Any],
    surface_pricing_path: Path,
) -> None:
    pricing_stub_path = pricing_stub_path.expanduser().resolve()
    if not pricing_stub_path.is_file():
        return
    data = json.loads(pricing_stub_path.read_text(encoding="utf-8"))
    items = data.setdefault("items", [])
    if not isinstance(items, list):
        return

    surface_items = surface_pricing.get("items") or []
    surface_ids = {item.get("target_id") for item in surface_items if isinstance(item, dict)}
    items[:] = [item for item in items if not (isinstance(item, dict) and item.get("target_id") in surface_ids)]
    items.extend(surface_items)

    meta = data.setdefault("meta", {})
    if isinstance(meta, dict):
        meta["scene_item_count"] = len(items)
        meta["surface_material_count"] = len(surface_items)
        meta["surface_material_pricing_json"] = str(surface_pricing_path.resolve())
    totals = data.setdefault("totals", {})
    if isinstance(totals, dict):
        totals["surface_material_total_value"] = surface_pricing.get("totals", {}).get("surface_material_total_value")
    write_json(pricing_stub_path, data)



def _maybe_apply_layout_postprocess(
    *,
    args: argparse.Namespace,
    scene_json_path: Path,
    run_dir: Path,
    tag: str,
) -> tuple[Path, dict[str, Any] | None]:
    if not (bool(getattr(args, "normalize_chandeliers", False)) or bool(getattr(args, "repair_furniture_overlaps", False))):
        return scene_json_path, None
    scene_json_path = scene_json_path.expanduser().resolve()
    if not scene_json_path.is_file():
        return scene_json_path, {"skipped_reason": "scene_json_missing", "input_scene_json": str(scene_json_path)}
    data = json.loads(scene_json_path.read_text(encoding="utf-8"))
    info: dict[str, Any] = {"input_scene_json": str(scene_json_path), "tag": tag}
    if bool(getattr(args, "normalize_chandeliers", False)):
        data, chandelier_info = normalize_chandelier_positions_in_scene(data)
        info["normalize_chandeliers"] = chandelier_info
    if bool(getattr(args, "repair_furniture_overlaps", False)):
        data, repair_info = repair_furniture_intersections_in_scene(data)
        info["repair_furniture_overlaps"] = repair_info
    out_path = (run_dir / f"{scene_json_path.stem}.layout_post.v1.json").resolve()
    write_json(out_path, data)
    info["output_scene_json"] = str(out_path)
    return out_path, info


def _maybe_apply_kitchen_stage(
    *,
    args: argparse.Namespace,
    artifacts: PlacementArtifacts,
    run_dir: Path,
    room_path: str,
    prompt_text: str,
    suffix: str,
) -> tuple[PlacementArtifacts, dict[str, Any] | None]:
    policy = str(getattr(args, "kitchens", "auto") or "auto").strip().lower()
    if policy in {"off", "false", "0", "no", "none"}:
        policy = "never"
    if policy in {"on", "true", "1", "yes"}:
        policy = "always"
    if policy not in {"auto", "always", "never"}:
        policy = "auto"

    material_catalog = Path(str(getattr(args, "kitchen_material_catalog", "") or "")).expanduser()
    if not material_catalog.is_absolute():
        material_catalog = (Path.cwd() / material_catalog).resolve()
    appliance_catalog = Path(str(getattr(args, "kitchen_appliance_catalog", "") or "")).expanduser()
    if str(getattr(args, "kitchen_appliance_catalog", "") or "").strip():
        if not appliance_catalog.is_absolute():
            appliance_catalog = (Path.cwd() / appliance_catalog).resolve()
    else:
        appliance_catalog = None

    print("🍳 kitchen: procedural гарнитур")
    next_artifacts, info = apply_kitchen_stage_to_artifacts(
        artifacts=artifacts,
        run_dir=run_dir,
        room_json_path=Path(room_path).expanduser().resolve(),
        material_catalog=material_catalog,
        appliance_catalog=appliance_catalog,
        prompt_text=prompt_text,
        mode=str(getattr(args, "kitchen_selection_mode", "optimal") or "optimal"),
        policy=policy,
        suffix=suffix,
        dining_policy=str(getattr(args, "kitchen_dining", "auto") or "auto"),
        accessories_policy=str(getattr(args, "kitchen_accessories", "auto") or "auto"),
        accessory_llm_settings={
            "provider": str(getattr(args, "kitchen_accessory_llm_provider", "none") or "none"),
            "ollama_url": str(getattr(args, "kitchen_accessory_ollama_url", "") or "http://127.0.0.1:11434"),
            "ollama_model": str(getattr(args, "kitchen_accessory_ollama_model", "") or "gpt-oss:20b"),
            "ollama_timeout": int(getattr(args, "kitchen_accessory_ollama_timeout", 180) or 180),
            "ollama_temperature": float(getattr(args, "kitchen_accessory_ollama_temperature", 0.2) or 0.2),
            "ollama_num_ctx": int(getattr(args, "kitchen_accessory_ollama_num_ctx", 8192) or 8192),
            "ollama_think": str(getattr(args, "kitchen_accessory_ollama_think", "low") or "low"),
        },
        kitchen_llm_settings={
            "provider": str(getattr(args, "kitchen_llm_provider", "none") or "none"),
            "ollama_url": str(getattr(args, "kitchen_ollama_url", "") or "http://127.0.0.1:11434"),
            "ollama_model": str(getattr(args, "kitchen_ollama_model", "") or "gpt-oss:20b"),
            "ollama_timeout": int(getattr(args, "kitchen_ollama_timeout", 180) or 180),
            "ollama_temperature": float(getattr(args, "kitchen_ollama_temperature", 0.1) or 0.1),
            "ollama_num_ctx": int(getattr(args, "kitchen_ollama_num_ctx", 8192) or 8192),
            "ollama_think": str(getattr(args, "kitchen_ollama_think", "low") or "low"),
        },
    )
    if info and info.get("replacement_count"):
        print(f"🍳 kitchen generated: {info.get('replacement_count')} assembly item(s)")
    elif info:
        print(f"⏭ kitchen: пропуск ({info.get('skipped_reason')})")
    return next_artifacts, info


def _is_fatal_disk_full_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "remote_disk_full" in text
        or "no space left on device" in text
        or "disk full" in text
    )


def _apply_supplier_bindings_for_artifacts(
    *,
    artifacts: PlacementArtifacts,
    run_dir: Path,
    bindings_json_path: Path,
    require_local_asset: bool,
    variant_suffix: str = "",
) -> dict[str, Any]:
    suffix = f".{variant_suffix.strip('.')}" if str(variant_suffix or "").strip() else ""
    supplier_placement_v1 = run_dir / f"placement_supplier{suffix}.v1.json"
    apply_supplier_bindings_to_json(
        input_json_path=artifacts.placement_v1,
        bindings_json_path=bindings_json_path,
        output_json_path=supplier_placement_v1,
        require_local_asset=require_local_asset,
    )

    supplier_scene_v1 = None
    if artifacts.scene_v1 and artifacts.scene_v1.is_file():
        supplier_scene_v1 = run_dir / f"scene_supplier{suffix}.v1.json"
        apply_supplier_bindings_to_json(
            input_json_path=artifacts.scene_v1,
            bindings_json_path=bindings_json_path,
            output_json_path=supplier_scene_v1,
            require_local_asset=require_local_asset,
        )

    supplier_data = json.loads(supplier_placement_v1.read_text(encoding="utf-8"))
    supplier_summary = ((supplier_data.get("meta") or {}).get("supplier_binding_summary") or {})
    return {
        "bindings_json": str(bindings_json_path.resolve()),
        "placement_v1": str(supplier_placement_v1.resolve()),
        "scene_v1": str(supplier_scene_v1.resolve()) if supplier_scene_v1 else None,
        "require_local_asset": bool(require_local_asset),
        "summary": supplier_summary,
    }


def _write_supplier_replacement_reports_for_artifacts(
    *,
    run_dir: Path,
    bindings_json_path: Path,
    supplier_info: dict[str, Any],
    variant_suffix: str = "",
    blender_build_report_path: str | Path | None = None,
) -> dict[str, Any]:
    scene_v1 = supplier_info.get("scene_v1")
    suffix = f".{variant_suffix.strip('.')}" if str(variant_suffix or "").strip() else ""
    return write_supplier_replacement_reports(
        bindings_json_path=bindings_json_path,
        run_dir=run_dir,
        supplier_scene_json_path=str(scene_v1) if scene_v1 else None,
        blender_build_report_path=blender_build_report_path,
        short_filename=f"supplier_replacements{suffix}.short.md",
        extended_filename=f"supplier_replacements{suffix}.full.md",
        html_filename=f"supplier_replacements{suffix}.html",
        summary_filename=f"supplier_replacements{suffix}.summary.json",
        mode=str(variant_suffix or "").strip(".") or None,
    )


def _load_json_if_file(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = Path(path).expanduser()
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _variant_total_price(summary: dict[str, Any], warnings: list[str], mode: str) -> float | None:
    total = 0.0
    found = False
    missing = 0
    for target in summary.get("targets") or []:
        if not isinstance(target, dict):
            continue
        if not target.get("chosen_candidate_id"):
            continue
        price = target.get("price")
        if price is None:
            missing += 1
            continue
        try:
            total += float(price)
            found = True
        except Exception:
            missing += 1
    if missing:
        warnings.append(f"{mode}: {missing} selected targets have no numeric price; total_price_estimate is partial.")
    return round(total, 2) if found else None


def _write_supplier_variants_comparison(run_dir: Path, variants: dict[str, Any]) -> Path | None:
    if not variants:
        return None
    warnings: list[str] = []
    modes = list(variants.keys())
    variant_payload: dict[str, Any] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for mode, info in variants.items():
        reports = info.get("reports") or {}
        rebind = info.get("rebind") or {}
        summary = _load_json_if_file(reports.get("summary_json")) or {}
        summaries[mode] = summary
        local_warnings = list(summary.get("warnings") or []) if isinstance(summary.get("warnings"), list) else []
        warnings.extend(f"{mode}: {x}" for x in local_warnings)
        variant_payload[mode] = {
            "bindings_path": info.get("bindings"),
            "scene_path": rebind.get("scene_v1"),
            "report_path": reports.get("html"),
            "summary_path": reports.get("summary_json"),
            "blend_path": (info.get("blender") or {}).get("blend_path"),
            "blend_exists": bool((info.get("blender") or {}).get("blend_exists")),
            "blender_status": (info.get("blender") or {}).get("blender_status") or "skipped",
            "blender_error": (info.get("blender") or {}).get("blender_error"),
            "counts": summary.get("counts") or {},
            "score_averages": summary.get("score_averages") or {},
            "total_price_estimate": _variant_total_price(summary, warnings, mode) if summary else None,
        }

    target_ids: set[str] = set()
    target_by_mode: dict[str, dict[str, dict[str, Any]]] = {}
    for mode, summary in summaries.items():
        targets = {}
        for target in summary.get("targets") or []:
            if not isinstance(target, dict):
                continue
            target_id = str(target.get("target_id") or "").strip()
            if not target_id:
                continue
            targets[target_id] = target
            target_ids.add(target_id)
        target_by_mode[mode] = targets

    differences: list[dict[str, Any]] = []
    for target_id in sorted(target_ids):
        row: dict[str, Any] = {"target_id": target_id, "category": None}
        ids: list[str] = []
        prices: dict[str, float | None] = {}
        scores: dict[str, float | None] = {}
        for mode in modes:
            target = target_by_mode.get(mode, {}).get(target_id) or {}
            if row["category"] is None:
                row["category"] = target.get("category")
            candidate_id = target.get("chosen_candidate_id")
            row[f"{mode}_candidate_id"] = candidate_id
            ids.append(str(candidate_id or ""))
            prices[mode] = target.get("price") if isinstance(target.get("price"), (int, float)) else None
            scores[mode] = target.get("final_score") if isinstance(target.get("final_score"), (int, float)) else None
        row["all_modes_same"] = len(set(ids)) <= 1
        if "best_match" in prices and "cheapest" in prices and prices["best_match"] is not None and prices["cheapest"] is not None:
            row["price_delta_best_vs_cheapest"] = round(float(prices["best_match"]) - float(prices["cheapest"]), 2)
        else:
            row["price_delta_best_vs_cheapest"] = None
        if "best_match" in scores and "cheapest" in scores and scores["best_match"] is not None and scores["cheapest"] is not None:
            row["score_delta_best_vs_cheapest"] = round(float(scores["best_match"]) - float(scores["cheapest"]), 6)
        else:
            row["score_delta_best_vs_cheapest"] = None
        differences.append(row)

    out = {
        "modes": modes,
        "variants": variant_payload,
        "target_differences": differences,
        "warnings": warnings,
    }
    out_path = run_dir / "supplier_variants.comparison.json"
    write_json(out_path, out)
    return out_path


def _write_supplier_variants_manifest(
    *,
    run_dir: Path,
    modes: list[str],
    variants: dict[str, Any],
    room_design_spec_path: str | None,
    comparison_json: str | None,
    validation_json: str | None = None,
    warnings: list[str] | None = None,
) -> Path:
    artifacts: dict[str, Any] = {}
    for mode in modes:
        info = variants.get(mode) or {}
        reports = info.get("reports") or {}
        rebind = info.get("rebind") or {}
        blender = info.get("blender") or {}
        artifacts[mode] = {
            "bindings": info.get("initial_bindings") or info.get("bindings"),
            "consistent_bindings": info.get("consistent_bindings"),
            "assets_bindings": info.get("bindings"),
            "scene_json": rebind.get("scene_v1"),
            "blend": blender.get("blend_path"),
            "html_report": reports.get("html"),
            "summary_json": reports.get("summary_json"),
        }
    out = {
        "run_dir": str(run_dir.resolve()),
        "room_design_spec_path": room_design_spec_path,
        "modes": modes,
        "artifacts": artifacts,
        "comparison_json": comparison_json,
        "validation_json": validation_json,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "warnings": warnings or [],
    }
    out_path = run_dir / "supplier_variants.manifest.json"
    write_json(out_path, out)
    return out_path


def _parse_supplier_build_modes(raw: str | None, selection_modes: list[str]) -> list[str]:
    requested = _parse_supplier_selection_modes(raw)
    if requested:
        allowed = set(selection_modes)
        return [mode for mode in requested if mode in allowed]
    return list(selection_modes)


def _mark_supplier_blender_skipped(variants: dict[str, Any], reason: str = "skip_blender") -> None:
    for info in variants.values():
        info["blender"] = {
            "blend_path": None,
            "blend_exists": False,
            "blender_status": "skipped",
            "blender_error": reason,
            "build_report": None,
        }


def _run_supplier_blender_variants(
    *,
    cfg_runtime: dict[str, Any],
    args: argparse.Namespace,
    run_dir: Path,
    layout_mode: str,
    effective_room_path: str,
    variants: dict[str, Any],
) -> None:
    build_modes = _parse_supplier_build_modes(getattr(args, "supplier_build_modes", None), list(variants.keys()))
    for mode_name, info in variants.items():
        if mode_name not in build_modes:
            info["blender"] = {
                "blend_path": None,
                "blend_exists": False,
                "blender_status": "skipped",
                "blender_error": "not_in_supplier_build_modes",
                "build_report": None,
            }
            continue
        scene_v1 = (info.get("rebind") or {}).get("scene_v1")
        if not scene_v1 or not Path(str(scene_v1)).expanduser().is_file():
            info["blender"] = {
                "blend_path": None,
                "blend_exists": False,
                "blender_status": "skipped",
                "blender_error": "scene_json_missing",
                "build_report": None,
            }
            continue
        try:
            result = run_blender_for_mode(
                cfg_runtime=cfg_runtime,
                args=args,
                room_path=effective_room_path,
                run_dir=run_dir,
                layout_mode=layout_mode,
                scene_json_path=Path(str(scene_v1)).expanduser().resolve(),
                variant_suffix=f"supplier.{mode_name}",
            )
            blend_path = result.get("blend_path")
            info["blender"] = {
                "blend_path": blend_path,
                "blend_exists": bool(blend_path and Path(str(blend_path)).is_file()),
                "blender_status": "ok",
                "blender_error": None,
                "build_report": result.get("build_report"),
                "render_path": result.get("render_path"),
                "gif_path": result.get("gif_path"),
                "blender_output": result.get("blender_output"),
                "keep_blend": result.get("keep_blend"),
            }
        except Exception as exc:
            info["blender"] = {
                "blend_path": str(Path(blender_outputs_for_mode(args, run_dir, layout_mode, variant_suffix=f"supplier.{mode_name}")[0] or "").resolve()),
                "blend_exists": False,
                "blender_status": "failed",
                "blender_error": f"{type(exc).__name__}: {exc}",
                "build_report": None,
            }


def _refresh_supplier_reports_after_blender(
    *,
    run_dir: Path,
    variants: dict[str, Any],
) -> None:
    for mode_name, info in variants.items():
        blender = info.get("blender") or {}
        build_report = blender.get("build_report")
        if not build_report or not Path(str(build_report)).is_file():
            continue
        bindings_path = info.get("bindings")
        if not bindings_path:
            continue
        reports = _write_supplier_replacement_reports_for_artifacts(
            run_dir=run_dir,
            bindings_json_path=Path(str(bindings_path)).expanduser().resolve(),
            supplier_info=info.get("rebind") or {},
            variant_suffix=mode_name,
            blender_build_report_path=build_report,
        )
        info["reports"] = reports


def _validate_supplier_variants_if_requested(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    variants: dict[str, Any],
) -> tuple[str | None, list[str]]:
    if not bool(getattr(args, "validate_supplier_variants", False)):
        return None, []
    bindings_paths = [str((info.get("bindings") or "")) for info in variants.values() if str(info.get("bindings") or "").strip()]
    if len(bindings_paths) <= 1:
        return None, ["supplier variant validation skipped: less than two bindings files"]
    out_path = run_dir / "supplier_variants.validation.json"
    argv: list[str] = []
    for path in bindings_paths:
        argv.extend(["--bindings", path])
    argv.extend(["--out", str(out_path.resolve())])
    code = supplier_variant_validator_main(argv)
    warnings: list[str] = []
    if out_path.is_file():
        data = json.loads(out_path.read_text(encoding="utf-8"))
        warnings.extend(str(x) for x in (data.get("warnings") or []))
        errors = [str(x) for x in (data.get("errors") or [])]
        warnings.extend(f"validator_error:{x}" for x in errors)
        if code != 0 and errors:
            raise RuntimeError(f"supplier variant validation failed: {errors[:3]}")
    return str(out_path.resolve()), warnings


def _finalize_supplier_variant_artifacts(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    manifest: dict[str, Any],
    manifest_path: Path,
    variants: dict[str, Any],
) -> dict[str, Any]:
    comparison_path = _write_supplier_variants_comparison(run_dir, variants)
    validation_path: str | None = None
    warnings: list[str] = []
    if comparison_path:
        comparison_data = json.loads(comparison_path.read_text(encoding="utf-8"))
        warnings.extend(str(x) for x in (comparison_data.get("warnings") or []))
    validation_path, validation_warnings = _validate_supplier_variants_if_requested(
        args=args,
        run_dir=run_dir,
        variants=variants,
    )
    warnings.extend(validation_warnings)
    supplier_manifest_path = _write_supplier_variants_manifest(
        run_dir=run_dir,
        modes=list(variants.keys()),
        variants=variants,
        room_design_spec_path=manifest.get("room_design_spec_json"),
        comparison_json=str(comparison_path.resolve()) if comparison_path else None,
        validation_json=validation_path,
        warnings=warnings,
    )
    manifest["supplier_variants"] = variants
    manifest["supplier_variants_comparison_json"] = str(comparison_path.resolve()) if comparison_path else None
    manifest["supplier_variants_manifest_json"] = str(supplier_manifest_path.resolve())
    if validation_path:
        manifest["supplier_variants_validation_json"] = validation_path
    write_json(manifest_path, manifest)
    return manifest


def _parse_elevations(raw: str) -> list[float]:
    out: list[float] = []
    for chunk in str(raw or "").split(","):
        chunk = chunk.strip()
        if chunk:
            out.append(float(chunk))
    return out or [0.0, 30.0, 45.0]


def _parse_supplier_gif_layers(raw: str | None) -> list[str]:
    allowed = {"interior", "kitchen", "surfaces", "windows", "curtains", "tables_chairs", "non_kitchen"}
    out: list[str] = []
    for chunk in str(raw or "interior").split(","):
        layer = chunk.strip().lower()
        if layer in allowed and layer not in out:
            out.append(layer)
    return out or ["interior"]


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


def _render_supplier_room_gifs(
    *,
    cfg_runtime: dict[str, Any],
    args: argparse.Namespace,
    run_dir: Path,
    layout_mode: str,
    supplier_scene_json_path: Path,
    supplier_blend_path: Path,
) -> dict[str, Any] | None:
    if bool(getattr(args, "skip_supplier_gif", False)):
        return None
    if not supplier_scene_json_path.is_file():
        return None
    use_reference_blend = supplier_blend_path.is_file()

    elevations = _parse_elevations(str(getattr(args, "supplier_gif_elevations", "0,30,45") or "0,30,45"))
    layers = _parse_supplier_gif_layers(str(getattr(args, "supplier_gif_layers", "interior") or "interior"))
    frames = int(getattr(args, "supplier_gif_frames", 36) or 36)
    fps = int(getattr(args, "supplier_gif_fps", 8) or 8)
    keep_frames = bool(getattr(args, "keep_supplier_gif_frames", False))
    out: list[dict[str, Any]] = []

    for layer in layers:
        render_layer = "all" if layer == "interior" else layer
        for elevation in elevations:
            suffix = f"{layer}.elev_{int(round(elevation)):02d}"
            frame_dir = run_dir / f"_frames_supplier_{suffix}"
            gif_path = run_dir / f"room_supplier.{suffix}.gif"
            if frame_dir.exists():
                shutil.rmtree(frame_dir, ignore_errors=True)

            cmd = [
                sys.executable,
                cfg_runtime["BLENDER_VIS_SCRIPT"],
                "--json",
                str(supplier_scene_json_path.resolve()),
                "--background",
                "--no-bbox-fallback",
                "--turntable-render-dir",
                str(frame_dir.resolve()),
                "--turntable-frames",
                str(frames),
                "--turntable-elevation-deg",
                str(float(elevation)),
                "--no-pack-assets",
            ]
            if use_reference_blend:
                cmd += ["--reference-blend", str(supplier_blend_path.resolve())]
            if render_layer == "all":
                cmd.append("--hide-room-shell")
            else:
                cmd += ["--render-layer", render_layer]
            if args.blender:
                cmd += ["--blender", args.blender]

            print("▶ Supplier room GIF:\n ", " ".join(cmd))
            subprocess.run(cmd, check=True)
            _render_gif_from_frames(frame_dir, gif_path, fps)
            if not keep_frames:
                shutil.rmtree(frame_dir, ignore_errors=True)
            out.append(
                {
                    "layer": layer,
                    "render_layer": render_layer,
                    "elevation_deg": float(elevation),
                    "gif": str(gif_path.resolve()),
                    "frames_dir": str(frame_dir.resolve()) if keep_frames else None,
                }
            )

    return {
        "supplier_scene_json": str(supplier_scene_json_path.resolve()),
        "supplier_blend": str(supplier_blend_path.resolve()) if use_reference_blend else None,
        "used_reference_blend": bool(use_reference_blend),
        "hide_room_shell": True,
        "layers": layers,
        "bbox": False,
        "orbit_center": "room_geometric_center",
        "orbit_radius_policy": "max(room_width,room_depth)*1.5",
        "frames": frames,
        "fps": fps,
        "outputs": out,
    }


def _acquire_supplier_assets_for_bindings(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    bindings_json_path: Path,
) -> tuple[Path, dict[str, Any]]:
    out_dir = Path(str(args.supplier_assets_dir or (run_dir / "supplier_assets"))).expanduser().resolve()
    db_path = Path(str(args.supplier_assets_db or (run_dir / "supplier_scene_assets.db"))).expanduser().resolve()
    enriched_bindings_path = run_dir / f"{bindings_json_path.stem}.assets.json"
    supplier_catalog_jsons = [Path(x).expanduser().resolve() for x in (args.supplier_catalog_json or []) if str(x).strip()]

    out_path = acquire_assets_for_bindings_json(
        bindings_json_path=bindings_json_path,
        output_json_path=enriched_bindings_path,
        db_path=db_path,
        out_dir=out_dir,
        blender_bin=args.supplier_assets_blender or args.blender,
        catalog_json_paths=supplier_catalog_jsons,
    )
    asset_data = json.loads(out_path.read_text(encoding="utf-8"))
    summary = ((asset_data.get("meta") or {}).get("asset_acquisition") or {})
    return out_path, {
        "bindings_json": str(out_path.resolve()),
        "db_path": str(db_path.resolve()),
        "out_dir": str(out_dir.resolve()),
        "summary": summary,
    }


def _resolve_supplier_bindings_json(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    layout_targets_json_path: str,
    supplier_user_preferences_json: str | None = None,
    room_design_spec: dict[str, Any] | None = None,
    selection_mode: str | None = None,
    out_suffix_override: str | None = None,
) -> Path | None:
    explicit = str(args.supplier_bindings_json or "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()

    supplier_catalog_jsons = [Path(x).expanduser().resolve() for x in (args.supplier_catalog_json or []) if str(x).strip()]
    if not supplier_catalog_jsons:
        return None

    sites = {str(x).strip() for x in (args.supplier_site or []) if str(x).strip()} or None
    catalog_rows = load_supplier_catalog_json(
        supplier_catalog_jsons,
        sites=sites,
        rich_only=bool(args.supplier_rich_only),
    )

    supplier_user_preferences: dict[str, Any] | None = None
    supplier_preferences_path = str(
        supplier_user_preferences_json
        or getattr(args, "supplier_user_preferences_json", "")
        or ""
    ).strip()
    if supplier_preferences_path:
        raw = read_supplier_matcher_json(supplier_preferences_path)
        if not isinstance(raw, dict):
            raise RuntimeError("supplier user preferences JSON must be an object")
        supplier_user_preferences = raw

    supplier_llm_provider = str(getattr(args, "supplier_llm_provider", "none") or "none").strip().lower()
    llm_settings = {
        "provider": supplier_llm_provider,
        "ollama_url": str(getattr(args, "supplier_ollama_url", None) or args.ollama_url or "http://127.0.0.1:11434"),
        "ollama_model": str(getattr(args, "supplier_ollama_model", None) or args.ollama_model or "gpt-oss:20b"),
        "ollama_timeout": int(getattr(args, "supplier_ollama_timeout", None) or args.ollama_timeout or 180),
        "ollama_temperature": float(getattr(args, "supplier_ollama_temperature", None) or 0.0),
        "top_n": int(getattr(args, "supplier_llm_top_n", None) or min(max(int(args.supplier_top_k), 1), 5)),
    }

    selection_strategy = str(getattr(args, "supplier_selection_strategy", "balanced") or "balanced").strip().lower()
    out_suffix = out_suffix_override or ("llm" if supplier_llm_provider != "none" else "heuristic")
    if not out_suffix_override and selection_strategy and selection_strategy != "balanced":
        out_suffix = f"{out_suffix}.{selection_strategy}"
    out_path = run_dir / f"base_supplier_bindings.{out_suffix}.json"
    result = build_bindings_with_candidates(
        targets_json_path=Path(layout_targets_json_path).expanduser().resolve(),
        catalog_rows=catalog_rows,
        top_k=int(args.supplier_top_k),
        selection_strategy=str(getattr(args, "supplier_selection_strategy", "balanced") or "balanced"),
        user_preferences=supplier_user_preferences,
        llm_settings=llm_settings,
        room_design_spec=room_design_spec,
        selection_mode=selection_mode,
    )
    write_json(out_path, result)
    return out_path


def _parse_supplier_selection_modes(raw: str | None) -> list[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    aliases = {
        "balanced": "optimal",
        "style": "best_match",
        "cheap_style": "optimal",
    }
    out: list[str] = []
    for part in re.split(r"[,;\\s]+", text):
        mode = aliases.get(part.strip().lower(), part.strip().lower())
        if mode not in {"cheapest", "optimal", "best_match"}:
            continue
        if mode not in out:
            out.append(mode)
    return out


def _build_room_design_spec_for_targets(
    *,
    run_dir: Path,
    prompt_text: str,
    layout_targets_json_path: str,
    style_profile: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    targets_path = Path(layout_targets_json_path).expanduser().resolve()
    targets_data = read_supplier_matcher_json(targets_path)
    if not isinstance(targets_data, dict):
        raise RuntimeError("layout targets JSON must be an object")
    spec = build_room_design_spec(
        user_prompt=prompt_text,
        layout_targets=targets_data,
        style_profile=style_profile,
    )
    out_path = run_dir / "room_design_spec.json"
    write_json(out_path, spec)
    return out_path, spec


def _run_supplier_modes_for_artifacts(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    artifacts: PlacementArtifacts,
    layout_targets_json_path: str,
    prompt_text: str,
    style_profile: dict[str, Any],
    style_supplier_preferences_path: Path | None,
    manifest: dict[str, Any],
    manifest_path: Path,
    manifest_key: str = "supplier_variants",
) -> tuple[Path | None, dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None, dict[str, Any]]:
    modes = _parse_supplier_selection_modes(getattr(args, "supplier_selection_modes", None))
    if not modes:
        modes = [str(getattr(args, "supplier_selection_mode", "") or "").strip() or None]  # type: ignore[list-item]

    spec_path, room_design_spec = _build_room_design_spec_for_targets(
        run_dir=run_dir,
        prompt_text=prompt_text,
        layout_targets_json_path=layout_targets_json_path,
        style_profile=style_profile,
    )
    manifest["room_design_spec_json"] = str(spec_path.resolve())

    variants: dict[str, Any] = {}
    primary_scene: Path | None = None
    primary_info: dict[str, Any] | None = None
    primary_report: dict[str, Any] | None = None
    primary_assets: dict[str, Any] | None = None
    preferred_mode = "optimal" if "optimal" in [m for m in modes if m] else (modes[0] or "optimal")

    for raw_mode in modes:
        mode = raw_mode or str(getattr(args, "supplier_selection_strategy", "balanced") or "balanced")
        mode_name = {"balanced": "optimal", "cheap_style": "optimal", "style": "best_match"}.get(str(mode), str(mode))
        bindings_path = _resolve_supplier_bindings_json(
            args=args,
            run_dir=run_dir,
            layout_targets_json_path=layout_targets_json_path,
            supplier_user_preferences_json=(
                str(style_supplier_preferences_path.resolve())
                if style_supplier_preferences_path and not str(getattr(args, "supplier_user_preferences_json", "") or "").strip()
                else None
            ),
            room_design_spec=room_design_spec,
            selection_mode=mode_name,
            out_suffix_override=f"{mode_name}",
        )
        if not bindings_path:
            continue
        initial_bindings_path = bindings_path
        consistent_path: Path | None = None
        bindings_data = json.loads(bindings_path.read_text(encoding="utf-8"))
        consistent_bindings = apply_supplier_scene_consistency(bindings_data)
        if consistent_bindings != bindings_data:
            consistent_path = run_dir / f"{bindings_path.stem}.consistent.json"
            write_json(consistent_path, consistent_bindings)
            bindings_path = consistent_path
        assets_bindings_path, assets_info = _acquire_supplier_assets_for_bindings(
            args=args,
            run_dir=run_dir,
            bindings_json_path=bindings_path,
        )
        supplier_info = _apply_supplier_bindings_for_artifacts(
            artifacts=artifacts,
            run_dir=run_dir,
            bindings_json_path=assets_bindings_path,
            require_local_asset=bool(args.supplier_require_local_asset),
            variant_suffix=mode_name,
        )
        report_info = _write_supplier_replacement_reports_for_artifacts(
            run_dir=run_dir,
            bindings_json_path=assets_bindings_path,
            supplier_info=supplier_info,
            variant_suffix=mode_name,
        )
        variants[mode_name] = {
            "initial_bindings": str(initial_bindings_path.resolve()),
            "consistent_bindings": str(consistent_path.resolve()) if consistent_path else str(initial_bindings_path.resolve()),
            "bindings": str(assets_bindings_path.resolve()),
            "assets": assets_info,
            "rebind": supplier_info,
            "reports": report_info,
            "blender": {
                "blend_path": None,
                "blend_exists": False,
                "blender_status": "skipped",
                "blender_error": "not_run_yet",
                "build_report": None,
            },
        }
        if mode_name == preferred_mode or primary_scene is None:
            primary_info = supplier_info
            primary_assets = assets_info
            primary_report = report_info
            if supplier_info.get("scene_v1"):
                primary_scene = Path(str(supplier_info["scene_v1"])).expanduser().resolve()
        manifest[manifest_key] = variants
        write_json(manifest_path, manifest)

    comparison_path = _write_supplier_variants_comparison(run_dir, variants)
    if comparison_path:
        manifest["supplier_variants_comparison_json"] = str(comparison_path.resolve())
        manifest[manifest_key] = variants
        supplier_manifest_path = _write_supplier_variants_manifest(
            run_dir=run_dir,
            modes=list(variants.keys()),
            variants=variants,
            room_design_spec_path=str(spec_path.resolve()),
            comparison_json=str(comparison_path.resolve()),
        )
        manifest["supplier_variants_manifest_json"] = str(supplier_manifest_path.resolve())
        write_json(manifest_path, manifest)

    return primary_scene, primary_info, primary_assets, primary_report, manifest


def _flooring_style_label(style_profile: dict[str, Any]) -> str | None:
    raw = str(style_profile.get("style_label") or "").strip().lower().replace("-", "_")
    aliases = {
        "modern": "contemporary",
        "industrial": "loft",
        "classicism": "classic",
        "neoclassical": "classic",
        "wabi_sabi": "japandi",
        "mid_century_modern": "contemporary",
        "art_deco": "classic",
        "rustic": "classic",
        "coastal": "scandinavian",
    }
    return aliases.get(raw, raw or None)


def _flooring_room_type(style_profile: dict[str, Any], scene_json_path: Path) -> str | None:
    raw = str(style_profile.get("room_type") or "").strip().lower().replace(" ", "_")
    aliases = {
        "bedroom": "bedroom",
        "livingroom": "living_room",
        "living_room": "living_room",
        "kitchen": "kitchen",
        "bathroom": "bathroom",
        "diningroom": "living_room",
        "dining_room": "living_room",
    }
    if raw in aliases:
        return aliases[raw]
    try:
        data = json.loads(scene_json_path.read_text(encoding="utf-8"))
        room = data.get("room") if isinstance(data, dict) else {}
        if isinstance(room, dict):
            scene_room = str(room.get("room_type") or "").strip().lower()
            return aliases.get(scene_room, scene_room or None)
    except Exception:
        return None
    return None


def _maybe_apply_flooring_to_scene(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    scene_json_path: Path,
    prompt_text: str,
    style_profile: dict[str, Any],
    room_id: str,
    suffix: str,
) -> tuple[Path, dict[str, Any] | None]:
    if bool(getattr(args, "no_flooring", False)):
        return scene_json_path, None

    materials_path = Path(str(getattr(args, "flooring_materials", "") or "")).expanduser()
    style_rules_path = Path(str(getattr(args, "flooring_style_rules", "") or "")).expanduser()
    if not materials_path.is_absolute():
        materials_path = (Path.cwd() / materials_path).resolve()
    if not style_rules_path.is_absolute():
        style_rules_path = (Path.cwd() / style_rules_path).resolve()

    if not (materials_path.is_file() or materials_path.is_dir()):
        print(f"⏭ flooring: каталог не найден, пропуск: {materials_path}")
        return scene_json_path, None
    if not style_rules_path.is_file():
        print(f"⏭ flooring: правила стилей не найдены, пропуск: {style_rules_path}")
        return scene_json_path, None

    selection_path = run_dir / f"flooring.selection{suffix}.v1.json"
    scene_out_path = run_dir / f"{scene_json_path.stem}.flooring.v1.json"
    style = _flooring_style_label(style_profile)
    room_type = _flooring_room_type(style_profile, scene_json_path)
    llm_settings = {
        "provider": str(getattr(args, "flooring_llm_provider", "ollama") or "ollama"),
        "ollama_url": str(getattr(args, "flooring_ollama_url", None) or getattr(args, "ollama_url", None) or "http://127.0.0.1:11434"),
        "ollama_model": str(getattr(args, "flooring_ollama_model", None) or getattr(args, "ollama_model", None) or "gpt-oss:20b"),
        "ollama_timeout": int(getattr(args, "flooring_ollama_timeout", None) or getattr(args, "ollama_timeout", None) or 180),
        "ollama_temperature": float(getattr(args, "flooring_ollama_temperature", 0.0) or 0.0),
        "ollama_num_ctx": int(getattr(args, "flooring_ollama_num_ctx", 8192) or 8192),
        "top_n": int(getattr(args, "flooring_llm_top_n", 5) or 5),
    }

    flooring_prompt_text = _flooring_prompt_for_selector(prompt_text, style_profile, run_dir)
    print("🧱 flooring: подбор покрытия пола")
    selection = run_flooring_selection(
        prompt=flooring_prompt_text,
        style=style,
        room_type=room_type,
        room_description=str(style_profile.get("style_hint") or ""),
        room_id=room_id,
        materials_path=materials_path,
        style_rules_path=style_rules_path,
        out_path=selection_path,
        top_k=int(getattr(args, "flooring_top_k", 10) or 10),
        llm_settings=llm_settings,
    )

    scene = json.loads(scene_json_path.read_text(encoding="utf-8"))
    scene_with_flooring = apply_flooring_to_scene(scene, selection)
    write_flooring_json(scene_with_flooring, scene_out_path)
    selected = selection.get("selected_material") or {}
    texture = selection.get("texture_candidate") or {}
    print(
        "🧱 flooring selected: "
        f"{selected.get('sku')} | {selected.get('name')} | "
        f"texture={texture.get('texture_abs_path') or texture.get('texture_path')} | "
        f"usable={bool(texture.get('usable_in_blender'))}"
    )
    return scene_out_path, {
        "selection_json": str(selection_path.resolve()),
        "scene_v1": str(scene_out_path.resolve()),
        "selected_sku": selected.get("sku"),
        "selected_name": selected.get("name"),
        "texture_path": texture.get("texture_abs_path") or texture.get("texture_path"),
        "texture_usable_in_blender": bool(texture.get("usable_in_blender")),
        "llm_rerank": selection.get("llm_rerank"),
    }


def _maybe_apply_wall_material_to_scene(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    scene_json_path: Path,
    prompt_text: str,
    style_profile: dict[str, Any],
    room_id: str,
    suffix: str,
) -> tuple[Path, dict[str, Any] | None]:
    if bool(getattr(args, "no_wall_material", False)):
        return scene_json_path, None

    materials_path = Path(str(getattr(args, "wall_materials", "") or "")).expanduser()
    if not materials_path.is_absolute():
        materials_path = (Path.cwd() / materials_path).resolve()
    if not (materials_path.is_file() or materials_path.is_dir()):
        print(f"⏭ wall material: каталог не найден, пропуск: {materials_path}")
        return scene_json_path, None

    selection_path = run_dir / f"wall_material.selection{suffix}.v1.json"
    scene_out_path = run_dir / f"{scene_json_path.stem}.wall_material.v1.json"
    style = _flooring_style_label(style_profile)
    room_type = _flooring_room_type(style_profile, scene_json_path)
    llm_settings = {
        "provider": str(getattr(args, "wall_llm_provider", "ollama") or "ollama"),
        "ollama_url": str(getattr(args, "wall_ollama_url", None) or getattr(args, "ollama_url", None) or "http://127.0.0.1:11434"),
        "ollama_model": str(getattr(args, "wall_ollama_model", None) or getattr(args, "ollama_model", None) or "gpt-oss:20b"),
        "ollama_timeout": int(getattr(args, "wall_ollama_timeout", None) or getattr(args, "ollama_timeout", None) or 180),
        "ollama_temperature": float(getattr(args, "wall_ollama_temperature", 0.0) or 0.0),
        "ollama_num_ctx": int(getattr(args, "wall_ollama_num_ctx", 8192) or 8192),
        "top_n": int(getattr(args, "wall_llm_top_n", 5) or 5),
    }

    wall_prompt_text = _flooring_prompt_for_selector(prompt_text, style_profile, run_dir)
    print("🧱 wall material: подбор покрытия стен")
    selection = run_wall_selection(
        prompt=wall_prompt_text,
        style=style,
        room_type=room_type,
        room_description=str(style_profile.get("style_hint") or ""),
        room_id=room_id,
        materials_path=materials_path,
        out_path=selection_path,
        top_k=int(getattr(args, "wall_top_k", 10) or 10),
        llm_settings=llm_settings,
    )

    scene = json.loads(scene_json_path.read_text(encoding="utf-8"))
    scene_with_wall = apply_wall_material_to_scene_with_catalog(scene, selection, materials_path=materials_path)
    write_wall_json(scene_with_wall, scene_out_path)
    selected = selection.get("selected_material") or {}
    print(
        "🧱 wall material selected: "
        f"{selected.get('sku')} | {selected.get('name')} | "
        f"avg={selected.get('average_hex') or selected.get('average_rgb')}"
    )
    return scene_out_path, {
        "selection_json": str(selection_path.resolve()),
        "scene_v1": str(scene_out_path.resolve()),
        "selected_sku": selected.get("sku"),
        "selected_name": selected.get("name"),
        "average_rgb": selected.get("average_rgb"),
        "average_hex": selected.get("average_hex"),
        "dominant_colors_hex": selected.get("dominant_colors_hex"),
        "llm_rerank": selection.get("llm_rerank"),
    }


def _scene_windows(scene: dict[str, Any]) -> list[dict[str, Any]]:
    room = scene.get("room")
    if not isinstance(room, dict):
        return []
    windows = room.get("windows")
    if not isinstance(windows, list):
        return []
    return [w for w in windows if isinstance(w, dict)]


def _scene_has_curtain_items(scene: dict[str, Any]) -> bool:
    items = scene.get("items") if isinstance(scene.get("items"), list) else scene.get("placements")
    if not isinstance(items, list):
        return False
    for item in items:
        if not isinstance(item, dict):
            continue
        text = " ".join(
            str(item.get(key) or "")
            for key in ("name", "category", "semantic_group", "id")
        ).lower()
        source = item.get("source")
        if isinstance(source, dict):
            text += " " + " ".join(str(v or "") for v in source.values()).lower()
        asset = item.get("asset")
        if isinstance(asset, dict):
            text += " " + str(asset.get("kind") or "").lower()
        if any(token in text for token in ("curtain", "shtor", "штор", "занавес", "window_covering")):
            return True
    return False


def _curtains_needed_for_scene(
    *,
    scene: dict[str, Any],
    prompt_text: str,
    style_profile: dict[str, Any],
    policy: str,
) -> tuple[bool, str]:
    if not _scene_windows(scene):
        return False, "missing_windows"
    if _scene_has_curtain_items(scene):
        return False, "existing_curtains"
    if policy == "always":
        return True, "policy_always"

    text_parts = [
        prompt_text,
        str(style_profile.get("expanded_prompt") or ""),
        str(style_profile.get("style_hint") or ""),
        str(style_profile.get("surface_design_brief") or ""),
        str(style_profile.get("chooser_prompt") or ""),
    ]
    text = " ".join(part for part in text_parts if part).lower()
    negative_tokens = (
        "no curtain",
        "no curtains",
        "without curtain",
        "without curtains",
        "без штор",
        "без занавес",
        "не нужны шторы",
        "шторы не нужны",
        "без жалюзи",
    )
    if any(token in text for token in negative_tokens):
        return False, "prompt_says_no_curtains"

    for key in ("needs_curtains", "wants_curtains", "curtains", "window_coverings"):
        if bool(style_profile.get(key)):
            return True, f"profile_{key}"

    explicit_tokens = (
        "curtain",
        "curtains",
        "drape",
        "drapes",
        "window treatment",
        "window covering",
        "tulle",
        "blind",
        "blinds",
        "штор",
        "занавес",
        "тюль",
        "гардин",
        "портьер",
        "жалюзи",
    )
    if any(token in text for token in explicit_tokens):
        return True, "prompt_mentions_curtains"

    room_type = str(style_profile.get("room_type") or "").strip().lower().replace(" ", "_")
    if room_type in {"bedroom", "livingroom", "living_room", "kids_room", "nursery", "детская", "спальня", "гостиная"}:
        return True, f"default_for_room_type:{room_type}"

    return False, "auto_not_requested"


def _maybe_apply_curtains_to_scene(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    scene_json_path: Path,
    prompt_text: str,
    style_profile: dict[str, Any],
    suffix: str,
) -> tuple[Path, dict[str, Any] | None]:
    policy = str(getattr(args, "curtains", "auto") or "auto").strip().lower()
    if bool(getattr(args, "no_curtains", False)):
        policy = "never"
    if policy in {"off", "false", "0", "no"}:
        policy = "never"
    if policy in {"on", "true", "1", "yes"}:
        policy = "always"
    if policy not in {"auto", "always", "never"}:
        policy = "auto"

    if policy == "never":
        return scene_json_path, None

    scene = json.loads(scene_json_path.read_text(encoding="utf-8"))
    needed, needed_reason = _curtains_needed_for_scene(
        scene=scene,
        prompt_text=prompt_text,
        style_profile=style_profile,
        policy=policy,
    )
    if not needed:
        print(f"⏭ curtains: пропуск ({needed_reason})")
        return scene_json_path, {
            "added_count": 0,
            "skipped_reason": needed_reason,
            "policy": policy,
        }

    materials_path = Path(str(getattr(args, "curtain_materials", "") or "")).expanduser()
    if not materials_path.is_absolute():
        materials_path = (Path.cwd() / materials_path).resolve()
    if not (materials_path.is_file() or materials_path.is_dir()):
        print(f"⏭ curtains: каталог не найден, пропуск: {materials_path}")
        return scene_json_path, None

    catalog, catalog_base_dir = load_curtain_catalog(materials_path)
    if not catalog:
        print(f"⏭ curtains: в каталоге нет пригодных товаров с локальными картинками: {materials_path}")
        return scene_json_path, None
    models_dir = Path(str(getattr(args, "curtain_models_dir", "") or "")).expanduser()
    if not models_dir.is_absolute():
        models_dir = (Path.cwd() / models_dir).resolve()
    curtain_model_paths = discover_curtain_models(models_dir)
    supplier_catalog_path = Path(str(getattr(args, "curtain_supplier_catalog", "") or "")).expanduser()
    if not supplier_catalog_path.is_absolute():
        supplier_catalog_path = (Path.cwd() / supplier_catalog_path).resolve()
    supplier_curtain_models = discover_supplier_curtain_models(
        supplier_catalog_path=supplier_catalog_path,
        manual_assets_root="data/sourse/suppliers/manual_assets/3ddd",
    )

    scene_out_path = run_dir / f"{scene_json_path.stem}.curtains.v1.json"
    seed = int(getattr(args, "curtain_seed", 0) or 0)
    if seed == 0:
        seed = int(getattr(args, "seed", 0) or 0)
    scene_with_curtains, info = apply_curtains_to_scene(
        scene,
        catalog=catalog,
        catalog_base_dir=catalog_base_dir,
        curtain_model_paths=curtain_model_paths,
        curtain_models=supplier_curtain_models,
        style_profile=style_profile,
        seed=seed,
    )
    if int(info.get("added_count", 0) or 0) <= 0:
        print(f"⏭ curtains: шторы не добавлены ({info.get('skipped_reason') or 'no_window_fit'})")
        return scene_json_path, info

    write_curtain_json(scene_out_path, scene_with_curtains)
    first = (info.get("selected") or [{}])[0]
    print(
        "🪟 curtains selected: "
        f"added={info.get('added_count')} | first={first.get('sku')} {first.get('name')} | "
        f"texture={first.get('texture_path')}"
    )
    return scene_out_path, {
        "scene_v1": str(scene_out_path.resolve()),
        "catalog_path": str(materials_path.resolve()),
        "models_dir": str(models_dir.resolve()),
        "supplier_catalog_path": str(supplier_catalog_path.resolve()),
        "policy": policy,
        "needed_reason": needed_reason,
        **info,
    }


def _flooring_prompt_for_selector(prompt_text: str, style_profile: dict[str, Any], run_dir: Path) -> str:
    parts = [str(style_profile.get("expanded_prompt") or prompt_text or "").strip()]
    style_hint = str(style_profile.get("style_hint") or "").strip()
    if style_hint:
        parts.append(f"Style/color context from style LLM: {style_hint}")
    surface_brief = str(style_profile.get("surface_design_brief") or "").strip()
    if surface_brief:
        parts.append(f"Surface design brief: {surface_brief}")
    preferred_colors = style_profile.get("preferred_colors")
    if isinstance(preferred_colors, list) and preferred_colors:
        parts.append("Preferred room colors: " + ", ".join(str(x) for x in preferred_colors if str(x).strip()))
    for key, label in (
        ("wall_palette", "Wall color targets"),
        ("floor_palette", "Floor color/material targets"),
        ("furniture_palette", "Furniture/object color targets"),
    ):
        values = style_profile.get(key)
        if isinstance(values, list) and values:
            parts.append(f"{label}: " + ", ".join(str(x) for x in values if str(x).strip()))
    material_family = style_profile.get("material_family")
    if isinstance(material_family, list) and material_family:
        parts.append("Preferred materials: " + ", ".join(str(x) for x in material_family if str(x).strip()))
    meta_path = run_dir / "infinigen_clean_meta.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            style_label = str(meta.get("style_label") or "").strip()
            room_semantic = str(meta.get("room_semantic") or "").strip()
            if style_label or room_semantic:
                parts.append(
                    "Infinigen generated scene context: "
                    f"style={style_label or 'unknown'}, room={room_semantic or 'unknown'}. "
                    "Choose a floor color/material that harmonizes with the generated Infinigen interior."
                )
        except Exception:
            pass
    return "\n".join(part for part in parts if part).strip() or str(prompt_text or "")


def _maybe_apply_fast_infinigen_profile(args: argparse.Namespace, style_profile: dict[str, Any]) -> None:
    fast_small = bool(getattr(args, "infinigen_fast_small", False))
    no_pose_cameras = bool(getattr(args, "infinigen_no_pose_cameras", False))
    solve_large = getattr(args, "infinigen_solve_steps_large", None)
    solve_medium = getattr(args, "infinigen_solve_steps_medium", None)
    solve_small = getattr(args, "infinigen_solve_steps_small", None)
    if not fast_small and not no_pose_cameras and solve_large is None and solve_medium is None and solve_small is None:
        return
    infinigen = style_profile.setdefault("infinigen", {})
    if not isinstance(infinigen, dict):
        raise RuntimeError("style_profile.infinigen must be an object")
    if fast_small:
        params = infinigen.setdefault("monkeypatch_params", {})
        if not isinstance(params, dict):
            raise RuntimeError("style_profile.infinigen.monkeypatch_params must be an object")
        params.update(
            {
                "obj_interior_obj_pct": 0.0,
                "obj_on_storage_pct": 0.0,
                "obj_on_nonstorage_pct": 0.0,
            }
        )
    overrides = infinigen.setdefault("overrides", [])
    if not isinstance(overrides, list):
        raise RuntimeError("style_profile.infinigen.overrides must be a list")

    override_map: dict[str, str] = {}
    if fast_small:
        override_map.update(
            {
                "compose_indoors.solve_medium_enabled": "False",
                "compose_indoors.solve_small_enabled": "False",
                "compose_indoors.solve_steps_large": "60",
                "compose_indoors.solve_steps_medium": "0",
                "compose_indoors.solve_steps_small": "0",
            }
        )
    if solve_large is not None:
        override_map["compose_indoors.solve_steps_large"] = str(max(0, int(solve_large)))
    if solve_medium is not None:
        medium_steps = max(0, int(solve_medium))
        override_map["compose_indoors.solve_medium_enabled"] = "True" if medium_steps > 0 else "False"
        override_map["compose_indoors.solve_steps_medium"] = str(medium_steps)
    if solve_small is not None:
        small_steps = max(0, int(solve_small))
        override_map["compose_indoors.solve_small_enabled"] = "True" if small_steps > 0 else "False"
        override_map["compose_indoors.solve_steps_small"] = str(small_steps)
    if no_pose_cameras:
        override_map["compose_indoors.pose_cameras_enabled"] = "False"
        override_map["compose_indoors.animate_cameras_enabled"] = "False"

    for key, value in override_map.items():
        overrides[:] = [item for item in overrides if not str(item).startswith(f"{key}=")]
        item = f"{key}={value}"
        if item not in overrides:
            overrides.append(item)


def run_pipeline_for_mode(
    cfg_runtime: dict[str, str],
    args: argparse.Namespace,
    room_path: str,
    run_dir: Path,
    layout_mode: str,
    prompt_text: str,
    style_profile_template: dict[str, Any],
) -> ModeOutputs:
    print(f"\n====== РЕЖИМ {layout_mode.upper()} ======")
    print(f"📁 mode_run_dir: {run_dir}")

    placer_spec = PLACER_SPECS[args.placer]
    chooser_required = bool(placer_spec.get("requires_object_selection", True))
    chooser_seed = int.from_bytes(secrets.token_bytes(8), "big")
    style_profile = deepcopy(style_profile_template)
    _maybe_apply_fast_infinigen_profile(args, style_profile)
    style_profile_path = run_dir / "style_profile.json"
    write_json(style_profile_path, style_profile)

    original_room_path = Path(room_path).expanduser().resolve()
    styled_room_path = run_dir / "room.style.v1.json"
    room_data = json.loads(original_room_path.read_text(encoding="utf-8"))
    styled_room_data = attach_style_hint_to_room_json(room_data, style_profile)
    write_json(styled_room_path, styled_room_data)
    effective_room_path = str(styled_room_path.resolve())

    chooser_prompt_text = str(style_profile.get("chooser_prompt") or prompt_text).strip() or prompt_text
    effective_prompt_text = chooser_prompt_text
    style_supplier_preferences = style_profile.get("supplier_preferences")
    style_supplier_preferences_path: Optional[Path] = None
    if isinstance(style_supplier_preferences, dict):
        style_supplier_preferences_path = run_dir / "style_supplier_preferences.json"
        write_json(style_supplier_preferences_path, style_supplier_preferences)

    (run_dir / "prompt.txt").write_text(prompt_text, encoding="utf-8")
    (run_dir / "prompt.styled.txt").write_text(effective_prompt_text, encoding="utf-8")
    (run_dir / "chooser_prompt.txt").write_text(chooser_prompt_text, encoding="utf-8")

    objects_path: Optional[Path] = None
    normalized_objects_path: Optional[Path] = None
    if chooser_required:
        objects_path = run_choose_stage(
            args=args,
            cfg_runtime=cfg_runtime,
            room_path=effective_room_path,
            prompt_text=chooser_prompt_text,
            run_dir=run_dir,
            seed=chooser_seed,
        )

        normalized_objects_path = run_dir / "objects.v1.json"
        normalize_json_artifact(
            cfg_runtime=cfg_runtime,
            input_path=objects_path,
            output_path=normalized_objects_path,
            target="objects",
        )
    else:
        print(f"⏭ Пропуск chooser для placer={args.placer}")

    run_manifest = {
        "room": effective_room_path,
        "room_original": str(original_room_path),
        "prompt": prompt_text,
        "prompt_styled": effective_prompt_text,
        "chooser_prompt": chooser_prompt_text,
        "chooser_seed": chooser_seed,
        "placer": args.placer,
        "layout_mode": layout_mode,
        "run_dir": str(run_dir),
        "style_profile_json": str(style_profile_path.resolve()),
        "style_room_json": str(styled_room_path.resolve()),
        "style": {
            "style_label": style_profile.get("style_label"),
            "room_type": style_profile.get("room_type"),
            "confidence": style_profile.get("confidence"),
            "style_hint": style_profile.get("style_hint"),
        },
        "supplier_preferences_json": (
            str(Path(args.supplier_user_preferences_json).expanduser().resolve())
            if str(getattr(args, "supplier_user_preferences_json", "") or "").strip()
            else str(style_supplier_preferences_path.resolve()) if style_supplier_preferences_path else None
        ),
        "objects_legacy": str(objects_path.resolve()) if objects_path else None,
        "objects_v1": str(normalized_objects_path.resolve()) if normalized_objects_path else None,
        "chooser_llm": {
            "provider": "ollama",
            "url": args.ollama_url,
            "model": args.ollama_model,
            "models": list(args.ollama_models) if getattr(args, "ollama_models", None) else [args.ollama_model],
            "timeout": args.ollama_timeout,
            "temperature": args.ollama_temperature,
            "max_attempts": args.ollama_max_attempts,
        },
        "plan_llm": {
            "models": list(args.plan_models),
            "think": args.plan_think,
            "temperature": args.plan_temperature,
        },
        "critic_llm": {
            "models": list(args.critic_models),
            "think": args.critic_think,
            "temperature": args.critic_temperature,
        },
    }
    manifest_path = run_dir / "run_manifest.json"
    write_json(manifest_path, run_manifest)

    if args.placer == "lego_gen":
        if normalized_objects_path is None:
            raise RuntimeError("placer=lego_gen требует objects.v1.json, но chooser stage был пропущен")
        lego_artifacts = run_lego_generate_from_scratch(
            cfg_runtime=cfg_runtime,
            args=args,
            room_path=effective_room_path,
            objects_v1_path=normalized_objects_path,
            run_dir=run_dir,
        )
        lego_artifacts, kitchen_info = _maybe_apply_kitchen_stage(
            args=args,
            artifacts=lego_artifacts,
            run_dir=run_dir,
            room_path=effective_room_path,
            prompt_text=effective_prompt_text,
            suffix="lego_gen",
        )
        lego_selection_stub = _build_layout_selection_stub_for_artifacts(
            artifacts=lego_artifacts,
            run_dir=run_dir,
            prefix="lego_gen",
        )

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["lego_gen"] = {
            "enabled": True,
            "placement_legacy": str(lego_artifacts.placement_legacy.resolve()),
            "placement_v1": str(lego_artifacts.placement_v1.resolve()),
            "scene_v1": str(lego_artifacts.scene_v1.resolve()) if lego_artifacts.scene_v1 else None,
            "scene_legacy": str(lego_artifacts.scene_legacy.resolve()) if lego_artifacts.scene_legacy else None,
            "layout_targets_json": lego_selection_stub["layout_targets_json"],
            "supplier_bindings_stub_json": lego_selection_stub["supplier_bindings_stub_json"],
            "scene_pricing_stub_json": lego_selection_stub["scene_pricing_stub_json"],
        }
        if kitchen_info is not None:
            manifest["kitchen_stage"] = kitchen_info

        supplier_scene_for_render: Optional[Path] = None
        supplier_scene_for_render, supplier_info, supplier_assets_info, supplier_report_info, manifest = _run_supplier_modes_for_artifacts(
            args=args,
            run_dir=run_dir,
            artifacts=lego_artifacts,
            layout_targets_json_path=lego_selection_stub["layout_targets_json"],
            prompt_text=effective_prompt_text,
            style_profile=style_profile,
            style_supplier_preferences_path=style_supplier_preferences_path,
            manifest=manifest,
            manifest_path=manifest_path,
            manifest_key="supplier_variants",
        )
        if supplier_info is not None:
            manifest["supplier_rebind"] = supplier_info
            manifest["supplier_assets"] = supplier_assets_info
            manifest["supplier_replacement_reports"] = supplier_report_info
        base_scene_for_render = choose_scene_for_render(lego_artifacts)
        base_scene_for_render, base_repair_info = maybe_repair_scene_json(
            args=args,
            scene_json_path=base_scene_for_render,
            run_dir=run_dir,
            tag="lego_gen_base",
        )
        if base_repair_info is not None:
            manifest["scene_repair_base"] = base_repair_info
        base_scene_for_render, base_layout_post_info = _maybe_apply_layout_postprocess(
            args=args,
            scene_json_path=base_scene_for_render,
            run_dir=run_dir,
            tag="lego_gen_base",
        )
        if base_layout_post_info is not None:
            manifest["layout_postprocess_base"] = base_layout_post_info
        base_scene_for_render, base_flooring_info = _maybe_apply_flooring_to_scene(
            args=args,
            run_dir=run_dir,
            scene_json_path=base_scene_for_render,
            prompt_text=prompt_text,
            style_profile=style_profile,
            room_id="room_001",
            suffix=".lego_gen_base",
        )
        if base_flooring_info is not None:
            manifest["flooring_base"] = base_flooring_info
            if isinstance(manifest.get("lego_gen"), dict):
                manifest["lego_gen"]["scene_v1_flooring"] = base_flooring_info.get("scene_v1")
        base_scene_for_render, base_wall_info = _maybe_apply_wall_material_to_scene(
            args=args,
            run_dir=run_dir,
            scene_json_path=base_scene_for_render,
            prompt_text=prompt_text,
            style_profile=style_profile,
            room_id="room_001",
            suffix=".lego_gen_base",
        )
        if base_wall_info is not None:
            manifest["wall_material_base"] = base_wall_info
            if isinstance(manifest.get("lego_gen"), dict):
                manifest["lego_gen"]["scene_v1_wall_material"] = base_wall_info.get("scene_v1")
        base_scene_for_render, base_curtain_info = _maybe_apply_curtains_to_scene(
            args=args,
            run_dir=run_dir,
            scene_json_path=base_scene_for_render,
            prompt_text=prompt_text,
            style_profile=style_profile,
            suffix=".lego_gen_base",
        )
        if base_curtain_info is not None:
            manifest["curtains_base"] = base_curtain_info
            if isinstance(manifest.get("lego_gen"), dict):
                manifest["lego_gen"]["scene_v1_curtains"] = base_curtain_info.get("scene_v1")
        surface_pricing_info = _write_surface_material_pricing(
            run_dir=run_dir,
            room_path=Path(effective_room_path).expanduser().resolve(),
            flooring_info=base_flooring_info,
            wall_info=base_wall_info,
            pricing_stub_json=lego_selection_stub.get("scene_pricing_stub_json"),
            suffix=".lego_gen_base",
        )
        if surface_pricing_info is not None:
            manifest["surface_materials_pricing_base"] = surface_pricing_info
        if supplier_scene_for_render and supplier_scene_for_render.is_file():
            supplier_scene_for_render, supplier_repair_info = maybe_repair_scene_json(
                args=args,
                scene_json_path=supplier_scene_for_render,
                run_dir=run_dir,
                tag="lego_gen_supplier",
            )
            if supplier_repair_info is not None:
                manifest["scene_repair_supplier"] = supplier_repair_info
            supplier_scene_for_render, supplier_layout_post_info = _maybe_apply_layout_postprocess(
                args=args,
                scene_json_path=supplier_scene_for_render,
                run_dir=run_dir,
                tag="lego_gen_supplier",
            )
            if supplier_layout_post_info is not None:
                manifest["layout_postprocess_supplier"] = supplier_layout_post_info
            supplier_scene_for_render, supplier_flooring_info = _maybe_apply_flooring_to_scene(
                args=args,
                run_dir=run_dir,
                scene_json_path=supplier_scene_for_render,
                prompt_text=prompt_text,
                style_profile=style_profile,
                room_id="room_001",
                suffix=".lego_gen_supplier",
            )
            if supplier_flooring_info is not None:
                manifest["flooring_supplier"] = supplier_flooring_info
                if isinstance(manifest.get("supplier_rebind"), dict):
                    manifest["supplier_rebind"]["scene_v1_flooring"] = supplier_flooring_info.get("scene_v1")
            supplier_scene_for_render, supplier_wall_info = _maybe_apply_wall_material_to_scene(
                args=args,
                run_dir=run_dir,
                scene_json_path=supplier_scene_for_render,
                prompt_text=prompt_text,
                style_profile=style_profile,
                room_id="room_001",
                suffix=".lego_gen_supplier",
            )
            if supplier_wall_info is not None:
                manifest["wall_material_supplier"] = supplier_wall_info
                if isinstance(manifest.get("supplier_rebind"), dict):
                    manifest["supplier_rebind"]["scene_v1_wall_material"] = supplier_wall_info.get("scene_v1")
            supplier_scene_for_render, supplier_curtain_info = _maybe_apply_curtains_to_scene(
                args=args,
                run_dir=run_dir,
                scene_json_path=supplier_scene_for_render,
                prompt_text=prompt_text,
                style_profile=style_profile,
                suffix=".lego_gen_supplier",
            )
            if supplier_curtain_info is not None:
                manifest["curtains_supplier"] = supplier_curtain_info
                if isinstance(manifest.get("supplier_rebind"), dict):
                    manifest["supplier_rebind"]["scene_v1_curtains"] = supplier_curtain_info.get("scene_v1")
            surface_pricing_info = _write_surface_material_pricing(
                run_dir=run_dir,
                room_path=Path(effective_room_path).expanduser().resolve(),
                flooring_info=supplier_flooring_info,
                wall_info=supplier_wall_info,
                pricing_stub_json=None,
                suffix=".lego_gen_supplier",
            )
            if surface_pricing_info is not None:
                manifest["surface_materials_pricing_supplier"] = surface_pricing_info
            if supplier_assets_info and supplier_assets_info.get("bindings_json"):
                supplier_report_info = _write_supplier_replacement_reports_for_artifacts(
                    run_dir=run_dir,
                    bindings_json_path=Path(str(supplier_assets_info["bindings_json"])).expanduser().resolve(),
                    supplier_info=supplier_info or {},
                    variant_suffix=str((supplier_report_info or {}).get("mode") or ""),
                )
                manifest["supplier_replacement_reports"] = supplier_report_info
        write_json(manifest_path, manifest)
        variants = manifest.get("supplier_variants") if isinstance(manifest.get("supplier_variants"), dict) else {}

        if args.skip_blender:
            _mark_supplier_blender_skipped(variants, reason="skip_blender")
            manifest = _finalize_supplier_variant_artifacts(
                args=args,
                run_dir=run_dir,
                manifest=manifest,
                manifest_path=manifest_path,
                variants=variants,
            )
            print(f"⏭ Пропуск Blender для режима {layout_mode}")
            print(f"\n✅ УСПЕХ! РЕЖИМ {layout_mode}")
            return ModeOutputs(base_artifacts=lego_artifacts, lego_artifacts=None)

        run_blender_for_mode(
            cfg_runtime=cfg_runtime,
            args=args,
            room_path=effective_room_path,
            run_dir=run_dir,
            layout_mode=layout_mode,
            scene_json_path=base_scene_for_render,
            variant_suffix="lego_gen",
        )

        if variants and bool(getattr(args, "build_supplier_blend", False)):
            _run_supplier_blender_variants(
                cfg_runtime=cfg_runtime,
                args=args,
                run_dir=run_dir,
                layout_mode=layout_mode,
                effective_room_path=effective_room_path,
                variants=variants,
            )
            _refresh_supplier_reports_after_blender(run_dir=run_dir, variants=variants)
            manifest = _finalize_supplier_variant_artifacts(
                args=args,
                run_dir=run_dir,
                manifest=manifest,
                manifest_path=manifest_path,
                variants=variants,
            )
        elif variants:
            _mark_supplier_blender_skipped(variants, reason="build_supplier_blend_disabled")
            manifest = _finalize_supplier_variant_artifacts(
                args=args,
                run_dir=run_dir,
                manifest=manifest,
                manifest_path=manifest_path,
                variants=variants,
            )

        if supplier_scene_for_render and supplier_scene_for_render.is_file() and not variants:
            run_blender_for_mode(
                cfg_runtime=cfg_runtime,
                args=args,
                room_path=effective_room_path,
                run_dir=run_dir,
                layout_mode=layout_mode,
                scene_json_path=supplier_scene_for_render,
                variant_suffix="lego_gen_supplier",
            )
            supplier_blend_out, _ = blender_outputs_for_mode(
                args,
                run_dir,
                layout_mode,
                variant_suffix="lego_gen_supplier",
            )
            supplier_gif_info = _render_supplier_room_gifs(
                cfg_runtime=cfg_runtime,
                args=args,
                run_dir=run_dir,
                layout_mode=layout_mode,
                supplier_scene_json_path=supplier_scene_for_render,
                supplier_blend_path=Path(str(supplier_blend_out)).expanduser().resolve(),
            )
            if supplier_gif_info is not None:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["supplier_room_gifs"] = supplier_gif_info
                write_json(manifest_path, manifest)

        print(f"\n✅ УСПЕХ! РЕЖИМ {layout_mode}")
        return ModeOutputs(base_artifacts=lego_artifacts, lego_artifacts=None)

    placement_out = run_dir / f"placement_{layout_mode}.json"
    base_artifacts: Optional[PlacementArtifacts] = None
    placement_attempts = 1 if args.placer == "ollama_llm" else int(args.max_attempts)

    for attempt in range(1, placement_attempts + 1):
        print(f"\n---------- ПОПЫТКА {attempt} ({layout_mode}) ----------")
        try:
            attempt_seed = int.from_bytes(secrets.token_bytes(8), "big")

            attempt_info = {
                "attempt": attempt,
                "attempt_seed": attempt_seed,
                "chooser_seed": chooser_seed,
                "layout_mode": layout_mode,
                "placer": args.placer,
                "objects_path": str(objects_path.resolve()) if objects_path else None,
                "objects_v1_path": str(normalized_objects_path.resolve()) if normalized_objects_path else None,
                "placement_legacy_path": str(placement_out.resolve()),
            }
            write_json(run_dir / f"attempt_{attempt:02d}.json", attempt_info)

            execute_placer(
                cfg_runtime=cfg_runtime,
                args=args,
                room_path=effective_room_path,
                objects_path=objects_path,
                layout_mode=layout_mode,
                seed=attempt_seed,
                out_path=placement_out,
                run_dir=run_dir,
                prompt_text=effective_prompt_text,
            )

            base_artifacts = build_scene_artifacts(
                cfg_runtime=cfg_runtime,
                room_path=effective_room_path,
                run_dir=run_dir,
                layout_mode=layout_mode,
                placement_out=placement_out,
                variant_suffix="",
            )
            base_artifacts, kitchen_info = _maybe_apply_kitchen_stage(
                args=args,
                artifacts=base_artifacts,
                run_dir=run_dir,
                room_path=effective_room_path,
                prompt_text=effective_prompt_text,
                suffix="base",
            )
            base_selection_stub = _build_layout_selection_stub_for_artifacts(
                artifacts=base_artifacts,
                run_dir=run_dir,
                prefix="base",
            )

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["base"] = {
                "placement_legacy": str(base_artifacts.placement_legacy.resolve()),
                "placement_v1": str(base_artifacts.placement_v1.resolve()),
                "scene_v1": str(base_artifacts.scene_v1.resolve()) if base_artifacts.scene_v1 else None,
                "scene_legacy": str(base_artifacts.scene_legacy.resolve()) if base_artifacts.scene_legacy else None,
                "layout_targets_json": base_selection_stub["layout_targets_json"],
                "supplier_bindings_stub_json": base_selection_stub["supplier_bindings_stub_json"],
                "scene_pricing_stub_json": base_selection_stub["scene_pricing_stub_json"],
            }
            if kitchen_info is not None:
                manifest["kitchen_stage"] = kitchen_info
            write_json(manifest_path, manifest)

            print(f"✅ placement stage success: {layout_mode}")
            break

        except Exception as e:
            print(f"❌ placement stage failed on attempt {attempt}: {e}")
            if _is_fatal_disk_full_error(e):
                raise RuntimeError(
                    "Placement aborted due to full disk on the remote/local worker. "
                    "Free space and rerun."
                ) from e
            if attempt >= placement_attempts:
                raise

    if base_artifacts is None:
        raise RuntimeError(f"Не удалось получить base placement для режима {layout_mode}")

    supplier_scene_for_render: Optional[Path] = None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    supplier_scene_for_render, supplier_info, supplier_assets_info, supplier_report_info, manifest = _run_supplier_modes_for_artifacts(
        args=args,
        run_dir=run_dir,
        artifacts=base_artifacts,
        layout_targets_json_path=base_selection_stub["layout_targets_json"],
        prompt_text=effective_prompt_text,
        style_profile=style_profile,
        style_supplier_preferences_path=style_supplier_preferences_path,
        manifest=manifest,
        manifest_path=manifest_path,
        manifest_key="supplier_variants",
    )
    if supplier_info is not None:
        manifest["supplier_rebind"] = supplier_info
        manifest["supplier_assets"] = supplier_assets_info
        manifest["supplier_replacement_reports"] = supplier_report_info
        write_json(manifest_path, manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_scene_for_render = choose_scene_for_render(base_artifacts)
    base_scene_for_render, base_repair_info = maybe_repair_scene_json(
        args=args,
        scene_json_path=base_scene_for_render,
        run_dir=run_dir,
        tag="base",
    )
    if base_repair_info is not None:
        manifest["scene_repair_base"] = base_repair_info
    base_scene_for_render, base_layout_post_info = _maybe_apply_layout_postprocess(
        args=args,
        scene_json_path=base_scene_for_render,
        run_dir=run_dir,
        tag="base",
    )
    if base_layout_post_info is not None:
        manifest["layout_postprocess_base"] = base_layout_post_info
    base_scene_for_render, base_flooring_info = _maybe_apply_flooring_to_scene(
        args=args,
        run_dir=run_dir,
        scene_json_path=base_scene_for_render,
        prompt_text=prompt_text,
        style_profile=style_profile,
        room_id="room_001",
        suffix=".base",
    )
    if base_flooring_info is not None:
        manifest["flooring_base"] = base_flooring_info
        if isinstance(manifest.get("base"), dict):
            manifest["base"]["scene_v1_flooring"] = base_flooring_info.get("scene_v1")
    base_scene_for_render, base_wall_info = _maybe_apply_wall_material_to_scene(
        args=args,
        run_dir=run_dir,
        scene_json_path=base_scene_for_render,
        prompt_text=prompt_text,
        style_profile=style_profile,
        room_id="room_001",
        suffix=".base",
    )
    if base_wall_info is not None:
        manifest["wall_material_base"] = base_wall_info
        if isinstance(manifest.get("base"), dict):
            manifest["base"]["scene_v1_wall_material"] = base_wall_info.get("scene_v1")
    base_scene_for_render, base_curtain_info = _maybe_apply_curtains_to_scene(
        args=args,
        run_dir=run_dir,
        scene_json_path=base_scene_for_render,
        prompt_text=prompt_text,
        style_profile=style_profile,
        suffix=".base",
    )
    if base_curtain_info is not None:
        manifest["curtains_base"] = base_curtain_info
        if isinstance(manifest.get("base"), dict):
            manifest["base"]["scene_v1_curtains"] = base_curtain_info.get("scene_v1")
    surface_pricing_info = _write_surface_material_pricing(
        run_dir=run_dir,
        room_path=Path(effective_room_path).expanduser().resolve(),
        flooring_info=base_flooring_info,
        wall_info=base_wall_info,
        pricing_stub_json=base_selection_stub.get("scene_pricing_stub_json"),
        suffix=".base",
    )
    if surface_pricing_info is not None:
        manifest["surface_materials_pricing_base"] = surface_pricing_info
    if supplier_scene_for_render and supplier_scene_for_render.is_file():
        supplier_scene_for_render, supplier_repair_info = maybe_repair_scene_json(
            args=args,
            scene_json_path=supplier_scene_for_render,
            run_dir=run_dir,
            tag="supplier",
        )
        if supplier_repair_info is not None:
            manifest["scene_repair_supplier"] = supplier_repair_info
        supplier_scene_for_render, supplier_layout_post_info = _maybe_apply_layout_postprocess(
            args=args,
            scene_json_path=supplier_scene_for_render,
            run_dir=run_dir,
            tag="supplier",
        )
        if supplier_layout_post_info is not None:
            manifest["layout_postprocess_supplier"] = supplier_layout_post_info
        supplier_scene_for_render, supplier_flooring_info = _maybe_apply_flooring_to_scene(
            args=args,
            run_dir=run_dir,
            scene_json_path=supplier_scene_for_render,
            prompt_text=prompt_text,
            style_profile=style_profile,
            room_id="room_001",
            suffix=".supplier",
        )
        if supplier_flooring_info is not None:
            manifest["flooring_supplier"] = supplier_flooring_info
            if isinstance(manifest.get("supplier_rebind"), dict):
                manifest["supplier_rebind"]["scene_v1_flooring"] = supplier_flooring_info.get("scene_v1")
        supplier_scene_for_render, supplier_wall_info = _maybe_apply_wall_material_to_scene(
            args=args,
            run_dir=run_dir,
            scene_json_path=supplier_scene_for_render,
            prompt_text=prompt_text,
            style_profile=style_profile,
            room_id="room_001",
            suffix=".supplier",
        )
        if supplier_wall_info is not None:
            manifest["wall_material_supplier"] = supplier_wall_info
            if isinstance(manifest.get("supplier_rebind"), dict):
                manifest["supplier_rebind"]["scene_v1_wall_material"] = supplier_wall_info.get("scene_v1")
        supplier_scene_for_render, supplier_curtain_info = _maybe_apply_curtains_to_scene(
            args=args,
            run_dir=run_dir,
            scene_json_path=supplier_scene_for_render,
            prompt_text=prompt_text,
            style_profile=style_profile,
            suffix=".supplier",
        )
        if supplier_curtain_info is not None:
            manifest["curtains_supplier"] = supplier_curtain_info
            if isinstance(manifest.get("supplier_rebind"), dict):
                manifest["supplier_rebind"]["scene_v1_curtains"] = supplier_curtain_info.get("scene_v1")
        surface_pricing_info = _write_surface_material_pricing(
            run_dir=run_dir,
            room_path=Path(effective_room_path).expanduser().resolve(),
            flooring_info=supplier_flooring_info,
            wall_info=supplier_wall_info,
            pricing_stub_json=None,
            suffix=".supplier",
        )
        if surface_pricing_info is not None:
            manifest["surface_materials_pricing_supplier"] = surface_pricing_info
        if supplier_assets_info and supplier_assets_info.get("bindings_json"):
            supplier_report_info = _write_supplier_replacement_reports_for_artifacts(
                run_dir=run_dir,
                bindings_json_path=Path(str(supplier_assets_info["bindings_json"])).expanduser().resolve(),
                supplier_info=supplier_info or {},
            )
            manifest["supplier_replacement_reports"] = supplier_report_info
    write_json(manifest_path, manifest)
    variants = manifest.get("supplier_variants") if isinstance(manifest.get("supplier_variants"), dict) else {}

    if args.skip_blender:
        _mark_supplier_blender_skipped(variants, reason="skip_blender")
        manifest = _finalize_supplier_variant_artifacts(
            args=args,
            run_dir=run_dir,
            manifest=manifest,
            manifest_path=manifest_path,
            variants=variants,
        )
        print(f"⏭ Пропуск Blender для режима {layout_mode}")
        print(f"\n✅ УСПЕХ! РЕЖИМ {layout_mode}")
        return ModeOutputs(base_artifacts=base_artifacts, lego_artifacts=None)

    run_blender_for_mode(
        cfg_runtime=cfg_runtime,
        args=args,
        room_path=effective_room_path,
        run_dir=run_dir,
        layout_mode=layout_mode,
        scene_json_path=base_scene_for_render,
        variant_suffix="",
    )

    if variants and bool(getattr(args, "build_supplier_blend", False)):
        _run_supplier_blender_variants(
            cfg_runtime=cfg_runtime,
            args=args,
            run_dir=run_dir,
            layout_mode=layout_mode,
            effective_room_path=effective_room_path,
            variants=variants,
        )
        _refresh_supplier_reports_after_blender(run_dir=run_dir, variants=variants)
        manifest = _finalize_supplier_variant_artifacts(
            args=args,
            run_dir=run_dir,
            manifest=manifest,
            manifest_path=manifest_path,
            variants=variants,
        )
    elif variants:
        _mark_supplier_blender_skipped(variants, reason="build_supplier_blend_disabled")
        manifest = _finalize_supplier_variant_artifacts(
            args=args,
            run_dir=run_dir,
            manifest=manifest,
            manifest_path=manifest_path,
            variants=variants,
        )

    if supplier_scene_for_render and supplier_scene_for_render.is_file() and not variants:
        run_blender_for_mode(
            cfg_runtime=cfg_runtime,
            args=args,
            room_path=effective_room_path,
            run_dir=run_dir,
            layout_mode=layout_mode,
            scene_json_path=supplier_scene_for_render,
            variant_suffix="supplier",
        )
        supplier_blend_out, _ = blender_outputs_for_mode(
            args,
            run_dir,
            layout_mode,
            variant_suffix="supplier",
        )
        supplier_gif_info = _render_supplier_room_gifs(
            cfg_runtime=cfg_runtime,
            args=args,
            run_dir=run_dir,
            layout_mode=layout_mode,
            supplier_scene_json_path=supplier_scene_for_render,
            supplier_blend_path=Path(str(supplier_blend_out)).expanduser().resolve(),
        )
        if supplier_gif_info is not None:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["supplier_room_gifs"] = supplier_gif_info
            write_json(manifest_path, manifest)

    print(f"\n✅ УСПЕХ! РЕЖИМ {layout_mode}")
    return ModeOutputs(base_artifacts=base_artifacts, lego_artifacts=None)


def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    p.add_argument("items", nargs="*")
    p.add_argument("--prompt", default=None)
    p.add_argument("--prompt-file", default=None)

    p.add_argument("--paths-config", default=DEFAULT_PATHS_CONFIG)
    p.add_argument("--room", default="__USE_CFG_DEFAULT__")

    p.add_argument("--prepared-info", default=None)
    p.add_argument("--future-root", default=None)

    p.add_argument("--placer", default=None)
    p.add_argument("--ml-model", default=None)
    p.add_argument("--ml-device", default=None)
    p.add_argument("--diffusion-steps", type=int, default=None)
    p.add_argument("--max-attempts", type=int, default=None)

    p.add_argument("--save-blend", default=None)
    p.add_argument("--render", default=None)
    p.add_argument("--blender-output", choices=["render", "gif", "both"], default="render", help="Visual output produced by Blender for every rendered scene")
    p.add_argument("--keep-blend", action=argparse.BooleanOptionalAction, default=False, help="Keep generated .blend scenes; by default only PNG/GIF/report artifacts remain")
    p.add_argument("--blender-gif-frames", type=int, default=36)
    p.add_argument("--blender-gif-elevation", type=float, default=30.0)
    p.add_argument("--blender-gif-fps", type=int, default=8)
    p.add_argument("--keep-blender-gif-frames", action="store_true")
    p.add_argument("--blender", default=None)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--skip-blender", action="store_true")
    p.add_argument("--no-bbox-fallback", action="store_true", help="Disable default bbox fallback for items without a resolved/imported mesh")
    p.add_argument("--no-import-glb", action="store_true", help="Compat flag, ignored by current Blender scene builder")
    p.add_argument("--normalize-chandeliers", action="store_true", help="Postprocess ceiling chandeliers into symmetric coverage positions at least 1m from walls")
    p.add_argument("--repair-furniture-overlaps", action="store_true", help="Postprocess movable furniture to reduce AABB overlaps and room-boundary overflow")

    p.add_argument("--run-dir", default=None)
    p.add_argument("--keep-tmp", action="store_true")

    p.add_argument("--remote-runner", default=None)
    p.add_argument("--remote-host", default=None)
    p.add_argument("--remote-port", type=int, default=None)
    p.add_argument("--remote-user", default=None)
    p.add_argument("--remote-key", default=None)
    p.add_argument("--remote-conda-env", default=None)
    p.add_argument("--remote-infinigen-src", default=None)

    p.add_argument("--ollama-url", default=None)
    p.add_argument("--ollama-model", default=None)
    p.add_argument("--ollama-models", nargs="*", default=None)
    p.add_argument("--ollama-timeout", type=int, default=None)
    p.add_argument("--ollama-temperature", type=float, default=None)
    p.add_argument("--ollama-max-attempts", type=int, default=None)
    p.add_argument("--chooser-llm-provider", choices=["none", "ollama"], default="ollama")
    p.add_argument("--style-llm-provider", choices=["none", "ollama"], default="ollama")
    p.add_argument("--style-ollama-url", default=None)
    p.add_argument("--style-ollama-model", default=None)
    p.add_argument("--style-ollama-models", nargs="*", default=None)
    p.add_argument("--style-ollama-timeout", type=int, default=None)
    p.add_argument("--style-ollama-temperature", type=float, default=None)
    p.add_argument("--style-llm-max-attempts", type=int, default=None)
    p.add_argument("--style-llm-think", choices=["low", "medium", "high"], default=None)
    p.add_argument("--style-llm-debug-dir", default=None)

    p.add_argument("--plan-model", default=None)
    p.add_argument("--plan-models", nargs="*", default=None)
    p.add_argument("--plan-think", choices=["none", "low"], default=None)
    p.add_argument("--llm-think", choices=["none", "low"], default=None)
    p.add_argument("--plan-temperature", type=float, default=None)

    p.add_argument("--critic-model", default=None)
    p.add_argument("--critic-models", nargs="*", default=None)
    p.add_argument("--critic-think", choices=["none", "low"], default=None)
    p.add_argument("--critic-temperature", type=float, default=None)
    p.add_argument("--max-scene-attempts", type=int, default=None)

    p.add_argument("--modes", default=None)
    p.add_argument("--supplier-bindings-json", default=None, help="Optional supplier_bindings json to apply after placement")
    p.add_argument(
        "--supplier-catalog-json",
        action="append",
        default=["data/sourse/suppliers/supplier_catalog_canonical.json"],
        help="Supplier catalog export JSON for automatic binding search; can be repeated",
    )
    p.add_argument("--supplier-site", action="append", default=None, help="Optional supplier source_site filter for automatic binding search")
    p.add_argument("--supplier-top-k", type=int, default=5, help="Top-K candidates for automatic supplier matcher")
    p.add_argument(
        "--supplier-selection-strategy",
        choices=["balanced", "cheapest", "cheap_style", "style"],
        default="balanced",
        help="Automatic supplier ranking strategy: cheapest, cheap_style, style, or balanced.",
    )
    p.add_argument("--supplier-selection-mode", choices=["cheapest", "optimal", "best_match"], default=None, help="Design-aware supplier selection mode for single bindings output")
    p.add_argument("--supplier-selection-modes", default=None, help="Comma-separated design-aware modes to build, e.g. cheapest,optimal,best_match")
    p.add_argument("--supplier-build-modes", default=None, help="Comma-separated supplier modes to build in Blender. Defaults to all selection modes.")
    p.add_argument("--build-supplier-blend", action="store_true", help="Compat flag: supplier blends are built when Blender is not skipped.")
    p.add_argument("--validate-supplier-variants", action="store_true", help="Validate multi-mode supplier bindings and write supplier_variants.validation.json")
    p.add_argument("--supplier-rich-only", action="store_true", help="Use only rich supplier cards during automatic binding search")
    p.add_argument("--supplier-user-preferences-json", default=None, help="Optional JSON with supplier matcher user preferences")
    p.add_argument("--supplier-llm-provider", choices=["none", "ollama"], default="none", help="Optional final LLM reranker after heuristic supplier top-K")
    p.add_argument("--supplier-ollama-url", default=None, help="Optional Ollama URL override for supplier matcher reranking")
    p.add_argument("--supplier-ollama-model", default=None, help="Optional Ollama model override for supplier matcher reranking")
    p.add_argument("--supplier-ollama-timeout", type=int, default=None, help="Optional timeout override in seconds for supplier matcher reranking")
    p.add_argument("--supplier-ollama-temperature", type=float, default=None, help="Optional temperature override for supplier matcher reranking")
    p.add_argument("--supplier-llm-top-n", type=int, default=None, help="How many top heuristic supplier candidates to send to the supplier LLM reranker")
    p.add_argument("--supplier-require-local-asset", action="store_true", help="Apply supplier replacement only for bindings with local downloaded assets")
    p.add_argument("--supplier-assets-dir", default=None, help="Directory for scene-specific downloaded supplier assets")
    p.add_argument("--supplier-assets-db", default=None, help="SQLite DB for scene-specific downloaded supplier assets")
    p.add_argument("--supplier-assets-blender", default=None, help="Optional Blender binary for supplier asset conversion")

    p.add_argument("--kitchens", choices=["auto", "always", "never"], default="auto", help="Procedural kitchen set stage policy")
    p.add_argument("--kitchen-material-catalog", default="data/floor_materials/basisrf/products.csv")
    p.add_argument("--kitchen-appliance-catalog", default="data/sourse/suppliers/supplier_catalog_canonical.json")
    p.add_argument("--kitchen-selection-mode", choices=["cheapest", "optimal", "best_match"], default="optimal")
    p.add_argument("--kitchen-dining", choices=["auto", "always", "never"], default="auto", help="Add supplier-compatible dining table/chair targets for kitchens")
    p.add_argument("--kitchen-accessories", choices=["auto", "always", "never"], default="auto", help="Add supplier-compatible countertop cooking-set/kitchenware targets")
    p.add_argument("--kitchen-llm-provider", choices=["none", "ollama"], default="none", help="Use LLM for kitchen palette, material top-k, appliance top-k and dining placement decisions")
    p.add_argument("--kitchen-ollama-url", default="http://127.0.0.1:11434")
    p.add_argument("--kitchen-ollama-model", default="gpt-oss:20b")
    p.add_argument("--kitchen-ollama-timeout", type=int, default=180)
    p.add_argument("--kitchen-ollama-temperature", type=float, default=0.1)
    p.add_argument("--kitchen-ollama-num-ctx", type=int, default=8192)
    p.add_argument("--kitchen-ollama-think", default="low")
    p.add_argument("--kitchen-accessory-llm-provider", choices=["none", "ollama"], default="none", help="Use LLM to choose varied kitchen accessories from the supplier inventory")
    p.add_argument("--kitchen-accessory-ollama-url", default="http://127.0.0.1:11434")
    p.add_argument("--kitchen-accessory-ollama-model", default="gpt-oss:20b")
    p.add_argument("--kitchen-accessory-ollama-timeout", type=int, default=180)
    p.add_argument("--kitchen-accessory-ollama-temperature", type=float, default=0.2)
    p.add_argument("--kitchen-accessory-ollama-num-ctx", type=int, default=8192)
    p.add_argument("--kitchen-accessory-ollama-think", default="low")

    p.add_argument("--no-flooring", action="store_true", help="Disable supplier floor covering selection and Blender floor texture application")
    p.add_argument("--flooring-materials", default="data/floor_materials")
    p.add_argument("--flooring-style-rules", default="config/flooring_style_rules.json")
    p.add_argument("--flooring-top-k", type=int, default=10)
    p.add_argument("--flooring-llm-provider", choices=["none", "ollama"], default="ollama")
    p.add_argument("--flooring-ollama-url", default=None)
    p.add_argument("--flooring-ollama-model", default=None)
    p.add_argument("--flooring-ollama-timeout", type=int, default=None)
    p.add_argument("--flooring-ollama-temperature", type=float, default=0.0)
    p.add_argument("--flooring-ollama-num-ctx", type=int, default=8192)
    p.add_argument("--flooring-llm-top-n", type=int, default=5)
    p.add_argument("--no-wall-material", action="store_true", help="Disable supplier wall covering selection")
    p.add_argument("--wall-materials", default="data/floor_materials")
    p.add_argument("--wall-top-k", type=int, default=10)
    p.add_argument("--wall-llm-provider", choices=["none", "ollama"], default="ollama")
    p.add_argument("--wall-ollama-url", default=None)
    p.add_argument("--wall-ollama-model", default=None)
    p.add_argument("--wall-ollama-timeout", type=int, default=None)
    p.add_argument("--wall-ollama-temperature", type=float, default=0.0)
    p.add_argument("--wall-ollama-num-ctx", type=int, default=8192)
    p.add_argument("--wall-llm-top-n", type=int, default=5)
    p.add_argument("--curtains", choices=["auto", "always", "never"], default="auto", help="Shtorystore curtain postprocess policy")
    p.add_argument("--no-curtains", action="store_true", help="Disable Shtorystore curtain postprocess for room windows")
    p.add_argument("--curtain-materials", default="data/floor_materials/shtorystore_curtains")
    p.add_argument("--curtain-models-dir", default="data/sourse/curtains_3d", help="Directory with curtain FBX/GLB/OBJ models; all files are cycled across windows")
    p.add_argument("--curtain-supplier-catalog", default="data/sourse/suppliers/supplier_catalog_canonical.json")
    p.add_argument("--curtain-seed", type=int, default=0, help="Seed for Shtorystore curtain product selection; 0 uses run seed")
    p.add_argument("--skip-supplier-gif", action="store_true", help="Disable supplier-only room GIF generation")
    p.add_argument("--supplier-gif-layers", default="interior", help="Comma-separated GIF layers: interior,kitchen,surfaces,windows,curtains,tables_chairs,non_kitchen")
    p.add_argument("--supplier-gif-frames", type=int, default=36)
    p.add_argument("--supplier-gif-elevations", default="0,30,45")
    p.add_argument("--supplier-gif-fps", type=int, default=8)
    p.add_argument("--keep-supplier-gif-frames", action="store_true")

    p.add_argument("--lego-postprocess", action="store_true")
    p.add_argument("--infinigen-src", default=None)
    p.add_argument("--infinigen-task", default=None)
    p.add_argument("--infinigen-configs", nargs="+", default=None)
    p.add_argument(
        "--infinigen-fast-small",
        action="store_true",
        help="Disable Infinigen small-object solve stage and lower loose/surface object density",
    )
    p.add_argument("--infinigen-no-pose-cameras", action="store_true", help="Disable Infinigen camera pose search; Blender renderer frames the room later.")
    p.add_argument("--infinigen-solve-steps-large", type=int, default=None)
    p.add_argument("--infinigen-solve-steps-medium", type=int, default=None)
    p.add_argument("--infinigen-solve-steps-small", type=int, default=None)
    p.add_argument(
        "--infinigen-rebind-selected-objects",
        action="store_true",
        help="Legacy mode: replace Infinigen generated items with chooser-selected assets after generation.",
    )
    p.add_argument("--lego-modes", default=None)
    p.add_argument("--lego-repo", default=None)
    p.add_argument("--lego-python", default=None)
    p.add_argument("--lego-helper-script", default=None)
    p.add_argument("--lego-tmp-root", default=None)
    p.add_argument("--lego-checkpoint-bedroom", default=None)
    p.add_argument("--lego-checkpoint-livingroom", default=None)
    p.add_argument("--lego-room-type", choices=["auto", "bedroom", "livingroom"], default="auto")
    p.add_argument("--lego-render-policy", choices=["base_only", "lego_only", "both"], default="both")
    p.add_argument("--lego-failure-policy", choices=["skip", "raise"], default="skip")
    p.add_argument(
        "--lego-generation-preset",
        choices=sorted(DEFAULT_LEGO_GENERATION_PRESETS.keys()),
        default=None,
    )
    p.add_argument("--lego-method", choices=["direct_map_once", "direct_map", "grad_nonoise", "grad_noise"], default=None)
    p.add_argument("--lego-outer-passes", type=int, default=None)
    p.add_argument("--lego-num-restarts", type=int, default=None)
    p.add_argument("--lego-init-pos-noise-std", type=float, default=None)
    p.add_argument("--lego-init-ang-noise-deg", type=float, default=None)
    p.add_argument("--lego-init-scene-mode", choices=["perturb", "random_full"], default=None)

    add_scene_repair_arguments(p)

    return p


def main() -> None:
    parser = build_cli()
    args = parser.parse_args()

    cfg_path = Path(args.paths_config).expanduser().resolve()
    cfg = load_yaml(cfg_path)
    cfg_base_dir = project_root_from_config(cfg, cfg_path)

    apply_config_defaults(args, cfg, cfg_base_dir)
    cfg_runtime = build_runtime_paths(cfg, cfg_base_dir)

    room_path = os.path.abspath((args.room or cfg_runtime["DEFAULT_ROOM_JSON"]).strip())
    modes = parse_modes(args, cfg)

    print(f"📦 modes: {', '.join(modes)}")
    print(f"🧭 paths-config: {cfg_path}")
    print(f"🤖 json ollama models: {', '.join(args.ollama_models)}")
    print(f"🧠 plan ollama models: {', '.join(args.plan_models)}")
    print(f"🧐 critic ollama models: {', '.join(args.critic_models)}")
    print(f"🧩 plan/critic/json think: {args.plan_think}/{args.critic_think}/{args.llm_think}")

    style_models = [str(x).strip() for x in (getattr(args, "style_ollama_models", None) or args.ollama_models or []) if str(x).strip()]
    if not style_models:
        style_models = [str(getattr(args, "style_ollama_model", None) or args.ollama_model or "gpt-oss:20b").strip()]
    style_think = str(getattr(args, "style_llm_think", None) or "").strip().lower()
    if style_think not in {"low", "medium", "high"}:
        style_think = "low"
    style_temperature = getattr(args, "style_ollama_temperature", None)
    if style_temperature is None:
        style_temperature = args.ollama_temperature if args.ollama_temperature is not None else 0.0
    print(f"🎨 style llm: provider={args.style_llm_provider}, models={', '.join(style_models)}")

    if args.lego_postprocess:
        lego_cfg = resolve_lego_generation_params(args)
        print(
            "🧩 lego generation: "
            f"preset={lego_cfg['preset']}, "
            f"method={lego_cfg['method']}, "
            f"init_scene_mode={lego_cfg['init_scene_mode']}, "
            f"outer_passes={lego_cfg['outer_passes']}, "
            f"num_restarts={lego_cfg['num_restarts']}, "
            f"init_pos_noise_std={lego_cfg['init_pos_noise_std']}, "
            f"init_ang_noise_deg={lego_cfg['init_ang_noise_deg']}"
        )

    prompt_text = read_prompt_from_args(args)
    style_profile_template = analyze_prompt_to_style_profile(
        prompt_text=prompt_text,
        room_path=room_path,
        provider=str(getattr(args, "style_llm_provider", "ollama") or "ollama"),
        ollama_url=str(getattr(args, "style_ollama_url", None) or args.ollama_url or "http://127.0.0.1:11434"),
        ollama_models=style_models,
        timeout_sec=int(getattr(args, "style_ollama_timeout", None) or args.ollama_timeout or 180),
        temperature=float(style_temperature),
        max_attempts=int(getattr(args, "style_llm_max_attempts", None) or args.ollama_max_attempts or 4),
        think=style_think,
        debug_dir=str(getattr(args, "style_llm_debug_dir", None) or ""),
    )
    print(
        "🎯 style selected: "
        f"{style_profile_template.get('style_label')} "
        f"(room={style_profile_template.get('room_type')}, "
        f"confidence={float(style_profile_template.get('confidence') or 0.0):.2f})"
    )
    created_run_dirs: list[Path] = []

    try:
        for layout_mode in modes:
            mode_run_dir, _ = make_mode_run_dir(cfg_runtime["TMP_ROOT"], layout_mode, args.run_dir)
            created_run_dirs.append(mode_run_dir)

            run_pipeline_for_mode(
                cfg_runtime=cfg_runtime,
                args=args,
                room_path=room_path,
                run_dir=mode_run_dir,
                layout_mode=layout_mode,
                prompt_text=prompt_text,
                style_profile_template=style_profile_template,
            )

        print("\n✅ ВСЕ РЕЖИМЫ ОТРАБОТАЛИ УСПЕШНО")

    finally:
        if not args.keep_tmp and not args.run_dir:
            for p in created_run_dirs:
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                    print(f"🗑 Удалён run_dir: {p}")


if __name__ == "__main__":
    main()
