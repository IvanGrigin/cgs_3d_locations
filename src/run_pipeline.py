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
    from .pipeline_artifacts import (
        blender_outputs_for_mode,
        build_scene_artifacts,
        choose_scene_for_render,
        normalize_json_artifact,
        run_blender_for_mode,
    )
    from .pipeline_scene_repair import add_scene_repair_arguments, maybe_repair_scene_json
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
    from pipeline_artifacts import (
        blender_outputs_for_mode,
        build_scene_artifacts,
        choose_scene_for_render,
        normalize_json_artifact,
        run_blender_for_mode,
    )
    from pipeline_scene_repair import add_scene_repair_arguments, maybe_repair_scene_json
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


def _scene_room_polygon(room: dict[str, Any]) -> list[tuple[float, float]]:
    raw = room.get("floor_polygon") or room.get("floor_polygon_xz") or []
    out: list[tuple[float, float]] = []
    if isinstance(raw, list):
        for point in raw:
            if not isinstance(point, dict):
                continue
            x = _to_float(point.get("x"))
            y = _to_float(point.get("y", point.get("z")))
            if x is not None and y is not None:
                out.append((float(x), float(y)))
    return out


def _poly_bounds(poly: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), max(xs), min(ys), max(ys)


def _point_in_poly_xy(x: float, y: float, poly: list[tuple[float, float]]) -> bool:
    inside = False
    n = len(poly)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / max(yj - yi, 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _dist_point_segment_xy(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    denom = vx * vx + vy * vy
    if denom <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, (wx * vx + wy * vy) / denom))
    return math.hypot(px - (ax + t * vx), py - (ay + t * vy))


def _dist_to_poly_edges_xy(x: float, y: float, poly: list[tuple[float, float]]) -> float:
    if len(poly) < 2:
        return 0.0
    return min(
        _dist_point_segment_xy(x, y, poly[i][0], poly[i][1], poly[(i + 1) % len(poly)][0], poly[(i + 1) % len(poly)][1])
        for i in range(len(poly))
    )


def _room_sample_points(poly: list[tuple[float, float]], *, step: float, wall_margin: float = 0.0) -> list[tuple[float, float]]:
    x_min, x_max, y_min, y_max = _poly_bounds(poly)
    points: list[tuple[float, float]] = []
    x = x_min
    while x <= x_max + 1e-9:
        y = y_min
        while y <= y_max + 1e-9:
            if _point_in_poly_xy(x, y, poly) and _dist_to_poly_edges_xy(x, y, poly) >= wall_margin:
                points.append((x, y))
            y += step
        x += step
    if points:
        return points
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    return [(cx, cy)] if _point_in_poly_xy(cx, cy, poly) else []


def _is_chandelier_item(item: dict[str, Any]) -> bool:
    text = f"{item.get('name') or ''} {item.get('category') or ''} {item.get('semantic_group') or ''}".lower()
    positive = ("chandelier", "ceilinglamp", "ceiling_lamp", "ceiling light", "ceiling_light", "pendant", "люстр", "потолоч")
    negative = ("floorlamp", "floor_lamp", "tablelamp", "table_lamp", "walllamp", "wall_lamp", "торшер", "настоль", "бра")
    return any(token in text for token in positive) and not any(token in text for token in negative)


def _shift_item_xy(item: dict[str, Any], dx: float, dy: float) -> None:
    pos = item.get("position_m")
    if isinstance(pos, list) and len(pos) >= 2:
        pos[0] = float(pos[0]) + dx
        pos[1] = float(pos[1]) + dy
    aabb = item.get("aabb")
    if isinstance(aabb, dict):
        for key in ("x_min", "x_max"):
            if key in aabb:
                aabb[key] = float(aabb[key]) + dx
        for key in ("y_min", "y_max"):
            if key in aabb:
                aabb[key] = float(aabb[key]) + dy


def normalize_chandelier_positions_in_scene(
    scene: dict[str, Any],
    *,
    wall_clearance_m: float = 1.0,
    sample_step_m: float = 0.35,
) -> tuple[dict[str, Any], dict[str, Any]]:
    updated = deepcopy(scene)
    room = updated.get("room") if isinstance(updated.get("room"), dict) else {}
    poly = _scene_room_polygon(room)
    placements = updated.get("placements") if isinstance(updated.get("placements"), list) else []
    chandeliers = [item for item in placements if isinstance(item, dict) and _is_chandelier_item(item)]
    info: dict[str, Any] = {
        "enabled": True,
        "chandelier_count": len(chandeliers),
        "wall_clearance_m": wall_clearance_m,
        "moved": [],
    }
    if len(poly) < 3 or not chandeliers:
        info["skipped_reason"] = "no_room_polygon_or_no_chandeliers"
        return updated, info

    candidate_points = _room_sample_points(poly, step=sample_step_m, wall_margin=wall_clearance_m)
    if not candidate_points:
        candidate_points = _room_sample_points(poly, step=sample_step_m, wall_margin=0.0)
        info["clearance_fallback"] = True
    if not candidate_points:
        info["skipped_reason"] = "no_valid_room_points"
        return updated, info

    x_min, x_max, y_min, y_max = _poly_bounds(poly)
    centroid = (0.5 * (x_min + x_max), 0.5 * (y_min + y_max))
    first = min(candidate_points, key=lambda p: math.hypot(p[0] - centroid[0], p[1] - centroid[1]))
    centers = [first]
    while len(centers) < len(chandeliers):
        centers.append(
            max(
                candidate_points,
                key=lambda p: min(math.hypot(p[0] - c[0], p[1] - c[1]) for c in centers),
            )
        )

    coverage_points = _room_sample_points(poly, step=sample_step_m, wall_margin=0.0)
    coverage_radius = max(
        (min(math.hypot(p[0] - c[0], p[1] - c[1]) for c in centers) for p in coverage_points),
        default=0.0,
    )
    for item, center in zip(chandeliers, centers):
        pos = item.get("position_m")
        if not (isinstance(pos, list) and len(pos) >= 2):
            continue
        old_xy = (float(pos[0]), float(pos[1]))
        dx, dy = center[0] - old_xy[0], center[1] - old_xy[1]
        _shift_item_xy(item, dx, dy)
        meta = item.setdefault("meta", {})
        if isinstance(meta, dict):
            meta["chandelier_normalized"] = True
            meta["chandelier_coverage_radius_m"] = round(coverage_radius, 3)
        info["moved"].append(
            {
                "id": item.get("id"),
                "old_xy": [round(old_xy[0], 4), round(old_xy[1], 4)],
                "new_xy": [round(center[0], 4), round(center[1], 4)],
                "move_m": round(math.hypot(dx, dy), 4),
            }
        )
    info["coverage_radius_m"] = round(coverage_radius, 4)
    return updated, info


def _item_rect_xy(item: dict[str, Any]) -> tuple[float, float, float, float] | None:
    aabb = item.get("aabb")
    if isinstance(aabb, dict):
        vals = [_to_float(aabb.get(k)) for k in ("x_min", "x_max", "y_min", "y_max")]
        if all(v is not None for v in vals):
            x_min, x_max, y_min, y_max = (float(v) for v in vals)  # type: ignore[arg-type]
            if x_max > x_min and y_max > y_min:
                return x_min, x_max, y_min, y_max
    pos = item.get("position_m")
    size = item.get("size_m")
    if isinstance(pos, list) and isinstance(size, list) and len(pos) >= 2 and len(size) >= 2:
        cx, cy = float(pos[0]), float(pos[1])
        sx, sy = max(0.0, float(size[0])), max(0.0, float(size[1]))
        return cx - sx * 0.5, cx + sx * 0.5, cy - sy * 0.5, cy + sy * 0.5
    return None


def _rect_area(rect: tuple[float, float, float, float]) -> float:
    return max(0.0, rect[1] - rect[0]) * max(0.0, rect[3] - rect[2])


def _rect_intersection_area(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[2], b[2]))


def _rect_shift(rect: tuple[float, float, float, float], dx: float, dy: float) -> tuple[float, float, float, float]:
    return rect[0] + dx, rect[1] + dx, rect[2] + dy, rect[3] + dy


def _rect_outside_room_area(rect: tuple[float, float, float, float], poly: list[tuple[float, float]]) -> float:
    x_min, x_max, y_min, y_max = _poly_bounds(poly)
    inside_bounds = (
        max(rect[0], x_min),
        min(rect[1], x_max),
        max(rect[2], y_min),
        min(rect[3], y_max),
    )
    outside = _rect_area(rect) - _rect_area(inside_bounds)
    corners = [(rect[0], rect[2]), (rect[0], rect[3]), (rect[1], rect[2]), (rect[1], rect[3])]
    if any(not _point_in_poly_xy(x, y, poly) for x, y in corners):
        outside = max(outside, _rect_area(rect) * 0.25)
    return max(0.0, outside)


def _is_movable_furniture_item(item: dict[str, Any]) -> bool:
    text = (
        f"{item.get('name') or ''} {item.get('label') or ''} {item.get('category') or ''} "
        f"{item.get('type') or ''} {item.get('semantic_group') or ''}"
    ).lower()
    blocked = (
        "rug",
        "lamp",
        "light",
        "plant",
        "window",
        "door",
        "ceiling",
        "wall",
        "floor",
        "decor",
        "vase",
        "book",
        "pillow",
        "clutter",
        "accessory",
        "ковер",
        "торшер",
        "люстр",
        "раст",
        "декор",
        "ваза",
        "книга",
        "подушка",
    )
    if any(token in text for token in blocked):
        return False
    furniture_tokens = (
        "cabinet",
        "shelf",
        "table",
        "desk",
        "sofa",
        "chair",
        "bed",
        "dresser",
        "wardrobe",
        "stand",
        "комод",
        "шкаф",
        "стеллаж",
        "стол",
        "диван",
        "кресл",
        "кровать",
        "тумб",
    )
    return any(token in text for token in furniture_tokens)


def _is_support_child_candidate(item: dict[str, Any]) -> bool:
    text = (
        f"{item.get('name') or ''} {item.get('label') or ''} {item.get('category') or ''} "
        f"{item.get('type') or ''} {item.get('semantic_group') or ''}"
    ).lower()
    blocked = ("rug", "wall", "floor", "ceiling", "window", "door", "ковер", "стена", "пол", "окно", "двер")
    return not any(token in text for token in blocked)


def _z_range(item: dict[str, Any]) -> tuple[float, float] | None:
    aabb = item.get("aabb")
    if isinstance(aabb, dict):
        z_min = _to_float(aabb.get("z_min"))
        z_max = _to_float(aabb.get("z_max"))
        if z_min is not None and z_max is not None and z_max >= z_min:
            return float(z_min), float(z_max)
    pos = item.get("position_m")
    size = item.get("size_m")
    if isinstance(pos, list) and isinstance(size, list) and len(pos) >= 3 and len(size) >= 3:
        cz = float(pos[2])
        sz = max(0.0, float(size[2]))
        return cz - sz * 0.5, cz + sz * 0.5
    return None


def _rect_contains_center(container: tuple[float, float, float, float], child: tuple[float, float, float, float], margin: float = 0.08) -> bool:
    cx = 0.5 * (child[0] + child[1])
    cy = 0.5 * (child[2] + child[3])
    return (container[0] - margin) <= cx <= (container[1] + margin) and (container[2] - margin) <= cy <= (container[3] + margin)


def _support_child_indices(
    *,
    anchor_index: int,
    placements: list[Any],
    rects: list[tuple[float, float, float, float] | None],
) -> set[int]:
    anchor = placements[anchor_index]
    if not isinstance(anchor, dict):
        return set()
    anchor_rect = rects[anchor_index]
    anchor_z = _z_range(anchor)
    if anchor_rect is None or anchor_z is None:
        return set()
    anchor_area = _rect_area(anchor_rect)
    children: set[int] = set()
    for idx, child in enumerate(placements):
        if idx == anchor_index or not isinstance(child, dict) or not _is_support_child_candidate(child):
            continue
        if _is_movable_furniture_item(child):
            continue
        child_rect = rects[idx]
        child_z = _z_range(child)
        if child_rect is None or child_z is None:
            continue
        child_area = _rect_area(child_rect)
        if child_area > anchor_area * 0.75:
            continue
        if not _rect_contains_center(anchor_rect, child_rect):
            continue
        overlap_ratio = _rect_intersection_area(anchor_rect, child_rect) / max(child_area, 1e-9)
        if overlap_ratio < 0.55:
            continue
        on_top = abs(child_z[0] - anchor_z[1]) <= 0.18
        inside = child_z[0] >= anchor_z[0] - 0.05 and child_z[1] <= anchor_z[1] + 0.10
        if on_top or inside:
            children.add(idx)
    return children


def _support_children_by_anchor(
    *,
    placements: list[Any],
    rects: list[tuple[float, float, float, float] | None],
) -> dict[int, set[int]]:
    raw = {
        idx: _support_child_indices(anchor_index=idx, placements=placements, rects=rects)
        for idx in range(len(placements))
        if rects[idx] is not None
    }
    candidates_by_child: dict[int, list[int]] = {}
    for anchor_idx, children in raw.items():
        for child_idx in children:
            candidates_by_child.setdefault(child_idx, []).append(anchor_idx)
    pruned: dict[int, set[int]] = {idx: set() for idx in raw}
    for child_idx, anchor_indices in candidates_by_child.items():
        child_z = _z_range(placements[child_idx]) if isinstance(placements[child_idx], dict) else None
        if child_z is None:
            continue

        def score(anchor_idx: int) -> tuple[float, float, float]:
            anchor = placements[anchor_idx]
            anchor_rect = rects[anchor_idx]
            anchor_z = _z_range(anchor) if isinstance(anchor, dict) else None
            if anchor_rect is None or anchor_z is None:
                return (9.0, 9.0, 9.0)
            top_delta = abs(child_z[0] - anchor_z[1])
            on_top_rank = 0.0 if top_delta <= 0.18 else 1.0
            return (on_top_rank, top_delta if on_top_rank == 0.0 else 0.0, _rect_area(anchor_rect))

        best_anchor = min(anchor_indices, key=score)
        pruned.setdefault(best_anchor, set()).add(child_idx)
    return pruned


def _move_group_indices(anchor_index: int, children_by_anchor: dict[int, set[int]]) -> set[int]:
    group = {anchor_index}
    pending = list(children_by_anchor.get(anchor_index, set()))
    while pending:
        idx = pending.pop()
        if idx in group:
            continue
        group.add(idx)
        pending.extend(children_by_anchor.get(idx, set()))
    return group


def _union_rect(rects_for_group: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    return (
        min(r[0] for r in rects_for_group),
        max(r[1] for r in rects_for_group),
        min(r[2] for r in rects_for_group),
        max(r[3] for r in rects_for_group),
    )


def _collision_penalty_for_group(
    group_rects: dict[int, tuple[float, float, float, float]],
    *,
    group_indices: set[int],
    rects: list[tuple[float, float, float, float] | None],
    movable_indices: set[int],
    poly: list[tuple[float, float]],
) -> float:
    penalty = sum(_rect_outside_room_area(rect, poly) * 2.0 for rect in group_rects.values())
    group_union = _union_rect(list(group_rects.values()))
    for idx, other in enumerate(rects):
        if idx in group_indices or other is None or idx not in movable_indices:
            continue
        penalty += _rect_intersection_area(group_union, other)
    return penalty


def _best_repair_shift_for_item(
    *,
    item_index: int,
    group_indices: set[int],
    rects: list[tuple[float, float, float, float] | None],
    movable_indices: set[int],
    poly: list[tuple[float, float]],
    search_step_m: float,
    max_shift_m: float,
) -> tuple[float, float, float, float] | None:
    old = rects[item_index]
    if old is None:
        return None
    old_group_rects = {idx: rects[idx] for idx in group_indices if rects[idx] is not None}
    if item_index not in old_group_rects:
        return None
    old_penalty = _collision_penalty_for_group(
        old_group_rects,
        group_indices=group_indices,
        rects=rects,
        movable_indices=movable_indices,
        poly=poly,
    )
    old_group_union = _union_rect(list(old_group_rects.values()))
    best = (old_penalty, _rect_area(old_group_union), 0.0, 0.0)
    steps = max(1, int(math.ceil(max_shift_m / search_step_m)))
    directions = [(0.0, 0.0)]
    for k in range(16):
        angle = (2.0 * math.pi * k) / 16.0
        directions.append((math.cos(angle), math.sin(angle)))
    for step_idx in range(1, steps + 1):
        radius = min(max_shift_m, step_idx * search_step_m)
        for ux, uy in directions[1:]:
            dx, dy = ux * radius, uy * radius
            cand_group_rects = {idx: _rect_shift(rect, dx, dy) for idx, rect in old_group_rects.items()}
            penalty = _collision_penalty_for_group(
                cand_group_rects,
                group_indices=group_indices,
                rects=rects,
                movable_indices=movable_indices,
                poly=poly,
            )
            cand_group_union = _union_rect(list(cand_group_rects.values()))
            old_overlap = _rect_intersection_area(old_group_union, cand_group_union)
            if penalty < best[0] - 1e-6 or (abs(penalty - best[0]) <= 1e-6 and old_overlap > best[1]):
                best = (penalty, old_overlap, dx, dy)
    if best[0] < old_penalty - 1e-6:
        return best
    return None


def repair_furniture_intersections_in_scene(
    scene: dict[str, Any],
    *,
    max_passes: int = 3,
    search_step_m: float = 0.15,
    max_shift_m: float = 1.2,
) -> tuple[dict[str, Any], dict[str, Any]]:
    updated = deepcopy(scene)
    room = updated.get("room") if isinstance(updated.get("room"), dict) else {}
    poly = _scene_room_polygon(room)
    placements = updated.get("placements") if isinstance(updated.get("placements"), list) else []
    info: dict[str, Any] = {"enabled": True, "passes": [], "moved": []}
    if len(poly) < 3 or not placements:
        info["skipped_reason"] = "no_room_polygon_or_no_placements"
        return updated, info

    movable_indices = {idx for idx, item in enumerate(placements) if isinstance(item, dict) and _is_movable_furniture_item(item)}
    rects = [_item_rect_xy(item) if isinstance(item, dict) else None for item in placements]
    children_by_anchor = _support_children_by_anchor(placements=placements, rects=rects)
    info["support_groups"] = [
        {
            "anchor_id": placements[idx].get("id") if isinstance(placements[idx], dict) else idx,
            "child_ids": [
                placements[child_idx].get("id") if isinstance(placements[child_idx], dict) else child_idx
                for child_idx in sorted(children)
            ],
        }
        for idx, children in sorted(children_by_anchor.items())
        if children
    ]
    for pass_idx in range(max(1, max_passes)):
        trouble: set[int] = set()
        for idx in movable_indices:
            rect = rects[idx]
            if rect is not None and _rect_outside_room_area(rect, poly) > 1e-6:
                trouble.add(idx)
        movable_list = sorted(movable_indices)
        for pos_i, i in enumerate(movable_list):
            ri = rects[i]
            if ri is None:
                continue
            for j in movable_list[pos_i + 1 :]:
                rj = rects[j]
                if rj is None:
                    continue
                if _rect_intersection_area(ri, rj) > 1e-6:
                    trouble.add(i if _rect_area(ri) <= _rect_area(rj) else j)
        pass_info = {"pass": pass_idx + 1, "trouble_count": len(trouble), "accepted": []}
        if not trouble:
            info["passes"].append(pass_info)
            break
        for idx in sorted(trouble):
            group_indices = _move_group_indices(idx, children_by_anchor)
            move = _best_repair_shift_for_item(
                item_index=idx,
                group_indices=group_indices,
                rects=rects,
                movable_indices=movable_indices,
                poly=poly,
                search_step_m=search_step_m,
                max_shift_m=max_shift_m,
            )
            if move is None:
                continue
            new_penalty, old_overlap, dx, dy = move
            item = placements[idx]
            old_rect = rects[idx]
            if old_rect is None or not isinstance(item, dict):
                continue
            moved_ids = []
            for move_idx in sorted(group_indices):
                move_item = placements[move_idx]
                move_rect = rects[move_idx]
                if move_rect is None or not isinstance(move_item, dict):
                    continue
                _shift_item_xy(move_item, dx, dy)
                rects[move_idx] = _rect_shift(move_rect, dx, dy)
                meta = move_item.setdefault("meta", {})
                if isinstance(meta, dict):
                    meta["furniture_overlap_repaired"] = True
                    meta["furniture_repair_anchor_id"] = item.get("id")
                    meta["furniture_repair_group_move"] = move_idx != idx
                moved_ids.append(move_item.get("id"))
            accepted = {
                "id": item.get("id"),
                "moved_ids": moved_ids,
                "dx": round(dx, 4),
                "dy": round(dy, 4),
                "move_m": round(math.hypot(dx, dy), 4),
                "new_penalty": round(new_penalty, 6),
                "old_new_overlap_area_m2": round(old_overlap, 6),
            }
            pass_info["accepted"].append(accepted)
            info["moved"].append(accepted)
        info["passes"].append(pass_info)
        if not pass_info["accepted"]:
            break
    info["moved_count"] = len(info["moved"])
    return updated, info


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
) -> dict[str, Any]:
    supplier_placement_v1 = run_dir / "placement_supplier.v1.json"
    apply_supplier_bindings_to_json(
        input_json_path=artifacts.placement_v1,
        bindings_json_path=bindings_json_path,
        output_json_path=supplier_placement_v1,
        require_local_asset=require_local_asset,
    )

    supplier_scene_v1 = None
    if artifacts.scene_v1 and artifacts.scene_v1.is_file():
        supplier_scene_v1 = run_dir / "scene_supplier.v1.json"
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
) -> dict[str, Any]:
    scene_v1 = supplier_info.get("scene_v1")
    return write_supplier_replacement_reports(
        bindings_json_path=bindings_json_path,
        run_dir=run_dir,
        supplier_scene_json_path=str(scene_v1) if scene_v1 else None,
    )


def _parse_elevations(raw: str) -> list[float]:
    out: list[float] = []
    for chunk in str(raw or "").split(","):
        chunk = chunk.strip()
        if chunk:
            out.append(float(chunk))
    return out or [0.0, 30.0, 45.0]


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
    if not supplier_blend_path.is_file():
        return None
    if supplier_blend_path.name != "scene_infinigen_clean_supplier.blend":
        return None

    elevations = _parse_elevations(str(getattr(args, "supplier_gif_elevations", "0,30,45") or "0,30,45"))
    frames = int(getattr(args, "supplier_gif_frames", 36) or 36)
    fps = int(getattr(args, "supplier_gif_fps", 8) or 8)
    keep_frames = bool(getattr(args, "keep_supplier_gif_frames", False))
    out: list[dict[str, Any]] = []

    for elevation in elevations:
        suffix = f"elev_{int(round(elevation)):02d}"
        frame_dir = run_dir / f"_frames_supplier_interior_{suffix}"
        gif_path = run_dir / f"room_supplier.interior.{suffix}.gif"
        if frame_dir.exists():
            shutil.rmtree(frame_dir, ignore_errors=True)

        cmd = [
            sys.executable,
            cfg_runtime["BLENDER_VIS_SCRIPT"],
            "--json",
            str(supplier_scene_json_path.resolve()),
            "--reference-blend",
            str(supplier_blend_path.resolve()),
            "--background",
            "--hide-room-shell",
            "--no-bbox-fallback",
            "--turntable-render-dir",
            str(frame_dir.resolve()),
            "--turntable-frames",
            str(frames),
            "--turntable-elevation-deg",
            str(float(elevation)),
            "--no-pack-assets",
        ]
        if args.blender:
            cmd += ["--blender", args.blender]

        print("▶ Supplier room GIF:\n ", " ".join(cmd))
        subprocess.run(cmd, check=True)
        _render_gif_from_frames(frame_dir, gif_path, fps)
        if not keep_frames:
            shutil.rmtree(frame_dir, ignore_errors=True)
        out.append(
            {
                "elevation_deg": float(elevation),
                "gif": str(gif_path.resolve()),
                "frames_dir": str(frame_dir.resolve()) if keep_frames else None,
            }
        )

    return {
        "supplier_scene_json": str(supplier_scene_json_path.resolve()),
        "supplier_blend": str(supplier_blend_path.resolve()),
        "hide_room_shell": True,
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
    out_suffix = "llm" if supplier_llm_provider != "none" else "heuristic"
    if selection_strategy and selection_strategy != "balanced":
        out_suffix = f"{out_suffix}.{selection_strategy}"
    out_path = run_dir / f"base_supplier_bindings.{out_suffix}.json"
    result = build_bindings_with_candidates(
        targets_json_path=Path(layout_targets_json_path).expanduser().resolve(),
        catalog_rows=catalog_rows,
        top_k=int(args.supplier_top_k),
        selection_strategy=str(getattr(args, "supplier_selection_strategy", "balanced") or "balanced"),
        user_preferences=supplier_user_preferences,
        llm_settings=llm_settings,
    )
    write_json(out_path, result)
    return out_path


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
    solve_large = getattr(args, "infinigen_solve_steps_large", None)
    solve_medium = getattr(args, "infinigen_solve_steps_medium", None)
    solve_small = getattr(args, "infinigen_solve_steps_small", None)
    if not fast_small and solve_large is None and solve_medium is None and solve_small is None:
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

        supplier_scene_for_render: Optional[Path] = None
        supplier_bindings_path = _resolve_supplier_bindings_json(
            args=args,
            run_dir=run_dir,
            layout_targets_json_path=lego_selection_stub["layout_targets_json"],
            supplier_user_preferences_json=(
                str(style_supplier_preferences_path.resolve())
                if style_supplier_preferences_path and not str(getattr(args, "supplier_user_preferences_json", "") or "").strip()
                else None
            ),
        )
        if supplier_bindings_path:
            supplier_bindings_path, supplier_assets_info = _acquire_supplier_assets_for_bindings(
                args=args,
                run_dir=run_dir,
                bindings_json_path=supplier_bindings_path,
            )
            supplier_info = _apply_supplier_bindings_for_artifacts(
                artifacts=lego_artifacts,
                run_dir=run_dir,
                bindings_json_path=supplier_bindings_path,
                require_local_asset=bool(args.supplier_require_local_asset),
            )
            supplier_report_info = _write_supplier_replacement_reports_for_artifacts(
                run_dir=run_dir,
                bindings_json_path=supplier_bindings_path,
                supplier_info=supplier_info,
            )
            manifest["supplier_rebind"] = supplier_info
            manifest["supplier_assets"] = supplier_assets_info
            manifest["supplier_replacement_reports"] = supplier_report_info
            if supplier_info.get("scene_v1"):
                supplier_scene_for_render = Path(str(supplier_info["scene_v1"])).expanduser().resolve()
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
            supplier_report_info = _write_supplier_replacement_reports_for_artifacts(
                run_dir=run_dir,
                bindings_json_path=supplier_bindings_path,
                supplier_info=supplier_info,
            )
            manifest["supplier_replacement_reports"] = supplier_report_info
        write_json(manifest_path, manifest)

        if args.skip_blender:
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

        if supplier_scene_for_render and supplier_scene_for_render.is_file():
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
    supplier_bindings_path = _resolve_supplier_bindings_json(
        args=args,
        run_dir=run_dir,
        layout_targets_json_path=base_selection_stub["layout_targets_json"],
        supplier_user_preferences_json=(
            str(style_supplier_preferences_path.resolve())
            if style_supplier_preferences_path and not str(getattr(args, "supplier_user_preferences_json", "") or "").strip()
            else None
        ),
    )
    if supplier_bindings_path:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        supplier_bindings_path, supplier_assets_info = _acquire_supplier_assets_for_bindings(
            args=args,
            run_dir=run_dir,
            bindings_json_path=supplier_bindings_path,
        )
        supplier_info = _apply_supplier_bindings_for_artifacts(
            artifacts=base_artifacts,
            run_dir=run_dir,
            bindings_json_path=supplier_bindings_path,
            require_local_asset=bool(args.supplier_require_local_asset),
        )
        supplier_report_info = _write_supplier_replacement_reports_for_artifacts(
            run_dir=run_dir,
            bindings_json_path=supplier_bindings_path,
            supplier_info=supplier_info,
        )
        manifest["supplier_rebind"] = supplier_info
        manifest["supplier_assets"] = supplier_assets_info
        manifest["supplier_replacement_reports"] = supplier_report_info
        if supplier_info.get("scene_v1"):
            supplier_scene_for_render = Path(str(supplier_info["scene_v1"])).expanduser().resolve()
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
        supplier_report_info = _write_supplier_replacement_reports_for_artifacts(
            run_dir=run_dir,
            bindings_json_path=supplier_bindings_path,
            supplier_info=supplier_info,
        )
        manifest["supplier_replacement_reports"] = supplier_report_info
    write_json(manifest_path, manifest)

    if args.skip_blender:
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

    if supplier_scene_for_render and supplier_scene_for_render.is_file():
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
    p.add_argument("--skip-supplier-gif", action="store_true", help="Disable supplier-only room GIF generation")
    p.add_argument("--supplier-gif-frames", type=int, default=36)
    p.add_argument("--supplier-gif-elevations", default="0,30,45")
    p.add_argument("--supplier-gif-fps", type=int, default=8)
    p.add_argument("--keep-supplier-gif-frames", action="store_true")

    p.add_argument("--lego-postprocess", action="store_true")
    p.add_argument("--infinigen-src", default=None)
    p.add_argument(
        "--infinigen-fast-small",
        action="store_true",
        help="Disable Infinigen small-object solve stage and lower loose/surface object density",
    )
    p.add_argument("--infinigen-solve-steps-large", type=int, default=None)
    p.add_argument("--infinigen-solve-steps-medium", type=int, default=None)
    p.add_argument("--infinigen-solve-steps-small", type=int, default=None)
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
