#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from copy import deepcopy
import math
from pathlib import Path
from typing import Any

try:
    from .pipeline_artifacts import read_json, write_json
except ImportError:
    from pipeline_artifacts import read_json, write_json


LOW_QUALITY_ASSET_STATUSES = {"proxy_generated_with_blender", "needs_blender_rebuild"}
SELECTED_BINDING_STATUSES = {"heuristic_top1_selected", "llm_reranked_top1_selected"}


def _infer_reference_scene_blend_path(input_json_path: str | Path, data: dict[str, Any]) -> str | None:
    input_path = Path(input_json_path).expanduser().resolve()
    json_dir = input_path.parent

    meta = data.get("meta") or {}
    placement_meta = meta.get("placement_meta") or {}
    raw_scene_blend = str(placement_meta.get("scene_blend") or "").strip()

    candidates: list[Path] = []
    candidates.append(json_dir / "scene_infinigen_clean.blend")
    candidates.append(json_dir / "infinigen_clean_scene.blend")
    if raw_scene_blend:
        raw_path = Path(raw_scene_blend).expanduser()
        candidates.append(raw_path)
        candidates.append(json_dir / raw_path.name)

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            return str(candidate.resolve())
    return None


def _candidate_size_m(candidate: dict[str, Any]) -> list[float] | None:
    try:
        width = candidate.get("width_cm")
        depth = candidate.get("depth_cm")
        height = candidate.get("height_cm")
        if width is None or depth is None or height is None:
            return None
        return [float(width) / 100.0, float(depth) / 100.0, float(height) / 100.0]
    except Exception:
        return None


def _aabb_center(aabb: dict[str, Any]) -> list[float]:
    return [
        0.5 * (float(aabb["x_min"]) + float(aabb["x_max"])),
        0.5 * (float(aabb["y_min"]) + float(aabb["y_max"])),
        0.5 * (float(aabb["z_min"]) + float(aabb["z_max"])),
    ]


def _item_aabb(item: dict[str, Any]) -> dict[str, float] | None:
    aabb = item.get("aabb") or {}
    if isinstance(aabb, dict) and all(k in aabb for k in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")):
        return {
            "x_min": float(aabb["x_min"]),
            "x_max": float(aabb["x_max"]),
            "y_min": float(aabb["y_min"]),
            "y_max": float(aabb["y_max"]),
            "z_min": float(aabb["z_min"]),
            "z_max": float(aabb["z_max"]),
        }
    pos = item.get("position_m")
    size = item.get("size_m")
    if not (isinstance(pos, list) and len(pos) == 3 and isinstance(size, list) and len(size) == 3):
        return None
    cx, cy, cz = [float(v) for v in pos]
    sx, sy, sz = [max(float(v), 0.0) for v in size]
    return {
        "x_min": cx - 0.5 * sx,
        "x_max": cx + 0.5 * sx,
        "y_min": cy - 0.5 * sy,
        "y_max": cy + 0.5 * sy,
        "z_min": cz - 0.5 * sz,
        "z_max": cz + 0.5 * sz,
    }


def _xy_inside_expanded(aabb: dict[str, float], point_xy: list[float], margin: float) -> bool:
    px, py = float(point_xy[0]), float(point_xy[1])
    return (
        aabb["x_min"] - margin <= px <= aabb["x_max"] + margin
        and aabb["y_min"] - margin <= py <= aabb["y_max"] + margin
    )


def _aabb_overlaps_expanded(
    anchor_aabb: dict[str, float],
    item_aabb: dict[str, float],
    *,
    x_margin: float,
    y_margin: float,
    z_margin_below: float,
    z_margin_above: float,
) -> bool:
    return (
        item_aabb["x_max"] >= anchor_aabb["x_min"] - x_margin
        and item_aabb["x_min"] <= anchor_aabb["x_max"] + x_margin
        and item_aabb["y_max"] >= anchor_aabb["y_min"] - y_margin
        and item_aabb["y_min"] <= anchor_aabb["y_max"] + y_margin
        and item_aabb["z_max"] >= anchor_aabb["z_min"] - z_margin_below
        and item_aabb["z_min"] <= anchor_aabb["z_max"] + z_margin_above
    )


def _item_position(item: dict[str, Any]) -> list[float] | None:
    pos = item.get("position_m")
    if isinstance(pos, list) and len(pos) == 3:
        return [float(pos[0]), float(pos[1]), float(pos[2])]
    aabb = _item_aabb(item)
    if aabb is None:
        return None
    return _aabb_center(aabb)


def _selected_supplier_binding(binding: dict[str, Any] | None) -> bool:
    if not isinstance(binding, dict):
        return False
    chosen = binding.get("chosen_candidate")
    if not isinstance(chosen, dict):
        return False
    if str(binding.get("selection_status") or "") not in SELECTED_BINDING_STATUSES:
        return False
    return ((binding.get("provenance") or {}).get("final_asset_source")) in {"supplier_catalog", "supplier_catalog_pending"}


def _binding_has_supported_local_asset(binding: dict[str, Any] | None) -> bool:
    if not _selected_supplier_binding(binding):
        return False
    chosen = binding.get("chosen_candidate") or {}
    local_path = str(chosen.get("asset_local_path") or "").strip()
    asset_format = str(chosen.get("asset_format") or "").strip().lower()
    asset_status = str(chosen.get("asset_status") or "").strip().lower()
    return bool(
        local_path
        and Path(local_path).expanduser().is_file()
        and asset_format in {"obj", "fbx", "glb", "gltf"}
        and asset_status not in LOW_QUALITY_ASSET_STATUSES
    )


def _normalized_anchor_xy(anchor_aabb: dict[str, float], item_center: list[float]) -> tuple[float, float]:
    width = max(anchor_aabb["x_max"] - anchor_aabb["x_min"], 1e-6)
    depth = max(anchor_aabb["y_max"] - anchor_aabb["y_min"], 1e-6)
    rel_x = (float(item_center[0]) - anchor_aabb["x_min"]) / width
    rel_y = (float(item_center[1]) - anchor_aabb["y_min"]) / depth
    return max(0.02, min(0.98, rel_x)), max(0.02, min(0.98, rel_y))


def _normalized_anchor_z(anchor_aabb: dict[str, float], item_aabb: dict[str, float]) -> float:
    height = max(anchor_aabb["z_max"] - anchor_aabb["z_min"], 1e-6)
    rel_z = (float(item_aabb["z_min"]) - anchor_aabb["z_min"]) / height
    return max(0.02, min(0.98, rel_z))


def _related_generated_item_actions(
    placements: list[dict[str, Any]],
    by_target_id: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    actions: dict[str, dict[str, Any]] = {}
    anchor_items: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for item in placements:
        item_id = str(item.get("id") or "").strip()
        binding = by_target_id.get(item_id)
        if _binding_has_supported_local_asset(binding):
            anchor_items.append((item, binding))

    support_surface_groups = {"desk", "side_table", "nightstand", "dresser", "shelf", "tv_stand", "wardrobe"}
    storage_volume_groups = {"dresser", "shelf", "tv_stand", "wardrobe"}
    tabletop_categories = {
        "BookStackFactory",
        "BookColumnFactory",
        "NatureShelfTrinketsFactory",
        "DeskLampFactory",
    }
    bed_soft_categories = {
        "BlanketFactory",
        "BoxComforterFactory",
        "PillowFactory",
        "MattressFactory",
        "TowelFactory",
    }

    for anchor_item, binding in anchor_items:
        anchor_id = str(anchor_item.get("id") or "").strip()
        anchor_group = str(binding.get("semantic_group") or "").strip().lower()
        anchor_aabb = _item_aabb(anchor_item)
        if not anchor_id or anchor_aabb is None:
            continue
        anchor_top = float(anchor_aabb["z_max"])

        for item in placements:
            item_id = str(item.get("id") or "").strip()
            if not item_id or item_id == anchor_id or item_id in actions:
                continue
            if _binding_has_supported_local_asset(by_target_id.get(item_id)):
                continue

            item_aabb = _item_aabb(item)
            item_pos = _item_position(item)
            if item_aabb is None or item_pos is None:
                continue
            category = str(item.get("category") or "").strip()

            if anchor_group == "bed" and category in bed_soft_categories:
                if _xy_inside_expanded(anchor_aabb, item_pos[:2], margin=0.18) and item_aabb["z_min"] <= anchor_top + 0.25:
                    actions[item_id] = {"action": "suppress", "anchor_id": anchor_id}
                continue

            if anchor_group in support_surface_groups and category in tabletop_categories:
                if anchor_group in storage_volume_groups:
                    inside_or_touching_storage = _aabb_overlaps_expanded(
                        anchor_aabb,
                        item_aabb,
                        x_margin=0.12,
                        y_margin=0.12,
                        z_margin_below=0.05,
                        z_margin_above=0.22,
                    )
                    on_storage_top = (
                        _xy_inside_expanded(anchor_aabb, item_pos[:2], margin=0.14)
                        and anchor_top - 0.08 <= item_aabb["z_min"] <= anchor_top + 0.4
                    )
                    if inside_or_touching_storage or on_storage_top:
                        rel_x, rel_y = _normalized_anchor_xy(anchor_aabb, item_pos)
                        actions[item_id] = {
                            "action": "reanchor",
                            "anchor_id": anchor_id,
                            "anchor_group": anchor_group,
                            "support_mode": "top" if on_storage_top else "volume",
                            "anchor_aabb": deepcopy(anchor_aabb),
                            "rel_x": rel_x,
                            "rel_y": rel_y,
                            "rel_z": _normalized_anchor_z(anchor_aabb, item_aabb),
                        }
                    continue

                on_top = (
                    _xy_inside_expanded(anchor_aabb, item_pos[:2], margin=0.14)
                    and anchor_top - 0.08 <= item_aabb["z_min"] <= anchor_top + 0.45
                )
                if on_top:
                    rel_x, rel_y = _normalized_anchor_xy(anchor_aabb, item_pos)
                    actions[item_id] = {
                        "action": "reanchor",
                        "anchor_id": anchor_id,
                        "anchor_group": anchor_group,
                        "support_mode": "top",
                        "anchor_aabb": deepcopy(anchor_aabb),
                        "rel_x": rel_x,
                        "rel_y": rel_y,
                        "rel_z": _normalized_anchor_z(anchor_aabb, item_aabb),
                    }

    return actions


def _rotation_aware_world_size(size_m: list[float], rotation_deg: float) -> list[float]:
    sx, sy, sz = [max(float(v), 1e-6) for v in size_m]
    theta = math.radians(float(rotation_deg or 0.0))
    c = abs(math.cos(theta))
    s = abs(math.sin(theta))
    return [sx * c + sy * s, sx * s + sy * c, sz]


def _apply_geometry_from_candidate(item: dict[str, Any], candidate_size_m: list[float]) -> None:
    selected_size = [max(float(v), 1e-6) for v in candidate_size_m]
    rotation_deg = float(item.get("rotation_deg", item.get("yaw_deg", 0.0)) or 0.0)
    world_size = _rotation_aware_world_size(selected_size, rotation_deg)
    constraints = item.get("constraints") or {}
    source_aabb = item.get("aabb") or {}

    if isinstance(source_aabb, dict) and all(k in source_aabb for k in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")):
        cx, cy, cz = _aabb_center(source_aabb)
        z_min_prev = float(source_aabb["z_min"])
        z_max_prev = float(source_aabb["z_max"])
    else:
        pos = item.get("position_m") or [0.0, 0.0, 0.0]
        cx, cy, cz = [float(pos[i]) for i in range(3)]
        z_min_prev = cz - 0.5 * world_size[2]
        z_max_prev = cz + 0.5 * world_size[2]

    mount_type = str((constraints or {}).get("mount_type") or "").strip().lower()
    if mount_type == "ceiling":
        z_max = z_max_prev
        z_min = z_max - world_size[2]
    elif mount_type == "wall":
        z_min = cz - 0.5 * world_size[2]
        z_max = cz + 0.5 * world_size[2]
    else:
        z_min = z_min_prev
        z_max = z_min + world_size[2]

    item["size_m"] = selected_size
    item["position_m"] = [cx, cy, 0.5 * (z_min + z_max)]
    item["aabb"] = {
        "x_min": cx - 0.5 * world_size[0],
        "x_max": cx + 0.5 * world_size[0],
        "y_min": cy - 0.5 * world_size[1],
        "y_max": cy + 0.5 * world_size[1],
        "z_min": z_min,
        "z_max": z_max,
    }


def _item_has_scene_geometry(item: dict[str, Any]) -> bool:
    aabb = item.get("aabb") or {}
    if isinstance(aabb, dict) and all(k in aabb for k in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")):
        return True
    size_m = item.get("size_m")
    return isinstance(size_m, list) and len(size_m) == 3


def _candidate_asset(candidate: dict[str, Any], require_local_asset: bool) -> tuple[dict[str, Any], bool]:
    local_path = str(candidate.get("asset_local_path") or "").strip()
    asset_format = str(candidate.get("asset_format") or "").strip().lower()
    asset_status = str(candidate.get("asset_status") or "").strip().lower()
    if (
        local_path
        and Path(local_path).expanduser().is_file()
        and asset_format in {"obj", "fbx", "glb", "gltf"}
        and asset_status not in LOW_QUALITY_ASSET_STATUSES
    ):
        return (
            {
                "mesh_path": str(Path(local_path).expanduser().resolve()),
                "mesh_fit_mode": "uniform",
            },
            False,
        )
    if require_local_asset:
        return {}, False
    return {}, True


def _replacement_mesh_fit_mode(binding: dict[str, Any], item: dict[str, Any]) -> str:
    support_groups = {
        "desk",
        "dresser",
        "nightstand",
        "side_table",
        "coffee_table",
        "shelf",
        "tv_stand",
        "wardrobe",
    }
    group = str(binding.get("semantic_group") or "").strip().lower()
    category = str(item.get("category") or "").strip().lower()
    constraints = item.get("constraints") or {}
    mount_type = str((constraints or {}).get("mount_type") or "").strip().lower()
    if group == "lamp_ceiling" or mount_type == "ceiling":
        return "stretch"
    if group in support_groups:
        return "stretch"
    if category in {
        "simpledeskfactory",
        "singlecabinetfactory",
        "cellshelffactory",
        "simplebookcasefactory",
    }:
        return "stretch"
    return "uniform"


def _should_keep_original_scene_item(item: dict[str, Any], binding: dict[str, Any]) -> bool:
    category = str(item.get("category") or "").strip().lower()
    semantic_group = str(binding.get("semantic_group") or "").strip().lower()
    if category in {
        "ceilinglightfactory",
        "floorlampfactory",
        "desklampfactory",
    }:
        return True
    if semantic_group in {
        "lamp_ceiling",
        "lamp_floor",
        "lamp_table",
    }:
        return True
    return False


def _should_apply_candidate_geometry(
    item: dict[str, Any],
    binding: dict[str, Any],
    candidate_size_m: list[float] | None,
    asset_block: dict[str, Any],
) -> bool:
    if candidate_size_m is None or not asset_block:
        return False
    constraints = item.get("constraints") or {}
    mount_type = str((constraints or {}).get("mount_type") or "").strip().lower()
    semantic_group = str(binding.get("semantic_group") or "").strip().lower()
    if mount_type == "ceiling" or semantic_group == "lamp_ceiling":
        return True
    return not _item_has_scene_geometry(item)


def _compact_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "unique_key",
        "source_site",
        "title",
        "brand",
        "collection",
        "category_raw",
        "category_norm",
        "semantic_group",
        "product_url",
        "model_page_url",
        "model_download_url",
        "model_download_landing_url",
        "model_vendor_url",
        "asset_status",
        "asset_format",
        "asset_local_path",
        "price_value",
        "price_currency",
        "style",
        "color",
        "materials",
        "width_cm",
        "depth_cm",
        "height_cm",
        "description",
        "score_breakdown",
    ]
    return {k: deepcopy(candidate.get(k)) for k in keys if k in candidate}


def apply_supplier_bindings_to_data(
    data: dict[str, Any],
    bindings_data: dict[str, Any],
    *,
    require_local_asset: bool = False,
) -> dict[str, Any]:
    out = deepcopy(data)
    placements = out.get("placements")
    if not isinstance(placements, list):
        placements = out.get("items")
    if not isinstance(placements, list):
        raise RuntimeError("Некорректный scene/placement JSON: нет placements/items")

    bindings = bindings_data.get("bindings") or []
    if not isinstance(bindings, list):
        raise RuntimeError("Некорректный supplier_bindings JSON: нет bindings")

    by_target_id = {
        str(b.get("target_id") or "").strip(): b
        for b in bindings
        if isinstance(b, dict) and str(b.get("target_id") or "").strip()
    }
    related_item_actions = _related_generated_item_actions(placements, by_target_id)

    replaced = 0
    placeholder_replaced = 0
    local_asset_replaced = 0
    suppressed_generated_count = 0

    new_items: list[dict[str, Any]] = []
    for item in placements:
        if not isinstance(item, dict):
            new_items.append(item)
            continue

        item_id = str(item.get("id") or "").strip()
        binding = by_target_id.get(item_id)
        chosen = binding.get("chosen_candidate") if isinstance(binding, dict) else None
        related_action = related_item_actions.get(item_id) if item_id else None
        if related_action and related_action.get("action") == "suppress" and not _selected_supplier_binding(binding):
            suppressed_generated_count += 1
            continue
        if not binding or not isinstance(chosen, dict):
            new_items.append(item)
            continue
        if str(binding.get("selection_status") or "") not in SELECTED_BINDING_STATUSES:
            new_items.append(item)
            continue
        if ((binding.get("provenance") or {}).get("final_asset_source")) not in {"supplier_catalog", "supplier_catalog_pending"}:
            new_items.append(item)
            continue
        if _should_keep_original_scene_item(item, binding):
            new_items.append(item)
            continue

        candidate_size_m = _candidate_size_m(chosen)

        asset_block, use_placeholder = _candidate_asset(chosen, require_local_asset=require_local_asset)
        if require_local_asset and not asset_block:
            new_items.append(item)
            continue
        if not asset_block and _item_has_scene_geometry(item):
            new_items.append(item)
            continue

        updated = deepcopy(item)
        original_name = updated.get("name")
        original_category = updated.get("category")

        updated["name"] = str(chosen.get("title") or original_name or "supplier_object")
        updated["category"] = str(binding.get("category") or original_category or updated["name"])
        updated["asset"] = asset_block
        # Preserve the original layout geometry whenever the generated scene
        # already has a concrete placement. Supplier assets should replace the
        # mesh, not collapse the scene to catalog dimensions.
        if _should_apply_candidate_geometry(updated, binding, candidate_size_m, asset_block):
            _apply_geometry_from_candidate(updated, candidate_size_m)
        if updated["asset"]:
            updated["asset"]["mesh_fit_mode"] = _replacement_mesh_fit_mode(binding, updated)

        source = deepcopy(updated.get("source") or {})
        source["asset_source"] = "supplier_catalog_local_asset" if asset_block else "supplier_catalog_placeholder"
        source["supplier_replaced"] = True
        source["supplier_target_id"] = item_id
        source["supplier_unique_key"] = chosen.get("unique_key")
        source["supplier_source_site"] = chosen.get("source_site")
        source["supplier_product_url"] = chosen.get("product_url") or chosen.get("model_page_url")
        source["supplier_model_url"] = chosen.get("model_download_url")
        source["placeholder_bbox"] = bool(use_placeholder)
        updated["source"] = source

        meta = deepcopy(updated.get("meta") or {})
        meta["placeholder_bbox"] = bool(use_placeholder)
        meta["supplier_binding_applied"] = True
        meta["supplier_binding_target_id"] = item_id
        meta["supplier_candidate"] = _compact_candidate(chosen)
        meta["supplier_selection_notes"] = deepcopy(binding.get("selection_notes") or [])
        meta["original_generated_item"] = {
            "id": item_id,
            "name": original_name,
            "category": original_category,
        }
        updated["meta"] = meta

        replaced += 1
        if use_placeholder:
            placeholder_replaced += 1
        else:
            local_asset_replaced += 1

        new_items.append(updated)
        continue

        
        
        
    # Re-anchor surviving generated decor/books/desk lamps to the updated
    # support AABBs of supplier-replaced furniture instead of deleting them.
    reanchored_count = 0
    final_anchor_aabbs: dict[str, dict[str, float]] = {}
    for item in new_items:
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("id") or "").strip()
        if not item_id:
            continue
        binding = by_target_id.get(item_id)
        if not _binding_has_supported_local_asset(binding):
            continue
        aabb = _item_aabb(item)
        if aabb is not None:
            final_anchor_aabbs[item_id] = aabb

    adjusted_items: list[dict[str, Any]] = []
    for item in new_items:
        if not isinstance(item, dict):
            adjusted_items.append(item)
            continue
        item_id = str(item.get("id") or "").strip()
        related_action = related_item_actions.get(item_id) if item_id else None
        if not related_action or related_action.get("action") != "reanchor":
            adjusted_items.append(item)
            continue

        anchor_id = str(related_action.get("anchor_id") or "").strip()
        anchor_aabb_new = final_anchor_aabbs.get(anchor_id)
        anchor_aabb_old = related_action.get("anchor_aabb")
        item_aabb_old = _item_aabb(item)
        if not anchor_id or not isinstance(anchor_aabb_old, dict) or anchor_aabb_new is None or item_aabb_old is None:
            adjusted_items.append(item)
            continue

        rel_x = float(related_action.get("rel_x", 0.5))
        rel_y = float(related_action.get("rel_y", 0.5))
        rel_z = float(related_action.get("rel_z", 0.5))
        support_mode = str(related_action.get("support_mode") or "top").strip().lower()
        anchor_group = str(related_action.get("anchor_group") or "").strip().lower()

        item_size = [
            max(item_aabb_old["x_max"] - item_aabb_old["x_min"], 1e-6),
            max(item_aabb_old["y_max"] - item_aabb_old["y_min"], 1e-6),
            max(item_aabb_old["z_max"] - item_aabb_old["z_min"], 1e-6),
        ]
        new_width = max(anchor_aabb_new["x_max"] - anchor_aabb_new["x_min"], 1e-6)
        new_depth = max(anchor_aabb_new["y_max"] - anchor_aabb_new["y_min"], 1e-6)
        new_height = max(anchor_aabb_new["z_max"] - anchor_aabb_new["z_min"], 1e-6)

        cx = anchor_aabb_new["x_min"] + rel_x * new_width
        cy = anchor_aabb_new["y_min"] + rel_y * new_depth

        support_sink = 0.025
        if anchor_group in {"shelf", "wardrobe"}:
            support_sink = 0.055
        elif anchor_group in {"dresser", "tv_stand"}:
            support_sink = 0.035

        if support_mode == "top":
            z_min = anchor_aabb_new["z_max"] - support_sink
        else:
            usable_height = max(new_height - 0.14, item_size[2] + 0.02)
            z_min = anchor_aabb_new["z_min"] + 0.07 + rel_z * usable_height
            z_min = min(z_min, anchor_aabb_new["z_max"] - item_size[2] - 0.03)
        z_max = z_min + item_size[2]

        updated = deepcopy(item)
        updated["position_m"] = [cx, cy, 0.5 * (z_min + z_max)]
        updated["size_m"] = item.get("size_m") or item_size
        updated["aabb"] = {
            "x_min": cx - 0.5 * item_size[0],
            "x_max": cx + 0.5 * item_size[0],
            "y_min": cy - 0.5 * item_size[1],
            "y_max": cy + 0.5 * item_size[1],
            "z_min": z_min,
            "z_max": z_max,
        }
        meta = deepcopy(updated.get("meta") or {})
        meta["supplier_support_anchor_target_id"] = anchor_id
        meta["supplier_support_mode"] = support_mode
        meta["supplier_support_anchor_group"] = anchor_group
        meta["supplier_support_reanchored"] = True
        updated["meta"] = meta
        reanchored_count += 1
        adjusted_items.append(updated)

    new_items = adjusted_items

    if "placements" in out and isinstance(out.get("placements"), list):
        out["placements"] = new_items
    elif "items" in out and isinstance(out.get("items"), list):
        out["items"] = new_items

    meta = deepcopy(out.get("meta") or {})
    meta["supplier_binding_summary"] = {
        "replaced_count": replaced,
        "placeholder_replaced_count": placeholder_replaced,
        "local_asset_replaced_count": local_asset_replaced,
        "suppressed_generated_related_count": suppressed_generated_count,
        "reanchored_generated_related_count": reanchored_count,
        "require_local_asset": bool(require_local_asset),
    }
    out["meta"] = meta
    return out


def apply_supplier_bindings_to_json(
    input_json_path: str | Path,
    bindings_json_path: str | Path,
    output_json_path: str | Path,
    *,
    require_local_asset: bool = False,
) -> Path:
    data = read_json(input_json_path)
    bindings = read_json(bindings_json_path)
    out = apply_supplier_bindings_to_data(data, bindings, require_local_asset=require_local_asset)
    reference_scene_blend = _infer_reference_scene_blend_path(input_json_path, data if isinstance(data, dict) else {})
    if reference_scene_blend:
        meta = deepcopy(out.get("meta") or {})
        placement_meta = deepcopy(meta.get("placement_meta") or {})
        placement_meta["scene_blend"] = reference_scene_blend
        meta["placement_meta"] = placement_meta
        out["meta"] = meta
    output_path = Path(output_json_path).expanduser().resolve()
    write_json(output_path, out)
    return output_path


def build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Apply supplier_bindings to scene/placement JSON.")
    ap.add_argument("--input-json", required=True, help="scene.v1.json or placement.v1.json")
    ap.add_argument("--bindings-json", required=True, help="supplier_bindings json with chosen_candidate")
    ap.add_argument("--out", required=True, help="Output JSON path")
    ap.add_argument("--require-local-asset", action="store_true", help="Replace only when candidate has local OBJ asset")
    return ap


def main() -> None:
    args = build_cli().parse_args()
    out_path = apply_supplier_bindings_to_json(
        input_json_path=args.input_json,
        bindings_json_path=args.bindings_json,
        output_json_path=args.out,
        require_local_asset=bool(args.require_local_asset),
    )
    data = read_json(out_path)
    summary = (data.get("meta") or {}).get("supplier_binding_summary") or {}
    print(f"replaced = {summary.get('replaced_count', 0)}")
    print(f"placeholder_replaced = {summary.get('placeholder_replaced_count', 0)}")
    print(f"local_asset_replaced = {summary.get('local_asset_replaced_count', 0)}")
    print(f"saved = {out_path}")


if __name__ == "__main__":
    main()
