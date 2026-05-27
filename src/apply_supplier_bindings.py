#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from copy import deepcopy
import math
from pathlib import Path
import sqlite3
from typing import Any

try:
    from .pipeline_artifacts import read_json, write_json
except ImportError:
    from pipeline_artifacts import read_json, write_json

ASSET_FALLBACK_MODE_NONE = "none"
ASSET_FALLBACK_MODE_FBX_OBJ_PROXY = "fbx_obj_proxy"
ASSET_FALLBACK_MODE_FBX_OBJ_TRELLIS_PROXY = "fbx_obj_trellis_proxy"
ASSET_FALLBACK_MODES = {
    ASSET_FALLBACK_MODE_NONE,
    ASSET_FALLBACK_MODE_FBX_OBJ_PROXY,
    ASSET_FALLBACK_MODE_FBX_OBJ_TRELLIS_PROXY,
}
TRELLIS_ASSET_STATUS = "trellis_generated_local_asset"

LOW_QUALITY_ASSET_STATUSES = {"proxy_generated_with_blender", "needs_blender_rebuild"}
SELECTED_BINDING_STATUSES = {
    "heuristic_top1_selected",
    "heuristic_first_acceptable_selected",
    "llm_reranked_top1_selected",
    "llm_reranked_first_acceptable_selected",
}


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


def _candidate_has_supported_local_asset(candidate: dict[str, Any] | None) -> bool:
    if not isinstance(candidate, dict):
        return False
    asset_status = str(candidate.get("asset_status") or "").strip().lower()
    if asset_status in LOW_QUALITY_ASSET_STATUSES:
        return False
    for local_path in _candidate_asset_paths(candidate):
        path_text = str(local_path).replace("\\", "/").lower()
        if path_text.endswith("/built/proxy.glb") or path_text.endswith("/proxy.glb"):
            continue
        ext = str(Path(local_path).suffix).lower().lstrip(".")
        if ext not in {"obj", "fbx", "glb", "gltf"}:
            continue
        if Path(local_path).is_file():
            return True
    return False


def _candidate_asset_paths(candidate: dict[str, Any] | None) -> list[str]:
    if not isinstance(candidate, dict):
        return []

    raw_values: list[Any] = []
    for key in (
        "asset_local_path",
        "local_asset_path",
        "mesh_path",
        "mesh_local_path",
        "obj_path",
        "fbx_path",
        "glb_path",
        "gltf_path",
        "file_path",
        "downloaded_path",
    ):
        value = candidate.get(key)
        if value:
            raw_values.append(value)

    for nested_key in ("asset", "source"):
        nested = candidate.get(nested_key)
        if isinstance(nested, dict):
            for key in (
                "asset_local_path",
                "local_asset_path",
                "mesh_path",
                "mesh_local_path",
                "obj_path",
                "fbx_path",
                "glb_path",
                "gltf_path",
                "file_path",
                "downloaded_path",
            ):
                value = nested.get(key)
                if value:
                    raw_values.append(value)

    extra = candidate.get("extra")
    if isinstance(extra, dict):
        trellis_asset = extra.get("trellis_generated_asset")
        if isinstance(trellis_asset, dict):
            for key in ("asset_local_path", "mesh_path", "glb_path", "gltf_path"):
                value = trellis_asset.get(key)
                if value:
                    raw_values.append(value)

    out: list[str] = []
    seen: set[str] = set()

    def add_file(path: Path) -> None:
        if not path.is_file():
            return  # pragma: no cover
        ext = path.suffix.lower().lstrip(".")
        if ext not in {"fbx", "obj", "glb", "gltf"}:
            return
        rp = str(path.expanduser().resolve())
        if rp not in seen:
            seen.add(rp)
            out.append(rp)

    for value in raw_values:
        values = value if isinstance(value, (list, tuple)) else [value]
        for raw in values:
            if raw is None:
                continue
            p = Path(str(raw)).expanduser()

            if p.is_file():
                add_file(p)
                continue

            if p.is_dir():
                for pattern in ("*.fbx", "*.FBX", "*.obj", "*.OBJ", "*.glb", "*.GLB", "*.gltf", "*.GLTF"):
                    for child in p.rglob(pattern):
                        add_file(child)

    return out

def _normalize_supplier_catalog_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    out = dict(candidate)
    dims = out.get("dimensions_cm") if isinstance(out.get("dimensions_cm"), dict) else {}
    for src_key, dst_key in [
        ("width", "width_cm"),
        ("depth", "depth_cm"),
        ("height", "height_cm"),
    ]:
        if out.get(dst_key) is None and dims.get(src_key) is not None:
            out[dst_key] = dims.get(src_key)
    if not str(out.get("asset_format") or "").strip() and out.get("asset_local_path"):
        out["asset_format"] = Path(str(out["asset_local_path"])).suffix.lstrip(".").lower()
    return out


def _compact_candidate_pool(binding: dict[str, Any], *, limit: int = 5) -> list[dict[str, Any]]:
    pool: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in [binding.get("chosen_candidate"), *(binding.get("top_candidates") or [])]:
        if not _candidate_has_supported_local_asset(candidate):
            continue
        compact = _compact_candidate(candidate)
        unique_key = str(compact.get("unique_key") or "").strip()
        if not unique_key or unique_key in seen:
            continue
        seen.add(unique_key)
        pool.append(compact)
        if len(pool) >= max(int(limit), 1):
            break
    return pool


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
    *,
    preserve_generated_bedding: bool = True,
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
        "FruitFactory",
    }
    strict_top_categories = {
        "DeskLampFactory",
        "FruitFactory",
        "PlantContainerFactory",
        "LargePlantContainerFactory",
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
            continue  # pragma: no cover
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
                continue  # pragma: no cover
            category = str(item.get("category") or "").strip()

            if anchor_group == "bed" and category in bed_soft_categories:
                if _xy_inside_expanded(anchor_aabb, item_pos[:2], margin=0.18) and item_aabb["z_min"] <= anchor_top + 0.25:
                    if not preserve_generated_bedding:
                        actions[item_id] = {
                            "action": "suppress",
                            "anchor_id": anchor_id,
                            "anchor_group": anchor_group,
                            "reason": "generated_bedding_suppressed_for_supplier_bed",
                        }
                        continue
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
                        "preserve_bedding": True,
                    }
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
                    if on_storage_top or (inside_or_touching_storage and category in strict_top_categories):
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
                    elif inside_or_touching_storage:
                        rel_x, rel_y = _normalized_anchor_xy(anchor_aabb, item_pos)
                        actions[item_id] = {
                            "action": "reanchor",
                            "anchor_id": anchor_id,
                            "anchor_group": anchor_group,
                            "support_mode": "volume",
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


def _candidate_asset(
    candidate: dict[str, Any],
    require_local_asset: bool,
    fallback_mode: str = ASSET_FALLBACK_MODE_NONE,
) -> tuple[dict[str, Any], bool]:
    if not isinstance(candidate, dict) or not candidate:
        return {}, False

    fallback_mode = str(fallback_mode or ASSET_FALLBACK_MODE_NONE).strip()
    use_fallback_proxy = fallback_mode in {ASSET_FALLBACK_MODE_FBX_OBJ_PROXY, ASSET_FALLBACK_MODE_FBX_OBJ_TRELLIS_PROXY}

    asset_status = str(candidate.get("asset_status") or "").strip().lower()
    extra = candidate.get("extra") if isinstance(candidate.get("extra"), dict) else {}
    trellis_extra = extra.get("trellis_generated_asset") if isinstance(extra.get("trellis_generated_asset"), dict) else {}
    is_proxy_generated_glb = (
        str(trellis_extra.get("asset_generation_method") or "").strip() == "proxy_glb_fallback"
        or str(trellis_extra.get("asset_source") or "").strip() == "supplier_catalog_procedural_proxy"
    )
    local_paths = _candidate_asset_paths(candidate)

    def make_asset_block(path: str, *, ext: str, source: str, fit_mode: str) -> dict[str, Any]:
        return {
            "mesh_path": path,
            "mesh_fit_mode": fit_mode,
            "kind": "supplier_catalog_asset",
            "source_kind": "supplier_catalog_local_asset",
            "asset_source": source,
            "asset_format": ext,
        }

    # 1. Prefer real FBX supplier assets.
    for p in local_paths:
        path_text = str(p).replace("\\", "/").lower()
        if path_text.endswith("/built/proxy.glb") or path_text.endswith("/proxy.glb"):
            continue
        ext = Path(p).suffix.lower().lstrip(".")
        if ext == "fbx" and Path(p).is_file():
            return (
                make_asset_block(
                    p,
                    ext=ext,
                    source="supplier_catalog_local_asset",
                    fit_mode="uniform",
                ),
                False,
            )

    # 2. Real GLB/GLTF supplier assets are valid too.
    # TRELLIS-generated GLB is stretched to target AABB because it is not metrically reliable.
    # Real supplier GLB/GLTF is kept uniform and fitted by the builder.
    for p in local_paths:
        path_text = str(p).replace("\\", "/").lower()
        if path_text.endswith("/built/proxy.glb") or path_text.endswith("/proxy.glb"):
            continue
        ext = Path(p).suffix.lower().lstrip(".")
        if ext in {"glb", "gltf"} and Path(p).is_file():
            is_trellis = asset_status == TRELLIS_ASSET_STATUS
            return (
                make_asset_block(
                    p,
                    ext=ext,
                    source=(
                        "supplier_catalog_procedural_proxy"
                        if is_proxy_generated_glb
                        else "trellis_generated_local_asset"
                        if is_trellis
                        else "supplier_catalog_local_asset"
                    ),
                    fit_mode="stretch" if is_trellis else "uniform",
                ),
                False,
            )

    # 3. Real OBJ supplier assets are valid. Previously this was only enabled in proxy fallback mode;
    # now OBJ is a first-class local asset because many supplier archives contain OBJ+MTL.
    for p in local_paths:
        path_text = str(p).replace("\\", "/").lower()
        if path_text.endswith("/built/proxy.glb") or path_text.endswith("/proxy.glb"):
            continue
        ext = Path(p).suffix.lower().lstrip(".")
        if ext == "obj" and Path(p).is_file():
            return (
                make_asset_block(
                    p,
                    ext=ext,
                    source="supplier_catalog_local_asset",
                    fit_mode="uniform",
                ),
                False,
            )

    # 4. Strict local mode: no local mesh means no replacement.
    if require_local_asset:
        return {}, False

    # 5. No real asset: fallback to procedural proxy if requested.
    if use_fallback_proxy:
        return (
            {
                "kind": "procedural_proxy",
                "mesh_fit_mode": "uniform",
                "source_kind": "supplier_catalog_proxy",
                "asset_source": "supplier_catalog_procedural_proxy",
            },
            False,
        )

    if asset_status in LOW_QUALITY_ASSET_STATUSES:
        return {}, True

    # 6. Old non-strict behavior: placeholder is allowed only without proxy mode.
    return {}, True


def _replacement_mesh_fit_mode(binding: dict[str, Any], item: dict[str, Any]) -> str:
    asset = item.get("asset") if isinstance(item.get("asset"), dict) else {}
    asset_source = str(asset.get("asset_source") or "").strip().lower()
    if asset_source in {"trellis_generated_local_asset", "supplier_catalog_procedural_proxy"}:
        # Generated GLBs are not metrically reliable. Uniform fitting often makes
        # them look undersized because a single long/empty axis limits the scale.
        return "trellis_stretch"

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
        return "uniform"
    if category in {
        "simpledeskfactory",
        "singlecabinetfactory",
        "cellshelffactory",
        "simplebookcasefactory",
    }:
        return "uniform"
    return "uniform"


def _should_keep_original_scene_item(item: dict[str, Any], binding: dict[str, Any]) -> bool:
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
    if semantic_group == "computer":
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


def _semantic_group_for_item(item: dict[str, Any], binding: dict[str, Any] | None = None) -> str:
    def normalize_group(raw: Any) -> str:
        text = str(raw or "").strip().lower()
        if not text:
            return ""
        known = {
            "lamp_ceiling",
            "lamp_table",
            "lamp_floor",
            "lamp_wall",
            "desk",
            "dining_table",
            "coffee_table",
            "side_table",
            "armchair",
            "chair",
            "sofa",
            "bed",
            "tv_stand",
            "tv_projector_screen",
            "computer",
        }
        if text in known:
            return text
        token_map = (
            ("ceilinglightfactory", "lamp_ceiling"),
            ("ceiling light", "lamp_ceiling"),
            ("chandelier", "lamp_ceiling"),
            ("люстр", "lamp_ceiling"),
            ("desklampfactory", "lamp_table"),
            ("desk lamp", "lamp_table"),
            ("floorlampfactory", "lamp_floor"),
            ("floor lamp", "lamp_floor"),
            ("simpledeskfactory", "desk"),
            ("deskfactory", "desk"),
            ("diningtablefactory", "dining_table"),
            ("coffeetablefactory", "coffee_table"),
            ("sidetablefactory", "side_table"),
            ("armchairfactory", "armchair"),
            ("chairfactory", "chair"),
            ("sofafactory", "sofa"),
            ("sofa", "sofa"),
            ("bedfactory", "bed"),
            ("bed", "bed"),
            ("tvstand", "tv_stand"),
            ("tv stand", "tv_stand"),
            ("tv_stand", "tv_stand"),
            ("wallmountedtvfactory", "tv_projector_screen"),
            ("television", "tv_projector_screen"),
            ("телевиз", "tv_projector_screen"),
            ("monitorfactory", "computer"),
            ("monitor factory", "computer"),
            ("computer", "computer"),
            ("laptop", "computer"),
            ("macbook", "computer"),
            ("imac", "computer"),
        )
        for token, semantic in token_map:
            if token in text:
                return semantic  # pragma: no cover
        return text

    meta = item.get("meta") or {}
    supplier_candidate = meta.get("supplier_candidate") if isinstance(meta.get("supplier_candidate"), dict) else {}
    group = normalize_group(
        supplier_candidate.get("semantic_group")
        or (binding or {}).get("semantic_group")
        or item.get("semantic_group")
        or ""
    )
    if group:
        if group == "tv_projector_screen":
            tv_text = " ".join(
                str(value or "").lower()
                for value in (
                    item.get("category"),
                    item.get("name"),
                    supplier_candidate.get("title"),
                    supplier_candidate.get("category_raw"),
                    supplier_candidate.get("category_norm"),
                )
            )
            if any(token in tv_text for token in ("monitor", "монитор", "gaming", "игров")):
                return "computer"
        return group

    category = str(item.get("category") or "").strip().lower()
    name = str(item.get("name") or "").strip().lower()
    text = f"{category} {name}"
    mapping = (
        ("ceilinglightfactory", "lamp_ceiling"),
        ("desklampfactory", "lamp_table"),
        ("floorlampfactory", "lamp_floor"),
        ("simpledeskfactory", "desk"),
        ("deskfactory", "desk"),
        ("diningtablefactory", "dining_table"),
        ("coffeetablefactory", "coffee_table"),
        ("sidetablefactory", "side_table"),
        ("armchairfactory", "armchair"),
        ("chairfactory", "chair"),
        ("sofafactory", "sofa"),
        ("bedfactory", "bed"),
        ("tvstand", "tv_stand"),
        ("television", "tv_projector_screen"),
        ("monitorfactory", "computer"),
        ("computer", "computer"),
        ("laptop", "computer"),
        ("macbook", "computer"),
        ("imac", "computer"),
    )
    for token, semantic in mapping:
        if token in text:
            return semantic
    if "tv stand" in text or "tv_stand" in text or ("тумб" in text and ("tv" in text or "тв" in text or "телевиз" in text)):
        return "tv_stand"
    if "tv" in text or "television" in text or "телевиз" in text:
        return "tv_projector_screen"
    if "bed" in text or "кровать" in text:
        return "bed"
    if "sofa" in text or "couch" in text or "диван" in text:
        return "sofa"
    if "люстр" in text or "chandelier" in text or "ceiling light" in text:
        return "lamp_ceiling"
    if "desk lamp" in text or "настоль" in text:
        return "lamp_table"
    if "dining" in text and "table" in text:
        return "dining_table"
    if "coffee" in text and "table" in text:
        return "coffee_table"
    if "desk" in text:
        return "desk"
    if "monitor" in text or "computer" in text or "laptop" in text or "keyboard" in text or "macbook" in text or "imac" in text:
        return "computer"
    if "chair" in text or "стул" in text:
        return "chair"
    if "armchair" in text or "кресл" in text:
        return "armchair"
    return ""


def _computer_text_kind(text: str) -> str:
    text = str(text or "").lower()
    if any(word in text for word in ("keyboard", "mouse", "клавиат", "мышь")) and not any(
        word in text for word in ("monitor", "display", "screen", "imac", "laptop", "macbook", "computer", "pc", "ноутбук")
    ):
        return "keyboard_mouse"
    if "imac" in text or "all-in-one" in text or "all in one" in text or "моноблок" in text:
        return "all_in_one"
    if any(word in text for word in ("laptop", "macbook", "notebook", "ноутбук")):
        return "laptop"
    if any(word in text for word in ("monitor", "display", "screen", "gaming", "игров", "монитор")):
        return "monitor"
    if any(word in text for word in ("computer", "desktop", "pc", "компьютер")):
        return "desktop"
    return "computer"


def _computer_item_kind(item: dict[str, Any]) -> str:
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    candidate = meta.get("supplier_candidate") if isinstance(meta.get("supplier_candidate"), dict) else {}
    text = " ".join(
        str(value or "")
        for value in (
            item.get("category"),
            item.get("name"),
            item.get("semantic_group"),
            candidate.get("title"),
            candidate.get("category_norm"),
            candidate.get("category_raw"),
        )
    )
    return _computer_text_kind(text)


def _looks_like_tv_text(text: str) -> bool:
    text = str(text or "").lower()
    reject = ("monitor", "монитор", "gaming", "игров", "keyboard", "mouse", "computer", "laptop", "macbook", "imac")
    if any(word in text for word in reject):
        return False
    return any(word in text for word in ("tv", "television", "телевиз", "smart tv", "oled tv", "uhd tv"))


def _candidate_from_supplier_catalog_json(
    category_norms: set[str],
    target_size: list[float],
    *,
    computer_kind: str | None = None,
) -> dict[str, Any] | None:
    catalog_path = Path("data/sourse/suppliers/supplier_catalog_canonical.json")
    if not catalog_path.is_file():
        return None  # pragma: no cover
    try:
        payload = read_json(catalog_path)
    except Exception:
        return None
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return None

    wants_computer = bool(category_norms & {"computer", "laptop_computer_keyboard_mouse", "computer_monitor"})
    target_w, target_d, target_h = target_size
    wants_tv = "tv_projector_screen" in category_norms and not wants_computer
    best: tuple[float, dict[str, Any]] | None = None
    for row in items:
        if not isinstance(row, dict):
            continue
        candidate = _normalize_supplier_catalog_candidate(row)
        category_norm = str(candidate.get("category_norm") or "").strip()
        if category_norm not in category_norms:
            continue
        if not _candidate_has_supported_local_asset(candidate):
            continue
        text = " ".join(
            str(candidate.get(key) or "").lower()
            for key in ("title", "category_raw", "description", "product_url")
        )
        if wants_tv:
            if not _looks_like_tv_text(text):
                continue
        if wants_computer:
            computer_words = ("mac", "imac", "macbook", "computer", "laptop", "keyboard", "monitor", "display", "desktop", "pc", "ноутбук", "компьютер", "монитор")
            reject_words = ("washing", "стираль", "router", "роутер", "coffee", "кофемаш", "tv", "television", "телевизор")
            if category_norm == "tv_projector_screen" and not any(word in text for word in ("monitor", "display", "screen", "gaming", "монитор", "игров")):
                continue
            if not any(word in text for word in computer_words) or any(word in text for word in reject_words):
                continue
        try:
            cw = float(candidate.get("width_cm") or target_w * 100.0) / 100.0
            cd = float(candidate.get("depth_cm") or target_d * 100.0) / 100.0
            ch = float(candidate.get("height_cm") or target_h * 100.0) / 100.0
        except Exception:  # pragma: no cover
            cw, cd, ch = target_w, target_d, target_h  # pragma: no cover
        size_score = abs(cw - target_w) + abs(cd - target_d) * 1.5 + abs(ch - target_h)
        if wants_tv:
            if cw < max(0.8, target_w * 0.72):
                size_score += 1.2
            if cd > 0.18:
                size_score += 0.45
        if wants_computer:
            title = str(candidate.get("title") or "").lower()
            cand_kind = _computer_text_kind(" ".join((title, str(candidate.get("category_norm") or ""), str(candidate.get("category_raw") or ""))))
            if computer_kind == "monitor":
                if cand_kind == "monitor":
                    size_score -= 0.75
                elif cand_kind == "all_in_one":
                    size_score -= 0.25
                elif cand_kind == "laptop":
                    size_score += 1.4
            elif computer_kind == "laptop":
                if cand_kind == "laptop":
                    size_score -= 0.6
                elif cand_kind in {"monitor", "all_in_one"}:
                    size_score += 0.45
            elif computer_kind == "all_in_one":
                if cand_kind == "all_in_one":
                    size_score -= 0.7
                elif cand_kind == "monitor":
                    size_score -= 0.15
                elif cand_kind == "laptop":
                    size_score += 0.9
            else:
                if cand_kind == "monitor":
                    size_score -= 0.35  # pragma: no cover
                elif cand_kind == "all_in_one":
                    size_score -= 0.2  # pragma: no cover
                elif cand_kind == "laptop":
                    size_score += 0.25  # pragma: no cover
        ready_bonus = -0.2 if str(candidate.get("asset_status") or "") == "local_dir_preferred" else 0.0
        source_bonus = -0.05 if str(candidate.get("source_site") or "") == "3ddd" else 0.0
        score = size_score + ready_bonus + source_bonus
        if wants_tv:
            candidate["semantic_group"] = "tv_projector_screen"
        elif wants_computer:
            candidate["semantic_group"] = "computer"
        if best is None or score < best[0]:
            best = (score, candidate)
    return best[1] if best is not None else None


def _room_center_xy(data: dict[str, Any], items: list[dict[str, Any]]) -> tuple[float, float]:
    room = data.get("room") or {}
    poly = room.get("floor_polygon") if isinstance(room, dict) else None
    pts: list[tuple[float, float]] = []
    if isinstance(poly, list):
        for pt in poly:
            if isinstance(pt, dict) and "x" in pt and "y" in pt:
                pts.append((float(pt["x"]), float(pt["y"])))
            elif isinstance(pt, list) and len(pt) >= 2:
                pts.append((float(pt[0]), float(pt[1])))
    if pts:
        return sum(x for x, _ in pts) / len(pts), sum(y for _, y in pts) / len(pts)

    aabbs = [_item_aabb(item) for item in items if isinstance(item, dict)]
    aabbs = [aabb for aabb in aabbs if aabb is not None]
    if aabbs:
        return (
            0.5 * (min(a["x_min"] for a in aabbs) + max(a["x_max"] for a in aabbs)),
            0.5 * (min(a["y_min"] for a in aabbs) + max(a["y_max"] for a in aabbs)),
        )
    return 0.0, 0.0


def _room_xy_bounds(data: dict[str, Any], items: list[dict[str, Any]]) -> dict[str, float] | None:
    room = data.get("room") or {}
    poly = room.get("floor_polygon") if isinstance(room, dict) else None
    points: list[tuple[float, float]] = []
    if isinstance(poly, list):
        for p in poly:
            if isinstance(p, dict) and "x" in p and "y" in p:
                points.append((float(p["x"]), float(p["y"])))
            elif isinstance(p, list) and len(p) >= 2:
                points.append((float(p[0]), float(p[1])))
    if not points:
        for item in items:
            if not isinstance(item, dict):
                continue  # pragma: no cover
            aabb = _item_aabb(item)
            if aabb:
                points.extend([(aabb["x_min"], aabb["y_min"]), (aabb["x_max"], aabb["y_max"])])
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return {"x_min": min(xs), "x_max": max(xs), "y_min": min(ys), "y_max": max(ys)}


def _point_in_room_xy(data: dict[str, Any], x: float, y: float) -> bool:
    room = data.get("room") or {}
    poly = room.get("floor_polygon") if isinstance(room, dict) else None
    points: list[tuple[float, float]] = []
    if isinstance(poly, list):
        for p in poly:
            if isinstance(p, dict) and "x" in p and "y" in p:
                points.append((float(p["x"]), float(p["y"])))
            elif isinstance(p, list) and len(p) >= 2:
                points.append((float(p[0]), float(p[1])))
    if len(points) < 3:
        return True
    inside = False
    j = len(points) - 1
    for i, (xi, yi) in enumerate(points):
        xj, yj = points[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / max(yj - yi, 1e-9) + xi):
            inside = not inside
        j = i
    return inside


def _set_item_center_xy(item: dict[str, Any], cx: float, cy: float) -> None:
    aabb = _item_aabb(item)
    pos = _item_position(item)
    if aabb is None or pos is None:
        return
    dx = float(cx) - pos[0]
    dy = float(cy) - pos[1]
    item["position_m"] = [float(cx), float(cy), pos[2]]
    item["aabb"] = {
        "x_min": aabb["x_min"] + dx,
        "x_max": aabb["x_max"] + dx,
        "y_min": aabb["y_min"] + dy,
        "y_max": aabb["y_max"] + dy,
        "z_min": aabb["z_min"],
        "z_max": aabb["z_max"],
    }


def _set_item_center_xyz(item: dict[str, Any], cx: float, cy: float, cz: float) -> None:
    aabb = _item_aabb(item)
    if aabb is None:
        return
    cur = _aabb_center(aabb)
    dx = float(cx) - cur[0]
    dy = float(cy) - cur[1]
    dz = float(cz) - cur[2]
    item["position_m"] = [float(cx), float(cy), float(cz)]
    item["aabb"] = {
        "x_min": aabb["x_min"] + dx,
        "x_max": aabb["x_max"] + dx,
        "y_min": aabb["y_min"] + dy,
        "y_max": aabb["y_max"] + dy,
        "z_min": aabb["z_min"] + dz,
        "z_max": aabb["z_max"] + dz,
    }


def _light_collides_xy(
    candidate_aabb: dict[str, float],
    items: list[dict[str, Any]],
    *,
    skip_id: str,
    by_target_id: dict[str, dict[str, Any]],
) -> int:
    collisions = 0
    for other in items:
        if not isinstance(other, dict):
            continue
        other_id = str(other.get("id") or "").strip()
        if other_id == skip_id:
            continue
        other_aabb = _item_aabb(other)
        if other_aabb is None:
            continue
        other_group = _semantic_group_for_item(other, by_target_id.get(other_id))
        z_overlaps = (
            candidate_aabb["z_max"] > other_aabb["z_min"] + 0.03
            and candidate_aabb["z_min"] < other_aabb["z_max"] - 0.03
        )
        if not z_overlaps or not _xy_aabb_overlap(candidate_aabb, other_aabb, margin=0.04):
            continue
        if other_group in {"computer", "chair", "armchair"}:
            collisions += 4
        else:
            collisions += 1
    return collisions


def _normalize_supported_light_placements(
    data: dict[str, Any],
    items: list[dict[str, Any]],
    by_target_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    support_groups = {"desk", "side_table", "nightstand", "dresser", "coffee_table", "shelf", "tv_stand"}
    moved: list[dict[str, Any]] = []

    for item in items:
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("id") or "").strip()
        group = _semantic_group_for_item(item, by_target_id.get(item_id))
        item_aabb = _item_aabb(item)
        if not item_id or item_aabb is None:
            continue
        sx = max(item_aabb["x_max"] - item_aabb["x_min"], 1e-6)
        sy = max(item_aabb["y_max"] - item_aabb["y_min"], 1e-6)
        sz = max(item_aabb["z_max"] - item_aabb["z_min"], 1e-6)
        old_pos = _aabb_center(item_aabb)

        if group == "lamp_table":
            supports: list[tuple[float, dict[str, Any], dict[str, float], str]] = []
            for anchor in items:
                if not isinstance(anchor, dict) or anchor is item:
                    continue
                anchor_id = str(anchor.get("id") or "").strip()
                anchor_group = _semantic_group_for_item(anchor, by_target_id.get(anchor_id))
                if anchor_group not in support_groups:
                    continue
                anchor_aabb = _item_aabb(anchor)
                if anchor_aabb is None:
                    continue  # pragma: no cover
                anchor_c = _aabb_center(anchor_aabb)
                dist = math.hypot(old_pos[0] - anchor_c[0], old_pos[1] - anchor_c[1])
                overlaps = _xy_aabb_overlap(item_aabb, anchor_aabb, margin=0.35)
                near_top = anchor_aabb["z_max"] - 0.15 <= item_aabb["z_min"] <= anchor_aabb["z_max"] + 0.55
                if overlaps or dist <= 1.2 or (anchor_group in {"desk", "side_table", "nightstand"} and dist <= 1.8):
                    bonus = -0.6 if overlaps and near_top else 0.0
                    supports.append((dist + bonus, anchor, anchor_aabb, anchor_group))
            if not supports:
                continue

            _score, anchor, anchor_aabb, anchor_group = min(supports, key=lambda row: row[0])
            margin = 0.08
            x_low = anchor_aabb["x_min"] + margin + 0.5 * sx
            x_high = anchor_aabb["x_max"] - margin - 0.5 * sx
            y_low = anchor_aabb["y_min"] + margin + 0.5 * sy
            y_high = anchor_aabb["y_max"] - margin - 0.5 * sy
            if x_low > x_high:
                x_low = x_high = 0.5 * (anchor_aabb["x_min"] + anchor_aabb["x_max"])
            if y_low > y_high:
                y_low = y_high = 0.5 * (anchor_aabb["y_min"] + anchor_aabb["y_max"])
            anchor_cx = 0.5 * (anchor_aabb["x_min"] + anchor_aabb["x_max"])
            anchor_cy = 0.5 * (anchor_aabb["y_min"] + anchor_aabb["y_max"])
            candidates = [
                (x_low, y_low),
                (x_low, y_high),
                (x_high, y_low),
                (x_high, y_high),
                (anchor_cx, y_low),
                (anchor_cx, y_high),
                (x_low, anchor_cy),
                (x_high, anchor_cy),
            ]
            z_min = anchor_aabb["z_max"] + 0.004
            best: tuple[float, float, float] | None = None
            for cx, cy in candidates:
                cand = {
                    "x_min": cx - 0.5 * sx,
                    "x_max": cx + 0.5 * sx,
                    "y_min": cy - 0.5 * sy,
                    "y_max": cy + 0.5 * sy,
                    "z_min": z_min,
                    "z_max": z_min + sz,
                }
                collision_score = _light_collides_xy(cand, items, skip_id=item_id, by_target_id=by_target_id)
                edge_preference = -0.15 if anchor_group == "desk" and (abs(cx - anchor_cx) > abs(cy - anchor_cy)) else 0.0
                travel = math.hypot(cx - old_pos[0], cy - old_pos[1])
                score = collision_score + travel * 0.05 + edge_preference
                if best is None or score < best[0]:
                    best = (score, cx, cy)
            if best is None:
                continue  # pragma: no cover
            _best_score, cx, cy = best
            updated_cz = z_min + 0.5 * sz
            _set_item_center_xyz(item, cx, cy, updated_cz)
            meta = item.setdefault("meta", {})
            if isinstance(meta, dict):
                meta["supplier_light_position_normalized"] = True
                meta["supplier_support_anchor_target_id"] = str(anchor.get("id") or "")
                meta["supplier_support_anchor_group"] = anchor_group
                meta["supplier_support_mode"] = "top"
            moved.append({"id": item_id, "group": group, "old_xyz": [round(v, 4) for v in old_pos], "new_xyz": [round(cx, 4), round(cy, 4), round(updated_cz, 4)]})
            continue

        if group == "lamp_floor":
            continue

    return items, {"moved_count": len(moved), "moved": moved}


def _collapse_ceiling_lights(
    data: dict[str, Any],
    items: list[dict[str, Any]],
    by_target_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ceiling_groups: dict[str, list[int]] = {}
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("id") or "").strip()
        if _semantic_group_for_item(item, by_target_id.get(item_id)) != "lamp_ceiling":
            continue
        meta = item.get("meta") or {}
        candidate = meta.get("supplier_candidate") if isinstance(meta.get("supplier_candidate"), dict) else {}
        asset = item.get("asset") if isinstance(item.get("asset"), dict) else {}
        signature = str(
            candidate.get("unique_key")
            or asset.get("mesh_path")
            or candidate.get("title")
            or ""
        ).strip()
        if not signature:
            signature = f"generated::{item.get('category') or item.get('name') or 'ceiling_light'}"  # pragma: no cover
        ceiling_groups.setdefault(signature, []).append(idx)

    moved: list[dict[str, Any]] = []
    kept_ids = [
        str(items[idx].get("id") or "").strip()
        for indices in ceiling_groups.values()
        for idx in indices
    ]
    if kept_ids:
        indices = [
            idx
            for indices in ceiling_groups.values()
            for idx in indices
        ]
        centers, coverage_radius = _ceiling_coverage_points(data, len(indices))
        for idx, center in zip(indices, centers):
            item = items[idx]
            old_pos = _item_position(item)
            if old_pos is None:
                continue  # pragma: no cover
            _set_item_center_xy(item, center[0], center[1])
            meta = item.setdefault("meta", {})
            if isinstance(meta, dict):
                meta["ceiling_supplier_coverage_normalized"] = True
                meta["ceiling_supplier_min_wall_clearance_m"] = 1.0
                meta["ceiling_supplier_coverage_radius_m"] = round(coverage_radius, 3)
            moved.append(
                {
                    "id": item.get("id"),
                    "old_xy": [round(old_pos[0], 4), round(old_pos[1], 4)],
                    "new_xy": [round(center[0], 4), round(center[1], 4)],
                }
            )
    return items, {
        "removed_count": 0,
        "kept_id": None,
        "kept_ids": [kid for kid in kept_ids if kid],
        "removed_ids": [],
        "count_preserved": True,
        "moved": moved,
    }


def _room_polygon_points(data: dict[str, Any]) -> list[tuple[float, float]]:
    room = data.get("room") if isinstance(data.get("room"), dict) else {}
    poly = room.get("floor_polygon") if isinstance(room.get("floor_polygon"), list) else []
    points: list[tuple[float, float]] = []
    for p in poly:
        if isinstance(p, dict) and "x" in p and "y" in p:
            points.append((float(p["x"]), float(p["y"])))
        elif isinstance(p, list) and len(p) >= 2:
            points.append((float(p[0]), float(p[1])))
    return points


def _point_segment_distance(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    denom = vx * vx + vy * vy
    t = 0.0 if denom <= 1e-9 else max(0.0, min(1.0, (wx * vx + wy * vy) / denom))
    qx, qy = ax + t * vx, ay + t * vy
    return math.hypot(px - qx, py - qy)


def _dist_to_room_edges(points: list[tuple[float, float]], x: float, y: float) -> float:
    if len(points) < 2:
        return 999.0
    return min(
        _point_segment_distance(x, y, ax, ay, bx, by)
        for (ax, ay), (bx, by) in zip(points, points[1:] + points[:1])
    )


def _ceiling_coverage_points(data: dict[str, Any], count: int) -> tuple[list[tuple[float, float]], float]:
    points = _room_polygon_points(data)
    bounds = _room_xy_bounds(data, [])
    if count <= 0 or bounds is None:
        return [], 0.0
    sample_step = 0.3
    candidates: list[tuple[float, float]] = []
    x = bounds["x_min"]
    while x <= bounds["x_max"] + 1e-9:
        y = bounds["y_min"]
        while y <= bounds["y_max"] + 1e-9:
            if _point_in_room_xy(data, x, y) and _dist_to_room_edges(points, x, y) >= 1.0:
                candidates.append((x, y))
            y += sample_step
        x += sample_step
    if not candidates:
        cx, cy = _room_center_xy(data, [])
        return [(cx, cy)] * count, 0.0

    coverage_points = [
        (x, y)
        for x, y in candidates
    ]

    def greedy_from(seed: tuple[float, float]) -> list[tuple[float, float]]:
        selected = [seed]
        while len(selected) < count:
            selected.append(max(candidates, key=lambda p: min(math.hypot(p[0] - s[0], p[1] - s[1]) for s in selected)))
        return selected

    seed_pool = candidates[:: max(1, len(candidates) // 12)] or candidates
    best_centers: list[tuple[float, float]] | None = None
    best_score: tuple[float, float] | None = None
    for seed in seed_pool:
        centers = greedy_from(seed)
        radius = max(min(math.hypot(p[0] - c[0], p[1] - c[1]) for c in centers) for p in coverage_points)
        min_pair = min(
            (math.hypot(a[0] - b[0], a[1] - b[1]) for i, a in enumerate(centers) for b in centers[i + 1 :]),
            default=0.0,
        )
        score = (radius, -min_pair)
        if best_score is None or score < best_score:
            best_score = score
            best_centers = centers
    return best_centers or candidates[:count], float(best_score[0] if best_score else 0.0)


def _table_requires_chair(item: dict[str, Any], group: str) -> bool:
    if group in {"desk", "dining_table"}:
        return True
    category = str(item.get("category") or "").strip().lower()
    name = str(item.get("name") or "").strip().lower()
    text = f"{category} {name}"
    if "simpledeskfactory" in text or "deskfactory" in text or "diningtablefactory" in text:
        return True
    if "coffee" in text or "side table" in text or "sidetable" in text or "nightstand" in text:
        return False
    return False


def _has_nearby_chair(item: dict[str, Any], items: list[dict[str, Any]], by_target_id: dict[str, dict[str, Any]]) -> bool:
    item_pos = _item_position(item)
    if item_pos is None:
        return False
    for other in items:
        if not isinstance(other, dict) or other is item:
            continue
        other_id = str(other.get("id") or "").strip()
        group = _semantic_group_for_item(other, by_target_id.get(other_id))
        if group not in {"chair", "armchair"}:
            continue  # pragma: no cover
        other_pos = _item_position(other)
        if other_pos is None:
            continue  # pragma: no cover
        if math.hypot(other_pos[0] - item_pos[0], other_pos[1] - item_pos[1]) <= 1.35:
            return True
    return False


def _has_usable_nearby_chair(item: dict[str, Any], items: list[dict[str, Any]], by_target_id: dict[str, dict[str, Any]]) -> bool:
    item_pos = _item_position(item)
    table_aabb = _item_aabb(item)
    if item_pos is None or table_aabb is None:
        return False
    for other in items:
        if not isinstance(other, dict) or other is item:
            continue
        other_id = str(other.get("id") or "").strip()
        group = _semantic_group_for_item(other, by_target_id.get(other_id))
        if group not in {"chair", "armchair"}:
            continue
        other_pos = _item_position(other)
        other_aabb = _item_aabb(other)
        if other_pos is None or other_aabb is None:
            continue  # pragma: no cover
        if math.hypot(other_pos[0] - item_pos[0], other_pos[1] - item_pos[1]) <= 1.35 and _chair_is_on_table_long_edge(table_aabb, other_aabb):
            return True
    return False


def _xy_aabb_overlap(a: dict[str, float], b: dict[str, float], margin: float = 0.0) -> bool:
    return (
        a["x_max"] > b["x_min"] - margin
        and a["x_min"] < b["x_max"] + margin
        and a["y_max"] > b["y_min"] - margin
        and a["y_min"] < b["y_max"] + margin
    )


def _door_keepout_aabbs(data: dict[str, Any], *, depth_m: float = 1.05, side_margin_m: float = 0.35) -> list[dict[str, float]]:
    room = data.get("room") if isinstance(data.get("room"), dict) else {}
    doors = room.get("doors") if isinstance(room.get("doors"), list) else []
    keepouts: list[dict[str, float]] = []
    for door in doors:
        if not isinstance(door, dict):
            continue
        segment = door.get("segment") if isinstance(door.get("segment"), dict) else {}
        try:
            x1 = float(segment["x1"])
            y1 = float(segment["y1"])
            x2 = float(segment["x2"])
            y2 = float(segment["y2"])
        except Exception:
            continue
        if abs(y2 - y1) <= abs(x2 - x1):
            x_min = min(x1, x2) - side_margin_m
            x_max = max(x1, x2) + side_margin_m
            if y1 <= 0.05:
                y_min, y_max = 0.0, depth_m
            else:
                y_min, y_max = y1 - depth_m, y1
        else:
            y_min = min(y1, y2) - side_margin_m
            y_max = max(y1, y2) + side_margin_m
            if x1 <= 0.05:
                x_min, x_max = 0.0, depth_m
            else:
                x_min, x_max = x1 - depth_m, x1
        keepouts.append({"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max, "z_min": 0.0, "z_max": 2.2})
    return keepouts


def _chair_side_for_table(table_aabb: dict[str, float], chair_aabb: dict[str, float]) -> str:
    table_cx = 0.5 * (table_aabb["x_min"] + table_aabb["x_max"])
    table_cy = 0.5 * (table_aabb["y_min"] + table_aabb["y_max"])
    chair_cx = 0.5 * (chair_aabb["x_min"] + chair_aabb["x_max"])
    chair_cy = 0.5 * (chair_aabb["y_min"] + chair_aabb["y_max"])
    dx = chair_cx - table_cx
    dy = chair_cy - table_cy
    if abs(dx) >= abs(dy):
        return "east" if dx >= 0 else "west"
    return "north" if dy >= 0 else "south"


def _chair_is_on_table_long_edge(table_aabb: dict[str, float], chair_aabb: dict[str, float]) -> bool:
    table_width = table_aabb["x_max"] - table_aabb["x_min"]
    table_depth = table_aabb["y_max"] - table_aabb["y_min"]
    side = _chair_side_for_table(table_aabb, chair_aabb)
    broad_sides = {"north", "south"} if table_width >= table_depth else {"east", "west"}
    return side in broad_sides


def _set_item_pose_from_aabb(
    item: dict[str, Any],
    aabb: dict[str, float],
    *,
    yaw_deg: float,
) -> None:
    size = [
        max(float(aabb["x_max"]) - float(aabb["x_min"]), 1e-6),
        max(float(aabb["y_max"]) - float(aabb["y_min"]), 1e-6),
        max(float(aabb["z_max"]) - float(aabb["z_min"]), 1e-6),
    ]
    center = _aabb_center(aabb)
    item["position_m"] = center
    item["size_m"] = size
    item["rotation_deg"] = float(yaw_deg)
    item["yaw_deg"] = float(yaw_deg)
    item["yaw_rad"] = math.radians(float(yaw_deg))
    item["aabb"] = {k: float(v) for k, v in aabb.items()}


def _catalog_candidate_asset(candidate: dict[str, Any]) -> dict[str, Any]:
    mesh_path = str(candidate.get("asset_local_path") or candidate.get("mesh_local_path") or "").strip()
    if mesh_path and Path(mesh_path).expanduser().is_file():
        return {"mesh_path": str(Path(mesh_path).expanduser().resolve()), "mesh_fit_mode": "uniform"}
    return {}


def _candidate_from_supplier_db(group: str, target_size: list[float]) -> dict[str, Any] | None:
    target_w, target_d, target_h = target_size
    candidates: list[dict[str, Any]] = []

    for asset_db in [
        Path("data/sourse/suppliers/site_assets_imodern_clean.db"),
    ]:
        if not asset_db.is_file():
            continue  # pragma: no cover
        try:
            with sqlite3.connect(str(asset_db)) as con:
                con.row_factory = sqlite3.Row
                rows = con.execute(
                    """
                    SELECT
                        a.unique_key, a.source_site, a.title, a.product_url,
                        a.asset_status, a.asset_format, a.asset_local_path,
                        json_extract(a.extra_json, '$.model_page_url') AS model_page_url,
                        json_extract(a.extra_json, '$.model_download_url') AS model_download_url,
                        p.brand, p.collection, p.category_raw, p.category_norm, p.width_cm,
                        p.depth_cm, p.height_cm, p.price_value, p.price_currency, p.style,
                        p.color, p.materials, p.description
                    FROM supplier_asset a
                    LEFT JOIN supplier_product p ON p.unique_key = a.unique_key
                    WHERE a.asset_local_path IS NOT NULL
                      AND a.asset_local_path != ''
                      AND (
                        lower(COALESCE(p.category_norm, '')) LIKE '%chair%'
                        OR lower(COALESCE(a.title, '')) LIKE '%chair%'
                        OR lower(COALESCE(a.title, '')) LIKE '%стул%'
                        OR lower(COALESCE(a.title, '')) LIKE '%кресл%'
                      )
                    """
                )
                for row in rows:
                    item = dict(row)
                    text = f"{item.get('category_norm') or ''} {item.get('title') or ''}".lower()
                    item["semantic_group"] = "chair" if ("стул" in text or "chair" in text and "armchair" not in text) else "armchair"
                    candidates.append(item)
        except sqlite3.Error:
            pass

    best: tuple[float, dict[str, Any]] | None = None
    for row in candidates:
        asset_path = str(row.get("asset_local_path") or row.get("mesh_local_path") or "").strip()
        if not asset_path or not Path(asset_path).expanduser().is_file():
            continue  # pragma: no cover
        fmt = str(row.get("asset_format") or row.get("mesh_format") or Path(asset_path).suffix.lstrip(".")).lower()
        if fmt not in {"obj", "fbx", "glb", "gltf"}:
            continue
        cw = float(row.get("width_cm") or target_w * 100.0) / 100.0
        cd = float(row.get("depth_cm") or target_d * 100.0) / 100.0
        ch = float(row.get("height_cm") or target_h * 100.0) / 100.0
        size_score = abs(cw - target_w) + abs(cd - target_d) + abs(ch - target_h) * 0.6
        group_score = 0.0 if str(row.get("semantic_group") or "") == group else 0.35
        score = group_score + size_score
        candidate = {
            "unique_key": row.get("unique_key"),
            "source_site": row.get("source_site"),
            "title": row.get("title") or "Supplier Chair",
            "brand": row.get("brand"),
            "collection": row.get("collection"),
            "category_raw": row.get("category_raw"),
            "category_norm": row.get("category_norm"),
            "semantic_group": row.get("semantic_group") or group,
            "product_url": row.get("product_url"),
            "model_page_url": row.get("model_page_url"),
            "model_download_url": row.get("model_download_url"),
            "model_download_landing_url": row.get("model_download_landing_url"),
            "model_vendor_url": row.get("model_vendor_url"),
            "asset_status": row.get("asset_status") or "local_supplier_asset",
            "asset_format": fmt,
            "asset_local_path": asset_path,
            "price_value": row.get("price_value"),
            "price_currency": row.get("price_currency"),
            "style": row.get("style"),
            "color": row.get("color"),
            "materials": row.get("materials"),
            "width_cm": cw * 100.0,
            "depth_cm": cd * 100.0,
            "height_cm": ch * 100.0,
            "description": row.get("description"),
        }
        if best is None or score < best[0]:
            best = (score, candidate)
    return best[1] if best is not None else None


def _make_supplier_chair_for_table(
    table: dict[str, Any],
    chair_id: str,
    candidate: dict[str, Any],
    chair_aabb: dict[str, float],
    *,
    yaw_deg: float,
    status: str,
    collision_count: int,
    tuck_depth_m: float,
) -> dict[str, Any]:
    table_id = str(table.get("id") or "").strip()
    group = str(candidate.get("semantic_group") or "chair").strip().lower()
    item = {
        "id": chair_id,
        "name": str(candidate.get("title") or "Supplier Chair"),
        "category": "ArmChairFactory" if group == "armchair" else "ChairFactory",
        "constraints": {},
        "asset": _catalog_candidate_asset(candidate),
        "source": {
            "placement_source": "supplier_affordance_postprocess",
            "generated_for_table_id": table_id,
            "asset_source": "supplier_catalog_local_asset",
            "supplier_replaced": True,
            "supplier_unique_key": candidate.get("unique_key"),
            "supplier_source_site": candidate.get("source_site"),
            "supplier_product_url": candidate.get("product_url") or candidate.get("model_page_url"),
            "supplier_model_url": candidate.get("model_download_url"),
            "placeholder_bbox": False,
        },
        "meta": {
            "supplier_binding_applied": True,
            "supplier_affordance_added": True,
            "affordance": "table_chair",
            "target_table_id": table_id,
            "placeholder_bbox": False,
            "supplier_candidate": _compact_candidate(candidate),
            "placement_status": status,
            "collision_count": collision_count,
            "tuck_depth_m": tuck_depth_m,
        },
    }
    _set_item_pose_from_aabb(item, chair_aabb, yaw_deg=yaw_deg)
    return item


def _ensure_table_chair_affordances(
    data: dict[str, Any],
    items: list[dict[str, Any]],
    by_target_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    existing_ids = {str(item.get("id") or "").strip() for item in items if isinstance(item, dict)}
    room_bounds = _room_xy_bounds(data, items)
    door_keepouts = _door_keepout_aabbs(data)
    added: list[dict[str, Any]] = []
    moved: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []

    for item in list(items):
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("id") or "").strip()
        group = _semantic_group_for_item(item, by_target_id.get(item_id))
        if not (item_id and _table_requires_chair(item, group) and not _has_usable_nearby_chair(item, items + added, by_target_id)):
            continue
        table_aabb = _item_aabb(item)
        if not table_aabb:
            continue  # pragma: no cover

        base_id = f"auto_chair_for_{item_id}"
        chair_id = base_id
        suffix = 2
        while chair_id in existing_ids:
            chair_id = f"{base_id}_{suffix}"  # pragma: no cover
            suffix += 1  # pragma: no cover
        existing_ids.add(chair_id)

        candidate_item: dict[str, Any] | None = None
        candidate_source = "supplier_catalog"
        existing_chairs: list[dict[str, Any]] = []
        for other in items:
            if not isinstance(other, dict) or other is item:
                continue
            other_id = str(other.get("id") or "").strip()
            other_group = _semantic_group_for_item(other, by_target_id.get(other_id))
            if other_group in {"chair", "armchair"} and _item_aabb(other):
                existing_chairs.append(other)

        if existing_chairs:
            item_pos = _item_position(item) or _aabb_center(table_aabb)
            candidate_item = min(
                existing_chairs,
                key=lambda other: math.hypot((_item_position(other) or [0, 0, 0])[0] - item_pos[0], (_item_position(other) or [0, 0, 0])[1] - item_pos[1]),
            )
            candidate_source = "existing_scene_item"
            source_aabb = _item_aabb(candidate_item) or {}
            sx = max(source_aabb.get("x_max", 0.5) - source_aabb.get("x_min", 0.0), 0.42)
            sy = max(source_aabb.get("y_max", 0.52) - source_aabb.get("y_min", 0.0), 0.42)
            sz = max(source_aabb.get("z_max", 0.9) - source_aabb.get("z_min", 0.0), 0.72)
            z_min = min(float(table_aabb.get("z_min", 0.0)), 0.0)
        else:
            target_size = [0.48, 0.55, 0.9]
            catalog_candidate = _candidate_from_supplier_db("chair", target_size)
            if not catalog_candidate:
                tables.append({"table_id": item_id, "chair_id": None, "placement_status": "missing_supplier_asset"})  # pragma: no cover
                continue  # pragma: no cover
            sx = max(float(catalog_candidate.get("width_cm") or 48.0) / 100.0, 0.42)
            sy = max(float(catalog_candidate.get("depth_cm") or 55.0) / 100.0, 0.42)
            sz = max(float(catalog_candidate.get("height_cm") or 90.0) / 100.0, 0.72)
            z_min = min(float(table_aabb.get("z_min", 0.0)), 0.0)

        tuck_depth = 0.18
        room_cx, room_cy = _room_center_xy(data, items)
        table_cx = 0.5 * (table_aabb["x_min"] + table_aabb["x_max"])
        table_cy = 0.5 * (table_aabb["y_min"] + table_aabb["y_max"])
        table_width = table_aabb["x_max"] - table_aabb["x_min"]
        table_depth = table_aabb["y_max"] - table_aabb["y_min"]
        broad_sides = {"north", "south"} if table_width >= table_depth else {"east", "west"}
        candidates = [
            ("south", table_cx, table_aabb["y_min"] - 0.5 * sy + tuck_depth, 180.0),
            ("north", table_cx, table_aabb["y_max"] + 0.5 * sy - tuck_depth, 0.0),
            ("west", table_aabb["x_min"] - 0.5 * sx + tuck_depth, table_cy, 90.0),
            ("east", table_aabb["x_max"] + 0.5 * sx - tuck_depth, table_cy, 270.0),
        ]
        candidates.sort(key=lambda c: (0 if c[0] in broad_sides else 1, math.hypot(c[1] - room_cx, c[2] - room_cy)))

        best: tuple[int, float, float, float, str, float, dict[str, float]] | None = None
        for side, cx, cy, yaw in candidates:
            if room_bounds:
                cx = min(max(cx, room_bounds["x_min"] + 0.5 * sx), room_bounds["x_max"] - 0.5 * sx)
                cy = min(max(cy, room_bounds["y_min"] + 0.5 * sy), room_bounds["y_max"] - 0.5 * sy)
            chair_aabb = {
                "x_min": cx - 0.5 * sx,
                "x_max": cx + 0.5 * sx,
                "y_min": cy - 0.5 * sy,
                "y_max": cy + 0.5 * sy,
                "z_min": z_min,
                "z_max": z_min + sz,
            }
            corners = [
                (chair_aabb["x_min"], chair_aabb["y_min"]),
                (chair_aabb["x_min"], chair_aabb["y_max"]),
                (chair_aabb["x_max"], chair_aabb["y_min"]),
                (chair_aabb["x_max"], chair_aabb["y_max"]),
            ]
            if side == "south":
                actual_tuck = max(0.0, chair_aabb["y_max"] - table_aabb["y_min"])
            elif side == "north":
                actual_tuck = max(0.0, table_aabb["y_max"] - chair_aabb["y_min"])
            elif side == "west":
                actual_tuck = max(0.0, chair_aabb["x_max"] - table_aabb["x_min"])  # pragma: no cover
            else:
                actual_tuck = max(0.0, table_aabb["x_max"] - chair_aabb["x_min"])
            back_clear = actual_tuck <= tuck_depth + 1e-6
            inside_room = all(_point_in_room_xy(data, x, y) for x, y in corners)
            inside_bounds = True
            if room_bounds:
                inside_bounds = (
                    chair_aabb["x_min"] >= room_bounds["x_min"]
                    and chair_aabb["x_max"] <= room_bounds["x_max"]
                    and chair_aabb["y_min"] >= room_bounds["y_min"]
                    and chair_aabb["y_max"] <= room_bounds["y_max"]
                )
            collisions = 0
            for occ in items + added:
                if not isinstance(occ, dict):
                    continue  # pragma: no cover
                occ_id = str(occ.get("id") or "").strip()
                if occ_id == item_id or (candidate_item is not None and occ is candidate_item):
                    continue
                occ_aabb = _item_aabb(occ)
                z_overlaps = bool(
                    occ_aabb
                    and chair_aabb["z_max"] > occ_aabb["z_min"] + 0.03
                    and chair_aabb["z_min"] < occ_aabb["z_max"] - 0.03
                )
                if occ_aabb and z_overlaps and _xy_aabb_overlap(chair_aabb, occ_aabb, margin=0.04):
                    collisions += 1
            door_conflicts = sum(1 for keepout in door_keepouts if _xy_aabb_overlap(chair_aabb, keepout, margin=0.0))
            collisions += door_conflicts * 5
            status = "valid" if inside_room and inside_bounds and back_clear and collisions == 0 else "best_effort"
            score = collisions + (0 if inside_room and inside_bounds and back_clear else 100)
            if best is None or score < best[0]:
                best = (score, cx, cy, yaw, status, actual_tuck, chair_aabb)
            if status == "valid":
                break

        if best is None:
            continue  # pragma: no cover
        score, cx, cy, yaw, status, actual_tuck, chair_aabb = best
        collision_count = max(0, score if score < 100 else score - 100)
        if candidate_item is not None:
            moved_item = deepcopy(candidate_item)
            _set_item_pose_from_aabb(moved_item, chair_aabb, yaw_deg=yaw)
            meta = deepcopy(moved_item.get("meta") or {})
            meta["supplier_affordance_moved"] = True
            meta["affordance"] = "table_chair"
            meta["target_table_id"] = item_id
            meta["placement_status"] = status
            meta["collision_count"] = collision_count
            meta["tuck_depth_m"] = actual_tuck
            moved_item["meta"] = meta
            for idx, existing in enumerate(items):
                if existing is candidate_item:
                    items[idx] = moved_item
                    break
            moved.append({"chair_id": str(moved_item.get("id") or ""), "table_id": item_id, "placement_status": status})
            tables.append({"table_id": item_id, "chair_id": str(moved_item.get("id") or ""), "placement_status": status, "source": candidate_source})
            continue

        catalog_candidate = _candidate_from_supplier_db("chair", [sx, sy, sz])
        if not catalog_candidate:
            tables.append({"table_id": item_id, "chair_id": None, "placement_status": "missing_supplier_asset"})  # pragma: no cover
            continue  # pragma: no cover
        supplier_chair = _make_supplier_chair_for_table(
            item,
            chair_id,
            catalog_candidate,
            chair_aabb,
            yaw_deg=yaw,
            status=status,
            collision_count=collision_count,
            tuck_depth_m=actual_tuck,
        )
        added.append(supplier_chair)
        tables.append({"table_id": item_id, "chair_id": chair_id, "placement_status": status, "source": candidate_source})

    if not added:
        return items, {"added_count": 0, "added_ids": [], "moved_count": len(moved), "moved": moved, "tables": tables}
    return items + added, {
        "added_count": len(added),
        "added_ids": [item["id"] for item in added],
        "moved_count": len(moved),
        "moved": moved,
        "tables": tables,
    }


def _scene_has_tv(items: list[dict[str, Any]], by_target_id: dict[str, dict[str, Any]]) -> bool:
    for item in items:
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("id") or "").strip()
        text = " ".join(str(value or "") for value in (item.get("category"), item.get("name"), item.get("semantic_group")))
        group = _semantic_group_for_item(item, by_target_id.get(item_id))
        if group == "tv_projector_screen" and _looks_like_tv_text(text):
            return True
        meta = item.get("meta") or {}
        candidate = meta.get("supplier_candidate") if isinstance(meta.get("supplier_candidate"), dict) else {}
        candidate_text = " ".join(
            str(candidate.get(key) or "")
            for key in ("title", "category_norm", "category_raw", "description")
        )
        if str(candidate.get("category_norm") or "") == "tv_projector_screen" and _looks_like_tv_text(candidate_text):
            return True
    return False


def _has_clear_tv_volume(aabb: dict[str, float], items: list[dict[str, Any]], *, ignore_id: str | None = None) -> bool:
    for item in items:
        if not isinstance(item, dict):
            continue
        if ignore_id and str(item.get("id") or "").strip() == ignore_id:
            continue
        other = _item_aabb(item)
        if other and _xy_aabb_overlap(aabb, other, margin=0.04):
            z_overlap = aabb["z_max"] > other["z_min"] + 0.02 and aabb["z_min"] < other["z_max"] - 0.02
            if z_overlap:
                return False
    return True


def _next_generated_id(prefix: str, existing_ids: set[str]) -> str:
    item_id = prefix
    suffix = 2
    while item_id in existing_ids:
        item_id = f"{prefix}_{suffix}"
        suffix += 1
    existing_ids.add(item_id)
    return item_id


def _make_supplier_tv_item(
    *,
    tv_id: str,
    candidate: dict[str, Any],
    aabb: dict[str, float],
    yaw_deg: float,
    affordance: str,
    anchor_id: str,
    placement_status: str,
) -> dict[str, Any]:
    item = {
        "id": tv_id,
        "name": str(candidate.get("title") or "Supplier TV"),
        "category": "WallMountedTVFactory",
        "constraints": {"mount_type": "wall"},
        "asset": _catalog_candidate_asset(candidate),
        "source": {
            "placement_source": "supplier_affordance_postprocess",
            "generated_for_anchor_id": anchor_id,
            "asset_source": "supplier_catalog_local_asset",
            "supplier_replaced": True,
            "supplier_unique_key": candidate.get("unique_key"),
            "supplier_source_site": candidate.get("source_site"),
            "supplier_product_url": candidate.get("product_url") or candidate.get("model_page_url"),
            "supplier_model_url": candidate.get("model_download_url"),
            "placeholder_bbox": False,
        },
        "meta": {
            "supplier_binding_applied": True,
            "supplier_affordance_added": True,
            "affordance": affordance,
            "target_anchor_id": anchor_id,
            "placeholder_bbox": False,
            "supplier_candidate": _compact_candidate(candidate),
            "placement_status": placement_status,
        },
    }
    _set_item_pose_from_aabb(item, aabb, yaw_deg=yaw_deg)
    return item


def _candidate_tv_size(candidate: dict[str, Any] | None) -> tuple[float, float, float]:
    if not isinstance(candidate, dict):
        return 1.1, 0.06, 0.65
    try:
        width = max(float(candidate.get("width_cm") or 110.0) / 100.0, 0.65)
        height = max(float(candidate.get("height_cm") or 65.0) / 100.0, 0.36)
        depth = max(min(float(candidate.get("depth_cm") or 6.0) / 100.0, 0.16), 0.035)
        return width, depth, height
    except Exception:
        return 1.1, 0.06, 0.65


def _candidate_size_m_or_fallback(candidate: dict[str, Any], fallback: list[float]) -> tuple[float, float, float]:
    try:
        return (
            max(float(candidate.get("width_cm") or fallback[0] * 100.0) / 100.0, 0.02),
            max(float(candidate.get("depth_cm") or fallback[1] * 100.0) / 100.0, 0.02),
            max(float(candidate.get("height_cm") or fallback[2] * 100.0) / 100.0, 0.02),
        )
    except Exception:
        return float(fallback[0]), float(fallback[1]), float(fallback[2])


def _ensure_computer_replacements(
    items: list[dict[str, Any]],
    by_target_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    replaced: list[dict[str, Any]] = []
    suppressed_ids: set[str] = set()
    suppress_keyboard_near: list[dict[str, float]] = []
    out: list[dict[str, Any]] = []

    def support_top_for_computer(item_id: str, source_aabb: dict[str, float]) -> float | None:
        support_groups = {"desk", "side_table", "nightstand", "dresser", "tv_stand", "shelf"}
        cx = 0.5 * (source_aabb["x_min"] + source_aabb["x_max"])
        cy = 0.5 * (source_aabb["y_min"] + source_aabb["y_max"])
        best: tuple[float, float] | None = None
        for other in items:
            if not isinstance(other, dict):
                continue  # pragma: no cover
            other_id = str(other.get("id") or "").strip()
            if other_id == item_id:
                continue
            other_group = _semantic_group_for_item(other, by_target_id.get(other_id))
            if other_group not in support_groups:
                continue
            other_aabb = _item_aabb(other)
            if other_aabb is None:
                continue  # pragma: no cover
            if not _xy_inside_expanded(other_aabb, [cx, cy], margin=0.18):
                continue  # pragma: no cover
            dz = abs(float(source_aabb["z_min"]) - float(other_aabb["z_max"]))
            if best is None or dz < best[0]:
                best = (dz, float(other_aabb["z_max"]))
        return None if best is None else best[1]

    for item in items:
        if not isinstance(item, dict):
            out.append(item)
            continue
        item_id = str(item.get("id") or "").strip()
        if _semantic_group_for_item(item, by_target_id.get(item_id)) != "computer":
            out.append(item)
            continue
        item_kind = _computer_item_kind(item)
        if item_kind == "keyboard_mouse":
            out.append(item)
            continue
        meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
        if isinstance(meta.get("supplier_candidate"), dict) and item.get("asset"):
            out.append(item)  # pragma: no cover
            continue  # pragma: no cover
        aabb = _item_aabb(item)
        if aabb is None:
            out.append(item)  # pragma: no cover
            continue  # pragma: no cover
        raw_size = item.get("size_m") if isinstance(item.get("size_m"), list) and len(item.get("size_m")) >= 3 else None
        target_size = [float(raw_size[0]), float(raw_size[1]), float(raw_size[2])] if raw_size else [
            aabb["x_max"] - aabb["x_min"],
            aabb["y_max"] - aabb["y_min"],
            aabb["z_max"] - aabb["z_min"],
        ]
        candidate = _candidate_from_supplier_catalog_json(
            {"laptop_computer_keyboard_mouse", "computer", "computer_monitor", "tv_projector_screen"},
            target_size,
            computer_kind=item_kind,
        )
        if not candidate:
            out.append(item)  # pragma: no cover
            continue  # pragma: no cover

        width, depth, height = _candidate_size_m_or_fallback(candidate, target_size)
        yaw = float(item.get("yaw_deg", item.get("rotation_deg", 0.0)) or 0.0)
        yaw_mod = yaw % 180.0
        if abs(yaw_mod - 90.0) < 45.0:
            sx, sy = width, depth  # pragma: no cover
        else:
            sx, sy = depth, width
        cx = 0.5 * (aabb["x_min"] + aabb["x_max"])
        cy = 0.5 * (aabb["y_min"] + aabb["y_max"])
        support_top = support_top_for_computer(item_id, aabb)
        z_min = float(aabb["z_min"])
        if support_top is not None:
            z_min = float(support_top) + 0.004
        new_aabb = {
            "x_min": cx - 0.5 * sx,
            "x_max": cx + 0.5 * sx,
            "y_min": cy - 0.5 * sy,
            "y_max": cy + 0.5 * sy,
            "z_min": z_min,
            "z_max": z_min + height,
        }
        updated = deepcopy(item)
        updated["name"] = str(candidate.get("title") or item.get("name") or "Computer")
        updated["asset"] = _catalog_candidate_asset(candidate)
        updated["source"] = {
            **(updated.get("source") if isinstance(updated.get("source"), dict) else {}),
            "asset_source": "supplier_catalog_local_asset",
            "supplier_replaced": True,
            "supplier_target_id": item_id,
            "supplier_unique_key": candidate.get("unique_key"),
            "supplier_source_site": candidate.get("source_site"),
            "supplier_product_url": candidate.get("product_url") or candidate.get("model_page_url"),
            "supplier_model_url": candidate.get("model_download_url"),
            "placeholder_bbox": False,
        }
        _set_item_pose_from_aabb(updated, new_aabb, yaw_deg=yaw)
        updated_meta = deepcopy(meta)
        updated_meta["supplier_binding_applied"] = True
        updated_meta["supplier_affordance_replaced"] = True
        updated_meta["affordance"] = "computer_replacement"
        updated_meta["computer_target_kind"] = item_kind
        updated_meta["computer_candidate_kind"] = _computer_text_kind(
            " ".join(str(candidate.get(key) or "") for key in ("title", "category_norm", "category_raw"))
        )
        updated_meta["placeholder_bbox"] = False
        updated_meta["supplier_candidate"] = _compact_candidate(candidate)
        updated["meta"] = updated_meta
        out.append(updated)
        if updated_meta["computer_candidate_kind"] in {"laptop", "all_in_one"}:
            suppress_keyboard_near.append(new_aabb)
        replaced.append({"id": item_id, "title": candidate.get("title"), "unique_key": candidate.get("unique_key")})

    if suppress_keyboard_near:
        filtered: list[dict[str, Any]] = []
        for item in out:
            if not isinstance(item, dict):
                filtered.append(item)  # pragma: no cover
                continue  # pragma: no cover
            item_id = str(item.get("id") or "").strip()
            if _computer_item_kind(item) == "keyboard_mouse":
                aabb = _item_aabb(item)
                if aabb and any(_xy_aabb_overlap(aabb, repl_aabb, margin=0.12) for repl_aabb in suppress_keyboard_near):
                    suppressed_ids.add(item_id)
                    continue
            filtered.append(item)
        out = filtered
    return out, {"replaced_count": len(replaced), "replaced": replaced, "suppressed_keyboard_ids": sorted(suppressed_ids)}


def _tv_aabb_on_stand(stand_aabb: dict[str, float], tv_size: tuple[float, float, float]) -> tuple[dict[str, float], float]:
    tv_w, tv_d, tv_h = tv_size
    cx = 0.5 * (stand_aabb["x_min"] + stand_aabb["x_max"])
    cy = 0.5 * (stand_aabb["y_min"] + stand_aabb["y_max"])
    z_min = max(stand_aabb["z_max"] + 0.08, 0.78)
    z_max = z_min + tv_h
    stand_w = stand_aabb["x_max"] - stand_aabb["x_min"]
    stand_d = stand_aabb["y_max"] - stand_aabb["y_min"]
    if stand_d > stand_w:
        aabb = {
            "x_min": cx - 0.5 * tv_d,
            "x_max": cx + 0.5 * tv_d,
            "y_min": cy - 0.5 * tv_w,
            "y_max": cy + 0.5 * tv_w,
            "z_min": z_min,
            "z_max": z_max,
        }
        return aabb, 90.0
    aabb = {
        "x_min": cx - 0.5 * tv_w,
        "x_max": cx + 0.5 * tv_w,
        "y_min": cy - 0.5 * tv_d,
        "y_max": cy + 0.5 * tv_d,
        "z_min": z_min,
        "z_max": z_max,
    }
    return aabb, 0.0


def _wall_tv_pose_for_anchor(
    data: dict[str, Any],
    items: list[dict[str, Any]],
    anchor: dict[str, Any],
    tv_size: tuple[float, float, float],
    *,
    anchor_group: str,
) -> tuple[dict[str, float], float] | None:
    bounds = _room_xy_bounds(data, items)
    anchor_pos = _item_position(anchor)
    if bounds is None or anchor_pos is None:
        return None
    room_cx, room_cy = _room_center_xy(data, items)
    yaw_rad = math.radians(float(anchor.get("rotation_deg", anchor.get("yaw_deg", 0.0)) or 0.0))
    dx = math.sin(yaw_rad)
    dy = -math.cos(yaw_rad)
    if abs(dx) < 0.2 and abs(dy) < 0.2:
        dx = room_cx - anchor_pos[0]  # pragma: no cover
        dy = room_cy - anchor_pos[1]  # pragma: no cover
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        dy = -1.0  # pragma: no cover

    tv_w, tv_d, tv_h = tv_size
    z_center = 1.45 if anchor_group == "bed" else 1.35
    z_min = max(0.75, z_center - 0.5 * tv_h)
    z_max = z_min + tv_h
    margin = 0.08

    if abs(dx) >= abs(dy):
        if dx >= 0:
            x_min = bounds["x_max"] - margin - tv_d
            x_max = bounds["x_max"] - margin
            yaw = 90.0
        else:
            x_min = bounds["x_min"] + margin
            x_max = bounds["x_min"] + margin + tv_d
            yaw = 270.0
        cy = max(bounds["y_min"] + 0.5 * tv_w, min(bounds["y_max"] - 0.5 * tv_w, anchor_pos[1]))
        aabb = {
            "x_min": x_min,
            "x_max": x_max,
            "y_min": cy - 0.5 * tv_w,
            "y_max": cy + 0.5 * tv_w,
            "z_min": z_min,
            "z_max": z_max,
        }
        return aabb, yaw

    if dy >= 0:
        y_min = bounds["y_max"] - margin - tv_d
        y_max = bounds["y_max"] - margin
        yaw = 180.0
    else:
        y_min = bounds["y_min"] + margin
        y_max = bounds["y_min"] + margin + tv_d
        yaw = 0.0
    cx = max(bounds["x_min"] + 0.5 * tv_w, min(bounds["x_max"] - 0.5 * tv_w, anchor_pos[0]))
    aabb = {
        "x_min": cx - 0.5 * tv_w,
        "x_max": cx + 0.5 * tv_w,
        "y_min": y_min,
        "y_max": y_max,
        "z_min": z_min,
        "z_max": z_max,
    }
    return aabb, yaw


def _ensure_tv_affordance(
    data: dict[str, Any],
    items: list[dict[str, Any]],
    by_target_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if _scene_has_tv(items, by_target_id):
        return items, {"status": "skipped", "added_count": 0, "added_ids": [], "skipped_reason": "tv_already_present"}

    room = data.get("room") if isinstance(data.get("room"), dict) else {}
    room_type = str(room.get("room_type") or room.get("type") or "").strip().lower()
    prompt_blob = " ".join(
        str(x or "")
        for x in (
            room.get("style_hint"),
            room.get("description"),
            (data.get("meta") or {}).get("prompt") if isinstance(data.get("meta"), dict) else None,
        )
    ).lower()
    tv_requested = any(word in prompt_blob for word in ("tv", "television", "smart tv", "home cinema", "кинотеатр", "телевиз"))
    if room_type == "bedroom" and not tv_requested:
        return items, {
            "status": "skipped",
            "added_count": 0,
            "added_ids": [],
            "skipped_reason": "bedroom_tv_not_requested",
        }

    candidate = _candidate_from_supplier_catalog_json({"tv_projector_screen"}, [1.1, 0.06, 0.65])
    if not candidate:
        return items, {"status": "skipped", "added_count": 0, "added_ids": [], "skipped_reason": "missing_supplier_tv_asset"}

    tv_size = _candidate_tv_size(candidate)
    existing_ids = {str(item.get("id") or "").strip() for item in items if isinstance(item, dict)}
    attempts: list[dict[str, Any]] = []

    for item in items:
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("id") or "").strip()
        if _semantic_group_for_item(item, by_target_id.get(item_id)) != "tv_stand":
            continue
        stand_aabb = _item_aabb(item)
        if not stand_aabb:
            continue
        tv_aabb, tv_yaw = _tv_aabb_on_stand(stand_aabb, tv_size)
        clear = _has_clear_tv_volume(tv_aabb, items, ignore_id=item_id)
        attempts.append({"anchor_id": item_id, "mode": "tv_stand_top", "clear": clear})
        if not clear:
            continue
        tv_id = _next_generated_id(f"auto_tv_for_{item_id}", existing_ids)
        tv_item = _make_supplier_tv_item(
            tv_id=tv_id,
            candidate=candidate,
            aabb=tv_aabb,
            yaw_deg=tv_yaw,
            affordance="tv_on_stand",
            anchor_id=item_id,
            placement_status="valid",
        )
        return items + [tv_item], {
            "status": "added",
            "added_count": 1,
            "added_ids": [tv_id],
            "anchor_id": item_id,
            "mode": "tv_stand_top",
            "attempts": attempts,
        }

    anchors: list[tuple[int, dict[str, Any], str]] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("id") or "").strip()
        group = _semantic_group_for_item(item, by_target_id.get(item_id))
        if group in {"sofa", "bed"}:
            anchors.append((0 if group == "sofa" else 1, item, group))
    anchors.sort(key=lambda x: x[0])

    for _rank, anchor, group in anchors:
        anchor_id = str(anchor.get("id") or "").strip()
        pose = _wall_tv_pose_for_anchor(data, items, anchor, tv_size, anchor_group=group)
        if pose is None:
            attempts.append({"anchor_id": anchor_id, "mode": f"opposite_{group}", "clear": False, "reason": "no_wall_pose"})  # pragma: no cover
            continue  # pragma: no cover
        tv_aabb, yaw = pose
        clear = _has_clear_tv_volume(tv_aabb, items, ignore_id=anchor_id)
        attempts.append({"anchor_id": anchor_id, "mode": f"opposite_{group}", "clear": clear})
        if not clear:
            continue
        tv_id = _next_generated_id(f"auto_tv_opposite_{anchor_id}", existing_ids)
        tv_item = _make_supplier_tv_item(
            tv_id=tv_id,
            candidate=candidate,
            aabb=tv_aabb,
            yaw_deg=yaw,
            affordance=f"tv_opposite_{group}",
            anchor_id=anchor_id,
            placement_status="valid",
        )
        return items + [tv_item], {
            "status": "added",
            "added_count": 1,
            "added_ids": [tv_id],
            "anchor_id": anchor_id,
            "mode": f"opposite_{group}",
            "attempts": attempts,
        }

    return items, {"status": "skipped", "added_count": 0, "added_ids": [], "skipped_reason": "no_clear_tv_location", "attempts": attempts}


def apply_supplier_bindings_to_data(
    data: dict[str, Any],
    bindings_data: dict[str, Any],
    *,
    require_local_asset: bool = False,
    fallback_mode: str = ASSET_FALLBACK_MODE_NONE,
    preserve_generated_bedding: bool = True,
) -> dict[str, Any]:
    out = deepcopy(data)
    processed_collection_key = "placements" if isinstance(out.get("placements"), list) else "items"
    placements = out.get(processed_collection_key)
    if not isinstance(placements, list):
        raise RuntimeError("Некорректный scene/placement JSON: нет placements/items")
    processed_item_ids = {
        str(item.get("id") or "").strip()
        for item in placements
        if isinstance(item, dict) and str(item.get("id") or "").strip()
    }

    bindings = bindings_data.get("bindings") or []
    if not isinstance(bindings, list):
        raise RuntimeError("Некорректный supplier_bindings JSON: нет bindings")

    by_target_id = {
        str(b.get("target_id") or "").strip(): b
        for b in bindings
        if isinstance(b, dict) and str(b.get("target_id") or "").strip()
    }
    related_item_actions = _related_generated_item_actions(
        placements,
        by_target_id,
        preserve_generated_bedding=preserve_generated_bedding,
    )

    replaced = 0
    placeholder_replaced = 0
    local_asset_replaced = 0
    proxy_asset_replaced = 0
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
            new_items.append(item)  # pragma: no cover
            continue  # pragma: no cover
        if ((binding.get("provenance") or {}).get("final_asset_source")) not in {"supplier_catalog", "supplier_catalog_pending"}:
            new_items.append(item)  # pragma: no cover
            continue  # pragma: no cover
        if _should_keep_original_scene_item(item, binding):
            new_items.append(item)  # pragma: no cover
            continue  # pragma: no cover

        candidate_size_m = _candidate_size_m(chosen)

        asset_block, use_placeholder = _candidate_asset(
            chosen,
            require_local_asset=require_local_asset,
            fallback_mode=fallback_mode,
        )
        if require_local_asset and not asset_block:
            new_items.append(item)  # pragma: no cover
            continue  # pragma: no cover
        if not asset_block and _item_has_scene_geometry(item):
            new_items.append(item)  # pragma: no cover
            continue  # pragma: no cover

        updated = deepcopy(item)
        original_name = updated.get("name")
        original_category = updated.get("category")
        original_source = deepcopy(updated.get("source") or {})
        original_aabb = deepcopy(updated.get("aabb") or updated.get("bbox") or {})
        original_position_m = deepcopy(updated.get("position_m"))
        original_size_m = deepcopy(updated.get("size_m"))

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
        asset_kind = str(asset_block.get("kind") or "").strip().lower()
        if asset_kind == "procedural_proxy":
            source["asset_source"] = "supplier_catalog_procedural_proxy"
        elif asset_block:
            source["asset_source"] = str(asset_block.get("asset_source") or "supplier_catalog_local_asset")
        else:
            source["asset_source"] = "supplier_catalog_placeholder"
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
        meta["supplier_candidate_pool"] = _compact_candidate_pool(binding, limit=5)
        meta["supplier_selection_notes"] = deepcopy(binding.get("selection_notes") or [])
        meta["original_generated_item"] = {
            "id": item_id,
            "name": original_name,
            "category": original_category,
            "source": original_source,
            "aabb": original_aabb,
            "position_m": original_position_m,
            "size_m": original_size_m,
            "blend_object_name": original_source.get("blend_object_name") if isinstance(original_source, dict) else None,
        }
        updated["meta"] = meta

        replaced += 1
        if use_placeholder:
            placeholder_replaced += 1
        elif source["asset_source"] == "supplier_catalog_procedural_proxy":
            proxy_asset_replaced += 1
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
            continue  # pragma: no cover
        binding = by_target_id.get(item_id)
        if not _binding_has_supported_local_asset(binding):
            continue
        aabb = _item_aabb(item)
        if aabb is not None:
            final_anchor_aabbs[item_id] = aabb

    adjusted_items: list[dict[str, Any]] = []
    preserved_bedding_ids: list[str] = []
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
            adjusted_items.append(item)  # pragma: no cover
            continue  # pragma: no cover

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

        if support_mode == "top":
            z_min = anchor_aabb_new["z_max"] + 0.004
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
        if related_action.get("preserve_bedding"):
            meta["supplier_bedding_preserved"] = True
            preserved_bedding_ids.append(item_id)
        updated["meta"] = meta
        reanchored_count += 1
        adjusted_items.append(updated)

    new_items = adjusted_items
    new_items, ceiling_light_postprocess = _collapse_ceiling_lights(out, new_items, by_target_id)
    new_items, supported_light_postprocess = _normalize_supported_light_placements(out, new_items, by_target_id)
    new_items, computer_postprocess = _ensure_computer_replacements(new_items, by_target_id)
    new_items, table_chair_postprocess = _ensure_table_chair_affordances(out, new_items, by_target_id)
    new_items, tv_postprocess = _ensure_tv_affordance(out, new_items, by_target_id)

    def sync_parallel_collection(existing_items: list[Any], updated_items: list[Any]) -> list[Any]:
        updated_by_id = {
            str(item.get("id") or "").strip(): item
            for item in updated_items
            if isinstance(item, dict) and str(item.get("id") or "").strip()
        }
        existing_ids = {
            str(item.get("id") or "").strip()
            for item in existing_items
            if isinstance(item, dict) and str(item.get("id") or "").strip()
        }
        seen_ids: set[str] = set()
        synced: list[Any] = []
        for item in existing_items:
            if not isinstance(item, dict):
                synced.append(item)
                continue
            item_id = str(item.get("id") or "").strip()
            if not item_id:
                synced.append(item)
                continue
            if item_id in processed_item_ids:
                updated = updated_by_id.get(item_id)
                if updated is None:
                    continue
                synced.append(deepcopy(updated))
                seen_ids.add(item_id)
            else:
                synced.append(item)
                seen_ids.add(item_id)

        for item in updated_items:
            if not isinstance(item, dict):
                synced.append(deepcopy(item))
                continue
            item_id = str(item.get("id") or "").strip()
            if item_id and (item_id in seen_ids or item_id in existing_ids):
                continue
            synced.append(deepcopy(item))
            if item_id:
                seen_ids.add(item_id)
        return synced

    if processed_collection_key == "placements":
        out["placements"] = new_items
        if isinstance(out.get("items"), list):
            out["items"] = sync_parallel_collection(out["items"], new_items)
    else:
        out["items"] = new_items  # pragma: no cover
        if isinstance(out.get("placements"), list):  # pragma: no cover
            out["placements"] = sync_parallel_collection(out["placements"], new_items)  # pragma: no cover

    meta = deepcopy(out.get("meta") or {})
    meta["supplier_binding_summary"] = {
        "replaced_count": replaced,
        "placeholder_replaced_count": placeholder_replaced,
        "local_asset_replaced_count": local_asset_replaced,
        "suppressed_generated_related_count": suppressed_generated_count,
        "reanchored_generated_related_count": reanchored_count,
        "ceiling_light_deduplicated_count": int(ceiling_light_postprocess.get("removed_count", 0) or 0),
        "supported_light_normalized_count": int(supported_light_postprocess.get("moved_count", 0) or 0),
        "missing_table_chair_added_count": int(table_chair_postprocess.get("added_count", 0) or 0),
        "table_chair_moved_count": int(table_chair_postprocess.get("moved_count", 0) or 0),
        "missing_tv_added_count": int(tv_postprocess.get("added_count", 0) or 0),
        "computer_replaced_count": int(computer_postprocess.get("replaced_count", 0) or 0),
        "unusable_table_suppressed_count": 0,
        "require_local_asset": bool(require_local_asset),
        "supplier_asset_fallback_mode": str(fallback_mode).strip() or ASSET_FALLBACK_MODE_NONE,
        "proxy_asset_replaced_count": int(proxy_asset_replaced),
    }
    meta["supplier_bed_postprocess"] = {
        "preserved_bedding_count": len(preserved_bedding_ids),
        "preserved_bedding_ids": preserved_bedding_ids,
        "policy": (
            "keep_generated_bedding_when_replacing_bed_frame"
            if preserve_generated_bedding
            else "suppress_generated_bedding_when_replacing_bed"
        ),
    }
    meta["supplier_postprocess"] = {
        "ceiling_lights": ceiling_light_postprocess,
        "supported_lights": supported_light_postprocess,
        "computer_replacements": computer_postprocess,
        "table_chair_affordance": table_chair_postprocess,
        "tv_affordance": tv_postprocess,
    }
    out["meta"] = meta
    return out


def apply_supplier_bindings_to_json(
    input_json_path: str | Path,
    bindings_json_path: str | Path,
    output_json_path: str | Path,
    *,
    require_local_asset: bool = False,
    fallback_mode: str = ASSET_FALLBACK_MODE_NONE,
    preserve_generated_bedding: bool = True,
) -> Path:
    data = read_json(input_json_path)
    bindings = read_json(bindings_json_path)
    out = apply_supplier_bindings_to_data(
        data,
        bindings,
        require_local_asset=require_local_asset,
        fallback_mode=fallback_mode,
        preserve_generated_bedding=preserve_generated_bedding,
    )
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
    ap.add_argument(
        "--supplier-asset-fallback-mode",
        choices=[ASSET_FALLBACK_MODE_NONE, ASSET_FALLBACK_MODE_FBX_OBJ_PROXY, ASSET_FALLBACK_MODE_FBX_OBJ_TRELLIS_PROXY],
        default=ASSET_FALLBACK_MODE_NONE,
        help="Fallback policy for unavailable local assets.",
    )
    ap.add_argument(
        "--suppress-generated-bedding",
        action="store_true",
        help="Remove generated bed soft parts when their bed is replaced by a supplier/TRELLIS asset.",
    )
    return ap


def main() -> None:
    args = build_cli().parse_args()
    out_path = apply_supplier_bindings_to_json(
        input_json_path=args.input_json,
        bindings_json_path=args.bindings_json,
        output_json_path=args.out,
        require_local_asset=bool(args.require_local_asset),
        fallback_mode=str(getattr(args, "supplier_asset_fallback_mode", ASSET_FALLBACK_MODE_NONE)),
        preserve_generated_bedding=not bool(getattr(args, "suppress_generated_bedding", False)),
    )
    data = read_json(out_path)
    summary = (data.get("meta") or {}).get("supplier_binding_summary") or {}
    print(f"replaced = {summary.get('replaced_count', 0)}")
    print(f"placeholder_replaced = {summary.get('placeholder_replaced_count', 0)}")
    print(f"local_asset_replaced = {summary.get('local_asset_replaced_count', 0)}")
    print(f"saved = {out_path}")


if __name__ == "__main__":
    main()  # pragma: no cover
