#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any


_CAMEL_RE_1 = re.compile(r"([a-z0-9])([A-Z])")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

_SEMANTIC_ALIASES: dict[str, tuple[str, ...]] = {
    "bed": ("bed", "king size bed", "double bed", "single bed", "kids bed"),
    "nightstand": ("nightstand", "bedside"),
    "wardrobe": ("wardrobe", "closet"),
    "dresser": ("drawer chest", "chest of drawers", "dresser", "cabinet"),
    "desk": ("desk", "dressing table", "vanity"),
    "tv_stand": ("tv stand", "television stand"),
    "computer": ("computer", "monitor", "laptop", "keyboard", "macbook", "imac"),
    "armchair": ("armchair", "easy chair"),
    "chair": ("chair", "dining chair", "office chair", "lounge chair"),
    "stool": ("stool", "pouf", "pouffe", "ottoman", "пуф", "табурет"),
    "dining_table": ("dining table", "dining_table", "обеденный стол"),
    "sofa": ("sofa", "loveseat", "chaise longue sofa"),
    "coffee_table": ("coffee table",),
    "side_table": ("side table", "corner table", "end table"),
    "lamp_table": ("table lamp", "desk lamp"),
    "lamp_ceiling": ("ceiling lamp", "ceiling light", "pendant lamp", "chandelier"),
    "lamp_floor": ("floor lamp",),
    "lamp_wall": ("wall lamp", "wall light"),
    "wall_art": ("wall art", "picture"),
    "rug": ("rug",),
    "shelf": ("shelf", "bookcase", "bookshelf"),
    "mirror": ("mirror",),
    "bathroom_sink": ("standing sink", "sink", "washbasin", "basin", "раковина", "умывальник"),
    "kitchenware": ("kitchenware", "cooking set", "kitchen decor", "набор для готовки", "мелочь для кухни", "посуда"),
    "kitchen_faucet": ("kitchen faucet", "kitchen_faucet", "смеситель для кухни"),
    "food_drink": ("food drink", "food_drink", "fruit plate", "еда и напитки", "фрукт"),
    "decorative_set": ("decorative set", "decorative_set", "olive and oil", "oil decorative", "декоративный набор"),
    "plant_planter_vase": ("plant planter vase", "plant_planter_vase", "flower vase", "flower bouquet", "букет", "ваза"),
    "plant": ("plant",),
}

_SUPPLIER_REPLACE_GROUPS = {
    "bed",
    "nightstand",
    "wardrobe",
    "dresser",
    "desk",
    "tv_stand",
    "computer",
    "armchair",
    "chair",
    "stool",
    "dining_table",
    "sofa",
    "coffee_table",
    "side_table",
    "lamp_table",
    "lamp_ceiling",
    "lamp_floor",
    "shelf",
    "mirror",
    "bathroom_sink",
    "kitchenware",
    "kitchen_faucet",
    "food_drink",
    "decorative_set",
    "plant_planter_vase",
}

_KEEP_GENERATED_GROUPS = {
    "lamp_wall",
    "wall_art",
    "rug",
    "plant",
}

_CATEGORY_TO_SEMANTIC_GROUP = {
    "bookshelf": "shelf",
    "ceiling_light": "lamp_ceiling",
    "chandelier": "lamp_ceiling",
    "floor_lamp": "lamp_floor",
    "table_lamp": "lamp_table",
    "wall_light": "lamp_wall",
    "runner_rug": "rug",
}

_PROCEDURAL_KEEP_CATEGORIES = {
    "decor_books",
    "decor_box",
    "decor_tray",
    "decor_vase",
    "pillow",
    "blanket",
    "rug",
    "runner_rug",
    "wall_art",
    "tv_accessory",
}

_WALL_REPLACE_CATEGORIES = {"mirror", "wall_light", "tv"}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _semantic_text(value: Any) -> str:
    s = str(value or "").strip()
    s = _CAMEL_RE_1.sub(r"\1 \2", s)
    s = s.replace("_", " ").replace("-", " ").replace("/", " ")
    s = _NON_ALNUM_RE.sub(" ", s.lower())
    return " ".join(s.split())


def _semantic_group(name: Any, category: Any = None, constraints: dict[str, Any] | None = None) -> str:
    text = " ".join(x for x in [_semantic_text(name), _semantic_text(category)] if x).strip()
    if "desk lamp" in text or "table lamp" in text:
        return "lamp_table"
    for group, aliases in _SEMANTIC_ALIASES.items():
        if any(alias in text for alias in aliases):
            return group

    mount_type = _semantic_text((constraints or {}).get("mount_type"))
    if mount_type == "ceiling":
        return "lamp_ceiling"
    if mount_type == "wall":
        return "lamp_wall"
    return text or "unknown"


def _physical_role(item: dict[str, Any]) -> str:
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    return str(meta.get("physical_role") or "").strip().lower()


def _category_key(item: dict[str, Any]) -> str:
    return str(item.get("category") or item.get("name") or "").strip().lower()


def _semantic_group_for_item(item: dict[str, Any], constraints: dict[str, Any]) -> str:
    category = _category_key(item)
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    source = item.get("source") if isinstance(item.get("source"), dict) else {}
    is_procedural = bool(_physical_role(item)) or bool(meta.get("procedural")) or str(source.get("placement_source") or "").startswith(
        "procedural_room_stage"
    )
    explicit = str(item.get("semantic_group") or "").strip().lower()
    if explicit in _CATEGORY_TO_SEMANTIC_GROUP:
        return _CATEGORY_TO_SEMANTIC_GROUP[explicit]
    if category in _CATEGORY_TO_SEMANTIC_GROUP:
        return _CATEGORY_TO_SEMANTIC_GROUP[category]
    if explicit and explicit not in {"bed", "wardrobe", "nightstand"}:
        return explicit
    if is_procedural and category:
        return category
    return _semantic_group(item.get("name"), item.get("category"), constraints)


def _replacement_policy_for_item(item: dict[str, Any], *, placeholder_bbox: bool) -> tuple[str, str]:
    category = _category_key(item)
    role = _physical_role(item)

    if placeholder_bbox:
        return "replace_with_supplier", "placeholder_bbox_requires_real_asset"
    if category in {"nightstand", "side_table", "floor_lamp"}:
        return "replace_with_supplier", f"{category}_always_replace_with_supplier"
    if category in {"rug", "runner_rug"}:
        return "keep_generated", "rug_kept_generated_until_carpet_catalog"
    if role == "solid_floor":
        return "replace_with_supplier", "solid_floor_furniture_or_floor_item"
    if role == "ceiling_mounted":
        return "replace_with_supplier", "ceiling_mounted_lighting"
    if role == "wall_mounted" and category in _WALL_REPLACE_CATEGORIES:
        return "replace_with_supplier", "wall_mounted_replaceable_item"
    if role == "wall_mounted" and category == "wall_art":
        return "keep_generated", "wall_art_generated_placeholder"
    if role in {"on_top", "soft_on_object", "soft_floor", "decorative_soft"}:
        return "keep_generated", f"{role}_kept_generated"
    if category in _PROCEDURAL_KEEP_CATEGORIES:
        return "keep_generated", "procedural_decor_kept_generated"
    return _default_replacement_policy(
        semantic_group=_semantic_group_for_item(item, item.get("constraints") if isinstance(item.get("constraints"), dict) else {}),
        placeholder_bbox=placeholder_bbox,
    )


def _layout_source_for_item(item: dict[str, Any], scene: dict[str, Any]) -> str:
    source = item.get("source") if isinstance(item.get("source"), dict) else {}
    scene_meta = scene.get("meta") if isinstance(scene.get("meta"), dict) else {}
    value = (
        source.get("placement_source")
        or source.get("generator")
        or scene_meta.get("placer")
        or scene_meta.get("mode")
        or "unknown_layout"
    )
    return str(value or "unknown_layout")


def _extract_position_m(item: dict[str, Any], aabb: dict[str, Any]) -> list[float]:
    pos = item.get("position_m")
    if isinstance(pos, list) and len(pos) == 3:
        return [float(pos[0]), float(pos[1]), float(pos[2])]

    return [
        0.5 * (float(aabb.get("x_min", 0.0)) + float(aabb.get("x_max", 0.0))),
        0.5 * (float(aabb.get("y_min", 0.0)) + float(aabb.get("y_max", 0.0))),
        0.5 * (float(aabb.get("z_min", 0.0)) + float(aabb.get("z_max", 0.0))),
    ]


def _extract_size_m(item: dict[str, Any], aabb: dict[str, Any]) -> list[float]:
    size = item.get("size_m")
    if isinstance(size, list) and len(size) == 3:
        return [float(size[0]), float(size[1]), float(size[2])]

    return [
        max(0.0, float(aabb.get("x_max", 0.0)) - float(aabb.get("x_min", 0.0))),
        max(0.0, float(aabb.get("y_max", 0.0)) - float(aabb.get("y_min", 0.0))),
        max(0.0, float(aabb.get("z_max", 0.0)) - float(aabb.get("z_min", 0.0))),
    ]


def _default_replacement_policy(
    *,
    semantic_group: str,
    placeholder_bbox: bool,
) -> tuple[str, str]:
    if placeholder_bbox:
        return "replace_with_supplier", "placeholder_bbox_requires_real_asset"
    if semantic_group in _SUPPLIER_REPLACE_GROUPS:
        return "replace_with_supplier", "major_furniture_group"
    if semantic_group in _KEEP_GENERATED_GROUPS:
        return "keep_generated", "decor_or_lighting_group"
    return "keep_generated", "fallback_keep_generated"


def extract_layout_targets(scene_or_placement_path: str | Path, out_path: str | Path) -> Path:
    src_path = Path(scene_or_placement_path).expanduser().resolve()
    dst_path = Path(out_path).expanduser().resolve()
    data = _read_json(src_path)

    placements = data.get("placements") or data.get("items") or []
    if not isinstance(placements, list):
        raise RuntimeError(f"Некорректный scene/placement JSON: {src_path}")

    targets: list[dict[str, Any]] = []
    for idx, item in enumerate(placements):
        if not isinstance(item, dict):
            continue

        aabb = item.get("aabb") or {}
        if not isinstance(aabb, dict):
            aabb = {}

        constraints = deepcopy(item.get("constraints") or {})
        meta = deepcopy(item.get("meta") or {})
        source = deepcopy(item.get("source") or {})
        category = str(item.get("category") or item.get("name") or f"object_{idx}")
        name = str(item.get("name") or item.get("category") or f"object_{idx}")

        placeholder_bbox = bool(meta.get("placeholder_bbox") or source.get("placeholder_bbox"))
        semantic_group = _semantic_group_for_item(item, constraints)
        replacement_policy, replacement_reason = _replacement_policy_for_item(item, placeholder_bbox=placeholder_bbox)
        layout_source = _layout_source_for_item(item, data)
        target = {
            "target_id": str(item.get("id") or f"target_{idx:04d}"),
            "name": name,
            "category": category,
            "semantic_group": semantic_group,
            "position_m": _extract_position_m(item, aabb),
            "size_m": _extract_size_m(item, aabb),
            "rotation_deg": float(item.get("rotation_deg", item.get("yaw_deg", 0.0)) or 0.0),
            "yaw_deg": float(item.get("yaw_deg", item.get("rotation_deg", 0.0)) or 0.0),
            "aabb": deepcopy(aabb),
            "mount_type": constraints.get("mount_type"),
            "constraints": constraints,
            "placeholder_bbox": placeholder_bbox,
            "replacement_policy": replacement_policy,
            "replacement_reason": replacement_reason,
            "layout_source": layout_source,
            "source": source,
            "meta": {
                "placement_meta": meta,
                "physical_role": meta.get("physical_role"),
                "layout_source": layout_source,
                "room_id": ((data.get("room") or {}).get("id") if isinstance(data.get("room"), dict) else None),
                "placement_index": idx,
            },
        }
        targets.append(target)

    artifact = {
        "schema": "layout_targets/v1",
        "source_json": str(src_path),
        "room": deepcopy(data.get("room") or {}),
        "meta": {
            "placer": (data.get("meta") or {}).get("placer") if isinstance(data.get("meta"), dict) else None,
            "mode": (data.get("meta") or {}).get("mode") if isinstance(data.get("meta"), dict) else None,
            "target_count": len(targets),
            "placeholder_bbox_count": sum(1 for x in targets if x["placeholder_bbox"]),
        },
        "targets": targets,
    }
    _write_json(dst_path, artifact)
    return dst_path


def build_supplier_bindings_stub(targets_path: str | Path, out_path: str | Path) -> Path:
    src_path = Path(targets_path).expanduser().resolve()
    dst_path = Path(out_path).expanduser().resolve()
    data = _read_json(src_path)
    targets = data.get("targets") or []
    if not isinstance(targets, list):
        raise RuntimeError(f"Некорректный layout_targets JSON: {src_path}")

    bindings = []
    for target in targets:
        if not isinstance(target, dict):
            continue
        bindings.append(
            {
                "target_id": str(target.get("target_id") or ""),
                "category": target.get("category"),
                "semantic_group": target.get("semantic_group"),
                "requested_size_m": deepcopy(target.get("size_m") or [0.0, 0.0, 0.0]),
                "replacement_policy": target.get("replacement_policy") or "keep_generated",
                "replacement_reason": target.get("replacement_reason"),
                "provenance": {
                    "layout_source": target.get("layout_source") or (target.get("meta") or {}).get("layout_source") or "unknown_layout",
                    "final_asset_source": "pending",
                    "allowed_asset_sources": ["generated_native", "supplier_catalog"],
                },
                "selection_status": (
                    "pending_candidate_search"
                    if target.get("replacement_policy") == "replace_with_supplier"
                    else "kept_generated_stub"
                ),
                "candidate_count": 0,
                "top_candidates": [],
                "chosen_candidate": None,
                "pricing": {
                    "status": "pending",
                    "currency": "RUB",
                    "generated_estimate_value": None,
                    "supplier_price_value": None,
                    "final_price_value": None,
                    "final_price_source": None,
                },
                "selection_notes": [
                    "stub_only",
                    "future_step: procedural_rank_then_llm_choice",
                    "future_step: split_scene_cost_between_generated_and_supplier_assets",
                ],
            }
        )

    artifact = {
        "schema": "supplier_bindings_stub/v1",
        "layout_targets_json": str(src_path),
        "meta": {
            "target_count": len(bindings),
            "status": "stub_initialized",
        },
        "bindings": bindings,
    }
    _write_json(dst_path, artifact)
    return dst_path


def build_scene_pricing_stub(bindings_path: str | Path, out_path: str | Path) -> Path:
    src_path = Path(bindings_path).expanduser().resolve()
    dst_path = Path(out_path).expanduser().resolve()
    data = _read_json(src_path)
    bindings = data.get("bindings") or []
    if not isinstance(bindings, list):
        raise RuntimeError(f"Некорректный supplier_bindings JSON: {src_path}")

    scene_items = []
    for binding in bindings:
        if not isinstance(binding, dict):
            continue
        scene_items.append(
            {
                "target_id": binding.get("target_id"),
                "category": binding.get("category"),
                "semantic_group": binding.get("semantic_group"),
                "replacement_policy": binding.get("replacement_policy"),
                "pricing_bucket": "supplier_catalog"
                if binding.get("replacement_policy") == "replace_with_supplier"
                else "generated_native",
                "price_status": "pending",
                "currency": "RUB",
                "final_price_value": None,
                "final_asset_source": "pending",
            }
        )

    generated_like = sum(1 for x in scene_items if x["pricing_bucket"] == "generated_native_or_supplier_pending")
    supplier_like = sum(1 for x in scene_items if x["pricing_bucket"] == "supplier_catalog")

    artifact = {
        "schema": "scene_pricing_stub/v1",
        "supplier_bindings_stub_json": str(src_path),
        "meta": {
            "status": "stub_initialized",
            "currency": "RUB",
            "scene_item_count": len(scene_items),
            "generated_native_or_supplier_pending_count": generated_like,
            "supplier_catalog_count": supplier_like,
        },
        "totals": {
            "generated_native_estimate_value": None,
            "supplier_catalog_total_value": None,
            "final_scene_total_value": None,
        },
        "items": scene_items,
    }
    _write_json(dst_path, artifact)
    return dst_path


def create_layout_selection_stub_artifacts(
    *,
    source_json_path: str | Path,
    run_dir: str | Path,
    prefix: str = "",
) -> dict[str, str]:
    run_dir_path = Path(run_dir).expanduser().resolve()
    stem = f"{prefix}_" if prefix else ""

    targets_path = run_dir_path / f"{stem}layout_targets.json"
    bindings_path = run_dir_path / f"{stem}supplier_bindings.stub.json"
    pricing_path = run_dir_path / f"{stem}scene_pricing.stub.json"

    extract_layout_targets(source_json_path, targets_path)
    build_supplier_bindings_stub(targets_path, bindings_path)
    build_scene_pricing_stub(bindings_path, pricing_path)
    return {
        "layout_targets_json": str(targets_path.resolve()),
        "supplier_bindings_stub_json": str(bindings_path.resolve()),
        "scene_pricing_stub_json": str(pricing_path.resolve()),
    }
