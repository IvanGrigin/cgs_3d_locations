from __future__ import annotations

from copy import deepcopy
from typing import Any


def _mm_to_m(value: Any) -> Any:
    if isinstance(value, dict):
        converted: dict[str, Any] = {}
        for key, val in value.items():
            if key.endswith("_mm") and isinstance(val, (int, float)):
                converted[key[:-3] + "_m"] = round(float(val) / 1000.0, 6)
            else:
                converted[key] = _mm_to_m(val)
        return converted
    if isinstance(value, list):
        return [_mm_to_m(item) for item in value]
    return value


def _material_summary(material: dict[str, Any] | None) -> dict[str, Any] | None:
    if not material:
        return None
    return {
        "sku": material.get("sku"),
        "name": material.get("name"),
        "brand": material.get("brand"),
        "url": material.get("url"),
        "price": material.get("price"),
        "price_currency": material.get("price_currency"),
        "availability": material.get("availability"),
        "kitchen_role": material.get("kitchen_role"),
        "dimensions": material.get("dimensions"),
        "visual": material.get("visual"),
        "flags": material.get("flags"),
        "image_url": material.get("image_url"),
        "local_image": material.get("local_image"),
    }


def _extract_material_bindings(selected_materials: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for target_role, entry in (selected_materials.get("materials") or {}).items():
        result[target_role] = {
            "target_role": target_role,
            "chosen_material": _material_summary(entry.get("chosen_material")),
            "final_score": entry.get("final_score"),
            "score_breakdown": entry.get("score_breakdown"),
            "top_candidates": [
                {
                    "material": _material_summary(candidate.get("material")),
                    "final_score": candidate.get("final_score"),
                    "score_breakdown": candidate.get("score_breakdown"),
                }
                for candidate in entry.get("top_candidates", [])
            ],
        }
    return result


def build_kitchen_assembly_json(
    target_id: str,
    layout_plan: dict[str, Any],
    selected_materials: dict[str, Any],
    bill_of_materials: dict[str, Any],
    design_spec: dict[str, Any],
    mode: str,
    appliance_assets: dict[str, Any] | None = None,
    position: list[float] | None = None,
    rotation: list[float] | None = None,
) -> dict[str, Any]:
    position = position or [0.0, 0.0, 0.0]
    rotation = rotation or [0.0, 0.0, 0.0]
    max_width_mm = max(
        [layout_plan.get("total_width_mm") or 0]
        + [
            m.get("x_mm", 0) + (m.get("width_mm", 0) if m.get("orientation", "x") == "x" else m.get("depth_mm", 0))
            for m in layout_plan.get("base_modules") or []
        ]
        + [
            m.get("x_mm", 0) + (m.get("width_mm", 0) if m.get("orientation", "x") == "x" else m.get("depth_mm", 0))
            for m in layout_plan.get("upper_modules") or []
        ]
    )
    max_height_mm = max(
        [0]
        + [m.get("z_mm", 0) + m.get("height_mm", 0) for m in layout_plan.get("base_modules") or []]
        + [m.get("z_mm", 0) + m.get("height_mm", 0) for m in layout_plan.get("upper_modules") or []]
    )
    max_depth_mm = max(
        [600]
        + [
            m.get("y_mm", 0) + (m.get("depth_mm", 0) if m.get("orientation", "x") == "x" else m.get("width_mm", 0))
            for m in layout_plan.get("base_modules") or []
        ]
        + [
            m.get("y_mm", 0) + (m.get("depth_mm", 0) if m.get("orientation", "x") == "x" else m.get("width_mm", 0))
            for m in layout_plan.get("upper_modules") or []
        ]
    )
    return {
        "id": target_id,
        "type": "procedural_assembly",
        "category": "kitchen_set",
        "assembly_type": "procedural_kitchen",
        "layout_type": layout_plan.get("layout_type"),
        "layout_variant": layout_plan.get("layout_variant"),
        "mode": mode,
        "position": position,
        "rotation": rotation,
        "dimensions": {
            "width_m": round(max_width_mm / 1000.0, 6),
            "depth_m": round(max_depth_mm / 1000.0, 6),
            "height_m": round(max_height_mm / 1000.0, 6),
        },
        "design_spec": design_spec,
        "layout_plan_mm": layout_plan,
        "material_bindings": _extract_material_bindings(selected_materials),
        "appliance_bindings": appliance_assets or {"appliances": {}, "warnings": []},
        "palette_consistency_score": selected_materials.get("palette_consistency_score"),
        "base_modules": _mm_to_m(deepcopy(layout_plan.get("base_modules") or [])),
        "countertop_segments": _mm_to_m(deepcopy(layout_plan.get("countertop_segments") or [])),
        "countertop": _mm_to_m(deepcopy(layout_plan.get("countertop"))),
        "backsplash_segments": _mm_to_m(deepcopy(layout_plan.get("backsplash_segments") or [])),
        "backsplash": _mm_to_m(deepcopy(layout_plan.get("backsplash"))),
        "upper_modules": _mm_to_m(deepcopy(layout_plan.get("upper_modules") or [])),
        "decor_items": _mm_to_m(deepcopy(layout_plan.get("decor_items") or [])),
        "bill_of_materials": bill_of_materials,
        "price_estimate": {
            "currency": bill_of_materials.get("currency", "RUB"),
            "total_material_price": bill_of_materials.get("total_material_price"),
            "total_estimated_price": bill_of_materials.get("total_estimated_price"),
        },
        "warnings": (
            list(layout_plan.get("warnings") or [])
            + list(selected_materials.get("warnings") or [])
            + list((appliance_assets or {}).get("warnings") or [])
        ),
    }
