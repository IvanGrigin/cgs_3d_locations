from __future__ import annotations

import math
from typing import Any

from .kitchen_constants import ROLE_PRICE_DEFAULTS_RUB


def _price(material: dict[str, Any] | None, role: str) -> float:
    if material:
        value = material.get("price")
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
        return ROLE_PRICE_DEFAULTS_RUB.get(material.get("kitchen_role") or role, ROLE_PRICE_DEFAULTS_RUB.get(role, 1000.0))
    return ROLE_PRICE_DEFAULTS_RUB.get(role, 1000.0)


def _material_binding(selected_materials: dict[str, Any], target_role: str) -> dict[str, Any] | None:
    entry = (selected_materials.get("materials") or {}).get(target_role)
    return entry.get("chosen_material") if entry else None


def _dim(material: dict[str, Any] | None, key: str, default: int) -> int:
    value = ((material or {}).get("dimensions") or {}).get(key)
    return int(value) if isinstance(value, (int, float)) and value > 0 else default


def _ceil_div(a: float, b: float) -> int:
    return int(math.ceil(a / max(1.0, b)))


def _facade_panels(layout_plan: dict[str, Any]) -> list[dict[str, int]]:
    panels: list[dict[str, int]] = []
    for module in layout_plan.get("base_modules") or []:
        if not module.get("has_facade", True):
            continue
        width = int(module.get("width_mm") or 0)
        height = int(module.get("height_mm") or 720)
        facade_layout = module.get("facade_layout")
        if facade_layout == "three_drawers":
            panels.extend({"width_mm": width, "height_mm": max(1, height // 3)} for _ in range(3))
        elif facade_layout == "two_doors":
            panels.extend([{"width_mm": max(1, width // 2), "height_mm": height}, {"width_mm": max(1, width // 2), "height_mm": height}])
        elif facade_layout == "oven_front":
            panels.append({"width_mm": width, "height_mm": 120})
        else:
            panels.append({"width_mm": width, "height_mm": height})
    for module in layout_plan.get("upper_modules") or []:
        width = int(module.get("width_mm") or 0)
        height = int(module.get("height_mm") or 720)
        if module.get("type") == "hood_cabinet":
            panels.append({"width_mm": width, "height_mm": min(height, 360)})
        elif width >= 700:
            panels.extend([{"width_mm": width // 2, "height_mm": height}, {"width_mm": width // 2, "height_mm": height}])
        else:
            panels.append({"width_mm": width, "height_mm": height})
    return panels


def _area_m2(panels: list[dict[str, int]]) -> float:
    return sum((p["width_mm"] / 1000.0) * (p["height_mm"] / 1000.0) for p in panels)


def _edge_length_m(panels: list[dict[str, int]]) -> float:
    return sum(2.0 * (p["width_mm"] / 1000.0 + p["height_mm"] / 1000.0) for p in panels)


def _body_area_m2(layout_plan: dict[str, Any]) -> float:
    base_count = len([m for m in layout_plan.get("base_modules") or [] if m.get("type") != "fridge_slot"])
    upper_count = len(layout_plan.get("upper_modules") or [])
    wide_bonus = sum(max(0, int(m.get("width_mm") or 600) - 600) / 600.0 for m in layout_plan.get("base_modules") or [])
    return round(base_count * 1.30 + upper_count * 1.00 + wide_bonus * 0.35, 3)


def _countertop_width_mm(layout_plan: dict[str, Any]) -> int:
    return sum(int(seg.get("width_mm") or 0) for seg in layout_plan.get("countertop_segments") or [])


def _backsplash_width_mm(layout_plan: dict[str, Any]) -> int:
    return sum(int(seg.get("width_mm") or 0) for seg in layout_plan.get("backsplash_segments") or [])


def _add_item(
    items: list[dict[str, Any]],
    role: str,
    material: dict[str, Any] | None,
    quantity: float,
    unit: str,
    unit_price: float | None = None,
    note: str | None = None,
) -> None:
    if quantity <= 0:
        return
    price = unit_price if unit_price is not None else _price(material, role)
    items.append(
        {
            "role": role,
            "sku": (material or {}).get("sku"),
            "name": (material or {}).get("name") or f"estimated_{role}",
            "kitchen_role": (material or {}).get("kitchen_role") or role,
            "unit": unit,
            "quantity": round(float(quantity), 3),
            "unit_price": round(float(price), 2),
            "total_price": round(float(price) * float(quantity), 2),
            "note": note,
        }
    )


def _asset_price(asset: dict[str, Any] | None, default: float) -> float:
    value = (asset or {}).get("price")
    if isinstance(value, (int, float)) and value > 100:
        return float(value)
    try:
        parsed = float(str(value).replace(" ", "").replace(",", ".")) if value is not None else 0.0
        if parsed > 100:
            return parsed
    except Exception:
        pass
    return default


def _add_decor_items(
    items: list[dict[str, Any]],
    layout_plan: dict[str, Any],
    appliance_assets: dict[str, Any] | None,
) -> float:
    role_defaults = {
        "flowers_vase": 4500.0,
        "oil_bottles_decor": 2200.0,
        "decorative_kitchen_set": 2800.0,
        "small_kitchen_appliance": 6500.0,
    }
    bindings = ((appliance_assets or {}).get("appliances") or {})
    total = 0.0
    for decor in layout_plan.get("decor_items") or []:
        role = str(decor.get("type") or "")
        if role not in role_defaults:
            continue
        chosen = ((bindings.get(role) or {}).get("chosen_asset") or {})
        default_price = float(decor.get("estimated_price") or role_defaults[role])
        unit_price = _asset_price(chosen, default_price)
        name = chosen.get("title") or role
        items.append(
            {
                "role": role,
                "sku": chosen.get("unique_key"),
                "name": name,
                "kitchen_role": "kitchen_accessory",
                "unit": "piece",
                "quantity": 1.0,
                "unit_price": round(unit_price, 2),
                "total_price": round(unit_price, 2),
                "note": f"decor_item_id={decor.get('id')}, placement={decor.get('placement')}",
            }
        )
        total += unit_price
    return total


def estimate_kitchen_bom(
    layout_plan: dict[str, Any],
    selected_materials: dict[str, Any],
    mode: str = "optimal",
    include_appliance_estimate: bool = False,
    appliance_assets: dict[str, Any] | None = None,
) -> dict[str, Any]:
    facade = _material_binding(selected_materials, "facade")
    body = _material_binding(selected_materials, "body")
    countertop = _material_binding(selected_materials, "countertop")
    backsplash = _material_binding(selected_materials, "backsplash")
    edge_band = _material_binding(selected_materials, "edge_band")
    wall_plinth = _material_binding(selected_materials, "wall_plinth")
    joint_profile = _material_binding(selected_materials, "joint_profile")
    end_profile = _material_binding(selected_materials, "end_profile")

    facade_panels = _facade_panels(layout_plan)
    facade_area = _area_m2(facade_panels)
    edge_length = _edge_length_m(facade_panels)
    body_area = _body_area_m2(layout_plan)
    countertop_width = _countertop_width_mm(layout_plan)
    backsplash_width = _backsplash_width_mm(layout_plan)
    facade_sheet_area = (_dim(facade, "length_mm", 2800) / 1000.0) * (_dim(facade, "width_mm", 1220) / 1000.0) * 0.85
    body_sheet_area = (_dim(body, "length_mm", 2800) / 1000.0) * (_dim(body, "width_mm", 2070) / 1000.0) * 0.85
    facade_sheet_count = _ceil_div(facade_area, facade_sheet_area)
    body_sheet_count = _ceil_div(body_area, body_sheet_area)
    countertop_piece_count = _ceil_div(countertop_width, _dim(countertop, "length_mm", 3000))
    backsplash_piece_count = _ceil_div(backsplash_width, _dim(backsplash, "length_mm", 3050))
    wall_plinth_count = _ceil_div(countertop_width, _dim(wall_plinth, "length_mm", 3000))

    items: list[dict[str, Any]] = []
    _add_item(items, "facade_sheet", facade, facade_sheet_count, "sheet", note=f"facade_area_m2={facade_area:.3f}")
    _add_item(items, "board_sheet", body, body_sheet_count, "sheet", note=f"body_area_m2≈{body_area:.3f}")
    _add_item(items, "countertop_slab", countertop, countertop_piece_count, "piece", note=f"countertop_width_mm={countertop_width}")
    _add_item(items, "backsplash_panel", backsplash, backsplash_piece_count, "piece", note=f"backsplash_width_mm={backsplash_width}, height=600mm")
    _add_item(items, "edge_band", edge_band, max(0.0, edge_length), "m", note="facade perimeter edge band")
    _add_item(items, "countertop_wall_plinth", wall_plinth, wall_plinth_count, "piece", note="between countertop and backsplash")
    if countertop_piece_count > 1:
        _add_item(items, "joint_profile", joint_profile, countertop_piece_count - 1, "piece", note="countertop joints")
    if countertop_width > 0:
        _add_item(items, "end_profile", end_profile, 2, "piece", note="visible countertop ends")
    decor_accessory_estimate = _add_decor_items(items, layout_plan, appliance_assets)

    base_count = len([m for m in layout_plan.get("base_modules") or [] if m.get("type") != "fridge_slot"])
    upper_count = len(layout_plan.get("upper_modules") or [])
    drawer_count = sum(1 for p in facade_panels if p["height_mm"] < 350)
    door_count = max(0, len(facade_panels) - drawer_count)
    hardware_estimate = base_count * 2200 + upper_count * 1800 + drawer_count * 1700 + door_count * 600
    sink_and_faucet_estimate = 9000 if any("sink" in m.get("cutouts", []) for m in layout_plan.get("base_modules") or []) else 0
    handwash_estimate = 6500 if any("entry_handwash" in m.get("cutouts", []) for m in layout_plan.get("base_modules") or []) else 0
    appliance_estimate = 0
    if include_appliance_estimate:
        appliance_prices = {"fridge": 45000, "washing_machine": 35000, "dishwasher": 38000, "oven": 30000}
        for module in layout_plan.get("base_modules") or []:
            appliance_estimate += appliance_prices.get(module.get("appliance"), 0)
        if any("cooktop" in m.get("cutouts", []) for m in layout_plan.get("base_modules") or []):
            appliance_estimate += 18000
    total_material_price = round(sum(item["total_price"] for item in items), 2)
    total_estimated_price = round(total_material_price + hardware_estimate + sink_and_faucet_estimate + handwash_estimate + appliance_estimate, 2)
    return {
        "mode": mode,
        "currency": "RUB",
        "items": items,
        "computed_quantities": {
            "facade_area_m2": round(facade_area, 3),
            "body_area_m2": round(body_area, 3),
            "edge_length_m": round(edge_length, 3),
            "countertop_width_mm": countertop_width,
            "backsplash_width_mm": backsplash_width,
            "facade_panel_count": len(facade_panels),
            "base_module_count": base_count,
            "upper_module_count": upper_count,
        },
        "estimates": {
            "hardware_estimate": round(hardware_estimate, 2),
            "sink_and_faucet_estimate": round(sink_and_faucet_estimate, 2),
            "entry_handwash_estimate": round(handwash_estimate, 2),
            "appliance_estimate": round(appliance_estimate, 2),
            "decor_accessory_estimate": round(decor_accessory_estimate, 2),
        },
        "total_material_price": total_material_price,
        "total_estimated_price": total_estimated_price,
    }
