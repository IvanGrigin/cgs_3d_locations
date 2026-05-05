from __future__ import annotations

from pathlib import Path
from typing import Any

from .kitchen_assembly_builder import build_kitchen_assembly_json
from .kitchen_appliance_matcher import select_kitchen_appliance_assets
from .kitchen_bom_estimator import estimate_kitchen_bom
from .kitchen_catalog_loader import load_kitchen_material_catalog
from .kitchen_constants import SELECTION_MODES
from .kitchen_design_spec import build_kitchen_design_spec
from .kitchen_layout_solver import solve_kitchen_layout
from .kitchen_material_matcher import select_kitchen_materials


def generate_kitchen_variants(
    material_catalog: str | Path | list[dict[str, Any]],
    user_prompt: str,
    room: dict[str, Any],
    kitchen_zone: dict[str, Any],
    required_appliances: dict[str, Any] | None = None,
    recommended_colors: dict[str, Any] | None = None,
    budget: dict[str, Any] | None = None,
    plumbing_point: dict[str, Any] | None = None,
    entry_zone: dict[str, Any] | None = None,
    appliance_catalog: str | Path | list[dict[str, Any]] | None = None,
    modes: list[str] | tuple[str, ...] = SELECTION_MODES,
    target_id: str = "kitchen_001",
    position: list[float] | None = None,
    rotation: list[float] | None = None,
) -> dict[str, dict[str, Any]]:
    materials = load_kitchen_material_catalog(material_catalog) if isinstance(material_catalog, (str, Path)) else material_catalog
    required_appliances = required_appliances or {"sink": True, "cooktop": True, "oven": True, "hood": True}
    design_spec = build_kitchen_design_spec(
        user_prompt=user_prompt,
        recommended_colors=recommended_colors or {},
        budget=budget or {},
        appliances=required_appliances,
        room_meta=room,
    )
    if entry_zone and entry_zone.get("has_entry_handwash"):
        design_spec["functional_requirements"]["entry_handwash"] = True
    layout_plan = solve_kitchen_layout(kitchen_zone, plumbing_point, entry_zone, required_appliances, design_spec)
    appliance_assets = (
        select_kitchen_appliance_assets(
            supplier_catalog=appliance_catalog,
            layout_plan=layout_plan,
            required_appliances=required_appliances,
            only_local_assets=True,
            top_n=5,
            user_prompt=user_prompt,
        )
        if appliance_catalog is not None
        else {"appliances": {}, "warnings": []}
    )
    variants: dict[str, dict[str, Any]] = {}
    for mode in modes:
        selected_materials = select_kitchen_materials(materials, design_spec, layout_plan, mode=mode, top_n=5)
        bom = estimate_kitchen_bom(layout_plan, selected_materials, mode=mode, include_appliance_estimate=False)
        variants[mode] = build_kitchen_assembly_json(
            target_id=f"{target_id}_{mode}",
            layout_plan=layout_plan,
            selected_materials=selected_materials,
            bill_of_materials=bom,
            design_spec=design_spec,
            mode=mode,
            appliance_assets=appliance_assets,
            position=position,
            rotation=rotation,
        )
    return variants


def is_kitchen_target(target: dict[str, Any]) -> bool:
    category = str(target.get("category") or target.get("type") or "").lower()
    return category in {"kitchen", "kitchen_set", "kitchen_cabinet_system", "kitchen_assembly"}


def build_kitchen_zone_from_target(target: dict[str, Any], room: dict[str, Any] | None = None) -> dict[str, Any]:
    dims = target.get("dimensions") or target.get("bbox_m") or target.get("size") or {}
    width_m = None
    if isinstance(dims, dict):
        width_m = dims.get("width_m") or dims.get("x") or dims.get("width")
    elif isinstance(dims, list) and dims:
        width_m = dims[0]
    try:
        width_mm = int(round(float(width_m) * 1000.0)) if width_m else 3000
    except Exception:
        width_mm = 3000
    return {
        "layout_type": target.get("layout_type") or "straight",
        "wall_id": target.get("wall_id") or (room or {}).get("default_kitchen_wall_id"),
        "available_width_mm": width_mm,
        "depth_mm": 600,
        "start_x_mm": 0,
        "end_x_mm": width_mm,
    }
