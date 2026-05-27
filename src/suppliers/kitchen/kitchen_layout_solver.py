from __future__ import annotations

from typing import Any

from .kitchen_constants import DISHWASHER_WIDTHS_MM, KITCHEN_DIMENSIONS_MM, STORAGE_FILL_WIDTHS_MM


STRAIGHT_LAYOUT_ALIASES = {
    "straight",
    "wall",
    "single_wall",
    "linear",
    "one_wall",
    "auto",
    "random",
    "однорядная",
    "линейная",
}

MIN_WIDTH_MM = 1500
MIN_SINK_COOKTOP_GAP_MM = 400
PREFERRED_SINK_COOKTOP_GAP_MM = 600


def _dim(key: str, default: int) -> int:
    try:
        return int(KITCHEN_DIMENSIONS_MM.get(key, default))
    except Exception:
        return default


def _as_int(value: Any, default: int) -> int:
    try:
        return default if value is None else int(round(float(value)))
    except Exception:
        return default


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "да", "истина"}
    return bool(value)


def _normalize_required(required_appliances: dict[str, Any] | None) -> dict[str, Any]:
    required = dict(required_appliances or {})
    required["sink"] = True
    required["faucet"] = True
    required["microwave"] = True
    required.setdefault("cooktop", True)
    required.setdefault("oven", bool(required.get("cooktop", True)))
    required.setdefault("hood", bool(required.get("cooktop", True)))
    required.setdefault("fridge", False)
    required.setdefault("dishwasher", False)
    required.setdefault("washing_machine", False)
    return required


def _module(
    module_type: str,
    width_mm: int,
    *,
    x_mm: int = 0,
    depth_mm: int | None = None,
    height_mm: int | None = None,
    appliance: str | None = None,
    facade_layout: str | None = None,
    cutouts: list[str] | None = None,
    has_countertop: bool = True,
    has_upper_cabinet: bool = True,
    has_facade: bool = True,
    role: str | None = None,
) -> dict[str, Any]:
    return {
        "type": module_type,
        "x_mm": x_mm,
        "y_mm": 0,
        "z_mm": _dim("plinth_height", 100),
        "width_mm": int(width_mm),
        "depth_mm": depth_mm if depth_mm is not None else _dim("base_depth", 560),
        "height_mm": height_mm if height_mm is not None else _dim("base_body_height", 720),
        "orientation": "x",
        "appliance": appliance,
        "facade_layout": facade_layout,
        "cutouts": cutouts or [],
        "has_countertop": has_countertop,
        "has_upper_cabinet": has_upper_cabinet,
        "has_facade": has_facade,
        "role": role or module_type,
    }


def _storage_module(width_mm: int, *, x_mm: int = 0, role: str = "storage") -> dict[str, Any]:
    facade_layout = "three_drawers" if width_mm >= 500 else "one_door"
    return _module("drawer_stack", width_mm, x_mm=x_mm, facade_layout=facade_layout, role=role)


def _base_cabinet(width_mm: int, *, role: str = "base_cabinet") -> dict[str, Any]:
    return _module("base_cabinet", width_mm, facade_layout="two_doors", role=role)


def _split_storage_width(width_mm: int) -> list[int]:
    remaining = max(0, int(width_mm))
    widths: list[int] = []
    for candidate in STORAGE_FILL_WIDTHS_MM:
        while remaining >= candidate:
            widths.append(candidate)
            remaining -= candidate
    if remaining >= 240:
        widths.append(remaining)
    elif remaining and widths:
        widths[-1] += remaining
    elif remaining:
        widths.append(remaining)
    return widths


def _sum_width(modules: list[dict[str, Any]]) -> int:
    return sum(int(module.get("width_mm") or 0) for module in modules)


def _constraint_center(constraints: dict[str, Any], role: str) -> int | None:
    data = constraints.get(role)
    if not isinstance(data, dict):
        return None
    if data.get("x_mm") is not None:
        return _as_int(data.get("x_mm"), 0)
    if data.get("center_x_mm") is not None:
        return _as_int(data.get("center_x_mm"), 0)
    if data.get("min_x_mm") is not None and data.get("max_x_mm") is not None:
        return (_as_int(data.get("min_x_mm"), 0) + _as_int(data.get("max_x_mm"), 0)) // 2
    return None


def _constraint_is_hard(constraints: dict[str, Any], role: str) -> bool:
    data = constraints.get(role)
    return isinstance(data, dict) and _as_bool(data.get("hard"), False)


def _place_modules_sequentially(modules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    x = 0
    result: list[dict[str, Any]] = []
    for idx, module in enumerate(modules, start=1):
        copied = dict(module)
        copied["id"] = f"base_{idx:03d}"
        copied["x_mm"] = x
        x += int(copied["width_mm"])
        result.append(copied)
    return result


def _fill_gaps_with_storage(intervals: list[dict[str, Any]], total_width_mm: int) -> list[dict[str, Any]]:
    ordered = sorted(intervals, key=lambda module: int(module.get("x_mm") or 0))
    result: list[dict[str, Any]] = []
    cursor = 0
    storage_index = 1

    for module in ordered:
        start = int(module.get("x_mm") or 0)
        width = int(module.get("width_mm") or 0)
        if start > cursor:
            for storage_width in _split_storage_width(start - cursor):
                result.append(_storage_module(storage_width, x_mm=cursor, role=f"storage_{storage_index}"))
                cursor += storage_width
                storage_index += 1
        result.append(module)
        cursor = max(cursor, start + width)

    if cursor < total_width_mm:
        for storage_width in _split_storage_width(total_width_mm - cursor):
            result.append(_storage_module(storage_width, x_mm=cursor, role=f"storage_{storage_index}"))
            cursor += storage_width
            storage_index += 1

    result = sorted(result, key=lambda module: int(module.get("x_mm") or 0))
    for idx, module in enumerate(result, start=1):
        module["id"] = f"base_{idx:03d}"
    return result


def _overlaps(left: dict[str, Any], right: dict[str, Any], gap_mm: int = 0) -> bool:
    lx = int(left.get("x_mm") or 0) - gap_mm
    rx = int(right.get("x_mm") or 0) - gap_mm
    lw = int(left.get("width_mm") or 0) + gap_mm * 2
    rw = int(right.get("width_mm") or 0) + gap_mm * 2
    return lx < rx + rw and lx + lw > rx


def _gap_between(left: dict[str, Any], right: dict[str, Any]) -> int:
    left_x = int(left.get("x_mm") or 0)
    right_x = int(right.get("x_mm") or 0)
    left_end = left_x + int(left.get("width_mm") or 0)
    right_end = right_x + int(right.get("width_mm") or 0)
    if left_end <= right_x:
        return right_x - left_end
    if right_end <= left_x:
        return left_x - right_end
    return -1


def _build_constrained_modules(
    available_width_mm: int,
    required: dict[str, Any],
    constraints: dict[str, Any],
    warnings: list[str],
) -> list[dict[str, Any]]:
    sink_center = _constraint_center(constraints, "sink")
    cooktop_center = _constraint_center(constraints, "cooktop") or _constraint_center(constraints, "gas")

    modules: list[dict[str, Any]] = []
    if sink_center is not None:
        sink_x = sink_center - 300
        if sink_x < 0 or sink_x + 600 > available_width_mm:
            if _constraint_is_hard(constraints, "sink"):
                raise ValueError(f"sink_constraint_unmet:{sink_center}mm")
            warnings.append(f"sink_constraint_ignored:{sink_center}mm")
        else:
            modules.append(_module("sink_cabinet", 600, x_mm=sink_x, facade_layout="two_doors", cutouts=["sink"], role="sink"))
            warnings.append(f"sink_constraint_satisfied:{sink_center}mm")

    if required.get("cooktop") and cooktop_center is not None:
        cooktop_x = cooktop_center - 300
        if cooktop_x < 0 or cooktop_x + 600 > available_width_mm:
            if _constraint_is_hard(constraints, "cooktop") or _constraint_is_hard(constraints, "gas"):
                raise ValueError(f"cooktop_constraint_unmet:{cooktop_center}mm")
            warnings.append(f"cooktop_constraint_ignored:{cooktop_center}mm")
        else:
            modules.append(
                _module(
                    "oven_cabinet",
                    600,
                    x_mm=cooktop_x,
                    appliance="oven" if required.get("oven", True) else None,
                    facade_layout="oven_front",
                    cutouts=["cooktop"],
                    role="cooking",
                )
            )
            warnings.append(f"cooktop_constraint_satisfied:{cooktop_center}mm")

    if not any(module["type"] == "sink_cabinet" for module in modules):
        modules.append(_module("sink_cabinet", 600, x_mm=max(0, min(available_width_mm - 600, 800)), facade_layout="two_doors", cutouts=["sink"], role="sink"))

    if required.get("cooktop") and not any("cooktop" in module.get("cutouts", []) for module in modules):
        sink = next(module for module in modules if module["type"] == "sink_cabinet")
        cooktop_x = max(0, min(available_width_mm - 600, int(sink["x_mm"]) + 600 + MIN_SINK_COOKTOP_GAP_MM))
        modules.append(_module("oven_cabinet", 600, x_mm=cooktop_x, appliance="oven" if required.get("oven", True) else None, facade_layout="oven_front", cutouts=["cooktop"], role="cooking"))

    for left_idx, left in enumerate(modules):
        for right in modules[left_idx + 1 :]:
            if {"sink", "cooking"} == {left.get("role"), right.get("role")}:
                if _gap_between(left, right) < MIN_SINK_COOKTOP_GAP_MM:
                    if _constraint_is_hard(constraints, "sink"):
                        raise ValueError(f"sink_constraint_unmet:{sink_center}mm")  # pragma: no cover
                    if _constraint_is_hard(constraints, "cooktop") or _constraint_is_hard(constraints, "gas"):
                        raise ValueError(f"cooktop_constraint_unmet:{cooktop_center}mm")  # pragma: no cover
                    raise ValueError("hard_functional_zone_overlap")
                continue
            if _overlaps(left, right):  # pragma: no cover
                raise ValueError("hard_functional_zone_overlap")  # pragma: no cover

    modules = sorted(modules, key=lambda module: int(module.get("x_mm") or 0))

    if required.get("fridge"):
        fridge = _module(
            "fridge_slot",
            _dim("fridge_width", 600),
            x_mm=0,
            depth_mm=_dim("fridge_depth", 650),
            height_mm=_dim("fridge_height", 1900),
            appliance="fridge",
            has_countertop=False,
            has_upper_cabinet=False,
            has_facade=False,
            role="fridge",
        )
        if any(_overlaps(fridge, module) for module in modules):
            fridge["x_mm"] = available_width_mm - int(fridge["width_mm"])
        if any(_overlaps(fridge, module) for module in modules):
            warnings.append("removed_due_to_insufficient_width:fridge_slot")
        else:
            modules.append(fridge)

    return _fill_gaps_with_storage(modules, available_width_mm)


def _build_default_modules(available_width_mm: int, required: dict[str, Any], warnings: list[str]) -> list[dict[str, Any]]:
    fridge_width = _dim("fridge_width", 600)
    dishwasher_width = DISHWASHER_WIDTHS_MM[0] if required.get("dishwasher") else 0
    cooktop_width = 600 if required.get("cooktop") else 0
    min_core = 600 + cooktop_width + dishwasher_width
    include_fridge = bool(required.get("fridge")) and available_width_mm >= min_core + fridge_width
    include_dishwasher = bool(required.get("dishwasher")) and available_width_mm >= min_core + (fridge_width if include_fridge else 0)

    if required.get("fridge") and not include_fridge:
        warnings.append("removed_due_to_insufficient_width:fridge_slot")
    if required.get("dishwasher") and not include_dishwasher:
        warnings.append("removed_due_to_insufficient_width:dishwasher_slot")

    modules: list[dict[str, Any]] = []
    if include_fridge:
        modules.append(
            _module(
                "fridge_slot",
                fridge_width,
                depth_mm=_dim("fridge_depth", 650),
                height_mm=_dim("fridge_height", 1900),
                appliance="fridge",
                has_countertop=False,
                has_upper_cabinet=False,
                has_facade=False,
                role="fridge",
            )
        )

    working_width = available_width_mm - _sum_width(modules)
    cooktop_enabled = bool(required.get("cooktop")) and working_width >= 1800
    if required.get("cooktop") and not cooktop_enabled:
        warnings.append("removed_due_to_insufficient_width:cooktop")

    if working_width >= 2400:
        modules.append(_base_cabinet(600))

    modules.append(_module("sink_cabinet", 600, facade_layout="two_doors", cutouts=["sink"], role="sink"))

    if include_dishwasher:
        modules.append(
            _module(
                "dishwasher_slot",
                dishwasher_width,
                depth_mm=600,
                height_mm=850,
                appliance="dishwasher",
                has_facade=False,
                role="water_appliance",
            )
        )

    if cooktop_enabled:
        if available_width_mm - _sum_width(modules) >= 1200:
            modules.append(_storage_module(600, role="drawer_stack"))
        modules.append(
            _module(
                "oven_cabinet",
                600,
                appliance="oven" if required.get("oven", True) else None,
                facade_layout="oven_front",
                cutouts=["cooktop"],
                role="cooking",
            )
        )

    residual = available_width_mm - _sum_width(modules)
    if residual > 0:
        if residual < 300:
            modules.append(_module("filler", residual, facade_layout="flat_panel", has_upper_cabinet=False, role="filler"))
        else:
            for width in _split_storage_width(residual):
                modules.append(_storage_module(width, role="storage"))

    return _place_modules_sequentially(modules)


def _countertop_cutout_for_module(module: dict[str, Any], segment_x_mm: int, cutout_type: str) -> dict[str, Any]:
    local_module_x = int(module["x_mm"]) - segment_x_mm
    if cutout_type == "sink":
        width = min(500, int(module["width_mm"]) - 80)
        depth = 400
        return {
            "type": "sink",
            "module_id": module["id"],
            "shape": "rounded_rect",
            "x_mm": local_module_x + max(60, (int(module["width_mm"]) - width) // 2),
            "y_mm": 160,
            "width_mm": width,
            "depth_mm": depth,
        }

    width = min(560, int(module["width_mm"]) - 40)
    depth = 490
    return {
        "type": "cooktop",
        "module_id": module["id"],
        "shape": "rect",
        "x_mm": local_module_x + max(20, (int(module["width_mm"]) - width) // 2),
        "y_mm": 55,
        "width_mm": width,
        "depth_mm": depth,
    }


def _build_countertop_segments(base_modules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []

    def flush() -> None:
        if not current:
            return
        x0 = int(current[0]["x_mm"])
        width = sum(int(module["width_mm"]) for module in current)
        cutouts: list[dict[str, Any]] = []
        for module in current:
            if "sink" in module.get("cutouts", []):
                cutouts.append(_countertop_cutout_for_module(module, x0, "sink"))
            if "cooktop" in module.get("cutouts", []):
                cutouts.append(_countertop_cutout_for_module(module, x0, "cooktop"))
        idx = len(segments) + 1
        segments.append(
            {
                "id": f"countertop_{idx:03d}",
                "x_mm": x0,
                "y_mm": 0,
                "z_mm": _dim("plinth_height", 100) + _dim("base_body_height", 720),
                "width_mm": width,
                "depth_mm": _dim("countertop_depth", 600),
                "thickness_mm": _dim("countertop_thickness", 38),
                "orientation": "x",
                "cutouts": cutouts,
            }
        )
        current.clear()

    for module in base_modules:
        if module.get("has_countertop", True):
            current.append(module)
        else:
            flush()
    flush()
    return segments


def _build_backsplash_segments(countertop_segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": f"backsplash_{idx:03d}",
            "x_mm": segment["x_mm"],
            "y_mm": 0,
            "z_mm": int(segment["z_mm"]) + int(segment["thickness_mm"]),
            "width_mm": segment["width_mm"],
            "height_mm": 600,
            "thickness_mm": _dim("backsplash_thickness", 4),
            "orientation": "x",
        }
        for idx, segment in enumerate(countertop_segments, start=1)
    ]


def _build_upper_modules(base_modules: list[dict[str, Any]], required: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    upper: list[dict[str, Any]] = []
    decor: list[dict[str, Any]] = []
    upper_z = _dim("plinth_height", 100) + _dim("base_body_height", 720) + _dim("countertop_thickness", 38) + 600
    cooktop_index = next((idx for idx, module in enumerate(base_modules) if "cooktop" in module.get("cutouts", [])), None)

    for base_index, base in enumerate(base_modules):
        if not base.get("has_upper_cabinet", True):
            continue
        if cooktop_index is not None and base_index == cooktop_index - 1:
            continue

        module_type = "wall_cabinet"
        height = _dim("upper_height", 720)
        depth = _dim("upper_depth", 320)

        if base["type"] == "sink_cabinet":
            module_type = "dish_dryer_cabinet"
        elif "cooktop" in base.get("cutouts", []):
            module_type = "hood_wall_mounted" if required.get("hood", True) else "open_hood_space"
            height = 360
        upper.append(
            {
                "id": f"upper_{len(upper) + 1:03d}",
                "type": module_type,
                "x_mm": base["x_mm"],
                "y_mm": 0,
                "z_mm": upper_z,
                "width_mm": base["width_mm"],
                "depth_mm": depth,
                "height_mm": height,
                "orientation": "x",
                "above_base_module_id": base["id"],
                "above": base.get("role") or base["type"],
            }
        )

    microwave_host = next((module for module in upper if module["type"] == "wall_cabinet" and int(module["width_mm"]) >= 500), None)
    if microwave_host:
        microwave_host["type"] = "microwave_open_shelf"
        microwave_host["height_mm"] = 360
        microwave_host["has_facade"] = False
        decor.append(
            {
                "id": "decor_microwave_001",
                "type": "microwave",
                "x_mm": int(microwave_host["x_mm"]) + int(microwave_host["width_mm"]) // 2,
                "y_mm": 160,
                "z_mm": int(microwave_host["z_mm"]) + 32,
                "orientation": "x",
                "upper_module_id": microwave_host["id"],
                "placement": "upper_open_shelf",
            }
        )
    else:
        segment_base = next(
            (
                base
                for base in base_modules
                if base.get("has_countertop", True)
                and not base.get("cutouts")
                and base.get("type") not in {"sink_cabinet", "oven_cabinet"}
            ),
            None,
        )
        if segment_base is None:
            segment_base = next(  # pragma: no cover
                (
                    base
                    for base in base_modules
                    if base.get("has_countertop", True) and not base.get("cutouts")
                ),
                None,
            )
        if segment_base:
            decor.append(
                {
                    "id": "decor_microwave_001",
                    "type": "microwave",
                    "placement": "countertop",
                    "x_mm": int(segment_base["x_mm"]) + int(segment_base["width_mm"]) // 2,
                    "y_mm": 390,
                    "z_mm": _dim("plinth_height", 100) + _dim("base_body_height", 720) + _dim("countertop_thickness", 38) + 2,
                    "orientation": "x",
                    "support_module_id": segment_base.get("id"),
                }
            )
            warnings.append("microwave_placement:countertop")

    return upper, decor, warnings


def _blocked_countertop_intervals(base_modules: list[dict[str, Any]], margin_mm: int = 120) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    for module in base_modules:
        x = int(module.get("x_mm") or 0)
        width = int(module.get("width_mm") or 0)
        if not module.get("has_countertop", True):
            continue
        if module.get("cutouts") or module.get("type") in {"sink_cabinet", "oven_cabinet", "dishwasher_slot"}:
            intervals.append((max(0, x - margin_mm), x + width + margin_mm))
    return intervals


def _find_countertop_spot(
    base_modules: list[dict[str, Any]],
    *,
    min_width_mm: int,
    preferred_y_mm: int,
    used_spots: list[tuple[int, int]],
) -> tuple[int, int, str | None] | None:
    blocked = [*_blocked_countertop_intervals(base_modules), *used_spots]
    candidates: list[tuple[int, int, str]] = []
    for module in base_modules:
        if not module.get("has_countertop", True):
            continue
        if module.get("type") in {"sink_cabinet", "oven_cabinet"}:
            continue
        start = int(module.get("x_mm") or 0)
        end = start + int(module.get("width_mm") or 0)
        free_start, free_end = start, end
        for left, right in blocked:
            if right <= free_start or left >= free_end:
                continue
            if left <= free_start < right:
                free_start = min(free_end, right)
            elif free_start < left < free_end:
                free_end = max(free_start, left)
        if free_end - free_start >= min_width_mm:
            candidates.append((free_start, free_end, str(module.get("id") or "")))
    if not candidates:
        return None
    start, end, module_id = max(candidates, key=lambda item: item[1] - item[0])
    x = (start + end) // 2
    used_spots.append((x - min_width_mm // 2 - 80, x + min_width_mm // 2 + 80))
    return x, preferred_y_mm, module_id or None


def _build_countertop_accessories(base_modules: list[dict[str, Any]], required: dict[str, Any]) -> list[dict[str, Any]]:
    if not _as_bool(required.get("decor_accessories"), True):
        return []

    top_z = _dim("plinth_height", 100) + _dim("base_body_height", 720) + _dim("countertop_thickness", 38) + 2
    used_spots: list[tuple[int, int]] = []
    items: list[dict[str, Any]] = []
    specs = [
        ("decor_flowers_vase_001", "flowers_vase", 260, 440, 4500),
        ("decor_oil_bottles_001", "oil_bottles_decor", 220, 130, 2200),
        ("decor_kitchen_set_001", "decorative_kitchen_set", 300, 360, 2800),
    ]

    for item_id, item_type, min_width, y_mm, estimate in specs:
        spot = _find_countertop_spot(base_modules, min_width_mm=min_width, preferred_y_mm=y_mm, used_spots=used_spots)
        if spot is None:
            continue
        x_mm, y_mm, module_id = spot
        items.append(
            {
                "id": item_id,
                "type": item_type,
                "placement": "countertop",
                "x_mm": x_mm,
                "y_mm": y_mm,
                "z_mm": top_z,
                "orientation": "x",
                "support_module_id": module_id,
                "estimated_price": estimate,
                "price_currency": "RUB",
            }
        )
    return items


def _build_functional_zones(base_modules: list[dict[str, Any]], upper_modules: list[dict[str, Any]], decor_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    zones: list[dict[str, Any]] = []
    for module in base_modules:
        role = module.get("role")
        if role in {"sink", "cooking", "fridge", "water_appliance"}:
            zones.append(
                {
                    "role": role,
                    "module_id": module["id"],
                    "type": module["type"],
                    "x_mm": module["x_mm"],
                    "width_mm": module["width_mm"],
                    "appliance": module.get("appliance"),
                }
            )
    for module in upper_modules:
        if module["type"] in {"hood_wall_mounted", "hood_compact_wall", "microwave_open_shelf"}:
            zones.append(
                {
                    "role": "hood" if module["type"].startswith("hood") else "microwave",
                    "module_id": module["id"],
                    "type": module["type"],
                    "x_mm": module["x_mm"],
                    "width_mm": module["width_mm"],
                }
            )
    for item in decor_items:
        if item["type"] == "microwave":
            zones.append({"role": "microwave", "item_id": item["id"], "placement": item.get("placement")})
        elif item["type"] in {"flowers_vase", "oil_bottles_decor", "decorative_kitchen_set"}:
            zones.append({"role": item["type"], "item_id": item["id"], "placement": item.get("placement")})
    return zones


def _build_openings(
    base_modules: list[dict[str, Any]],
    upper_modules: list[dict[str, Any]],
    countertop_segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    openings: list[dict[str, Any]] = []
    for segment in countertop_segments:
        for cutout in segment.get("cutouts", []):
            openings.append(
                {
                    "type": f"{cutout['type']}_cutout",
                    "countertop_segment_id": segment["id"],
                    "module_id": cutout.get("module_id"),
                    "x_mm": int(segment["x_mm"]) + int(cutout["x_mm"]),
                    "y_mm": cutout["y_mm"],
                    "width_mm": cutout["width_mm"],
                    "depth_mm": cutout["depth_mm"],
                    "z_mm": int(segment["z_mm"]) + int(segment["thickness_mm"]),
                }
            )
    for module in base_modules:
        if module["type"] == "fridge_slot":
            openings.append({"type": "fridge_space", "module_id": module["id"], "x_mm": module["x_mm"], "width_mm": module["width_mm"], "depth_mm": module["depth_mm"], "height_mm": module["height_mm"]})
    for module in upper_modules:
        if module["type"] == "microwave_open_shelf":
            openings.append({"type": "microwave_niche", "module_id": module["id"], "x_mm": module["x_mm"], "width_mm": module["width_mm"], "depth_mm": module["depth_mm"], "height_mm": module["height_mm"]})
        if module["type"].startswith("hood"):
            openings.append({"type": "hood_space", "module_id": module["id"], "x_mm": module["x_mm"], "width_mm": module["width_mm"], "height_mm": module["height_mm"]})
    return openings


def _build_cabinet_breakdown(base_modules: list[dict[str, Any]], upper_modules: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    facades: list[dict[str, Any]] = []
    for module in base_modules:
        if module.get("has_facade", True):
            facades.append({"module_id": module["id"], "tier": "base", "layout": module.get("facade_layout") or "flat_panel", "width_mm": module["width_mm"], "height_mm": module["height_mm"]})
    for module in upper_modules:
        if module["type"] not in {"hood_wall_mounted", "microwave_open_shelf"}:
            facades.append({"module_id": module["id"], "tier": "upper", "layout": "one_lift_or_two_doors", "width_mm": module["width_mm"], "height_mm": module["height_mm"]})
    return {"facades": facades}


def _build_asset_targets(functional_zones: list[dict[str, Any]]) -> list[dict[str, Any]]:
    targets = []
    for role in ("fridge", "sink", "cooktop", "hood", "microwave", "water_appliance"):
        if any(zone["role"] == role or (role == "cooktop" and zone["role"] == "cooking") for zone in functional_zones):
            targets.append({"role": role, "prefer_fbx": True})
    for role in ("flowers_vase", "oil_bottles_decor", "decorative_kitchen_set"):
        if any(zone["role"] == role for zone in functional_zones):
            targets.append({"role": role, "prefer_fbx": True, "optional": True})
    targets.extend(
        [
            {"role": "body", "prefer_material": "board_sheet"},
            {"role": "facade", "prefer_material": "facade_sheet"},
            {"role": "countertop", "prefer_material": "countertop_slab"},
            {"role": "backsplash", "prefer_material": "backsplash_panel"},
        ]
    )
    return targets


def solve_kitchen_layout(
    kitchen_zone: dict[str, Any],
    plumbing_point: dict[str, Any] | None,
    entry_zone: dict[str, Any] | None,
    required_appliances: dict[str, Any] | None,
    design_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del design_spec
    warnings: list[str] = []
    requested_layout = str((kitchen_zone or {}).get("layout_type") or "straight").strip().lower()
    if requested_layout not in STRAIGHT_LAYOUT_ALIASES:
        warnings.append(f"layout_forced_to_straight:{requested_layout}")

    available_width_mm = max(MIN_WIDTH_MM, _as_int((kitchen_zone or {}).get("available_width_mm"), 3000))
    constraints = (kitchen_zone or {}).get("constraints") if isinstance((kitchen_zone or {}).get("constraints"), dict) else {}
    del plumbing_point
    required = _normalize_required(required_appliances)

    if constraints:
        base_modules = _build_constrained_modules(available_width_mm, required, constraints, warnings)
    else:
        base_modules = _build_default_modules(available_width_mm, required, warnings)

    countertop_segments = _build_countertop_segments(base_modules)
    backsplash_segments = _build_backsplash_segments(countertop_segments)
    upper_modules, decor_items, upper_warnings = _build_upper_modules(base_modules, required)
    decor_items.extend(_build_countertop_accessories(base_modules, required))
    warnings.extend(upper_warnings)

    has_cooktop = any("cooktop" in module.get("cutouts", []) for module in base_modules)
    if not has_cooktop:
        warnings.append("cooking_mode:microwave_only")
    elif required.get("hood"):
        warnings.append("hood_style:wall_mounted")

    result = {
        "layout_type": "straight",
        "layout_variant": {
            "name": "linear_open_hood",
            "upper_strategy": "full_storage_with_open_hood_gap",
            "hood_style": "wall_mounted",
            "reserve_open_hood_gap": True,
        },
        "wall_id": (kitchen_zone or {}).get("wall_id"),
        "total_width_mm": available_width_mm,
        "used_width_mm": _sum_width(base_modules),
        "coordinate_system": {"x_axis": "along_wall", "y_axis": "from_wall", "z_axis": "up"},
        "dimensions": {
            "base_depth_mm": _dim("base_depth", 560),
            "countertop_depth_mm": _dim("countertop_depth", 600),
            "countertop_top_z_mm": _dim("plinth_height", 100) + _dim("base_body_height", 720) + _dim("countertop_thickness", 38),
            "backsplash_height_mm": 600,
        },
        "base_modules": base_modules,
        "countertop_segments": countertop_segments,
        "countertop": countertop_segments[0] if countertop_segments else None,
        "backsplash_segments": backsplash_segments,
        "backsplash": backsplash_segments[0] if backsplash_segments else None,
        "upper_modules": upper_modules,
        "decor_items": decor_items,
        "warnings": [f"layout_variant:linear_open_hood", *warnings],
    }
    return result
