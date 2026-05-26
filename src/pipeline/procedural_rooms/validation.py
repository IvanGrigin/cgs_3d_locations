from __future__ import annotations

from collections import deque
from typing import Any

from .geometry import AABB, as_float, object_footprint_inside_polygon, point_in_polygon, polygon_bounds, Vec2
from .placement_engine import COLLISION_IGNORE_CATEGORIES
from .room_context import RoomContext


NON_SOLID_FALLBACK_CATEGORIES = {
    "rug",
    "runner_rug",
    "pillow",
    "blanket",
    "wall_art",
    "mirror",
    "wall_light",
    "wall_hooks",
    "curtain",
    "headboard",
    "ceiling_light",
    "tv",
    "decor_books",
    "decor_vase",
    "decor_box",
    "decor_tray",
    "table_lamp",
    "bath_mat",
    "towel_rack",
    "toilet_paper_holder",
    "hygiene_shower",
    "soap_dispenser",
    "toothbrush_cup",
    "shampoo_bottle",
    "air_freshener",
}


def _aabb_from_item(item: dict[str, Any]) -> AABB | None:
    aabb = item.get("aabb")
    if not isinstance(aabb, dict):
        return None
    return AABB(
        x_min=as_float(aabb.get("x_min")),
        x_max=as_float(aabb.get("x_max")),
        y_min=as_float(aabb.get("y_min")),
        y_max=as_float(aabb.get("y_max")),
        z_min=as_float(aabb.get("z_min")),
        z_max=as_float(aabb.get("z_max")),
    )


def is_solid_floor_obstacle(item: dict[str, Any]) -> bool:
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    role = str(meta.get("physical_role") or "").strip().lower()
    if role:
        return role == "solid_floor"

    category = str(item.get("category", "")).strip().lower()
    constraints = item.get("constraints") if isinstance(item.get("constraints"), dict) else {}
    mount_type = str(item.get("mount_type") or constraints.get("mount_type") or "").strip().lower()
    layer = str(meta.get("density_layer", "")).strip().lower()

    if mount_type in {"wall", "ceiling", "on_top"}:
        return False
    if meta.get("allow_collision"):
        return False
    if meta.get("support_relation") == "on_top":
        return False
    if category in NON_SOLID_FALLBACK_CATEGORIES:
        return False
    if category in COLLISION_IGNORE_CATEGORIES and layer in {"soft_decor", "decor", "wall_decor", "textile", "lighting", "electronics"}:
        return False
    if layer in {"soft_decor", "decor", "wall_decor", "textile", "lighting", "electronics"}:
        return False
    return True


def _is_soft_validation_item(item: dict[str, Any]) -> bool:
    return not is_solid_floor_obstacle(item)


def _aabb_contains_xy(aabb: AABB, x: float, y: float, margin: float = 0.0) -> bool:
    return aabb.x_min - margin <= x <= aabb.x_max + margin and aabb.y_min - margin <= y <= aabb.y_max + margin


def _gap_between(a: AABB, b: AABB) -> float:
    x_overlap = not (a.x_max <= b.x_min or b.x_max <= a.x_min)
    y_overlap = not (a.y_max <= b.y_min or b.y_max <= a.y_min)
    if y_overlap:
        return max(b.x_min - a.x_max, a.x_min - b.x_max, 0.0)
    if x_overlap:
        return max(b.y_min - a.y_max, a.y_min - b.y_max, 0.0)
    return max(
        min(abs(b.x_min - a.x_max), abs(a.x_min - b.x_max)),
        min(abs(b.y_min - a.y_max), abs(a.y_min - b.y_max)),
    )


def _collision_margin_for_pair(a: dict[str, Any], b: dict[str, Any]) -> float:
    meta_a = a.get("meta") if isinstance(a.get("meta"), dict) else {}
    meta_b = b.get("meta") if isinstance(b.get("meta"), dict) else {}
    if bool(meta_a.get("compact_bathroom_template")) and bool(meta_b.get("compact_bathroom_template")):
        return 0.0
    if str(meta_a.get("door_swing_assumption") or "") == "outward_or_sliding" and str(meta_b.get("door_swing_assumption") or "") == "outward_or_sliding":
        return 0.0
    return 0.02


def _sample_access_points(aabb: AABB, *, clearance: float = 0.45, step: float = 0.15) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    y = aabb.y_min
    while y <= aabb.y_max + 1e-6:
        points.append((aabb.x_min - clearance, y))
        points.append((aabb.x_max + clearance, y))
        y += step
    x = aabb.x_min
    while x <= aabb.x_max + 1e-6:
        points.append((x, aabb.y_min - clearance))
        points.append((x, aabb.y_max + clearance))
        x += step
    return points


def _reachable_points(ctx: RoomContext, solid_aabbs: list[AABB], *, step: float = 0.12) -> set[tuple[int, int]]:
    min_x, min_y, max_x, max_y = polygon_bounds(ctx.polygon)

    def to_cell(x: float, y: float) -> tuple[int, int]:
        return (round((x - min_x) / step), round((y - min_y) / step))

    def from_cell(cell: tuple[int, int]) -> tuple[float, float]:
        return (min_x + cell[0] * step, min_y + cell[1] * step)

    def is_free(x: float, y: float) -> bool:
        if not point_in_polygon(Vec2(x, y), ctx.polygon):
            return False
        return not any(_aabb_contains_xy(aabb, x, y, margin=0.03) for aabb in solid_aabbs)

    starts: list[tuple[int, int]] = []
    for zone in ctx.door_clearance_zones:
        x = zone.x_min
        while x <= zone.x_max + 1e-6:
            y = zone.y_min
            while y <= zone.y_max + 1e-6:
                if is_free(x, y):
                    starts.append(to_cell(x, y))
                y += step
            x += step
    if not starts and is_free(ctx.centroid.x, ctx.centroid.y):
        starts.append(to_cell(ctx.centroid.x, ctx.centroid.y))

    seen: set[tuple[int, int]] = set(starts)
    queue: deque[tuple[int, int]] = deque(starts)
    max_i = round((max_x - min_x) / step)
    max_j = round((max_y - min_y) / step)
    while queue:
        cell = queue.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (cell[0] + di, cell[1] + dj)
            if nxt in seen or nxt[0] < 0 or nxt[1] < 0 or nxt[0] > max_i or nxt[1] > max_j:
                continue
            x, y = from_cell(nxt)
            if not is_free(x, y):
                continue
            seen.add(nxt)
            queue.append(nxt)
    return seen


def _bedroom_functional_clearance_violations(ctx: RoomContext, solid_items: list[tuple[dict[str, Any], AABB]]) -> list[dict[str, Any]]:
    if ctx.room_type != "bedroom":
        return []

    violations: list[dict[str, Any]] = []
    room_min_x, _room_min_y, room_max_x, _room_max_y = ctx.bounds
    room_width = room_max_x - room_min_x
    solid_aabbs = [aabb for _item, aabb in solid_items]
    reachable = _reachable_points(ctx, solid_aabbs)
    min_x, min_y, _max_x, _max_y = polygon_bounds(ctx.polygon)
    step = 0.12

    def reachable_xy(x: float, y: float) -> bool:
        cell = (round((x - min_x) / step), round((y - min_y) / step))
        return cell in reachable

    targets = [(item, aabb) for item, aabb in solid_items if str(item.get("category") or "") in {"bed", "dresser", "wardrobe"}]
    for item, aabb in targets:
        if not any(reachable_xy(x, y) for x, y in _sample_access_points(aabb)):
            violations.append({"id": item.get("id"), "category": item.get("category"), "reason": "no_continuous_passage_from_door"})

    for item, aabb in solid_items:
        if str(item.get("category") or "") != "bed":
            continue
        if room_width <= 2.1:
            left_gap = aabb.x_min - room_min_x
            right_gap = room_max_x - aabb.x_max
            if max(left_gap, right_gap) < 0.45:
                violations.append(
                    {
                        "id": item.get("id"),
                        "category": item.get("category"),
                        "reason": "bed_splits_tiny_room_width",
                        "left_gap_m": round(left_gap, 3),
                        "right_gap_m": round(right_gap, 3),
                    }
                )

    blockers = [(item, aabb) for item, aabb in solid_items if str(item.get("category") or "") in {"bed", "bench"}]
    for item, aabb in solid_items:
        if str(item.get("category") or "") != "wardrobe":
            continue
        for blocker, blocker_aabb in blockers:
            gap = _gap_between(aabb, blocker_aabb)
            if gap < 0.55:
                violations.append(
                    {
                        "id": item.get("id"),
                        "category": item.get("category"),
                        "reason": "wardrobe_access_gap_too_small",
                        "against_id": blocker.get("id"),
                        "gap_m": round(gap, 3),
                    }
                )
                break

    return violations


def _generic_access_violations(ctx: RoomContext, solid_items: list[tuple[dict[str, Any], AABB]]) -> list[dict[str, Any]]:
    if not solid_items:
        return []
    solid_aabbs = [aabb for _item, aabb in solid_items]
    reachable = _reachable_points(ctx, solid_aabbs)
    min_x, min_y, _max_x, _max_y = polygon_bounds(ctx.polygon)
    step = 0.12

    def reachable_xy(x: float, y: float) -> bool:
        cell = (round((x - min_x) / step), round((y - min_y) / step))
        return cell in reachable

    violations: list[dict[str, Any]] = []
    for item, aabb in solid_items:
        constraints = item.get("constraints") if isinstance(item.get("constraints"), dict) else {}
        category = str(item.get("category") or "").strip().lower()
        requires_access = bool(constraints.get("requires_access")) or category in {
            "bed",
            "wardrobe",
            "wardrobe_module",
            "dresser",
            "desk",
            "chair",
            "sofa",
            "coffee_table",
            "tv_stand",
            "toilet",
            "sink",
            "bathtub",
            "shower",
            "washing_machine",
        }
        if not requires_access:
            continue
        clearance = 0.32 if ctx.min_side_m < 1.6 else 0.45
        if not any(reachable_xy(x, y) for x, y in _sample_access_points(aabb, clearance=clearance)):
            violations.append({"id": item.get("id"), "category": category, "reason": "no_access_point_reachable"})
    return violations


def validate_placements(ctx: RoomContext, placements: list[dict[str, Any]]) -> dict[str, Any]:
    collisions: list[dict[str, Any]] = []
    outside: list[dict[str, Any]] = []
    aabb_bounds_violations: list[dict[str, Any]] = []
    vertical_bounds_violations: list[dict[str, Any]] = []
    door_violations: list[dict[str, Any]] = []
    window_violations: list[dict[str, Any]] = []
    soft_floor_solid_overlaps: list[dict[str, Any]] = []
    aabb_center_mismatches: list[dict[str, Any]] = []
    orientation_contract_missing: list[dict[str, Any]] = []
    clearance_contract_missing: list[dict[str, Any]] = []
    required_missing: list[dict[str, Any]] = []

    solid_items: list[tuple[dict[str, Any], AABB]] = []
    soft_floor_items: list[tuple[dict[str, Any], AABB]] = []
    soft_item_count = 0
    room_min_x, room_min_y, room_max_x, room_max_y = ctx.bounds
    eps = 1e-6
    for item in placements:
        category = str(item.get("category", ""))
        meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
        role = str(meta.get("physical_role") or "").strip().lower()
        if meta.get("procedural") and role in {"solid_floor", "wall_mounted", "ceiling_mounted"} and not isinstance(item.get("orientation_rule"), dict):
            orientation_contract_missing.append({"id": item.get("id"), "category": category, "physical_role": role})
        if meta.get("procedural") and role == "solid_floor" and not isinstance(item.get("clearance_rule"), dict):
            clearance_contract_missing.append({"id": item.get("id"), "category": category, "physical_role": role})
        aabb = _aabb_from_item(item)
        if aabb is not None:
            if (
                aabb.x_min < room_min_x - eps
                or aabb.x_max > room_max_x + eps
                or aabb.y_min < room_min_y - eps
                or aabb.y_max > room_max_y + eps
            ):
                aabb_bounds_violations.append({"id": item.get("id"), "category": category, "aabb": aabb.to_json()})
            if aabb.z_min < -eps or aabb.z_max > ctx.ceiling_height_m + eps:
                vertical_bounds_violations.append(
                    {
                        "id": item.get("id"),
                        "category": category,
                        "z_min": aabb.z_min,
                        "z_max": aabb.z_max,
                        "room_height_m": ctx.ceiling_height_m,
                    }
                )
            pos = item.get("position_m")
            if isinstance(pos, list) and len(pos) >= 3:
                cx = (aabb.x_min + aabb.x_max) * 0.5
                cy = (aabb.y_min + aabb.y_max) * 0.5
                cz = (aabb.z_min + aabb.z_max) * 0.5
                dx = abs(cx - as_float(pos[0]))
                dy = abs(cy - as_float(pos[1]))
                dz = abs(cz - as_float(pos[2]))
                center_eps = 0.01
                if dx > center_eps or dy > center_eps or dz > center_eps:
                    aabb_center_mismatches.append(
                        {
                            "id": item.get("id"),
                            "category": category,
                            "position_m": [as_float(pos[0]), as_float(pos[1]), as_float(pos[2])],
                            "aabb_center_m": [cx, cy, cz],
                            "delta_m": [dx, dy, dz],
                        }
                    )
        if _is_soft_validation_item(item):
            soft_item_count += 1
            role = str((item.get("meta") if isinstance(item.get("meta"), dict) else {}).get("physical_role") or "").strip().lower()
            if aabb is not None and (role == "soft_floor" or category in {"bath_mat", "rug", "runner_rug"}):
                soft_floor_items.append((item, aabb))
            continue
        if aabb is None:
            continue
        solid_items.append((item, aabb))

        pos = item.get("position_m") or [0.0, 0.0, 0.0]
        size = item.get("size_m") or [0.0, 0.0, 0.0]
        yaw = as_float(item.get("yaw_deg", item.get("rotation_deg")), 0.0)
        if not object_footprint_inside_polygon(Vec2(as_float(pos[0]), as_float(pos[1])), size[:2], yaw, ctx.polygon):
            outside.append({"id": item.get("id"), "category": category})

        for zone in ctx.door_clearance_zones:
            if aabb.intersects_xy(zone, margin=0.0):
                if ctx.room_type == "toilet" and category == "toilet":
                    continue
                if bool(meta.get("door_clearance_exempt")):
                    continue
                door_violations.append({"id": item.get("id"), "category": category})
        for zone in ctx.window_clearance_zones:
            if aabb.intersects_xy(zone, margin=0.0):
                window_violations.append({"id": item.get("id"), "category": category})

    for i in range(len(solid_items)):
        item_a, aabb_a = solid_items[i]
        for j in range(i + 1, len(solid_items)):
            item_b, aabb_b = solid_items[j]
            margin = _collision_margin_for_pair(item_a, item_b)
            if aabb_a.intersects_xy(aabb_b, margin=margin):
                collisions.append(
                    {
                        "a": item_a.get("id"),
                        "b": item_b.get("id"),
                        "category_a": item_a.get("category"),
                        "category_b": item_b.get("category"),
                    }
                )
    if ctx.room_type in {"bathroom", "toilet"}:
        for soft_item, soft_aabb in soft_floor_items:
            for solid_item, solid_aabb in solid_items:
                if soft_aabb.intersects_xy(solid_aabb, margin=0.0):
                    soft_floor_solid_overlaps.append(
                        {
                            "soft_id": soft_item.get("id"),
                            "solid_id": solid_item.get("id"),
                            "soft_category": soft_item.get("category"),
                            "solid_category": solid_item.get("category"),
                        }
                    )

    functional_violations = _generic_access_violations(ctx, solid_items)
    functional_violations.extend(_bedroom_functional_clearance_violations(ctx, solid_items))
    categories = {str(item.get("category") or "").strip().lower() for item in placements if isinstance(item, dict)}
    if ctx.room_type == "toilet" and "toilet" not in categories:
        required_missing.append({"category": "toilet", "reason": "required_toilet_missing"})
    if ctx.room_type == "bathroom":
        if not ({"sink", "vanity"} & categories):
            required_missing.append({"category": "sink", "reason": "required_sink_missing"})
        if not ({"bathtub", "shower"} & categories):
            required_missing.append({"category": "bathtub_or_shower", "reason": "required_bathing_fixture_missing"})
    if ctx.room_type == "bedroom":
        if "bed" not in categories:
            required_missing.append({"category": "bed", "reason": "required_bed_missing"})
        if not ({"wardrobe", "wardrobe_module", "nightstand", "dresser"} & categories):
            required_missing.append({"category": "wardrobe_or_nightstand", "reason": "required_storage_or_nightstand_missing"})
    if ctx.room_type == "living_room" and "sofa" not in categories:
        required_missing.append({"category": "sofa", "reason": "required_sofa_missing"})

    return {
        "collisions": collisions,
        "outside_room": outside,
        "aabb_bounds_violations": aabb_bounds_violations,
        "vertical_bounds_violations": vertical_bounds_violations,
        "door_clearance_violations": door_violations,
        "window_clearance_violations": window_violations,
        "soft_floor_solid_overlaps": soft_floor_solid_overlaps,
        "aabb_center_mismatches": aabb_center_mismatches,
        "orientation_contract_missing": orientation_contract_missing,
        "clearance_contract_missing": clearance_contract_missing,
        "functional_clearance_violations": functional_violations,
        "required_missing": required_missing,
        "accessibility_ok": not collisions
        and not aabb_bounds_violations
        and not vertical_bounds_violations
        and not door_violations
        and not window_violations
        and not soft_floor_solid_overlaps
        and not aabb_center_mismatches
        and not outside
        and not functional_violations
        and not required_missing,
        "solid_item_count": len(solid_items),
        "soft_item_count": soft_item_count,
        "total_item_count": len(placements),
    }
