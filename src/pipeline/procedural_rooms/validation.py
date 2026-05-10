from __future__ import annotations

from typing import Any

from .geometry import AABB, as_float, object_footprint_inside_polygon, Vec2
from .placement_engine import COLLISION_IGNORE_CATEGORIES
from .room_context import RoomContext


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


def _is_soft_validation_item(item: dict[str, Any]) -> bool:
    category = str(item.get("category", "")).strip().lower()
    mount_type = str(item.get("mount_type", "")).strip().lower()
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    layer = str(meta.get("density_layer", "")).strip().lower()

    if category in COLLISION_IGNORE_CATEGORIES:
        return True
    if mount_type in {"wall", "ceiling"}:
        return True
    if meta.get("allow_collision"):
        return True
    if meta.get("support_relation") == "on_top":
        return True
    if layer in {"soft_decor", "decor", "wall_decor", "textile", "lighting", "electronics"}:
        return True
    return False


def validate_placements(ctx: RoomContext, placements: list[dict[str, Any]]) -> dict[str, Any]:
    collisions: list[dict[str, Any]] = []
    outside: list[dict[str, Any]] = []
    door_violations: list[dict[str, Any]] = []

    solid_items: list[tuple[dict[str, Any], AABB]] = []
    soft_item_count = 0
    for item in placements:
        category = str(item.get("category", ""))
        if _is_soft_validation_item(item):
            soft_item_count += 1
            continue
        aabb = _aabb_from_item(item)
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
                door_violations.append({"id": item.get("id"), "category": category})

    for i in range(len(solid_items)):
        item_a, aabb_a = solid_items[i]
        for j in range(i + 1, len(solid_items)):
            item_b, aabb_b = solid_items[j]
            if aabb_a.intersects_xy(aabb_b, margin=0.02):
                collisions.append(
                    {
                        "a": item_a.get("id"),
                        "b": item_b.get("id"),
                        "category_a": item_a.get("category"),
                        "category_b": item_b.get("category"),
                    }
                )

    return {
        "collisions": collisions,
        "outside_room": outside,
        "door_clearance_violations": door_violations,
        "accessibility_ok": not collisions and not door_violations and not outside,
        "solid_item_count": len(solid_items),
        "soft_item_count": soft_item_count,
        "total_item_count": len(placements),
    }
