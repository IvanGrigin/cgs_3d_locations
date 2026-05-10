from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Any, Sequence

from .geometry import (
    AABB,
    Vec2,
    aabb_from_box,
    as_float,
    clamp,
    local_axes_from_yaw,
    nearest_corner_candidates,
    normalize_angle_deg,
    object_footprint_inside_polygon,
    polygon_bounds,
    polygon_centroid,
    wall_inside_normal,
    yaw_for_local_y_to_vector,
)
from .object_specs import ObjectSpec
from .room_context import RoomContext


COLLISION_IGNORE_CATEGORIES = {
    "rug",
    "runner_rug",
    "wall_art",
    "wall_light",
    "tv",
    "mirror",
    "wall_hooks",
    "ceiling_light",
    "floor_lamp",
    "table_lamp",
    "plant",
    "pillow",
    "blanket",
    "decor_books",
    "decor_vase",
    "decor_box",
    "decor_tray",
    "storage_basket",
    "umbrella_stand",
}


def clamp_center_inside_room_for_aabb(
    center: Vec2,
    size_m: Sequence[float],
    *,
    polygon: Sequence[Vec2],
    yaw_deg: float = 0.0,
    margin: float = 0.03,
) -> Vec2:
    """Clamp an object's center so its yawed AABB stays inside room bounds."""
    x_min, y_min, x_max, y_max = polygon_bounds(polygon)
    footprint_aabb = aabb_from_box([0.0, 0.0, 0.0], size_m, yaw_deg)
    half_x = max(abs(footprint_aabb.x_min), abs(footprint_aabb.x_max)) + margin
    half_y = max(abs(footprint_aabb.y_min), abs(footprint_aabb.y_max)) + margin

    low_x = x_min + half_x
    high_x = x_max - half_x
    low_y = y_min + half_y
    high_y = y_max - half_y
    if low_x > high_x:
        low_x = high_x = (x_min + x_max) * 0.5
    if low_y > high_y:
        low_y = high_y = (y_min + y_max) * 0.5

    clamped = Vec2(clamp(center.x, low_x, high_x), clamp(center.y, low_y, high_y))
    if object_footprint_inside_polygon(clamped, size_m[:2], yaw_deg, polygon):
        return clamped

    # Bbox clamping is enough for rectangular rooms. For slightly irregular
    # polygons, move inward toward the room centroid until the footprint fits.
    centroid = polygon_centroid(polygon)
    for step in (0.25, 0.5, 0.75, 1.0):
        candidate = Vec2(
            clamp(clamped.x + (centroid.x - clamped.x) * step, low_x, high_x),
            clamp(clamped.y + (centroid.y - clamped.y) * step, low_y, high_y),
        )
        if object_footprint_inside_polygon(candidate, size_m[:2], yaw_deg, polygon):
            return candidate
    return clamped


@dataclass
class PlacementEngine:
    ctx: RoomContext
    rng: random.Random
    source_name: str
    generator_name: str
    archetype: str
    placements: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    _counter: int = 0

    def next_id(self, category: str) -> str:
        self._counter += 1
        safe_category = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in category.lower())
        return f"{self.ctx.room_type}_{safe_category}_{self._counter:04d}"

    def existing_aabbs(self, *, include_soft: bool = False) -> list[AABB]:
        result: list[AABB] = []
        for item in self.placements:
            category = str(item.get("category", ""))
            if not include_soft and category in COLLISION_IGNORE_CATEGORIES:
                continue
            if item.get("meta", {}).get("allow_collision"):
                continue
            aabb_json = item.get("aabb")
            if not isinstance(aabb_json, dict):
                continue
            result.append(
                AABB(
                    x_min=as_float(aabb_json.get("x_min")),
                    x_max=as_float(aabb_json.get("x_max")),
                    y_min=as_float(aabb_json.get("y_min")),
                    y_max=as_float(aabb_json.get("y_max")),
                    z_min=as_float(aabb_json.get("z_min")),
                    z_max=as_float(aabb_json.get("z_max")),
                )
            )
        return result

    def can_place(
        self,
        center: Vec2,
        size_m: Sequence[float],
        yaw_deg: float,
        *,
        allow_collision: bool = False,
        margin: float = 0.03,
        ignore_door_clearance: bool = False,
    ) -> tuple[bool, str]:
        if not object_footprint_inside_polygon(center, size_m[:2], yaw_deg, self.ctx.polygon):
            return False, "outside_room_polygon"

        aabb = aabb_from_box([center.x, center.y, as_float(size_m[2]) * 0.5], size_m, yaw_deg)

        if not ignore_door_clearance:
            for zone in self.ctx.door_clearance_zones:
                if aabb.intersects_xy(zone, margin=0.0):
                    return False, "door_clearance_collision"

        if allow_collision:
            return True, "ok"

        for other in self.existing_aabbs():
            if aabb.intersects_xy(other, margin=margin):
                return False, "object_collision"

        return True, "ok"

    def make_item(
        self,
        spec: ObjectSpec,
        center: Vec2,
        yaw_deg: float,
        *,
        z_center: float | None = None,
        name: str | None = None,
        category: str | None = None,
        layer: str | None = None,
        mount_type: str | None = None,
        wall_contact_side: str | None = None,
        extra_meta: dict[str, Any] | None = None,
        constraints: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        size = [float(spec.size_m[0]), float(spec.size_m[1]), float(spec.size_m[2])]
        z = z_center if z_center is not None else size[2] * 0.5
        yaw = normalize_angle_deg(yaw_deg)
        aabb = aabb_from_box([center.x, center.y, z], size, yaw)
        item_category = category or spec.category
        meta = {
            "procedural": True,
            "density_layer": layer or spec.layer,
            "replace_with_supplier": spec.replace_with_supplier,
            "allow_collision": bool(spec.allow_collision),
        }
        if extra_meta:
            meta.update(extra_meta)

        return {
            "id": self.next_id(item_category),
            "name": name or spec.name,
            "category": item_category,
            "position_m": [center.x, center.y, z],
            "size_m": size,
            "rotation_deg": yaw,
            "yaw_deg": yaw,
            "yaw_rad": math.radians(yaw),
            "aabb": aabb.to_json(),
            "mount_type": mount_type or spec.mount_type,
            "wall_contact_side": wall_contact_side,
            "constraints": constraints or {"requires_access": bool(spec.requires_access)},
            "asset": {
                "kind": "procedural_placeholder",
                "mesh_fit_mode": "fit",
            },
            "source": {
                "placement_source": self.source_name,
                "generator": self.generator_name,
                "archetype": self.archetype,
            },
            "meta": meta,
        }

    def add_item(
        self,
        spec: ObjectSpec,
        center: Vec2,
        yaw_deg: float,
        *,
        z_center: float | None = None,
        name: str | None = None,
        category: str | None = None,
        layer: str | None = None,
        allow_collision: bool | None = None,
        margin: float = 0.03,
        ignore_door_clearance: bool = False,
        mount_type: str | None = None,
        wall_contact_side: str | None = None,
        extra_meta: dict[str, Any] | None = None,
        constraints: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        collision_allowed = spec.allow_collision if allow_collision is None else allow_collision
        ok, reason = self.can_place(
            center,
            spec.size_m,
            yaw_deg,
            allow_collision=collision_allowed,
            margin=margin,
            ignore_door_clearance=ignore_door_clearance,
        )
        if not ok:
            self.rejected.append(
                {
                    "category": category or spec.category,
                    "name": name or spec.name,
                    "reason": reason,
                    "center": [center.x, center.y],
                    "size_m": list(spec.size_m),
                    "yaw_deg": yaw_deg,
                }
            )
            return None

        item = self.make_item(
            spec,
            center,
            yaw_deg,
            z_center=z_center,
            name=name,
            category=category,
            layer=layer,
            mount_type=mount_type,
            wall_contact_side=wall_contact_side,
            extra_meta=extra_meta,
            constraints=constraints,
        )
        self.placements.append(item)
        return item

    def add_wall_aligned(
        self,
        spec: ObjectSpec,
        wall_id: str,
        along_center_m: float,
        *,
        offset_from_wall_m: float | None = None,
        name: str | None = None,
        category: str | None = None,
        layer: str | None = None,
        allow_collision: bool | None = None,
        margin: float = 0.03,
        ignore_door_clearance: bool = False,
        extra_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        wall = next((w for w in self.ctx.walls if w.id == wall_id), None)
        if wall is None:
            return None

        normal = wall_inside_normal(wall, self.ctx.polygon)
        depth = spec.size_m[1]
        offset = depth * 0.5 if offset_from_wall_m is None else offset_from_wall_m
        along = clamp(along_center_m, spec.size_m[0] * 0.5, max(spec.size_m[0] * 0.5, wall.length - spec.size_m[0] * 0.5))
        base = wall.point_at(along)
        center = base + normal * offset
        yaw = yaw_for_local_y_to_vector(normal)
        return self.add_item(
            spec,
            center,
            yaw,
            name=name,
            category=category,
            layer=layer,
            allow_collision=allow_collision,
            margin=margin,
            ignore_door_clearance=ignore_door_clearance,
            wall_contact_side="back",
            extra_meta={**(extra_meta or {}), "wall_id": wall_id, "wall_along_m": along},
        )

    def add_on_top(
        self,
        parent: dict[str, Any] | None,
        spec: ObjectSpec,
        *,
        local_offset_xy: tuple[float, float] = (0.0, 0.0),
        yaw_delta_deg: float = 0.0,
        name: str | None = None,
        category: str | None = None,
        layer: str | None = None,
    ) -> dict[str, Any] | None:
        if not parent:
            return None
        parent_pos = parent.get("position_m") or [0.0, 0.0, 0.0]
        parent_size = parent.get("size_m") or [0.0, 0.0, 0.0]
        parent_yaw = as_float(parent.get("yaw_deg", parent.get("rotation_deg")), 0.0)
        ux, uy = local_axes_from_yaw(parent_yaw)
        center_xy = Vec2(as_float(parent_pos[0]), as_float(parent_pos[1]))
        center_xy = center_xy + ux * local_offset_xy[0] + uy * local_offset_xy[1]
        z = as_float(parent_pos[2]) + as_float(parent_size[2]) * 0.5 + spec.size_m[2] * 0.5
        return self.add_item(
            spec,
            center_xy,
            parent_yaw + yaw_delta_deg,
            z_center=z,
            name=name,
            category=category,
            layer=layer,
            allow_collision=True,
            ignore_door_clearance=True,
            extra_meta={"parent_id": parent.get("id"), "support_relation": "on_top"},
        )

    def add_near(
        self,
        anchor: dict[str, Any] | None,
        spec: ObjectSpec,
        *,
        local_offset_xy: tuple[float, float],
        yaw_delta_deg: float = 0.0,
        name: str | None = None,
        category: str | None = None,
        layer: str | None = None,
        allow_collision: bool | None = None,
    ) -> dict[str, Any] | None:
        if not anchor:
            return None
        parent_pos = anchor.get("position_m") or [0.0, 0.0, 0.0]
        parent_yaw = as_float(anchor.get("yaw_deg", anchor.get("rotation_deg")), 0.0)
        ux, uy = local_axes_from_yaw(parent_yaw)
        center_xy = Vec2(as_float(parent_pos[0]), as_float(parent_pos[1]))
        center_xy = center_xy + ux * local_offset_xy[0] + uy * local_offset_xy[1]
        return self.add_item(
            spec,
            center_xy,
            parent_yaw + yaw_delta_deg,
            name=name,
            category=category,
            layer=layer,
            allow_collision=allow_collision,
            extra_meta={"anchor_id": anchor.get("id"), "placement_relation": "near"},
        )

    def add_wall_art(
        self,
        wall_id: str,
        along_center_m: float,
        spec: ObjectSpec,
        *,
        z_center: float = 1.55,
        name: str | None = None,
        category: str | None = None,
        layer: str | None = None,
    ) -> dict[str, Any] | None:
        wall = next((w for w in self.ctx.walls if w.id == wall_id), None)
        if wall is None:
            return None
        normal = wall_inside_normal(wall, self.ctx.polygon)
        along = clamp(along_center_m, spec.size_m[0] * 0.5, max(spec.size_m[0] * 0.5, wall.length - spec.size_m[0] * 0.5))
        center = wall.point_at(along) + normal * 0.035
        yaw = yaw_for_local_y_to_vector(normal)
        return self.add_item(
            spec,
            center,
            yaw,
            z_center=z_center,
            name=name,
            category=category,
            layer=layer,
            allow_collision=True,
            ignore_door_clearance=True,
            mount_type="wall",
            wall_contact_side="back",
            extra_meta={"wall_id": wall_id, "wall_along_m": along},
        )

    def add_ceiling_light(self, category: str = "ceiling_light", name: str = "Ceiling light") -> dict[str, Any]:
        from .object_specs import ObjectSpec

        size = (0.45, 0.45, 0.18)
        spec = ObjectSpec(category=category, name=name, size_m=size, layer="lighting", mount_type="ceiling", allow_collision=True)
        center = self.ctx.centroid
        item = self.make_item(
            spec,
            center,
            0.0,
            z_center=max(2.2, self.ctx.ceiling_height_m - size[2] * 0.5),
            mount_type="ceiling",
            extra_meta={"ceiling_mounted": True},
        )
        self.placements.append(item)
        return item

    def add_corner_object(self, spec: ObjectSpec, *, preferred_index: int = 0, category: str | None = None, name: str | None = None) -> dict[str, Any] | None:
        corners = nearest_corner_candidates(self.ctx.polygon, inset=max(spec.size_m[0], spec.size_m[1]) * 0.7)
        if not corners:
            return None
        ordered = corners[preferred_index:] + corners[:preferred_index]
        first_rejection: dict[str, Any] | None = None
        for center in ordered:
            clamped_center = clamp_center_inside_room_for_aabb(center, spec.size_m, polygon=self.ctx.polygon, margin=0.03)
            ok, reason = self.can_place(clamped_center, spec.size_m, 0.0, allow_collision=spec.allow_collision, margin=0.02)
            if not ok:
                if first_rejection is None:
                    first_rejection = {
                        "category": category or spec.category,
                        "name": name or spec.name,
                        "reason": reason,
                        "center": [clamped_center.x, clamped_center.y],
                        "size_m": list(spec.size_m),
                        "yaw_deg": 0.0,
                    }
                continue
            item = self.make_item(spec, clamped_center, 0.0, category=category, name=name)
            self.placements.append(item)
            return item
        if first_rejection:
            self.rejected.append(first_rejection)
        return None

    def clone_item_with_new_size(self, item: dict[str, Any], new_size_m: Sequence[float]) -> dict[str, Any]:
        cloned = copy.deepcopy(item)
        cloned["size_m"] = [float(new_size_m[0]), float(new_size_m[1]), float(new_size_m[2])]
        pos = cloned.get("position_m") or [0.0, 0.0, 0.0]
        yaw = as_float(cloned.get("yaw_deg", cloned.get("rotation_deg")), 0.0)
        cloned["aabb"] = aabb_from_box(pos, new_size_m, yaw).to_json()
        return cloned
