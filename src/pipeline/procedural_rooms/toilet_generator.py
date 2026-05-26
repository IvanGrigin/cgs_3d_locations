from __future__ import annotations

import random
from typing import Any

from .geometry import Vec2, as_float, choose_longest_wall, choose_wall_most_opposite, local_axes_from_yaw
from .object_specs import TOILET_SPECS, Density, ObjectSpec, density_rank
from .placement_engine import PlacementEngine, clamp_center_inside_room_for_aabb
from .room_context import RoomContext
from .sanitary_layout_solver import generate_sanitary_toilet


def _wall_candidates(ctx: RoomContext, rng: random.Random, *, min_length: float = 0.0) -> list[Any]:
    walls = [w for w in ctx.walls if w.length >= min_length and not w.has_door]
    if not walls:
        walls = [w for w in ctx.walls if w.length >= min_length]
    walls = sorted(walls, key=lambda w: (w.has_window, -w.length, w.id))
    if len(walls) > 1:
        first = walls[:1]
        rest = walls[1:]
        rng.shuffle(rest)
        walls = first + rest
    return walls


def _door_wall(ctx: RoomContext) -> Any | None:
    for wall in ctx.walls:
        if wall.has_door:
            return wall
    for door in ctx.doors:
        wall_id = str(door.get("wall_id") or "")
        wall = next((w for w in ctx.walls if w.id == wall_id), None)
        if wall:
            return wall
    return None


def _preferred_toilet_wall(ctx: RoomContext) -> Any | None:
    door_wall = _door_wall(ctx)
    if door_wall:
        return choose_wall_most_opposite(ctx.walls, door_wall, ctx.polygon, avoid_windows=False, avoid_doors=True)
    return choose_longest_wall(ctx.walls, avoid_windows=False, avoid_doors=True)


def _add_required_wall_item(
    engine: PlacementEngine,
    ctx: RoomContext,
    spec_keys: list[str],
    *,
    layer: str,
    category: str | None = None,
    front_target: str,
    preferred_wall: Any | None = None,
) -> dict[str, Any] | None:
    for key in spec_keys:
        spec = TOILET_SPECS[key]
        walls = [preferred_wall] if preferred_wall and preferred_wall.length >= spec.size_m[0] + 0.05 else []
        walls.extend(w for w in _wall_candidates(ctx, engine.rng, min_length=spec.size_m[0] + 0.05) if w not in walls)
        for wall in walls:
            factors = [0.5, 0.33, 0.67, 0.18, 0.82]
            for factor in factors:
                item = engine.add_wall_aligned(
                    spec,
                    wall.id,
                    wall.length * factor,
                    layer=layer,
                    category=category,
                    margin=0.02,
                    front_target=front_target,
                    extra_meta={"required": True},
                )
                if item:
                    return item
    return None


def _add_fallback_center_item(
    engine: PlacementEngine,
    ctx: RoomContext,
    spec: ObjectSpec,
    *,
    category: str | None = None,
    front_target: str,
) -> dict[str, Any] | None:
    center = clamp_center_inside_room_for_aabb(ctx.centroid, spec.size_m, polygon=ctx.polygon, yaw_deg=0.0, margin=0.03)
    return engine.add_item(
        spec,
        center,
        0.0,
        category=category,
        margin=0.01,
        ignore_door_clearance=True,
        ignore_window_clearance=True,
        front_target=front_target,
        extra_meta={"required": True, "fallback_center": True},
    )


def _add_wall_mount_near(
    engine: PlacementEngine,
    ctx: RoomContext,
    anchor: dict[str, Any] | None,
    spec_key: str,
    *,
    name: str | None = None,
    z_center: float = 1.2,
    along_delta_m: float = 0.0,
) -> dict[str, Any] | None:
    spec = TOILET_SPECS[spec_key]
    wall_id = ""
    along = None
    if anchor:
        meta = anchor.get("meta") if isinstance(anchor.get("meta"), dict) else {}
        wall_id = str(meta.get("wall_id") or "")
        along = as_float(meta.get("wall_along_m"), None)
    wall = next((w for w in ctx.walls if w.id == wall_id), None)
    if wall is None:
        wall = choose_longest_wall(ctx.walls, avoid_windows=False, avoid_doors=True)
        along = wall.length * 0.5 if wall else None
    if wall is None or along is None:
        return None
    item = engine.add_wall_art(wall.id, along + along_delta_m, spec, z_center=z_center, name=name, category=spec.category)
    if item and anchor:
        meta = item.setdefault("meta", {})
        if isinstance(meta, dict):
            meta["anchor_id"] = anchor.get("id")
            meta["placement_relation"] = "near"
    return item


def generate_toilet(ctx: RoomContext, *, density: Density, seed: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    solved = generate_sanitary_toilet(ctx, density=density, seed=seed)
    if solved is not None:
        return solved

    rng = random.Random(seed)
    engine = PlacementEngine(
        ctx=ctx,
        rng=rng,
        source_name="procedural_room_stage",
        generator_name="toilet_generator",
        archetype="compact_wc_accessible_front",
    )

    toilet = _add_required_wall_item(
        engine,
        ctx,
        ["toilet", "compact_toilet"],
        layer="primary",
        category="toilet",
        front_target="door",
        preferred_wall=_preferred_toilet_wall(ctx),
    )
    if toilet is None:
        toilet = _add_fallback_center_item(engine, ctx, TOILET_SPECS["compact_toilet"], category="toilet", front_target="door")

    sink = None
    if ctx.area_m2 >= 1.4 or ctx.min_side_m >= 0.9:
        preferred_wall = None
        if toilet:
            wall_id = str((toilet.get("meta") if isinstance(toilet.get("meta"), dict) else {}).get("wall_id") or "")
            toilet_wall = next((w for w in ctx.walls if w.id == wall_id), None)
            if toilet_wall:
                preferred_wall = choose_wall_most_opposite(ctx.walls, toilet_wall, ctx.polygon, avoid_windows=False, avoid_doors=True)
        for key in ["sink", "corner_sink"]:
            spec = TOILET_SPECS[key]
            walls = [preferred_wall] if preferred_wall else []
            walls.extend(w for w in _wall_candidates(ctx, rng, min_length=spec.size_m[0]) if w not in walls)
            for wall in [w for w in walls if w]:
                for factor in (0.22, 0.78, 0.5):
                    sink = engine.add_wall_aligned(
                        spec,
                        wall.id,
                        wall.length * factor,
                        layer="primary",
                        category="sink",
                        margin=0.02,
                        front_target="door",
                        extra_meta={"required_if_space": True},
                    )
                    if sink:
                        break
                if sink:
                    break
            if sink:
                break

    cabinet = None
    if density_rank(density) >= 2 and ctx.area_m2 >= 1.8:
        spec = TOILET_SPECS["toilet_cabinet"]
        occupied_wall_ids = {
            str((item.get("meta") if isinstance(item.get("meta"), dict) else {}).get("wall_id") or "")
            for item in (toilet, sink)
            if item
        }
        walls = [w for w in _wall_candidates(ctx, rng, min_length=spec.size_m[0]) if w.id not in occupied_wall_ids]
        walls.extend(w for w in _wall_candidates(ctx, rng, min_length=spec.size_m[0]) if w not in walls)
        for wall in walls:
            for factor in (0.5, 0.24, 0.76):
                cabinet = engine.add_wall_aligned(
                    spec,
                    wall.id,
                    wall.length * factor,
                    layer="storage",
                    category="toilet_cabinet",
                    margin=0.02,
                    front_target="door",
                    extra_meta={"required_if_space": True},
                )
                if cabinet:
                    break
            if cabinet:
                break

    _add_wall_mount_near(engine, ctx, toilet, "toilet_paper_holder", name="Toilet paper holder", z_center=0.75, along_delta_m=0.35)
    if density_rank(density) >= 2:
        _add_wall_mount_near(engine, ctx, toilet, "hygiene_shower", name="Hygiene shower", z_center=0.85, along_delta_m=-0.32)

    if sink:
        engine.add_on_top(sink, TOILET_SPECS["soap_dispenser"], local_offset_xy=(-0.10, -0.02), name="Hand soap dispenser")
        _add_wall_mount_near(engine, ctx, sink, "mirror", name="Mirror above sink", z_center=1.42)
    elif density_rank(density) >= 2:
        _add_wall_mount_near(engine, ctx, toilet, "wall_shelf", name="Small wall shelf", z_center=1.55, along_delta_m=-0.45)

    if density_rank(density) >= 2 and ctx.area_m2 >= 1.6:
        if toilet:
            engine.add_near(
                toilet,
                TOILET_SPECS["small_bin"],
                local_offset_xy=(0.34, 0.05),
                allow_collision=False,
                front_target="door",
            )
    if density_rank(density) >= 3:
        shelf = _add_wall_mount_near(engine, ctx, toilet or sink, "wall_shelf", name="Decor shelf", z_center=1.55, along_delta_m=0.55)
        if shelf:
            engine.add_on_top(shelf, TOILET_SPECS["air_freshener"], local_offset_xy=(0.0, 0.0), name="Air freshener")

    engine.add_ceiling_light(name="Toilet ceiling light")

    report = {
        "generator": "toilet_generator",
        "archetype": engine.archetype,
        "required": {"toilet": bool(toilet)},
        "optional": {"sink": bool(sink), "toilet_cabinet": bool(cabinet)},
        "density": density,
        "rejected": engine.rejected,
    }
    return engine.placements, report
