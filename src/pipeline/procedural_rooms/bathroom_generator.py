from __future__ import annotations

import random
from typing import Any

from .geometry import Vec2, as_float, choose_longest_wall, choose_wall_most_opposite, local_axes_from_yaw
from .object_specs import BATHROOM_SPECS, Density, ObjectSpec, density_rank
from .placement_engine import PlacementEngine, clamp_center_inside_room_for_aabb
from .room_context import RoomContext


def _candidate_walls(ctx: RoomContext, rng: random.Random, *, min_length: float = 0.0, avoid_windows: bool = False) -> list[Any]:
    walls = [
        wall
        for wall in ctx.walls
        if wall.length >= min_length and not wall.has_door and (not avoid_windows or not wall.has_window)
    ]
    if not walls:
        walls = [wall for wall in ctx.walls if wall.length >= min_length and not wall.has_door]
    if not walls:
        walls = [wall for wall in ctx.walls if wall.length >= min_length]
    walls = sorted(walls, key=lambda wall: (wall.has_window, -wall.length, wall.id))
    if len(walls) > 1:
        head = walls[:1]
        rest = walls[1:]
        rng.shuffle(rest)
        walls = head + rest
    return walls


def _try_wall_specs(
    engine: PlacementEngine,
    ctx: RoomContext,
    keys: list[str],
    *,
    category: str | None,
    layer: str,
    front_target: str,
    avoid_windows: bool = False,
    required: bool = False,
) -> dict[str, Any] | None:
    for key in keys:
        spec = BATHROOM_SPECS[key]
        for wall in _candidate_walls(ctx, engine.rng, min_length=spec.size_m[0] + 0.05, avoid_windows=avoid_windows):
            factors = [0.5, 0.625, 0.375, 0.24, 0.76, 0.36, 0.64]
            for factor in factors:
                item = engine.add_wall_aligned(
                    spec,
                    wall.id,
                    wall.length * factor,
                    category=category or spec.category,
                    layer=layer,
                    margin=0.02,
                    front_target=front_target,
                    extra_meta={"required": required},
                )
                if item:
                    return item
    return None


def _fallback_center_fixture(
    engine: PlacementEngine,
    ctx: RoomContext,
    spec: ObjectSpec,
    *,
    category: str,
    front_target: str,
) -> dict[str, Any] | None:
    center = clamp_center_inside_room_for_aabb(ctx.centroid, spec.size_m, polygon=ctx.polygon, yaw_deg=0.0, margin=0.03)
    return engine.add_item(
        spec,
        center,
        0.0,
        category=category,
        layer="primary",
        margin=0.01,
        ignore_door_clearance=True,
        ignore_window_clearance=True,
        front_target=front_target,
        extra_meta={"required": True, "fallback_center": True},
    )


def _wall_id_and_along(item: dict[str, Any] | None) -> tuple[str, float | None]:
    if not item:
        return "", None
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    return str(meta.get("wall_id") or ""), as_float(meta.get("wall_along_m"), None)


def _add_wall_near(
    engine: PlacementEngine,
    ctx: RoomContext,
    anchor: dict[str, Any] | None,
    spec_key: str,
    *,
    z_center: float,
    category: str | None = None,
    name: str | None = None,
    along_delta_m: float = 0.0,
) -> dict[str, Any] | None:
    wall_id, along = _wall_id_and_along(anchor)
    wall = next((w for w in ctx.walls if w.id == wall_id), None)
    if wall is None:
        wall = choose_longest_wall(ctx.walls, avoid_windows=False, avoid_doors=True)
        along = wall.length * 0.5 if wall else None
    if wall is None or along is None:
        return None
    spec = BATHROOM_SPECS[spec_key]
    return engine.add_wall_art(wall.id, along + along_delta_m, spec, z_center=z_center, category=category or spec.category, name=name)


def _add_bath_mat_in_front(engine: PlacementEngine, fixture: dict[str, Any] | None) -> dict[str, Any] | None:
    if not fixture:
        return None
    spec = BATHROOM_SPECS["bath_mat"]
    fixture_pos = fixture.get("position_m") or [0.0, 0.0, 0.0]
    fixture_size = fixture.get("size_m") or [0.0, 0.0, 0.0]
    fixture_yaw = as_float(fixture.get("yaw_deg", fixture.get("rotation_deg")), 0.0)
    ux, front = local_axes_from_yaw(fixture_yaw)
    base = Vec2(as_float(fixture_pos[0]), as_float(fixture_pos[1]))
    front_distance = as_float(fixture_size[1]) * 0.5 + spec.size_m[1] * 0.5 + 0.06
    for lateral in (0.0, -0.18, 0.18, -0.32, 0.32):
        center = base + front * front_distance + ux * lateral
        ok, reason = engine.can_place(center, spec.size_m, fixture_yaw, allow_collision=False, margin=0.02)
        if not ok:
            continue
        return engine.add_item(
            spec,
            center,
            fixture_yaw,
            allow_collision=True,
            layer="textile",
            front_target=fixture.get("id"),
            extra_meta={"anchor_id": fixture.get("id"), "placement_relation": "in_front_of"},
        )
    engine.rejected.append(
        {
            "category": spec.category,
            "name": spec.name,
            "reason": "bath_mat_no_front_clearance",
            "anchor_id": fixture.get("id"),
            "size_m": list(spec.size_m),
        }
    )
    return None


def generate_bathroom(ctx: RoomContext, *, density: Density, seed: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    engine = PlacementEngine(
        ctx=ctx,
        rng=rng,
        source_name="procedural_room_stage",
        generator_name="bathroom_generator",
        archetype="wet_wall_and_accessible_sink",
    )

    # Required bathing fixture: prefer bathtub when area permits, otherwise shower.
    bathing = None
    if ctx.area_m2 >= 3.8 and ctx.max_side_m >= 1.55:
        bathing = _try_wall_specs(
            engine,
            ctx,
            ["bathtub", "compact_bathtub", "compact_shower"],
            category=None,
            layer="primary",
            front_target="room_center",
            avoid_windows=False,
            required=True,
        )
    if bathing is None:
        bathing = _try_wall_specs(
            engine,
            ctx,
            ["compact_shower", "shower", "compact_bathtub"],
            category=None,
            layer="primary",
            front_target="door",
            avoid_windows=False,
            required=True,
        )
    if bathing is None:
        bathing = _fallback_center_fixture(engine, ctx, BATHROOM_SPECS["compact_shower"], category="shower", front_target="door")

    # Required sink. Prefer a wall opposite/adjacent to the wet fixture.
    sink = None
    preferred_wall = None
    if bathing:
        wall_id, _along = _wall_id_and_along(bathing)
        fixture_wall = next((w for w in ctx.walls if w.id == wall_id), None)
        if fixture_wall:
            preferred_wall = choose_wall_most_opposite(ctx.walls, fixture_wall, ctx.polygon, avoid_windows=False, avoid_doors=True)

    sink_keys = ["sink", "compact_sink"]
    for key in sink_keys:
        spec = BATHROOM_SPECS[key]
        walls = [preferred_wall] if preferred_wall else []
        walls.extend(w for w in _candidate_walls(ctx, rng, min_length=spec.size_m[0], avoid_windows=False) if w not in walls)
        for wall in [w for w in walls if w]:
            for factor in (0.5, 0.25, 0.75):
                sink = engine.add_wall_aligned(
                    spec,
                    wall.id,
                    wall.length * factor,
                    category="sink",
                    layer="primary",
                    margin=0.02,
                    front_target="door",
                    extra_meta={"required": True},
                )
                if sink:
                    break
            if sink:
                break
        if sink:
            break
    if sink is None:
        sink = _fallback_center_fixture(engine, ctx, BATHROOM_SPECS["compact_sink"], category="sink", front_target="door")

    if sink:
        engine.add_on_top(sink, BATHROOM_SPECS["soap_dispenser"], local_offset_xy=(-0.14, -0.02), name="Soap dispenser")
        engine.add_on_top(sink, BATHROOM_SPECS["toothbrush_cup"], local_offset_xy=(0.14, -0.02), name="Toothbrush cup")
        _add_wall_near(engine, ctx, sink, "mirror", z_center=1.45, category="mirror", name="Mirror above sink")

    if bathing:
        _add_bath_mat_in_front(engine, bathing)
        _add_wall_near(engine, ctx, bathing, "towel_rack", z_center=1.45, category="towel_rack", name="Towel rack", along_delta_m=0.55)
        if density_rank(density) >= 2:
            _add_wall_near(engine, ctx, bathing, "wall_shelf", z_center=1.25, category="shelf", name="Shower shelf", along_delta_m=-0.45)

    # Optional toilet in larger bathrooms; standalone WC generator always provides one.
    toilet = None
    if ctx.area_m2 >= 4.2 and density_rank(density) >= 2:
        toilet = _try_wall_specs(
            engine,
            ctx,
            ["toilet", "compact_toilet"],
            category="toilet",
            layer="primary",
            front_target="door",
            required=False,
        )

    if ctx.area_m2 >= 4.5 and density_rank(density) >= 2:
        _try_wall_specs(
            engine,
            ctx,
            ["washing_machine"],
            category="washing_machine",
            layer="appliance",
            front_target="door",
            required=False,
        )

    if density_rank(density) >= 2 and ctx.area_m2 >= 2.8:
        engine.add_corner_object(BATHROOM_SPECS["laundry_basket"], preferred_index=1, category="laundry_basket", name="Laundry basket")

    if density_rank(density) >= 3:
        shelf = _add_wall_near(engine, ctx, sink or bathing, "wall_shelf", z_center=1.62, category="shelf", name="Extra bathroom shelf", along_delta_m=0.45)
        if shelf:
            engine.add_on_top(shelf, BATHROOM_SPECS["shampoo_bottle"], local_offset_xy=(-0.12, 0.0), name="Shampoo bottle")
            engine.add_on_top(shelf, BATHROOM_SPECS["shampoo_bottle"], local_offset_xy=(0.12, 0.0), name="Conditioner bottle")

    engine.add_ceiling_light(name="Bathroom ceiling light")

    report = {
        "generator": "bathroom_generator",
        "archetype": engine.archetype,
        "required": {"sink": bool(sink), "bathing_fixture": bool(bathing)},
        "bathing_fixture_category": bathing.get("category") if bathing else None,
        "toilet_added": bool(toilet),
        "density": density,
        "rejected": engine.rejected,
    }
    return engine.placements, report
