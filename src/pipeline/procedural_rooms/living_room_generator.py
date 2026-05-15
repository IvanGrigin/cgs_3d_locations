from __future__ import annotations

import random
from typing import Any

from .geometry import Vec2, as_float, choose_longest_wall, choose_wall_most_opposite, local_axes_from_yaw, wall_inside_normal, yaw_for_local_y_to_vector
from .object_specs import LIVING_ROOM_SPECS, Density, ObjectSpec, density_rank
from .placement_engine import PlacementEngine, clamp_center_inside_room_for_aabb
from .room_context import RoomContext


def _sofa_spec_key(ctx: RoomContext, density: Density) -> str:
    if ctx.area_m2 >= 24.0 and density_rank(density) >= 2:
        return "sectional_sofa"
    if ctx.area_m2 >= 13.5:
        return "sofa_3"
    return "sofa_2"


def generate_living_room(ctx: RoomContext, *, density: Density, seed: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    engine = PlacementEngine(
        ctx=ctx,
        rng=rng,
        source_name="procedural_room_stage",
        generator_name="living_room_generator",
        archetype="sofa_facing_tv_wall",
    )

    tv_wall = choose_longest_wall(ctx.walls, avoid_windows=True, avoid_doors=True, min_length=1.8)
    if tv_wall is None:
        tv_wall = choose_longest_wall(ctx.walls, avoid_windows=False, avoid_doors=True)
    if tv_wall is None:
        return [], {"generator": "living_room_generator", "status": "no_wall"}

    tv_stand = engine.add_wall_aligned(
        LIVING_ROOM_SPECS["tv_stand"],
        tv_wall.id,
        tv_wall.length * 0.5,
        layer="primary",
        front_target="sofa",
    )
    if tv_stand:
        engine.add_wall_art(tv_wall.id, tv_wall.length * 0.5, LIVING_ROOM_SPECS["tv"], z_center=1.25, category="tv", name="Wall-mounted TV")

    sofa_keys = [_sofa_spec_key(ctx, density)]
    for fallback_key in ("sofa_2", "sofa_3"):
        if fallback_key not in sofa_keys:
            sofa_keys.append(fallback_key)
    sofa_spec = LIVING_ROOM_SPECS[sofa_keys[0]]
    sofa_wall = choose_wall_most_opposite(ctx.walls, tv_wall, ctx.polygon, avoid_windows=False, avoid_doors=True)

    sofa = None
    for sofa_key in sofa_keys:
        sofa_spec = LIVING_ROOM_SPECS[sofa_key]
        if sofa_wall:
            for along_factor in (0.5, 0.68, 0.78, 0.32, 0.22):
                sofa = engine.add_wall_aligned(
                    sofa_spec,
                    sofa_wall.id,
                    sofa_wall.length * along_factor,
                    layer="primary",
                    margin=0.02,
                    front_target="tv",
                )
                if sofa is not None:
                    break
        if sofa is not None:
            break

    # If wall-aligned sofa does not work, place it in front of TV at a controlled distance.
    if sofa is None and tv_stand:
        tv_pos = tv_stand.get("position_m") or [ctx.centroid.x, ctx.centroid.y, 0.0]
        tv_yaw = as_float(tv_stand.get("yaw_deg"), 0.0)
        _, tv_forward = local_axes_from_yaw(tv_yaw)
        tv_right, _ = local_axes_from_yaw(tv_yaw)
        for sofa_key in sofa_keys:
            sofa_spec = LIVING_ROOM_SPECS[sofa_key]
            for distance in (min(max(1.45, ctx.min_side_m * 0.48), 2.6), 1.7, 1.2):
                for lateral in (0.0, 0.35, -0.35):
                    center = Vec2(as_float(tv_pos[0]), as_float(tv_pos[1])) + tv_forward * distance + tv_right * lateral
                    center = clamp_center_inside_room_for_aabb(center, sofa_spec.size_m, polygon=ctx.polygon, yaw_deg=tv_yaw + 180.0, margin=0.04)
                    sofa = engine.add_item(
                        sofa_spec,
                        center,
                        tv_yaw + 180.0,
                        layer="primary",
                        margin=0.02,
                        front_target=tv_stand.get("id") if tv_stand else "room_center",
                    )
                    if sofa is not None:
                        break
                if sofa is not None:
                    break
            if sofa is not None:
                break

    if sofa is None:
        for sofa_key in ("sofa_2",):
            sofa_spec = LIVING_ROOM_SPECS[sofa_key]
            center = clamp_center_inside_room_for_aabb(ctx.centroid, sofa_spec.size_m, polygon=ctx.polygon, yaw_deg=0.0, margin=0.04)
            sofa = engine.add_item(
                sofa_spec,
                center,
                0.0,
                layer="primary",
                margin=0.01,
                ignore_door_clearance=True,
                ignore_window_clearance=True,
                front_target="room_center",
                extra_meta={"required_fallback": True},
            )
            if sofa:
                break
    if sofa is None:
        return engine.placements, {"generator": "living_room_generator", "status": "sofa_rejected", "rejected": engine.rejected}

    # Coffee table and rug are tied to sofa.
    coffee = engine.add_near(
        sofa,
        LIVING_ROOM_SPECS["coffee_table"],
        local_offset_xy=(0.0, sofa_spec.size_m[1] * 0.5 + LIVING_ROOM_SPECS["coffee_table"].size_m[1] * 0.5 + 0.28),
        allow_collision=False,
        layer="primary",
        front_target=sofa.get("id"),
    )
    engine.add_near(
        sofa,
        LIVING_ROOM_SPECS["rug"],
        local_offset_xy=(0.0, sofa_spec.size_m[1] * 0.55),
        allow_collision=True,
        layer="textile",
    )

    if coffee:
        if density_rank(density) >= 2:
            engine.add_on_top(coffee, LIVING_ROOM_SPECS["decor_tray"], local_offset_xy=(0.0, 0.0))
            engine.add_on_top(coffee, LIVING_ROOM_SPECS["decor_books"], local_offset_xy=(-0.25, 0.07))
        if density_rank(density) >= 3:
            engine.add_on_top(coffee, LIVING_ROOM_SPECS["decor_vase"], local_offset_xy=(0.28, -0.08))
            for i, (dx, dy) in enumerate([(-0.22, -0.15), (0.0, 0.16), (0.24, 0.13), (0.12, -0.18)], start=1):
                engine.add_on_top(
                    coffee,
                    LIVING_ROOM_SPECS["decor_books"],
                    local_offset_xy=(dx, dy),
                    name=f"Coffee table book stack {i}",
                )
            remote_spec = ObjectSpec("tv_accessory", "Remote control", (0.18, 0.06, 0.025), "electronics", allow_collision=True)
            engine.add_on_top(coffee, remote_spec, local_offset_xy=(0.35, 0.12), name="Remote control")

    # Side tables, lamps, soft decor.
    if density_rank(density) >= 2:
        side_offset = sofa_spec.size_m[0] * 0.5 + LIVING_ROOM_SPECS["side_table"].size_m[0] * 0.5 + 0.12
        side_l = engine.add_near(sofa, LIVING_ROOM_SPECS["side_table"], local_offset_xy=(-side_offset, 0.05), layer="secondary")
        if side_l:
            side_l.setdefault("meta", {})["front_target"] = sofa.get("id")
        side_r = engine.add_near(sofa, LIVING_ROOM_SPECS["side_table"], local_offset_xy=(side_offset, 0.05), layer="secondary")
        if side_r:
            side_r.setdefault("meta", {})["front_target"] = sofa.get("id")
        if side_l:
            engine.add_on_top(side_l, LIVING_ROOM_SPECS["table_lamp"], local_offset_xy=(0.0, 0.0))
            if density_rank(density) >= 3:
                engine.add_on_top(side_l, LIVING_ROOM_SPECS["decor_books"], local_offset_xy=(-0.08, 0.08), name="Side table books left")
                engine.add_on_top(side_l, LIVING_ROOM_SPECS["decor_vase"], local_offset_xy=(0.10, -0.06), name="Side table vase left")
        if side_r and density_rank(density) >= 3:
            engine.add_on_top(side_r, LIVING_ROOM_SPECS["plant_small"], local_offset_xy=(0.0, 0.0))
            engine.add_on_top(side_r, LIVING_ROOM_SPECS["decor_books"], local_offset_xy=(0.08, 0.08), name="Side table books right")
            engine.add_on_top(side_r, LIVING_ROOM_SPECS["decor_tray"], local_offset_xy=(-0.10, -0.06), name="Side table tray right")
        engine.add_corner_object(LIVING_ROOM_SPECS["floor_lamp"], preferred_index=0)
        engine.add_corner_object(LIVING_ROOM_SPECS["plant_large"], preferred_index=1)

    if density_rank(density) >= 3:
        for i, x in enumerate([-0.78, -0.52, -0.26, 0.0, 0.26, 0.52, 0.78]):
            engine.add_on_top(sofa, LIVING_ROOM_SPECS["pillow"], local_offset_xy=(x, -sofa_spec.size_m[1] * 0.22), name=f"Sofa pillow {i + 1}")
        engine.add_on_top(sofa, LIVING_ROOM_SPECS["blanket"], local_offset_xy=(sofa_spec.size_m[0] * 0.22, 0.0), name="Sofa throw blanket")
        engine.add_on_top(sofa, LIVING_ROOM_SPECS["blanket"], local_offset_xy=(-sofa_spec.size_m[0] * 0.22, 0.03), name="Layered sofa blanket")

    # Bookcases and console on free walls.
    if density_rank(density) >= 2:
        used_ids = {tv_wall.id, sofa_wall.id if sofa_wall else ""}
        bookshelves: list[dict[str, Any]] = []
        for wall in sorted(ctx.walls, key=lambda w: w.length, reverse=True):
            if wall.id in used_ids or wall.has_door:
                continue
            if wall.length >= 1.7:
                left = engine.add_wall_aligned(LIVING_ROOM_SPECS["bookshelf"], wall.id, wall.length * 0.25, layer="storage", margin=0.02)
                right = None
                if density_rank(density) >= 3 and wall.length >= 2.8:
                    right = engine.add_wall_aligned(LIVING_ROOM_SPECS["bookshelf"], wall.id, wall.length * 0.75, layer="storage", margin=0.02)
                bookshelves.extend([x for x in (left, right) if x])
                if left or right:
                    break
        if density_rank(density) >= 3:
            shelf_book = ObjectSpec("decor_books", "Bookshelf book stack", (0.22, 0.16, 0.18), "decor", allow_collision=True)
            shelf_box = ObjectSpec("decor_box", "Bookshelf storage box", (0.26, 0.18, 0.16), "decor", allow_collision=True)
            shelf_vase = ObjectSpec("decor_vase", "Bookshelf vase", (0.16, 0.16, 0.28), "decor", allow_collision=True)
            for shelf_index, shelf in enumerate(bookshelves, start=1):
                for i, (dx, dy, spec) in enumerate([
                    (-0.22, -0.06, shelf_book),
                    (0.02, -0.06, shelf_book),
                    (0.24, -0.06, shelf_box),
                    (-0.14, 0.08, shelf_vase),
                    (0.18, 0.08, shelf_book),
                ], start=1):
                    engine.add_on_top(shelf, spec, local_offset_xy=(dx, dy), name=f"Bookshelf {shelf_index} item {i}")

    if density_rank(density) >= 3:
        for wall in sorted(ctx.walls, key=lambda w: w.length, reverse=True):
            if wall.id == tv_wall.id or wall.has_door:
                continue
            console = engine.add_wall_aligned(LIVING_ROOM_SPECS["console_table"], wall.id, wall.length * 0.5, layer="secondary", margin=0.02)
            if console:
                engine.add_on_top(console, LIVING_ROOM_SPECS["decor_vase"], local_offset_xy=(-0.25, 0.0))
                engine.add_on_top(console, LIVING_ROOM_SPECS["decor_books"], local_offset_xy=(0.2, 0.0))
                engine.add_on_top(console, LIVING_ROOM_SPECS["decor_tray"], local_offset_xy=(0.0, 0.08), name="Console decorative tray")
                engine.add_wall_art(wall.id, wall.length * 0.5, LIVING_ROOM_SPECS["wall_art"], z_center=1.55)
                break

        for wall in sorted(ctx.walls, key=lambda w: w.length, reverse=True):
            if wall.has_door:
                continue
            art_count = 4 if wall.length >= 4.0 else 3
            for i in range(art_count):
                engine.add_wall_art(
                    wall.id,
                    wall.length * (i + 1) / (art_count + 1),
                    LIVING_ROOM_SPECS["wall_art"],
                    z_center=1.42 + 0.1 * (i % 2),
                    name=f"Living room wall art {wall.id}-{i + 1}",
                )

        if tv_stand:
            speaker_spec = ObjectSpec("tv_accessory", "Compact speaker", (0.18, 0.16, 0.22), "electronics", allow_collision=True)
            box_spec = ObjectSpec("tv_accessory", "Media box", (0.32, 0.22, 0.07), "electronics", allow_collision=True)
            for i, (dx, spec) in enumerate([(-0.55, speaker_spec), (0.55, speaker_spec), (0.0, box_spec)], start=1):
                engine.add_on_top(tv_stand, spec, local_offset_xy=(dx, 0.0), name=f"TV accessory {i}")

    # Armchairs around coffee table for medium/large rooms.
    if density_rank(density) >= 2 and ctx.area_m2 >= 15.0 and coffee:
        coffee_pos = coffee.get("position_m") or [ctx.centroid.x, ctx.centroid.y, 0.0]
        sofa_yaw = as_float(sofa.get("yaw_deg"), 0.0)
        ux, uy = local_axes_from_yaw(sofa_yaw)
        center_base = Vec2(as_float(coffee_pos[0]), as_float(coffee_pos[1]))
        for side, sign in [("left", -1.0), ("right", 1.0)]:
            center = center_base + ux * (sign * 1.25)
            item = engine.add_item(
                LIVING_ROOM_SPECS["armchair"],
                center,
                sofa_yaw + sign * 70.0,
                layer="secondary",
                extra_meta={"seating_group_side": side, "front_target": coffee.get("id")},
            )
            if density_rank(density) < 3 and item:
                break

    # Optional dining zone in large rooms.
    if density_rank(density) >= 3 and ctx.area_m2 >= 22.0:
        center = Vec2(
            ctx.bounds[0] + ctx.width_m * 0.25,
            ctx.bounds[1] + ctx.depth_m * 0.25,
        )
        table = engine.add_item(LIVING_ROOM_SPECS["dining_table"], center, 0.0, layer="secondary", front_target="room_center")
        if table:
            for idx, (dx, dy, yaw) in enumerate([(0.0, -0.72, 0.0), (0.0, 0.72, 180.0), (-0.95, 0.0, 90.0), (0.95, 0.0, 270.0)]):
                engine.add_near(
                    table,
                    LIVING_ROOM_SPECS["dining_chair"],
                    local_offset_xy=(dx, dy),
                    yaw_delta_deg=yaw,
                    name=f"Dining chair {idx + 1}",
                    layer="secondary",
                    front_target=table.get("id"),
                )

    engine.add_ceiling_light()

    report = {
        "generator": "living_room_generator",
        "archetype": engine.archetype,
        "tv_wall_id": tv_wall.id,
        "sofa_wall_id": sofa_wall.id if sofa_wall else None,
        "density": density,
        "rejected": engine.rejected,
    }
    return engine.placements, report
