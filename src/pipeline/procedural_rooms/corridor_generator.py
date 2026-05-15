from __future__ import annotations

import random
from typing import Any

from .geometry import Vec2, choose_longest_wall
from .object_specs import CORRIDOR_SPECS, Density, density_rank
from .placement_engine import PlacementEngine
from .room_context import RoomContext


def generate_corridor(ctx: RoomContext, *, density: Density, seed: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    engine = PlacementEngine(
        ctx=ctx,
        rng=rng,
        source_name="procedural_room_stage",
        generator_name="corridor_generator",
        archetype="linear_storage_along_wall",
    )

    main_wall = choose_longest_wall(ctx.walls, avoid_windows=True, avoid_doors=True)
    if main_wall is None:
        main_wall = choose_longest_wall(ctx.walls, avoid_windows=False, avoid_doors=True)
    if main_wall is None:
        return [], {"generator": "corridor_generator", "status": "no_wall"}

    width = ctx.min_side_m

    # Runner rug along the corridor axis. It is allowed to overlap the central clear path.
    runner = CORRIDOR_SPECS["runner_rug"]
    if ctx.width_m >= ctx.depth_m:
        rug_size = (min(ctx.width_m * 0.75, max(1.2, ctx.width_m - 0.7)), 0.75, runner.size_m[2])
        rug_yaw = 90.0
    else:
        rug_size = (0.75, min(ctx.depth_m * 0.75, max(1.2, ctx.depth_m - 0.7)), runner.size_m[2])
        rug_yaw = 0.0

    from .object_specs import ObjectSpec

    rug_spec = ObjectSpec(
        category=runner.category,
        name=runner.name,
        size_m=rug_size,
        layer=runner.layer,
        allow_collision=True,
    )
    engine.add_item(rug_spec, ctx.centroid, rug_yaw, allow_collision=True, layer="textile", ignore_door_clearance=True)

    # Furniture depth is limited by corridor width.
    if width >= 1.0:
        shoe = CORRIDOR_SPECS["shoe_cabinet"]
        if width < 1.2:
            shoe = ObjectSpec(
                category=shoe.category,
                name="Extra narrow shoe cabinet",
                size_m=(0.75, 0.24, 0.85),
                layer=shoe.layer,
            )
        shoe_item = engine.add_wall_aligned(shoe, main_wall.id, main_wall.length * 0.35, layer="storage", margin=0.02)
    else:
        shoe_item = None

    if width >= 1.25 and density_rank(density) >= 2:
        bench = CORRIDOR_SPECS["bench"]
        if width < 1.45:
            bench = ObjectSpec(
                category=bench.category,
                name="Narrow entry bench",
                size_m=(0.85, 0.30, 0.45),
                layer=bench.layer,
            )
        b = engine.add_wall_aligned(bench, main_wall.id, main_wall.length * 0.68, layer="secondary", margin=0.02)
        if b and density_rank(density) >= 3:
            engine.add_on_top(b, CORRIDOR_SPECS["storage_basket"], local_offset_xy=(-0.23, 0.0), name="Bench basket left")
            engine.add_on_top(b, CORRIDOR_SPECS["storage_basket"], local_offset_xy=(0.23, 0.0), name="Bench basket right")
            engine.add_on_top(b, CORRIDOR_SPECS["key_tray"], local_offset_xy=(0.0, 0.0), name="Bench key tray")

    if width >= 1.5 and density_rank(density) >= 2:
        engine.add_wall_aligned(CORRIDOR_SPECS["coat_rack"], main_wall.id, main_wall.length * 0.12, layer="storage", margin=0.02)

    if width >= 1.7 and density_rank(density) >= 3:
        engine.add_wall_aligned(CORRIDOR_SPECS["wardrobe_narrow"], main_wall.id, main_wall.length * 0.88, layer="storage", margin=0.02)

    # Opposite wall receives mirror/hooks/art because they are thin and safe.
    opposite = None
    for wall in sorted(ctx.walls, key=lambda w: w.length, reverse=True):
        if wall.id != main_wall.id and not wall.has_door:
            opposite = wall
            break
    if opposite is None:
        opposite = main_wall

    engine.add_wall_art(opposite.id, opposite.length * 0.5, CORRIDOR_SPECS["mirror"], z_center=1.25, category="mirror", name="Full-height mirror")
    if density_rank(density) >= 2:
        engine.add_wall_art(main_wall.id, main_wall.length * 0.52, CORRIDOR_SPECS["wall_hooks"], z_center=1.55, category="wall_hooks", name="Wall hooks")

    if density_rank(density) >= 3:
        # Series of small wall art along the opposite wall.
        count = 5 if opposite.length >= 4.5 else (4 if opposite.length >= 2.4 else 3)
        for i in range(count):
            along = opposite.length * (i + 1) / (count + 1)
            engine.add_wall_art(opposite.id, along, CORRIDOR_SPECS["wall_art"], z_center=1.55, name=f"Corridor wall art {i + 1}")

        hook_count = 4 if main_wall.length >= 3.5 else 3
        for i in range(hook_count):
            engine.add_wall_art(
                main_wall.id,
                main_wall.length * (i + 1) / (hook_count + 1),
                CORRIDOR_SPECS["wall_hooks"],
                z_center=1.62,
                category="wall_hooks",
                name=f"Wall hook rail {i + 1}",
            )

        # Small console and decor if corridor is wide enough.
        if width >= 1.35:
            console = engine.add_wall_aligned(CORRIDOR_SPECS["console_table"], opposite.id, opposite.length * 0.28, layer="secondary", margin=0.02)
            if console:
                engine.add_on_top(console, CORRIDOR_SPECS["key_tray"], local_offset_xy=(0.0, 0.0))
                engine.add_on_top(console, CORRIDOR_SPECS["small_plant"], local_offset_xy=(0.25, 0.0))
                engine.add_on_top(console, CORRIDOR_SPECS["storage_basket"], local_offset_xy=(-0.25, 0.0), name="Console small basket")
        if shoe_item:
            engine.add_on_top(shoe_item, CORRIDOR_SPECS["key_tray"], local_offset_xy=(-0.18, 0.0), name="Shoe cabinet key tray")
            engine.add_on_top(shoe_item, CORRIDOR_SPECS["small_plant"], local_offset_xy=(0.18, 0.0), name="Shoe cabinet small plant")
        engine.add_corner_object(CORRIDOR_SPECS["umbrella_stand"], preferred_index=0)
        engine.add_corner_object(CORRIDOR_SPECS["storage_basket"], preferred_index=1, name="Corner storage basket")

    # Light series: a corridor usually needs several ceiling lights.
    lights = 1
    if ctx.max_side_m > 3.5:
        lights = 2
    if ctx.max_side_m > 5.5:
        lights = 3
    if density_rank(density) >= 3:
        lights = max(lights, 3 if ctx.max_side_m >= 4.0 else 2)

    for i in range(lights):
        light = engine.add_ceiling_light(name=f"Corridor ceiling light {i + 1}")
        if lights > 1:
            if ctx.width_m >= ctx.depth_m:
                x = ctx.bounds[0] + ctx.width_m * (i + 1) / (lights + 1)
                y = ctx.centroid.y
            else:
                x = ctx.centroid.x
                y = ctx.bounds[1] + ctx.depth_m * (i + 1) / (lights + 1)
            engine.move_item_xy(light, x, y)

    report = {
        "generator": "corridor_generator",
        "archetype": engine.archetype,
        "main_wall_id": main_wall.id,
        "density": density,
        "corridor_width_m": width,
        "rejected": engine.rejected,
    }
    return engine.placements, report
