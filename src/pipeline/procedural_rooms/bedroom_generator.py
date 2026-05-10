from __future__ import annotations

import random
from typing import Any

from .geometry import Vec2, choose_longest_wall, choose_wall_most_opposite, wall_inside_normal
from .object_specs import BEDROOM_SPECS, Density, ObjectSpec, density_rank
from .placement_engine import PlacementEngine
from .room_context import RoomContext


def _bed_key(ctx: RoomContext) -> str:
    if ctx.area_m2 < 7.0 or ctx.min_side_m < 2.4:
        return "single_bed"
    if ctx.area_m2 < 13.0 or ctx.min_side_m < 3.0:
        return "double_bed"
    return "queen_bed"


def generate_bedroom(ctx: RoomContext, *, density: Density, seed: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    engine = PlacementEngine(
        ctx=ctx,
        rng=rng,
        source_name="procedural_room_stage",
        generator_name="bedroom_generator",
        archetype="bed_against_long_wall",
    )

    bed_spec = BEDROOM_SPECS[_bed_key(ctx)]
    bed_wall = choose_longest_wall(ctx.walls, avoid_windows=True, avoid_doors=True, min_length=bed_spec.size_m[0] + 0.35)
    if bed_wall is None:
        bed_wall = choose_longest_wall(ctx.walls, avoid_windows=False, avoid_doors=True)
    if bed_wall is None:
        return [], {"generator": "bedroom_generator", "status": "no_wall"}

    bed = engine.add_wall_aligned(
        bed_spec,
        bed_wall.id,
        bed_wall.length * 0.5,
        ignore_door_clearance=False,
        extra_meta={"role": "main_bed"},
    )

    if bed is None:
        # Try a smaller bed before giving up.
        bed_spec = BEDROOM_SPECS["single_bed"]
        bed = engine.add_wall_aligned(
            bed_spec,
            bed_wall.id,
            bed_wall.length * 0.5,
            ignore_door_clearance=False,
            extra_meta={"role": "main_bed", "fallback": "single_bed"},
        )

    if bed is None:
        return [], {"generator": "bedroom_generator", "status": "bed_rejected", "rejected": engine.rejected}

    bed_width = bed_spec.size_m[0]
    night_spec = BEDROOM_SPECS["nightstand"]
    night_offset_x = bed_width * 0.5 + night_spec.size_m[0] * 0.5 + 0.08
    night_offset_y = -0.02
    left_ns = engine.add_near(
        bed,
        night_spec,
        local_offset_xy=(-night_offset_x, night_offset_y),
        allow_collision=False,
        layer="secondary",
    )
    right_ns = engine.add_near(
        bed,
        night_spec,
        local_offset_xy=(night_offset_x, night_offset_y),
        allow_collision=False,
        layer="secondary",
    )

    if density_rank(density) >= 2:
        if left_ns:
            engine.add_on_top(left_ns, BEDROOM_SPECS["table_lamp"], local_offset_xy=(0.0, 0.0))
            engine.add_on_top(left_ns, BEDROOM_SPECS["decor_books"], local_offset_xy=(0.07, -0.05))
        if right_ns:
            engine.add_on_top(right_ns, BEDROOM_SPECS["table_lamp"], local_offset_xy=(0.0, 0.0))
            engine.add_on_top(right_ns, BEDROOM_SPECS["decor_box"], local_offset_xy=(-0.06, -0.05))

    if density_rank(density) >= 3:
        # Soft decor on bed.
        pillow_xs = [-0.42, -0.14, 0.14, 0.42]
        for i, x in enumerate(pillow_xs):
            engine.add_on_top(
                bed,
                BEDROOM_SPECS["pillow"],
                local_offset_xy=(x, -bed_spec.size_m[1] * 0.35),
                name=f"Decorative pillow {i + 1}",
            )
        engine.add_on_top(
            bed,
            BEDROOM_SPECS["blanket"],
            local_offset_xy=(0.0, bed_spec.size_m[1] * 0.18),
            name="Folded blanket",
        )
        engine.add_on_top(
            bed,
            BEDROOM_SPECS["blanket"],
            local_offset_xy=(0.0, bed_spec.size_m[1] * 0.34),
            name="Layered bed throw",
        )
        for i, x in enumerate([-0.30, 0.30]):
            engine.add_on_top(
                bed,
                BEDROOM_SPECS["decor_books"],
                local_offset_xy=(x, bed_spec.size_m[1] * 0.08),
                name=f"Bedside reading stack {i + 1}",
            )

    # Rug is intentionally allowed to overlap bed.
    if density_rank(density) >= 1:
        engine.add_near(
            bed,
            BEDROOM_SPECS["rug"],
            local_offset_xy=(0.0, bed_spec.size_m[1] * 0.22),
            allow_collision=True,
            layer="textile",
        )

    # Wardrobe modules on an opposite/free wall.
    wardrobe_wall = choose_wall_most_opposite(ctx.walls, bed_wall, ctx.polygon, avoid_windows=True, avoid_doors=True)
    wardrobe_items: list[dict[str, Any]] = []
    if wardrobe_wall:
        module = BEDROOM_SPECS["wardrobe_module"]
        max_modules = 1
        if ctx.area_m2 >= 9.0:
            max_modules = 2
        if ctx.area_m2 >= 13.0:
            max_modules = 3
        if density_rank(density) >= 3 and ctx.area_m2 >= 16.0:
            max_modules = 4
        available_modules = min(max_modules, int(max(1.0, wardrobe_wall.length - 0.4) // module.size_m[0]))
        total_width = available_modules * module.size_m[0]
        start = max(module.size_m[0] * 0.5, wardrobe_wall.length * 0.5 - total_width * 0.5 + module.size_m[0] * 0.5)
        for i in range(available_modules):
            item = engine.add_wall_aligned(
                module,
                wardrobe_wall.id,
                start + i * module.size_m[0],
                name=f"Wardrobe module {i + 1}",
                category="wardrobe",
                layer="storage",
                margin=0.02,
            )
            if item:
                wardrobe_items.append(item)

    if density_rank(density) >= 3:
        box_spec = BEDROOM_SPECS["decor_box"]
        for i, wardrobe in enumerate(wardrobe_items[:4]):
            engine.add_on_top(
                wardrobe,
                box_spec,
                local_offset_xy=(0.0, 0.0),
                name=f"Wardrobe top storage box {i + 1}",
                layer="decor",
            )

    # Dresser on a remaining wall.
    if density_rank(density) >= 2:
        for wall in sorted(ctx.walls, key=lambda w: w.length, reverse=True):
            if wall.id in {bed_wall.id, wardrobe_wall.id if wardrobe_wall else ""}:
                continue
            dresser = engine.add_wall_aligned(
                BEDROOM_SPECS["dresser"],
                wall.id,
                wall.length * 0.5,
                layer="storage",
                margin=0.02,
            )
            if dresser:
                engine.add_on_top(dresser, BEDROOM_SPECS["decor_vase"], local_offset_xy=(-0.25, 0.0))
                if density_rank(density) >= 3:
                    engine.add_on_top(dresser, BEDROOM_SPECS["decor_books"], local_offset_xy=(0.18, 0.02))
                    engine.add_on_top(dresser, BEDROOM_SPECS["decor_box"], local_offset_xy=(0.42, 0.0))
                break

    # Bench at bed foot.
    if density_rank(density) >= 2 and ctx.area_m2 >= 11.0:
        engine.add_near(
            bed,
            BEDROOM_SPECS["bench"],
            local_offset_xy=(0.0, bed_spec.size_m[1] * 0.5 + BEDROOM_SPECS["bench"].size_m[1] * 0.5 + 0.12),
            allow_collision=False,
            layer="secondary",
        )

    # Optional desk for large bedrooms.
    if density_rank(density) >= 3 and ctx.area_m2 >= 14.0:
        for wall in sorted(ctx.walls, key=lambda w: w.length, reverse=True):
            if wall.id == bed_wall.id:
                continue
            desk = engine.add_wall_aligned(BEDROOM_SPECS["desk"], wall.id, wall.length * 0.28, layer="secondary", margin=0.02)
            if desk:
                engine.add_near(
                    desk,
                    BEDROOM_SPECS["chair"],
                    local_offset_xy=(0.0, BEDROOM_SPECS["desk"].size_m[1] * 0.5 + BEDROOM_SPECS["chair"].size_m[1] * 0.5 + 0.08),
                    allow_collision=False,
                )
                engine.add_on_top(desk, BEDROOM_SPECS["table_lamp"], local_offset_xy=(-0.35, 0.0))
                break

    # Wall decor above bed.
    if density_rank(density) >= 2:
        engine.add_wall_art(bed_wall.id, bed_wall.length * 0.5, BEDROOM_SPECS["wall_art"], z_center=1.55)
    if density_rank(density) >= 3:
        engine.add_wall_art(bed_wall.id, max(0.4, bed_wall.length * 0.5 - 0.7), BEDROOM_SPECS["sconce"], z_center=1.45)
        engine.add_wall_art(bed_wall.id, min(bed_wall.length - 0.4, bed_wall.length * 0.5 + 0.7), BEDROOM_SPECS["sconce"], z_center=1.45)
        for wall in sorted(ctx.walls, key=lambda w: w.length, reverse=True):
            if wall.id == bed_wall.id:
                continue
            if wall.has_door:
                continue
            art_count = 3 if wall.length >= 3.0 else 2
            for i in range(art_count):
                along = wall.length * (i + 1) / (art_count + 1)
                engine.add_wall_art(
                    wall.id,
                    along,
                    BEDROOM_SPECS["wall_art"],
                    z_center=1.45 + 0.08 * (i % 2),
                    name=f"Bedroom gallery art {wall.id}-{i + 1}",
                )

    # Plants/floor lamps in corners.
    if density_rank(density) >= 2:
        engine.add_corner_object(BEDROOM_SPECS["plant"], preferred_index=0)
    if density_rank(density) >= 3:
        engine.add_corner_object(BEDROOM_SPECS["floor_lamp"], preferred_index=1)

        # Very-high density should feel lived-in without adding more floor
        # obstacles. These objects are either on existing surfaces or mounted.
        small_vase = BEDROOM_SPECS["decor_vase"]
        small_box = BEDROOM_SPECS["decor_box"]
        for idx, parent in enumerate([x for x in [left_ns, right_ns] if x], start=1):
            engine.add_on_top(parent, small_vase, local_offset_xy=(-0.10, 0.07), name=f"Nightstand vase {idx}")
            engine.add_on_top(parent, BEDROOM_SPECS["decor_books"], local_offset_xy=(0.10, -0.07), name=f"Nightstand book pair {idx}")
            engine.add_on_top(parent, small_box, local_offset_xy=(0.05, 0.08), name=f"Nightstand small box {idx}")

        tray_spec = ObjectSpec("decor_tray", "Bedroom catch-all tray", (0.34, 0.22, 0.05), "decor", allow_collision=True)
        for parent in wardrobe_items[:2]:
            engine.add_on_top(parent, tray_spec, local_offset_xy=(0.0, 0.0), name="Wardrobe decorative tray")

    engine.add_ceiling_light()

    report = {
        "generator": "bedroom_generator",
        "archetype": engine.archetype,
        "bed_wall_id": bed_wall.id,
        "density": density,
        "rejected": engine.rejected,
    }
    return engine.placements, report
