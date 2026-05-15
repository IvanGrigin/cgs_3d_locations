from __future__ import annotations

import random
from typing import Any

from .geometry import Vec2, aabb_from_box, choose_longest_wall, choose_wall_most_opposite, wall_inside_normal, yaw_for_local_y_to_vector
from .object_specs import BEDROOM_SPECS, Density, ObjectSpec, density_rank
from .placement_engine import PlacementEngine
from .room_context import RoomContext


def _spec_with_size(base: ObjectSpec, size_m: tuple[float, float, float], *, name: str | None = None) -> ObjectSpec:
    return ObjectSpec(
        base.category,
        name or base.name,
        size_m,
        base.layer,
        mount_type=base.mount_type,
        replace_with_supplier=base.replace_with_supplier,
        allow_collision=base.allow_collision,
        requires_access=base.requires_access,
        support_surface=base.support_surface,
        front_target_hint=base.front_target_hint,
    )


def _wall_room_depth(ctx: RoomContext, wall: Any) -> float:
    normal = wall_inside_normal(wall, ctx.polygon)
    origin = wall.point_at(wall.length * 0.5)
    distances = [(point - origin).dot(normal) for point in ctx.polygon]
    positive = [value for value in distances if value > 0.01]
    return max(positive) if positive else ctx.max_side_m


def _bed_side_margin(ctx: RoomContext, bed_width: float) -> float:
    if bed_width >= 1.8:
        return 0.25
    if bed_width >= 1.4:
        return 0.08 if ctx.min_side_m <= 2.15 else 0.14
    return 0.08


def _bed_foot_clearance(ctx: RoomContext) -> float:
    if ctx.min_side_m <= 2.15 or ctx.area_m2 < 7.5:
        return 0.35
    if ctx.min_side_m <= 2.7 or ctx.is_long_narrow:
        return 0.45
    return 0.60


def _bed_size_options(ctx: RoomContext) -> list[str]:
    options: list[str] = []
    if ctx.area_m2 >= 15.0 and ctx.min_side_m >= 3.2:
        options.append("queen_bed")
    if ctx.area_m2 >= 10.5 and ctx.min_side_m >= 2.8:
        options.append("double_bed")
    if ctx.area_m2 >= 6.0 and ctx.min_side_m >= 1.9:
        options.append("compact_double_bed")
    options.append("single_bed")
    return options


def _bed_fits_wall(ctx: RoomContext, wall: Any, spec: ObjectSpec) -> bool:
    bed_width, bed_depth, _ = spec.size_m
    if wall.length < bed_width + 2.0 * _bed_side_margin(ctx, bed_width):
        return False
    if _wall_room_depth(ctx, wall) < bed_depth + _bed_foot_clearance(ctx):
        return False
    return any(_bed_clearance_ok(ctx, _wall_aligned_aabb(ctx, wall, spec, along)) for along in _bed_along_candidates(ctx, wall, spec))


def _wall_aligned_aabb(ctx: RoomContext, wall: Any, spec: ObjectSpec, along_m: float) -> Any:
    normal = wall_inside_normal(wall, ctx.polygon)
    center = wall.point_at(along_m) + normal * (spec.size_m[1] * 0.5)
    yaw = yaw_for_local_y_to_vector(normal)
    return aabb_from_box([center.x, center.y, spec.size_m[2] * 0.5], spec.size_m, yaw)


def _bed_along_candidates(ctx: RoomContext, wall: Any, spec: ObjectSpec) -> list[float]:
    width = spec.size_m[0]
    center = wall.length * 0.5
    candidates = [center]
    if ctx.min_side_m <= 2.1 and wall.length - width >= 0.45:
        edge = 0.02
        candidates = [
            width * 0.5 + edge,
            wall.length - width * 0.5 - edge,
            center,
        ]
    out: list[float] = []
    for value in candidates:
        value = max(width * 0.5, min(wall.length - width * 0.5, value))
        if all(abs(value - existing) > 0.01 for existing in out):
            out.append(value)
    return out


def _bed_clearance_ok(ctx: RoomContext, aabb: Any) -> bool:
    for zone in ctx.window_clearance_zones:
        if aabb.intersects_xy(zone, margin=0.0):
            return False
    if ctx.min_side_m <= 2.1:
        room_min_x, _room_min_y, room_max_x, _room_max_y = ctx.bounds
        left_gap = aabb.x_min - room_min_x
        right_gap = room_max_x - aabb.x_max
        if max(left_gap, right_gap) < 0.45:
            return False
    return True


def _select_bed_plan(ctx: RoomContext) -> tuple[ObjectSpec, Any | None, float | None]:
    for key in _bed_size_options(ctx):
        spec = BEDROOM_SPECS[key]
        bed_wall = _preferred_bed_wall(ctx, spec.size_m[0])
        if bed_wall is not None and _bed_fits_wall(ctx, bed_wall, spec):
            return spec, bed_wall, _select_bed_along(ctx, bed_wall, spec)
        for wall in sorted(ctx.walls, key=lambda item: (item.has_door, item.has_window, -item.length)):
            if _bed_fits_wall(ctx, wall, spec):
                return spec, wall, _select_bed_along(ctx, wall, spec)
    return BEDROOM_SPECS["single_bed"], None, None


def _select_bed_along(ctx: RoomContext, wall: Any, spec: ObjectSpec) -> float:
    for along in _bed_along_candidates(ctx, wall, spec):
        if _bed_clearance_ok(ctx, _wall_aligned_aabb(ctx, wall, spec, along)):
            return along
    return wall.length * 0.5


def _tiny_bed_wall(ctx: RoomContext, bed_width: float) -> Any | None:
    candidates = [
        wall
        for wall in ctx.walls
        if not wall.has_door and wall.length >= bed_width + 0.25
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda wall: (wall.length, wall.has_window))


def _wall_name(ctx: RoomContext, wall_id: str) -> str:
    for wall in ctx.room.get("walls") or []:
        if isinstance(wall, dict) and str(wall.get("id") or "") == wall_id:
            return str(wall.get("name") or "").strip().lower()
    return ""


def _door_reference_point(ctx: RoomContext) -> Vec2 | None:
    wall_by_id = {wall.id: wall for wall in ctx.walls}
    points: list[Vec2] = []
    for door in ctx.doors:
        wall = wall_by_id.get(str(door.get("wall_id") or ""))
        if wall is None:
            continue
        width = float(door.get("width") or 0.8)
        s = float(door.get("s") or 0.0)
        points.append(wall.point_at(s + width * 0.5))
    if not points:
        return None
    return Vec2(sum(p.x for p in points) / len(points), sum(p.y for p in points) / len(points))


def _preferred_bed_wall(ctx: RoomContext, bed_width: float) -> Any | None:
    usable = [
        wall
        for wall in ctx.walls
        if not wall.has_door and wall.length >= bed_width + 0.18
    ]
    if not usable:
        return None

    named = [
        wall
        for wall in usable
        if any(token in _wall_name(ctx, wall.id) for token in ("far", "headboard", "bed"))
    ]
    if named:
        return max(named, key=lambda wall: wall.length)

    door_point = _door_reference_point(ctx)
    if ctx.is_long_narrow or ctx.min_side_m <= 2.7:
        short_candidates = [
            wall
            for wall in usable
            if wall.length <= ctx.min_side_m + 0.35 and not wall.has_window
        ]
        if short_candidates:
            if door_point is not None:
                return max(short_candidates, key=lambda wall: (wall.point_at(wall.length * 0.5) - door_point).length())
            return max(short_candidates, key=lambda wall: wall.length)

    if door_point is not None:
        return max(usable, key=lambda wall: (wall.point_at(wall.length * 0.5) - door_point).length())
    return choose_longest_wall(usable, avoid_windows=True, avoid_doors=True, min_length=bed_width + 0.18)


def _add_window_curtains(engine: PlacementEngine, ctx: RoomContext) -> None:
    for wall in ctx.walls:
        if not wall.has_window:
            continue
        width = max(0.75, min(wall.length - 0.12, BEDROOM_SPECS["curtains"].size_m[0]))
        spec = ObjectSpec(
            "curtain",
            "Pale green curtains along window wall",
            (width, 0.06, 2.15),
            "wall_decor",
            mount_type="wall",
            replace_with_supplier=False,
            allow_collision=True,
        )
        engine.add_wall_art(wall.id, wall.length * 0.5, spec, z_center=1.45, name=spec.name, category="curtain", layer="wall_decor")


def _add_foreground_pouf(engine: PlacementEngine, ctx: RoomContext) -> None:
    spec = BEDROOM_SPECS["stool_pouf"]
    x_min, y_min, x_max, y_max = ctx.bounds
    candidates = [
        Vec2(x_min + 0.45, y_min + 0.55),
        Vec2((x_min + x_max) * 0.5, y_min + 0.95),
        Vec2((x_min + x_max) * 0.5, y_min + 1.25),
        Vec2(x_min + 0.55, y_min + 1.05),
    ]
    for center in candidates:
        item = engine.add_item(
            spec,
            center,
            0.0,
            name="Small round pale green pouf stool",
            category="stool",
            layer="secondary",
            front_target="room_center",
        )
        if item:
            constraints = item.setdefault("constraints", {})
            if isinstance(constraints, dict):
                constraints.update({"style": "soft classic", "color": "pale green olive", "materials": "soft fabric"})
            return


def _add_small_console(engine: PlacementEngine, ctx: RoomContext, bed_wall_id: str) -> dict[str, Any] | None:
    spec = BEDROOM_SPECS["console_dresser"]
    preferred = sorted(
        [wall for wall in ctx.walls if wall.id != bed_wall_id and not wall.has_window],
        key=lambda wall: (
            0 if "entrance" in _wall_name(ctx, wall.id) else 1,
            1 if wall.has_door else 0,
            -wall.length,
        ),
    )
    for wall in preferred:
        for along_ratio in (0.28, 0.45, 0.65):
            item = engine.add_wall_aligned(
                spec,
                wall.id,
                wall.length * along_ratio,
                name="Small white console dresser",
                category="dresser",
                layer="storage",
                margin=0.02,
                front_target="room_center",
            )
            if item:
                constraints = item.setdefault("constraints", {})
                if isinstance(constraints, dict):
                    constraints.update({"style": "soft classic", "color": "white cream", "materials": "matte painted wood"})
                return item
    return None


def _nightstand_plan_for_space(ctx: RoomContext, side_space: float) -> tuple[ObjectSpec, float] | None:
    gap = 0.05 if ctx.min_side_m <= 2.1 else 0.04
    if side_space < 0.30 + gap:
        return None
    width = min(0.60, max(0.30, side_space - gap))
    depth = min(0.44, max(0.30, width * 0.85))
    base = BEDROOM_SPECS["nightstand"]
    spec = _spec_with_size(
        base,
        (width, depth, 0.52 if width <= 0.36 else 0.55),
        name="Compact nightstand" if width <= 0.36 else "Nightstand",
    )
    return spec, gap


def _item_intersects_window_clearance(ctx: RoomContext, item: dict[str, Any] | None) -> bool:
    if not item:
        return False
    aabb = item.get("aabb") if isinstance(item.get("aabb"), dict) else {}
    try:
        x_min = float(aabb.get("x_min"))
        x_max = float(aabb.get("x_max"))
        y_min = float(aabb.get("y_min"))
        y_max = float(aabb.get("y_max"))
    except Exception:
        return False
    for zone in ctx.window_clearance_zones:
        if not (x_max <= zone.x_min or zone.x_max <= x_min or y_max <= zone.y_min or zone.y_max <= y_min):
            return True
    return False


def _remove_placement(engine: PlacementEngine, item: dict[str, Any] | None) -> None:
    if not item:
        return
    item_id = item.get("id")
    engine.placements = [candidate for candidate in engine.placements if candidate.get("id") != item_id]


def generate_bedroom(ctx: RoomContext, *, density: Density, seed: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    engine = PlacementEngine(
        ctx=ctx,
        rng=rng,
        source_name="procedural_room_stage",
        generator_name="bedroom_generator",
        archetype="bed_against_long_wall",
    )

    bed_spec, bed_wall, bed_along_m = _select_bed_plan(ctx)
    if bed_wall is None and ctx.min_side_m <= 2.1:
        bed_wall = _tiny_bed_wall(ctx, bed_spec.size_m[0])
    if bed_wall is None:
        bed_wall = choose_longest_wall(ctx.walls, avoid_windows=True, avoid_doors=True, min_length=bed_spec.size_m[0] + 0.35)
    if bed_wall is None:
        bed_wall = choose_longest_wall(ctx.walls, avoid_windows=False, avoid_doors=True)
    if bed_wall is None:
        return [], {"generator": "bedroom_generator", "status": "no_wall"}

    bed = engine.add_wall_aligned(
        bed_spec,
        bed_wall.id,
        bed_along_m if bed_along_m is not None else bed_wall.length * 0.5,
        ignore_door_clearance=False,
        ignore_window_clearance=True,
        front_target="room_center",
        extra_meta={
            "role": "main_bed",
            "bed_size_policy": {
                "selected_width_m": bed_spec.size_m[0],
                "minimum_widths_m": {"single": 0.90, "one_and_half": 1.40, "double": 1.80},
                "wall_fraction": round(bed_spec.size_m[0] / max(bed_wall.length, 0.01), 4),
                "side_margin_m": round((bed_wall.length - bed_spec.size_m[0]) * 0.5, 4),
                "wall_along_fraction": round((bed_along_m if bed_along_m is not None else bed_wall.length * 0.5) / max(bed_wall.length, 0.01), 4),
                "foot_clearance_policy_m": _bed_foot_clearance(ctx),
            },
        },
    )

    if bed is None:
        # Try a smaller bed before giving up.
        bed_spec = BEDROOM_SPECS["single_bed"]
        bed = engine.add_wall_aligned(
            bed_spec,
            bed_wall.id,
            bed_wall.length * 0.5,
            ignore_door_clearance=False,
            ignore_window_clearance=True,
            front_target="room_center",
            extra_meta={"role": "main_bed", "fallback": "single_bed", "bed_size_policy": {"selected_width_m": bed_spec.size_m[0]}},
        )

    if bed is None:
        return [], {"generator": "bedroom_generator", "status": "bed_rejected", "rejected": engine.rejected}

    bed_meta = bed.get("meta") if isinstance(bed.get("meta"), dict) else {}
    bed_wall_along_m = float(bed_meta.get("wall_along_m") or bed_wall.length * 0.5)
    bed_width = bed_spec.size_m[0]
    bed_constraints = bed.setdefault("constraints", {})
    if isinstance(bed_constraints, dict):
        bed_constraints.update({"style": "soft classic", "color": "cream beige pale green", "materials": "fabric padded upholstery light wood"})

    mural_base = BEDROOM_SPECS["mural"]
    mural_spec = _spec_with_size(
        mural_base,
        (max(0.75, min(mural_base.size_m[0], bed_wall.length - 0.12)), mural_base.size_m[1], mural_base.size_m[2]),
    )
    engine.add_wall_art(
        bed_wall.id,
        bed_wall_along_m,
        mural_spec,
        z_center=1.42,
        name="Soft Italian street garden mural accent wall",
        category="wall_art",
        layer="wall_decor",
    )
    headboard_spec = ObjectSpec(
        "headboard",
        "Cream padded square panel headboard",
        (min(bed_wall.length - 0.18, bed_width + 0.12), 0.08, 1.08),
        "secondary",
        mount_type="wall",
        replace_with_supplier=False,
        allow_collision=True,
    )
    engine.add_wall_art(
        bed_wall.id,
        bed_wall_along_m,
        headboard_spec,
        z_center=0.62,
        name=headboard_spec.name,
        category="headboard",
        layer="wall_decor",
    )

    start_side_space = max(0.0, bed_wall_along_m - bed_width * 0.5)
    end_side_space = max(0.0, bed_wall.length - (bed_wall_along_m + bed_width * 0.5))
    start_night_plan = _nightstand_plan_for_space(ctx, start_side_space)
    end_night_plan = _nightstand_plan_for_space(ctx, end_side_space)
    night_offset_y = -0.02
    left_ns = None
    right_ns = None
    if start_night_plan:
        night_spec, night_gap = start_night_plan
        night_offset_x = bed_width * 0.5 + night_spec.size_m[0] * 0.5 + night_gap
        left_ns = engine.add_near(
            bed,
            night_spec,
            local_offset_xy=(-night_offset_x, night_offset_y),
            allow_collision=False,
            layer="secondary",
            ignore_window_clearance=True,
            front_target=bed.get("id"),
        )
    if end_night_plan:
        night_spec, night_gap = end_night_plan
        night_offset_x = bed_width * 0.5 + night_spec.size_m[0] * 0.5 + night_gap
        right_ns = engine.add_near(
            bed,
            night_spec,
            local_offset_xy=(night_offset_x, night_offset_y),
            allow_collision=False,
            layer="secondary",
            ignore_window_clearance=True,
            front_target=bed.get("id"),
        )
    if any(wall.has_window for wall in ctx.walls):
        if _item_intersects_window_clearance(ctx, left_ns):
            _remove_placement(engine, left_ns)
            left_ns = None
        if _item_intersects_window_clearance(ctx, right_ns):
            _remove_placement(engine, right_ns)
            right_ns = None
    for ns in (left_ns, right_ns):
        if ns:
            constraints = ns.setdefault("constraints", {})
            if isinstance(constraints, dict):
                constraints.update({"style": "soft classic", "color": "white cream light wood", "materials": "matte painted wood"})

    if density_rank(density) >= 2:
        if left_ns:
            engine.add_on_top(left_ns, BEDROOM_SPECS["table_lamp"], local_offset_xy=(0.0, 0.0))
            engine.add_on_top(left_ns, BEDROOM_SPECS["decor_books"], local_offset_xy=(0.07, -0.05))
        if right_ns:
            engine.add_on_top(right_ns, BEDROOM_SPECS["table_lamp"], local_offset_xy=(0.0, 0.0))
            engine.add_on_top(right_ns, BEDROOM_SPECS["decor_box"], local_offset_xy=(-0.06, -0.05))

        # High-density bedrooms should not render as a bare mattress.
        for i, x in enumerate((-0.28, 0.28), start=1):
            engine.add_on_top(
                bed,
                BEDROOM_SPECS["pillow"],
                local_offset_xy=(x, -bed_spec.size_m[1] * 0.35),
                name=f"Bed pillow {i}",
            )
        engine.add_on_top(
            bed,
            BEDROOM_SPECS["blanket"],
            local_offset_xy=(0.0, bed_spec.size_m[1] * 0.18),
            name="Bed blanket",
        )

    if density_rank(density) >= 3:
        # Extra soft decor on top of the high-density bedding baseline.
        pillow_xs = [-0.42, 0.42]
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
            local_offset_xy=(0.0, bed_spec.size_m[1] * 0.34),
            name="Layered bed throw",
        )
        # Product-asset/TRELLIS runs should not place hard decorative stacks on
        # the mattress; they read as stray blocks after bed replacement.

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
    wardrobe_wall = None if (ctx.is_long_narrow or ctx.min_side_m <= 2.7 or ctx.area_m2 <= 10.5) else choose_wall_most_opposite(ctx.walls, bed_wall, ctx.polygon, avoid_windows=True, avoid_doors=True)
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
                front_target="room_center",
            )
            if item:
                wardrobe_items.append(item)

    if not wardrobe_items and not (left_ns or right_ns):
        # Tiny rooms still need at least one storage/nightstand class object.
        narrow = BEDROOM_SPECS["wardrobe_narrow"]
        for wall in sorted(ctx.walls, key=lambda w: (w.has_door, w.has_window, -w.length)):
            item = engine.add_wall_aligned(
                narrow,
                wall.id,
                wall.length * 0.72,
                name="Narrow wardrobe",
                category="wardrobe",
                layer="storage",
                margin=0.01,
                front_target="room_center",
                extra_meta={"required_fallback": True},
            )
            if item:
                wardrobe_items.append(item)
                break

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

    # Compact console/dresser for narrow bedrooms; full dresser for larger rooms.
    console = None
    if density_rank(density) >= 2:
        console = _add_small_console(engine, ctx, bed_wall.id)

    if density_rank(density) >= 2 and not console and not (ctx.is_long_narrow or ctx.min_side_m <= 2.7):
        for wall in sorted(ctx.walls, key=lambda w: w.length, reverse=True):
            if wall.id in {bed_wall.id, wardrobe_wall.id if wardrobe_wall else ""}:
                continue
            dresser = engine.add_wall_aligned(
                BEDROOM_SPECS["dresser"],
                wall.id,
                wall.length * 0.5,
                layer="storage",
                margin=0.02,
                front_target="room_center",
            )
            if dresser:
                engine.add_on_top(dresser, BEDROOM_SPECS["decor_vase"], local_offset_xy=(-0.25, 0.0))
                if density_rank(density) >= 3:
                    engine.add_on_top(dresser, BEDROOM_SPECS["decor_books"], local_offset_xy=(0.18, 0.02))
                    engine.add_on_top(dresser, BEDROOM_SPECS["decor_box"], local_offset_xy=(0.42, 0.0))
                break

    # Bench at bed foot.
    if density_rank(density) >= 2 and ctx.area_m2 >= 11.0:
        bench = engine.add_near(
            bed,
            BEDROOM_SPECS["bench"],
            local_offset_xy=(0.0, bed_spec.size_m[1] * 0.5 + BEDROOM_SPECS["bench"].size_m[1] * 0.5 + 0.12),
            allow_collision=False,
            layer="secondary",
            front_target=bed.get("id"),
        )
        if bench:
            meta = bench.setdefault("meta", {})
            if isinstance(meta, dict):
                meta["anchor_id"] = bed.get("id")
                meta["placement_relation"] = "at_foot_of"
                meta["clearance_to_anchor_m"] = 0.12

    # Optional desk for large bedrooms.
    if density_rank(density) >= 3 and ctx.area_m2 >= 14.0:
        for wall in sorted(ctx.walls, key=lambda w: w.length, reverse=True):
            if wall.id == bed_wall.id:
                continue
            desk = engine.add_wall_aligned(
                BEDROOM_SPECS["desk"],
                wall.id,
                wall.length * 0.28,
                layer="secondary",
                margin=0.02,
                front_target="chair",
            )
            if desk:
                engine.add_near(
                    desk,
                    BEDROOM_SPECS["chair"],
                    local_offset_xy=(0.0, BEDROOM_SPECS["desk"].size_m[1] * 0.5 + BEDROOM_SPECS["chair"].size_m[1] * 0.5 + 0.08),
                    allow_collision=False,
                    front_target=desk.get("id"),
                )
                engine.add_on_top(desk, BEDROOM_SPECS["table_lamp"], local_offset_xy=(-0.35, 0.0))
                break

    # Wall decor above bed.
    if density_rank(density) == 2:
        art_added = False
        for wall in sorted(ctx.walls, key=lambda w: w.length, reverse=True):
            if wall.id == bed_wall.id or wall.has_door or wall.has_window:
                continue
            art_count = 2 if wall.length >= 2.4 else 1
            for i in range(art_count):
                along = wall.length * (i + 1) / (art_count + 1)
                engine.add_wall_art(
                    wall.id,
                    along,
                    BEDROOM_SPECS["wall_art"],
                    z_center=1.45 + 0.08 * (i % 2),
                    name=f"Bedroom framed wall art {wall.id}-{i + 1}",
                )
                art_added = True
            break
        if not art_added:
            for i, offset in enumerate((-0.62, 0.62)):
                along = max(0.35, min(bed_wall.length - 0.35, bed_wall_along_m + offset))
                engine.add_wall_art(
                    bed_wall.id,
                    along,
                    BEDROOM_SPECS["wall_art"],
                    z_center=1.72,
                    name=f"Bedroom framed wall art above bed {i + 1}",
                )
    if density_rank(density) >= 3:
        engine.add_wall_art(bed_wall.id, max(0.4, bed_wall.length * 0.5 - 0.7), BEDROOM_SPECS["sconce"], z_center=1.45)
        engine.add_wall_art(bed_wall.id, min(bed_wall.length - 0.4, bed_wall.length * 0.5 + 0.7), BEDROOM_SPECS["sconce"], z_center=1.45)
        for wall in sorted(ctx.walls, key=lambda w: w.length, reverse=True):
            if wall.id == bed_wall.id:
                continue
            if wall.has_door:
                continue
            if wall.has_window:
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
    if density_rank(density) >= 2 and ctx.area_m2 > 7.0:
        engine.add_corner_object(BEDROOM_SPECS["plant"], preferred_index=0)
    if density_rank(density) >= 3 and ctx.area_m2 > 7.0:
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
    _add_window_curtains(engine, ctx)
    if density_rank(density) >= 2:
        _add_foreground_pouf(engine, ctx)

    report = {
        "generator": "bedroom_generator",
        "archetype": engine.archetype,
        "bed_wall_id": bed_wall.id,
        "density": density,
        "rejected": engine.rejected,
    }
    return engine.placements, report
