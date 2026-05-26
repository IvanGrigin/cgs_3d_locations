from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations
from typing import Any, Iterable, Sequence

from .geometry import Vec2, as_float, clamp
from .object_specs import BATHROOM_SPECS, SANITARY_REAL_ASSETS, TOILET_SPECS, Density, ObjectSpec, density_rank
from .placement_engine import PlacementEngine
from .room_context import RoomContext


EPS = 1e-7


class AxisWall(str, Enum):
    SOUTH = "south"
    NORTH = "north"
    WEST = "west"
    EAST = "east"


@dataclass(frozen=True)
class LocalRect:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def depth(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.depth)

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) * 0.5

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) * 0.5

    def inflate(self, margin: float) -> "LocalRect":
        return LocalRect(self.x1 - margin, self.y1 - margin, self.x2 + margin, self.y2 + margin)

    def inside(self, width: float, depth: float) -> bool:
        return self.x1 >= -EPS and self.y1 >= -EPS and self.x2 <= width + EPS and self.y2 <= depth + EPS

    def intersects(self, other: "LocalRect", *, strict: bool = True) -> bool:
        if strict:
            return not (
                self.x2 <= other.x1 + EPS
                or other.x2 <= self.x1 + EPS
                or self.y2 <= other.y1 + EPS
                or other.y2 <= self.y1 + EPS
            )
        return not (
            self.x2 < other.x1 - EPS
            or other.x2 < self.x1 - EPS
            or self.y2 < other.y1 - EPS
            or other.y2 < self.y1 - EPS
        )

    def intersection_area(self, other: "LocalRect") -> float:
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        return (x2 - x1) * (y2 - y1)

    def to_json(self) -> dict[str, float]:
        return {
            "x1": round(self.x1, 4),
            "y1": round(self.y1, 4),
            "x2": round(self.x2, 4),
            "y2": round(self.y2, 4),
        }


@dataclass(frozen=True)
class LocalDoor:
    wall: AxisWall
    center: float
    width: float


@dataclass(frozen=True)
class LocalGeometry:
    x0: float
    y0: float
    width: float
    depth: float
    doors: tuple[LocalDoor, ...]
    wall_by_axis: dict[AxisWall, Any]


@dataclass
class PlanItem:
    category: str
    spec_key: str
    spec: ObjectSpec
    rect: LocalRect
    wall: AxisWall
    layer: str
    required: bool = False
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def access_rect(self) -> LocalRect | None:
        clearance = _front_clearance_for_category(self.category)
        if clearance <= 0.0:
            return None
        return _access_rect_for(self.rect, self.wall, clearance)


@dataclass
class LayoutCandidate:
    template: str
    items: list[PlanItem] = field(default_factory=list)
    score: float = 0.0
    failures: list[str] = field(default_factory=list)

    def categories(self) -> set[str]:
        return {item.category for item in self.items}

    def solid_items(self) -> list[PlanItem]:
        return [item for item in self.items if item.spec.mount_type != "wall"]


def generate_sanitary_toilet(
    ctx: RoomContext,
    *,
    density: Density,
    seed: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    geom = _local_geometry_from_context(ctx)
    if geom is None:
        return None

    rng = random.Random(seed)
    candidates = list(_generate_toilet_candidates(ctx, geom, density, rng))
    materialized = _materialize_ranked_candidate(
        ctx,
        geom,
        rng,
        candidates,
        generator_name="toilet_generator",
        required_groups={"toilet"},
        allow_toilet_door_overlap=True,
    )
    if materialized is None:
        return None
    best, engine, added = materialized
    toilet = _first_added(added, "toilet")
    sink = _first_added(added, "sink")
    cabinet = _first_added(added, "toilet_cabinet")

    _add_toilet_accessories(engine, ctx, density, toilet, sink)
    engine.add_ceiling_light(name="Toilet ceiling light")

    report = {
        "generator": "toilet_generator",
        "archetype": engine.archetype,
        "solver": "sanitary_layout_solver",
        "template": best.template,
        "score": round(best.score, 4),
        "required": {"toilet": bool(toilet)},
        "optional": {"sink": bool(sink), "toilet_cabinet": bool(cabinet)},
        "density": density,
        "rejected": engine.rejected,
    }
    return engine.placements, report


def generate_sanitary_bathroom(
    ctx: RoomContext,
    *,
    density: Density,
    seed: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    geom = _local_geometry_from_context(ctx)
    if geom is None:
        return None

    rng = random.Random(seed)
    candidates = list(_generate_bathroom_candidates(ctx, geom, density, rng))
    materialized = _materialize_ranked_candidate(
        ctx,
        geom,
        rng,
        candidates,
        generator_name="bathroom_generator",
        required_groups={"bath", "sink"},
    )
    if materialized is None:
        return None
    best, engine, added = materialized
    bathing = _first_added(added, "bathtub") or _first_added(added, "shower")
    sink = _first_added(added, "sink") or _first_added(added, "vanity")
    cabinet = _first_added(added, "toilet_cabinet")

    _add_bathroom_accessories(engine, ctx, density, bathing, sink)
    engine.add_ceiling_light(name="Bathroom ceiling light")

    report = {
        "generator": "bathroom_generator",
        "archetype": engine.archetype,
        "solver": "sanitary_layout_solver",
        "template": best.template,
        "score": round(best.score, 4),
        "required": {"sink": bool(sink), "bathing_fixture": bool(bathing)},
        "bathing_fixture_category": bathing.get("category") if bathing else None,
        "toilet_added": False,
        "optional": {"toilet_cabinet": bool(cabinet)},
        "density": density,
        "rejected": engine.rejected,
    }
    return engine.placements, report


def _local_geometry_from_context(ctx: RoomContext) -> LocalGeometry | None:
    x_min, y_min, x_max, y_max = ctx.bounds
    width = max(0.0, x_max - x_min)
    depth = max(0.0, y_max - y_min)
    if width < 0.75 or depth < 0.75:
        return None

    tol = 0.08
    wall_by_axis: dict[AxisWall, Any] = {}
    for wall in ctx.walls:
        if _near(wall.start.y, y_min, tol) and _near(wall.end.y, y_min, tol):
            _keep_longest_wall(wall_by_axis, AxisWall.SOUTH, wall)
        elif _near(wall.start.y, y_max, tol) and _near(wall.end.y, y_max, tol):
            _keep_longest_wall(wall_by_axis, AxisWall.NORTH, wall)
        elif _near(wall.start.x, x_min, tol) and _near(wall.end.x, x_min, tol):
            _keep_longest_wall(wall_by_axis, AxisWall.WEST, wall)
        elif _near(wall.start.x, x_max, tol) and _near(wall.end.x, x_max, tol):
            _keep_longest_wall(wall_by_axis, AxisWall.EAST, wall)

    doors: list[LocalDoor] = []
    for door in ctx.doors:
        local = _local_door_from_opening(ctx, wall_by_axis, door, x_min, y_min)
        if local is not None:
            doors.append(local)
    if not doors:
        wall = next((axis for axis, segment in wall_by_axis.items() if getattr(segment, "has_door", False)), None)
        if wall is None:
            wall = AxisWall.SOUTH
        doors.append(LocalDoor(wall=wall, center=_wall_length(width, depth, wall) * 0.5, width=0.75))

    return LocalGeometry(
        x0=x_min,
        y0=y_min,
        width=width,
        depth=depth,
        doors=tuple(doors),
        wall_by_axis=wall_by_axis,
    )


def _generate_toilet_candidates(
    ctx: RoomContext,
    geom: LocalGeometry,
    density: Density,
    rng: random.Random,
) -> Iterable[LayoutCandidate]:
    door = geom.doors[0]
    toilet_walls = _preferred_walls(door.wall)
    sink_walls = _preferred_walls(door.wall, include_door_wall=True)
    cabinet_allowed = density_rank(density) >= 2 and ctx.area_m2 >= 1.65
    sink_allowed = ctx.area_m2 >= 1.15 or min(geom.width, geom.depth) >= 0.88

    for toilet_key in ("toilet", "compact_toilet"):
        toilet_spec = TOILET_SPECS[toilet_key]
        for toilet_wall in toilet_walls:
            for toilet_center in _center_candidates(geom, toilet_wall, toilet_spec, prefer_edges=False):
                toilet = _plan_item(geom, toilet_spec, toilet_key, "toilet", toilet_wall, toilet_center, "primary", required=True)
                base = LayoutCandidate(template="wc_opposite_door", items=[toilet])
                if not _candidate_ok(geom, base, allow_toilet_door_overlap=True):
                    continue

                expanded = False
                if sink_allowed:
                    for sink_key, sink_spec in _toilet_sink_specs(ctx):
                        for sink_wall in [w for w in sink_walls if w != toilet_wall] + [toilet_wall]:
                            for sink_center in _center_candidates(geom, sink_wall, sink_spec, prefer_edges=True):
                                candidate = _clone_with(base, _plan_item(
                                    geom,
                                    sink_spec,
                                    sink_key,
                                    "sink",
                                    sink_wall,
                                    sink_center,
                                    "primary",
                                    required=False,
                                ))
                                if not _candidate_ok(geom, candidate, allow_toilet_door_overlap=True):
                                    continue
                                expanded = True
                                yield from _with_optional_cabinet(geom, candidate, TOILET_SPECS, cabinet_allowed)

                if not expanded:
                    yield from _with_optional_cabinet(geom, base, TOILET_SPECS, cabinet_allowed)


def _generate_bathroom_candidates(
    ctx: RoomContext,
    geom: LocalGeometry,
    density: Density,
    rng: random.Random,
) -> Iterable[LayoutCandidate]:
    if _assume_outward_tiny_bathroom_door(ctx, geom):
        yield from _generate_tiny_bathroom_corner_candidates(ctx, geom, density, rng)
        return

    door = geom.doors[0]
    wet_walls = _preferred_walls(door.wall, include_door_wall=False)
    sink_walls = _preferred_walls(door.wall, include_door_wall=True)
    cabinet_allowed = density_rank(density) >= 2 and ctx.area_m2 >= 2.8
    ignore_door_clearance = _assume_outward_tiny_bathroom_door(ctx, geom)

    if ctx.area_m2 >= 3.8 and max(geom.width, geom.depth) >= 1.55:
        wet_keys = ("bathtub", "compact_bathtub", "shower", "compact_shower")
    elif ctx.area_m2 < 3.8:
        wet_keys = ("wet_room_shower", "compact_shower", "corner_shower_1x1", "compact_bathtub")
    elif min(geom.width, geom.depth) >= 0.98:
        wet_keys = ("corner_shower_1x1", "shower", "compact_shower", "compact_bathtub")
    else:
        wet_keys = ("shower", "compact_shower", "compact_bathtub")

    if ctx.area_m2 >= 4.0:
        sink_keys = ("vanity", "sink", "compact_sink")
    else:
        sink_keys = ("sink", "compact_sink")

    for wet_key in wet_keys:
        wet_spec = BATHROOM_SPECS[wet_key]
        for wet_wall in wet_walls:
            for wet_center in _center_candidates(geom, wet_wall, wet_spec, prefer_edges=True):
                wet = _plan_item(
                    geom,
                    wet_spec,
                    wet_key,
                    wet_spec.category,
                    wet_wall,
                    wet_center,
                    "primary",
                    required=True,
                    metadata={"corner_fixture": True} if wet_key == "corner_shower_1x1" else None,
                )
                base = LayoutCandidate(template="bath_far_corner", items=[wet])
                if not _candidate_ok(geom, base, require_access=False, ignore_door_clearance=ignore_door_clearance):
                    continue

                for sink_key in sink_keys:
                    sink_spec = BATHROOM_SPECS[sink_key]
                    for sink_wall in [w for w in sink_walls if w != wet_wall] + [wet_wall]:
                        for sink_center in _center_candidates(geom, sink_wall, sink_spec, prefer_edges=True):
                            candidate = _clone_with(base, _plan_item(
                                geom,
                                sink_spec,
                                sink_key,
                                sink_spec.category,
                                sink_wall,
                                sink_center,
                                "primary" if sink_spec.category == "sink" else "storage",
                                required=True,
                            ))
                            if not _candidate_ok(geom, candidate, ignore_door_clearance=ignore_door_clearance):
                                continue
                            yield from _with_optional_cabinet(geom, candidate, BATHROOM_SPECS, cabinet_allowed)


def _generate_tiny_bathroom_corner_candidates(
    ctx: RoomContext,
    geom: LocalGeometry,
    density: Density,
    rng: random.Random,
) -> Iterable[LayoutCandidate]:
    door = geom.doors[0]
    shower_key, shower_spec = _tiny_bathroom_shower_spec(ctx, geom)
    wet_wall = opposite_wall(door.wall)
    sink_walls = _tiny_bathroom_sink_wall_order(door.wall, wet_wall)
    wet_centers = _center_candidates(geom, wet_wall, shower_spec, prefer_edges=True)
    wet_centers.sort(key=lambda center_s: -_door_to_wall_center_distance(geom, door, wet_wall, center_s))

    for wet_center in wet_centers:
        wet = _plan_item(
            geom,
            shower_spec,
            shower_key,
            "shower",
            wet_wall,
            wet_center,
            "primary",
            required=True,
            metadata={
                "compact_bathroom_template": True,
                "door_clearance_exempt": True,
                "door_swing_assumption": "outward_or_sliding",
                "corner_fixture": True,
            },
        )
        base = LayoutCandidate(template="tiny_bathroom_1x2_corner_shower", items=[wet])
        if not _candidate_ok(geom, base, require_access=False, ignore_door_clearance=True):
            continue

        for sink_key, sink_spec in _tiny_bathroom_sink_specs(ctx):
            for sink_wall in sink_walls:
                sink_centers = _tiny_bathroom_sink_centers(geom, wet, sink_wall, sink_spec, door)
                for sink_center in sink_centers:
                    candidate = _clone_with(base, _plan_item(
                        geom,
                        sink_spec,
                        sink_key,
                        "sink",
                        sink_wall,
                        sink_center,
                        "primary",
                        required=True,
                        metadata={
                            "compact_bathroom_template": True,
                            "door_clearance_exempt": True,
                            "door_swing_assumption": "outward_or_sliding",
                            "corner_fixture": True,
                        },
                    ))
                    if not _candidate_ok(geom, candidate, ignore_door_clearance=True):
                        continue
                    yield candidate


def _assume_outward_tiny_bathroom_door(ctx: RoomContext, geom: LocalGeometry) -> bool:
    return bool(
        ctx.room_type == "bathroom"
        and ctx.area_m2 <= 2.45
        and min(geom.width, geom.depth) <= 1.55
        and max(geom.width, geom.depth) <= 2.25
        and geom.doors
    )


def _tiny_bathroom_sink_specs(ctx: RoomContext) -> list[tuple[str, ObjectSpec]]:
    specs: list[tuple[str, ObjectSpec]] = []
    if ctx.area_m2 <= 1.55 or ctx.min_side_m <= 1.05:
        specs.append(
            (
                "micro_sink",
                ObjectSpec(
                    "sink",
                    "Micro bathroom sink",
                    (0.30, 0.24, 0.30),
                    "primary",
                    requires_access=True,
                    support_surface=True,
                    front_target_hint="door",
                    proxy_base_type="sink",
                    proxy_subclass="wall_mounted_sink",
                    proxy_material="ceramic",
                    proxy_color="#f5f2ea",
                    asset_mesh_path=SANITARY_REAL_ASSETS["sink_lago_wall_hung"],
                    asset_fit_mode="uniform",
                ),
            )
        )
    if ctx.area_m2 <= 2.45:
        specs.append(
            (
                "narrow_sink",
                ObjectSpec(
                    "sink",
                    "Narrow bathroom sink",
                    (0.38, 0.28, 0.32),
                    "primary",
                    requires_access=True,
                    support_surface=True,
                    front_target_hint="door",
                    proxy_base_type="sink",
                    proxy_subclass="wall_mounted_sink",
                    proxy_material="ceramic",
                    proxy_color="#f5f2ea",
                    asset_mesh_path=SANITARY_REAL_ASSETS["sink_lago_wall_hung"],
                    asset_fit_mode="uniform",
                ),
            )
        )
    specs.append(("compact_sink", BATHROOM_SPECS["compact_sink"]))
    return specs


def _tiny_bathroom_shower_spec(ctx: RoomContext, geom: LocalGeometry) -> tuple[str, ObjectSpec]:
    if ctx.area_m2 <= 3.1 and min(geom.width, geom.depth) <= 1.55:
        return "wet_room_shower", BATHROOM_SPECS["wet_room_shower"]
    if min(geom.width, geom.depth) < 1.18 or ctx.area_m2 < 1.65:
        return "compact_shower", BATHROOM_SPECS["compact_shower"]
    return "corner_shower_1x1", BATHROOM_SPECS["corner_shower_1x1"]


def _tiny_bathroom_sink_wall_order(door_wall: AxisWall, wet_wall: AxisWall) -> list[AxisWall]:
    side_walls = [wall for wall in all_walls() if wall not in {door_wall, wet_wall}]
    if set(side_walls) == {AxisWall.WEST, AxisWall.EAST}:
        return [AxisWall.WEST, AxisWall.EAST]
    if set(side_walls) == {AxisWall.SOUTH, AxisWall.NORTH}:
        return [AxisWall.NORTH, AxisWall.SOUTH]
    return side_walls


def _door_point(geom: LocalGeometry, door: LocalDoor) -> tuple[float, float]:
    if door.wall == AxisWall.SOUTH:
        return door.center, 0.0
    if door.wall == AxisWall.NORTH:
        return door.center, geom.depth
    if door.wall == AxisWall.WEST:
        return 0.0, door.center
    return geom.width, door.center


def _wall_center_point(geom: LocalGeometry, wall: AxisWall, center_s: float) -> tuple[float, float]:
    if wall == AxisWall.SOUTH:
        return center_s, 0.0
    if wall == AxisWall.NORTH:
        return center_s, geom.depth
    if wall == AxisWall.WEST:
        return 0.0, center_s
    return geom.width, center_s


def _door_to_wall_center_distance(geom: LocalGeometry, door: LocalDoor, wall: AxisWall, center_s: float) -> float:
    dx0, dy0 = _door_point(geom, door)
    x1, y1 = _wall_center_point(geom, wall, center_s)
    return math.hypot(x1 - dx0, y1 - dy0)


def _tiny_bathroom_sink_centers(
    geom: LocalGeometry,
    wet: PlanItem,
    sink_wall: AxisWall,
    sink_spec: ObjectSpec,
    door: LocalDoor,
) -> list[float]:
    length = _wall_length(geom.width, geom.depth, sink_wall)
    half = sink_spec.size_m[0] * 0.5
    lo = half
    hi = length - half
    if hi < lo:
        return []

    raw: list[float] = []
    gap = 0.08
    if sink_wall in {AxisWall.WEST, AxisWall.EAST}:
        if wet.rect.cy >= geom.depth * 0.5:
            raw.append(wet.rect.y1 - half - gap)
        else:
            raw.append(wet.rect.y2 + half + gap)
    else:
        if wet.rect.cx >= geom.width * 0.5:
            raw.append(wet.rect.x1 - half - gap)
        else:
            raw.append(wet.rect.x2 + half + gap)
    raw.extend(_center_candidates(geom, sink_wall, sink_spec, prefer_edges=True))

    centers: list[float] = []
    for value in raw:
        center = clamp(value, lo, hi)
        if all(abs(center - old) > 0.02 for old in centers):
            centers.append(center)

    def score(center_s: float) -> float:
        distance = _door_to_wall_center_distance(geom, door, sink_wall, center_s)
        # Keep the first custom center ahead of generic edge candidates when it
        # is legal; it places the sink just outside the shower's 1x1 corner.
        custom_bonus = 10.0 if centers and abs(center_s - centers[0]) < 0.02 else 0.0
        return custom_bonus + distance

    centers.sort(key=score, reverse=True)
    return centers


def _with_optional_cabinet(
    geom: LocalGeometry,
    base: LayoutCandidate,
    library: dict[str, ObjectSpec],
    allowed: bool,
) -> Iterable[LayoutCandidate]:
    yielded = False
    if allowed:
        spec = library["toilet_cabinet"]
        occupied = {item.wall for item in base.solid_items()}
        walls = [w for w in _preferred_walls(geom.doors[0].wall, include_door_wall=True) if w not in occupied]
        walls.extend(w for w in _preferred_walls(geom.doors[0].wall, include_door_wall=True) if w not in walls)
        for wall in walls:
            for center in _center_candidates(geom, wall, spec, prefer_edges=True):
                candidate = _clone_with(base, _plan_item(
                    geom,
                    spec,
                    "toilet_cabinet",
                    "toilet_cabinet",
                    wall,
                    center,
                    "storage",
                    required=False,
                ))
                if _candidate_ok(geom, candidate, allow_toilet_door_overlap=True):
                    yielded = True
                    yield candidate
                    break
            if yielded:
                break
    yield base


def _toilet_sink_specs(ctx: RoomContext) -> list[tuple[str, ObjectSpec]]:
    specs: list[tuple[str, ObjectSpec]] = []
    if ctx.area_m2 < 2.2:
        specs.append(
            (
                "micro_sink",
                ObjectSpec(
                    "sink",
                    "Micro handwash sink",
                    (0.30, 0.24, 0.30),
                    "primary",
                    requires_access=True,
                    support_surface=True,
                    front_target_hint="door",
                    proxy_base_type="sink",
                    proxy_subclass="wall_mounted_sink",
                    proxy_material="ceramic",
                    proxy_color="#f5f2ea",
                    asset_mesh_path=SANITARY_REAL_ASSETS["sink_lago_wall_hung"],
                    asset_fit_mode="uniform",
                ),
            )
        )
    specs.extend([("corner_sink", TOILET_SPECS["corner_sink"]), ("sink", TOILET_SPECS["sink"])])
    return specs


def _choose_best_candidate(
    geom: LocalGeometry,
    candidates: Sequence[LayoutCandidate],
    *,
    required_groups: set[str],
    allow_toilet_door_overlap: bool = False,
    ignore_door_clearance: bool = False,
) -> LayoutCandidate | None:
    best: LayoutCandidate | None = None
    for candidate in candidates:
        if not _candidate_ok(geom, candidate, allow_toilet_door_overlap=allow_toilet_door_overlap, ignore_door_clearance=ignore_door_clearance):
            continue
        candidate.score = _score_candidate(geom, candidate, required_groups)
        if best is None or candidate.score > best.score:
            best = candidate
    return best


def _materialize_ranked_candidate(
    ctx: RoomContext,
    geom: LocalGeometry,
    rng: random.Random,
    candidates: Sequence[LayoutCandidate],
    *,
    generator_name: str,
    required_groups: set[str],
    allow_toilet_door_overlap: bool = False,
) -> tuple[LayoutCandidate, PlacementEngine, list[dict[str, Any]]] | None:
    ranked: list[LayoutCandidate] = []
    ignore_door_clearance = ctx.room_type == "bathroom" and _assume_outward_tiny_bathroom_door(ctx, geom)
    for candidate in candidates:
        if not _candidate_ok(
            geom,
            candidate,
            allow_toilet_door_overlap=allow_toilet_door_overlap,
            ignore_door_clearance=ignore_door_clearance,
        ):
            continue
        candidate.score = _score_candidate(geom, candidate, required_groups)
        ranked.append(candidate)
    ranked.sort(key=lambda candidate: candidate.score, reverse=True)

    for candidate in ranked:
        engine = PlacementEngine(
            ctx=ctx,
            rng=rng,
            source_name="procedural_room_stage",
            generator_name=generator_name,
            archetype=f"sanitary_solver/{candidate.template}",
        )
        added = _materialize_candidate(engine, geom, candidate)
        categories = {item.get("category") for item in added}
        has_toilet = "toilet" not in required_groups or "toilet" in categories
        has_bath = "bath" not in required_groups or bool({"bathtub", "shower"} & categories)
        has_sink = "sink" not in required_groups or bool({"sink", "vanity"} & categories)
        if has_toilet and has_bath and has_sink:
            return candidate, engine, added
    return None


def _score_candidate(geom: LocalGeometry, candidate: LayoutCandidate, required_groups: set[str]) -> float:
    categories = candidate.categories()
    score = 0.0
    if "toilet" in required_groups and "toilet" in categories:
        score += 100.0
    if "bath" in required_groups and ({"bathtub", "shower"} & categories):
        score += 100.0
    if "sink" in required_groups and ({"sink", "vanity"} & categories):
        score += 100.0
    if "sink" in categories or "vanity" in categories:
        score += 22.0
    if "toilet_cabinet" in categories:
        score += 7.0

    door = geom.doors[0]
    center_zone = LocalRect(geom.width * 0.34, geom.depth * 0.34, geom.width * 0.66, geom.depth * 0.66)
    room_area = geom.width * geom.depth
    for item in candidate.solid_items():
        if item.wall == opposite_wall(door.wall):
            score += 8.0
        if item.category in {"bathtub", "shower"} and item.wall == opposite_wall(door.wall):
            score += 16.0
        if item.category == "bathtub" and room_area >= 3.8:
            score += 28.0
        if item.spec_key == "wet_room_shower" and room_area < 3.8:
            score += 34.0
        elif item.category == "shower" and item.spec_key != "wet_room_shower" and room_area < 3.8:
            score -= 18.0
        if item.category in {"bathtub", "shower"}:
            score += _near_corner_score(geom, item.rect) * 7.0
        if item.wall == door.wall:
            score -= 12.0
        score -= item.rect.intersection_area(center_zone) * 18.0

    score += min(18.0, sum(item.rect.area for item in candidate.solid_items()) / max(EPS, geom.width * geom.depth) * 80.0)
    return score


def _candidate_ok(
    geom: LocalGeometry,
    candidate: LayoutCandidate,
    *,
    require_access: bool = True,
    allow_toilet_door_overlap: bool = False,
    ignore_door_clearance: bool = False,
) -> bool:
    solids = candidate.solid_items()
    for item in solids:
        if not item.rect.inside(geom.width, geom.depth):
            candidate.failures.append(f"{item.category}:outside_room")
            return False
        if not ignore_door_clearance:
            for zone in _door_clearance_zones(geom):
                if item.rect.intersects(zone, strict=True):
                    if allow_toilet_door_overlap and item.category == "toilet":
                        continue
                    candidate.failures.append(f"{item.category}:door_clearance")
                    return False
    for a, b in combinations(solids, 2):
        if a.rect.intersects(b.rect, strict=True):
            candidate.failures.append(f"{a.category}/{b.category}:collision")
            return False
    if require_access and not _accessibility_ok(geom, solids):
        candidate.failures.append("accessibility")
        return False
    return True


def _accessibility_ok(geom: LocalGeometry, solids: Sequence[PlanItem]) -> bool:
    targets = [(item.category, item.access_rect) for item in solids if item.required or item.category in {"toilet", "sink", "vanity", "bathtub", "shower"}]
    targets = [(cat, rect) for cat, rect in targets if rect is not None and rect.area > 0.005]
    if not targets:
        return True

    step = 0.10
    nx = max(1, int(math.floor(geom.width / step))) + 1
    ny = max(1, int(math.floor(geom.depth / step))) + 1
    obstacles = [item.rect.inflate(0.02) for item in solids]
    start = _start_point_from_door(geom, geom.doors[0])

    def cell_center(ix: int, iy: int) -> tuple[float, float]:
        return min(geom.width - EPS, (ix + 0.5) * step), min(geom.depth - EPS, (iy + 0.5) * step)

    def blocked(x: float, y: float) -> bool:
        probe = LocalRect(x - step * 0.2, y - step * 0.2, x + step * 0.2, y + step * 0.2)
        return any(probe.intersects(obs, strict=True) for obs in obstacles)

    def to_cell(x: float, y: float) -> tuple[int, int]:
        return max(0, min(nx - 1, int(x / step))), max(0, min(ny - 1, int(y / step)))

    start_cell = to_cell(*start)
    sx, sy = cell_center(*start_cell)
    if blocked(sx, sy):
        for zone in _door_clearance_zones(geom):
            found = None
            for ix in range(nx):
                for iy in range(ny):
                    x, y = cell_center(ix, iy)
                    if LocalRect(x, y, x + EPS, y + EPS).intersects(zone, strict=False) and not blocked(x, y):
                        found = (ix, iy)
                        break
                if found:
                    break
            if found:
                start_cell = found
                break
        else:
            return False

    reachable = _bfs(nx, ny, start_cell, cell_center, blocked)
    for _category, target in targets:
        assert target is not None
        ok = False
        ix1 = max(0, int(target.x1 / step) - 1)
        ix2 = min(nx - 1, int(target.x2 / step) + 1)
        iy1 = max(0, int(target.y1 / step) - 1)
        iy2 = min(ny - 1, int(target.y2 / step) + 1)
        for ix in range(ix1, ix2 + 1):
            for iy in range(iy1, iy2 + 1):
                if (ix, iy) not in reachable:
                    continue
                x, y = cell_center(ix, iy)
                if LocalRect(x, y, x + EPS, y + EPS).intersects(target, strict=False):
                    ok = True
                    break
            if ok:
                break
        if not ok:
            return False
    return True


def _materialize_candidate(engine: PlacementEngine, geom: LocalGeometry, candidate: LayoutCandidate) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    bathroom_outward_door = engine.ctx.room_type == "bathroom" and _assume_outward_tiny_bathroom_door(engine.ctx, geom)
    tiny_sink_wall = next((item.wall for item in candidate.items if item.category in {"sink", "vanity"}), None)
    for plan in candidate.items:
        is_corner_shower_plan = (
            plan.category == "shower"
            and (plan.metadata.get("corner_fixture") or plan.spec_key in {"corner_shower_1x1", "compact_shower", "micro_shower"})
        )
        yaw = _yaw_for_wall(plan.wall)
        if is_corner_shower_plan:
            yaw = _corner_shower_yaw(plan.wall, tiny_sink_wall)
        wall = geom.wall_by_axis.get(plan.wall)
        center = Vec2(geom.x0 + plan.rect.cx, geom.y0 + plan.rect.cy)
        extra_meta = {
            "required": bool(plan.required),
            "sanitary_solver": True,
            "solver_template": candidate.template,
            "solver_spec_key": plan.spec_key,
            "solver_wall": plan.wall.value,
            "solver_rect": plan.rect.to_json(),
        }
        if bathroom_outward_door:
            extra_meta["door_clearance_exempt"] = True
            extra_meta["door_swing_assumption"] = "outward_or_sliding"
        if is_corner_shower_plan:
            extra_meta["corner_shower_yaw"] = yaw
            extra_meta["corner_shower_sharp_corner"] = _corner_shower_target_corner(plan.wall, tiny_sink_wall)
            extra_meta["asset_yaw_offset_deg"] = 45.0
            extra_meta["asset_footprint_min_rect_deg"] = 45.0
            extra_meta["asset_footprint_min_rect_size_m"] = [1.005, 1.005]
        if wall is not None:
            along = _project_along_wall(wall, center)
            extra_meta["wall_id"] = wall.id
            extra_meta["wall_along_m"] = along
        extra_meta.update(plan.metadata)
        z_center = None
        mount_type = None
        if _is_wall_hung_sink_plan(plan):
            z_center = _wall_hung_sink_z_center(plan)
            mount_type = "wall"
            extra_meta["wall_hung_fixture"] = True
            if plan.spec.asset_mesh_path == SANITARY_REAL_ASSETS["sink_lago_wall_hung"]:
                extra_meta["asset_yaw_offset_deg"] = 180.0
                extra_meta["asset_orientation_note"] = "Lago wall-hung sink mesh back faces local +Y; rotate visual mesh so mounting side is against wall."
        item = engine.add_item(
            plan.spec,
            center,
            yaw,
            z_center=z_center,
            name=plan.name,
            category=plan.category,
            layer=plan.layer,
            mount_type=mount_type,
            margin=0.015,
            ignore_door_clearance=(engine.ctx.room_type == "toilet" and plan.category == "toilet") or bathroom_outward_door,
            wall_contact_side="back",
            extra_meta=extra_meta,
            front_target="door" if plan.category in {"toilet", "sink", "vanity"} else "room_center",
        )
        if item is not None:
            result.append(item)
    return result


def _is_wall_hung_sink_plan(plan: PlanItem) -> bool:
    if plan.category == "sink":
        return True
    if plan.category != "vanity":
        return False
    return plan.spec_key in {"compact_sink", "corner_sink", "micro_sink"} or plan.metadata.get("compact_bathroom_template")


def _wall_hung_sink_z_center(plan: PlanItem) -> float:
    target_top = 0.88 if plan.metadata.get("compact_bathroom_template") else 0.86
    return max(0.55, target_top - plan.spec.size_m[2] * 0.5)


def _add_toilet_accessories(
    engine: PlacementEngine,
    ctx: RoomContext,
    density: Density,
    toilet: dict[str, Any],
    sink: dict[str, Any] | None,
) -> None:
    if _has_real_asset(TOILET_SPECS["toilet_paper_holder"]):
        _add_wall_mount_near(engine, ctx, toilet, TOILET_SPECS["toilet_paper_holder"], z_center=0.75, along_delta_m=0.34, name="Toilet paper holder")
    if density_rank(density) >= 2 and _has_real_asset(TOILET_SPECS["hygiene_shower"]):
        _add_wall_mount_near(engine, ctx, toilet, TOILET_SPECS["hygiene_shower"], z_center=0.85, along_delta_m=-0.30, name="Hygiene shower")
    if sink:
        if _has_real_asset(TOILET_SPECS["soap_dispenser"]):
            engine.add_on_top(sink, TOILET_SPECS["soap_dispenser"], local_offset_xy=(-0.08, -0.02), name="Hand soap dispenser")
        if _has_real_asset(TOILET_SPECS["mirror"]):
            _add_wall_mount_near(engine, ctx, sink, TOILET_SPECS["mirror"], z_center=1.42, along_delta_m=0.0, name="Mirror above sink")
    elif density_rank(density) >= 2 and _has_real_asset(TOILET_SPECS["wall_shelf"]):
        _add_wall_mount_near(engine, ctx, toilet, TOILET_SPECS["wall_shelf"], z_center=1.55, along_delta_m=-0.45, name="Small wall shelf")


def _add_bathroom_accessories(
    engine: PlacementEngine,
    ctx: RoomContext,
    density: Density,
    bathing: dict[str, Any],
    sink: dict[str, Any],
) -> None:
    bathing_category = str(bathing.get("category") or "").strip().lower() if bathing else ""
    bathing_meta = bathing.get("meta") if isinstance(bathing.get("meta"), dict) else {}
    if (
        bathing_category == "shower"
        and bathing_meta.get("solver_spec_key") == "wet_room_shower"
        and _has_real_asset(BATHROOM_SPECS["shower_mixer"])
    ):
        mixer = _add_wall_mount_near(
            engine,
            ctx,
            bathing,
            BATHROOM_SPECS["shower_mixer"],
            z_center=1.28,
            name="Wall shower mixer and hand shower",
        )
        if mixer is not None:
            meta = mixer.setdefault("meta", {})
            if isinstance(meta, dict):
                meta["preserve_imported_group"] = True
                meta["consolidate_import_group_glb"] = True
                meta["asset_orientation_note"] = "Keep the shower rail, mixer body, hose, and hand shower as one imported wall-mounted group."
    if _has_real_asset(BATHROOM_SPECS["soap_dispenser"]):
        engine.add_on_top(sink, BATHROOM_SPECS["soap_dispenser"], local_offset_xy=(-0.12, -0.02), name="Soap dispenser")
    if _has_real_asset(BATHROOM_SPECS["toothbrush_cup"]):
        engine.add_on_top(sink, BATHROOM_SPECS["toothbrush_cup"], local_offset_xy=(0.12, -0.02), name="Toothbrush cup")
    if _has_real_asset(BATHROOM_SPECS["mirror"]):
        _add_wall_mount_near(engine, ctx, sink, BATHROOM_SPECS["mirror"], z_center=1.45, name="Mirror above sink")
    if _has_real_asset(BATHROOM_SPECS["towel_rack"]):
        if not _add_tiny_bathroom_towel_rack(engine, ctx, bathing, sink):
            _add_wall_mount_near(engine, ctx, bathing, BATHROOM_SPECS["towel_rack"], z_center=1.35, along_delta_m=0.52, name="Towel rack")
    if density_rank(density) >= 2 and _has_real_asset(BATHROOM_SPECS["wall_shelf"]):
        _add_wall_mount_near(engine, ctx, bathing, BATHROOM_SPECS["wall_shelf"], z_center=1.25, along_delta_m=-0.42, name="Shower shelf")


def _has_real_asset(spec: ObjectSpec) -> bool:
    return bool(spec.asset_mesh_path)


def _axis_from_item_meta(item: dict[str, Any] | None) -> AxisWall | None:
    if not item:
        return None
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    raw = str(meta.get("solver_wall") or "").strip().lower()
    try:
        return AxisWall(raw)
    except ValueError:
        return None


def _add_tiny_bathroom_towel_rack(
    engine: PlacementEngine,
    ctx: RoomContext,
    bathing: dict[str, Any],
    sink: dict[str, Any],
) -> dict[str, Any] | None:
    geom = _local_geometry_from_context(ctx)
    if geom is None or not _assume_outward_tiny_bathroom_door(ctx, geom):
        return None
    wet_wall = _axis_from_item_meta(bathing)
    sink_wall = _axis_from_item_meta(sink)
    door_wall = geom.doors[0].wall if geom.doors else None
    if wet_wall is None:
        return None

    preferred_axes = [
        wall
        for wall in all_walls()
        if wall not in {door_wall, wet_wall, sink_wall}
    ]
    fallback_axes = [
        wall
        for wall in all_walls()
        if wall not in {door_wall, wet_wall}
    ]
    for axis in [*preferred_axes, *[wall for wall in fallback_axes if wall not in preferred_axes]]:
        wall = geom.wall_by_axis.get(axis)
        if wall is None:
            continue
        along = _free_towel_along_for_tiny_bathroom(geom, axis, bathing, sink, wall)
        if along is None:
            continue
        item = engine.add_wall_art(
            wall.id,
            along,
            BATHROOM_SPECS["towel_rack"],
            z_center=1.35,
            name="Towel rack",
            category="towel_rack",
        )
        if item is not None:
            meta = item.setdefault("meta", {})
            if isinstance(meta, dict):
                meta["anchor_id"] = bathing.get("id")
                meta["placement_relation"] = "free_wall_near_tiny_shower"
            return item
    return None


def _free_towel_along_for_tiny_bathroom(
    geom: LocalGeometry,
    axis: AxisWall,
    bathing: dict[str, Any],
    sink: dict[str, Any],
    wall: Any,
) -> float | None:
    spec = BATHROOM_SPECS["towel_rack"]
    half = spec.size_m[0] * 0.5
    wet_rect_raw = (bathing.get("meta") if isinstance(bathing.get("meta"), dict) else {}).get("solver_rect")
    wet_rect = None
    if isinstance(wet_rect_raw, dict):
        try:
            wet_rect = LocalRect(
                float(wet_rect_raw["x1"]),
                float(wet_rect_raw["y1"]),
                float(wet_rect_raw["x2"]),
                float(wet_rect_raw["y2"]),
            )
        except Exception:
            wet_rect = None
    sink_pos = sink.get("position_m") if isinstance(sink.get("position_m"), list) else []
    sink_x = as_float(sink_pos[0], geom.x0 + geom.width * 0.5) - geom.x0
    sink_y = as_float(sink_pos[1], geom.y0 + geom.depth * 0.45) - geom.y0

    if axis in {AxisWall.WEST, AxisWall.EAST}:
        upper_limit = geom.depth - half
        if wet_rect is not None:
            upper_limit = min(upper_limit, wet_rect.y1 - 0.12)
        if upper_limit < half:
            return None
        local_y = clamp(sink_y, half, upper_limit)
        local_x = 0.0 if axis == AxisWall.WEST else geom.width
    else:
        upper_limit = geom.width - half
        if upper_limit < half:
            return None
        local_x = clamp(sink_x, half, upper_limit)
        local_y = 0.0 if axis == AxisWall.SOUTH else geom.depth
    return _project_along_wall(wall, Vec2(geom.x0 + local_x, geom.y0 + local_y))


def _add_wall_mount_near(
    engine: PlacementEngine,
    ctx: RoomContext,
    anchor: dict[str, Any],
    spec: ObjectSpec,
    *,
    z_center: float,
    along_delta_m: float = 0.0,
    name: str | None = None,
) -> dict[str, Any] | None:
    meta = anchor.get("meta") if isinstance(anchor.get("meta"), dict) else {}
    wall_id = str(meta.get("wall_id") or "")
    along = as_float(meta.get("wall_along_m"), None)
    wall = next((w for w in ctx.walls if w.id == wall_id), None)
    if wall is None or along is None:
        wall = next((w for w in ctx.walls if not w.has_door), ctx.walls[0] if ctx.walls else None)
        along = wall.length * 0.5 if wall else None
    if wall is None or along is None:
        return None
    item = engine.add_wall_art(
        wall.id,
        along + along_delta_m,
        spec,
        z_center=z_center,
        name=name,
        category=spec.category,
    )
    if item is not None:
        item.setdefault("meta", {})["anchor_id"] = anchor.get("id")
        item.setdefault("meta", {})["placement_relation"] = "near"
    return item


def _plan_item(
    geom: LocalGeometry,
    spec: ObjectSpec,
    spec_key: str,
    category: str,
    wall: AxisWall,
    center_s: float,
    layer: str,
    *,
    required: bool,
    metadata: dict[str, Any] | None = None,
) -> PlanItem:
    half = spec.size_m[0] * 0.5
    depth = spec.size_m[1]
    if wall == AxisWall.SOUTH:
        rect = LocalRect(center_s - half, 0.0, center_s + half, depth)
    elif wall == AxisWall.NORTH:
        rect = LocalRect(center_s - half, geom.depth - depth, center_s + half, geom.depth)
    elif wall == AxisWall.WEST:
        rect = LocalRect(0.0, center_s - half, depth, center_s + half)
    else:
        rect = LocalRect(geom.width - depth, center_s - half, geom.width, center_s + half)
    return PlanItem(category=category, spec_key=spec_key, spec=spec, rect=rect, wall=wall, layer=layer, required=required, metadata=metadata or {})


def _center_candidates(geom: LocalGeometry, wall: AxisWall, spec: ObjectSpec, *, prefer_edges: bool) -> list[float]:
    length = _wall_length(geom.width, geom.depth, wall)
    lo = spec.size_m[0] * 0.5
    hi = length - spec.size_m[0] * 0.5
    if hi < lo:
        return []
    if prefer_edges:
        raw = [lo + 0.02, hi - 0.02, length * 0.5, length * 0.28, length * 0.72, length * 0.38, length * 0.62]
    else:
        raw = [length * 0.5, length * 0.38, length * 0.62, lo + 0.02, hi - 0.02, length * 0.28, length * 0.72]
    centers: list[float] = []
    for value in raw:
        c = clamp(value, lo, hi)
        if all(abs(c - old) > 0.02 for old in centers):
            centers.append(c)
    return centers


def _door_clearance_zones(geom: LocalGeometry) -> list[LocalRect]:
    zones: list[LocalRect] = []
    for door in geom.doors:
        half = door.width * 0.5 + 0.20
        depth = 1.0
        c = door.center
        if door.wall == AxisWall.SOUTH:
            zones.append(LocalRect(max(0.0, c - half), 0.0, min(geom.width, c + half), min(geom.depth, depth)))
        elif door.wall == AxisWall.NORTH:
            zones.append(LocalRect(max(0.0, c - half), max(0.0, geom.depth - depth), min(geom.width, c + half), geom.depth))
        elif door.wall == AxisWall.WEST:
            zones.append(LocalRect(0.0, max(0.0, c - half), min(geom.width, depth), min(geom.depth, c + half)))
        else:
            zones.append(LocalRect(max(0.0, geom.width - depth), max(0.0, c - half), geom.width, min(geom.depth, c + half)))
    return zones


def _access_rect_for(rect: LocalRect, wall: AxisWall, clearance: float) -> LocalRect:
    if wall == AxisWall.SOUTH:
        return LocalRect(rect.x1, rect.y2, rect.x2, rect.y2 + clearance)
    if wall == AxisWall.NORTH:
        return LocalRect(rect.x1, rect.y1 - clearance, rect.x2, rect.y1)
    if wall == AxisWall.WEST:
        return LocalRect(rect.x2, rect.y1, rect.x2 + clearance, rect.y2)
    return LocalRect(rect.x1 - clearance, rect.y1, rect.x1, rect.y2)


def _front_clearance_for_category(category: str) -> float:
    return {
        "toilet": 0.58,
        "sink": 0.50,
        "vanity": 0.58,
        "bathtub": 0.58,
        "shower": 0.60,
        "toilet_cabinet": 0.42,
        "washing_machine": 0.58,
    }.get(category, 0.0)


def _start_point_from_door(geom: LocalGeometry, door: LocalDoor) -> tuple[float, float]:
    offset = 0.18
    if door.wall == AxisWall.SOUTH:
        return door.center, min(geom.depth - EPS, offset)
    if door.wall == AxisWall.NORTH:
        return door.center, max(EPS, geom.depth - offset)
    if door.wall == AxisWall.WEST:
        return min(geom.width - EPS, offset), door.center
    return max(EPS, geom.width - offset), door.center


def _bfs(nx: int, ny: int, start: tuple[int, int], cell_center_fn, blocked_fn) -> set[tuple[int, int]]:
    queue = [start]
    seen = {start}
    head = 0
    while head < len(queue):
        ix, iy = queue[head]
        head += 1
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx_i = ix + dx
            ny_i = iy + dy
            if not (0 <= nx_i < nx and 0 <= ny_i < ny):
                continue
            cell = (nx_i, ny_i)
            if cell in seen:
                continue
            x, y = cell_center_fn(nx_i, ny_i)
            if blocked_fn(x, y):
                continue
            seen.add(cell)
            queue.append(cell)
    return seen


def _preferred_walls(door_wall: AxisWall, *, include_door_wall: bool = False) -> list[AxisWall]:
    adjacent = [w for w in all_walls() if w not in {door_wall, opposite_wall(door_wall)}]
    walls = [opposite_wall(door_wall), *adjacent]
    if include_door_wall:
        walls.append(door_wall)
    return walls


def opposite_wall(wall: AxisWall) -> AxisWall:
    return {
        AxisWall.SOUTH: AxisWall.NORTH,
        AxisWall.NORTH: AxisWall.SOUTH,
        AxisWall.WEST: AxisWall.EAST,
        AxisWall.EAST: AxisWall.WEST,
    }[wall]


def all_walls() -> tuple[AxisWall, AxisWall, AxisWall, AxisWall]:
    return AxisWall.SOUTH, AxisWall.NORTH, AxisWall.WEST, AxisWall.EAST


def _wall_length(width: float, depth: float, wall: AxisWall) -> float:
    return width if wall in {AxisWall.SOUTH, AxisWall.NORTH} else depth


def _yaw_for_wall(wall: AxisWall) -> float:
    return {
        AxisWall.SOUTH: 0.0,
        AxisWall.NORTH: 180.0,
        AxisWall.WEST: 270.0,
        AxisWall.EAST: 90.0,
    }[wall]


def _corner_shower_target_corner(wet_wall: AxisWall, sink_wall: AxisWall | None) -> str:
    # Keep the curved shower door opening toward the free side of the room.
    # The 1x2 tiny-bathroom layout moves the sink to the opposite wall instead
    # of mirroring the shower, otherwise the sink lands in front of the shower
    # entry.
    if wet_wall == AxisWall.NORTH:
        return "north_west"
    if wet_wall == AxisWall.SOUTH:
        return "south_east" if sink_wall == AxisWall.WEST else "south_west"
    if wet_wall == AxisWall.WEST:
        return "south_west" if sink_wall == AxisWall.NORTH else "north_west"
    return "south_east" if sink_wall == AxisWall.NORTH else "north_east"


def _corner_shower_yaw(wet_wall: AxisWall, sink_wall: AxisWall | None) -> float:
    # Logical yaw for the 1x1 room footprint. The DIWO OBJ visual mesh itself
    # is authored diagonally; Blender receives asset_yaw_offset_deg=45 so the
    # real minimum footprint rect is fitted without rotating this logical AABB.
    return {
        "north_west": 0.0,
        "south_west": 90.0,
        "south_east": 180.0,
        "north_east": 270.0,
    }[_corner_shower_target_corner(wet_wall, sink_wall)]


def _near_corner_score(geom: LocalGeometry, rect: LocalRect) -> float:
    corners = [(0.0, 0.0), (geom.width, 0.0), (0.0, geom.depth), (geom.width, geom.depth)]
    distance = min(abs(rect.cx - x) + abs(rect.cy - y) for x, y in corners)
    return max(0.0, 1.5 - distance)


def _clone_with(candidate: LayoutCandidate, item: PlanItem) -> LayoutCandidate:
    return LayoutCandidate(template=candidate.template, items=[*candidate.items, item])


def _first_added(items: Sequence[dict[str, Any]], category: str) -> dict[str, Any] | None:
    return next((item for item in items if item.get("category") == category), None)


def _local_door_from_opening(
    ctx: RoomContext,
    wall_by_axis: dict[AxisWall, Any],
    door: dict[str, Any],
    x0: float,
    y0: float,
) -> LocalDoor | None:
    wall_id = str(door.get("wall_id") or "")
    axis = next((axis for axis, wall in wall_by_axis.items() if wall.id == wall_id), None)
    wall = wall_by_axis.get(axis) if axis is not None else None
    if axis is None or wall is None:
        return None

    width = as_float(door.get("width"), 0.75)
    if "center" in door:
        center = as_float(door.get("center"))
    else:
        center_s = as_float(door.get("s"), wall.length * 0.5 - width * 0.5) + width * 0.5
        point = wall.point_at(center_s)
        center = point.x - x0 if axis in {AxisWall.SOUTH, AxisWall.NORTH} else point.y - y0
    return LocalDoor(wall=axis, center=clamp(center, 0.0, _wall_length(ctx.width_m, ctx.depth_m, axis)), width=width)


def _keep_longest_wall(wall_by_axis: dict[AxisWall, Any], axis: AxisWall, wall: Any) -> None:
    current = wall_by_axis.get(axis)
    if current is None or wall.length > current.length:
        wall_by_axis[axis] = wall


def _project_along_wall(wall: Any, point: Vec2) -> float:
    return clamp((point - wall.start).dot(wall.tangent), 0.0, wall.length)


def _near(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol
