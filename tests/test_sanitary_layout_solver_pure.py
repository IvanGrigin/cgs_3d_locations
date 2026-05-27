from __future__ import annotations

import random
from types import SimpleNamespace

import pytest

from src.pipeline.procedural_rooms.geometry import Vec2
from src.pipeline.procedural_rooms.object_specs import BATHROOM_SPECS, TOILET_SPECS, ObjectSpec, density_rank, normalize_density
from src.pipeline.procedural_rooms.room_context import build_room_context
from src.pipeline.procedural_rooms import sanitary_layout_solver as solver


def room_scene(room_type: str = "bathroom", width: float = 2.0, depth: float = 2.4, *, doors: bool = True) -> dict:
    room = {
        "id": "room_unit",
        "room_type": room_type,
        "width_m": width,
        "depth_m": depth,
        "floor_polygon": [{"x": 0, "y": 0}, {"x": width, "y": 0}, {"x": width, "y": depth}, {"x": 0, "y": depth}],
    }
    if doors:
        door_width = 0.7
        room["doors"] = [{"id": "door_1", "wall_id": "w0", "s": width * 0.5 - door_width * 0.5, "width": door_width}]
    return {"room": room}


def ctx(room_type: str = "bathroom", width: float = 2.0, depth: float = 2.4, *, doors: bool = True):
    return build_room_context(room_scene(room_type, width, depth, doors=doors))


def geom(room_type: str = "bathroom", width: float = 2.0, depth: float = 2.4, *, doors: bool = True) -> solver.LocalGeometry:
    result = solver._local_geometry_from_context(ctx(room_type, width, depth, doors=doors))
    assert result is not None
    return result


def test_local_rect_properties_intersections_and_serialization():
    assert normalize_density("minimal") == "normal"
    assert normalize_density("maximum") == "very_high"
    assert normalize_density("unexpected") == "high"
    assert density_rank("very_high") == 3

    rect = solver.LocalRect(0.1, 0.2, 1.1, 1.7)
    assert rect.width == pytest.approx(1.0)
    assert rect.depth == pytest.approx(1.5)
    assert rect.area == pytest.approx(1.5)
    assert rect.cx == pytest.approx(0.6)
    assert rect.cy == pytest.approx(0.95)
    inflated = rect.inflate(0.1)
    assert (inflated.x1, inflated.y1, inflated.x2, inflated.y2) == pytest.approx((0.0, 0.1, 1.2, 1.8))
    assert rect.inside(2.0, 2.0)
    assert not rect.inside(0.5, 2.0)
    assert rect.intersects(solver.LocalRect(1.0, 1.0, 1.5, 2.0))
    assert not rect.intersects(solver.LocalRect(1.1, 1.7, 1.4, 2.0), strict=True)
    assert rect.intersects(solver.LocalRect(1.1, 1.7, 1.4, 2.0), strict=False)
    assert rect.intersection_area(solver.LocalRect(0.6, 0.7, 1.6, 2.0)) == pytest.approx(0.5)
    assert rect.to_json() == {"x1": 0.1, "y1": 0.2, "x2": 1.1, "y2": 1.7}


def test_local_geometry_door_fallback_and_wall_helpers():
    g = geom("toilet", 1.4, 2.0)
    assert (g.x0, g.y0, g.width, g.depth) == pytest.approx((0.0, 0.0, 1.4, 2.0))
    assert g.doors == (solver.LocalDoor(solver.AxisWall.SOUTH, 0.7, 0.7),)
    assert set(g.wall_by_axis) == set(solver.all_walls())

    fallback = geom("bathroom", 2.0, 2.0, doors=False)
    assert fallback.doors[0].wall == solver.AxisWall.SOUTH
    assert fallback.doors[0].center == pytest.approx(1.0)

    assert solver.opposite_wall(solver.AxisWall.WEST) == solver.AxisWall.EAST
    assert solver._wall_length(1.4, 2.0, solver.AxisWall.NORTH) == 1.4
    assert solver._wall_length(1.4, 2.0, solver.AxisWall.EAST) == 2.0
    assert solver._yaw_for_wall(solver.AxisWall.SOUTH) == 0.0
    assert solver._yaw_for_wall(solver.AxisWall.NORTH) == 180.0
    assert solver._preferred_walls(solver.AxisWall.SOUTH) == [
        solver.AxisWall.NORTH,
        solver.AxisWall.WEST,
        solver.AxisWall.EAST,
    ]
    assert solver._preferred_walls(solver.AxisWall.SOUTH, include_door_wall=True)[-1] == solver.AxisWall.SOUTH


def test_plan_item_candidates_door_clearance_and_access_rects():
    g = geom("bathroom", 2.0, 2.0)
    spec = BATHROOM_SPECS["compact_sink"]

    south = solver._plan_item(g, spec, "compact_sink", "sink", solver.AxisWall.SOUTH, 1.0, "primary", required=True)
    north = solver._plan_item(g, spec, "compact_sink", "sink", solver.AxisWall.NORTH, 1.0, "primary", required=True)
    west = solver._plan_item(g, spec, "compact_sink", "sink", solver.AxisWall.WEST, 1.0, "primary", required=True)
    east = solver._plan_item(g, spec, "compact_sink", "sink", solver.AxisWall.EAST, 1.0, "primary", required=True)

    assert south.rect == solver.LocalRect(0.76, 0.0, 1.24, 0.32)
    assert north.rect.y1 == pytest.approx(1.68)
    assert west.rect.x2 == pytest.approx(0.32)
    assert east.rect.x1 == pytest.approx(1.68)
    assert (south.access_rect.x1, south.access_rect.y1, south.access_rect.x2, south.access_rect.y2) == pytest.approx((0.76, 0.32, 1.24, 0.82))
    assert (north.access_rect.x1, north.access_rect.y1, north.access_rect.x2, north.access_rect.y2) == pytest.approx((0.76, 1.18, 1.24, 1.68))
    assert (west.access_rect.x1, west.access_rect.y1, west.access_rect.x2, west.access_rect.y2) == pytest.approx((0.32, 0.76, 0.82, 1.24))
    assert (east.access_rect.x1, east.access_rect.y1, east.access_rect.x2, east.access_rect.y2) == pytest.approx((1.18, 0.76, 1.68, 1.24))

    centers = solver._center_candidates(g, solver.AxisWall.SOUTH, spec, prefer_edges=True)
    assert centers[0] == pytest.approx(spec.size_m[0] * 0.5 + 0.02)
    assert len(centers) == len(set(round(c, 3) for c in centers))
    assert solver._center_candidates(g, solver.AxisWall.SOUTH, ObjectSpec("x", "Too wide", (3.0, 0.2, 0.2), "x"), prefer_edges=True) == []

    zones = solver._door_clearance_zones(g)
    assert (zones[0].x1, zones[0].y1, zones[0].x2, zones[0].y2) == pytest.approx((0.45, 0.0, 1.55, 1.0))
    assert solver._start_point_from_door(g, solver.LocalDoor(solver.AxisWall.NORTH, 1.0, 0.7)) == pytest.approx((1.0, 1.82))
    assert solver._start_point_from_door(g, solver.LocalDoor(solver.AxisWall.WEST, 1.0, 0.7)) == pytest.approx((0.18, 1.0))
    assert solver._start_point_from_door(g, solver.LocalDoor(solver.AxisWall.EAST, 1.0, 0.7)) == pytest.approx((1.82, 1.0))


def test_candidate_validation_scoring_and_bfs_paths():
    g = geom("bathroom", 2.4, 2.4)
    shower = solver._plan_item(g, BATHROOM_SPECS["compact_shower"], "compact_shower", "shower", solver.AxisWall.NORTH, 0.6, "primary", required=True)
    sink = solver._plan_item(g, BATHROOM_SPECS["compact_sink"], "compact_sink", "sink", solver.AxisWall.EAST, 1.4, "primary", required=True)
    candidate = solver.LayoutCandidate("ok", [shower, sink])

    assert solver._candidate_ok(g, candidate)
    assert solver._accessibility_ok(g, candidate.solid_items())
    assert solver._score_candidate(g, candidate, {"bath", "sink"}) > 200
    assert solver._choose_best_candidate(g, [candidate], required_groups={"bath", "sink"}) is candidate

    outside = solver.LayoutCandidate(
        "outside",
        [solver.PlanItem("sink", "compact_sink", BATHROOM_SPECS["compact_sink"], solver.LocalRect(-0.5, 0, 0.2, 0.3), solver.AxisWall.SOUTH, "primary", True)],
    )
    assert not solver._candidate_ok(g, outside)
    assert outside.failures == ["sink:outside_room"]

    blocked_door = solver.LayoutCandidate(
        "door",
        [solver._plan_item(g, BATHROOM_SPECS["compact_sink"], "compact_sink", "sink", solver.AxisWall.SOUTH, 1.2, "primary", required=True)],
    )
    assert not solver._candidate_ok(g, blocked_door)
    assert blocked_door.failures == ["sink:door_clearance"]

    collision = solver.LayoutCandidate("collision", [shower, solver._clone_with(solver.LayoutCandidate("base", [shower]), shower).items[-1]])
    assert not solver._candidate_ok(g, collision, ignore_door_clearance=True)
    assert collision.failures[-1] == "shower/shower:collision"

    seen = solver._bfs(3, 3, (0, 0), lambda ix, iy: (ix, iy), lambda x, y: x == 1 and y == 0)
    assert (0, 0) in seen
    assert (1, 0) not in seen
    assert (2, 2) in seen


def test_tiny_bathroom_helpers_choose_compact_wall_placements():
    c = ctx("bathroom", 1.0, 1.5)
    g = solver._local_geometry_from_context(c)
    assert g is not None
    assert solver._assume_outward_tiny_bathroom_door(c, g)

    shower_key, shower_spec = solver._tiny_bathroom_shower_spec(c, g)
    assert shower_key == "wet_room_shower"
    assert shower_spec.category == "shower"
    assert [key for key, _ in solver._tiny_bathroom_sink_specs(c)] == ["micro_sink", "narrow_sink", "compact_sink"]
    assert solver._tiny_bathroom_sink_wall_order(solver.AxisWall.SOUTH, solver.AxisWall.NORTH) == [solver.AxisWall.WEST, solver.AxisWall.EAST]

    door = g.doors[0]
    wet = solver._plan_item(g, shower_spec, shower_key, "shower", solver.AxisWall.NORTH, 0.8, "primary", required=True)
    sink_centers = solver._tiny_bathroom_sink_centers(g, wet, solver.AxisWall.WEST, solver._tiny_bathroom_sink_specs(c)[0][1], door)
    assert sink_centers
    assert sink_centers[0] > 0.0
    assert solver._door_point(g, door) == pytest.approx((0.5, 0.0))
    assert solver._wall_center_point(g, solver.AxisWall.NORTH, 0.5) == pytest.approx((0.5, 1.5))
    assert solver._door_to_wall_center_distance(g, door, solver.AxisWall.NORTH, 0.5) == pytest.approx(1.5)
    assert solver._corner_shower_target_corner(solver.AxisWall.SOUTH, solver.AxisWall.WEST) == "south_east"
    assert solver._corner_shower_yaw(solver.AxisWall.WEST, solver.AxisWall.NORTH) == 90.0
    assert solver._near_corner_score(g, solver.LocalRect(0.0, 0.0, 0.3, 0.3)) > 1.0


def test_candidate_generators_emit_valid_toilet_and_bathroom_layouts():
    toilet_ctx = ctx("toilet", 1.4, 2.0)
    toilet_geom = solver._local_geometry_from_context(toilet_ctx)
    assert toilet_geom is not None
    toilet_candidate = next(solver._generate_toilet_candidates(toilet_ctx, toilet_geom, "high", random.Random(1)))
    assert {"toilet", "sink"} <= toilet_candidate.categories()

    bath_ctx = ctx("bathroom", 2.2, 2.4)
    bath_geom = solver._local_geometry_from_context(bath_ctx)
    assert bath_geom is not None
    bath_candidate = next(solver._generate_bathroom_candidates(bath_ctx, bath_geom, "high", random.Random(2)))
    assert ({"sink", "vanity"} & bath_candidate.categories()) and ({"bathtub", "shower"} & bath_candidate.categories())

    tiny_ctx = ctx("bathroom", 1.2, 1.8)
    tiny_geom = solver._local_geometry_from_context(tiny_ctx)
    assert tiny_geom is not None
    tiny_candidate = next(solver._generate_bathroom_candidates(tiny_ctx, tiny_geom, "very_high", random.Random(3)))
    assert tiny_candidate.template == "tiny_bathroom_1x2_corner_shower"


def test_public_generators_return_reports_and_solver_metadata(monkeypatch):
    def plan(local_geom, specs, key, category, wall, center, **kwargs):
        return solver._plan_item(local_geom, specs[key], key, category, wall, center, "primary", required=True, **kwargs)

    def toilet_candidates(room_ctx, local_geom, _density, _rng):
        return [solver.LayoutCandidate("test_toilet", [plan(local_geom, TOILET_SPECS, "toilet", "toilet", solver.AxisWall.NORTH, local_geom.width * 0.5)])]

    def bathroom_candidates(room_ctx, local_geom, _density, _rng):
        shower = plan(local_geom, BATHROOM_SPECS, "compact_shower", "shower", solver.AxisWall.NORTH, min(0.8, local_geom.width * 0.7))
        sink = plan(
            local_geom,
            BATHROOM_SPECS,
            "compact_sink",
            "sink",
            solver.AxisWall.WEST,
            local_geom.depth * 0.45,
            metadata={"compact_bathroom_template": room_ctx.width_m <= 1.25},
        )
        template = "tiny_bathroom_1x2_corner_shower" if room_ctx.width_m <= 1.25 else "test_bathroom"
        return [solver.LayoutCandidate(template, [shower, sink])]

    monkeypatch.setattr(solver, "_generate_toilet_candidates", toilet_candidates)
    monkeypatch.setattr(solver, "_generate_bathroom_candidates", bathroom_candidates)

    toilet_result = solver.generate_sanitary_toilet(ctx("toilet", 1.4, 2.0), density="high", seed=1)
    assert toilet_result is not None
    toilet_items, toilet_report = toilet_result
    assert toilet_report["generator"] == "toilet_generator"
    assert toilet_report["required"] == {"toilet": True}
    assert any(item["category"] == "ceiling_light" for item in toilet_items)
    assert any(item["meta"].get("sanitary_solver") for item in toilet_items if item["category"] == "toilet")

    bathroom_result = solver.generate_sanitary_bathroom(ctx("bathroom", 2.2, 2.4), density="high", seed=2)
    assert bathroom_result is not None
    bathroom_items, bathroom_report = bathroom_result
    assert bathroom_report["generator"] == "bathroom_generator"
    assert bathroom_report["required"] == {"sink": True, "bathing_fixture": True}
    assert bathroom_report["bathing_fixture_category"] in {"bathtub", "shower"}
    assert any(item["category"] == "ceiling_light" for item in bathroom_items)

    tiny_result = solver.generate_sanitary_bathroom(ctx("bathroom", 1.2, 1.8), density="very_high", seed=3)
    assert tiny_result is not None
    tiny_items, tiny_report = tiny_result
    assert tiny_report["template"] == "tiny_bathroom_1x2_corner_shower"
    assert any(item["meta"].get("door_swing_assumption") == "outward_or_sliding" for item in tiny_items)


class FakeEngine:
    def __init__(self, room_ctx):
        self.ctx = room_ctx
        self.archetype = "fake"
        self.placements: list[dict] = []
        self.rejected: list[dict] = []
        self.add_item_calls: list[dict] = []

    def add_item(self, spec, center, yaw, **kwargs):
        item = {
            "id": f"{spec.category}_{len(self.placements)}",
            "category": kwargs.get("category") or spec.category,
            "name": kwargs.get("name") or spec.name,
            "position_m": [center.x, center.y, kwargs.get("z_center") or spec.size_m[2] * 0.5],
            "size_m": list(spec.size_m),
            "yaw_deg": yaw,
            "meta": dict(kwargs.get("extra_meta") or {}),
        }
        self.add_item_calls.append({"spec": spec, "center": center, "yaw": yaw, "kwargs": kwargs})
        self.placements.append(item)
        return item

    def add_wall_art(self, wall_id, along_center_m, spec, *, z_center=1.55, name=None, category=None, layer=None):
        item = {
            "id": f"wall_{len(self.placements)}",
            "category": category or spec.category,
            "name": name or spec.name,
            "position_m": [0.0, along_center_m, z_center],
            "size_m": list(spec.size_m),
            "meta": {"wall_id": wall_id, "wall_along_m": along_center_m},
        }
        self.placements.append(item)
        return item

    def add_on_top(self, parent, spec, *, local_offset_xy=(0.0, 0.0), yaw_delta_deg=0.0, name=None, category=None, layer=None):
        item = {
            "id": f"top_{len(self.placements)}",
            "category": category or spec.category,
            "name": name or spec.name,
            "position_m": [0.0, 0.0, 1.0],
            "size_m": list(spec.size_m),
            "meta": {"parent_id": parent.get("id"), "support_relation": "on_top", "offset": list(local_offset_xy)},
        }
        self.placements.append(item)
        return item

    def add_ceiling_light(self, *args, **kwargs):
        self.placements.append({"id": "ceiling", "category": "ceiling_light", "meta": {}})


def test_materialize_candidate_writes_solver_and_wall_hung_metadata():
    c = ctx("bathroom", 1.2, 1.8)
    g = solver._local_geometry_from_context(c)
    assert g is not None
    wet = solver._plan_item(
        g,
        BATHROOM_SPECS["compact_shower"],
        "compact_shower",
        "shower",
        solver.AxisWall.NORTH,
        0.8,
        "primary",
        required=True,
        metadata={"corner_fixture": True},
    )
    sink = solver._plan_item(g, BATHROOM_SPECS["compact_sink"], "compact_sink", "sink", solver.AxisWall.WEST, 0.7, "primary", required=True)
    candidate = solver.LayoutCandidate("tiny_bathroom_1x2_corner_shower", [wet, sink])
    engine = FakeEngine(c)

    added = solver._materialize_candidate(engine, g, candidate)
    assert [item["category"] for item in added] == ["shower", "sink"]
    assert added[0]["meta"]["corner_shower_yaw"] == 0.0
    assert added[0]["meta"]["asset_yaw_offset_deg"] == 45.0
    assert added[1]["meta"]["wall_hung_fixture"] is True
    assert added[1]["meta"]["asset_yaw_offset_deg"] == 180.0
    assert engine.add_item_calls[1]["kwargs"]["mount_type"] == "wall"


def test_accessory_helpers_use_anchor_wall_metadata(monkeypatch):
    c = ctx("bathroom", 2.2, 2.4)
    engine = FakeEngine(c)
    anchor = {"id": "anchor", "category": "sink", "position_m": [1.0, 1.0, 0.5], "size_m": [0.5, 0.3, 0.4], "meta": {"wall_id": "w2", "wall_along_m": 0.8}}

    item = solver._add_wall_mount_near(engine, c, anchor, BATHROOM_SPECS["mirror"], z_center=1.4, along_delta_m=0.2, name="Mirror")
    assert item is not None
    assert item["meta"]["anchor_id"] == "anchor"
    assert item["meta"]["placement_relation"] == "near"
    assert item["meta"]["wall_along_m"] == pytest.approx(1.0)

    monkeypatch.setattr(solver, "_has_real_asset", lambda spec: True)
    solver._add_toilet_accessories(engine, c, "very_high", anchor, anchor)
    assert any(item["name"] == "Hand soap dispenser" for item in engine.placements)
    assert any(item["name"] == "Mirror above sink" for item in engine.placements)

    wet_room = {"id": "shower", "category": "shower", "position_m": [1.0, 1.8, 0.5], "size_m": [0.4, 0.4, 0.1], "meta": {"wall_id": "w2", "wall_along_m": 0.6, "solver_spec_key": "wet_room_shower", "solver_wall": "north"}}
    solver._add_bathroom_accessories(engine, c, "very_high", wet_room, anchor)
    mixer = next(item for item in engine.placements if item["name"] == "Wall shower mixer and hand shower")
    assert mixer["meta"]["preserve_imported_group"] is True
    assert any(item["name"] == "Shower shelf" for item in engine.placements)


def test_tiny_towel_rack_helpers_find_free_wall():
    c = ctx("bathroom", 1.2, 1.8)
    g = solver._local_geometry_from_context(c)
    assert g is not None
    engine = FakeEngine(c)
    bathing = {
        "id": "shower",
        "category": "shower",
        "position_m": [0.8, 1.7, 0.02],
        "meta": {"solver_wall": "north", "solver_rect": {"x1": 0.5, "y1": 1.5, "x2": 1.0, "y2": 1.8}},
    }
    sink = {"id": "sink", "category": "sink", "position_m": [0.1, 0.7, 0.7], "meta": {"solver_wall": "west"}}

    wall = g.wall_by_axis[solver.AxisWall.EAST]
    along = solver._free_towel_along_for_tiny_bathroom(g, solver.AxisWall.EAST, bathing, sink, wall)
    assert along is not None

    rack = solver._add_tiny_bathroom_towel_rack(engine, c, bathing, sink)
    assert rack is not None
    assert rack["category"] == "towel_rack"
    assert rack["meta"]["anchor_id"] == "shower"
    assert rack["meta"]["placement_relation"] == "free_wall_near_tiny_shower"


def test_small_invalid_contexts_and_parsers_return_none_or_defaults():
    tiny = build_room_context(room_scene("bathroom", 0.5, 0.7))
    assert solver._local_geometry_from_context(tiny) is None
    assert solver.generate_sanitary_bathroom(tiny, density="normal", seed=0) is None

    assert solver._axis_from_item_meta(None) is None
    assert solver._axis_from_item_meta({"meta": {"solver_wall": "bad"}}) is None
    assert solver._axis_from_item_meta({"meta": {"solver_wall": "east"}}) == solver.AxisWall.EAST
    assert solver._is_wall_hung_sink_plan(
        solver.PlanItem("vanity", "compact_sink", BATHROOM_SPECS["compact_sink"], solver.LocalRect(0, 0, 1, 1), solver.AxisWall.SOUTH, "primary")
    )
    assert solver._wall_hung_sink_z_center(
        solver.PlanItem("sink", "micro_sink", BATHROOM_SPECS["compact_sink"], solver.LocalRect(0, 0, 1, 1), solver.AxisWall.SOUTH, "primary", metadata={"compact_bathroom_template": True})
    ) == pytest.approx(0.7)


def test_sanitary_solver_remaining_failure_and_fallback_edges(monkeypatch):
    tiny_toilet = build_room_context(room_scene("toilet", 0.5, 0.7))
    assert solver.generate_sanitary_toilet(tiny_toilet, density="normal", seed=0) is None

    monkeypatch.setattr(solver, "_materialize_ranked_candidate", lambda *args, **kwargs: None)
    assert solver.generate_sanitary_toilet(ctx("toilet", 1.4, 2.0), density="normal", seed=0) is None
    assert solver.generate_sanitary_bathroom(ctx("bathroom", 2.2, 2.4), density="normal", seed=0) is None

    g = geom("bathroom", 1.4, 1.8)
    decor_spec = ObjectSpec("decor", "Decor", (0.2, 0.2, 0.2), "decor")
    decor = solver.PlanItem("decor", "decor", decor_spec, solver.LocalRect(0.2, 0.2, 0.4, 0.4), solver.AxisWall.SOUTH, "decor", required=False)
    assert decor.access_rect is None
    assert solver._accessibility_ok(g, [decor])

    blocking_sink = solver.PlanItem(
        "sink",
        "blocking_sink",
        BATHROOM_SPECS["compact_sink"],
        solver.LocalRect(0.0, 0.0, g.width, g.depth),
        solver.AxisWall.SOUTH,
        "primary",
        required=True,
    )
    assert not solver._accessibility_ok(g, [blocking_sink])

    toilet = solver._plan_item(g, TOILET_SPECS["toilet"], "toilet", "toilet", solver.AxisWall.SOUTH, g.width * 0.5, "primary", required=True)
    candidate = solver.LayoutCandidate("door_toilet", [toilet])
    assert solver._candidate_ok(g, candidate, allow_toilet_door_overlap=True, require_access=False)

    base = solver.LayoutCandidate("base", [toilet])
    assert list(solver._with_optional_cabinet(g, base, TOILET_SPECS, allowed=False)) == [base]

    compact_ctx = ctx("bathroom", 1.1, 3.0)
    compact_geom = solver._local_geometry_from_context(compact_ctx)
    assert compact_geom is not None
    assert solver._tiny_bathroom_shower_spec(compact_ctx, compact_geom)[0] == "compact_shower"
    corner_ctx = ctx("bathroom", 1.6, 2.0)
    corner_geom = solver._local_geometry_from_context(corner_ctx)
    assert corner_geom is not None
    assert solver._tiny_bathroom_shower_spec(corner_ctx, corner_geom)[0] == "corner_shower_1x1"

    huge_sink = ObjectSpec("sink", "Huge sink", (5.0, 0.3, 0.3), "primary", requires_access=True)
    wet = solver._plan_item(g, BATHROOM_SPECS["compact_shower"], "compact_shower", "shower", solver.AxisWall.NORTH, 0.8, "primary", required=True)
    assert solver._tiny_bathroom_sink_centers(g, wet, solver.AxisWall.WEST, huge_sink, g.doors[0]) == []
    assert solver._tiny_bathroom_sink_wall_order(solver.AxisWall.WEST, solver.AxisWall.EAST) == [solver.AxisWall.NORTH, solver.AxisWall.SOUTH]

    for wall in solver.all_walls():
        point = solver._door_point(g, solver.LocalDoor(wall, 0.4, 0.7))
        assert len(point) == 2
        center = solver._wall_center_point(g, wall, 0.4)
        assert len(center) == 2
    assert solver._corner_shower_target_corner(solver.AxisWall.NORTH, solver.AxisWall.WEST) == "north_west"
    assert solver._corner_shower_target_corner(solver.AxisWall.EAST, solver.AxisWall.SOUTH) == "north_east"
    assert solver._corner_shower_yaw(solver.AxisWall.EAST, solver.AxisWall.NORTH) == 180.0
    assert solver._first_added([{"category": "sink"}], "toilet") is None

    wall_by_axis = {}
    short = SimpleNamespace(length=1.0)
    long = SimpleNamespace(length=2.0)
    solver._keep_longest_wall(wall_by_axis, solver.AxisWall.SOUTH, short)
    solver._keep_longest_wall(wall_by_axis, solver.AxisWall.SOUTH, long)
    solver._keep_longest_wall(wall_by_axis, solver.AxisWall.SOUTH, short)
    assert wall_by_axis[solver.AxisWall.SOUTH] is long

    assert solver._local_door_from_opening(ctx("bathroom", 2.0, 2.0), {}, {"wall_id": "missing"}, 0.0, 0.0) is None
    assert solver._add_wall_mount_near(FakeEngine(ctx("bathroom", 2.0, 2.0)), SimpleNamespace(walls=[]), {"id": "x", "meta": {}}, BATHROOM_SPECS["mirror"], z_center=1.4) is None

    monkeypatch.setattr(solver, "_has_real_asset", lambda spec: True)
    c = ctx("toilet", 1.4, 2.0)
    engine = FakeEngine(c)
    toilet_anchor = {"id": "toilet", "category": "toilet", "position_m": [0.7, 1.4, 0.4], "size_m": [0.4, 0.6, 0.8], "meta": {"wall_id": "w2", "wall_along_m": 0.7}}
    solver._add_toilet_accessories(engine, c, "very_high", toilet_anchor, sink=None)
    assert any(item.get("name") == "Small wall shelf" for item in engine.placements)


def test_sanitary_solver_additional_clearance_towel_and_candidate_edges(monkeypatch):
    g = geom("bathroom", 2.0, 2.0)
    all_door_geom = solver.LocalGeometry(
        x0=0.0,
        y0=0.0,
        width=2.0,
        depth=2.0,
        doors=(
            solver.LocalDoor(solver.AxisWall.NORTH, 1.0, 0.7),
            solver.LocalDoor(solver.AxisWall.WEST, 1.0, 0.7),
            solver.LocalDoor(solver.AxisWall.EAST, 1.0, 0.7),
        ),
        wall_by_axis=g.wall_by_axis,
    )
    zones = solver._door_clearance_zones(all_door_geom)
    assert zones[0].y1 == pytest.approx(1.0)
    assert zones[1].x1 == pytest.approx(0.0)
    assert zones[2].x1 == pytest.approx(1.0)

    c = ctx("bathroom", 1.2, 1.8)
    tiny_geom = solver._local_geometry_from_context(c)
    assert tiny_geom is not None
    engine = FakeEngine(c)
    no_wall_bathing = {"id": "shower", "category": "shower", "position_m": [0.8, 1.7, 0.02], "meta": {}}
    sink = {"id": "sink", "category": "sink", "position_m": [0.6, 0.7, 0.7], "meta": {"solver_wall": "west"}}
    assert solver._add_tiny_bathroom_towel_rack(engine, c, no_wall_bathing, sink) is None

    bad_rect_bathing = {
        "id": "shower",
        "category": "shower",
        "position_m": [0.8, 1.7, 0.02],
        "meta": {"solver_rect": {"x1": "bad"}, "solver_wall": "north"},
    }
    assert solver._free_towel_along_for_tiny_bathroom(tiny_geom, solver.AxisWall.WEST, bad_rect_bathing, sink, tiny_geom.wall_by_axis[solver.AxisWall.WEST]) is not None

    blocked_bathing = {
        "id": "shower",
        "category": "shower",
        "position_m": [0.8, 1.7, 0.02],
        "meta": {"solver_rect": {"x1": 0.0, "y1": 0.0, "x2": 1.2, "y2": 0.01}, "solver_wall": "north"},
    }
    assert solver._free_towel_along_for_tiny_bathroom(tiny_geom, solver.AxisWall.WEST, blocked_bathing, sink, tiny_geom.wall_by_axis[solver.AxisWall.WEST]) is None
    narrow_geom = solver.LocalGeometry(0.0, 0.0, 0.1, 2.0, tiny_geom.doors, tiny_geom.wall_by_axis)
    assert solver._free_towel_along_for_tiny_bathroom(narrow_geom, solver.AxisWall.NORTH, bad_rect_bathing, sink, tiny_geom.wall_by_axis[solver.AxisWall.NORTH]) is None

    no_along_engine = FakeEngine(c)
    monkeypatch.setattr(solver, "_free_towel_along_for_tiny_bathroom", lambda *_args, **_kwargs: None)
    bathing = {"id": "shower", "category": "shower", "position_m": [0.8, 1.7, 0.02], "meta": {"solver_wall": "north"}}
    assert solver._add_tiny_bathroom_towel_rack(no_along_engine, c, bathing, sink) is None

    wall = next(iter(tiny_geom.wall_by_axis.values()))
    direct_door = solver._local_door_from_opening(c, {solver.AxisWall.NORTH: wall}, {"wall_id": wall.id, "center": 999, "width": 0.6}, 0.0, 0.0)
    assert direct_door is not None
    assert direct_door.center <= solver._wall_length(c.width_m, c.depth_m, solver.AxisWall.NORTH)

    small_toilet = build_room_context(room_scene("toilet", 0.86, 1.1))
    small_geom = solver._local_geometry_from_context(small_toilet)
    assert small_geom is not None
    toilet_candidate = next(solver._generate_toilet_candidates(small_toilet, small_geom, "normal", random.Random(1)))
    assert "sink" not in toilet_candidate.categories()

    bathroom_area_small = ctx("bathroom", 1.5, 2.2)
    bathroom_area_geom = solver._local_geometry_from_context(bathroom_area_small)
    assert bathroom_area_geom is not None
    assert next(solver._generate_bathroom_candidates(bathroom_area_small, bathroom_area_geom, "normal", random.Random(1)))

    bathroom_narrow = ctx("bathroom", 0.9, 4.3)
    bathroom_narrow_geom = solver._local_geometry_from_context(bathroom_narrow)
    assert bathroom_narrow_geom is not None
    assert next(solver._generate_bathroom_candidates(bathroom_narrow, bathroom_narrow_geom, "normal", random.Random(2)))

    assert solver._tiny_bathroom_sink_specs(ctx("bathroom", 0.9, 1.2))[0][0] == "micro_sink"
