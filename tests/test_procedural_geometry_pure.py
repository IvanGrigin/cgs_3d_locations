from __future__ import annotations

import pytest

from src.pipeline.procedural_rooms import geometry as g


def square() -> list[g.Vec2]:
    return [g.Vec2(0, 0), g.Vec2(2, 0), g.Vec2(2, 2), g.Vec2(0, 2)]


def test_vec2_aabb_and_numeric_edges() -> None:
    assert g.Vec2(2, 0).cross(g.Vec2(0, 3)) == 6
    assert g.Vec2(0, 0).normalized() == g.Vec2(1, 0)
    assert g.Vec2(1.5, 2.5).as_list() == [1.5, 2.5]
    assert g.as_float(object(), 4.5) == 4.5
    assert g.point_from_mapping("bad") == g.Vec2(0, 0)

    a = g.AABB(0, 1, 0, 1, 0, 1)
    assert not a.intersects_3d(g.AABB(2, 3, 2, 3, 0, 1))
    assert not a.intersects_3d(g.AABB(0, 1, 0, 1, 2, 3))
    assert a.intersects_3d(g.AABB(0.5, 1.5, 0.5, 1.5, 0.5, 1.5))


def test_polygon_helpers_and_angles() -> None:
    closed = [{"x": 0, "y": 0}, {"x": 1, "z": 0}, {"x": 0, "y": 0}]
    assert len(g.polygon_from_json(closed)) == 2
    assert g.polygon_area([g.Vec2(0, 0), g.Vec2(1, 0)]) == 0
    assert g.polygon_area(square()) == pytest.approx(4.0)
    assert g.signed_polygon_area([g.Vec2(0, 0), g.Vec2(1, 0)]) == 0
    assert g.signed_polygon_area(square()) == pytest.approx(4.0)
    assert g.polygon_centroid([]) == g.Vec2(0, 0)
    assert g.polygon_centroid([g.Vec2(0, 0), g.Vec2(1, 0)]) == g.Vec2(0.5, 0)
    assert g.polygon_bounds([]) == (0, 0, 0, 0)
    assert not g.point_in_polygon(g.Vec2(0, 0), [g.Vec2(0, 0), g.Vec2(1, 0)])

    class NegativeModulo:
        def __mod__(self, _other):
            return -5.0

    assert g.normalize_angle_deg(NegativeModulo()) == 355.0


def test_wall_segment_building_selection_and_clearance() -> None:
    poly = square()
    walls = g.build_wall_segments(
        poly,
        walls_json=[
            {"id": "bad", "from_vertex": "x", "to_vertex": "y"},
            {"id": "outside", "from_vertex": 99, "to_vertex": 1},
            {"id": "w1", "from_vertex": 1, "to_vertex": 2},
        ],
        doors=[{"wall_id": "bad"}],
        windows=[{"wall_id": "w1"}],
    )
    assert [wall.id for wall in walls] == ["bad", "w1"]
    assert walls[0].has_door is True
    assert walls[1].has_window is True

    generated = g.build_wall_segments(poly)
    assert len(generated) == 4
    assert g.choose_longest_wall([], min_length=1.0) is None
    assert g.choose_longest_wall(walls, avoid_doors=True, avoid_windows=True) in walls
    assert g.choose_wall_most_opposite([walls[0]], walls[0], poly) == walls[0]

    clearance = g.opening_clearance_aabb(walls[0], {"width": 0.5, "s": 0.1}, poly, clearance_depth=0.6)
    assert clearance is not None
    assert clearance.z_max > clearance.z_min


def test_footprint_and_corner_helpers() -> None:
    poly = square()
    assert g.object_footprint_inside_polygon(g.Vec2(1, 1), [0.5, 0.5], 0, [])
    assert not g.object_footprint_inside_polygon(g.Vec2(5, 5), [1, 1], 0, poly, tolerance=0.01)
    corners = g.nearest_corner_candidates(poly, inset=0.1)
    assert len(corners) == 4
    assert g.nearest_corner_candidates([]) == []
