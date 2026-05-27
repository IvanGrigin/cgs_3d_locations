from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = str(ROOT / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src.pipeline import infinigen_scene_improvers as imp  # noqa: E402


def test_scene_room_polygon_and_primary_curtain_model() -> None:
    room = {"floor_polygon": [{"x": 0, "y": 0}, {"x": 4, "y": 0}, {"x": 4, "y": 3}, {"x": 0, "y": 3}]}
    polygon = imp._scene_room_polygon(room)
    assert polygon == [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)]

    assert imp._is_primary_plain_curtain_model({"asset_local_path": "/tmp/shtora.fbx", "title": "Штора"}) is True
    assert imp._is_primary_plain_curtain_model({"asset_local_path": "/tmp/grommet_window.fbx", "title": "Шторка"}) is False


def test_select_curtain_products_is_rank_and_deterministic(tmp_path: Path) -> None:
    catalog = [
        {"sku": "c1", "name": "plain curtain", "price": 100, "category": "шторы"},
        {"sku": "c2", "name": "blackout curtain", "price": 200, "category": "шторы"},
        {"sku": "c3", "name": "random", "price": 150, "category": "curtains"},
    ]
    style = {"style_hint": "scandinavian", "room_type": "bedroom"}
    result = imp.select_curtain_products(catalog, count=2, style_profile=style, seed=42)
    assert len(result) == 2
    assert result[0]["sku"] in {"c1", "c2"}
    second_run = imp.select_curtain_products(catalog, count=2, style_profile=style, seed=42)
    assert second_run == result


def test_apply_curtains_to_scene_adds_item_and_keeps_room(tmp_path: Path) -> None:
    tex = tmp_path / "tex.jpg"
    tex.write_text("x", encoding="utf-8")
    catalog = [
        {
            "sku": "c1",
            "name": "linen",
            "product_url": "https://example.com/1",
            "local_image_paths": [str(tex.name)],
        }
    ]
    room = {
        "floor_polygon": [{"x": 0, "y": 0}, {"x": 3, "y": 0}, {"x": 3, "y": 3}, {"x": 0, "y": 3}],
        "width": 3,
        "height": 3,
        "ceiling_height": 2.8,
        "walls": [
            {"id": "w0", "from_vertex": 0, "to_vertex": 1},
            {"id": "w1", "from_vertex": 1, "to_vertex": 2},
            {"id": "w2", "from_vertex": 2, "to_vertex": 3},
            {"id": "w3", "from_vertex": 3, "to_vertex": 0},
        ],
        "windows": [
            {
                "id": "win1",
                "wall_id": "w0",
                "s": 0.5,
                "width": 1.0,
                "height": 1.0,
                "z0": 0.4,
            }
        ],
    }
    scene = {"room": room, "placements": []}

    out, info = imp.apply_curtains_to_scene(
        deepcopy(scene),
        catalog=catalog,
        catalog_base_dir=tmp_path,
        seed=3,
    )
    assert info["added_count"] == 1
    assert len(out["placements"]) == 1
    assert out["placements"][0]["id"] == "curtain_win1"


def test_repair_furniture_intersections_no_room_polygon_shortcut() -> None:
    scene = {
        "room": {"id": "r1"},
        "placements": [
            {
                "id": "desk",
                "name": "desk",
                "size_m": [1, 1, 1],
                "aabb": {"x_min": 0, "x_max": 0.8, "y_min": 0, "y_max": 0.8, "z_min": 0, "z_max": 1},
            }
        ],
    }
    out, info = imp.repair_furniture_intersections_in_scene(scene)
    assert info["skipped_reason"] == "no_room_polygon_or_no_placements"
    assert out == scene


def test_repair_furniture_intersections_shifts_overlapping_movable_items() -> None:
    room = {
        "floor_polygon": [{"x": 0, "y": 0}, {"x": 4, "y": 0}, {"x": 4, "y": 4}, {"x": 0, "y": 4}]
    }
    scene = {
        "room": room,
        "placements": [
            {
                "id": "chair_01",
                "name": "chair",
                "size_m": [1.0, 1.0, 1.0],
                "aabb": {
                    "x_min": 1.0,
                    "x_max": 2.0,
                    "y_min": 1.0,
                    "y_max": 2.0,
                    "z_min": 0.0,
                    "z_max": 1.0,
                },
            },
            {
                "id": "chair_02",
                "name": "chair",
                "size_m": [1.0, 1.0, 1.0],
                "aabb": {
                    "x_min": 1.4,
                    "x_max": 2.4,
                    "y_min": 1.4,
                    "y_max": 2.4,
                    "z_min": 0.0,
                    "z_max": 1.0,
                },
            },
        ],
    }
    out, info = imp.repair_furniture_intersections_in_scene(scene, max_passes=1, search_step_m=0.5, max_shift_m=0.8)
    assert info["enabled"] is True
    assert info["passes"][0]["trouble_count"] >= 1
    assert len(out["placements"]) == 2


def test_geometry_helpers_chandelier_and_door_passage_paths() -> None:
    poly = [(0.0, 0.0), (3.0, 0.0), (3.0, 2.0), (0.0, 2.0)]
    assert imp._to_float("bad") is None
    assert imp._poly_bounds(poly) == (0.0, 3.0, 0.0, 2.0)
    assert imp._point_in_poly_xy(1.0, 1.0, poly)
    assert not imp._point_in_poly_xy(4.0, 1.0, poly)
    assert imp._dist_point_segment_xy(1.0, 1.0, 0.0, 0.0, 0.0, 0.0) > 0
    assert imp._dist_to_poly_edges_xy(1.5, 1.0, poly) == 1.0
    assert imp._room_sample_points(poly, step=1.0, wall_margin=0.25)
    assert imp._circle_overlap_area(1.0, 3.0) == 0.0
    assert imp._circle_overlap_area(1.0, 0.0) == 3.141592653589793
    centers, radius, overlap = imp._select_chandelier_centers(
        count=2,
        candidate_points=[(0.5, 0.5), (2.5, 0.5), (2.5, 1.5), (0.5, 1.5)],
        coverage_points=imp._room_sample_points(poly, step=1.0),
        centroid=(1.5, 1.0),
    )
    assert len(centers) == 2
    assert radius >= 0.0
    assert overlap >= 0.0
    assert imp._is_chandelier_item({"category": "ceiling_light", "name": "small chandelier"})
    assert not imp._is_chandelier_item({"category": "floor_lamp", "name": "floorlamp chandelier"})

    scene = {
        "room": {"floor_polygon": [{"x": 0, "y": 0}, {"x": 1.2, "y": 0}, {"x": 1.2, "y": 1.2}, {"x": 0, "y": 1.2}]},
        "placements": [
            {"id": "c1", "name": "ceiling chandelier", "position_m": [0.1, 0.1, 2.5], "aabb": {"x_min": 0, "x_max": 0.2, "y_min": 0, "y_max": 0.2}},
            {"id": "c2", "name": "ceiling chandelier", "position_m": [0.2, 0.2, 2.5]},
        ],
    }
    out, info = imp.normalize_chandelier_positions_in_scene(scene, wall_clearance_m=1.0, sample_step_m=0.5)
    assert info["small_room_center_fallback"] is True
    assert info["skipped_extra_chandelier_count"] == 1
    assert out["placements"][0]["meta"]["chandelier_normalized"] is True

    no_poly, info = imp.normalize_chandelier_positions_in_scene({"room": {}, "placements": []})
    assert info["skipped_reason"] == "no_room_polygon_or_no_chandeliers"
    assert no_poly["placements"] == []

    room = {
        "floor_polygon": [{"x": 0, "y": 0}, {"x": 4, "y": 0}, {"x": 4, "y": 4}, {"x": 0, "y": 4}],
        "walls": [{"id": "w0", "from_vertex": 0, "to_vertex": 1}],
        "doors": [{"wall_id": "w0", "s": 1.0, "width": 1.0}],
    }
    passage = imp._door_clearance_context(room, imp._scene_room_polygon(room))
    assert passage["keepouts"]
    assert passage["passage_points"]
    assert imp._passage_penalty_for_group({0: (0.9, 2.1, 0.0, 1.2)}, passage) > 0
    assert imp._room_wall_segments({}, poly)["w0"] == (poly[0], poly[1])
    assert imp._opening_segment_xy({"segment": {"x1": 0, "y1": 0, "x2": 1, "y2": 0}}, {}) == ((0.0, 0.0), (1.0, 0.0))


def test_rect_support_group_and_curtain_edge_branches(tmp_path: Path) -> None:
    anchor = {
        "id": "desk",
        "category": "desk",
        "position_m": [1.0, 1.0, 0.5],
        "size_m": [1.0, 1.0, 1.0],
    }
    child = {
        "id": "book",
        "category": "book",
        "position_m": [1.0, 1.0, 1.05],
        "size_m": [0.2, 0.2, 0.1],
    }
    plant = {
        "id": "plant",
        "category": "rug",
        "position_m": [1.0, 1.0, 1.05],
        "size_m": [0.2, 0.2, 0.1],
    }
    placements = [anchor, child, plant]
    rects = [imp._item_rect_xy(item) for item in placements]
    assert rects[0] == (0.5, 1.5, 0.5, 1.5)
    assert imp._rect_area(rects[0]) == 1.0
    assert imp._rect_intersection_area(rects[0], rects[1]) > 0
    assert imp._rect_shift(rects[0], 1, -1) == (1.5, 2.5, -0.5, 0.5)
    assert imp._rect_outside_room_area((-1, 1, -1, 1), [(0, 0), (2, 0), (2, 2), (0, 2)]) > 0
    assert imp._is_movable_furniture_item(anchor)
    assert not imp._is_movable_furniture_item(plant)
    assert imp._is_support_child_candidate(child)
    assert imp._z_range(child) == (1.0, 1.1)
    assert imp._rect_contains_center(rects[0], rects[1])
    assert imp._support_child_indices(anchor_index=0, placements=placements, rects=rects) == {1}
    children = imp._support_children_by_anchor(placements=placements, rects=rects)
    assert children[0] == {1}
    assert imp._move_group_indices(0, children) == {0, 1}
    assert imp._union_rect([rects[0], rects[1]]) == (0.5, 1.5, 0.5, 1.5)
    assert imp._rect_from_points([(0, 1), (2, 3)]) == (0, 2, 1, 3)
    assert imp._rect_contains_point_with_margin((0, 1, 0, 1), 1.05, 0.5, 0.1)

    no_texture, info = imp.apply_curtains_to_scene(
        {
            "room": {
                "floor_polygon": [{"x": 0, "y": 0}, {"x": 3, "y": 0}],
                "walls": [{"id": "w0", "from_vertex": 0, "to_vertex": 1}],
                "windows": [{"id": "w", "wall_id": "w0"}],
            },
            "placements": [],
        },
        catalog=[{"sku": "missing", "local_image_paths": ["missing.jpg"]}],
        catalog_base_dir=tmp_path,
    )
    assert info["added_count"] == 0
    assert no_texture["placements"] == []

    assert imp.apply_curtains_to_scene({"placements": []}, catalog=[], catalog_base_dir=tmp_path)[1]["skipped_reason"] == "missing_room"
    assert imp.apply_curtains_to_scene({"room": {}, "placements": []}, catalog=[], catalog_base_dir=tmp_path)[1]["skipped_reason"] == "missing_windows"
    assert imp.apply_curtains_to_scene({"room": {"windows": [{}]}, "placements": []}, catalog=[], catalog_base_dir=tmp_path)[1]["skipped_reason"] == "missing_walls"
    assert imp._resolve_catalog_image({"local_image_paths": "bad"}, tmp_path) is None
    assert imp._score_curtain(
        {"name": "blackout beige atlas", "category": "шторы", "price": 10, "image_selection_note": "fallback_only_one_gallery_image"},
        "classic",
        "bedroom",
    ) > 0


def test_infinigen_improver_remaining_geometry_and_curtain_edges(tmp_path: Path) -> None:
    assert imp._scene_room_polygon({"floor_polygon": [{"x": 0, "z": 1}, ["bad"], {"x": "bad", "y": 2}]}) == [(0.0, 1.0)]
    assert not imp._point_in_poly_xy(0.0, 0.0, [(0.0, 0.0), (1.0, 0.0)])
    assert imp._dist_to_poly_edges_xy(0.0, 0.0, []) == 0.0
    assert imp._chandelier_coverage_radius([], [(0.0, 0.0)]) == 0.0
    assert imp._room_sample_points([(0.0, 0.0), (0.4, 0.0), (0.4, 0.4), (0.0, 0.4)], step=1.0, wall_margin=0.1) == [(0.2, 0.2)]
    assert imp._circle_overlap_area(0.0, 0.0) == 0.0
    assert imp._select_chandelier_centers(count=0, candidate_points=[(0, 0)], coverage_points=[(0, 0)], centroid=(0, 0)) == ([], 0.0, 0.0)
    centers, _radius, _overlap = imp._select_chandelier_centers(
        count=3,
        candidate_points=[(0.0, 0.0)],
        coverage_points=[(0.0, 0.0), (1.0, 0.0)],
        centroid=(0.0, 0.0),
    )
    assert centers == [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]

    large_scene = {
        "room": {
            "floor_polygon": [
                {"x": 0, "y": 0},
                {"x": 4, "y": 0},
                {"x": 4, "y": 4},
                {"x": 0, "y": 4},
            ]
        },
        "placements": [
            {"id": "badpos", "name": "ceiling chandelier"},
            {"id": "c1", "name": "ceiling chandelier", "position_m": [0.2, 0.2, 2.6]},
        ],
    }
    out, info = imp.normalize_chandelier_positions_in_scene(large_scene, wall_clearance_m=0.25, sample_step_m=1.0)
    assert info["candidate_point_count"] > 0
    assert out["placements"][1]["meta"]["chandelier_normalized"] is True

    concave = {
        "room": {
            "floor_polygon": [
                {"x": 0, "y": 0},
                {"x": 2, "y": 0},
                {"x": 2, "y": 0.4},
                {"x": 0.4, "y": 0.4},
                {"x": 0.4, "y": 2},
                {"x": 0, "y": 2},
            ]
        },
        "placements": [{"id": "c1", "name": "ceiling chandelier", "position_m": [0.1, 0.1, 2.6]}],
    }
    _out, skipped = imp.normalize_chandelier_positions_in_scene(concave, wall_clearance_m=2.0, sample_step_m=0.5)
    assert skipped["skipped_reason"] == "no_valid_room_points_with_wall_clearance"

    assert imp._item_rect_xy({"aabb": {"x_min": 1, "x_max": 0, "y_min": 0, "y_max": 1}}) is None
    assert imp._z_range({"aabb": {"z_min": 2, "z_max": 1}}) is None
    assert imp._support_child_indices(anchor_index=0, placements=["bad"], rects=[(0, 1, 0, 1)]) == set()
    assert imp._support_children_by_anchor(placements=[{"id": "bad"}, {"id": "child"}], rects=[None, (0, 1, 0, 1)]) == {1: set()}
    anchor = {"id": "desk", "category": "desk", "position_m": [0.5, 0.5, 0.5], "size_m": [1.0, 1.0, 1.0]}
    no_rect_child = {"id": "vase", "category": "vase", "position_m": [0.5, 0.5, 1.05], "size_m": [0.1, 0.1, 0.1]}
    outside_child = {"id": "book", "category": "book", "position_m": [2.0, 2.0, 1.05], "size_m": [0.1, 0.1, 0.1]}
    low_overlap_child = {"id": "lamp", "category": "lamp", "position_m": [1.05, 0.5, 1.05], "size_m": [0.4, 0.4, 0.1]}
    assert imp._support_child_indices(
        anchor_index=0,
        placements=[anchor, no_rect_child, outside_child, low_overlap_child],
        rects=[(0, 1, 0, 1), None, (1.95, 2.05, 1.95, 2.05), (0.85, 1.25, 0.3, 0.7)],
    ) == set()

    wall_segments = imp._room_wall_segments({"walls": [{"bad": True}, {"id": "w", "from_vertex": 0, "to_vertex": 1}]}, [(0, 0), (1, 0)])
    assert wall_segments["w"] == ((0, 0), (1, 0))
    assert imp._opening_segment_xy({"wall_id": "missing"}, wall_segments) is None
    assert imp._opening_segment_xy({"wall_id": "w", "s": "bad"}, wall_segments) is None
    assert imp._door_clearance_context({"doors": []}, [(0, 0), (1, 0), (1, 1)])["keepouts"] == []
    assert imp._passage_penalty_for_group({}, {"keepouts": [(0, 1, 0, 1)]}) == 0.0

    tex = tmp_path / "curtain.jpg"
    tex.write_bytes(b"jpg")
    model = tmp_path / "shtora.fbx"
    model.write_text("fbx", encoding="utf-8")
    scene = {
        "room": {
            "floor_polygon": [{"x": 0, "y": 0}, {"x": 3, "y": 0}, {"x": 3, "y": 3}, {"x": 0, "y": 3}],
            "walls": [{"id": "w0", "from_vertex": 0, "to_vertex": 1}],
            "openings": [
                {"id": "win", "type": "window", "wall_id": "w0", "s": 0.4, "width": 0.8, "height": 1.0, "z0": 0.8}
            ],
            "ceiling_height": 2.7,
        },
        "placements": [{"id": "bad", "category": "x"}],
    }
    out, info = imp.apply_curtains_to_scene(
        scene,
        catalog=[{"sku": "c1", "name": "plain curtain", "local_image_paths": [tex.name]}],
        catalog_base_dir=tmp_path,
        curtain_models=[{"title": "Shtora", "asset_local_path": str(model), "asset_status": "local_file"}],
        seed=1,
    )
    assert info["added_count"] == 1
    added = out["placements"][-1]
    assert added["asset"]["mesh_path"] == str(model.resolve())
    assert added["meta"]["curtain_model"]["title"] == "Shtora"

    assert imp._is_primary_plain_curtain_model("/tmp/not_shtora.obj") is False
    assert imp._is_primary_plain_curtain_model("/tmp/curtain 2/shtora.fbx") is False
    assert imp._score_curtain({"name": "однотон gray", "category": "шторы"}, "scandinavian", "living_room") > 1.0
    assert imp._score_curtain({"name": "вашим дизайном", "category": "шторы"}, "", "") < 0.0
    assert imp.select_curtain_products([], count=1) == []
    assert imp.select_curtain_products([{"sku": "x"}], count=0) == []
    assert imp._wall_points({"floor_polygon": "bad"}, {}) is None
    assert imp._wall_points({"floor_polygon": [{"x": 0, "y": 0}]}, {"from_vertex": "bad", "to_vertex": 0}) is None

    base_room = {
        "floor_polygon": [{"x": 0, "y": 0}, {"x": 3, "y": 0}, {"x": 3, "y": 3}, {"x": 0, "y": 3}],
        "walls": [{"id": "w0", "from_vertex": 0, "to_vertex": 1}],
        "windows": [{"id": "win", "wall_id": "w0", "s": 0.4, "width": 0.8, "height": 4.0, "z0": 0.1}],
        "ceiling_height_m": 3.4,
        "floor_z": 0.0,
    }
    empty_info = imp.apply_curtains_to_scene(
        {"room": base_room, "placements": []},
        catalog=[],
        catalog_base_dir=tmp_path,
    )[1]
    assert empty_info["skipped_reason"] == "empty_catalog"
    invalid_items_info = imp.apply_curtains_to_scene(
        {"room": base_room, "placements": "bad"},
        catalog=[{"sku": "c1", "local_image_paths": [tex.name]}],
        catalog_base_dir=tmp_path,
    )[1]
    assert invalid_items_info["skipped_reason"] == "invalid_items"
    skipped_window_scene = {
        "room": {**base_room, "windows": ["bad", {"id": "missing", "wall_id": "nope"}, {"id": "badwall", "wall_id": "w0"}]},
        "placements": [],
    }
    skipped_window_scene["room"]["walls"] = [{"id": "w0", "from_vertex": "bad", "to_vertex": 1}]
    skipped, skipped_info = imp.apply_curtains_to_scene(
        skipped_window_scene,
        catalog=[{"sku": "c1", "local_image_paths": [tex.name]}],
        catalog_base_dir=tmp_path,
    )
    assert skipped_info["added_count"] == 0
    assert skipped["placements"] == []

    model_path = tmp_path / "shtora.fbx"
    duplicate_scene = {
        "room": {**base_room, "openings": {"windows": base_room["windows"]}, "windows": []},
        "items": [{"id": "curtain_win"}],
    }
    out, info = imp.apply_curtains_to_scene(
        duplicate_scene,
        catalog=[{"sku": "c1", "name": "plain", "local_image_paths": [tex.name]}],
        catalog_base_dir=tmp_path,
        curtain_models=["bad"],
        curtain_model_paths=[str(model_path)],
        style_profile={"style_label": "modern"},
    )
    assert info["added_count"] == 1
    assert info["model_count"] == 1
    assert out["items"][-1]["id"] == "curtain_win_2"
    assert out["items"][-1]["asset"]["kind"] == "curtain_fbx_textured"


def test_infinigen_improver_remaining_support_and_repair_edges(monkeypatch) -> None:
    def raw_children(*, anchor_index, placements, rects):
        if anchor_index in {0, 1}:
            return {2}
        return set()

    monkeypatch.setattr(imp, "_support_child_indices", raw_children)
    placements = [
        {"id": "desk", "category": "desk", "position_m": [0.5, 0.5, 0.5], "size_m": [1.0, 1.0, 1.0]},
        {"id": "bad_anchor", "category": "desk"},
        {"id": "book", "category": "book", "position_m": [0.5, 0.5, 1.05], "size_m": [0.2, 0.2, 0.1]},
    ]
    children = imp._support_children_by_anchor(
        placements=placements,
        rects=[(0, 1, 0, 1), (0, 1, 0, 1), (0.4, 0.6, 0.4, 0.6)],
    )
    assert children[0] == {2}
    assert children[1] == set()

    monkeypatch.setattr(imp, "_support_child_indices", lambda **_kwargs: {1})
    no_child_z = imp._support_children_by_anchor(
        placements=[placements[0], {"id": "child_without_z"}],
        rects=[(0, 1, 0, 1), (0.4, 0.6, 0.4, 0.6)],
    )
    assert no_child_z == {0: set(), 1: set()}

    room = {
        "floor_polygon": [{"x": 0, "y": 0}, {"x": 4, "y": 0}, {"x": 4, "y": 4}, {"x": 0, "y": 4}]
    }
    scene = {
        "room": room,
        "placements": [
            {
                "id": "chair_1",
                "name": "chair",
                "aabb": {"x_min": 0.2, "x_max": 0.7, "y_min": 0.2, "y_max": 0.7, "z_min": 0, "z_max": 1},
            },
            {
                "id": "chair_2",
                "name": "chair",
                "aabb": {"x_min": 2.0, "x_max": 2.5, "y_min": 2.0, "y_max": 2.5, "z_min": 0, "z_max": 1},
            },
        ],
    }
    out, info = imp.repair_furniture_intersections_in_scene(scene, max_passes=2)
    assert info["passes"][0]["trouble_count"] == 0
    assert info["moved_count"] == 0
    assert out == scene
