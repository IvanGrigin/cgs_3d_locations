from __future__ import annotations

import argparse
import json

import pytest

from src.pipeline import relationship_graph_stage as rg


def aabb(x1, x2, y1, y2, z1, z2):
    return {"x_min": x1, "x_max": x2, "y_min": y1, "y_max": y2, "z_min": z1, "z_max": z2}


def relationship_scene() -> dict:
    return {
        "room": {
            "floor_polygon": [{"x": 0, "y": 0}, {"x": 4, "y": 0}, {"x": 4, "y": 4}, {"x": 0, "y": 4}],
            "ceiling_height_m": 2.8,
        },
        "items": [
            {
                "id": "desk.main",
                "name": "working desk",
                "category": "desk",
                "aabb": aabb(0.4, 1.6, 1.2, 1.9, 0.0, 0.75),
                "yaw_deg": 0,
            },
            {
                "id": "chair.main",
                "name": "office chair",
                "category": "office chair",
                "aabb": aabb(0.75, 1.25, 0.25, 0.75, 0.0, 0.85),
                "yaw_deg": 0,
            },
            {
                "id": "laptop.main",
                "name": "open laptop",
                "category": "laptop",
                "aabb": aabb(0.8, 1.2, 1.35, 1.65, 0.75, 0.85),
                "yaw_deg": 0,
            },
            {
                "id": "art.main",
                "name": "wall art poster",
                "category": "wall art",
                "aabb": aabb(0.5, 1.2, 0.02, 0.06, 1.2, 1.8),
                "placement_type": "wall",
            },
        ],
    }


def test_aabb_and_basic_utility_helpers(monkeypatch):
    box = rg.AABB.from_any(aabb(0, 2, 0, 1, 0, 0.5))
    assert box is not None
    assert box.width == 2
    assert box.depth == 1
    assert box.height == 0.5
    assert box.center == (1.0, 0.5, 0.25)
    assert box.area_xy == 2
    assert box.volume == 1
    assert box.to_dict()["x_max"] == 2
    assert box.moved_center_xy(3, 4).center_xy == (3.0, 4.0)
    assert box.moved_bottom_z(2).z_max == 2.5
    assert box.translated(1, 2, 3).center == (2.0, 2.5, 3.25)
    assert box.contains_xy(rg.AABB(0.5, 1.5, 0.2, 0.8, 0, 1))
    assert box.overlap_xy_area(rg.AABB(1, 3, 0.5, 1.5, 0, 1)) == 0.5
    assert box.overlap_volume(rg.AABB(1, 3, 0.5, 1.5, 0.25, 1.0)) == 0.125
    assert rg.AABB.from_any({"bad": 1}) is None
    assert rg.AABB.from_any({"x_min": object(), "x_max": 1, "y_min": 0, "y_max": 1, "z_min": 0, "z_max": 1}) is None

    assert rg._safe_id("desk.main №1") == "desk_main_1"
    assert rg._float_or_none("12,5 cm") == 12.5
    assert rg._float_or_none(True) is None
    assert rg._float_or_none("no digits") is None

    class BadMatch:
        def group(self, _idx):
            return "bad"

    monkeypatch.setattr(rg.re, "search", lambda *_args, **_kwargs: BadMatch())
    assert rg._float_or_none("123") is None
    assert rg._deep_text_values({"a": ["x", {"b": 2}]}) == ["x", "2"]
    assert len(rg._deep_text_values(list(range(100)), limit=3)) == 3
    assert rg._axis_offset_deg("+X") == 0
    assert rg._axis_offset_deg("-Y") == 270
    assert rg._unit_vec_from_yaw(0, "+Y") == pytest.approx((0.0, 1.0))


def test_collect_items_rule_edges_and_validation():
    scene = relationship_scene()
    items = rg.collect_items(scene, prompt="office workspace")
    by_id = {item.object_id: item for item in items}

    assert set(by_id) == {"desk_main", "chair_main", "laptop_main", "art_main"}
    assert by_id["desk_main"].semantic_group == "desk"
    assert by_id["chair_main"].semantic_group == "office_chair"
    assert by_id["laptop_main"].role == "accessory"
    assert by_id["art_main"].placement_type == "wall"
    assert rg._room_polygon(scene) == [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]

    room = rg._room_bounds_from_data(scene, items)
    assert room.to_dict() == aabb(0, 4, 0, 4, 0.0, 2.8)

    edges = rg.build_rule_based_edges(items, room)
    edge_keys = {edge.key() for edge in edges}
    assert ("chair_main", "faces", "desk_main") in edge_keys
    assert ("chair_main", "in_front_of", "desk_main") in edge_keys
    assert ("laptop_main", "on_top_of", "desk_main") in edge_keys
    assert ("art_main", "mounted_on_wall", rg.WALL_TARGET_ID) in edge_keys

    validation = rg.validate_relationship_graph(items, edges, room)
    assert validation["counts"]["item_count"] == 4
    assert validation["counts"]["edge_count"] == len(edges)
    assert validation["relation_scores"]["support"] is not None
    assert validation["score"] > 0.5


def test_relationship_graph_builds_applies_and_annotates(tmp_path):
    data, info = rg.build_relationship_graph(
        relationship_scene(),
        prompt="рабочая зона",
        options=rg.StageOptions(apply_placement=True, validate=True),
    )

    assert data["relationship_graph"]["schema"] == rg.GRAPH_SCHEMA
    assert info["schema"] == rg.SCHEMA
    assert info["summary"]["item_count"] == 4
    assert info["summary"]["edge_count"] >= 4
    assert data["items"][0]["meta"]["relationship_graph"]["semantic_group"] == "desk"
    assert any(edge["relation_type"] == "on_top_of" for edge in data["relationship_graph"]["edges"])

    args = argparse.Namespace(
        relationship_graph=True,
        relationship_graph_apply_placement=False,
        relationship_graph_no_validate=False,
        relationship_graph_no_repair=False,
        relationship_graph_add_missing_supports=False,
        relationship_graph_min_score=0.75,
    )
    input_path = tmp_path / "scene.json"
    input_path.write_text(json.dumps(relationship_scene()), encoding="utf-8")
    out_path, stage_info = rg.maybe_apply_relationship_graph_stage(
        args=args,
        run_dir=tmp_path,
        scene_json_path=input_path,
        prompt_text="office",
        tag="unit",
    )
    assert out_path.is_file()
    assert stage_info is not None
    assert stage_info["tag"] == "unit"
    assert stage_info["output_scene_json"] == str(out_path)


def test_relation_dedupe_and_llm_resolution():
    items = rg.collect_items(relationship_scene(), prompt="office")
    source = next(item for item in items if item.object_id == "chair_main")
    target = next(item for item in items if item.object_id == "desk_main")

    duplicate_edges = rg._dedupe_edges(
        [
            rg.RelationEdge(source.object_id, "faces", target.object_id, "bad_class", "soft", 0.4, {"a": 1}, "rule"),
            rg.RelationEdge(source.object_id, "faces", target.object_id, "orientation", "hard", 0.8, {"b": 2}, "llm"),
            rg.RelationEdge(source.object_id, "not_a_relation", target.object_id, "orientation"),
            rg.RelationEdge(source.object_id, "inside", target.object_id, "bad_class", "urgent"),
        ]
    )
    assert len(duplicate_edges) == 2
    assert duplicate_edges[0].constraint_level == "hard"
    assert duplicate_edges[0].weight == 0.8
    assert duplicate_edges[0].params == {"a": 1, "b": 2}
    assert duplicate_edges[0].relation_class == "orientation"
    assert duplicate_edges[1].relation_class == "containment"
    assert duplicate_edges[1].constraint_level == "hard"

    llm_edges = rg._resolve_llm_relations(
        [
            {
                "from_object_id": "chair_main",
                "to_object_id": "desk_main",
                "relation_type": "faces",
                "constraint_level": "soft",
                "weight": "0.6",
                "params": {"tolerance_deg": 20},
            },
            {"from_group": "laptop", "to_group": "desk", "relation": "on_top_of"},
            {"from_object_id": "chair_main", "to_object_id": rg.ROOM_CENTER_ID, "relation": "faces"},
            {"relation": "bad"},
        ],
        items,
    )
    assert {edge.key() for edge in llm_edges} == {
        ("chair_main", "faces", "desk_main"),
        ("laptop_main", "on_top_of", "desk_main"),
        ("chair_main", "faces", rg.ROOM_CENTER_ID),
    }


def test_anchor_generation_and_relation_aware_placement_branches():
    scene = {
        "room": {"floor_polygon": [[0, 0], [5, 0], [5, 4], [0, 4]], "ceiling_height": 3.0},
        "items": [
            {"id": "bed", "name": "bed", "category": "bed", "aabb": aabb(-0.5, 1.0, 1.0, 2.6, 0, 0.65), "yaw_deg": 0},
            {"id": "pillow", "name": "pillow", "category": "pillow", "aabb": aabb(0, 0.4, 0, 0.3, 0, 0.15)},
            {"id": "rug", "name": "rug", "category": "rug", "aabb": aabb(4.8, 5.6, 3.6, 4.4, 0, 0.02)},
            {"id": "art", "name": "wall art", "category": "wall art", "placement_type": "wall", "aabb": aabb(4.0, 4.6, 3.9, 4.0, 1.1, 1.6)},
            {"id": "chair_a", "name": "chair", "category": "chair", "aabb": aabb(0, 0.5, 0, 0.5, 0, 0.8), "yaw_deg": 180},
            {"id": "table", "name": "dining table", "category": "dining table", "aabb": aabb(2.0, 3.0, 1.5, 2.2, 0, 0.75)},
        ],
    }
    items = rg.collect_items(scene, prompt="sleep dining work")
    by_id = {it.object_id: it for it in items}
    bed_anchors = rg.generate_anchors_for_item(by_id["bed"])
    assert "top.pillow_left" in bed_anchors
    assert rg._local_to_world_anchor(by_id["bed"], bed_anchors["top.center"])[2] == pytest.approx(0.65)
    assert rg._anchor_name_for_placement_area("pillow_area", by_id["bed"], by_id["pillow"]) == "top.pillow_left"

    room = rg._room_bounds_from_data(scene, items)
    edges = [
        rg._edge(by_id["pillow"], "on_top_of", by_id["bed"], params={"placement_area": "pillow_area"}),
        rg._edge(by_id["rug"], "under", by_id["table"]),
        rg._edge(by_id["chair_a"], "around", by_id["table"]),
        rg._edge(by_id["chair_a"], "faces", rg.ROOM_CENTER_ID, params={"target_xy": [2.5, 2.0]}),
        rg._edge(by_id["art"], "above", by_id["bed"], params={"vertical_gap_m": {"min": 0.2, "max": 0.4}}),
    ]
    info = rg.apply_relation_aware_placement(scene, items, edges, room, rg.StageOptions(apply_placement=True))
    assert info["applied"] is True
    assert info["changed_count"] >= 4
    assert by_id["pillow"].aabb.z_min > by_id["bed"].aabb.z_max
    assert by_id["rug"].aabb.z_min == pytest.approx(0.002)
    assert 0 <= by_id["chair_a"].yaw_deg <= 360
    assert by_id["bed"].aabb.x_min >= room.x_min

    skipped = rg.apply_relation_aware_placement(scene, items, edges, room, rg.StageOptions(apply_placement=False))
    assert skipped == {"applied": False, "changed_count": 0, "changes": []}


def test_validation_error_paths_cli_and_relation_loading(tmp_path, capsys):
    scene = relationship_scene()
    items = rg.collect_items(scene, prompt="office")
    by_id = {it.object_id: it for it in items}
    room = rg._room_bounds_from_data(scene, items)
    invalid_edges = [
        rg.RelationEdge("missing", "faces", "desk_main", "orientation"),
        rg.RelationEdge("laptop_main", "on_top_of", "chair_main", "support", "hard"),
        rg.RelationEdge("laptop_main", "on_top_of", "desk_main", "support", "hard"),
        rg.RelationEdge("desk_main", "on_top_of", "laptop_main", "support", "hard"),
        rg.RelationEdge("chair_main", "faces", rg.ROOM_CENTER_ID, "orientation", "hard", params={"target_xy": [0, 0]}),
    ]
    validation = rg.validate_relationship_graph(items, invalid_edges, room)
    problems = {row.get("problem") for row in validation["errors"]}
    assert "from_object_missing" in problems
    assert "invalid_support_target" in problems
    assert validation["counts"]["error_count"] >= 2
    assert rg._detect_support_cycle({"a": "b", "b": "c", "c": "a"}) == ["a", "b", "c", "a"]
    assert rg._score_wall_relation(by_id["art_main"], room) > 0.8

    parser = argparse.ArgumentParser()
    rg.add_relationship_graph_arguments(parser)
    parsed = parser.parse_args(["--relationship-graph", "--relationship-graph-apply-placement", "--relationship-graph-no-validate"])
    assert parsed.relationship_graph is True
    assert parsed.relationship_graph_no_validate is True

    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("office prompt", encoding="utf-8")
    assert rg._read_prompt_from_cli(argparse.Namespace(prompt_file=str(prompt_file), prompt="fallback")) == "office prompt"
    assert rg._read_prompt_from_cli(argparse.Namespace(prompt_file=str(tmp_path / "missing.txt"), prompt="fallback")) == "fallback"

    relations_path = tmp_path / "relations.json"
    relations_path.write_text(json.dumps({"edges": [{"relation": "faces"}, "bad"]}), encoding="utf-8")
    assert rg._load_llm_relations(str(relations_path)) == [{"relation": "faces"}]
    list_relations = tmp_path / "relations_list.json"
    list_relations.write_text(json.dumps([{"relation": "near"}, 3]), encoding="utf-8")
    assert rg._load_llm_relations(str(list_relations)) == [{"relation": "near"}]
    with pytest.raises(FileNotFoundError):
        rg._load_llm_relations(str(tmp_path / "none.json"))
    bad_relations = tmp_path / "bad_relations.json"
    bad_relations.write_text(json.dumps({"x": []}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="LLM relations JSON"):
        rg._load_llm_relations(str(bad_relations))

    input_path = tmp_path / "scene.json"
    out_path = tmp_path / "out.json"
    input_path.write_text(json.dumps(scene), encoding="utf-8")
    rc = rg.main([
        "--input",
        str(input_path),
        "--out",
        str(out_path),
        "--prompt-file",
        str(prompt_file),
        "--llm-relations-json",
        str(relations_path),
        "--apply-placement",
    ])
    assert rc == 0
    assert out_path.is_file()
    assert "relationship_graph_stage:" in capsys.readouterr().out


def test_rule_based_edges_cover_all_major_room_zones_and_relation_helpers():
    def item(object_id, category, box, zone, **extra):
        return {
            "id": object_id,
            "name": category,
            "category": category,
            "aabb": box,
            "zone_id": zone,
            "yaw_deg": extra.pop("yaw", 0),
            **extra,
        }

    scene = {
        "room": {"floor_polygon": [[0, 0], [8, 0], [8, 7], [0, 7]], "ceiling_height_m": 3.0},
        "items": [
            item("desk", "desk", aabb(0.5, 1.9, 0.5, 1.2, 0, 0.75), "work_zone"),
            item("office_chair", "office chair", aabb(0.8, 1.2, 1.5, 2.0, 0, 0.9), "work_zone"),
            item("monitor", "monitor", aabb(0.9, 1.4, 0.55, 0.65, 0.75, 1.2), "work_zone"),
            item("keyboard", "keyboard", aabb(0.9, 1.4, 0.9, 1.0, 0.75, 0.8), "work_zone"),
            item("mouse", "mouse", aabb(1.45, 1.6, 0.9, 1.0, 0.75, 0.8), "work_zone"),
            item("mug", "mug", aabb(1.55, 1.7, 0.7, 0.85, 0.75, 0.9), "work_zone"),
            item("dining_table", "dining table", aabb(3.0, 4.4, 0.7, 1.7, 0, 0.75), "dining_zone"),
            item("dining_chair_a", "dining chair", aabb(3.2, 3.7, 2.0, 2.5, 0, 0.9), "dining_zone"),
            item("plate", "plate", aabb(3.25, 3.55, 0.95, 1.25, 0.75, 0.78), "dining_zone"),
            item("bowl", "bowl", aabb(3.7, 4.0, 0.95, 1.25, 0.75, 0.9), "dining_zone"),
            item("vase", "vase", aabb(3.65, 3.9, 1.2, 1.45, 0.75, 1.2), "dining_zone"),
            item("bed", "bed", aabb(0.5, 2.5, 3.0, 5.0, 0, 0.65), "sleeping_zone"),
            item("nightstand_l", "nightstand", aabb(0.0, 0.4, 3.0, 3.6, 0, 0.6), "sleeping_zone"),
            item("nightstand_r", "nightstand", aabb(2.6, 3.0, 3.0, 3.6, 0, 0.6), "sleeping_zone"),
            item("pillow", "pillow", aabb(0.9, 1.4, 3.2, 3.6, 0.65, 0.85), "sleeping_zone"),
            item("blanket", "blanket", aabb(0.7, 2.3, 3.8, 4.8, 0.65, 0.72), "sleeping_zone"),
            item("phone", "phone", aabb(2.7, 2.9, 3.2, 3.4, 0.6, 0.65), "sleeping_zone"),
            item("sofa", "sofa", aabb(4.7, 6.8, 3.0, 3.9, 0, 0.85), "living_zone"),
            item("coffee_table", "coffee table", aabb(5.0, 6.0, 4.3, 5.0, 0, 0.45), "living_zone"),
            item("tv", "tv", aabb(5.1, 5.9, 6.8, 6.9, 1.0, 1.7), "living_zone", placement_type="wall"),
            item("tv_stand", "tv stand", aabb(4.9, 6.1, 6.2, 6.6, 0, 0.5), "living_zone"),
            item("remote", "remote", aabb(5.2, 5.4, 4.5, 4.7, 0.45, 0.5), "living_zone"),
            item("rug", "rug", aabb(4.5, 6.5, 4.0, 5.4, 0, 0.03), "living_zone"),
            item("plant", "plant", aabb(7.2, 7.7, 0.5, 1.0, 0, 1.2), "living_zone"),
            item("wardrobe", "wardrobe", aabb(7.0, 7.8, 2.0, 3.4, 0, 2.2), "storage_zone"),
            item("shelf", "shelf", aabb(6.6, 7.7, 3.8, 4.2, 0, 1.8), "storage_zone"),
            item("book", "book", aabb(6.8, 7.0, 3.85, 4.05, 1.0, 1.08), "storage_zone"),
            item("dresser", "dresser", aabb(0.3, 1.5, 5.8, 6.4, 0, 0.9), "sleeping_zone"),
            item("mirror", "mirror", aabb(0.5, 1.3, 6.45, 6.5, 1.0, 1.8), "sleeping_zone", placement_type="wall"),
            item("wall_art", "wall art", aabb(1.0, 2.0, 5.05, 5.1, 1.2, 1.7), "sleeping_zone", placement_type="wall"),
            item("sink", "sink", aabb(3.0, 3.8, 5.8, 6.3, 0, 0.9), "bathroom_zone"),
            item("toilet", "toilet", aabb(4.2, 4.8, 5.8, 6.5, 0, 0.8), "bathroom_zone"),
            item("soap", "soap dispenser", aabb(3.1, 3.25, 5.95, 6.1, 0.9, 1.1), "bathroom_zone"),
            item("toothbrush", "toothbrush cup", aabb(3.35, 3.5, 5.95, 6.1, 0.9, 1.1), "bathroom_zone"),
        ],
    }

    items = rg.collect_items(scene, prompt="office dining bedroom living bathroom storage")
    by_id = {it.object_id: it for it in items}
    room = rg._room_bounds_from_data(scene, items)
    edges = rg.build_rule_based_edges(items, room)
    keys = {edge.key() for edge in edges}

    assert ("office_chair", "in_front_of", "desk") in keys
    assert ("office_chair", "faces", "desk") in keys
    assert ("dining_chair_a", "around", "dining_table") in keys
    assert ("monitor", "on_top_of", "desk") in keys
    assert ("plate", "on_top_of", "dining_table") in keys
    assert ("pillow", "on_top_of", "bed") in keys
    assert ("nightstand_l", "next_to", "bed") in keys
    assert ("coffee_table", "in_front_of", "sofa") in keys
    assert ("sofa", "faces", "tv") in keys
    assert ("tv", "mounted_on_wall", rg.WALL_TARGET_ID) in keys
    assert ("tv", "visible_from", "sofa") in keys
    assert ("tv", "above", "tv_stand") in keys
    assert ("wardrobe", "against_wall", rg.WALL_TARGET_ID) in keys
    assert ("mirror", "above", "dresser") in keys
    assert ("wall_art", "mounted_on_wall", rg.WALL_TARGET_ID) in keys
    assert any(key[0] == "rug" and key[1] == "under" for key in keys)
    assert ("plant", "near", rg.WALL_TARGET_ID) in keys
    assert ("toilet", "against_wall", rg.WALL_TARGET_ID) in keys
    assert ("soap", "on_top_of", "sink") in keys
    assert ("toothbrush", "on_top_of", "sink") in keys

    assert rg._support_targets_for_accessory(by_id["mug"]) == {"desk"}
    assert "nightstand" in rg._support_targets_for_accessory(by_id["phone"])
    assert "shelf" in rg._support_targets_for_accessory(by_id["book"])
    assert rg._default_placement_area(by_id["keyboard"], by_id["desk"]) == "front_center"
    assert rg._default_placement_area(by_id["phone"], by_id["nightstand_r"]) == "front"
    assert rg._default_placement_area(by_id["blanket"], by_id["bed"]) == "center"
    assert rg._default_placement_area(by_id["remote"], by_id["coffee_table"]) == "center"
    assert rg._infer_side_preference(by_id["nightstand_l"], by_id["bed"]) == "left"
    assert rg._sofa_orientation_target(by_id["sofa"], items).object_id == "tv"

    chair = by_id["office_chair"]
    desk = by_id["desk"]
    for relation, side in [("left_of", "left"), ("right_of", "right"), ("behind", "back"), ("next_to", "auto")]:
        moved = rg._aabb_for_proximity_relation(
            chair,
            desk,
            rg._edge(chair, relation, desk, params={"side_preference": side, "distance_m": {"min": 0.1, "max": 0.2}}),
        )
        assert moved is not None
    assert rg._aabb_for_around_relation(by_id["dining_chair_a"], by_id["dining_table"], rg._edge(by_id["dining_chair_a"], "around", by_id["dining_table"]), items) is not None

    data, info = rg.build_relationship_graph(scene, prompt="office dining bedroom", options=rg.StageOptions(apply_placement=True, validate=True))
    assert info["summary"]["edge_count"] >= len(edges)
    assert data["relationship_stage"]["validation"]["counts"]["edge_count"] >= len(edges)


def test_helper_branches_for_collection_zones_and_raw_updates(tmp_path: Path):
    scene = {
        "placements": [
            "ignored",
            {
                "id": "rot.list",
                "category": "chair",
                "role": "primary",
                "rotation": [0, 0, 45],
                "aabb": aabb(0, 1, 0, 1, 0, 1),
                "constraints": {"mount_type": "wall"},
            },
            {
                "id": "rot.short",
                "category": "chair",
                "rotation": [270],
                "aabb": aabb(1, 2, 0, 1, 0, 1),
                "constraints": {"mount_type": "ceiling"},
            },
            {"id": "ward", "category": "wardrobe", "aabb": aabb(0, 1, 2, 3, 0, 2)},
            {"id": "sofa", "category": "sofa", "aabb": aabb(2, 3, 2, 3, 0, 1)},
            {"id": "dish", "category": "plate", "aabb": aabb(3, 3.2, 0, 0.2, 0.7, 0.8)},
            {"id": "kit", "category": "kitchen set", "aabb": aabb(3, 4, 1, 2, 0, 2)},
            {"id": "sink", "category": "sink", "aabb": aabb(4, 4.8, 1, 2, 0, 1)},
            {"id": "decor", "semantic_group": "small_decor", "aabb": aabb(4, 4.2, 3, 3.2, 0, 0.2), "meta": "not-dict"},
            {"id": "alias", "semantic_group": "bookshelf", "aabb": aabb(0, 1, 4, 5, 0, 1)},
        ]
    }

    assert rg._items_key({}) == "items"
    assert rg.collect_items({"items": "bad"}) == []
    items = rg.collect_items(scene, prompt="kitchen")
    by_id = {it.object_id: it for it in items}

    assert by_id["rot_list"].role == "main"
    assert by_id["rot_list"].yaw_deg == 45
    assert by_id["rot_short"].yaw_deg == 270
    assert by_id["rot_list"].placement_type == "wall"
    assert by_id["rot_short"].placement_type == "ceiling"
    assert by_id["ward"].zone_id == "storage_zone"
    assert by_id["sofa"].zone_id == "living_zone"
    assert by_id["dish"].zone_id == "kitchen_zone"
    assert by_id["kit"].zone_id == "kitchen_zone"
    assert by_id["sink"].zone_id == "bathroom_zone"
    assert by_id["decor"].zone_id == "decor_zone"
    assert by_id["alias"].semantic_group == "shelf"
    assert rg._placement_type_from_item({}, "unknown_group", "accessory") == "floor"

    list_position = {"bbox": aabb(0, 1, 0, 1, 0, 1), "position": [0]}
    rg._write_aabb_to_item(list_position, rg.AABB(1, 2, 3, 4, 5, 6))
    assert list_position["position"] == [1.5, 3.5, 5]
    dict_position = {"position": {"x": 0}}
    rg._write_aabb_to_item(dict_position, rg.AABB(1, 2, 3, 4, 5, 6))
    assert dict_position["position"] == {"x": 1.5, "y": 3.5, "z": 5}

    bounds_from_items = rg._room_bounds_from_data({"room": {"z_min": 1, "z_max": 4}}, items)
    assert bounds_from_items.x_max == pytest.approx(4.8)
    assert bounds_from_items.z_min == 1
    assert rg._room_bounds_from_data({"room": {}}, []).to_dict()["x_max"] == 5.0

    no_aabb = rg.ItemRef("no", {}, 0, "decor_zone", "chair", "secondary", None, 10, "+Y", "floor")
    assert rg._yaw_to_face(no_aabb, (1, 1)) == 10
    assert rg.generate_anchors_for_item(no_aabb) == {}
    assert rg._local_to_world_anchor(no_aabb, {}) == (0.0, 0.0, 0.0)
    assert rg._items_by_group(items, zone_id="kitchen_zone")["plate"][0].object_id == "dish"
    assert rg._find_items(items, ["chair"], role_preference="main")[0].object_id == "rot_list"
    assert rg._best_target_for(by_id["decor"], items, ["nonexistent"], allow_cross_zone=True) is None
    assert rg._best_target_for(by_id["decor"], items, ["desk"], allow_cross_zone=False) is None

    raw = {"rotation": [0]}
    rg._set_yaw_deg_on_item(raw, 123)
    assert raw["rotation"] == [0, 0.0, 123.0]
    raw = {"rotation_deg": 0}
    rg._set_yaw_deg_on_item(raw, 33)
    assert raw["rotation_deg"] == 33
    raw = {}
    rg._set_yaw_deg_on_item(raw, 44)
    assert raw["yaw_deg"] == 44

    rg._annotate_items_with_relationship_meta(scene, items, [rg._edge(by_id["decor"], "near", rg.WALL_TARGET_ID)])
    assert isinstance(by_id["decor"].raw["meta"], dict)

    disabled_args = argparse.Namespace(relationship_graph=False)
    missing_path = tmp_path / "missing.json"
    assert rg.maybe_apply_relationship_graph_stage(
        args=disabled_args,
        run_dir=tmp_path,
        scene_json_path=missing_path,
        prompt_text="",
    ) == (missing_path, None)


def test_validation_problem_branches_and_stage_skip_paths(tmp_path: Path):
    scene = {
        "room": {"floor_polygon": [[0, 0], [5, 0], [5, 5], [0, 5]]},
        "items": [
            {"id": "desk", "category": "desk", "aabb": aabb(0, 1, 0, 1, 0, 0.7)},
            {"id": "shelf", "semantic_group": "shelf", "category": "shelf", "aabb": aabb(4, 4.8, 0, 1, 0, 1.7)},
            {"id": "chair", "category": "chair", "aabb": aabb(4, 4.5, 4, 4.5, 0, 0.8), "yaw_deg": 0},
            {"id": "chair2", "category": "chair", "aabb": aabb(3.5, 4, 3.5, 4, 0, 0.8)},
            {"id": "laptop", "category": "laptop", "aabb": aabb(4, 4.2, 4, 4.2, 0, 0.1)},
            {"id": "book", "category": "book", "aabb": aabb(3.2, 3.4, 3.2, 3.4, 0, 0.1)},
            {"id": "sofa", "category": "sofa", "aabb": aabb(2, 3, 2, 3, 0, 1)},
            {"id": "decor", "category": "vase", "aabb": aabb(3, 3.2, 3, 3.2, 0, 0.3)},
            {"id": "no_box", "category": "desk"},
        ],
    }
    items = rg.collect_items(scene)
    by_id = {it.object_id: it for it in items}
    room = rg._room_bounds_from_data(scene, items)
    edges = [
        rg.RelationEdge("chair", "faces", "missing_target", "orientation", "hard"),
        rg.RelationEdge("laptop", "on_top_of", rg.ROOM_CENTER_ID, "support", "hard"),
        rg.RelationEdge("laptop", "on_top_of", "chair", "support", "hard"),
        rg.RelationEdge("laptop", "on_top_of", "desk", "support", "hard"),
        rg.RelationEdge("laptop", "on_top_of", "desk", "support", "hard"),
        rg.RelationEdge("decor", "on_top_of", "desk", "support", "soft"),
        rg.RelationEdge("desk", "on_top_of", "shelf", "support", "hard"),
        rg.RelationEdge("shelf", "on_top_of", "desk", "support", "hard"),
        rg.RelationEdge("chair", "faces", "no_box", "orientation", "soft"),
        rg.RelationEdge("chair", "faces", "desk", "orientation", "soft"),
        rg.RelationEdge("chair", "near", "desk", "proximity", "hard"),
        rg.RelationEdge("decor", "near", "desk", "proximity", "soft"),
    ]

    validation = rg.validate_relationship_graph(items, edges, room)
    problems = {row.get("problem") for row in validation["errors"]} | {row.get("problem") for row in validation["warnings"]}
    assert "to_object_missing" in problems
    assert "support_target_missing" in problems
    assert "invalid_support_target" in problems
    assert "support_relation_not_satisfied" in problems
    assert "support_relation_weak" in problems
    assert "multiple_hard_support_relations" in problems
    assert "orientation_target_has_no_aabb" in problems
    assert "faces_relation_weak" in problems
    assert "proximity_relation_not_satisfied" in problems
    assert "proximity_relation_weak" in problems
    assert "accessory_without_support_or_context_relation" in problems
    assert "chair_without_faces_relation" in problems
    assert "sofa_without_orientation_target" in problems
    assert "support_cycle" in problems
    assert rg._score_on_top_of(by_id["no_box"], by_id["desk"]) == 0

    args = argparse.Namespace(
        relationship_graph=True,
        relationship_graph_apply_placement=False,
        relationship_graph_no_validate=False,
        relationship_graph_no_repair=False,
        relationship_graph_add_missing_supports=False,
        relationship_graph_min_score=0.75,
    )
    missing_path = tmp_path / "missing.json"
    _, missing_info = rg.maybe_apply_relationship_graph_stage(
        args=args,
        run_dir=tmp_path,
        scene_json_path=missing_path,
        prompt_text="",
    )
    assert missing_info["skipped_reason"] == "scene_json_missing"

    list_path = tmp_path / "list.json"
    list_path.write_text("[]", encoding="utf-8")
    _, list_info = rg.maybe_apply_relationship_graph_stage(
        args=args,
        run_dir=tmp_path,
        scene_json_path=list_path,
        prompt_text="",
    )
    assert list_info["skipped_reason"] == "input_json_not_object"
    assert rg._load_llm_relations(None) is None

    with pytest.raises(RuntimeError, match="Input JSON root"):
        rg.main(["--input", str(list_path), "--out", str(tmp_path / "out.json")])


def test_relationship_remaining_rule_placement_and_score_edges():
    scene = {
        "room": {"floor_polygon": [[0, 0], [6, 0], [6, 6], [0, 6]], "ceiling_height_m": 3.0},
        "items": [
            {"id": "sofa", "category": "sofa", "aabb": aabb(1, 2, 1, 2, 0, 0.8)},
            {"id": "mirror", "category": "mirror", "aabb": aabb(0.1, 0.2, 2, 3, 1, 1.8), "placement_type": "wall"},
            {"id": "tooth", "category": "toothbrush", "aabb": aabb(3, 3.2, 3, 3.2, 0, 0.2)},
            {"id": "decor", "semantic_group": "small_decor", "aabb": aabb(4, 4.2, 4, 4.2, 0, 0.2)},
            {"id": "chair", "category": "chair", "aabb": aabb(2, 2.5, 2, 2.5, 0, 0.8), "yaw_deg": 0},
            {"id": "table", "category": "dining table", "aabb": aabb(3, 4, 1, 2, 0, 0.75)},
            {"id": "ch0", "category": "dining chair", "aabb": aabb(3, 3.4, 2.4, 2.8, 0, 0.8)},
            {"id": "ch1", "category": "dining chair", "aabb": aabb(3.4, 3.8, 2.4, 2.8, 0, 0.8)},
            {"id": "ch2", "category": "dining chair", "aabb": aabb(3.8, 4.2, 2.4, 2.8, 0, 0.8)},
            {"id": "ch3", "category": "dining chair", "aabb": aabb(4.2, 4.6, 2.4, 2.8, 0, 0.8)},
            {"id": "ch4", "category": "dining chair", "aabb": aabb(4.6, 5.0, 2.4, 2.8, 0, 0.8)},
        ],
    }
    items = rg.collect_items(scene)
    by_id = {item.object_id: item for item in items}
    room = rg._room_bounds_from_data(scene, items)
    edges = rg.build_rule_based_edges(items, room)
    keys = {edge.key() for edge in edges}
    center_items = rg.collect_items({"items": [{"id": "solo_sofa", "category": "sofa", "aabb": aabb(1, 2, 1, 2, 0, 0.8)}]})
    center_room = rg._room_bounds_from_data(scene, center_items)
    assert ("solo_sofa", "faces", rg.ROOM_CENTER_ID) in {edge.key() for edge in rg.build_rule_based_edges(center_items, center_room)}
    assert ("mirror", "mounted_on_wall", rg.WALL_TARGET_ID) in keys
    assert rg._support_targets_for_accessory(by_id["decor"]) == set(rg.TABLE_GROUPS | rg.STORAGE_GROUPS)
    assert rg._default_placement_area(by_id["decor"], by_id["table"]) == "center"
    no_aabb = rg.ItemRef("no", {}, 0, "z", "chair", "secondary", None, 0, "+Y", "floor")
    assert rg._infer_side_preference(no_aabb, by_id["table"]) == "auto"
    assert rg._sofa_orientation_target(by_id["sofa"], [by_id["sofa"]]) is None

    info = rg.apply_relation_aware_placement(
        scene,
        items,
        [
            rg.RelationEdge("missing", "faces", rg.ROOM_CENTER_ID, "orientation"),
            rg.RelationEdge("chair", "faces", rg.ROOM_CENTER_ID, "orientation", params={}),
            rg.RelationEdge("chair", "next_to", "table", "proximity", params={"side_preference": "diagonal"}),
        ],
        room,
        rg.StageOptions(apply_placement=True),
    )
    assert info["applied"] is True
    assert by_id["chair"].yaw_deg != 0

    changed: list[dict] = []
    new_item = rg.ItemRef("new", {}, 0, "z", "chair", "secondary", None, 0, "+Y", "floor")
    rg._update_item_aabb(new_item, rg.AABB(0, 1, 0, 1, 0, 1), changed, "unit")
    assert new_item.aabb is not None and changed == []
    rg._clamp_item_to_room(no_aabb, room, changed, "clamp")

    assert rg._aabb_for_proximity_relation(no_aabb, by_id["table"], rg.RelationEdge("no", "near", "table", "proximity")) is None
    assert rg._aabb_for_around_relation(no_aabb, by_id["table"], rg.RelationEdge("no", "around", "table", "group"), items) is None
    outsider = rg.ItemRef("outsider", {}, 99, "dining_zone", "vase", "accessory", rg.AABB(0, 0.2, 0, 0.2, 0, 0.2), 0, "+Y", "floor")
    assert rg._aabb_for_around_relation(outsider, by_id["table"], rg.RelationEdge("outsider", "around", "table", "group"), items) is not None
    assert rg._aabb_for_around_relation(by_id["ch4"], by_id["table"], rg.RelationEdge("ch4", "around", "table", "group"), items) is not None

    assert rg._score_faces(no_aabb, (1, 1), rg.RelationEdge("no", "faces", "table", "orientation")) == 0.0
    assert rg._score_proximity(no_aabb, by_id["table"], rg.RelationEdge("no", "near", "table", "proximity")) == 0.0
    far = rg.ItemRef("far", {}, 0, "z", "chair", "secondary", rg.AABB(100, 101, 100, 101, 0, 1), 0, "+Y", "floor")
    assert rg._score_proximity(far, by_id["table"], rg.RelationEdge("far", "near", "table", "proximity", params={"distance_m": {"min": 0.1, "max": 0.2}})) == 0.0
    assert rg._score_wall_relation(no_aabb, room) == 0.0


def test_llm_relation_resolution_edge_cases():
    items = rg.collect_items(relationship_scene(), prompt="office")
    edges = rg._resolve_llm_relations(
        [
            "bad",
            {"from_object_id": "chair_main", "to_object_id": "desk_main", "relation_type": "near", "constraint_level": "urgent"},
            {"from_object_id": "missing", "to_object_id": "desk_main", "relation_type": "near"},
            {"from_group": "chair", "to_group": "missing", "relation": "faces"},
            {"from_object_id": "chair_main", "to_object_id": rg.WALL_TARGET_ID, "relation": "against_wall", "priority": "decorative"},
        ],
        items,
    )

    by_key = {edge.key(): edge for edge in edges}
    assert by_key[("chair_main", "near", "desk_main")].constraint_level == "hard"
    assert by_key[("chair_main", "against_wall", rg.WALL_TARGET_ID)].constraint_level == "decorative"
