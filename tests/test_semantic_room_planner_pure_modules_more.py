from __future__ import annotations

import json
import urllib.error

import pytest

from src.pipeline.semantic_room_planner import anchors
from src.pipeline.semantic_room_planner import catalog_queries as cq
from src.pipeline.semantic_room_planner import geometry_analyzer as ga
from src.pipeline.semantic_room_planner import llm_client
from src.pipeline.semantic_room_planner import llm_steps
from src.pipeline.semantic_room_planner import normalizer
from src.pipeline.semantic_room_planner import placement_solver as ps
from src.pipeline.semantic_room_planner import relation_rules as rr
from src.pipeline.semantic_room_planner import repair
from src.pipeline.semantic_room_planner import schemas
from src.pipeline.semantic_room_planner import scene_export as se
from src.pipeline.semantic_room_planner import semantic_sanitizer as ss
from src.pipeline.semantic_room_planner import validation
from src.pipeline.semantic_room_planner import zone_templates as zt


def _obj(
    oid: str,
    subclass: str,
    zone_id: str = "zone_work",
    zone_type: str = "work_zone",
    role: str = "secondary",
    placement_type: str = "floor",
    dims: tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> dict:
    return {
        "id": oid,
        "subclass": subclass,
        "zone_id": zone_id,
        "zone_type": zone_type,
        "role": role,
        "placement_type": placement_type,
        "dimensions_m": {"width": dims[0], "depth": dims[1], "height": dims[2]},
        "label_en": subclass,
        "label_ru": subclass,
    }


def test_geometry_normalization_analysis_and_zone_template_minimums() -> None:
    raw = {
        "schema": "scene.v1",
        "room": {
            "id": "r",
            "type_hint": "bedroom",
            "floor_polygon": [{"x": 0, "y": 0}, {"x": 4, "y": 0}, {"x": 4, "y": 3}, {"x": 0, "y": 3}],
            "openings": {"doors": [{"center": {"x": 0.2, "y": 1.0}}]},
        },
    }
    normalized = ga.normalize_room_input(raw, "prompt")
    analyzed = ga.analyze_room_geometry(normalized)
    assert analyzed["area_m2"] == 12
    assert analyzed["has_known_door"] is True
    assert "Window position is unknown." in analyzed["assumptions"]
    assert len(analyzed["walls"]) == 4
    assert ga._extract_polygon({"scene": {"floor_polygon": [[0, 0], [1, 0], [0, 1]]}})[1] == {"x": 1.0, "y": 0.0}

    with pytest.raises(ValueError):
        ga.normalize_room_input({"room": {"floor_polygon": [[0, 0], [1, 0]]}})
    with pytest.raises(ValueError, match="missing"):
        ga._extract_polygon({})

    degenerate = ga.analyze_room_geometry({"floor_polygon": [[0, 0], [1, 0], [1, 0], [2, 0]]})
    assert degenerate["center"] == {"x": 1.0, "y": 0.0}
    assert any(w["length_m"] == 0.0 for w in degenerate["walls"])
    assert "Door position is unknown." in degenerate["assumptions"]
    clockwise = ga.analyze_room_geometry(
        {
            "floor_polygon": [[0, 0], [0, 1], [1, 1], [1, 0]],
            "room": {"openings": {"doors": [{"id": "d"}], "windows": [{"id": "w"}]}},
        }
    )
    assert clockwise["walls"][0]["normal_to_inside"] == {"x": 1.0, "y": -0.0}
    assert clockwise["assumptions"] == []

    work_items = zt.apply_zone_template_minimums({"id": "z", "type": "work_zone"}, [])
    assert {"desk", "office_chair"} <= {item["subclass"] for item in work_items}
    partial = zt.apply_zone_template_minimums({"id": "z", "type": "dining_zone"}, [{"subclass": "dining_table", "role": "main"}])
    assert sum(1 for item in partial if item["subclass"] == "dining_chair") == 4
    assert "desk" in zt.allowed_subclasses_for_zone("work_zone")
    assert "desk" in zt.structural_subclasses_for_zone("work_zone")
    sofa_anchors = anchors.generate_anchors([{"id": "sofa1", "subclass": "sofa"}])
    assert "front.coffee_table" in sofa_anchors["objects"]["sofa1"]["anchors"]


def test_semantic_small_branch_edges(tmp_path, monkeypatch) -> None:
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="root"):
        schemas.read_json(bad_json)
    assert schemas.as_float(object(), 4.5) == 4.5

    normalized = normalizer.normalize_objects(
        [{"zone_id": "decor", "source": "fallback", "objects": ["bad", {"subclass": "book", "role": "accessory"}]}],
        [{"id": "decor", "type": "decor_zone"}],
    )
    assert [obj["source"] for obj in normalized["objects"]] == ["fallback_template"]
    custom_query = cq.fallback_catalog_queries(
        {"id": "custom1", "subclass": "custom_chair", "label_en": "lounge chair", "label_ru": "кресло", "color": "green", "material": "velvet"}
    )
    assert custom_query["catalog_queries"]["ru"][0] == "кресло green velvet"
    assert custom_query["negative_keywords"] == ["children", "bar"]
    assert cq.generate_catalog_queries([{"id": "custom1", "subclass": "custom_chair"}], {"provider": "none"})["source"] == "fallback_templates"

    assert se._as_float(None, 7.0) == 7.0
    assert se._as_float(object(), 3.0) == 3.0
    assert se._extract_yaw_deg({"rotation": [12]}) == 12.0
    assert se._extract_yaw_deg({"rotation": {"unknown": 1}}) == 0.0
    assert se._proxy_fallback_group("unmapped_large_fixture") == "closed_cabinet"
    mesh_item = {"mesh_path": "local.glb"}
    assert se.ensure_procedural_proxy_asset(mesh_item, "chair", {}) is mesh_item
    asset_mesh_item = {"asset": {"kind": "supplier_model", "mesh_path": "local.glb"}}
    assert se.ensure_procedural_proxy_asset(asset_mesh_item, "chair", {}) is asset_mesh_item

    assert validation._opening_points(["bad", {"center": {"x": 1, "z": 2}}, {"from": {"x": 0, "y": 0}, "to": {"x": 2, "y": 2}}]) == [
        (1.0, 2.0),
        (1.0, 1.0),
    ]
    bad_validation = validation.validate_geometry(
        {"area_m2": 4.0, "bbox": {"x_min": 0, "x_max": 2, "y_min": 0, "y_max": 2, "width_m": 2, "depth_m": 2}},
        [
            _obj("floor_missing", "chair", role="secondary", placement_type="floor"),
            _obj("loose", "book", role="accessory", placement_type="floor"),
            _obj("support", "desk", role="secondary", placement_type="floor"),
            _obj("top", "book", role="accessory", placement_type="floor"),
            _obj("inside", "book", role="accessory", placement_type="floor"),
        ],
        {"edges": [rr._edge("top", "on_top_of", "support", "support"), rr._edge("inside", "inside", "support", "containment")]},
        {
            "placements": [
                {"object_id": "support", "aabb": {"x_min": 0.5, "x_max": 1.0, "y_min": 0.5, "y_max": 1.0, "z_min": 0, "z_max": 0.5}},
                {"object_id": "top", "aabb": {"x_min": 1.5, "x_max": 1.8, "y_min": 1.5, "y_max": 1.8, "z_min": 0.1, "z_max": 0.2}},
                {"object_id": "inside", "aabb": {"x_min": 1.5, "x_max": 1.8, "y_min": 1.5, "y_max": 1.8, "z_min": 0.0, "z_max": 0.1}},
                {"object_id": "loose", "aabb": {"x_min": 0.0, "x_max": 0.2, "y_min": 0.0, "y_max": 0.2, "z_min": 0.0, "z_max": 0.1}},
            ]
        },
    )
    assert any("missing placement" in error for error in bad_validation["hard_errors"])
    assert any("on_top_of z mismatch" in error for error in bad_validation["hard_errors"])
    assert any("inside relation placed near bottom" in warning for warning in bad_validation["soft_warnings"])

    repaired = repair.repair_scene({}, [], {}, {"placements": []}, {"is_valid": True}, max_iterations=2)
    assert repaired["final_status"] == "success"
    assert repaired["iterations"] == []

    def fake_loads(text):
        if str(text).startswith("{"):
            return []
        raise json.JSONDecodeError("no", str(text), 0)

    monkeypatch.setattr(llm_client.json, "loads", fake_loads)
    with pytest.raises(ValueError, match="not an object"):
        llm_client.extract_json_object('prefix {"a": 1} suffix')


def test_semantic_sanitizer_theme_classification_caps_and_repair() -> None:
    plant_theme = ss.extract_theme_spec("many plants biophilic bedroom")
    avoid_theme = ss.extract_theme_spec("без растений и without plants")
    assert plant_theme["theme_tags"] == ["plants", "biophilic"]
    assert "potted_plant" in avoid_theme["avoid_objects"]
    assert ss._plant_like_text("ceramic pot with leaves")
    assert ss._classify_from_text({"name": "small hanging plant"}) == "hanging_planter"
    assert ss._classify_from_text({"label_en": "desk lamp"}) == "desk"
    assert ss._normalize_item_subclass({"subclass": "plant"}) == "potted_plant"
    assert ss._normalize_item_subclass({"subclass": "wall_shelf"}) == "shelf"

    raw_items = {
        "source": "llm",
        "objects": [
            "bad",
            {"subclass": "object"},
            {"name": "mini desktop plant", "x": 10},
            {"label_en": "wardrobe"},
            {"subclass": "pillow", "quantity": 99},
        ],
    }
    cleaned = ss.sanitize_zone_items("zone_sleeping", "sleeping_zone", raw_items, plant_theme)
    subclasses = [item["subclass"] for item in cleaned["objects"]]
    assert "small_potted_plant" in subclasses
    assert "wardrobe" not in subclasses
    assert any("non-object" in warning for warning in cleaned["warnings"])
    assert all("x" not in item for item in cleaned["objects"])

    many = [_obj(f"p{i}", "potted_plant", role="accessory") for i in range(8)]
    many += [_obj(f"s{i}", "small_potted_plant", role="accessory") for i in range(8)]
    repaired = ss.repair_semantic_objects(
        many + [_obj("bad", "unknown"), dict(_obj("dup", "book"), id="p0")],
        [{"id": "z", "type": "sleeping_zone"}],
        plant_theme,
        max_total_objects=6,
    )
    assert len(repaired["objects"]) <= 6
    assert any("Dropped" in warning for warning in repaired["warnings"])

    no_plants = ss.repair_semantic_objects([_obj("p", "potted_plant"), _obj("b", "book")], [], avoid_theme)
    assert [item["id"] for item in no_plants["objects"]] == ["b"]


def test_llm_steps_fallbacks_prompts_and_retry(monkeypatch) -> None:
    state = {
        "prompt": "детская для мальчик racing single bed student desk много растений",
        "input": {"room": {"type_hint": ""}},
        "room_intent": {"room_type": "bedroom", "required_functions": ["sleeping", "working", "storage"]},
    }
    assert llm_steps._settings({"provider": " Ollama ", "use_llm_catalog_queries": True}, "s")["provider"] == "ollama"
    assert llm_steps._provider_is_none({"provider": "none"})
    assert llm_steps._extract_theme("racing car room") == "cars/racing"
    assert llm_steps._prompt_preferences("single bed no plants")["bed_preference"] == "single"
    assert llm_steps.build_room_intent_prompt(state)[0]["role"] == "system"
    assert "Allowed_subclasses" in llm_steps.build_zone_items_prompt(state, {"id": "z", "type": "sleeping_zone"})[1]["content"]
    assert "Relation enum" in llm_steps.build_zone_relations_prompt(state, {"id": "z"}, {"objects": []})[0]["content"]
    assert llm_steps.build_catalog_queries_prompt({"id": "o"})[1]["content"]

    intent = llm_steps.run_room_intent_step(state, {"provider": "none"})
    zones = llm_steps.run_zones_step(state | {"room_intent": intent}, {"provider": "none"})
    assert intent["room_type"] == "bedroom"
    assert {zone["type"] for zone in zones["zones"]} >= {"sleeping_zone", "work_zone", "storage_zone"}
    sleeping = {"id": "zone_sleeping", "type": "sleeping_zone"}
    fallback_items = llm_steps.run_zone_items_step(state, sleeping, {"provider": "none"})
    assert any(item["subclass"] == "bed" for item in fallback_items["objects"])
    assert any("racing_rug" in warning or "excessive" in warning for warning in fallback_items["warnings"])
    assert llm_steps.run_zone_relations_step(state, sleeping, fallback_items, {"provider": "none"})["relations"]

    calls = []

    def fake_call_json_llm(messages, **kwargs):
        calls.append(kwargs["step_name"])
        if kwargs["step_name"].endswith("_retry"):
            return {"objects": [{"subclass": "book"}]}
        return {"objects": [{"subclass": "object"}]}

    monkeypatch.setattr(llm_steps, "call_json_llm", fake_call_json_llm)
    retried = llm_steps.run_zone_items_step({"prompt": ""}, sleeping, {"provider": "ollama"})
    assert calls == ["04_zone_items_zone_sleeping", "04_zone_items_zone_sleeping_retry"]
    assert retried["objects"][0]["subclass"] == "book"


def test_llm_steps_room_type_zone_and_theme_fallback_matrix() -> None:
    assert "Zone type enum" in llm_steps.build_zones_prompt({"prompt": "x"})[0]["content"]

    prompts = {
        "kitchen": ("modern kitchen cooking", "kitchen", "cooking"),
        "living": ("cozy living room", "living_room", "living"),
        "bathroom": ("small bathroom", "bathroom", "bathing"),
        "toilet": ("compact toilet wc", "toilet", "toilet"),
        "dining": ("formal dining room", "dining_room", "dining"),
        "office": ("quiet office кабинет", "office", None),
    }
    for label, (prompt, room_type, required_function) in prompts.items():
        intent = llm_steps.run_room_intent_step({"prompt": prompt, "input": {"room": {"type_hint": "room"}}}, {"provider": "none"})
        assert intent["room_type"] == room_type, label
        if required_function:
            assert required_function in intent["required_functions"]

    whole_home = llm_steps.run_zones_step(
        {
            "prompt": "entire home все комнаты",
            "room_intent": {"room_type": "apartment", "required_functions": []},
        },
        {"provider": "none"},
    )
    assert {zone["type"] for zone in whole_home["zones"]} >= {"living_zone", "kitchen_zone", "bathroom_zone", "toilet_zone"}

    zone_cases = [
        ({"room_type": "kitchen", "required_functions": ["cooking", "dining"]}, {"kitchen_zone", "dining_zone"}),
        ({"room_type": "living_room", "required_functions": ["living"]}, {"living_zone"}),
        ({"room_type": "bathroom", "required_functions": ["bathing", "toilet"]}, {"bathroom_zone", "toilet_zone"}),
        ({"room_type": "toilet", "required_functions": ["toilet"]}, {"toilet_zone"}),
        ({"room_type": "dining_room", "required_functions": ["dining"]}, {"dining_zone"}),
        ({"room_type": "office", "required_functions": ["storage"]}, {"work_zone"}),
        ({"room_type": "unknown", "required_functions": []}, {"living_zone"}),
    ]
    for intent, expected_types in zone_cases:
        zones = llm_steps.run_zones_step({"prompt": "", "room_intent": intent}, {"provider": "none"})
        assert {zone["type"] for zone in zones["zones"]} >= expected_types

    plant_state = {
        "prompt": "biophilic bedroom many plants",
        "theme_spec": {"theme_tags": ["plants"], "avoid_objects": []},
    }
    for zone_type, expected in [
        ("sleeping_zone", set()),
        ("work_zone", set()),
        ("storage_zone", {"small_potted_plant", "potted_plant"}),
    ]:
        items = llm_steps.run_zone_items_step(plant_state, {"id": f"z_{zone_type}", "type": zone_type}, {"provider": "none"})
        subclasses = {item["subclass"] for item in items["objects"]}
        assert expected <= subclasses
        assert items["schema"] == "zone_items/v1"

    fallback = llm_steps.run_zone_items_step({"prompt": ""}, {"id": "z_unknown", "type": "unknown_zone"}, {"provider": "none"})
    assert fallback["objects"] == []

    calls = []

    def raises_retry(*_args, **kwargs):
        calls.append(kwargs.get("step_name"))
        if len(calls) == 1:
            return {"objects": [{"subclass": "object"}]}
        raise RuntimeError("llm offline")

    original = llm_steps.call_json_llm
    llm_steps.call_json_llm = raises_retry
    try:
        failed = llm_steps.run_zone_items_step({"prompt": ""}, {"id": "z", "type": "sleeping_zone"}, {"provider": "ollama"})
    finally:
        llm_steps.call_json_llm = original
    assert any("semantic retry failed" in warning for warning in failed.get("warnings", []))


def test_relation_rules_resolve_augment_validate_and_graph() -> None:
    objects = [
        _obj("desk", "desk", role="main", dims=(1.4, 0.7, 0.75)),
        _obj("chair", "office_chair"),
        _obj("laptop", "laptop", role="accessory", placement_type="support", dims=(0.3, 0.2, 0.03)),
        _obj("mug", "mug", role="accessory", placement_type="support", dims=(0.08, 0.08, 0.1)),
        _obj("bed", "bed", "zone_sleep", "sleeping_zone", "main", dims=(1.8, 2.0, 0.55)),
        _obj("night", "nightstand", "zone_sleep", "sleeping_zone", dims=(0.5, 0.4, 0.5)),
        _obj("pillow", "pillow", "zone_sleep", "sleeping_zone", "accessory", "support", dims=(0.5, 0.35, 0.1)),
        _obj("rug", "rug", "zone_sleep", "sleeping_zone", "accessory", dims=(2.0, 1.5, 0.02)),
        _obj("sofa", "sofa", "zone_living", "living_zone", "main", dims=(2.0, 0.9, 0.8)),
        _obj("coffee", "coffee_table", "zone_living", "living_zone", dims=(0.9, 0.55, 0.4)),
        _obj("tv", "tv", "zone_living", "living_zone", dims=(1.0, 0.1, 0.6)),
        _obj("remote", "remote", "zone_living", "living_zone", "accessory", "support", dims=(0.12, 0.05, 0.02)),
        _obj("counter", "kitchen_counter", "zone_kitchen", "kitchen_zone", "main", dims=(2.0, 0.65, 0.9)),
        _obj("stove", "stove", "zone_kitchen", "kitchen_zone", "accessory", "support", dims=(0.55, 0.45, 0.08)),
        _obj("sink", "sink", "zone_bath", "bathroom_zone", "main", dims=(0.6, 0.45, 0.85)),
        _obj("mirror", "mirror", "zone_bath", "bathroom_zone", "accessory", "wall", dims=(0.5, 0.04, 0.7)),
    ]
    by_id = rr._object_by_id(objects)
    assert rr.normalize_relation_class("on_top_of") == "support"
    assert rr._target_subclass(rr._edge("desk", "against_wall", "room_wall", None), by_id) == "room_wall"
    assert rr.is_relation_allowed(rr._edge("laptop", "on_top_of", "desk", None), by_id) == (True, "")
    assert rr.is_relation_allowed(rr._edge("desk", "on_top_of", "laptop", None), by_id)[0] is False

    resolved = rr.resolve_relations_by_subclass(
        [{"zone_id": "zone_work", "source": "template", "relations": [{"from_subclass": "office_chair", "relation_type": "faces", "to_subclass": "desk"}]}],
        objects,
    )
    assert resolved[0]["from_id"] == "chair"
    augmented = rr.augment_relations_with_rules(objects, resolved)
    rels = {(edge["from_id"], edge["relation_type"], edge["to_id"]) for edge in augmented}
    assert ("chair", "in_front_of", "desk") in rels
    assert ("laptop", "on_top_of", "desk") in rels
    assert ("pillow", "on_top_of", "bed") in rels
    assert ("remote", "on_top_of", "coffee") in rels
    assert ("mirror", "mounted_on_wall", "room_wall") in rels

    invalid = rr.validate_relation_targets_exist(
        augmented + [rr._edge("missing", "bad_rel", "missing2", "semantic")],
        objects + [dict(objects[0], id="desk")],
        [{"id": "zone_sleep", "type": "sleeping_zone"}, {"id": "zone_empty", "type": "work_zone"}],
    )
    assert invalid["is_valid"] is False
    assert any("duplicate object ids" == err for err in invalid["errors"])
    graph = rr.build_relationship_graph(objects, augmented)
    assert graph["nodes"][0]["id"] == "desk"


def test_relation_rules_remaining_auto_rules_and_validation_edges() -> None:
    assert rr.is_relation_allowed(rr._edge("missing", "on_top_of", "desk", "support"), {}) == (False, "missing from_id")
    assert rr.is_relation_allowed(rr._edge("empty", "near", "desk", "proximity"), {"empty": {"id": "empty"}})[1] == "empty source subclass"
    assert rr.is_relation_allowed(rr._edge("book", "inside", "missing", "containment"), {"book": _obj("book", "book")})[1] == "missing to_id"

    tiny_desk = _obj("tiny_desk", "desk", dims=(0.1, 0.1, 0.7))
    big_laptop = _obj("big_laptop", "laptop", placement_type="support", dims=(1.0, 1.0, 0.05))
    objects_by_id = {"tiny_desk": tiny_desk, "big_laptop": big_laptop}
    assert "larger than support" in rr.is_relation_allowed(rr._edge("big_laptop", "on_top_of", "tiny_desk", "support"), objects_by_id)[1]
    assert "containment not allowed" in rr.is_relation_allowed(rr._edge("big_laptop", "inside", "tiny_desk", "containment"), objects_by_id)[1]
    big_book = _obj("big_book", "book", placement_type="support", dims=(1.0, 1.0, 0.1))
    small_shelf = _obj("small_shelf", "shelf", dims=(0.2, 0.2, 1.0))
    assert "larger than container" in rr.is_relation_allowed(
        rr._edge("big_book", "inside", "small_shelf", "containment"),
        {"big_book": big_book, "small_shelf": small_shelf},
    )[1]
    assert "floor-only object" in rr.is_relation_allowed(
        rr._edge("potted", "on_top_of", "floor", "support"),
        {"potted": _obj("potted", "potted_plant")},
    )[1]

    assert rr.augment_relations_with_rules([_obj("lonely_chair", "chair", "zd", "dining_zone")], []) == []
    objects = [
        _obj("k_counter", "kitchen_counter", "zk", "kitchen_zone", "main", dims=(1.8, 0.6, 0.9)),
        _obj("k_table", "kitchen_table", "zk", "kitchen_zone", "main", dims=(1.0, 0.8, 0.75)),
        _obj("k_chair", "chair", "zk", "kitchen_zone"),
        _obj("pan", "pan", "zk", "kitchen_zone", "accessory", "support", dims=(0.25, 0.2, 0.08)),
        _obj("d_table", "dining_table", "zd", "dining_zone", "main", dims=(1.4, 0.9, 0.75)),
        _obj("d_chair", "dining_chair", "zd", "dining_zone"),
        _obj("plate", "plate", "zd", "dining_zone", "accessory", "support", dims=(0.22, 0.22, 0.03)),
        _obj("vase", "vase", "zd", "dining_zone", "accessory", "support", dims=(0.18, 0.18, 0.35)),
        _obj("sofa2", "sofa", "zl", "living_zone", "main", dims=(1.8, 0.8, 0.8)),
        _obj("coffee2", "coffee_table", "zl", "living_zone", dims=(0.8, 0.5, 0.35)),
        _obj("lpillow", "pillow", "zl", "living_zone", "accessory", "support", dims=(0.35, 0.25, 0.1)),
        _obj("lblanket", "blanket", "zl", "living_zone", "accessory", "support", dims=(0.7, 0.5, 0.08)),
        _obj("bed2", "bed", "zs", "sleeping_zone", "main", dims=(1.6, 2.0, 0.55)),
        _obj("night2", "nightstand", "zs", "sleeping_zone", dims=(0.45, 0.4, 0.5)),
        _obj("lamp2", "table_lamp", "zs", "sleeping_zone", "accessory", "support", dims=(0.2, 0.2, 0.45)),
        _obj("phone2", "phone", "zs", "sleeping_zone", "accessory", "support", dims=(0.08, 0.04, 0.02)),
        _obj("art2", "wall_art", "zs", "sleeping_zone", "accessory", "wall", dims=(0.5, 0.03, 0.4)),
        _obj("stand", "plant_stand", "zp", "decor_zone", dims=(0.3, 0.3, 0.6)),
        _obj("potted", "potted_plant", "zp", "decor_zone", "accessory", dims=(0.25, 0.25, 0.6)),
        _obj("shelf2", "shelf", "zp", "decor_zone", dims=(0.9, 0.3, 1.4)),
        _obj("smallplant", "small_potted_plant", "zp", "decor_zone", "accessory", "support", dims=(0.18, 0.18, 0.25)),
        _obj("hanging", "hanging_planter", "zp", "decor_zone", "accessory", "wall", dims=(0.25, 0.25, 0.45)),
        _obj("bath_sink", "sink", "zb", "bathroom_zone", "main", dims=(0.55, 0.45, 0.8)),
        _obj("rack", "towel_rack", "zb", "bathroom_zone", "accessory", "wall", dims=(0.5, 0.04, 0.12)),
        _obj("hand_towel", "hand_towel", "zb", "bathroom_zone", "accessory", "support", dims=(0.35, 0.02, 0.25)),
        _obj("shower", "shower", "zb", "bathroom_zone", "main", dims=(0.8, 0.8, 2.0)),
        _obj("shampoo", "shampoo_bottle", "zb", "bathroom_zone", "accessory", "support", dims=(0.08, 0.08, 0.18)),
        _obj("bathmat", "bath_mat", "zb", "bathroom_zone", "accessory", dims=(0.6, 0.4, 0.02)),
        _obj("toilet", "toilet", "zt", "toilet_zone", "main", dims=(0.45, 0.65, 0.75)),
        _obj("brush", "toilet_brush", "zt", "toilet_zone", "accessory", dims=(0.12, 0.12, 0.45)),
        _obj("wardrobe2", "wardrobe", "zstore", "storage_zone", "main", dims=(1.0, 0.6, 1.8)),
        _obj("store_shelf", "shelf", "zstore", "storage_zone", "main", dims=(0.8, 0.3, 1.4)),
        _obj("storage_box", "storage_box", "zstore", "storage_zone", "accessory", "support", dims=(0.3, 0.25, 0.25)),
        _obj("store_book", "book", "zstore", "storage_zone", "accessory", "support", dims=(0.12, 0.08, 0.03)),
        _obj("toycar", "toy_car", "zstore", "storage_zone", "accessory", "support", dims=(0.12, 0.06, 0.05)),
        _obj("loose_desk", "desk", "zloose", "loose_zone", "main", dims=(1.0, 0.5, 0.75)),
        _obj("loose_book", "book", "zloose", "loose_zone", "accessory", "support", dims=(0.1, 0.08, 0.03)),
    ]
    existing = [rr._edge("loose_desk", "on_top_of", "loose_book", "support", level="hard", source="bad_input")]
    augmented = rr.augment_relations_with_rules(objects, existing)
    triples = {(edge["from_id"], edge["relation_type"], edge["to_id"]) for edge in augmented}
    assert ("k_chair", "around", "k_table") in triples
    assert ("plate", "on_top_of", "d_table") in triples
    assert ("lpillow", "on_top_of", "sofa2") not in triples
    assert ("lblanket", "near", "sofa2") in triples
    assert ("lamp2", "on_top_of", "night2") in triples
    assert ("potted", "on_top_of", "stand") in triples
    assert ("smallplant", "inside", "shelf2") in triples
    assert ("hanging", "mounted_on_wall", "room_wall") in triples
    assert ("hand_towel", "near", "rack") in triples
    assert ("shampoo", "inside", "shower") in triples
    assert ("brush", "near", "toilet") in triples
    assert ("storage_box", "inside", "wardrobe2") in triples
    assert ("store_book", "inside", "store_shelf") in triples
    assert ("toycar", "inside", "store_shelf") in triples
    assert ("loose_book", "on_top_of", "loose_desk") in triples
    assert any(edge.get("source") == "semantic_repair" for edge in augmented)

    bad_wall = dict(_obj("bad_wall", "wall_art", placement_type="floor"), label_en="", label_ru="")
    empty = dict(_obj("empty", ""), label_en="", label_ru="")
    validation_result = rr.validate_relation_targets_exist(
        [rr._edge("big_book", "inside", "small_shelf", "containment")],
        [big_book, small_shelf, bad_wall, empty],
        [{"id": "zmissing", "type": "work_zone"}],
    )
    assert validation_result["is_valid"] is False
    assert any("wall-only object has floor placement" in str(error) for error in validation_result["errors"])
    assert any("empty subclass" in str(error) for error in validation_result["errors"])
    assert any("larger than container" in str(error) for error in validation_result["errors"])


def test_placement_solver_branches_and_geometry_validation() -> None:
    room = {
        "area_m2": 16.0,
        "bbox": {"x_min": 0.0, "x_max": 4.0, "y_min": 0.0, "y_max": 4.0, "width_m": 4.0, "depth_m": 4.0},
        "center": {"x": 2.0, "y": 2.0},
        "openings": {"doors": [{"x": 0.4, "y": 0.4}], "windows": [{"from": {"x": 3, "y": 2}, "to": {"x": 4, "y": 2}}]},
    }
    objects = [
        _obj("bed", "bed", "z", "sleeping_zone", "main", dims=(1.2, 1.4, 0.5)),
        _obj("night", "nightstand", "z", "sleeping_zone", dims=(0.4, 0.35, 0.45)),
        _obj("pillow", "pillow", "z", "sleeping_zone", "accessory", "support", dims=(0.3, 0.2, 0.08)),
        _obj("wall_art", "wall_art", "z", "sleeping_zone", "accessory", "wall", dims=(0.5, 0.04, 0.5)),
        _obj("rug", "rug", "z", "sleeping_zone", "accessory", dims=(1.6, 1.1, 0.02)),
        _obj("shelf", "shelf", "z", "sleeping_zone", dims=(0.7, 0.35, 1.6)),
        _obj("book", "book", "z", "sleeping_zone", "accessory", "support", dims=(0.12, 0.08, 0.03)),
    ]
    edges = [
        rr._edge("night", "next_to", "bed", "proximity"),
        rr._edge("pillow", "on_top_of", "bed", "support", params={"placement_area": "top.left_front"}),
        rr._edge("wall_art", "above", "bed", "wall"),
        rr._edge("rug", "under", "bed", "proximity"),
        rr._edge("shelf", "against_wall", "room_wall", "wall"),
        rr._edge("book", "inside", "shelf", "containment"),
        rr._edge("night", "faces", "bed", "orientation"),
    ]
    assert ps._find_obj(objects, "bed")["subclass"] == "bed"
    assert ps._slot_offset("bad_slot", {"x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1}, {"width": 0.1, "depth": 0.1}) == (0.0, 0.0)
    assert ps._clamp_center_to_bbox(-10, 10, {"width": 1, "depth": 1}, room["bbox"])[2] is True
    solved = ps.solve_placements(room, objects, {"edges": edges}, anchors.generate_anchors(objects), {"max_candidates_per_object": 6, "max_total_candidate_combinations": 16})
    placements = {item["object_id"]: item for item in solved["placements"]}
    assert placements["pillow"]["position"]["z"] > placements["bed"]["position"]["z"]
    assert placements["book"]["position"]["z"] >= placements["shelf"]["aabb"]["z_min"]
    assert solved["solver_limits"]["candidate_evaluations"] > 0

    bad_validation = validation.validate_geometry(
        room,
        objects + [_obj("huge", "wardrobe", "z", "sleeping_zone", "main", dims=(9.0, 9.0, 2.0))],
        {"edges": edges},
        {"placements": list(placements.values()) + [{"object_id": "huge", "position": {"x": 0.4, "y": 0.4, "z": 0}, "aabb": {"x_min": -4, "x_max": 5, "y_min": -4, "y_max": 5, "z_min": 0, "z_max": 2}, "warnings": ["manual warning"]}]},
    )
    assert bad_validation["is_valid"] is False
    assert any("object dimensions" in err for err in bad_validation["hard_errors"])
    assert "manual warning" in bad_validation["soft_warnings"]

    repaired = repair.repair_scene(room, objects, {"edges": edges}, solved, bad_validation, max_iterations=2)
    assert repaired["schema"] == "repair_report/v1"
    assert repaired["max_iterations"] == 2


def test_placement_solver_remaining_geometric_branches() -> None:
    room = {
        "bbox": {"x_min": 0.0, "x_max": 3.0, "y_min": 0.0, "y_max": 3.0, "width_m": 3.0, "depth_m": 3.0},
        "center": {"x": 1.5, "y": 1.5},
    }
    dims = {"width": 0.4, "depth": 0.4, "height": 0.4}
    bbox = room["bbox"]
    assert ps._is_free_floor_aabb(ps._aabb(-1, -1, 0, 0.2, 0.2, 0.2), bbox, {}, "x") is False
    assert ps._find_free_floor_position(room, dims, {}, "x", preferred=[(1.0, 1.0)]) == (1.0, 1.0)
    assert ps._find_free_floor_position(room, {"width": 10, "depth": 10, "height": 1}, {}, "x") is None
    assert ps.generate_main_object_candidates(room, _obj("table", "dining_table"), [], {}, 1)[0]["reason"] == "near room center"

    target_template = {
        "aabb": {"x_min": 1.2, "x_max": 1.8, "y_min": 1.2, "y_max": 1.8, "z_min": 0, "z_max": 1.0},
        "position": {"x": 1.5, "y": 1.5, "z": 0},
    }
    for point, yaw in [({"x": 2.95, "y": 1.5, "z": 0}, -90.0), ({"x": 1.5, "y": 0.02, "z": 0}, 180.0), ({"x": 1.5, "y": 2.95, "z": 0}, 0.0)]:
        target = dict(target_template, position=point)
        assert ps._wall_position_near_target(room, dims, target)[2] == yaw

    small_support = {
        "aabb": {"x_min": 1.0, "x_max": 2.0, "y_min": 1.0, "y_max": 2.0, "z_min": 0.0, "z_max": 0.8},
        "position": {"x": 1.5, "y": 1.5, "z": 0},
        "subclass": "drawer",
    }
    inside = ps._inside_position(_obj("sock", "book", dims=(0.1, 0.1, 0.1)), small_support, rr._edge("sock", "inside", "drawer", "containment"), {})
    assert inside[2] > 0
    occupied = {"support": set(ps.GENERIC_SLOT_ORDER)}
    assert ps._choose_support_slot(_obj("extra", "unknown"), rr._edge("extra", "on_top_of", "support", "support"), occupied) == "center"

    objects = [
        _obj("sofa", "sofa", "z", "living_zone", "main", dims=(1.2, 0.8, 0.7)),
        _obj("nearplant", "plant", "z", "living_zone", dims=(0.3, 0.3, 0.7)),
        _obj("coffee", "coffee_table", "z", "living_zone", dims=(0.5, 0.35, 0.35)),
        _obj("around1", "chair", "z", "living_zone", dims=(0.35, 0.35, 0.7)),
        _obj("around2", "chair", "z", "living_zone", dims=(0.35, 0.35, 0.7)),
        _obj("mirror", "mirror", "z", "living_zone", "accessory", "wall", dims=(0.4, 0.05, 0.6)),
        _obj("poster", "poster", "z", "living_zone", "accessory", "wall", dims=(0.3, 0.04, 0.3)),
        _obj("toy", "toy", "z", "living_zone", "accessory", "floor", dims=(0.2, 0.2, 0.2)),
        _obj("box", "box", "z", "living_zone", "accessory", "floor", dims=(0.25, 0.25, 0.25)),
    ]
    edges = [
        rr._edge("nearplant", "near", "sofa", "proximity"),
        rr._edge("coffee", "in_front_of", "sofa", "proximity"),
        rr._edge("around1", "around", "sofa", "proximity"),
        rr._edge("around2", "around", "sofa", "proximity"),
        rr._edge("mirror", "mounted_on_wall", "sofa", "wall"),
        rr._edge("box", "under", "sofa", "support"),
    ]
    solved = ps.solve_placements(room, objects, {"edges": edges}, {}, {"max_candidates_per_object": 4, "max_total_candidate_combinations": 64})
    placements = {item["object_id"]: item for item in solved["placements"]}
    assert "near sofa" in placements["nearplant"]["placement_reason"]
    assert "in front of sofa" in placements["coffee"]["placement_reason"]
    assert "distributed around sofa" in placements["around1"]["placement_reason"]
    assert placements["mirror"]["placement_reason"].startswith("against")
    assert placements["poster"]["position"]["z"] == 1.2
    assert placements["toy"]["placement_reason"] == "Placed by fallback grid."
    assert "Shifted to nearest free floor slot" in placements["box"]["placement_reason"]


def test_normalizer_catalog_queries_scene_export_and_llm_client(tmp_path, monkeypatch) -> None:
    zones = [{"id": "zone_work", "type": "work_zone"}]
    normalized = normalizer.normalize_objects(
        [
            {
                "zone_id": "zone_work",
                "source": "llm",
                "objects": [
                    {"subclass": "desk", "role": "main"},
                    {"type": "mouse", "quantity": 2, "role": "accessory", "dimensions_m": {"width": 10, "depth": 10, "height": 10}},
                ],
            }
        ],
        zones,
    )
    objects = normalized["objects"]
    assert any(item["subclass"] == "desk" for item in objects)
    assert sum(1 for item in objects if item["subclass"] == "mouse") == 2
    assert all(item["dimensions_m"]["width"] <= 3.0 for item in objects)

    fallback_query = cq.fallback_catalog_queries({"id": "chair1", "subclass": "office_chair", "color": "black", "material": "fabric"})
    assert fallback_query["negative_keywords"] == ["children", "bar"]
    monkeypatch.setattr(cq, "call_json_llm", lambda *a, **k: {"items": [{"object_id": objects[0]["id"], "catalog_queries": {"ru": ["x"]}}]})
    llm_queries = cq.generate_catalog_queries(objects, {"provider": "ollama", "use_llm_catalog_queries": True, "llm_catalog_max_objects": 1})
    assert llm_queries["source"] == "llm_batch_with_fallback"
    monkeypatch.setattr(cq, "call_json_llm", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
    assert cq.generate_catalog_queries(objects, {"provider": "ollama", "use_llm_catalog_queries": True})["source"] == "fallback_after_llm_catalog_error"

    item = {"id": "x", "semantic_group": "mug", "rotation": {"z": 45}, "meta": "bad"}
    se.normalize_rotation_for_legacy_blender(item)
    assert item["rotation"] == [0.0, 0.0, 45.0]
    se.ensure_procedural_proxy_asset(item, "mug", {"width": "0,2", "depth": 0.1, "height": 0.08})
    assert item["asset"]["fallback_subclass"] == "decor_box"
    assert se.ensure_procedural_proxy_asset({"mesh_path": "local.obj"}, "desk", {})["mesh_path"] == "local.obj"

    state = {
        "input": {"prompt": "p", "room": {"id": "r", "floor_polygon": []}},
        "room_geometry": {"bbox": {}, "center": {}},
        "room_intent": {},
        "zones": zones,
        "objects": [objects[0]],
        "relationship_graph": {"edges": []},
        "anchors": anchors.generate_anchors([objects[0]]),
        "placements": {"placements": [{"object_id": objects[0]["id"], "position": {"x": 1, "y": 1, "z": 0}, "rotation_z_deg": 90, "aabb": {"x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1, "z_min": 0, "z_max": 1}}]},
        "geometry_validation": {"is_valid": True, "soft_warnings": ["soft"]},
        "relationship_validation": {},
        "catalog_queries": {"items": [{"object_id": objects[0]["id"], "q": 1}]},
    }
    exported = se.export_scene_plan(state, tmp_path)
    assert exported["scene_v1"]["schema"] == "scene.v1"
    assert (tmp_path / "scene.semantic.v1.json").is_file()

    assert llm_client.extract_json_object("```json\n{\"a\": 1}\n```") == {"a": 1}
    assert llm_client.extract_json_object("prefix {\"a\": {\"b\": \"}\"}} suffix")["a"]["b"] == "}"
    assert llm_client.extract_json_object("prefix {\"a\":\"x\\\\ny\"} suffix") == {"a": "x\\ny"}
    with pytest.raises(ValueError):
        llm_client.extract_json_object("no json")
    with pytest.raises(ValueError):
        llm_client.extract_json_object("prefix {\"a\": 1")

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return json.dumps({"message": {"content": "{\"ok\": true}"}}).encode()

    monkeypatch.setattr(llm_client.urllib.request, "urlopen", lambda req, timeout: FakeResponse())
    monkeypatch.setattr(llm_client.time, "sleep", lambda *_: None)
    assert llm_client.call_json_llm([{"role": "user", "content": "x"}], provider="ollama", max_attempts=1) == {"ok": True}
    assert llm_client.call_json_llm([{"role": "user", "content": "x"}], provider="ollama", max_attempts=1, debug_dir=tmp_path / "debug_success", step_name="ok") == {"ok": True}
    assert (tmp_path / "debug_success" / "ok.01.response.txt").is_file()
    with pytest.raises(RuntimeError):
        llm_client.call_json_llm([], provider="none")
    with pytest.raises(RuntimeError):
        llm_client.call_json_llm([], provider="unsupported", max_attempts=1)
    monkeypatch.setattr(llm_client.urllib.request, "urlopen", lambda req, timeout: (_ for _ in ()).throw(urllib.error.URLError("down")))
    with pytest.raises(RuntimeError):
        llm_client.call_json_llm([{"role": "user", "content": "x"}], provider="ollama", max_attempts=2, debug_dir=tmp_path / "debug")
    with pytest.raises(RuntimeError):
        llm_client.call_json_llm([{"role": "user", "content": "x"}], provider="openrouter", max_attempts=1)
    class FakeOpenRouterResponse(FakeResponse):
        def read(self):
            return json.dumps({"choices": [{"message": {"content": "{\"r\": 2}"}}]}).encode()

    monkeypatch.setenv("OPENROUTER_API_KEY", "key")
    monkeypatch.setattr(llm_client.urllib.request, "urlopen", lambda req, timeout: FakeOpenRouterResponse())
    assert llm_client.call_json_llm([], provider="openrouter", model="m", max_attempts=1) == {"r": 2}
