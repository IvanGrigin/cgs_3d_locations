from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = str(ROOT / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src.pipeline import kitchen_stage as ks  # noqa: E402


def test_room_is_kitchen_by_prompt_and_metadata() -> None:
    assert ks._room_is_kitchen({"room_type": "kitchen"}, "cozy kitchen with island")
    assert ks._room_is_kitchen({"type": "something", "name": "Кухня"}, "open area")
    assert not ks._room_is_kitchen({"room_type": "bedroom"}, "loft bedroom")


def test_prompt_kitchen_width_parses_meters() -> None:
    assert ks._prompt_kitchen_width_m("Make kitchen 3.2m wide") == 3.2
    assert ks._prompt_kitchen_width_m("Ширина кухни 1,6 м") == 1.6
    assert ks._prompt_kitchen_width_m("no size here") is None


def test_room_size_and_default_wall_target(tmp_path: Path) -> None:
    room = {"width": 1.0, "depth": 1.2}
    assert ks._room_size(room) == (1.8, 1.8)
    assert ks._room_polygon_xy({"floor_polygon": [{"x": 0, "y": 0}, {"x": 2, "y": 0}, {"x": 2, "y": 1}, {"x": 0, "y": 1}]}) == [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)]

    target = ks._default_kitchen_target({}, prompt_text="width 2.5 м")
    assert target["id"] == "kitchen_001"
    assert target["category"] == "kitchen_set"
    assert target["kitchen_width_m"] >= 1.5


def test_append_kitchen_companion_targets_adds_table_and_chairs(tmp_path: Path, monkeypatch) -> None:
    room = {"width_m": 4.0, "depth_m": 3.0}
    items = [{"id": "wall_item", "category": "Lighting"}]

    monkeypatch.setattr(
        ks,
        "plan_dining_with_llm",
        lambda room, prompt_text, llm_settings=None: {
            "status": "ok",
            "add_dining": True,
            "chair_count": 2,
            "table": {"width_m": 1.0, "depth_m": 0.7, "y_m": 1.0, "yaw_deg": 0.0},
        },
    )

    additions = ks._append_kitchen_companion_targets(
        items,
        room=room,
        prompt_text="kitchen with dining",
        add_dining=True,
        add_accessories=False,
        inventory_index=None,
        accessory_llm_settings={"provider": "none"},
        dining_llm_settings={"provider": "none"},
    )
    assert additions
    roles = {item["role"] for item in additions}
    assert "dining_table" in roles
    assert "dining_chair" in roles
    assert len(items) >= 3
    assert any(item.get("category") == "dining_table" for item in items)


def test_default_accessory_plan_and_kitchen_wall_interval() -> None:
    assert ks._default_accessory_plan(dining_possible=True)[0]["role"] == "countertop_cooking_set"
    wall = {
        "a": (0.0, 0.0),
        "b": (4.0, 0.0),
        "u": (1.0, 0.0),
        "length": 4.0,
        "id": "w0",
        "n": (0.0, 1.0),
    }
    opening = {"wall_id": "w0", "s": 1.0, "width": 0.5}
    interval = ks._opening_interval_on_wall(opening, wall)
    assert interval == (0.63, 1.37)


def test_kitchen_stage_fallback_when_material_catalog_missing(tmp_path: Path) -> None:
    from src.pipeline_config import PlacementArtifacts

    place_v1 = tmp_path / "placement.v1.json"
    place_v1.write_text("{}", encoding="utf-8")
    scene_v1 = tmp_path / "scene.v1.json"
    scene_v1.write_text(json.dumps({"room": {"id": "r1"}}), encoding="utf-8")

    artifacts = PlacementArtifacts(
        placement_legacy=tmp_path / "placement_legacy.json",
        placement_v1=place_v1,
        scene_v1=scene_v1,
        scene_legacy=None,
    )
    updated, info = ks.apply_kitchen_stage_to_artifacts(
        artifacts=artifacts,
        run_dir=tmp_path,
        room_json_path=tmp_path / "room.json",
        material_catalog=tmp_path / "missing.json",
        appliance_catalog=None,
        prompt_text="kitchen",
        mode="balanced",
        policy="always",
        suffix="t1",
        dining_policy="auto",
        accessories_policy="auto",
        accessory_llm_settings={"provider": "none"},
        kitchen_llm_settings={"provider": "none"},
    )
    assert updated is artifacts
    assert info == {"skipped_reason": "material_catalog_missing", "material_catalog": str((tmp_path / "missing.json").resolve())}


def test_kitchen_stage_geometry_inventory_and_llm_helpers(tmp_path: Path, monkeypatch) -> None:
    room_path = tmp_path / "room.json"
    room_path.write_text(json.dumps({"room": {"room_type": "kitchen", "width_m": 4, "depth_m": 3}}), encoding="utf-8")
    assert ks._room_dict(None, room_path)["room_type"] == "kitchen"
    assert ks._float("bad", 2.5) == 2.5
    assert ks._item_aabb({"position_m": [1, 2, 1], "size_m": [2, 1, 2]})["x_min"] == 0.0

    room = {
        "floor_polygon": [{"x": 0, "y": 0}, {"x": 4, "y": 0}, {"x": 4, "y": 3}, {"x": 0, "y": 3}],
        "walls": [{"id": "w0", "from_vertex": 0, "to_vertex": 1}, {"id": "bad", "from_vertex": 99, "to_vertex": 0}],
        "doors": [{"wall_id": "w0", "s": 0.8, "width": 0.8}],
        "windows": [{"wall_id": "w0", "segment": {"x1": 2.0, "y1": 0.0, "x2": 2.4, "y2": 0.0}}],
    }
    walls = ks._wall_candidates(room)
    assert len(walls) == 1
    free = ks._free_wall_intervals(room, walls[0])
    assert free
    selected = ks._select_kitchen_wall_target(room, "кухня 2.5 м")
    assert selected and selected["wall_id"] == "w0"
    assert selected["kitchen_width_m"] <= 2.5
    assert abs(ks._default_kitchen_target({"width_m": 2, "depth_m": 4}, "3m")["rotation"][2]) == 90.0

    old_items = [
        {"id": "kit", "category": "KitchenCabinet"},
        {"id": "chair", "meta": {"companion_role": "dining_chair"}},
        {"id": "keep", "category": "sofa"},
    ]
    kept, removed = ks._remove_existing_kitchen_stage_objects(old_items, remove_dining=True)
    assert [item["id"] for item in kept] == ["keep"]
    assert {item["id"] for item in removed} == {"kit", "chair"}
    assert ks._has_companion_role([{"meta": {"companion_role": "tableware_set"}}], "tableware_set")

    inventory_index = {
        "kitchen_buckets": {
            "kitchenware": [
                {"unique_key": "used", "title": "Used plate", "asset_local_path": "/x.glb"},
                {"unique_key": "best", "title": "Ceramic plate set", "description": "tableware plate", "model_download_url": "https://x"},
            ],
            "food_fruit": [{"unique_key": "fruit", "title": "Fruit bowl", "width_cm": 30, "depth_cm": 20, "height_cm": 12}],
            "oil_bottles_decor": [],
            "flowers_vases": [],
        }
    }
    used = {"used"}
    picked = ks._pick_kitchen_inventory_item(inventory_index, bucket="kitchenware", prefer_terms=("plate",), used_keys=used)
    assert picked["unique_key"] == "best"
    assert "best" in used
    assert ks._inventory_rows_for_bucket(inventory_index, "food_fruit")[0]["unique_key"] == "fruit"
    assert ks._inventory_item_by_key(inventory_index, "fruit")["title"] == "Fruit bowl"
    assert ks._extract_llm_text({"message": {"content": " ok "}}) == "ok"
    assert ks._parse_json_object("```json\n{\"items\": []}\n```") == {"items": []}
    assert ks._bucket_candidates_for_llm(inventory_index, "kitchenware", limit=1)[0]["unique_key"] == "used"

    plan, info = ks._plan_kitchen_accessories_with_llm(
        inventory_index=inventory_index,
        room={"width_m": 4, "depth_m": 3},
        prompt_text="kitchen",
        dining_possible=False,
        llm_settings={"provider": "none"},
    )
    assert info["status"] == "skipped"
    assert plan

    fake_module = type(sys)("src.LLMModule.ollama_client")
    fake_module.chat_json = lambda **kwargs: {"response": json.dumps({"items": [
        {"role": "Oil bottles!", "bucket": "oil_bottles_decor", "surface": "countertop", "reason": "style"},
        {"role": "Bad", "bucket": "chairs", "surface": "countertop"},
    ]})}
    monkeypatch.setitem(sys.modules, "src.LLMModule.ollama_client", fake_module)
    llm_plan, llm_info = ks._plan_kitchen_accessories_with_llm(
        inventory_index=inventory_index,
        room={"width_m": 4, "depth_m": 3},
        prompt_text="kitchen",
        dining_possible=True,
        llm_settings={"provider": "ollama", "ollama_url": "http://fake"},
    )
    assert llm_info["status"] == "ok"
    assert llm_plan[0]["role"] == "oil_bottles"


def test_kitchen_stage_targets_replacement_and_report(tmp_path: Path, monkeypatch) -> None:
    assert ks._make_aabb(1, 1, 0, 2, 2, 1) == {"x_min": 0.0, "x_max": 2.0, "y_min": 0.0, "y_max": 2.0, "z_min": 0, "z_max": 1}
    target = ks._target_item(
        item_id="item1",
        name="Accessory",
        category="decor",
        center_xy=(1.0, 1.0),
        size=(0.2, 0.3, 0.4),
        z_min=0.9,
        meta={"x": 1},
    )
    assert target["mount_type"] == "surface"
    assert ks._chair_yaw_back_away_from_table((2, 1), (1, 1)) == 270.0
    assert ks._dimension_m_from_row({"dimensions_cm": {"width": 50}}, "width", 1.0) == 0.5
    assert ks._accessory_size_from_row({"width_cm": 500, "depth_cm": 2, "height_cm": 90}, (0.3, 0.3, 0.3)) == (0.62, 0.1, 0.7)
    assert ks._fallback_size_for_bucket("food_fruit") == (0.28, 0.28, 0.18)
    assert ks._surface_slots(surface="countertop", room_width=4, room_depth=3, dining_possible=True)
    packed = ks._pack_surface_items(
        [{"size": (0.4, 0.2, 0.1)}, {"size": (0.4, 0.2, 0.1)}],
        surface="countertop",
        room_width=3,
        room_depth=2,
        dining_possible=False,
    )
    assert len(packed) == 2

    assembly = {
        "id": "asm1",
        "mode": "balanced",
        "layout_type": "straight",
        "dimensions": {"width_m": 2.4, "depth_m": 0.6, "height_m": 2.2},
        "price_estimate": {"total_estimated_price": 123.45},
        "appliance_bindings": {
            "appliances": {
                "sink": {"chosen_asset": {"unique_key": "sink1", "title": "Sink"}, "top_candidates": [{}, {}]},
            },
            "unavailable_assets": {"oven": [{}, {}]},
        },
    }
    summary = ks._appliance_summary(assembly)
    assert summary["sink"]["top_candidate_count"] == 2
    assert summary["oven"]["unavailable_candidate_count"] == 2

    original = {"id": "kitchen_target", "name": "Old kitchen", "aabb": {"x_min": 0, "x_max": 2.4, "y_min": 0, "y_max": 0.6, "z_min": 0, "z_max": 2.2}}
    scene_item = ks._assembly_to_scene_item(assembly, original)
    assert scene_item["category"] == "kitchen_set"
    assert scene_item["asset"]["kind"] == "procedural_kitchen"
    assert scene_item["meta"]["source_target_id"] == "kitchen_target"

    monkeypatch.setattr(ks, "generate_kitchen_variants", lambda **kwargs: {"balanced": assembly})
    monkeypatch.setattr(ks, "build_kitchen_zone_from_target", lambda target, room: {"available_width_mm": 2000})
    monkeypatch.setattr(ks, "plan_dining_with_llm", lambda **kwargs: {"status": "ok", "add_dining": False})
    doc = {"placements": [{"id": "existing", "category": "KitchenCabinet"}], "meta": {}}
    material_catalog = tmp_path / "materials.json"
    material_catalog.write_text("[]", encoding="utf-8")
    out, replacements, additions = ks._replace_kitchens_in_doc(
        doc,
        room={"room_type": "kitchen", "width_m": 4, "depth_m": 3},
        material_catalog=material_catalog,
        appliance_catalog=None,
        inventory_index=None,
        prompt_text="kitchen",
        mode="balanced",
        add_if_missing=True,
        add_dining=True,
        add_accessories=False,
        accessory_llm_settings={"provider": "none"},
        kitchen_llm_settings={"provider": "none"},
    )
    assert replacements[0]["assembly_id"] == "asm1"
    assert additions == []
    assert out["placements"][0]["asset"]["kind"] == "procedural_kitchen"
    assert out["meta"]["kitchen_stage"]["replacement_count"] == 1

    report = ks.write_kitchen_report(tmp_path, replacements, [{"id": "chair", "role": "dining_chair", "category": "chair"}], suffix="unit")
    assert Path(report["summary_json"]).is_file()
    assert "Kitchen stage" in Path(report["markdown"]).read_text(encoding="utf-8")


def test_append_kitchen_accessory_targets_with_inventory_and_duplicate_roles(monkeypatch) -> None:
    inventory_index = {
        "kitchen_buckets": {
            "kitchenware": [
                {"unique_key": "plate", "title": "Ceramic plate set", "category_norm": "kitchenware", "asset_local_path": "/tmp/plate.glb"},
                {"unique_key": "cup", "title": "Cup set", "category_norm": "kitchenware", "asset_local_path": "/tmp/cup.glb"},
            ],
            "food_fruit": [{"unique_key": "fruit", "title": "Fruit bowl", "category_norm": "food_drink"}],
            "oil_bottles_decor": [{"unique_key": "oil", "title": "Oil bottles", "category_norm": "decorative_set"}],
            "flowers_vases": [{"unique_key": "vase", "title": "Small vase", "category_norm": "plant_planter_vase"}],
        }
    }
    monkeypatch.setattr(ks, "plan_dining_with_llm", lambda **_kwargs: {"status": "ok", "add_dining": True})
    monkeypatch.setattr(
        ks,
        "_plan_kitchen_accessories_with_llm",
        lambda **_kwargs: (
            [
                {"role": "plate set", "bucket": "kitchenware", "surface": "dining_table", "unique_key": "plate", "llm_reason": "serve dinner"},
                {"role": "plate set", "bucket": "kitchenware", "surface": "dining_table", "unique_key": "cup", "llm_reason": "duplicate role"},
                {"role": "oil bottles", "bucket": "oil_bottles_decor", "surface": "countertop"},
                {"role": "fruit bowl", "bucket": "food_fruit", "surface": "countertop"},
                {"role": "bad surface", "bucket": "flowers_vases", "surface": "shelf"},
            ],
            {"status": "ok", "model": "mock"},
        ),
    )

    items: list[dict] = []
    additions = ks._append_kitchen_companion_targets(
        items,
        room={"width_m": 4.0, "depth_m": 3.2},
        prompt_text="kitchen with dining and decor",
        add_dining=True,
        add_accessories=True,
        inventory_index=inventory_index,
        accessory_llm_settings={"provider": "none"},
        dining_llm_settings={"provider": "none"},
    )

    roles = {row["role"] for row in additions}
    assert {"dining_table", "dining_chair", "plate_set", "plate_set_2", "oil_bottles", "fruit_bowl", "bad_surface"} <= roles
    accessory_items = [item for item in items if item.get("meta", {}).get("support_surface")]
    assert len(accessory_items) == 5
    assert any(item["meta"].get("supplier_preferred_unique_key") == "plate" for item in accessory_items)
    assert any(item["meta"].get("kitchen_accessory_llm_reason") == "serve dinner" for item in accessory_items)


def test_apply_kitchen_stage_success_writes_scene_and_placement(tmp_path: Path, monkeypatch) -> None:
    from src.pipeline_config import PlacementArtifacts

    material_catalog = tmp_path / "materials.json"
    material_catalog.write_text("[]", encoding="utf-8")
    appliance_catalog = tmp_path / "appliances.json"
    appliance_catalog.write_text(json.dumps({"kitchen_buckets": {}}), encoding="utf-8")
    placement_v1 = tmp_path / "placement.v1.json"
    scene_v1 = tmp_path / "scene.v1.json"
    payload = {
        "room": {"room_type": "kitchen", "width_m": 4.0, "depth_m": 3.0},
        "placements": [{"id": "kitchen_target", "category": "KitchenCabinet", "aabb": {"x_min": 0, "x_max": 2.4, "y_min": 0, "y_max": 0.6, "z_min": 0, "z_max": 2.2}}],
    }
    placement_v1.write_text(json.dumps(payload), encoding="utf-8")
    scene_v1.write_text(json.dumps(payload), encoding="utf-8")

    assembly = {
        "id": "asm_full",
        "mode": "balanced",
        "layout_type": "straight",
        "dimensions": {"width_m": 2.4, "depth_m": 0.6, "height_m": 2.2},
        "price_estimate": {"total_estimated_price": 1000},
    }
    monkeypatch.setattr(ks, "generate_kitchen_variants", lambda **_kwargs: {"balanced": assembly})
    monkeypatch.setattr(ks, "build_kitchen_zone_from_target", lambda target, room: {"available_width_mm": 2400})
    monkeypatch.setattr(ks, "plan_dining_with_llm", lambda **_kwargs: {"status": "ok", "add_dining": False})

    updated, info = ks.apply_kitchen_stage_to_artifacts(
        artifacts=PlacementArtifacts(
            placement_legacy=tmp_path / "placement_legacy.json",
            placement_v1=placement_v1,
            scene_v1=scene_v1,
            scene_legacy=None,
        ),
        run_dir=tmp_path,
        room_json_path=tmp_path / "missing_room.json",
        material_catalog=material_catalog,
        appliance_catalog=appliance_catalog,
        prompt_text="kitchen",
        mode="balanced",
        policy="always",
        suffix="success",
        dining_policy="never",
        accessories_policy="never",
        accessory_llm_settings={"provider": "none"},
        kitchen_llm_settings={"provider": "none"},
    )

    assert info and info["replacement_count"] == 1
    assert updated.placement_v1.name == "placement_kitchen.success.v1.json"
    assert updated.scene_v1.name == "scene_kitchen.success.v1.json"
    assert json.loads(updated.placement_v1.read_text(encoding="utf-8"))["placements"][0]["asset"]["kind"] == "procedural_kitchen"
    assert Path(info["reports"]["summary_json"]).is_file()


def test_kitchen_stage_remaining_geometry_inventory_and_llm_edges(tmp_path: Path, monkeypatch) -> None:
    assert ks._room_dict(None, None) == {}
    assert ks._item_aabb({"dimensions": {"width_m": 2, "depth_m": 0.7, "height_m": 2.4}})["x_max"] == 2.5
    assert ks._room_polygon_xy({"floor_polygon": ["bad", {"x": "nan", "y": 0}, {"x": 1, "z": 2}]}) == [(1.0, 2.0)]
    assert ks._polygon_signed_area([(0, 0), (1, 0)]) == 0.0

    fallback_walls = ks._wall_candidates({"width_m": 3, "depth_m": 2})
    assert len(fallback_walls) == 4
    generated_walls = ks._wall_candidates({"floor_polygon": [{"x": 0, "y": 0}, {"x": 2, "y": 0}, {"x": 2, "y": 2}]})
    assert len(generated_walls) == 3
    assert ks._wall_candidates({"floor_polygon": [{"x": 0, "y": 0}, {"x": 0, "y": 0}], "walls": [{"from_vertex": 0, "to_vertex": 1}, "bad"]}) == []

    wall = {"id": "w0", "a": (0.0, 0.0), "u": (1.0, 0.0), "length": 2.0}
    assert ks._opening_interval_on_wall({"wall_id": "other", "s": 1.0}, wall) is None
    assert ks._opening_interval_on_wall({"wall_id": "w0"}, wall) is None
    assert ks._free_wall_intervals({"doors": [{"wall_id": "w0", "s": 1.0, "width": 2.0}]}, wall) == []
    assert ks._select_kitchen_wall_target({"width_m": 1.0, "depth_m": 1.0, "doors": [{"wall_id": "w0", "s": 0.5, "width": 5.0}]}, "") is None
    original_select = ks._select_kitchen_wall_target
    monkeypatch.setattr(ks, "_select_kitchen_wall_target", lambda *_args, **_kwargs: None)
    target = ks._default_kitchen_target({"width_m": 1.0, "depth_m": 4.0}, "2m")
    monkeypatch.setattr(ks, "_select_kitchen_wall_target", original_select)
    assert target["meta"]["along_long_wall"] is True

    assert not ks._has_target_like(["bad"], ("kitchen",))
    assert ks._is_infinigen_kitchen_object({"meta": {"procedural_assembly": "kitchen"}})
    assert ks._is_kitchen_stage_dining_object({"id": "kitchen_dining_chair_001"})
    kept, removed = ks._remove_existing_kitchen_stage_objects([{"id": "chair", "meta": {"companion_role": "dining_chair"}}, "raw"], remove_dining=False)
    assert kept[0]["id"] == "chair" and removed == []
    assert ks._has_companion_role(["bad"], "dining_table") is False

    assert ks._load_kitchen_selection_index(None) is None
    bad_catalog = tmp_path / "bad_catalog.json"
    bad_catalog.write_text("{bad", encoding="utf-8")
    assert ks._load_kitchen_selection_index(bad_catalog) is None
    assert ks._pick_kitchen_inventory_item({"kitchen_buckets": {"x": ["bad"]}}, bucket="x") is None
    assert ks._inventory_rows_for_bucket(None, "x") == []
    assert ks._inventory_item_by_key(None, "x") is None
    assert ks._extract_llm_text({"response": " ok "}) == "ok"
    assert ks._extract_llm_text({"x": 1}) == '{"x": 1}'
    assert ks._parse_json_object("prefix {\"items\": []} suffix") == {"items": []}
    assert ks._parse_json_object("[]") == {}

    plan, info = ks._plan_kitchen_accessories_with_llm(
        inventory_index=None,
        room={},
        prompt_text="kitchen",
        dining_possible=True,
        llm_settings={"provider": "openai"},
    )
    assert info["reason"] == "unsupported_provider"
    assert plan[-1]["surface"] == "dining_table"

    fake_module = type(sys)("src.LLMModule.ollama_client")
    fake_module.chat_json = lambda **_kwargs: {"message": {"content": json.dumps({"items": [{"bucket": "chairs", "surface": "countertop"}]})}}
    monkeypatch.setitem(sys.modules, "src.LLMModule.ollama_client", fake_module)
    fallback_plan, fallback_info = ks._plan_kitchen_accessories_with_llm(
        inventory_index=None,
        room={},
        prompt_text="kitchen",
        dining_possible=False,
        llm_settings={"provider": "ollama"},
    )
    assert fallback_info["status"] == "fallback"
    assert fallback_plan

    fake_module.chat_json = lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
    failed_plan, failed_info = ks._plan_kitchen_accessories_with_llm(
        inventory_index=None,
        room={},
        prompt_text="kitchen",
        dining_possible=False,
        llm_settings={"provider": "ollama"},
    )
    assert failed_info["status"] == "failed"
    assert failed_plan

    assert ks._room_size({"width": "bad", "depth": "bad"}) == (3.2, 3.0)
    assert ks._dimension_m_from_row({}, "width", 0.4) == 0.4
    assert ks._dimension_m_from_row({"width_cm": -1}, "width", 0.4) == 0.4
    assert ks._pack_surface_items([], surface="countertop", room_width=2, room_depth=2, dining_possible=False) == []
    assert ks._surface_slots(surface="dining_table", room_width=2.7, room_depth=2.0, dining_possible=True)

    monkeypatch.setattr(ks, "plan_dining_with_llm", lambda **_kwargs: {"status": "ok", "add_dining": True, "chair_count": 6})
    items: list[dict] = []
    additions = ks._append_kitchen_companion_targets(
        items,
        room={"width_m": 4, "depth_m": 3.2},
        prompt_text="kitchen dining",
        add_dining=True,
        add_accessories=False,
        inventory_index=None,
        accessory_llm_settings={"provider": "none"},
        dining_llm_settings={"provider": "none"},
    )
    assert sum(1 for row in additions if row["role"] == "dining_chair") == 4

    assembly = {"id": "a", "dimensions": {"width_m": 2.0, "depth_m": 0.5, "height_m": 2.2}}
    negative = ks._assembly_to_scene_item(assembly, {"id": "k", "rotation": [0, 0, -90], "aabb": {"x_min": 1, "x_max": 1.5, "y_min": 2, "y_max": 4, "z_min": 0, "z_max": 2}})
    assert negative["position"] == [1.0, 4.0, 0.0]
    bad_yaw = ks._assembly_to_scene_item(assembly, {"id": "k", "rotation": "bad", "aabb": {"x_min": 1, "x_max": 3, "y_min": 2, "y_max": 2.5, "z_min": 0, "z_max": 2}})
    assert bad_yaw["position"] == [1.0, 2.0, 0.0]


def test_kitchen_stage_replace_and_apply_skip_edges(tmp_path: Path, monkeypatch) -> None:
    from src.pipeline_config import PlacementArtifacts

    material_catalog = tmp_path / "materials.json"
    material_catalog.write_text("[]", encoding="utf-8")
    room = {"room_type": "bedroom", "width_m": 4.0, "depth_m": 3.0}
    unchanged, replacements, additions = ks._replace_kitchens_in_doc(
        {"meta": {}},
        room=room,
        material_catalog=material_catalog,
        appliance_catalog=None,
        inventory_index=None,
        prompt_text="bedroom",
        mode="balanced",
        add_if_missing=False,
        add_dining=False,
        add_accessories=False,
        accessory_llm_settings={"provider": "none"},
        kitchen_llm_settings={"provider": "none"},
    )
    assert unchanged == {"meta": {}}
    assert replacements == additions == []

    no_target, replacements, _ = ks._replace_kitchens_in_doc(
        {"items": [{"id": "sofa", "category": "sofa"}]},
        room=room,
        material_catalog=material_catalog,
        appliance_catalog=None,
        inventory_index=None,
        prompt_text="bedroom",
        mode="balanced",
        add_if_missing=False,
        add_dining=False,
        add_accessories=False,
        accessory_llm_settings={"provider": "none"},
        kitchen_llm_settings={"provider": "none"},
    )
    assert no_target["items"][0]["id"] == "sofa"
    assert replacements == []

    assembly = {"id": "asm", "mode": "balanced", "layout_type": "straight", "dimensions": {"width_m": 2.4, "depth_m": 0.6, "height_m": 2.2}}
    monkeypatch.setattr(ks, "generate_kitchen_variants", lambda **_kwargs: {"balanced": assembly})
    monkeypatch.setattr(ks, "build_kitchen_zone_from_target", lambda target, room: {})
    monkeypatch.setattr(ks, "plan_dining_with_llm", lambda **_kwargs: {"status": "ok", "add_dining": False})
    added_doc, replacements, _ = ks._replace_kitchens_in_doc(
        {"items": [{"id": "old", "category": "KitchenCabinet"}], "meta": {}},
        room={"room_type": "kitchen", "width_m": 4, "depth_m": 3},
        material_catalog=material_catalog,
        appliance_catalog=tmp_path / "missing_appliances.json",
        inventory_index=None,
        prompt_text="kitchen",
        mode="balanced",
        add_if_missing=True,
        add_dining=True,
        add_accessories=False,
        accessory_llm_settings={"provider": "none"},
        kitchen_llm_settings={"provider": "none"},
    )
    assert replacements[0]["removed_infinigen_kitchen_item_count"] == 1
    assert added_doc["items"][0]["category"] == "kitchen_set"

    artifacts = PlacementArtifacts(
        placement_legacy=tmp_path / "legacy.json",
        placement_v1=tmp_path / "placement.json",
        scene_v1=tmp_path / "scene.json",
        scene_legacy=None,
    )
    artifacts.placement_v1.write_text(json.dumps({"items": []}), encoding="utf-8")
    artifacts.scene_v1.write_text("[]", encoding="utf-8")
    assert ks.apply_kitchen_stage_to_artifacts(
        artifacts=artifacts,
        run_dir=tmp_path,
        room_json_path=tmp_path / "room.json",
        material_catalog=material_catalog,
        appliance_catalog=None,
        prompt_text="bedroom",
        mode="balanced",
        policy="never",
        suffix="skip",
    ) == (artifacts, None)

    list_placement = tmp_path / "list_placement.json"
    list_placement.write_text("[]", encoding="utf-8")
    list_artifacts = PlacementArtifacts(placement_legacy=tmp_path / "l.json", placement_v1=list_placement, scene_v1=None, scene_legacy=None)
    assert ks.apply_kitchen_stage_to_artifacts(
        artifacts=list_artifacts,
        run_dir=tmp_path,
        room_json_path=tmp_path / "room.json",
        material_catalog=material_catalog,
        appliance_catalog=None,
        prompt_text="kitchen",
        mode="balanced",
        policy="always",
        suffix="list",
    ) == (list_artifacts, None)
