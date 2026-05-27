import json
from pathlib import Path

from src.apply_supplier_bindings import apply_supplier_bindings_to_data


def _room_scene(placements):
    return {
        "schema": "scene.v1",
        "room": {
            "type": "livingroom",
            "floor_polygon": [
                {"x": 0.0, "y": 0.0},
                {"x": 6.0, "y": 0.0},
                {"x": 6.0, "y": 6.0},
                {"x": 0.0, "y": 6.0},
            ],
            "floor_z": 0.0,
            "ceiling_height": 3.0,
        },
        "placements": placements,
    }


def _placement(item_id, category, x, y, z, sx=0.5, sy=0.5, sz=0.5, **extra):
    item = {
        "id": item_id,
        "name": category,
        "category": category,
        "position_m": [x, y, z],
        "size_m": [sx, sy, sz],
        "aabb": {
            "x_min": x - sx / 2,
            "x_max": x + sx / 2,
            "y_min": y - sy / 2,
            "y_max": y + sy / 2,
            "z_min": z - sz / 2,
            "z_max": z + sz / 2,
        },
    }
    item.update(extra)
    return item


def test_supplier_postprocess_deduplicates_ceiling_lights_to_room_center(tmp_path: Path) -> None:
    model_path = tmp_path / "lamp.obj"
    model_path.write_text("# obj\n", encoding="utf-8")
    placements = [
        _placement(f"light_{idx}", "CeilingLightFactory", x, y, 2.7)
        for idx, (x, y) in enumerate([(1.0, 1.0), (1.0, 5.0), (5.0, 1.0), (5.0, 5.0)])
    ]
    bindings = {
        "bindings": [
            {
                "target_id": item["id"],
                "category": "CeilingLightFactory",
                "semantic_group": "lamp_ceiling",
                "selection_status": "heuristic_top1_selected",
                "provenance": {"final_asset_source": "supplier_catalog"},
                "chosen_candidate": {
                    "title": "Ceiling lamp",
                    "semantic_group": "lamp_ceiling",
                    "asset_local_path": str(model_path),
                    "asset_format": "obj",
                    "asset_status": "archive_extracted_preferred",
                    "width_cm": 60,
                    "depth_cm": 60,
                    "height_cm": 30,
                },
            }
            for item in placements
        ]
    }

    out = apply_supplier_bindings_to_data(_room_scene(placements), bindings)

    lights = [p for p in out["placements"] if p["category"] == "CeilingLightFactory"]
    assert len(lights) == 4
    assert [p["position_m"][:2] for p in lights] != [[1.0, 1.0], [1.0, 5.0], [5.0, 1.0], [5.0, 5.0]]
    for light in lights:
        assert 1.0 <= light["position_m"][0] <= 5.0
        assert 1.0 <= light["position_m"][1] <= 5.0
        assert light["meta"]["ceiling_supplier_coverage_normalized"] is True
    summary = out["meta"]["supplier_binding_summary"]
    assert summary["ceiling_light_deduplicated_count"] == 0
    assert out["meta"]["supplier_postprocess"]["ceiling_lights"]["removed_count"] == 0
    assert out["meta"]["supplier_postprocess"]["ceiling_lights"]["count_preserved"] is True
    assert len(out["meta"]["supplier_postprocess"]["ceiling_lights"]["moved"]) == 4


def test_supplier_postprocess_adds_supplier_chair_for_desk_without_chair(tmp_path: Path, monkeypatch) -> None:
    model_path = tmp_path / "chair.fbx"
    model_path.write_text("fbx\n", encoding="utf-8")

    def fake_candidate(group, target_size):
        return {
            "unique_key": "test::chair",
            "source_site": "test",
            "title": "Supplier Chair",
            "semantic_group": "chair",
            "asset_status": "archive_extracted_preferred",
            "asset_format": "fbx",
            "asset_local_path": str(model_path),
            "width_cm": 48,
            "depth_cm": 55,
            "height_cm": 90,
        }

    monkeypatch.setattr("src.apply_supplier_bindings._candidate_from_supplier_db", fake_candidate)
    desk = _placement("desk_1", "SimpleDeskFactory", 3.0, 3.0, 0.4, sx=1.2, sy=0.7, sz=0.8)
    lamp = _placement(
        "lamp_1",
        "DeskLampFactory",
        3.0,
        3.0,
        0.95,
        sx=0.2,
        sy=0.2,
        sz=0.4,
        meta={"supplier_support_anchor_target_id": "desk_1"},
    )

    out = apply_supplier_bindings_to_data(_room_scene([desk, lamp]), {"bindings": []})

    assert [p["id"] for p in out["placements"]] == ["desk_1", "lamp_1", "auto_chair_for_desk_1"]
    chair = out["placements"][2]
    assert chair["category"] == "ChairFactory"
    assert chair["asset"]["mesh_path"] == str(model_path)
    assert chair["meta"]["supplier_affordance_added"] is True
    assert "supplier_generated_affordance" not in chair["meta"]
    assert chair["meta"]["target_table_id"] == "desk_1"
    assert chair["meta"]["placement_status"] == "valid"
    pp = out["meta"]["supplier_postprocess"]["table_chair_affordance"]
    assert pp["added_count"] == 1
    assert pp["added_ids"] == ["auto_chair_for_desk_1"]
    assert pp["tables"] == [
        {"table_id": "desk_1", "chair_id": "auto_chair_for_desk_1", "placement_status": "valid", "source": "supplier_catalog"}
    ]
    summary = out["meta"]["supplier_binding_summary"]
    assert summary["missing_table_chair_added_count"] == 1
    assert summary["unusable_table_suppressed_count"] == 0


def test_supplier_postprocess_keeps_desk_with_nearby_chair() -> None:
    desk = _placement("desk_1", "SimpleDeskFactory", 3.0, 3.0, 0.4, sx=1.2, sy=0.7, sz=0.8)
    chair = _placement("chair_1", "ChairFactory", 3.0, 2.15, 0.45, sx=0.5, sy=0.5, sz=0.9)

    out = apply_supplier_bindings_to_data(_room_scene([desk, chair]), {"bindings": []})

    assert [p["id"] for p in out["placements"]] == ["desk_1", "chair_1"]
    assert out["meta"]["supplier_postprocess"]["table_chair_affordance"]["added_count"] == 0


def test_supplier_postprocess_moves_existing_chair_to_wide_side() -> None:
    desk = _placement("desk_1", "SimpleDeskFactory", 3.0, 3.0, 0.4, sx=1.2, sy=0.7, sz=0.8)
    chair = _placement("chair_1", "ChairFactory", 5.0, 5.0, 0.45, sx=0.5, sy=0.5, sz=0.9)

    out = apply_supplier_bindings_to_data(_room_scene([desk, chair]), {"bindings": []})

    assert [p["id"] for p in out["placements"]] == ["desk_1", "chair_1"]
    moved = out["placements"][1]
    assert moved["meta"]["supplier_affordance_moved"] is True
    assert moved["meta"]["target_table_id"] == "desk_1"
    assert moved["aabb"]["x_min"] >= desk["aabb"]["x_min"]
    assert moved["aabb"]["x_max"] <= desk["aabb"]["x_max"]
    assert moved["aabb"]["y_min"] < desk["aabb"]["y_min"]
    assert moved["aabb"]["y_max"] <= desk["aabb"]["y_min"] + 0.19
    pp = out["meta"]["supplier_postprocess"]["table_chair_affordance"]
    assert pp["added_count"] == 0
    assert pp["moved_count"] == 1


def test_supplier_postprocess_adds_tv_on_empty_tv_stand(tmp_path: Path, monkeypatch) -> None:
    model_path = tmp_path / "tv.fbx"
    model_path.write_text("fbx\n", encoding="utf-8")

    def fake_tv_candidate(category_norms, target_size):
        return {
            "unique_key": "test::tv",
            "source_site": "test",
            "title": "Supplier TV",
            "category_norm": "tv_projector_screen",
            "semantic_group": "tv_projector_screen",
            "asset_status": "local_dir_preferred",
            "asset_format": "fbx",
            "asset_local_path": str(model_path),
            "width_cm": 110,
            "depth_cm": 6,
            "height_cm": 65,
        }

    monkeypatch.setattr("src.apply_supplier_bindings._candidate_from_supplier_catalog_json", fake_tv_candidate)
    stand = _placement("tv_stand_1", "TVStandFactory", 3.0, 1.0, 0.25, sx=1.6, sy=0.45, sz=0.5)
    sofa = _placement("sofa_1", "SofaFactory", 3.0, 4.5, 0.45, sx=2.0, sy=0.9, sz=0.9)

    out = apply_supplier_bindings_to_data(_room_scene([stand, sofa]), {"bindings": []})

    assert [p["id"] for p in out["placements"]] == ["tv_stand_1", "sofa_1", "auto_tv_for_tv_stand_1"]
    tv = out["placements"][2]
    assert tv["category"] == "WallMountedTVFactory"
    assert tv["asset"]["mesh_path"] == str(model_path)
    assert tv["meta"]["supplier_affordance_added"] is True
    assert tv["meta"]["affordance"] == "tv_on_stand"
    assert tv["aabb"]["z_min"] > stand["aabb"]["z_max"]
    pp = out["meta"]["supplier_postprocess"]["tv_affordance"]
    assert pp["added_count"] == 1
    assert pp["mode"] == "tv_stand_top"
    assert out["meta"]["supplier_binding_summary"]["missing_tv_added_count"] == 1


def test_supplier_postprocess_adds_wall_tv_opposite_sofa(tmp_path: Path, monkeypatch) -> None:
    model_path = tmp_path / "tv.fbx"
    model_path.write_text("fbx\n", encoding="utf-8")

    def fake_tv_candidate(category_norms, target_size):
        return {
            "unique_key": "test::tv",
            "source_site": "test",
            "title": "Supplier TV",
            "category_norm": "tv_projector_screen",
            "semantic_group": "tv_projector_screen",
            "asset_status": "local_dir_preferred",
            "asset_format": "fbx",
            "asset_local_path": str(model_path),
            "width_cm": 110,
            "depth_cm": 6,
            "height_cm": 65,
        }

    monkeypatch.setattr("src.apply_supplier_bindings._candidate_from_supplier_catalog_json", fake_tv_candidate)
    sofa = _placement("sofa_1", "SofaFactory", 3.0, 1.0, 0.45, sx=2.0, sy=0.9, sz=0.9)

    out = apply_supplier_bindings_to_data(_room_scene([sofa]), {"bindings": []})

    assert [p["id"] for p in out["placements"]] == ["sofa_1", "auto_tv_opposite_sofa_1"]
    tv = out["placements"][1]
    assert tv["meta"]["affordance"] == "tv_opposite_sofa"
    assert tv["aabb"]["y_min"] >= 0.0
    assert tv["aabb"]["y_max"] < 0.25
    pp = out["meta"]["supplier_postprocess"]["tv_affordance"]
    assert pp["added_count"] == 1
    assert pp["mode"] == "opposite_sofa"


def test_supplier_postprocess_does_not_add_tv_when_tv_exists(tmp_path: Path, monkeypatch) -> None:
    def fail_candidate(category_norms, target_size):
        raise AssertionError("TV candidate lookup should be skipped")

    monkeypatch.setattr("src.apply_supplier_bindings._candidate_from_supplier_catalog_json", fail_candidate)
    sofa = _placement("sofa_1", "SofaFactory", 3.0, 1.0, 0.45, sx=2.0, sy=0.9, sz=0.9)
    tv = _placement("tv_1", "TelevisionFactory", 3.0, 5.8, 1.2, sx=1.1, sy=0.06, sz=0.65)

    out = apply_supplier_bindings_to_data(_room_scene([sofa, tv]), {"bindings": []})

    assert [p["id"] for p in out["placements"]] == ["sofa_1", "tv_1"]
    pp = out["meta"]["supplier_postprocess"]["tv_affordance"]
    assert pp["added_count"] == 0
    assert pp["skipped_reason"] == "tv_already_present"
