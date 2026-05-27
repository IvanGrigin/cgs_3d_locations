import json
import math
import sys

import pytest

from src.tools import normalize_scene_format as nsf


def test_basic_helpers_and_json_roundtrip(tmp_path):
    path = tmp_path / "nested" / "payload.json"
    payload = {"ok": True, "items": [1, 2, 3]}

    nsf.save_json(path, payload)

    assert nsf.load_json(path) == payload
    assert nsf.is_number(1.5)
    assert not nsf.is_number(True)
    assert nsf.as_float("2.5") == 2.5
    assert nsf.as_float("bad", default=7.0) == 7.0
    assert nsf.as_int("12") == 12
    assert nsf.as_int("bad", default=4) == 4
    assert nsf.as_str(None, default="fallback") == "fallback"
    assert nsf.ensure_list3((1, "2", "bad")) == [1.0, 2.0, 0.0]
    assert nsf.ensure_list3([1, 2]) is None
    assert nsf.as_list3([1, 2], default=[3, 4, 5]) == [3, 4, 5]
    assert nsf.radians_from_deg(180) == pytest.approx(math.pi)
    assert nsf.degrees_from_rad(math.pi / 2) == pytest.approx(90.0)
    assert nsf.quantize_rot_0_90_180_270(269.0) == 270
    assert nsf.first_non_none(None, 0, "x") == 0
    assert nsf.deep_copy_dict([1, 2, 3]) == {}


def test_geometry_room_and_kind_helpers():
    aabb = nsf.build_aabb_from_center_size([1.0, 2.0, 0.5], [2.0, 4.0, 1.0])

    assert aabb == {"x_min": 0.0, "x_max": 2.0, "y_min": 0.0, "y_max": 4.0, "z_min": 0.0, "z_max": 1.0}
    assert nsf.center_from_aabb(aabb) == [1.0, 2.0, 0.5]
    assert nsf.size_from_aabb(aabb) == [2.0, 4.0, 1.0]
    assert nsf.build_center_from_xy_floor_and_size([3.0, 4.0], 0.2, [1.0, 1.0, 1.6]) == [3.0, 4.0, 1.0]
    assert nsf.mm3_to_m3([1000, 2000, 3000]) == [1.0, 2.0, 3.0]
    assert nsf.mm3_to_m3([1000, 2000]) is None

    assert nsf.normalize_room({"room": {"id": "r1"}, "source": "nested"}) == {"id": "r1", "source": "nested", "units": "m"}
    assert nsf.detect_input_kind({"schema": "objects.v1"}) == "objects_v1"
    assert nsf.detect_input_kind({"schema": "placement.v1"}) == "placement_v1"
    assert nsf.detect_input_kind({"schema": "scene.v1"}) == "scene_v1"
    assert nsf.detect_input_kind({"seed": 1, "items": [{"name": "chair"}]}) == "objects_like"
    assert nsf.detect_input_kind({"room": {}, "items": [{"center": [0, 0, 0]}]}) == "cube_or_old_scene"
    assert nsf.detect_input_kind({"placements": [{"center": [0, 0, 0]}]}) == "placement_like"
    assert nsf.auto_target_from_input_kind("cube_or_old_scene") == "scene"

    with pytest.raises(ValueError, match="target"):
        nsf.auto_target_from_input_kind("unknown")
    with pytest.raises(ValueError, match="JSON"):
        nsf.detect_input_kind([])
    with pytest.raises(ValueError, match="determine|определить"):
        nsf.detect_input_kind({"x": 1})


def test_objects_conversion_preserves_assets_meta_and_bounds():
    converted = nsf.convert_to_objects_v1(
        {
            "seed": "9",
            "items": [
                {
                    "uid": "bed-1",
                    "class_name": "Bed",
                    "asset_meta": {
                        "category": "sleeping",
                        "model_id": "m1",
                        "style": "minimal",
                        "size_x": 2.0,
                        "size_y": 1.8,
                        "size_z": 0.7,
                    },
                    "min_size_mm": [1900, 1700, 600],
                    "max_size_mm": [2100, 1900, 800],
                    "mesh_path": "bed.glb",
                    "color": ["0.1", "bad", 0.3, 1.0],
                    "custom_field": {"kept": True},
                }
            ],
            "run_id": "abc",
        }
    )

    obj = converted["objects"][0]
    assert converted["schema"] == "objects.v1"
    assert converted["seed"] == 9
    assert converted["meta"]["run_id"] == "abc"
    assert obj["id"] == "bed-1"
    assert obj["name"] == "Bed"
    assert obj["category"] == "sleeping"
    assert obj["size_m"] == pytest.approx([2.0, 1.8, 0.7])
    assert obj["size_min_m"] == [1.9, 1.7, 0.6]
    assert obj["size_max_m"] == [2.1, 1.9, 0.8]
    assert obj["color"] == [0.1, 0.7, 0.3, 1.0]
    assert obj["asset"]["model_id"] == "m1"
    assert obj["asset"]["mesh_path"] == "bed.glb"
    assert obj["meta"]["style"] == "minimal"
    assert obj["meta"]["custom_field"] == {"kept": True}

    assert nsf.normalize_objects_data({"objects": []})["objects"] == []
    with pytest.raises(ValueError, match="objects/items/placements"):
        nsf.convert_to_objects_v1({"bad": []})


def test_placement_and_scene_conversion_routes():
    source = {
        "placer": "test",
        "mode": "smoke",
        "placements": [
            {
                "object_id": "desk-1",
                "type": "Desk",
                "category": "desk",
                "position_room_xy_m": [2.0, 3.0],
                "z_floor_m": 0.1,
                "size": [1.2, 0.6, 0.8],
                "yaw_rad": math.pi / 2,
                "constraints": {"mount_type": "floor"},
                "placement_source": "manual",
                "server_index": 7,
                "forward": [0, 1, 0],
                "extra": "kept",
            },
            {
                "id": "lamp-1",
                "name": "Lamp",
                "bbox": {"x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1, "z_min": 0, "z_max": 2},
                "rotation": 181,
            },
        ],
        "llm_attempts_used": 3,
    }

    placement = nsf.convert_to_placement_v1(source)

    first = placement["placements"][0]
    second = placement["placements"][1]
    assert placement["schema"] == "placement.v1"
    assert placement["placer"] == "test"
    assert placement["mode"] == "smoke"
    assert placement["meta"]["llm_attempts_used"] == 3
    assert first["position_m"] == [2.0, 3.0, 0.5]
    assert first["rotation_deg"] == 90
    assert first["yaw_rad"] == pytest.approx(math.pi / 2)
    assert first["mount_type"] == "floor"
    assert first["source"] == {"placement_source": "manual", "server_index": 7}
    assert first["meta"]["forward"] == [0, 1, 0]
    assert first["meta"]["extra"] == "kept"
    assert second["position_m"] == [0.5, 0.5, 1.0]
    assert second["size_m"] == [1.0, 1.0, 2.0]
    assert second["rotation_deg"] == 180

    scene = nsf.convert_to_scene_v1({"room": {"id": "r1"}, **source})
    assert scene["schema"] == "scene.v1"
    assert scene["room"]["units"] == "m"
    assert len(scene["placements"]) == 2
    assert nsf.normalize_scene_data({"room": {"id": "r1"}, "items": []})["placements"] == []
    assert nsf.normalize_placement_data({"items": []})["placements"] == []

    with pytest.raises(ValueError, match="room"):
        nsf.convert_to_scene_v1({"placements": []})
    with pytest.raises(ValueError, match="placements/items"):
        nsf.convert_to_scene_v1({"room": {}})


def test_build_scene_convert_json_and_cli(tmp_path, monkeypatch, capsys):
    room = {"room": {"id": "r2"}}
    placement = {"placer": "unit", "placements": [{"id": "o1", "name": "Chair", "size_m": [1, 1, 1]}]}

    scene = nsf.build_scene_from_room_and_placement(room, placement)
    assert scene["schema"] == "scene.v1"
    assert scene["room"]["id"] == "r2"
    assert scene["meta"]["placer"] == "unit"
    assert scene["placements"][0]["position_m"] == [0.0, 0.0, 0.5]

    assert nsf.convert_json({"objects": []}, "auto")["schema"] == "objects.v1"
    assert nsf.convert_json({"placements": []}, "placement")["schema"] == "placement.v1"
    with pytest.raises(ValueError, match="target"):
        nsf.convert_json({"objects": []}, "bad")

    input_path = tmp_path / "input.json"
    output_path = tmp_path / "out.json"
    input_path.write_text(json.dumps({"objects": [{"name": "Box", "size_m": [1, 2, 3]}]}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["normalize_scene_format.py", "--input", str(input_path), "--output", str(output_path), "--target", "objects", "--print-kind"])
    nsf.main()
    assert json.loads(output_path.read_text(encoding="utf-8"))["schema"] == "objects.v1"
    assert "input_kind" in capsys.readouterr().out

    room_path = tmp_path / "room.json"
    placement_path = tmp_path / "placement.json"
    scene_path = tmp_path / "scene.json"
    room_path.write_text(json.dumps({"room": {"id": "r3"}}), encoding="utf-8")
    placement_path.write_text(json.dumps(placement), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["normalize_scene_format.py", "--room", str(room_path), "--placement", str(placement_path), "--output", str(scene_path), "--target", "scene"])
    nsf.main()
    assert json.loads(scene_path.read_text(encoding="utf-8"))["schema"] == "scene.v1"

    monkeypatch.setattr(sys, "argv", ["normalize_scene_format.py", "--output", str(tmp_path / "missing.json")])
    with pytest.raises(RuntimeError, match="--input"):
        nsf.main()


def test_normalize_scene_format_remaining_extractors_and_routes():
    assert nsf.deep_copy_dict({"a": {"b": 1}}) == {"a": {"b": 1}}
    assert nsf.detect_input_kind({"room": {}, "placements": []}) == "scene_like"
    assert nsf.detect_input_kind({"items": [{"center": [0, 0, 0]}]}) == "placement_like"
    assert nsf.detect_input_kind({"items": [{"name": "Lamp"}]}) == "items_like"
    assert nsf.auto_target_from_input_kind("placement_like") == "placement"

    asset_obj = {
        "asset": {"format": "glb"},
        "mesh_path": "model.glb",
        "mesh_fit_mode": "uniform",
        "mesh_texture_dirs": ["textures"],
    }
    assert nsf.extract_asset_block(asset_obj) == {
        "format": "glb",
        "mesh_path": "model.glb",
        "mesh_fit_mode": "uniform",
        "mesh_texture_dirs": ["textures"],
    }
    assert nsf.extract_name({"asset_meta": {"super-category": "decor"}}) == "decor"
    assert nsf.extract_category({"asset_meta": {"super_category": "storage"}}) == "storage"
    assert nsf.extract_meta_block_from_object({"meta": {"kept": True}})["kept"] is True
    assert nsf.extract_size_m_from_object_like({"size_m": [1, 2, 3]}) == [1.0, 2.0, 3.0]
    assert nsf.extract_size_m_from_object_like({"size": [4, 5, 6]}) == [4.0, 5.0, 6.0]
    assert nsf.extract_size_m_from_object_like({"asset_meta": {"size_x": 1, "size_y": 2, "size_z": 3}}) == [1.0, 2.0, 3.0]
    assert nsf.extract_size_bounds_m({"size_m": [1, 2, 3]}) == ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0])

    assert nsf.extract_position_m({"position_m": [1, 2, 3]}, [1, 1, 1], None) == [1.0, 2.0, 3.0]
    assert nsf.extract_position_m({"center": [4, 5, 6]}, [1, 1, 1], None) == [4.0, 5.0, 6.0]
    assert nsf.extract_position_m({"translation_m": [7, 8, 9]}, [1, 1, 1], None) == [7.0, 8.0, 9.0]
    assert nsf.extract_position_m({"position": [10, 11, 12]}, [1, 1, 1], None) == [10.0, 11.0, 12.0]
    assert nsf.extract_position_m_from_placement_item({"bbox": {"x_min": 0, "x_max": 2, "y_min": 0, "y_max": 2, "z_min": 0, "z_max": 2}}, [1, 1, 1]) == [1.0, 1.0, 1.0]
    assert nsf.extract_position_m_from_placement_item({}, [1, 1, 2]) == [0.0, 0.0, 1.0]

    assert nsf.extract_placement_size_m({"bbox_size_m": [1, 2, 3]}, None) == [1.0, 2.0, 3.0]
    assert nsf.extract_placement_size_m({"min_size_mm": [1000, 2000, 3000], "max_size_mm": [1000, 2000, 3000]}, None) == [1.0, 2.0, 3.0]
    assert nsf.extract_placement_size_m({"asset_meta": {"size_x": 1, "size_y": 2, "size_z": 3}}, None) == [1.0, 2.0, 3.0]
    assert nsf.extract_placement_size_m({}, None) is None
    assert nsf.extract_rotation_info({"rotation_deg": 91})[0] == 90
    assert nsf.extract_rotation_info({"yaw_deg": 181})[0] == 180
    assert nsf.extract_mount_type({"mount_type": " wall "}) == "wall"
    assert nsf.extract_source_block_for_placement({"source": {"origin": "unit"}, "placement_source": "manual"}, "fallback")["origin"] == "unit"

    objects = nsf.convert_to_objects_v1(
        {
            "placements": [
                {
                    "name": "Converted",
                    "asset_meta": {"category": "decor", "style": "modern"},
                    "asset_source": "catalog",
                    "future_jid": "jid1",
                }
            ],
            "meta": {"batch": 1},
        }
    )
    assert objects["objects"][0]["asset"]["model_id"] == "jid1"
    assert objects["objects"][0]["meta"]["style"] == "modern"

    scene_v1 = {"schema": "scene.v1", "room": {"id": "r"}, "placements": []}
    converted_scene = nsf.convert_json(scene_v1, "auto")
    assert converted_scene["schema"] == "scene.v1"
    assert converted_scene["room"]["units"] == "m"
    assert nsf.normalize_scene_data(scene_v1)["room"]["units"] == "m"
    assert nsf.normalize_placement_data({"schema": "placement.v1", "placements": []})["schema"] == "placement.v1"
