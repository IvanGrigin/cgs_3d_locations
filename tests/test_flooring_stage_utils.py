from pathlib import Path

from src.pipeline.flooring_stage import (
    _as_float,
    _floor_material_scene_payload,
    _infer_floor_tile_size_m,
    apply_flooring_to_scene,
    load_json,
    run_flooring_selection,
    write_json,
)


def test_as_float_handles_strings_and_invalid():
    assert _as_float("123") == 123.0
    assert _as_float("") is None
    assert _as_float("abc") is None
    assert _as_float(None) is None


def test_infer_floor_tile_size_defaults_and_clamps():
    assert _infer_floor_tile_size_m({"plank_length_mm": 1000}) == 1.0
    assert _infer_floor_tile_size_m({"plank_length_mm": 5000}) == 2.4
    assert _infer_floor_tile_size_m({"plank_width_mm": 250}) == 1.0
    assert _infer_floor_tile_size_m({}) == 1.2


def test_floor_material_scene_payload_prefers_local_texture_candidate():
    selection = {
        "selected_material": {
            "sku": "F123",
            "name": "Lacquer",
            "product_url": "https://example.com",
            "local_image_paths": ["local/path/tex.jpg"],
            "image_urls": ["https://cdn.local/tex.jpg"],
            "material_type": "laminate",
            "decor": "oak",
            "design": "wood",
            "tone": "dark",
            "plank_length_mm": 1320,
            "plank_width_mm": 190,
            "thickness_mm": 12,
            "class": 33,
        },
        "texture_candidate": {
            "texture_abs_path": "/abs/texture.jpg",
            "usable_in_blender": True,
            "reason": "ok",
            "analysis": {"color_variation": {"v": 0.1}},
        },
    }
    payload = _floor_material_scene_payload(selection)
    assert payload["source"] == "supplier_catalog"
    assert payload["texture_path"] == "/abs/texture.jpg"
    assert payload["texture_tiling"]["tile_size_m"] == 1.32
    assert payload["texture_color_variation"] == {"v": 0.1}


def test_apply_flooring_to_scene_prefers_matching_room_and_room_fallbacks(tmp_path: Path):
    selection = {"selected_material": {"sku": "F123", "material_type": "laminate"}, "room_id": "r-2"}

    scene = {
        "rooms": [
            {"id": "r-1", "name": "old"},
            {"id": "r-2", "name": "target"},
        ]
    }
    with_id = apply_flooring_to_scene(scene, selection)
    assert with_id["rooms"][1]["floor_material"]["sku"] == "F123"

    scene_without_match = {"rooms": [{"id": "r-1", "name": "first"}]}
    fallback = apply_flooring_to_scene(scene_without_match, selection)
    assert fallback["rooms"][0]["floor_material"]["sku"] == "F123"

    scene_without_rooms = {"room": {"id": "r-main"}}
    legacy = apply_flooring_to_scene(scene_without_rooms, selection)
    assert legacy["room"]["floor_material"]["sku"] == "F123"


def test_flooring_run_selection_and_json_helpers(monkeypatch, tmp_path: Path):
    class FakeSelection:
        def to_dict(self):
            return {"room_id": "r-1", "selected_material": {"material_type": "laminate"}}

    class FakeSelector:
        def __init__(self, *_, **__):
            pass

        def select(self, *_, **__):
            return FakeSelection()

        def save_selection(self, selection: FakeSelection, path: Path) -> None:
            path.write_text("ok", encoding="utf-8")

    monkeypatch.setattr("src.pipeline.flooring_stage.FloorMaterialSelector", FakeSelector)

    out_path = tmp_path / "selection.json"
    result = run_flooring_selection(
        "prompt",
        "modern",
        "living",
        "small room",
        "r-1",
        materials_path=tmp_path / "materials.json",
        style_rules_path=tmp_path / "rules.json",
        out_path=out_path,
        top_k=1,
    )
    assert result["selected_material"]["material_type"] == "laminate"
    assert out_path.read_text(encoding="utf-8") == "ok"

    json_path = tmp_path / "simple.json"
    payload = {"x": 1}
    write_json(payload, json_path)
    assert load_json(json_path) == payload
