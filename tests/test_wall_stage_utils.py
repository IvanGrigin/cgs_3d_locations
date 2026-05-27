from pathlib import Path

from src.pipeline import wall_stage
from src.pipeline.wall_stage import (
    apply_wall_material_to_scene,
    apply_wall_material_to_scene_with_catalog,
    _resolve_texture_path,
    _wall_material_scene_payload,
    load_json,
    run_wall_selection,
    write_json,
)


def test_resolve_texture_path_variants(tmp_path: Path):
    assert _resolve_texture_path("", tmp_path) is None
    assert _resolve_texture_path("https://example.com/tex.jpg", tmp_path) == "https://example.com/tex.jpg"

    rel_root = tmp_path / "textures"
    rel_root.mkdir()
    rel_file = rel_root / "tex.jpg"
    rel_file.write_text("x", encoding="utf-8")

    assert _resolve_texture_path("textures/tex.jpg", Path(str(tmp_path / "materials.json"))) == str(rel_file.resolve())

    abs_file = tmp_path / "abs.jpg"
    abs_file.write_text("x", encoding="utf-8")
    assert _resolve_texture_path(str(abs_file), Path(str(tmp_path / "materials.json"))) == str(abs_file.resolve())

    missing = _resolve_texture_path("missing.tex", Path(str(tmp_path / "materials.json")))
    assert missing.endswith("missing.tex")


def test_resolve_texture_path_handles_resolve_errors(monkeypatch, tmp_path: Path):
    original_resolve = Path.resolve

    def flaky_resolve(self: Path, *args, **kwargs):
        if self.name == "broken.jpg":
            raise OSError("unit failure")
        return original_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", flaky_resolve)

    tex = tmp_path / "broken.jpg"
    tex.write_text("x", encoding="utf-8")
    assert _resolve_texture_path(str(tex), tmp_path / "materials.json") == str(tex)


def test_wall_material_scene_payload_and_catalog_path_resolution(tmp_path: Path):
    material_path = tmp_path / "catalog" / "wall.jpg"
    material_path.parent.mkdir()
    material_path.write_text("x", encoding="utf-8")

    selection = {
        "selected_material": {
            "sku": "W1",
            "name": "Paper",
            "product_url": "https://example.com/wall",
            "local_image_paths": ["catalog/wall.jpg"],
            "image_urls": ["https://cdn.local/wall.jpg"],
            "material_type": "wallpaper",
            "base_material": "paper",
            "color": "gray",
            "tone": "light",
            "pattern": "plain",
            "average_rgb": [1, 2, 3],
            "average_hex": "#010203",
            "dominant_colors_rgb": [[1, 2, 3]],
            "dominant_colors_hex": ["#010203"],
        },
    }

    materials_path = tmp_path / "materials.json"
    payload = _wall_material_scene_payload(selection, materials_path=materials_path)
    assert payload["texture_path"] == str(material_path.resolve())

    room = {"rooms": [{"id": "r-1"}]}
    scene = apply_wall_material_to_scene_with_catalog(room, selection, materials_path=materials_path)
    assert scene["rooms"][0]["wall_material"]["sku"] == "W1"
    assert scene["rooms"][0]["wall_material"]["texture_path"] == str(material_path.resolve())


def test_apply_wall_material_to_scene_fallbacks_and_run_selection(monkeypatch, tmp_path: Path):
    scene = {"room": {"id": "old"}}
    selection = {"selected_material": {"sku": "W2"}, "room_id": "r-main"}
    assert apply_wall_material_to_scene(scene, selection)["room"]["wall_material"]["sku"] == "W2"

    class FakeSelection:
        def to_dict(self):
            return {"room_id": "r-main", "selected_material": {"material_type": "wallpaper"}}

    class FakeSelector:
        def __init__(self, *_, **__):
            pass

        def select(self, *_, **__):
            return FakeSelection()

        def save_selection(self, selection: FakeSelection, path: Path) -> None:
            path.write_text("ok", encoding="utf-8")

    monkeypatch.setattr("src.pipeline.wall_stage.WallMaterialSelector", FakeSelector)

    out_path = tmp_path / "wall_selection.json"
    result = run_wall_selection(
        "prompt",
        "modern",
        "bedroom",
        "small room",
        "r-main",
        materials_path=tmp_path / "catalog.jsonl",
        out_path=out_path,
        top_k=1,
    )
    assert result["selected_material"]["material_type"] == "wallpaper"
    assert out_path.read_text(encoding="utf-8") == "ok"


def test_wall_material_room_list_branches_and_json_helpers(tmp_path: Path):
    selection = {"selected_material": {"sku": "W3"}, "room_id": "target"}
    scene = {"rooms": [{"id": "other"}, {"id": "target"}]}
    updated = apply_wall_material_to_scene(scene, selection)
    assert updated["rooms"][1]["wall_material"]["sku"] == "W3"
    assert "wall_material" not in scene["rooms"][1]

    fallback = apply_wall_material_to_scene({"rooms": [{"id": "first"}]}, {"selected_material": {"sku": "W4"}, "room_id": "missing"})
    assert fallback["rooms"][0]["wall_material"]["sku"] == "W4"

    material_path = tmp_path / "tex.jpg"
    material_path.write_text("x", encoding="utf-8")
    with_catalog = apply_wall_material_to_scene_with_catalog(
        {"rooms": [{"id": "first"}]},
        {"selected_material": {"sku": "W5", "local_image_paths": [str(material_path)]}, "room_id": "missing"},
        tmp_path / "catalog.json",
    )
    assert with_catalog["rooms"][0]["wall_material"]["texture_path"] == str(material_path.resolve())

    dict_path = tmp_path / "nested" / "scene.json"
    write_json({"ok": True}, dict_path)
    assert load_json(dict_path) == {"ok": True}


def test_apply_wall_material_with_catalog_matching_and_root_fallbacks(tmp_path: Path):
    material_path = tmp_path / "tex.jpg"
    material_path.write_text("x", encoding="utf-8")
    selection = {"selected_material": {"sku": "W7", "local_image_paths": [str(material_path)]}, "room_id": "target"}

    matched = apply_wall_material_to_scene_with_catalog({"rooms": [{"id": "target"}]}, selection, tmp_path / "catalog.json")
    assert matched["rooms"][0]["wall_material"]["sku"] == "W7"

    root = apply_wall_material_to_scene_with_catalog({"room": {"id": "root"}}, selection, tmp_path / "catalog.json")
    assert root["room"]["wall_material"]["sku"] == "W7"


def test_run_wall_selection_passes_llm_settings(monkeypatch, tmp_path: Path):
    seen = {}

    class FakeSelection:
        def to_dict(self):
            return {"selected_material": {"sku": "W6"}}

    class FakeSelector:
        def __init__(self, *, materials_path):
            seen["materials_path"] = materials_path

        def select(self, **kwargs):
            seen.update(kwargs)
            return FakeSelection()

        def save_selection(self, selection, path):
            seen["saved"] = path

    monkeypatch.setattr(wall_stage, "WallMaterialSelector", FakeSelector)
    result = run_wall_selection(
        "prompt",
        None,
        None,
        None,
        "room",
        materials_path=tmp_path / "wall.jsonl",
        out_path=tmp_path / "selection.json",
        top_k=3,
        llm_settings={"provider": "none"},
    )
    assert result["selected_material"]["sku"] == "W6"
    assert seen["llm_settings"] == {"provider": "none"}
    assert seen["saved"].name == "selection.json"
