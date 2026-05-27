from __future__ import annotations

import csv
import json
from pathlib import Path

from src.ChooseObject import floor_material_normalizer as floor
from src.ChooseObject import wall_material_normalizer as wall


def test_floor_normalizer_helpers_catalog_loading_and_surface_items(tmp_path: Path):
    assert floor._safe_json("", {"x": 1}) == {"x": 1}
    assert floor._safe_json("null", {"x": 1}) == {"x": 1}
    assert floor._parse_float("1 234,50 мм") == 1234.50
    assert floor._parse_int("33 класс") == 33
    assert floor._rgb_to_hex([-1, 16, 300]) == "#0010ff"
    assert floor.normalize_availability("http://schema.org/OutOfStock") == "out_of_stock"
    assert floor.normalize_material_type("кварцвинил SPC") == "vinyl_or_spc"
    assert floor.normalize_design_and_decor("серый бетон и мрамор")[1] == "marble"
    assert floor.normalize_tone("венге темный")[0] == "dark"
    assert floor.normalize_chamfer("4V четырехсторонняя") == "four_sided"
    assert floor._package_area("Ламинат 2,13 м²", {}) == 2.13
    assert "premium" in floor._style_tags("parquet_board", "walnut", "wood", "dark")
    assert "bathroom" in floor._room_suitability("vinyl_or_spc", True)[0]

    try:
        from PIL import Image
    except ImportError:
        Image = None
    image_rel = "imgs/floor.jpg"
    if Image is not None:
        image_path = tmp_path / image_rel
        image_path.parent.mkdir()
        Image.new("RGB", (16, 16), (110, 90, 70)).save(image_path)
        colors = floor.analyze_floor_material_colors(tmp_path, [image_rel], k=2)
        assert colors["average_hex"] is not None
        assert colors["dominant_colors_rgb"]

    row = {
        "url": "https://mosplitka.ru/product/x/",
        "name": "Керамогранит серый бетон",
        "sku": "F1",
        "brand": "B",
        "price": "1000",
        "availability": "в наличии",
        "properties_json": json.dumps({"Тип": "Керамогранит", "Дизайн": "бетон", "Класс": "33", "Толщина": "9"}),
        "images_json": json.dumps(["https://example.test/f.jpg"]),
        "local_image_paths_json": json.dumps([image_rel]),
        "room_recommendations_json": json.dumps([{"room": "bathroom"}]),
        "style_recommendations_json": json.dumps([{"style": "loft"}]),
        "parse_status": "ok",
    }
    material = floor.normalize_product(row, base_dir=tmp_path, analyze_images=Image is not None)
    assert material.source == "mosplitka"
    assert material.material_type == "porcelain_tile"
    assert "bathroom" in material.room_suitability
    assert "loft" in material.style_tags

    jsonl = tmp_path / "normalized_floor_materials.jsonl"
    floor.write_jsonl([material, material], jsonl)
    loaded = floor.load_normalized_materials(jsonl)
    assert len(loaded) == 1
    assert loaded[0].local_image_paths
    assert floor.load_normalized_materials(tmp_path)[0].sku == "F1"

    surface_item = {
        "version": "surface_material.v1",
        "source": "surface",
        "sku": "S1",
        "name": "Surface floor",
        "url": "https://example.test/s",
        "price": 12.0,
        "availability": "in_stock",
        "normalized": {
            "is_selectable_floor": True,
            "material_type": "porcelain_tile",
            "visual_pattern": "stone",
            "base_color": "gray",
            "tone": "neutral",
            "style_tags": ["minimalism"],
            "rooms": ["bathroom"],
            "tile_width_cm": "30",
            "tile_height_cm": "60",
        },
        "material_image": {"path": image_rel, "image_url": "https://example.test/img.jpg"},
        "raw_properties": {"Класс": "33", "Страна": "RU"},
        "text_facts": {"country": "RU"},
    }
    converted = floor._floor_material_from_catalog_item(surface_item, tmp_path)
    assert converted is not None
    assert converted.plank_width_mm == 300
    assert converted.water_resistant is True
    assert floor._floor_material_from_catalog_item({"version": "other"}, tmp_path) is None
    assert floor._dedupe_floor_materials([material, material]) == [material]

    csv_path = tmp_path / "products.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    out_jsonl = tmp_path / "out.jsonl"
    normalized = floor.normalize_domlenta_catalog(csv_path, out_jsonl, analyze_images=False)
    assert len(normalized) == 1
    assert out_jsonl.is_file()


def test_floor_normalizer_remaining_error_and_fallback_branches(tmp_path: Path, monkeypatch):
    import builtins
    import sys
    import types

    assert floor._safe_json("{bad", {"x": 1}) == {"x": 1}
    assert floor._parse_float("abc") is None
    assert floor._parse_int("abc") is None
    assert floor._rgb_to_hex(None) is None
    assert floor._rgb_to_hex([1, 2]) is None
    assert floor._kmeans_rgb([]) == []
    assert floor._kmeans_rgb([(0, 0, 0), (10, 0, 0), (250, 0, 0), (255, 0, 0)], k=2, iterations=2)
    assert floor.analyze_floor_material_colors(tmp_path, ["missing.jpg"]) == {
        "average_rgb": None,
        "average_hex": None,
        "dominant_colors_rgb": [],
        "dominant_colors_hex": [],
    }
    assert floor.normalize_availability("unknown stock text") == "unknown"
    assert floor.normalize_material_type("инженерная доска") == "engineered_wood"
    assert floor.normalize_material_type("паркетная доска") == "parquet_board"
    assert floor.normalize_material_type("линолеум") == "linoleum"
    assert floor.normalize_material_type("плитка") == "ceramic_tile"
    assert floor.normalize_material_type("not a floor") == "unknown_floor_material"
    assert floor.normalize_design_and_decor("под дерево однотон") == ("plain", "plain")
    assert floor.normalize_tone("white") == ("white", "white")
    assert floor.normalize_tone("gray") == ("gray", "gray")
    assert floor.normalize_tone("натурал") == ("natural", "brown")
    assert floor.normalize_tone("dark oak") == ("dark", "brown")
    assert floor.normalize_tone("uncolored") == (None, None)
    assert floor.normalize_chamfer("без фаски") == "none"
    assert floor.normalize_chamfer("2V двухсторонняя") == "two_sided"
    assert floor._package_area("No area", {"Площадь упаковки": "3,5 м²"}) == 3.5
    assert {"ash", "light", "natural"} <= set(floor._style_tags("laminate", "ash", "wood", "light"))
    assert {"classic", "baroque", "premium"} <= set(floor._style_tags("tile", "marble", "marble", "beige"))
    assert floor._room_suitability("unknown", False) == ([], [])

    real_float = builtins.float

    def broken_float(value):
        if value == "12":
            raise ValueError("boom")
        return real_float(value)

    monkeypatch.setattr(builtins, "float", broken_float)
    assert floor._parse_float("12") is None
    monkeypatch.setattr(builtins, "float", real_float)

    real_import = builtins.__import__

    def missing_pil_import(name, *args, **kwargs):
        if name == "PIL":
            raise ImportError("no pil")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_pil_import)
    assert floor._sample_image_pixels(tmp_path / "x.jpg") == []
    monkeypatch.setattr(builtins, "__import__", real_import)

    class FakeImage:
        def __init__(self, pixels):
            self._pixels = pixels

        def draft(self, *_args, **_kwargs):
            raise RuntimeError("draft unsupported")

        def thumbnail(self, _size):
            return None

        def convert(self, _mode):
            return self

        def getdata(self):
            return self._pixels

    fake_image_mod = types.SimpleNamespace(open=lambda _path: FakeImage([(255, 255, 255), (0, 0, 0), (10, 20, 30)]))
    monkeypatch.setitem(sys.modules, "PIL", types.SimpleNamespace(Image=fake_image_mod))
    assert floor._sample_image_pixels(tmp_path / "fake.jpg") == [(10, 20, 30)]

    fake_image_mod.open = lambda _path: FakeImage([(250, 250, 250), (0, 0, 0)])
    assert floor._sample_image_pixels(tmp_path / "blank.jpg") == [(250, 250, 250), (0, 0, 0)]

    fake_image_mod.open = lambda _path: FakeImage([(i % 255, 20, 30) for i in range(30)])
    assert len(floor._sample_image_pixels(tmp_path / "many.jpg", max_pixels=5)) == 5

    fake_image_mod.open = lambda _path: (_ for _ in ()).throw(OSError("bad image"))
    assert floor._sample_image_pixels(tmp_path / "bad.jpg") == []
    monkeypatch.delitem(sys.modules, "PIL", raising=False)

    try:
        from PIL import Image
    except ImportError:
        Image = None
    if Image is not None:
        image_path = tmp_path / "colors.png"
        Image.new("RGB", (2, 2), (20, 40, 60)).save(image_path)
        no_palette = floor.analyze_floor_material_colors(tmp_path, [str(image_path)], k=0)
        assert no_palette["average_rgb"] == [20, 40, 60]
        assert no_palette["dominant_colors_rgb"] == []

    bad_json_row = {
        "url": "https://domlenta.ru/product/laminat-1/",
        "name": "теплый пол дуб",
        "properties_json": "[]",
        "images_json": "{}",
        "local_image_paths_json": "{}",
        "room_recommendations_json": "{}",
        "style_recommendations_json": "{}",
        "availability": "unknown",
    }
    normalized = floor.normalize_product(bad_json_row, base_dir=tmp_path, analyze_images=False)
    assert normalized.warm_floor_compatible is True
    assert normalized.image_urls == []
    assert normalized.local_image_paths == []

    original_resolve = Path.resolve

    def flaky_resolve(self: Path, *args, **kwargs):
        if self.name == "local.jpg":
            raise OSError("unit resolve failure")
        return original_resolve(self, *args, **kwargs)

    local_img = tmp_path / "local.jpg"
    local_img.write_text("x", encoding="utf-8")
    monkeypatch.setattr(Path, "resolve", flaky_resolve)
    assert floor._abs_local_paths([str(local_img)], tmp_path) == [str(local_img)]
    assert wall._abs_local_paths([str(local_img)], tmp_path) == [str(local_img)]

    products_csv = tmp_path / "broken_products.csv"
    with products_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["name", "url"])
        writer.writeheader()
        writer.writerow({"name": "Fallback", "url": "https://example.test/f"})
    monkeypatch.setattr(floor, "normalize_product", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("bad")))
    fallback = floor.normalize_domlenta_catalog(products_csv, tmp_path / "fallback.jsonl", analyze_images=False)
    assert fallback[0].name == "Fallback"

    invalid_jsonl = tmp_path / "surface_materials_catalog.jsonl"
    invalid_jsonl.write_text("{bad\n", encoding="utf-8")
    assert invalid_jsonl in floor._surface_catalog_paths(tmp_path)
    assert floor._load_floor_materials_from_dir(tmp_path) == []

    blank_jsonl = tmp_path / "blank.jsonl"
    blank_jsonl.write_text("\n", encoding="utf-8")
    assert floor.load_normalized_materials(blank_jsonl) == []
    assert floor._floor_material_from_catalog_item(None, tmp_path) is None
    assert floor._floor_material_from_catalog_item(
        {"version": "surface_material.v1", "normalized": {"is_selectable_floor": False}},
        tmp_path,
    ) is None


def test_wall_normalizer_helpers_catalog_loading_and_surface_items(tmp_path: Path):
    assert wall._safe_json("", []) == []
    assert wall._parse_float("10,05 м") == 10.05
    assert wall._rgb_to_hex([255, 0, 128]) == "#ff0080"
    assert wall.normalize_availability("out_of_stock") == "out_of_stock"
    assert wall.normalize_wall_material_type("Фотообои") == "photo_wallpaper"
    assert wall.normalize_base_material("винил на флизелине") == "nonwoven"
    assert wall.normalize_color_and_tone("оливково зеленый")[0] == "green"
    assert wall.normalize_pattern("геометрический орнамент") == "geometric"
    assert "baroque" in wall._style_tags("wallpaper", "black", "dark", "damask")

    try:
        from PIL import Image
    except ImportError:
        Image = None
    image_rel = "imgs/wall.jpg"
    if Image is not None:
        image_path = tmp_path / image_rel
        image_path.parent.mkdir(exist_ok=True)
        Image.new("RGB", (16, 16), (180, 170, 150)).save(image_path)
        colors = wall.analyze_wallpaper_colors(tmp_path, [image_rel], k=2)
        assert colors["average_hex"] is not None

    row = {
        "url": "https://example.test/w",
        "name": "Обои флизелиновые зеленые листья",
        "sku": "W1",
        "brand": "WB",
        "price": "500",
        "availability": "in_stock",
        "categories": "Обои",
        "breadcrumbs": "Обои",
        "description": "ботанический рисунок",
        "properties_json": json.dumps({"Тип": "Обои", "Цвет": "Зеленый", "Рисунок": "листья", "Материал": "Флизелин", "Ширина": "106", "Длина": "10"}),
        "images_json": json.dumps(["https://example.test/w.jpg"]),
        "local_image_paths_json": json.dumps([image_rel]),
        "parse_status": "ok",
    }
    material = wall.normalize_product(row, base_dir=tmp_path, analyze_images=Image is not None)
    assert material.material_type == "wallpaper"
    assert material.base_material == "nonwoven"
    assert material.pattern == "botanical"
    assert material.width_cm == 106
    assert material.length_m == 10

    jsonl = tmp_path / "normalized_wall_materials.jsonl"
    wall.write_jsonl([material, material], jsonl)
    loaded = wall.load_normalized_wall_materials(jsonl)
    assert len(loaded) == 1
    assert loaded[0].sku == "W1"
    assert wall.load_normalized_wall_materials(tmp_path)[0].sku == "W1"

    surface_item = {
        "version": "surface_material.v1",
        "source": "surface",
        "sku": "SW1",
        "name": "Surface wall",
        "url": "https://example.test/sw",
        "price": 30,
        "availability": "in_stock",
        "normalized": {
            "is_selectable_wall": True,
            "material_type": "wallpaper",
            "visual_pattern": "plain",
            "base_color": "beige",
            "tone": "warm_light",
            "surface_finish": "matte",
            "style_tags": ["scandinavian"],
            "rooms": ["bedroom"],
            "width_cm": "106",
            "length_m": "10",
        },
        "material_image": {"path": image_rel, "image_url": "https://example.test/sw.jpg"},
        "raw_properties": {"Страна": "RU", "Материал основы": "Флизелин"},
        "text_facts": {"base_material": "nonwoven"},
        "text_description_ru": "plain beige wallpaper",
    }
    converted = wall._wall_material_from_catalog_item(surface_item, tmp_path)
    assert converted is not None
    assert converted.color == "beige"
    assert converted.width_cm == 106
    assert wall._wall_material_from_catalog_item({"version": "other"}, tmp_path) is None
    assert wall._dedupe_wall_materials([material, material]) == [material]

    csv_path = tmp_path / "wall_products.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    out_jsonl = tmp_path / "walls.jsonl"
    normalized = wall.normalize_domlenta_wallpapers_catalog(csv_path, out_jsonl, analyze_images=False)
    assert len(normalized) == 1
    assert out_jsonl.is_file()


def test_wall_normalizer_remaining_error_and_fallback_branches(tmp_path: Path, monkeypatch):
    import builtins
    import sys
    import types

    assert wall._safe_json("{bad", []) == []
    assert wall._rgb_to_hex(None) is None
    assert wall.normalize_availability("unknown") == "unknown"
    assert wall.normalize_wall_material_type("жидкие обои") == "liquid_wallpaper"
    assert wall.normalize_wall_material_type("стеновая панель") == "wall_panel"
    assert wall.normalize_wall_material_type("краска интерьерная") == "paint"
    assert wall.normalize_wall_material_type("unknown") == "unknown_wall_material"
    assert wall.normalize_base_material("бумага") == "paper"
    assert wall.normalize_base_material("стеклообои") == "fiberglass"
    assert wall.normalize_color_and_tone("темный") == (None, "dark")
    assert wall.normalize_color_and_tone("светлый") == (None, "light")
    assert wall.normalize_color_and_tone("no color") == (None, None)
    assert wall.normalize_pattern("no pattern") is None
    assert {"loft", "industrial"} <= set(wall._style_tags("wallpaper", None, None, "concrete"))
    assert wall._kmeans_rgb([]) == []
    assert wall._kmeans_rgb([(0, 0, 0), (10, 0, 0), (250, 0, 0), (255, 0, 0)], k=2, iterations=2)
    assert wall.analyze_wallpaper_colors(tmp_path, ["missing.jpg"]) == {
        "average_rgb": None,
        "average_hex": None,
        "dominant_colors_rgb": [],
        "dominant_colors_hex": [],
    }

    real_float = builtins.float

    def broken_float(value):
        if value == "12":
            raise ValueError("boom")
        return real_float(value)

    monkeypatch.setattr(builtins, "float", broken_float)
    assert wall._parse_float("12") is None
    monkeypatch.setattr(builtins, "float", real_float)

    real_import = builtins.__import__

    def missing_pil_import(name, *args, **kwargs):
        if name == "PIL":
            raise ImportError("no pil")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_pil_import)
    assert wall._sample_image_pixels(tmp_path / "x.jpg") == []
    monkeypatch.setattr(builtins, "__import__", real_import)

    class FakeImage:
        def __init__(self, pixels):
            self._pixels = pixels

        def convert(self, _mode):
            return self

        def thumbnail(self, _size):
            return None

        def getdata(self):
            return self._pixels

    fake_image_mod = types.SimpleNamespace(open=lambda _path: FakeImage([(255, 255, 255), (0, 0, 0), (10, 20, 30)]))
    monkeypatch.setitem(sys.modules, "PIL", types.SimpleNamespace(Image=fake_image_mod))
    assert wall._sample_image_pixels(tmp_path / "fake.jpg") == [(10, 20, 30)]

    fake_image_mod.open = lambda _path: FakeImage([(i % 255, 30, 40) for i in range(30)])
    assert len(wall._sample_image_pixels(tmp_path / "many.jpg", max_pixels=5)) == 5

    fake_image_mod.open = lambda _path: (_ for _ in ()).throw(OSError("bad image"))
    assert wall._sample_image_pixels(tmp_path / "bad.jpg") == []
    monkeypatch.delitem(sys.modules, "PIL", raising=False)

    bad_json_row = {
        "url": "https://example.test/wall-1/",
        "name": "панель светлая",
        "properties_json": "[]",
        "images_json": "{}",
        "local_image_paths_json": "{}",
        "availability": "unknown",
    }
    normalized = wall.normalize_product(bad_json_row, base_dir=tmp_path, analyze_images=False)
    assert normalized.material_type == "wall_panel"
    assert normalized.image_urls == []
    assert normalized.local_image_paths == []

    products_csv = tmp_path / "broken_wall_products.csv"
    with products_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["name", "url"])
        writer.writeheader()
        writer.writerow({"name": "Fallback wall", "url": "https://example.test/w"})
    monkeypatch.setattr(wall, "normalize_product", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("bad")))
    fallback = wall.normalize_domlenta_wallpapers_catalog(products_csv, tmp_path / "fallback_wall.jsonl", analyze_images=False)
    assert fallback[0].name == "Fallback wall"

    invalid_jsonl = tmp_path / "surface_materials_wall.jsonl"
    invalid_jsonl.write_text("{bad\n", encoding="utf-8")
    assert invalid_jsonl in wall._surface_catalog_paths(tmp_path)
    assert wall._load_wall_materials_from_dir(tmp_path) == []

    blank_jsonl = tmp_path / "blank_walls.jsonl"
    blank_jsonl.write_text("\n", encoding="utf-8")
    assert wall.load_normalized_wall_materials(blank_jsonl) == []
    assert wall._wall_material_from_catalog_item(None, tmp_path) is None
    assert wall._wall_material_from_catalog_item(
        {"version": "surface_material.v1", "normalized": {"is_selectable_wall": False}},
        tmp_path,
    ) is None
