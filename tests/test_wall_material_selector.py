import json
from pathlib import Path

from src.ChooseObject.wall_material_normalizer import WallMaterial, normalize_product, write_jsonl
from src.ChooseObject.wall_material_selector import WallMaterialSelector
from src.pipeline.wall_stage import apply_wall_material_to_scene


def _sample_row(name: str, props: dict, **overrides):
    row = {
        "url": "https://domlenta.ru/product/sample-801156/",
        "final_url": "https://domlenta.ru/product/sample-801156/",
        "name": name,
        "sku": "801156",
        "brand": "ER",
        "price": "1200",
        "price_currency": "RUB",
        "availability": "http://schema.org/InStock",
        "description": "Описание товара",
        "breadcrumbs": "Декоративные обои",
        "categories": "Обои",
        "properties_json": json.dumps(props, ensure_ascii=False),
        "images_json": json.dumps(["https://example.com/1.jpg"]),
        "local_image_paths_json": json.dumps([]),
        "parse_status": "ok",
        "error": "",
    }
    row.update(overrides)
    return row


def _write_materials(tmp_path: Path, materials: list[WallMaterial]) -> Path:
    path = tmp_path / "wall_materials.jsonl"
    write_jsonl(materials, path)
    return path


def test_normalize_wallpaper_color_pattern():
    material = normalize_product(_sample_row(
        "Обои флизелиновые Euro Decor Serena 7344-11 серые Россия",
        {"Тип": "Обои", "Цвет": "Серый", "Рисунок": "однотонный", "Материал": "Флизелин"},
    ))
    assert material.material_type == "wallpaper"
    assert material.base_material == "nonwoven"
    assert material.color == "gray"
    assert material.pattern == "plain"


def test_scandinavian_prefers_light_plain_over_dark_damask(tmp_path):
    light = normalize_product(_sample_row(
        "Обои флизелиновые бежевые однотонные",
        {"Тип": "Обои", "Цвет": "Бежевый", "Рисунок": "однотонный"},
        sku="light",
    ))
    dark = normalize_product(_sample_row(
        "Обои виниловые черные дамаск",
        {"Тип": "Обои", "Цвет": "Черный", "Рисунок": "дамаск"},
        sku="dark",
    ))
    selector = WallMaterialSelector(_write_materials(tmp_path, [dark, light]))
    selection = selector.select("Светлая спальня в скандинавском стиле, бежевые стены без рисунка", "scandinavian", "bedroom", top_k=2)
    assert selection.selected_material.sku == "light"


def test_apply_wall_material_to_scene_adds_room_wall_material():
    selection = {
        "room_id": "room_001",
        "selected_material": {
            "sku": "801156",
            "name": "Обои",
            "product_url": "https://example.com",
            "local_image_paths": ["images/801156/01.jpg"],
            "image_urls": [],
            "material_type": "wallpaper",
            "base_material": "nonwoven",
            "color": "beige",
            "tone": "warm_light",
            "pattern": "plain",
            "average_rgb": [210, 198, 180],
            "average_hex": "#d2c6b4",
            "dominant_colors_rgb": [[210, 198, 180]],
            "dominant_colors_hex": ["#d2c6b4"],
        },
    }
    scene = {"room": {"id": "room_001"}}
    updated = apply_wall_material_to_scene(scene, selection)
    assert updated["room"]["wall_material"]["sku"] == "801156"
    assert updated["room"]["wall_material"]["average_rgb"] == [210, 198, 180]


def test_llm_rerank_receives_color_payload(monkeypatch, tmp_path):
    light = normalize_product(_sample_row("Обои серые однотонные", {"Тип": "Обои", "Цвет": "Серый"}, sku="gray"))
    green = normalize_product(_sample_row("Обои зеленые ботанические", {"Тип": "Обои", "Цвет": "Зеленый", "Рисунок": "листья"}, sku="green"))
    light.average_rgb = [190, 190, 188]
    light.average_hex = "#bebebc"
    green.average_rgb = [80, 130, 90]
    green.average_hex = "#50825a"

    captured = {}

    def fake_chat_json(**kwargs):
        captured["prompt"] = kwargs["user_prompt"]
        return {"message": {"content": json.dumps({"chosen_sku": "green", "ordered_skus": ["green", "gray"], "reason": "green accent requested"})}}

    import src.LLMModule.ollama_client as ollama_client
    monkeypatch.setattr(ollama_client, "chat_json", fake_chat_json)

    selector = WallMaterialSelector(_write_materials(tmp_path, [light, green]))
    selection = selector.select(
        "зеленые стены с растительным мотивом",
        "contemporary",
        "bedroom",
        top_k=2,
        llm_settings={"provider": "ollama", "top_n": 2, "ollama_model": "stub"},
    )
    assert selection.selected_material.sku == "green"
    assert "average_rgb" in captured["prompt"]
