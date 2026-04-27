import csv
import json
from pathlib import Path

from src.ChooseObject.floor_material_normalizer import FloorMaterial, normalize_product, write_jsonl
from src.ChooseObject.floor_material_selector import FloorMaterialSelector
from src.pipeline.flooring_stage import apply_flooring_to_scene


def _sample_row(name: str, props: dict, **overrides):
    row = {
        "url": "https://domlenta.ru/product/sample-799719/",
        "final_url": "https://domlenta.ru/product/sample-799719/",
        "name": name,
        "sku": "799719",
        "brand": "SWISSKRONO",
        "price": "2636.51",
        "price_currency": "RUB",
        "availability": "http://schema.org/InStock",
        "description": "Описание товара",
        "breadcrumbs": "Напольные покрытия",
        "categories": "Ламинат",
        "properties_json": json.dumps(props, ensure_ascii=False),
        "images_json": json.dumps(["https://example.com/1.jpg"]),
        "local_image_paths_json": json.dumps(["images/799719/01.jpg"]),
        "parse_status": "ok",
        "error": "",
    }
    row.update(overrides)
    return row


def _rules_path() -> Path:
    return Path("config/flooring_style_rules.json")


def _write_materials(tmp_path, materials):
    path = tmp_path / "materials.jsonl"
    write_jsonl(materials, path)
    return path


def test_normalize_laminate():
    row = _sample_row(
        "Ламинат SWISSKRONO HOME STANDARD ДУБ ЛИВИНЬО 33 класс 10 мм 1,845 м²",
        {
            "Тип": "Ламинат",
            "Декор": "Под дуб",
            "Название декора": "ДУБ ЛИВИНЬО",
            "Толщина планки": "10",
            "Класс": "33",
            "Оттенок": "Тёмный",
            "Фаска": "Четырехсторонняя",
        },
    )
    material = normalize_product(row)
    assert material.material_type == "laminate"
    assert material.decor == "oak"
    assert material.design == "wood"
    assert material.tone == "dark"
    assert material.class_value == 33
    assert material.thickness_mm == 10
    assert material.package_area_m2 == 1.845
    assert material.chamfer == "four_sided"


def test_scandinavian_bedroom_prefers_light_wood_over_dark_concrete(tmp_path):
    light = normalize_product(_sample_row(
        "Ламинат светлый дуб натуральный 33 класс 8 мм 2,0 м²",
        {"Тип": "Ламинат", "Декор": "Под дуб", "Оттенок": "Светлый", "Класс": "33", "Толщина": "8"},
        sku="1",
    ))
    concrete = normalize_product(_sample_row(
        "Керамогранит темный бетон 33 класс 10 мм",
        {"Тип": "Керамогранит", "Дизайн": "Под бетон", "Оттенок": "Тёмный", "Класс": "33"},
        sku="2",
    ))
    selector = FloorMaterialSelector(_write_materials(tmp_path, [concrete, light]), _rules_path())
    selection = selector.select("Светлая спальня в скандинавском стиле, натуральный дуб", "scandinavian", "bedroom", top_k=2)
    assert selection.selected_material.sku == "1"


def test_bathroom_does_not_choose_ordinary_laminate_when_vinyl_exists(tmp_path):
    laminate = normalize_product(_sample_row(
        "Ламинат дуб обычный 33 класс 8 мм",
        {"Тип": "Ламинат", "Декор": "Под дуб", "Класс": "33"},
        sku="lam",
    ))
    vinyl = normalize_product(_sample_row(
        "Кварцвиниловая плитка SPC камень светлый 43 класс",
        {"Тип": "Кварцвинил SPC", "Дизайн": "Камень", "Класс": "43"},
        sku="spc",
    ))
    selector = FloorMaterialSelector(_write_materials(tmp_path, [laminate, vinyl]), _rules_path())
    selection = selector.select("Ванная с влагостойким покрытием под камень", "minimalism", "bathroom", top_k=2)
    assert selection.selected_material.sku == "spc"


def test_hallway_boosts_class_33(tmp_path):
    low = normalize_product(_sample_row(
        "Ламинат дуб 31 класс светлый",
        {"Тип": "Ламинат", "Декор": "Под дуб", "Класс": "31", "Оттенок": "Светлый"},
        sku="31",
    ))
    high = normalize_product(_sample_row(
        "Ламинат дуб 33 класс светлый",
        {"Тип": "Ламинат", "Декор": "Под дуб", "Класс": "33", "Оттенок": "Светлый"},
        sku="33",
    ))
    selector = FloorMaterialSelector(_write_materials(tmp_path, [low, high]), _rules_path())
    selection = selector.select("Светлая прихожая с дубовым полом", "contemporary", "hallway", top_k=2)
    assert selection.selected_material.sku == "33"


def test_selection_json_shape(tmp_path):
    material = normalize_product(_sample_row(
        "Ламинат светлый дуб 33 класс 8 мм 2,0 м²",
        {"Тип": "Ламинат", "Декор": "Под дуб", "Оттенок": "Светлый", "Класс": "33"},
    ))
    selector = FloorMaterialSelector(_write_materials(tmp_path, [material]), _rules_path())
    selection = selector.select("Светлая спальня", "scandinavian", "bedroom")
    out = tmp_path / "flooring.selection.v1.json"
    selector.save_selection(selection, out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["version"] == "flooring.selection.v1"
    assert "request" in data
    assert "selected_material" in data
    assert "selection_reason" in data
    assert "top_candidates" in data


def test_apply_flooring_to_scene_adds_room_floor_material():
    selection = {
        "room_id": "room_001",
        "selected_material": {
            "sku": "799719",
            "name": "Ламинат",
            "product_url": "https://example.com",
            "local_image_paths": ["images/799719/01.jpg"],
            "image_urls": [],
            "material_type": "laminate",
            "decor": "oak",
            "design": "wood",
            "tone": "light",
            "plank_length_mm": 1380,
            "plank_width_mm": 191,
            "thickness_mm": 10,
            "class": 33,
        },
    }
    scene = {"room": {"id": "room_001"}}
    updated = apply_flooring_to_scene(scene, selection)
    assert updated["room"]["floor_material"]["sku"] == "799719"
    assert updated["room"]["floor_material"]["texture_path"] == "images/799719/01.jpg"


def test_llm_rerank_can_choose_candidate(monkeypatch, tmp_path):
    light = normalize_product(_sample_row(
        "Ламинат светлый дуб натуральный 33 класс 8 мм 2,0 м²",
        {"Тип": "Ламинат", "Декор": "Под дуб", "Оттенок": "Светлый", "Класс": "33"},
        sku="light",
    ))
    dark = normalize_product(_sample_row(
        "Ламинат темный дуб табак 33 класс 10 мм 1,8 м²",
        {"Тип": "Ламинат", "Декор": "Под дуб", "Оттенок": "Тёмный", "Класс": "33"},
        sku="dark",
    ))

    def fake_chat_json(**kwargs):
        return {"message": {"content": json.dumps({"chosen_sku": "dark", "ordered_skus": ["dark", "light"], "reason": "prompt asks dark oak"})}}

    import src.LLMModule.ollama_client as ollama_client
    monkeypatch.setattr(ollama_client, "chat_json", fake_chat_json)

    selector = FloorMaterialSelector(_write_materials(tmp_path, [light, dark]), _rules_path())
    selection = selector.select(
        "спальня дуб",
        "scandinavian",
        "bedroom",
        top_k=2,
        llm_settings={"provider": "ollama", "top_n": 2, "ollama_model": "stub"},
    )
    assert selection.selected_material.sku == "dark"
    assert selection.llm_rerank["status"] == "applied"
