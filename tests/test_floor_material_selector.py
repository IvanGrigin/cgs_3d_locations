import csv
import builtins
import json
from pathlib import Path

from src.ChooseObject.floor_material_normalizer import FloorMaterial, normalize_product, write_jsonl
from src.ChooseObject.floor_material_selector import FlooringRequest, FloorMaterialSelector, FloorPromptAnalyzer, _extract_ollama_text, _json_loads_or
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


def test_prompt_analyzer_json_helpers_and_score_branches(tmp_path):
    analyzer = FloorPromptAnalyzer()
    request = analyzer.build_flooring_request(
        "темная ванная без ламинат, SPC под камень, теплый пол, без сучков",
        style="modern",
        room_type="wc",
    )
    assert request.style == "contemporary"
    assert request.room_type == "bathroom"
    assert "vinyl_or_spc" in request.preferred_material_types
    assert "ламинат" in request.avoid_terms
    assert request.technical_requirements["water_resistant"] is True
    assert request.technical_requirements["warm_floor_compatible"] is True
    assert request.visual_requirements["avoid_strong_color_variation"] is True

    assert _extract_ollama_text({"message": {"content": "msg"}}) == "msg"
    assert _extract_ollama_text({"response": "resp"}) == "resp"
    assert _extract_ollama_text({"content": "content"}) == "content"
    assert _json_loads_or("prefix {\"x\": 1} suffix", {}) == {"x": 1}
    assert _json_loads_or("not json", {"fallback": True}) == {"fallback": True}

    laminate = FloorMaterial(
        sku="lam",
        name="Ламинат черный",
        material_type="laminate",
        design="wood",
        decor="oak",
        tone="black",
        class_=31,
        water_resistant=False,
        warm_floor_compatible=False,
        availability="out_of_stock",
        style_tags=["dark", "wood"],
        room_suitability=["bedroom"],
        bad_for=["bathroom"],
        search_text="ламинат черный дуб",
    )
    spc = FloorMaterial(
        sku="spc",
        name="SPC stone",
        material_type="vinyl_or_spc",
        design="stone",
        tone="gray",
        class_=43,
        water_resistant=True,
        warm_floor_compatible=True,
        availability="in_stock",
        style_tags=["stone", "waterproof"],
        room_suitability=["bathroom", "kitchen"],
        local_image_paths=["missing.jpg"],
        search_text="spc stone gray",
    )
    selector = FloorMaterialSelector(_write_materials(tmp_path, [laminate, spc]), _rules_path())
    filtered = selector.filter_materials([laminate, spc], request)
    assert filtered == [spc]
    bad_candidate = selector.score_material(laminate, request)
    good_candidate = selector.score_material(spc, request)
    assert good_candidate.final_score > bad_candidate.final_score
    assert "water_resistance_required" in bad_candidate.penalties
    assert "out_of_stock" in bad_candidate.penalties
    assert selector._llm_candidate_payload(good_candidate)["sku"] == "spc"

    top, info = selector._llm_rerank_candidates([good_candidate, bad_candidate], request, {"provider": "openai"})
    assert top[0].material.sku == "spc"
    assert info["status"] == "skipped"


def test_texture_analysis_crop_cache_and_llm_failure_paths(monkeypatch, tmp_path):
    try:
        from PIL import Image
    except ImportError:
        return

    image_dir = tmp_path / "images" / "sku"
    image_dir.mkdir(parents=True)
    texture = image_dir / "01.jpg"
    image = Image.new("RGB", (160, 160), "white")
    for x in range(32, 128):
        for y in range(32, 128):
            image.putpixel((x, y), (145 + (x % 4), 112 + (y % 4), 70))
    image.save(texture)

    material = FloorMaterial(
        sku="sku",
        name="Texture material",
        material_type="laminate",
        design="wood",
        decor="oak",
        tone="natural",
        local_image_paths=["images/sku/01.jpg"],
        image_urls=["https://example.test/fallback.jpg"],
        search_text="oak texture",
    )
    selector = FloorMaterialSelector(_write_materials(tmp_path, [material]), _rules_path())
    resolved = selector._resolve_texture_path("images/sku/01.jpg")
    assert resolved == texture
    assert selector._resolve_texture_path("missing.jpg") is None
    analysis = selector.analyze_texture_image(texture)
    assert analysis["usable_in_blender"] is True
    assert analysis["color_variation"]["variation_score"] < 0.6
    selected = selector.select_texture_candidate(material)
    assert selected["usable_in_blender"] is True
    assert "cropped_texture_path" in selected["analysis"]
    assert selector.select_texture_candidate(material) is selected
    assert Path(selected["texture_abs_path"]).is_file()

    all_white = tmp_path / "white.jpg"
    Image.new("RGB", (80, 80), "white").save(all_white)
    white_analysis = selector.analyze_texture_image(all_white)
    assert white_analysis["usable_in_blender"] is False
    assert white_analysis["reason"] == "all_white_after_mask"
    assert selector.analyze_texture_image(tmp_path / "not-image.jpg")["usable_in_blender"] is False

    no_local = FloorMaterial(sku="fallback", name="fallback", image_urls=["https://example.test/x.jpg"])
    fallback = selector.select_texture_candidate(no_local)
    assert fallback["reason"] == "no_local_images_to_analyze"
    assert fallback["texture_path"] == "https://example.test/x.jpg"

    light = FloorMaterial(sku="light", name="light", material_type="laminate", tone="light", image_urls=["u"])
    dark = FloorMaterial(sku="dark", name="dark", material_type="laminate", tone="dark", image_urls=["u"])
    candidates = [
        selector.score_material(light, selector.analyzer.build_flooring_request("light", "scandinavian", "bedroom")),
        selector.score_material(dark, selector.analyzer.build_flooring_request("light", "scandinavian", "bedroom")),
    ]

    import src.LLMModule.ollama_client as ollama_client

    monkeypatch.setattr(ollama_client, "chat_json", lambda **_kwargs: {"message": {"content": "{\"chosen_sku\": \"unknown\"}"}})
    unchanged, info = selector._llm_rerank_candidates(candidates, candidates[0].material and selector.analyzer.build_flooring_request("light", "scandinavian", "bedroom"), {"provider": "ollama", "top_n": 2})
    assert unchanged == candidates
    assert info["reason"] == "ollama_returned_unknown_sku"

    monkeypatch.setattr(ollama_client, "chat_json", lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("down")))
    unchanged, info = selector._llm_rerank_candidates(candidates, selector.analyzer.build_flooring_request("light", "scandinavian", "bedroom"), {"provider": "ollama", "top_n": 2})
    assert unchanged == candidates
    assert info["status"] == "failed"


def test_floor_selector_remaining_error_and_scoring_branches(monkeypatch, tmp_path):
    analyzer = FloorPromptAnalyzer()
    assert analyzer.extract_room_type("plain text without room") == "unknown_room"
    assert analyzer.normalize_room_type(None) is None
    assert analyzer.extract_technical_requirements("", "bedroom")["min_class"] == 32
    black_parquet = analyzer.build_flooring_request("black паркет в спальню", style="coastal", room_type=None)
    assert black_parquet.style == "scandinavian"
    assert {"black", "dark"} <= set(black_parquet.preferred_tones)
    assert {"parquet_board", "engineered_wood"} <= set(black_parquet.preferred_material_types)
    assert _extract_ollama_text(None) == ""
    assert _json_loads_or("prefix {bad json} suffix", {"fallback": True}) == {"fallback": True}

    known = [
        FloorMaterial(sku=f"known{i}", name=f"known {i}", material_type="laminate", parse_status="ok")
        for i in range(3)
    ]
    unknown = [
        FloorMaterial(sku=f"unknown{i}", name=f"unknown {i}", material_type="unknown_floor_material", parse_status="ok")
        for i in range(4)
    ]
    spc = [
        FloorMaterial(sku=f"spc{i}", name=f"spc {i}", material_type="vinyl_or_spc", parse_status="ok")
        for i in range(3)
    ]
    selector = FloorMaterialSelector(_write_materials(tmp_path, known + unknown + spc), _rules_path())

    request = analyzer.build_flooring_request("bedroom", "minimalism", "bedroom")
    assert selector.filter_materials(known + unknown, request) == known
    avoid_laminate = analyzer.build_flooring_request("без ламинат", "minimalism", "bedroom")
    assert selector.filter_materials(known + spc, avoid_laminate) == spc
    prefer_spc = analyzer.build_flooring_request("spc", "minimalism", "bedroom")
    assert selector.filter_materials(known + spc, prefer_spc) == spc

    failed_material = FloorMaterial(sku="failed", name="failed", parse_status="failed")
    fallback_selector = FloorMaterialSelector(_write_materials(tmp_path / "fallback", [failed_material]), _rules_path())
    fallback_selection = fallback_selector.select("anything", top_k=1)
    assert fallback_selection.selected_material.sku == "failed"
    empty_selector = FloorMaterialSelector(_write_materials(tmp_path / "empty", []), _rules_path())
    empty_selection = empty_selector.select("anything", top_k=1)
    assert empty_selection.selected_material is None
    assert empty_selection.texture_candidate is None

    selector.style_rules["custom"] = {"avoid_tones": ["dark"], "avoid_designs": ["tile"]}
    dark_tile = FloorMaterial(
        sku="dark_tile",
        name="dark tile",
        material_type="laminate",
        design="tile",
        tone="dark",
        class_=32,
        water_resistant=True,
        warm_floor_compatible=None,
        availability="in_stock",
        image_urls=["u"],
        search_text="dark tile",
    )
    strong_variation_request = analyzer.build_flooring_request("light", "custom", "kitchen")
    strong_variation_request.avoid_tones = ["dark"]
    strong_variation_request.visual_requirements = {
        "avoid_strong_color_variation": True,
        "max_color_variation_score": 0.1,
    }
    monkeypatch.setattr(
        selector,
        "select_texture_candidate",
        lambda _material: {
            "analysis": {
                "color_variation": {
                    "variation_score": 0.8,
                    "natural_darkening_risk": True,
                }
            }
        },
    )
    candidate = selector.score_material(dark_tile, strong_variation_request)
    assert "avoid_tone:dark" in candidate.penalties
    assert "style_avoid_tone:dark" in candidate.penalties
    assert "style_avoid_design:tile" in candidate.penalties
    assert "color_variation_too_high:0.800" in candidate.penalties
    warm_score, warm_penalties = selector._technical_score(
        FloorMaterial(sku="warm_unknown", name="warm unknown", warm_floor_compatible=None),
        FlooringRequest(technical_requirements={"warm_floor_compatible": True}),
    )
    assert warm_score > 0.55
    assert warm_penalties == []

    children_request = analyzer.build_flooring_request("детская dark", "minimalism", "children")
    children_score, children_penalties, _notes = selector._room_score(
        FloorMaterial(sku="kid", name="kid", class_=32, tone="dark"),
        children_request,
    )
    assert children_score < 0.7
    assert children_penalties == []

    candidates = [selector.score_material(known[0], request), selector.score_material(spc[0], request)]
    one_candidate, no_info = selector._llm_rerank_candidates(candidates[:1], request, {"provider": "ollama"})
    assert one_candidate == candidates[:1]
    assert no_info is None
    no_sku = selector.score_material(FloorMaterial(sku="", name="no sku"), request)
    one_sku, no_info = selector._llm_rerank_candidates([candidates[0], no_sku], request, {"provider": "ollama", "top_n": 2})
    assert one_sku == [candidates[0], no_sku]
    assert no_info is None

    original_import = builtins.__import__

    def import_without_ollama(name, *args, **kwargs):
        if name in {"src.LLMModule.ollama_client", "LLMModule.ollama_client"}:
            raise ImportError("blocked")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_ollama)
    unchanged, info = selector._llm_rerank_candidates(candidates, request, {"provider": "ollama", "top_n": 2})
    assert unchanged == candidates
    assert info["status"] == "failed"
    monkeypatch.setattr(builtins, "__import__", original_import)

    import src.LLMModule.ollama_client as ollama_client

    monkeypatch.setattr(ollama_client, "chat_json", lambda **_kwargs: {"message": {"content": "[]"}})
    unchanged, info = selector._llm_rerank_candidates(candidates, request, {"provider": "ollama", "top_n": 2})
    assert unchanged == candidates
    assert "LLM did not return JSON object" in info["reason"]
    monkeypatch.setattr(
        ollama_client,
        "chat_json",
        lambda **_kwargs: {"message": {"content": json.dumps({"chosen_sku": candidates[1].material.sku})}},
    )
    reranked, info = selector._llm_rerank_candidates(candidates, request, {"provider": "ollama", "top_n": 2})
    assert reranked[0].material.sku == candidates[1].material.sku
    assert info["ordered_skus"][0] == candidates[1].material.sku

    existing_texture = tmp_path / "absolute.jpg"
    existing_texture.write_bytes(b"img")
    assert selector._resolve_texture_path("") is None
    assert selector._resolve_texture_path(str(existing_texture.resolve())) == existing_texture.resolve()
    assert selector._percentile([], 0.5) == 0.0
    assert selector._analyze_color_variation([], width=1, min_x=0, min_y=1, max_x=0, max_y=0)["reason"] == "empty_texture_crop"

    original_import = builtins.__import__

    def import_without_pillow(name, *args, **kwargs):
        if name == "PIL" or name.startswith("PIL."):
            raise ImportError("no pillow")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_pillow)
    assert selector.analyze_texture_image(existing_texture)["reason"] == "Pillow is not installed"
    monkeypatch.setattr(builtins, "__import__", original_import)

    try:
        from PIL import Image
    except ImportError:
        return

    small = tmp_path / "small.jpg"
    Image.new("RGB", (20, 20), "white").save(small)
    material = FloorMaterial(sku="crop", name="crop")
    bad_bbox = {"crop_bbox": [0, 0, 10], "usable_in_blender": True}
    assert selector._save_color_variation_map(small, bad_bbox, material, 1) is None
    assert selector._save_texture_crop(small, bad_bbox, material, 1) is None
    small_crop = {"crop_bbox": [0, 0, 20, 20], "thumbnail_width": 20, "thumbnail_height": 20, "usable_in_blender": True}
    assert selector._save_color_variation_map(small, small_crop, material, 1) is None
    assert selector._save_texture_crop(small, small_crop, material, 1) is None

    monkeypatch.setattr(Image, "open", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("broken")))
    assert selector._save_color_variation_map(small, small_crop, material, 1) is None
    assert selector._save_texture_crop(small, small_crop, material, 1) is None
