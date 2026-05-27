import json
import sqlite3
import sys
import types
from pathlib import Path

import pytest

from src import supplier_layout_matcher as m


def good_target(**overrides):
    target = {
        "target_id": "bed_1",
        "category": "bed",
        "semantic_group": "bed",
        "size_m": [1.6, 2.0, 0.9],
        "replacement_policy": "replace_with_supplier",
        "constraints": {"style": "modern", "material": "wood", "color": "brown"},
    }
    target.update(overrides)
    return target


def good_row(asset_path=None, **overrides):
    row = {
        "unique_key": "bed-key",
        "source_site": "homeconcept",
        "source_url": "https://example.test/bed",
        "title": "Modern brown wooden bed",
        "title_en": "Modern brown wooden bed",
        "title_ru": "Современная коричневая деревянная кровать",
        "category_raw": "bed",
        "category_raw_en": "bed",
        "category_norm": "bed",
        "semantic_group": "bed",
        "brand": "Brand",
        "collection": "Line",
        "product_url": "https://example.test/bed",
        "price_value": 50000,
        "price_currency": "RUB",
        "style": "modern",
        "style_llm": "modern",
        "style_llm_confidence": 0.9,
        "style_llm_quality_score": 8,
        "style_llm_secondary": ["minimalism"],
        "color": "brown",
        "color_en": "brown",
        "materials": "wood",
        "materials_en": "wood",
        "description": "Modern solid wood bed for bedroom",
        "description_short_en": "Brown wood bed",
        "search_text_en": "modern brown wood bed",
        "width_cm": 160,
        "depth_cm": 200,
        "height_cm": 100,
        "room": "bedroom",
        "availability": "free",
        "images_json": json.dumps(["https://example.test/bed.jpg"]),
        "extra_json": "{}",
        "model_format": "glb",
        "asset_status": "ready",
        "asset_format": "glb",
        "asset_local_path": str(asset_path) if asset_path else "",
    }
    row.update(overrides)
    return row


def test_preferences_styles_and_json_helpers(tmp_path):
    payload_path = tmp_path / "payload.json"
    m.write_json(payload_path, {"ok": True})
    assert m.read_json(payload_path) == {"ok": True}

    assert m._json_loads_or('{"x": 1}', {}) == {"x": 1}
    assert m._json_loads_or("bad", {"fallback": True}) == {"fallback": True}
    assert m._is_truthy("yes")
    assert not m._is_truthy("no")
    assert m._dedup_keep_order(["a", "a", "", "b"]) == ["a", "b"]
    assert m._value_to_text_list(["red, green", ("blue",)]) == ["red", "green", "blue"]
    assert set(m._normalize_color_preference_list("grey, дуб")) == {"brown", "gray"}
    assert m._normalize_site_list(["HomeConcept", "homeconcept"]) == ["homeconcept"]
    assert m._normalize_brand_list("Brand, Other") == ["Brand", "Other"]
    assert m._safe_float("1.25") == 1.25
    assert m._safe_float("x") is None

    assert m._normalize_style_label("mid-century") == "mid_century_modern"
    assert {"japandi", "eco_organic"} <= m._extract_styles_from_text("Japandi wabi sabi bedroom")

    raw_preferences = {
        "global": {"max_price": "1000", "preferred_color": "beige", "allowed_sites": "a,b"},
        "by_semantic_group": {"bed": {"avoid_colors": ["red"], "strict_color": "1"}},
        "by_target_id": {"bed_1": {"preferred_brand": "Brand", "require_model_url": True}},
    }
    normalized = m._normalize_user_preferences(raw_preferences)
    prefs = m._target_user_preferences(
        {"target_id": "bed_1", "semantic_group": "bed"},
        {"user_preferences": normalized},
    )
    assert prefs["max_price_rub"] == 1000.0
    assert prefs["preferred_colors"] == ["beige"]
    assert prefs["avoid_colors"] == ["red"]
    assert prefs["allowed_sites"] == ["a", "b"]
    assert prefs["preferred_brands"] == ["Brand"]
    assert prefs["strict_color"] is True
    assert prefs["require_model_url"] is True

    assert m._selection_strategy({"supplier_selection_strategy": "unknown"}) == "balanced"
    assert m._selection_strategy({"supplier_selection_strategy": "best_visual_reference"}) == "style"


def test_dimensions_assets_tokens_and_fit_helpers(tmp_path):
    asset = tmp_path / "model.glb"
    asset.write_bytes(b"glb")
    proxy = tmp_path / "proxy.glb"
    proxy.write_bytes(b"glb")

    assert m._dimension_value_to_cm("1200", "mm") == 120.0
    assert m._dimension_value_to_cm("1.2", "m") == 120.0
    assert m._dimension_value_to_cm("1.2", None) == 120.0
    row = good_row(asset, title="Ширина 160 см глубина 200 см высота 100 см")
    assert m._infer_dimensions_cm_from_text(row)["width"] == 160.0
    assert m._row_dimension_cm(row, "depth") == 200.0
    assert m._product_size_m(row) == [1.6, 2.0, 1.0]
    assert m._effective_target_size_m({"semantic_group": "lamp_ceiling", "size_m": [0.2, 0.2, 0.05]}) == [0.6, 0.6, 0.45]
    assert m._has_full_dimensions(row)
    assert m._has_category(row)
    assert m._row_is_rich(row)

    assert m._candidate_has_ready_real_asset(row)
    assert not m._candidate_has_ready_real_asset(good_row(proxy, asset_local_path=str(proxy), asset_format="glb"))
    assert m._candidate_has_downloadable_asset({"model_download_url": "https://x.test/model.rar", "model_format": "rar"})
    assert not m._candidate_has_downloadable_asset({"model_download_url": "https://drive.google.com/file", "model_download_filename": "view"})

    assert {"wood", "brown", "bed"} <= m._normalize_text_tokens("Wood BedFactory brown oak bed")
    assert m._normalize_color_token("grey") == "gray"
    assert m._rgb_to_basic_color_tokens([0.1, 0.12, 0.13]) == {"black"}
    assert "green" in m._extract_color_tokens(["olive", [0.1, 0.5, 0.2]])
    assert "bed" in m._target_category_tokens(good_target())
    assert "wood" in m._target_design_tokens(good_target())
    assert "brown" in m._target_color_tokens({"constraints": {"color": "brown"}})
    assert "modern" in m._target_query_tokens(good_target())
    assert "bed" in m._row_query_tokens(row)
    assert "bed" in m._row_category_tokens(row)
    assert "wood" in m._row_design_tokens(row)
    assert "brown" in m._row_color_tokens(row)
    assert "wood" in m._row_material_tokens(row)
    assert m._same_family("chair", "armchair")
    assert m._target_requires_exact_group("bed")

    sink_bad = m._bathroom_sink_quality_info({"title": "Living room decorative vase", "category_raw": "decor"})
    assert sink_bad["bathroom_sink_quality_reject_reason"] == "bathroom_sink_false_visual_context"
    sink_ok = m._bathroom_sink_quality_info({"title": "wall mounted sink basin", "height_cm": 20})
    assert sink_ok["bathroom_sink_quality_reject_reason"] is None

    fits, fit_info = m._fits_inside_bbox([1, 2, 1], [2, 1, 1])
    assert fits and fit_info["bbox_fit_orientation"] == "swapped_xy"
    soft, soft_info = m._passes_rescalable_fit("bed", [1.6, 2.0, 0.9], [1.7, 2.1, 1.0])
    assert soft and soft_info["passes_rescalable_fit"]
    assert m._size_distance([1, 1, 1], [1, 2, 1]) > 0
    assert m._axis_log_distance(1, 2) > 0
    dim_info = m._dimension_priority_info(good_target(), row)
    assert dim_info["dimension_priority"] == "bed_footprint_first"
    size_rank, _dist, size_breakdown = m._size_match_info(good_target(), row)
    assert size_rank == 0
    axis_info = m._axis_distance_info(good_target(), row, size_breakdown)
    assert axis_info["oriented_candidate_size_m"]
    fill = m._bbox_fill_info(good_target(), row)
    assert fill["passes_min_fill"]


def test_scoring_ranking_llm_and_policy_helpers(monkeypatch, tmp_path):
    asset = tmp_path / "model.glb"
    asset.write_bytes(b"glb")
    target = good_target()
    row = good_row(asset, width_cm=165)
    context = {
        "prompt_tokens": m._normalize_text_tokens("modern brown wood bedroom"),
        "room_style_tokens": m._normalize_text_tokens("modern minimalism"),
        "style_label": "modern",
        "prompt_text": "modern brown wood bedroom",
        "supplier_selection_strategy": "balanced",
        "supplier_selection_mode": "optimal",
        "user_preferences": m._normalize_user_preferences(
            {"preferred_color": "brown", "preferred_brand": "Brand", "max_price": 100000, "allowed_sites": "homeconcept"}
        ),
    }

    assert m._infer_row_group({"category_norm": "tv_projector_screen", "title": "gaming monitor"}) == "computer"
    category_rank, category_info = m._category_match_info(target, row)
    assert category_rank == 0
    assert category_info["category_match"] == "exact_group"
    design_score, design_info = m._design_match_info(target, row)
    assert design_score > 0 and design_info["material_match"]
    style_rank, style_score, style_info = m._style_match_info(target, row, context)
    assert style_rank == 0 and style_score > 100
    prompt_score, prompt_info = m._prompt_match_info(target, row, context)
    assert prompt_score > 0 and prompt_info["color_match"]
    query_score, query_info = m._query_match_info(target, row)
    assert query_score > 0 and query_info["query_overlap_count"] > 0
    prefs_ok, pref_score, pref_info = m._user_preference_match_info(target, row, context)
    assert prefs_ok and pref_score > 0 and pref_info["user_preferences_applied"]
    assert m._source_policy_match_info({"source_site": "3ddd", "availability": "pro"})[0] is False
    assert m._source_policy_match_info({"source_site": "3ddd", "extra_json": '{"api_type": "free"}'})[0] is True

    assert m._extract_ollama_text({"message": {"content": " hi "}}) == "hi"
    assert m._parse_json_object_from_text("```json\n{\"x\": 1}\n```") == {"x": 1}
    payload = m._llm_candidate_payload({**row, "score_breakdown": {"query_score": 1}, "images_json": row["images_json"]})
    assert payload["unique_key"] == "bed-key"
    assert payload["image_count"] == 1
    reranked, report = m._llm_rerank_candidates(
        target=target,
        top_candidates=[{"unique_key": "a"}, {"unique_key": "b"}],
        context=context,
        llm_settings={"provider": "none"},
    )
    assert [c["unique_key"] for c in reranked] == ["a", "b"]
    assert report is None

    fake_module = types.SimpleNamespace(
        chat_json=lambda **_kwargs: {"message": {"content": '{"chosen_unique_key": "b", "ordered_unique_keys": ["b", "a"]}'}}
    )
    monkeypatch.setitem(sys.modules, "src.LLMModule.ollama_client", fake_module)
    reranked, report = m._llm_rerank_candidates(
        target=target,
        top_candidates=[{"unique_key": "a"}, {"unique_key": "b"}],
        context=context,
        llm_settings={"provider": "ollama", "top_n": 2},
    )
    assert [c["unique_key"] for c in reranked] == ["b", "a"]
    assert report["status"] == "applied"

    assert m._candidate_axis_m(row, "width") == 1.65
    assert m._luxury_ceiling_intent({"prompt_text": "classic chandelier"})
    assert m._target_room_type_from_context({"room_design_spec": {"room_type": "bedroom"}}) == "bedroom"
    assert "bed" in m._candidate_title_tokens(row)
    assert m._candidate_oriented_xy_ratio([1, 2, 1], [2, 1, 1])["orientation"] == "swapped_xy"
    assert m._hard_dimension_reject_info(
        {"semantic_group": "lamp_ceiling", "category": "ceiling_light", "size_m": [0.4, 0.4, 0.1]},
        {"title": "Large chandelier", "category_norm": "ceiling_light", "width_cm": 100, "depth_cm": 100, "height_cm": 50},
        {"room_type": "bedroom"},
    )["hard_dimension_reject_reason"] == "rejected_bedroom_chandelier_not_requested"

    monkeypatch.setattr(m, "candidate_identity_gate", lambda _target, _row: (True, {"identity_gate_checked": True, "identity_gate_passed": True}))
    ranked = m._rank_candidate(target, row, context)
    assert ranked is not None
    rank_key, reasons = ranked
    assert isinstance(rank_key, tuple)
    assert reasons["category_match"] == "exact_group"
    assert reasons["has_real_asset"] is True

    candidate = m._candidate_from_scored(rank_key, row, {**reasons, "final_score": 42}, selection_mode="optimal")
    assert candidate["unique_key"] == "bed-key"
    accepted, accept_info = m._candidate_acceptability(target, candidate)
    assert accepted
    assert accept_info["asset_ok"]
    assert m._candidate_has_viable_asset_hint({"asset_format": "max"}) == (False, "max_only_asset")
    assert m._candidate_acceptance_thresholds("bed", False)["min_query_score"] >= 16
    chosen_idx, chosen_candidate, choose_report = m._choose_accepted_candidate_for_mode(
        [(0, {**candidate, "price_value": 500}), (1, {**candidate, "unique_key": "cheap", "price_value": 100})],
        "cheapest",
    )
    assert chosen_idx == 1
    assert chosen_candidate["unique_key"] == "cheap"
    assert choose_report["policy"] == "lowest_price_among_all_suitable"


def test_catalog_json_loading_context_and_build_bindings(monkeypatch, tmp_path):
    asset = tmp_path / "model.glb"
    asset.write_bytes(b"glb")
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps({"items": [good_row(asset, width_cm=165), {**good_row(asset, width_cm=165), "unique_key": "bed-key", "color": ""}]}),
        encoding="utf-8",
    )
    rows = m.load_supplier_catalog_json([catalog_path])
    assert len(rows) == 2
    merged_rows = m._merge_catalog_rows(rows)
    assert len(merged_rows) == 1
    assert merged_rows[0]["semantic_group"] == "bed"

    targets_path = tmp_path / "targets.json"
    targets_path.write_text(json.dumps({"room": {"style_hint": "modern"}, "targets": [good_target()]}), encoding="utf-8")
    (tmp_path / "prompt.txt").write_text("modern brown wooden bedroom", encoding="utf-8")
    context = m._load_matcher_context(
        targets_path,
        {"room": {"style_hint": "modern"}},
        user_preferences={"preferred_color": "brown"},
        selection_strategy="unknown",
        selection_mode="optimal",
    )
    assert context["supplier_selection_strategy"] == "balanced"
    assert "modern" in context["prompt_tokens"]

    assert m._target_is_large_furniture_candidate(good_target())
    assert not m._target_is_large_furniture_candidate({"category": "PillowFactory", "semantic_group": "", "size_m": [0.2, 0.2, 0.1]})

    source_scene = tmp_path / "scene.json"
    source_scene.write_text(
        json.dumps({"placements": [{"id": "bed_1", "color": [0.5, 0.3, 0.2], "constraints": {"brand": "Brand"}}]}),
        encoding="utf-8",
    )
    enriched = m._enrich_targets_from_source_scene({"source_json": str(source_scene)}, [good_target(constraints={})])
    assert enriched[0]["color_rgb"] == [0.5, 0.3, 0.2]
    assert enriched[0]["constraints"]["brand"] == "Brand"

    images = m._candidate_images({"images_json": '[{"url": "u1"}, "u2", "u2"]'})
    assert images == ["u1", "u2"]
    assert m._candidate_dimensions_m(good_row(asset)) == {"width": 1.6, "depth": 2.0, "height": 1.0}
    assert m._candidate_generation_prompt(good_row(asset), good_target()).startswith("Generate a realistic 3D model")
    reference = m._build_generation_reference(good_row(asset), good_target())
    assert reference["has_local_asset"] is True
    diagnostics = m._selection_diagnostics({"score_breakdown": {"size_score": 1}, **good_row(asset)}, "optimal")
    assert diagnostics["has_local_asset"] is True

    monkeypatch.setattr(m, "candidate_identity_gate", lambda _target, _row: (True, {"identity_gate_checked": True, "identity_gate_passed": True}))
    bindings = m.build_bindings_with_candidates(
        targets_json_path=targets_path,
        catalog_rows=merged_rows,
        top_k=3,
        selection_strategy="balanced",
        selection_mode="optimal",
    )
    assert bindings["meta"]["matched_target_count"] == 1
    binding = bindings["bindings"][0]
    assert binding["selection_status"] == "heuristic_top1_selected"
    assert binding["chosen_candidate"]["unique_key"] == "bed-key"


def test_sqlite_catalog_loading_build_cli_main_and_no_candidate_paths(monkeypatch, tmp_path, capsys):
    product_db = tmp_path / "products.db"
    with sqlite3.connect(product_db) as con:
        con.execute(
            """
            CREATE TABLE supplier_product (
                unique_key TEXT, source_site TEXT, source_url TEXT, parsed_at TEXT,
                external_id TEXT, category_raw TEXT, category_norm TEXT, title TEXT,
                brand TEXT, collection TEXT, product_url TEXT, model_link_type TEXT,
                model_download_url TEXT, model_download_landing_url TEXT, model_vendor_url TEXT,
                model_extraction_method TEXT, model_download_filename TEXT, model_format TEXT,
                model_page_url TEXT, price_value REAL, price_currency TEXT, style TEXT,
                color TEXT, description TEXT, width_cm REAL, depth_cm REAL, height_cm REAL,
                materials TEXT, room TEXT, availability TEXT, images_json TEXT, extra_json TEXT
            )
            """
        )
        con.execute(
            """
            CREATE TABLE supplier_asset (
                unique_key TEXT, asset_status TEXT, asset_format TEXT, asset_local_path TEXT
            )
            """
        )
        con.execute(
            "INSERT INTO supplier_product VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "chair-db",
                "homeconcept",
                "source",
                "now",
                "1",
                "chair",
                "chair",
                "Modern chair",
                "Brand",
                "Line",
                "url",
                "",
                "",
                "",
                "",
                "",
                "",
                "glb",
                "",
                1000,
                "RUB",
                "modern",
                "brown",
                "wood chair",
                50,
                60,
                90,
                "wood",
                "bedroom",
                "free",
                json.dumps(["img.jpg"]),
                "{}",
            ),
        )
        con.execute("INSERT INTO supplier_asset VALUES (?,?,?,?)", ("chair-db", "ready", "glb", str(tmp_path / "chair.glb")))

    rows = m.load_supplier_catalog([product_db], sites={"homeconcept"}, rich_only=False)
    assert len(rows) == 1
    assert rows[0]["asset_format"] == "glb"
    assert rows[0]["semantic_group"] == "chair"
    assert m.load_supplier_catalog([product_db], sites={"other"}, rich_only=False) == []

    mesh_db = tmp_path / "mesh.db"
    with sqlite3.connect(mesh_db) as con:
        con.execute(
            """
            CREATE TABLE supplier_mesh_catalog (
                unique_key TEXT, source_site TEXT, source_url TEXT, parsed_at TEXT,
                external_id TEXT, category_raw TEXT, category_norm TEXT, title TEXT,
                brand TEXT, collection TEXT, product_url TEXT, model_link_type TEXT,
                model_download_url TEXT, model_download_landing_url TEXT, model_vendor_url TEXT,
                model_extraction_method TEXT, model_download_filename TEXT, model_format TEXT,
                model_page_url TEXT, price_value REAL, price_currency TEXT, style TEXT,
                color TEXT, description TEXT, width_cm REAL, depth_cm REAL, height_cm REAL,
                materials TEXT, room TEXT, availability TEXT, images_json TEXT, extra_json TEXT,
                mesh_status TEXT, mesh_format TEXT, mesh_local_path TEXT, semantic_group TEXT
            )
            """
        )
        con.execute(
            "INSERT INTO supplier_mesh_catalog VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "mesh-bed",
                "loft",
                "source",
                "now",
                "2",
                "bed",
                "bed",
                "Modern bed",
                "Brand",
                "Line",
                "url",
                "",
                "",
                "",
                "",
                "",
                "",
                "fbx",
                "",
                2000,
                "RUB",
                "modern",
                "brown",
                "wood bed",
                160,
                200,
                100,
                "wood",
                "bedroom",
                "free",
                json.dumps(["bed.jpg"]),
                "{}",
                "ready",
                "fbx",
                str(tmp_path / "bed.fbx"),
                "",
            ),
        )
    mesh_rows = m.load_supplier_catalog([mesh_db], rich_only=False)
    assert mesh_rows[0]["unique_key"] == "mesh-bed"
    assert mesh_rows[0]["semantic_group"] == "bed"

    parser = m.build_cli()
    parsed = parser.parse_args(
        [
            "--targets",
            "targets.json",
            "--supplier-json",
            "catalog.json",
            "--site",
            "homeconcept",
            "--top-k",
            "7",
            "--selection-mode",
            "cheapest_top20",
            "--preferred-color",
            "brown",
            "--avoid-color",
            "red",
            "--require-real-asset",
            "--llm-provider",
            "none",
            "--out",
            "bindings.json",
        ]
    )
    assert parsed.top_k == 7
    assert parsed.selection_mode == "cheapest_top20"
    assert parsed.require_real_asset is True

    target_keep = good_target(target_id="keep", replacement_policy="keep_generated")
    target_small = good_target(target_id="small", semantic_group="pillow", category="pillow", size_m=[0.2, 0.2, 0.1])
    target_no_candidate = good_target(target_id="nope", semantic_group="bed", category="bed")
    targets = tmp_path / "targets_main.json"
    targets.write_text(json.dumps({"targets": [target_keep, target_small, target_no_candidate]}), encoding="utf-8")
    no_candidate = m.build_bindings_with_candidates(
        targets_json_path=targets,
        catalog_rows=[],
        top_k=2,
        selection_mode="optimal",
    )
    statuses = [b["selection_status"] for b in no_candidate["bindings"]]
    assert statuses == ["kept_generated_stub", "kept_generated_stub", "no_candidates_found"]

    asset = tmp_path / "bed.glb"
    asset.write_bytes(b"glb")
    catalog_json = tmp_path / "catalog_main.json"
    catalog_json.write_text(json.dumps({"items": [good_row(asset)]}), encoding="utf-8")
    out = tmp_path / "bindings_out.json"
    monkeypatch.setattr(m, "candidate_identity_gate", lambda _target, _row: (True, {"identity_gate_checked": True, "identity_gate_passed": True}))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "supplier_layout_matcher",
            "--targets",
            str(targets),
            "--supplier-json",
            str(catalog_json),
            "--top-k",
            "3",
            "--selection-mode",
            "optimal",
            "--preferred-color",
            "brown",
            "--out",
            str(out),
        ],
    )
    m.main()
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["schema"] == "supplier_bindings/v1"
    assert "saved =" in capsys.readouterr().out


def test_matcher_edge_reject_branches_and_color_dimension_variants(tmp_path):
    assert m._value_to_text_list({"not": "a list"}) == ["{'not': 'a list'}"]
    assert m._value_to_text_list("") == []
    assert m._dimension_value_to_cm("bad", "cm") is None
    assert m._product_size_m({"width_cm": "bad", "depth_cm": 1, "height_cm": 2}) is None
    assert m._effective_target_size_m({"size_m": ["bad"]}) == [1e-6, 1e-6, 1e-6]

    length_row = {"title": "console length 120 cm width 40 cm height 75 cm"}
    inferred = m._infer_dimensions_cm_from_text(length_row)
    assert inferred["depth"] == 120.0
    triple = m._infer_dimensions_cm_from_text({"title": "size 120x40x75 cm"})
    assert triple == {"width": 120.0, "depth": 40.0, "height": 75.0}

    colors = {
        "white": [0.95, 0.95, 0.95],
        "gray": [0.5, 0.5, 0.5],
        "red": [1.0, 0.0, 0.0],
        "orange": [1.0, 0.35, 0.0],
        "yellow": [1.0, 0.95, 0.0],
        "blue": [0.0, 0.0, 1.0],
        "purple": [0.55, 0.0, 0.85],
        "beige": [0.95, 0.86, 0.65],
    }
    for expected, rgb in colors.items():
        assert expected in m._rgb_to_basic_color_tokens(rgb)
    assert m._rgb_to_basic_color_tokens(["x", 1, 2]) == set()
    assert m._rgb_to_basic_color_tokens([0.7, 0.7, 0.7]) == set()
    row_colors = m._row_image_color_tokens(
        {"image_color_features": {"colors": {"top5": [{"basic_color": "olive"}, "bad", {"rgb": [0.0, 0.0, 1.0]}]}}}
    )
    assert {"green", "blue"} <= row_colors

    assert not m._candidate_has_ready_real_asset({"asset_local_path": str(tmp_path / "missing.glb"), "asset_format": "glb", "asset_status": "ready"})
    assert m._candidate_has_downloadable_asset({"model_download_url": "https://yadi.sk/d/model"})
    assert m._candidate_has_downloadable_asset({"mesh_source_url": "https://x.test/model.obj?download=1"})
    assert not m._candidate_has_downloadable_asset({"model_download_url": "https://drive.google.com/file", "model_download_filename": "view"})

    missing_sink = m._bathroom_sink_quality_info({"title": "wall mounted cabinet", "category_raw": "bathroom"})
    assert missing_sink["bathroom_sink_quality_reject_reason"] == "bathroom_sink_missing_explicit_sink_terms"
    tall_sink = m._bathroom_sink_quality_info({"title": "bathroom sink", "category_raw": "bathroom", "width_cm": 50, "depth_cm": 40, "height_cm": 95})
    assert tall_sink["bathroom_sink_quality_reject_reason"] == "bathroom_sink_standalone_tall_not_wall_or_countertop"

    assert m._fits_inside_bbox([1, 1, 1], []) == (False, {"fit_checked": False, "fits_bbox": False})
    assert m._passes_rescalable_fit("decor", [1, 1, 1], [1, 1, 1])[1]["rescalable_fit_checked"] is False
    assert m._dimension_priority_info(good_target(semantic_group="desk"), good_row(width_cm="", depth_cm="", height_cm=""))["dimension_priority"] == "missing_candidate_size"
    assert m._dimension_priority_info(good_target(semantic_group="desk"), good_row(width_cm=120, depth_cm=60, height_cm=75))["dimension_priority"] == "support_surface_height_first"
    assert m._dimension_priority_info(good_target(semantic_group="chair"), good_row(width_cm=45, depth_cm=50, height_cm=80))["dimension_priority"] == "overall_size_first"
    assert m._axis_distance_info(good_target(), {"width_cm": "", "depth_cm": "", "height_cm": ""}, {})["oriented_candidate_size_m"] is None
    assert m._axis_distance_info(good_target(), good_row(width_cm=200, depth_cm=160, height_cm=100), {"bbox_fit_orientation": "swapped_xy"})["oriented_candidate_size_m"][:2] == [1.6, 2.0]
    assert m._bbox_fill_info(good_target(semantic_group="decor"), good_row(width_cm=10, depth_cm=10, height_cm=10))["passes_min_fill"] is True
    assert m._bbox_fill_info(good_target(), {"width_cm": "", "depth_cm": "", "height_cm": ""})["passes_min_fill"] is False

    assert m._infer_row_group({"title": "luxury chandelier", "category_raw": "ceiling lamp"}) == "lamp_ceiling"
    assert m._infer_row_group({"title": "floor lamp", "category_raw": "lighting"}) == "lamp_floor"
    assert m._infer_row_group({"title": "small lamp", "category_raw": "lighting"}) == "lamp_table"
    same_rank, same_info = m._category_match_info(good_target(semantic_group="chair"), {"semantic_group": "armchair"})
    assert same_rank == 1 and same_info["category_match"] == "same_family"
    mismatch_rank, mismatch_info = m._category_match_info(good_target(semantic_group="bed"), {"semantic_group": "chair"})
    assert mismatch_rank == 3 and mismatch_info["category_match"] == "exact_group_required_mismatch"
    size_rank, _dist, size_info = m._size_match_info(good_target(), {"semantic_group": "bed"})
    assert size_rank == 2 and size_info["candidate_size_m"] is None

    no_style_rank, _score, no_style_info = m._style_match_info(
        {"semantic_group": "chair", "category": "chair", "size_m": [0.5, 0.6, 0.9]},
        good_row(style_llm="", style=""),
        {"style_label": ""},
    )
    assert no_style_rank == 2 and no_style_info["style_selection_match"] == "no_target_style"
    low_style_rank, _score, low_style_info = m._style_match_info(
        good_target(),
        good_row(style_llm="baroque", style_llm_confidence=0.1, style_llm_quality_score=1),
        {"style_label": "modern"},
    )
    assert low_style_rank == 2 and low_style_info["style_selection_match"] == "candidate_style_unknown_or_low_quality"
    compat_rank, _score, compat_info = m._style_match_info(good_target(), good_row(style_llm="minimalism"), {"style_label": "modern"})
    assert compat_rank == 1 and compat_info["style_selection_match"] == "compatible_style"
    mismatch_rank, _score, mismatch_info = m._style_match_info(
        good_target(constraints={"style": "industrial"}),
        good_row(style_llm="classic", style="", style_llm_secondary=[], style_llm_confidence=0.9, style_llm_quality_score=8),
        {"style_label": "industrial"},
    )
    assert mismatch_rank == 3 and mismatch_info["style_selection_match"] == "style_mismatch"

    reject_cases = [
        ({"allowed_sites": "other"}, "site_not_allowed_by_user_preferences"),
        ({"disallowed_sites": "homeconcept"}, "site_explicitly_disallowed_by_user_preferences"),
        ({"max_price": 1}, "price_above_user_max_price"),
        ({"avoid_colors": "brown"}, "color_explicitly_avoided_by_user_preferences"),
        ({"preferred_color": "white", "strict_color": True}, "strict_color_requested_but_candidate_color_mismatch"),
        ({"require_real_asset": True}, "real_asset_required_by_user_preferences"),
        ({"require_model_url": True}, "model_url_required_by_user_preferences"),
    ]
    for prefs, expected_reason in reject_cases:
        ok, _bonus, info = m._user_preference_match_info(good_target(), good_row(asset_local_path="", asset_format=""), {"user_preferences": m._normalize_user_preferences(prefs)})
        assert not ok
        assert info["user_preference_reject_reason"] == expected_reason

    assert m._row_extra_dict({"extra": {"x": 1}}) == {"x": 1}
    assert m._three_ddd_access_type({"availability": "free"}) == "free"
    assert m._extract_ollama_text({"response": " ok "}) == "ok"
    assert m._extract_ollama_text({"x": 1}) == '{"x": 1}'
    assert m._parse_json_object_from_text("prefix {\"x\": 2} suffix") == {"x": 2}
    with pytest.raises(RuntimeError, match="LLM did not return JSON"):
        m._parse_json_object_from_text("not json")


def test_sidecar_asset_acceptability_and_llm_rerank_edges(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sidecar_dir = tmp_path / "reports" / "supplier_image_colors"
    sidecar_dir.mkdir(parents=True)
    sidecar = sidecar_dir / "supplier_catalog_canonical.image_colors.jsonl"
    sidecar.write_text(
        "\n".join(
            [
                "",
                json.dumps({"status": "bad", "unique_key": "skip"}),
                json.dumps({"status": "ok", "unique_key": "", "image": "none"}),
                json.dumps({"status": "ok", "unique_key": "u1", "image": "img.jpg", "foreground_ratio": 0.5, "colors": {"top5": []}, "color_tokens": ["brown"], "method": "mock"}),
            ]
        ),
        encoding="utf-8",
    )
    assert m._load_image_color_feature_sidecar(Path("other.json")) == {}
    features = m._load_image_color_feature_sidecar(Path("supplier_catalog_canonical.json"))
    assert features["u1"]["color_tokens"] == ["brown"]

    assert m._candidate_has_viable_asset_hint(good_row(asset_status="needs_blender_rebuild", asset_format="", model_format="", images_json="[]")) == (False, "asset_status:needs_blender_rebuild")
    assert m._candidate_has_viable_asset_hint(good_row(asset_format="max", model_format="", images_json="[]")) == (False, "max_only_asset")
    assert m._candidate_has_viable_asset_hint(good_row(asset_format="fbx", asset_local_path="")) == (True, "local_asset:fbx")
    assert m._candidate_has_viable_asset_hint(good_row(asset_format="", model_format="rar", images_json="[]")) == (True, "downloadable_asset:rar")
    assert m._candidate_has_viable_asset_hint(good_row(asset_format="", model_format="", preview_local_path="preview.jpg", images_json="[]")) == (True, "preview_image_asset_reference")
    assert m._candidate_has_viable_asset_hint(good_row(asset_format="", model_format="", preview_local_path="", images=["a.jpg"], images_json="[]")) == (True, "product_image_asset_reference")
    assert m._candidate_has_viable_asset_hint(good_row(asset_format="", model_format="", preview_local_path="", images=[], images_json='["a.jpg"]')) == (True, "product_image_asset_reference")

    target = good_target(semantic_group="chair", category="chair", size_m=[0.5, 0.6, 0.9])
    base_breakdown = {
        "size_missing": False,
        "passes_min_fill": True,
        "category_match": "exact_group",
        "primary_axis_distance": 0.01,
        "secondary_axis_distance": 0.01,
        "query_score": 100.0,
        "query_overlap_count": 3,
    }
    assert m._candidate_acceptability(target, {"score_breakdown": dict(base_breakdown), "asset_format": "fbx"})[0]
    failed_identity = dict(base_breakdown, identity_gate_checked=True, identity_gate_passed=False)
    assert m._candidate_acceptability(target, {"score_breakdown": failed_identity, "asset_format": "fbx"})[1]["reject_reason"] == "identity_gate_failed"
    assert m._candidate_acceptability(target, {"score_breakdown": dict(base_breakdown, passes_min_fill=False), "asset_format": "fbx"})[1]["reject_reason"] == "fails_min_fill"
    assert m._candidate_acceptability(target, {"score_breakdown": dict(base_breakdown, category_match="mismatch"), "asset_format": "fbx"})[1]["reject_reason"].startswith("category_match")
    assert m._candidate_acceptability(target, {"score_breakdown": dict(base_breakdown, primary_axis_distance=99), "asset_format": "fbx"})[1]["reject_reason"] == "primary_axis_distance_too_large"
    missing_breakdown = {
        "size_missing": True,
        "category_match": "same_family",
        "relaxed_missing_size_match": False,
        "query_overlap_count": 0,
        "query_score": 0,
    }
    assert m._candidate_acceptability(target, {"score_breakdown": missing_breakdown, "asset_format": "fbx"})[1]["reject_reason"].startswith("missing_size_requires_exact_group")
    missing_breakdown["category_match"] = "exact_group"
    assert m._candidate_acceptability(target, {"score_breakdown": missing_breakdown, "asset_format": "fbx"})[1]["reject_reason"] == "missing_size_without_relaxed_match"
    missing_breakdown["relaxed_missing_size_match"] = True
    assert m._candidate_acceptability(target, {"score_breakdown": missing_breakdown, "asset_format": "fbx"})[1]["reject_reason"] == "missing_size_without_query_overlap"

    unsupported, info = m._llm_rerank_candidates(target=target, top_candidates=[good_row(unique_key="a"), good_row(unique_key="b")], context={}, llm_settings={"provider": "openai"})
    assert unsupported[0]["unique_key"] == "a"
    assert info["reason"] == "unsupported_provider:openai"

    fake_module = types.ModuleType("src.LLMModule.ollama_client")
    fake_module.chat_json = lambda **_kwargs: {"response": json.dumps({"chosen_unique_key": "Second", "ordered_unique_keys": ["Second", "a"], "reason": "visual"})}
    monkeypatch.setitem(sys.modules, "src.LLMModule.ollama_client", fake_module)
    ranked, info = m._llm_rerank_candidates(
        target=target,
        top_candidates=[good_row(unique_key="a", title="First model"), good_row(unique_key="b model", title="Second model")],
        context={},
        llm_settings={"provider": "ollama", "top_n": 2, "ollama_model": "mock"},
    )
    assert info["status"] == "applied"
    assert ranked[0]["unique_key"] == "b model"

    fake_module.chat_json = lambda **_kwargs: {"response": json.dumps({"chosen_unique_key": "unknown"})}
    _ranked, info = m._llm_rerank_candidates(target=target, top_candidates=[good_row(unique_key="a"), good_row(unique_key="b")], context={}, llm_settings={"provider": "ollama"})
    assert info["reason"] == "ollama_returned_unknown_candidate"


def test_ranker_reject_modes_catalog_json_and_price_selection_edges(tmp_path, monkeypatch):
    asset = tmp_path / "asset.glb"
    asset.write_bytes(b"glb")
    monkeypatch.setattr(m, "candidate_identity_gate", lambda _target, _row: (True, {"identity_gate_checked": True, "identity_gate_passed": True}))

    light_target = good_target(
        target_id="light",
        category="ceiling_light",
        semantic_group="lamp_ceiling",
        size_m=[0.3, 0.3, 0.08],
    )
    bedroom_ctx = {"room_design_spec": {"room_type": "bedroom"}, "prompt_text": ""}
    oversized_light = good_row(
        asset,
        unique_key="large-light",
        title="large flush ceiling lamp",
        semantic_group="lamp_ceiling",
        category_norm="ceiling_light",
        width_cm=120,
        depth_cm=110,
        height_cm=20,
    )
    assert m._hard_dimension_reject_info(light_target, oversized_light, bedroom_ctx)["hard_dimension_reject_reason"] == "rejected_oversized_for_target_aabb"
    taller_light_target = {**light_target, "size_m": [0.65, 0.65, 0.2]}
    missing_height_light = {**oversized_light, "width_cm": 130, "depth_cm": 130, "height_cm": ""}
    assert m._hard_dimension_reject_info(taller_light_target, missing_height_light, bedroom_ctx)["hard_dimension_reject_reason"] == "rejected_missing_height_for_oversized_light"
    tiny_target = good_target(target_id="tiny", semantic_group="decor", category="decor", size_m=[0.35, 0.35, 0.1])
    huge_decor = good_row(asset, unique_key="huge", semantic_group="decor", category_norm="decor", width_cm=82, depth_cm=82, height_cm=10)
    assert m._hard_dimension_reject_info(tiny_target, huge_decor, {})["hard_dimension_reject_reason"] == "rejected_very_small_target_large_candidate"
    assert m._hard_dimension_reject_info(good_target(semantic_group="bed"), good_row(asset, semantic_group="chair", category_norm="chair"), {}) is None
    assert m._bedroom_ceiling_light_reject_reason(light_target, oversized_light, bedroom_ctx) == "rejected_oversized_for_target_aabb"

    rank_target = good_target(target_id="desk", semantic_group="desk", category="desk", size_m=[1.2, 0.6, 0.75])
    rank_row = good_row(
        asset,
        unique_key="desk",
        semantic_group="desk",
        category_norm="desk",
        title="modern brown wood desk",
        width_cm=120,
        depth_cm=60,
        height_cm=75,
        price_value=1100,
    )
    for strategy in ("cheapest", "cheap_style", "style", "balanced"):
        ranked = m._rank_candidate(rank_target, rank_row, {"supplier_selection_strategy": strategy, "prompt_text": "modern brown wood desk"})
        assert ranked is not None
        assert ranked[1]["supplier_selection_strategy"] == strategy
    ranked_budget = m._rank_candidate(
        good_target(target_id="desk", semantic_group="desk", category="desk", size_m=[1.2, 0.6, 0.75], constraints={"budget_rub": "bad"}),
        rank_row,
        {"supplier_selection_strategy": "balanced"},
    )
    assert ranked_budget is not None and ranked_budget[1]["price_distance_ratio"] is None

    assert m._rank_candidate(rank_target, good_row(asset, source_site="3ddd", availability="pro"), {}) is None
    monkeypatch.setattr(m, "candidate_identity_gate", lambda _target, _row: (False, {"identity_gate_checked": True, "identity_gate_passed": False}))
    assert m._rank_candidate(rank_target, rank_row, {}) is None
    monkeypatch.setattr(m, "candidate_identity_gate", lambda _target, _row: (True, {"identity_gate_checked": True, "identity_gate_passed": True}))
    assert m._rank_candidate(rank_target, good_row(asset, semantic_group="bed", category_norm="bed"), {}) is None
    sink_bad = good_row(asset_local_path="", asset_format="", semantic_group="bathroom_sink", category_norm="sink")
    assert m._rank_candidate(good_target(semantic_group="bathroom_sink", category="sink"), sink_bad, {}) is None

    relaxed_target = good_target(target_id="plant", semantic_group="plant_planter_vase", category="decor", size_m=[0.25, 0.25, 0.5])
    relaxed_row = good_row(
        asset_local_path="",
        asset_format="",
        model_format="rar",
        model_download_url="https://example.test/model.rar",
        unique_key="plant",
        semantic_group="plant_planter_vase",
        category_norm="decor",
        width_cm="",
        depth_cm="",
        height_cm="",
        title="decor plant vase",
        search_text_en="decor plant vase",
    )
    relaxed_ranked = m._rank_candidate(relaxed_target, relaxed_row, {"prompt_text": "decor plant vase"})
    assert relaxed_ranked is not None
    assert relaxed_ranked[1]["relaxed_missing_size_match"] is True

    base_breakdown = {
        "size_missing": False,
        "passes_min_fill": True,
        "category_match": "exact_group",
        "primary_axis_distance": 0.01,
        "secondary_axis_distance": 0.01,
        "query_score": 100.0,
        "query_overlap_count": 3,
    }
    assert m._candidate_has_viable_asset_hint({"model_download_url": "https://example.test/model.zip", "model_download_filename": "model.zip"}) == (True, "downloadable_asset")
    assert m._candidate_has_viable_asset_hint({"images_json": "[]"}) == (False, "no_viable_asset")
    assert m._candidate_acceptability(good_target(semantic_group="bathroom_sink"), {"score_breakdown": base_breakdown})[1]["reject_reason"].startswith("bathroom_sink_requires_viable_asset")
    assert m._candidate_acceptability(rank_target, {"score_breakdown": dict(base_breakdown, secondary_axis_distance=99), "asset_format": "fbx"})[1]["reject_reason"] == "secondary_axis_distance_too_large"
    assert m._candidate_acceptability(rank_target, {"score_breakdown": dict(base_breakdown, query_score=0), "asset_format": "fbx"})[1]["reject_reason"] == "query_match_too_weak"
    missing_low_query = {
        "size_missing": True,
        "category_match": "exact_group",
        "relaxed_missing_size_match": True,
        "query_overlap_count": 10,
        "query_score": 0,
    }
    assert m._candidate_acceptability(rank_target, {"score_breakdown": missing_low_query, "asset_format": "fbx"})[1]["reject_reason"] == "missing_size_query_match_too_weak"

    assert m._candidate_images({"images": ["u1", {"url": "u2"}, 123]}) == ["u1", "u2"]
    assert m._candidate_dimensions_m({"width_cm": "50", "depth_cm": "", "height_cm": "70"}) == {"width": 0.5, "depth": None, "height": 0.7}
    assert m._candidate_price_number({"price_value": ""}) is None
    with pytest.raises(ValueError):
        m._choose_accepted_candidate_for_mode([], "optimal")
    cheap_candidates = [
        (21, {"unique_key": "later", "price_value": 1, "final_score": 1}),
        (22, {"unique_key": "fallback", "price_value": 2, "final_score": 99}),
    ]
    chosen_idx, chosen, policy = m._choose_accepted_candidate_for_mode(cheap_candidates, "cheapest_top20")
    assert chosen_idx == 21 and chosen["unique_key"] == "later"
    assert policy["policy"] == "lowest_price_among_top20_suitable"

    single_catalog = tmp_path / "single_catalog.json"
    single_catalog.write_text(json.dumps(good_row(asset, unique_key="single")), encoding="utf-8")
    assert m.load_supplier_catalog_json([single_catalog])[0]["unique_key"] == "single"
    bad_bindings = tmp_path / "bindings.json"
    bad_bindings.write_text(json.dumps({"bindings": []}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="bindings"):
        m.load_supplier_catalog_json([bad_bindings])
    bad_catalog = tmp_path / "bad_catalog.json"
    bad_catalog.write_text(json.dumps({"items": [123, good_row(asset, unique_key="", color=""), {**good_row(asset, unique_key="filtered"), "source_site": "other"}]}), encoding="utf-8")
    filtered_rows = m.load_supplier_catalog_json([bad_catalog], sites={"homeconcept"})
    assert len(filtered_rows) == 1 and filtered_rows[0]["unique_key"] == ""
    inferred_catalog = tmp_path / "inferred_catalog.json"
    inferred_catalog.write_text(json.dumps({"items": [{**good_row(asset, width_cm=None, depth_cm=None, height_cm=None), "title": "size 120x40x75 cm"}]}), encoding="utf-8")
    inferred_row = m.load_supplier_catalog_json([inferred_catalog])[0]
    assert inferred_row["dimensions_inferred_from_text"]["width"] == 120.0

    product_db = tmp_path / "products_no_asset.db"
    with sqlite3.connect(product_db) as con:
        con.execute(
            """
            CREATE TABLE supplier_product (
                unique_key TEXT, source_site TEXT, source_url TEXT, parsed_at TEXT,
                external_id TEXT, category_raw TEXT, category_norm TEXT, title TEXT,
                brand TEXT, collection TEXT, product_url TEXT, model_link_type TEXT,
                model_download_url TEXT, model_download_landing_url TEXT, model_vendor_url TEXT,
                model_extraction_method TEXT, model_download_filename TEXT, model_format TEXT,
                model_page_url TEXT, price_value REAL, price_currency TEXT, style TEXT,
                color TEXT, description TEXT, width_cm REAL, depth_cm REAL, height_cm REAL,
                materials TEXT, room TEXT, availability TEXT, images_json TEXT, extra_json TEXT
            )
            """
        )
        con.execute(
            "INSERT INTO supplier_product VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("plain", "homeconcept", "src", "now", "1", "chair", "chair", "", "Brand", "Line", "url", "", "", "", "", "", "", "glb", "", 10, "RUB", "modern", "brown", "desc", 50, 60, 90, "wood", "bedroom", "free", "[]", "{}"),
        )
        con.execute(
            "INSERT INTO supplier_product VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("plain2", "homeconcept", "src", "now", "2", "chair", "chair", "Plain chair", "Brand", "Line", "url", "", "", "", "", "", "", "glb", "", 10, "RUB", "modern", "brown", "desc", 50, 60, 90, "wood", "bedroom", "free", "[]", "{}"),
        )
    assert [row["unique_key"] for row in m.load_supplier_catalog([product_db], rich_only=False)] == ["plain2"]

    style_profile = tmp_path / "style_profile.json"
    style_profile.write_text(json.dumps({"style_label": "modern", "description": "modern calm"}), encoding="utf-8")
    targets_path = tmp_path / "bad_targets.json"
    targets_path.write_text(json.dumps({"targets": {"bad": True}}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="layout_targets"):
        m.build_bindings_with_candidates(targets_json_path=targets_path, catalog_rows=[], top_k=1)

    missing_source = m._enrich_targets_from_source_scene({"source_json": str(tmp_path / "missing.json")}, [good_target()])
    assert missing_source[0]["target_id"] == "bed_1"
    assert m._enrich_targets_from_source_scene({"source_json": ""}, [good_target()])[0]["target_id"] == "bed_1"
    assert not m._target_is_large_furniture_candidate({"semantic_group": "unknown", "size_m": [9, 9, 9]})
    assert m._target_is_large_furniture_candidate({"semantic_group": "lamp_table", "size_m": ["bad"], "force_supplier_replacement": True})


def test_build_bindings_design_price_and_rejection_paths(tmp_path, monkeypatch):
    asset = tmp_path / "asset.glb"
    asset.write_bytes(b"glb")
    monkeypatch.setattr(m, "candidate_identity_gate", lambda _target, _row: (True, {"identity_gate_checked": True, "identity_gate_passed": True}))
    monkeypatch.setattr(m, "build_price_stats", lambda _rows: {"RUB": {"min": 1, "max": 100}})
    monkeypatch.setattr(
        m,
        "rank_candidate_for_mode",
        lambda **kwargs: (
            50.0 if kwargs["row"].get("unique_key") == "expensive" else 40.0,
            {"gate_passed": True, "candidate_score_hard_reject_reason": None, "mode_score": 1.0},
        ),
    )
    monkeypatch.setattr(
        m,
        "_candidate_acceptability",
        lambda _target, _candidate: (True, {"accepted": True, "reject_reason": None, "asset_ok": True}),
    )
    targets_path = tmp_path / "targets.json"
    targets_path.write_text(
        json.dumps(
            {
                "targets": [
                    123,
                    good_target(target_id="keep", replacement_policy="keep_generated"),
                    good_target(target_id="small", semantic_group="unknown", category="decor", size_m=[0.1, 0.1, 0.1]),
                    good_target(target_id="sink", semantic_group="bathroom_sink", category="sink", size_m=[0.5, 0.45, 0.2]),
                    good_target(target_id="bed"),
                ]
            }
        ),
        encoding="utf-8",
    )
    rows = [
        good_row(asset, unique_key="blocked", source_site="3ddd", availability="pro"),
        good_row(asset, unique_key="wrong", semantic_group="bed", category_norm="bed"),
        good_row(asset, unique_key="expensive", price_value=5000),
        good_row(asset, unique_key="cheap", price_value=100),
    ]
    bindings = m.build_bindings_with_candidates(
        targets_json_path=targets_path,
        catalog_rows=rows,
        top_k=1,
        selection_mode="cheapest_top20",
        room_design_spec={"room_type": "bedroom"},
    )
    by_id = {binding["target_id"]: binding for binding in bindings["bindings"]}
    assert by_id["keep"]["selection_status"] == "kept_generated_stub"
    assert by_id["small"]["selection_notes"] == ["kept_generated_small_or_nonfurniture_target"]
    assert by_id["sink"]["candidate_count"] == 0
    assert by_id["bed"]["chosen_candidate"]["unique_key"] == "cheap"
    assert any("selected_by_price_policy_not_top1" in note for note in by_id["bed"]["selection_notes"])
    assert bindings["meta"]["final_selection_policy"] == "lowest_price_among_top20_suitable_after_gates"

    monkeypatch.setattr(
        m,
        "rank_candidate_for_mode",
        lambda **_kwargs: (0.0, {"gate_passed": False, "candidate_score_hard_reject_reason": "design_gate_failed"}),
    )
    rejected = m.build_bindings_with_candidates(
        targets_json_path=targets_path,
        catalog_rows=rows[-2:],
        top_k=2,
        selection_mode="best_visual_reference",
        room_design_spec={"room_type": "bedroom"},
    )
    bed_binding = {binding["target_id"]: binding for binding in rejected["bindings"]}["bed"]
    assert bed_binding["selection_status"] == "no_candidates_found"
    assert bed_binding["rejection_summary"]["design_gate_failed"] == 2


def test_remaining_supplier_matcher_helper_edges(tmp_path, monkeypatch):
    asset = tmp_path / "asset.glb"
    asset.write_bytes(b"glb")
    proxy = tmp_path / "built" / "proxy.glb"
    proxy.parent.mkdir()
    proxy.write_bytes(b"proxy")
    unsupported = tmp_path / "asset.txt"
    unsupported.write_text("txt", encoding="utf-8")

    original_row_dimension_cm = m._row_dimension_cm
    monkeypatch.setattr(m, "_row_dimension_cm", lambda _row, _axis: object())
    assert m._product_size_m({}) is None
    monkeypatch.setattr(m, "_row_dimension_cm", original_row_dimension_cm)

    assert m._has_category({"category_raw_en": "chair"})
    assert m._has_category({"category_raw_ru": "стул"})
    assert m._has_category({"category_norm": "chair"})
    rich_base = {
        "title": "t",
        "price_value": 1,
        "width_cm": 10,
        "depth_cm": 10,
        "height_cm": 10,
        "category_raw": "chair",
        "brand": "b",
    }
    assert m._row_is_rich({**rich_base, "description_short_en": "short"})
    assert m._row_is_rich({**rich_base, "description_short_ru": "коротко"})
    assert m._row_is_rich({**rich_base, "vlm_description_text": "visual"})

    assert not m._candidate_has_ready_real_asset(good_row(proxy, asset_local_path=str(proxy)))
    assert not m._candidate_has_ready_real_asset(good_row(unsupported, asset_local_path=str(unsupported), asset_format="txt"))
    assert not m._candidate_has_ready_real_asset(good_row(asset, asset_status="needs_blender_rebuild"))

    assert m._rgb_to_basic_color_tokens("bad") == set()
    assert m._rgb_to_basic_color_tokens([object(), 0, 0]) == set()
    assert "red" in m._rgb_to_basic_color_tokens([1.0, 0.0, 0.0])

    assert m._infer_row_group({"title": "compact chair", "category_norm": "", "category_raw": ""}) == "chair"
    assert m._infer_row_group({"title": "mystery object", "category_norm": "", "category_raw": ""}) == "mystery object"

    mismatch_rank, mismatch_info = m._category_match_info(
        good_target(semantic_group="bed", category="bed"),
        good_row(asset, semantic_group="lamp_ceiling", category_norm="ceiling_light"),
    )
    assert mismatch_rank == 3
    assert mismatch_info["category_match"] == "exact_group_required_mismatch"

    score, design_info = m._design_match_info(
        good_target(constraints={"brand": "Brand", "style": "modern", "material": "wood", "color": "brown"}),
        good_row(asset),
    )
    assert score > 0
    assert design_info["brand_match"]

    assert m._row_style_llm_info(good_row(asset, style_llm_secondary=json.dumps("minimalism")))["secondary"] == ["minimalism"]


def test_remaining_supplier_matcher_llm_rerank_edges(monkeypatch):
    candidates = [
        good_row(unique_key="chair model", title="Chair A"),
        good_row(unique_key="table", title="Table B"),
        good_row(unique_key="tail", title="Tail C"),
    ]

    assert m._llm_rerank_candidates(
        target=good_target(),
        top_candidates=candidates,
        context={},
        llm_settings={"provider": "none"},
    ) == (candidates, None)
    assert m._llm_rerank_candidates(
        target=good_target(),
        top_candidates=candidates[:1],
        context={},
        llm_settings={"provider": "ollama"},
    ) == (candidates[:1], None)
    unsupported_rows, unsupported_info = m._llm_rerank_candidates(
        target=good_target(),
        top_candidates=candidates,
        context={},
        llm_settings={"provider": "openai"},
    )
    assert unsupported_rows == candidates
    assert unsupported_info["reason"] == "unsupported_provider:openai"

    import builtins

    real_import = builtins.__import__

    def fail_ollama_import(name, *args, **kwargs):
        if name in {"src.LLMModule.ollama_client", "LLMModule.ollama_client"}:
            raise ImportError("missing ollama")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_ollama_import)
    failed_rows, failed_info = m._llm_rerank_candidates(
        target=good_target(),
        top_candidates=candidates,
        context={},
        llm_settings={"provider": "ollama", "top_n": 2},
    )
    assert failed_rows == candidates
    assert failed_info["status"] == "failed"
    assert failed_info["reason"].startswith("ollama_import_failed:")

    calls = []

    def fake_chat_json(**_kwargs):
        calls.append(_kwargs)
        return {"response": json.dumps({"chosen_unique_key": "chair", "ordered_unique_keys": ["table"], "reason": "ok"})}

    fake_module = types.SimpleNamespace(chat_json=fake_chat_json)

    def fake_import(name, *args, **kwargs):
        if name in {"src.LLMModule.ollama_client", "LLMModule.ollama_client"}:
            return fake_module
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    ranked, info = m._llm_rerank_candidates(
        target=good_target(target_id="chair_target"),
        top_candidates=candidates,
        context={"prompt_text": "modern chair", "selection_strategy": "style", "selection_mode": "optimal"},
        llm_settings={"provider": "ollama", "top_n": 2, "ollama_model": "mock"},
    )
    assert calls
    assert info["status"] == "applied"
    assert info["chosen_unique_key"] == "chair model"
    assert [row["unique_key"] for row in ranked] == ["chair model", "table", "tail"]
    assert ranked[0]["llm_rank"] == 1

    def bad_chat_json(**_kwargs):
        return {"response": json.dumps({"chosen_unique_key": "unknown"})}

    fake_module.chat_json = bad_chat_json
    unknown_rows, unknown_info = m._llm_rerank_candidates(
        target=good_target(),
        top_candidates=candidates[:2],
        context={},
        llm_settings={"provider": "ollama"},
    )
    assert unknown_rows == candidates[:2]
    assert unknown_info["reason"] == "ollama_returned_unknown_candidate"

    def raising_chat_json(**_kwargs):
        raise RuntimeError("offline")

    fake_module.chat_json = raising_chat_json
    offline_rows, offline_info = m._llm_rerank_candidates(
        target=good_target(),
        top_candidates=candidates[:2],
        context={},
        llm_settings={"provider": "ollama"},
    )
    assert offline_rows == candidates[:2]
    assert offline_info["reason"].startswith("ollama_rerank_failed:RuntimeError:")


def test_remaining_supplier_matcher_hard_reject_and_rank_edges(tmp_path, monkeypatch):
    asset = tmp_path / "asset.glb"
    asset.write_bytes(b"glb")
    ceiling_target = good_target(
        target_id="ceiling",
        semantic_group="lamp_ceiling",
        category="ceiling_light",
        size_m=[0.4, 0.4, 0.05],
    )
    bedroom_context = {"room_design_spec": {"room_type": "bedroom"}}
    chandelier = good_row(
        asset,
        semantic_group="lamp_ceiling",
        category_norm="ceiling_light",
        title="large chandelier",
        width_cm=50,
        depth_cm=50,
        height_cm=40,
    )
    assert m._hard_dimension_reject_info(ceiling_target, chandelier, bedroom_context)["hard_dimension_reject_reason"] == "rejected_bedroom_chandelier_not_requested"
    large_light = good_row(
        asset,
        semantic_group="lamp_ceiling",
        category_norm="ceiling_light",
        title="large ceiling lamp",
        width_cm=120,
        depth_cm=100,
        height_cm=40,
    )
    assert m._hard_dimension_reject_info(ceiling_target, large_light, bedroom_context)["hard_dimension_reject_reason"] == "rejected_oversized_for_target_aabb"
    missing_height = good_row(
        asset,
        semantic_group="lamp_ceiling",
        category_norm="ceiling_light",
        title="wide ceiling lamp",
        width_cm=130,
        depth_cm=130,
        height_cm=None,
    )
    wide_ceiling_target = {**ceiling_target, "size_m": [0.7, 0.7, 0.05]}
    assert m._hard_dimension_reject_info(wide_ceiling_target, missing_height, bedroom_context)["hard_dimension_reject_reason"] == "rejected_missing_height_for_oversized_light"
    assert m._hard_dimension_reject_info(
        good_target(semantic_group="bed", category="bed"),
        good_row(asset, semantic_group="lamp_ceiling", category_norm="ceiling_light"),
        {},
    ) is None
    assert m._bedroom_ceiling_light_reject_reason(
        good_target(semantic_group="decorative_set", category="decor", size_m=[0.35, 0.35, 0.1]),
        good_row(asset, semantic_group="decorative_set", category_norm="decor", width_cm=81, depth_cm=81, height_cm=20),
        {},
    ) is None

    base_target = good_target()
    base_row = good_row(asset)
    monkeypatch.setattr(m, "_source_policy_match_info", lambda _row: (False, {"source_policy_passed": False}))
    assert m._rank_candidate(base_target, base_row, {}) is None

    monkeypatch.setattr(m, "_source_policy_match_info", lambda _row: (True, {"source_policy_passed": True}))
    monkeypatch.setattr(m, "_hard_dimension_reject_info", lambda _target, _row, _context: {"hard_dimension_reject_reason": "blocked"})
    assert m._rank_candidate(base_target, base_row, {}) is None

    monkeypatch.setattr(m, "_hard_dimension_reject_info", lambda _target, _row, _context: None)
    monkeypatch.setattr(m, "candidate_identity_gate", lambda _target, _row: (False, {"identity_gate_passed": False}))
    assert m._rank_candidate(base_target, base_row, {}) is None

    monkeypatch.setattr(m, "candidate_identity_gate", lambda _target, _row: (True, {"identity_gate_passed": True}))
    monkeypatch.setattr(m, "_category_match_info", lambda _target, _row: (3, {"category_match": "mismatch"}))
    assert m._rank_candidate(base_target, base_row, {}) is None

    monkeypatch.setattr(m, "_category_match_info", lambda _target, _row: (0, {"category_match": "exact_group"}))
    monkeypatch.setattr(m, "_bedroom_ceiling_light_reject_reason", lambda _target, _row, _context: "rejected_oversized_for_target_aabb")
    assert m._rank_candidate(base_target, base_row, {}) is None

    monkeypatch.setattr(m, "_bedroom_ceiling_light_reject_reason", lambda _target, _row, _context: None)
    monkeypatch.setattr(m, "_size_match_info", lambda _target, _row: (1, 99.0, {"candidate_size_m": [9, 9, 9]}))
    assert m._rank_candidate(base_target, base_row, {}) is None

    monkeypatch.setattr(m, "_size_match_info", lambda _target, _row: (0, 0.0, {"candidate_size_m": [1, 1, 1]}))
    monkeypatch.setattr(m, "_bbox_fill_info", lambda _target, _row: {"passes_min_fill": False})
    assert m._rank_candidate(base_target, base_row, {}) is None

    monkeypatch.setattr(
        m,
        "_bbox_fill_info",
        lambda _target, _row: {"passes_min_fill": True, "fill_orientation": "direct"},
    )
    monkeypatch.setattr(m, "_bathroom_sink_quality_info", lambda _row: {"bathroom_sink_quality_reject_reason": "bad_sink"})
    assert m._rank_candidate(good_target(semantic_group="bathroom_sink", category="sink"), good_row(asset, semantic_group="bathroom_sink", category_norm="sink"), {}) is None

    monkeypatch.setattr(m, "_bathroom_sink_quality_info", lambda _row: {})
    monkeypatch.setattr(m, "_user_preference_match_info", lambda _target, _row, _context: (False, 0.0, {"preferences_ok": False}))
    assert m._rank_candidate(base_target, base_row, {}) is None


def test_supplier_matcher_additional_context_catalog_llm_and_cli_edges(tmp_path, monkeypatch):
    asset = tmp_path / "asset.glb"
    asset.write_bytes(b"glb")

    assert "red" in m._rgb_to_basic_color_tokens([1.0, 0.0, 0.08])
    mismatch_rank, mismatch_info = m._category_match_info(
        good_target(semantic_group="decorative_set", category="decor"),
        good_row(asset, semantic_group="lamp_floor", category_norm="lamp"),
    )
    assert mismatch_rank == 3
    assert mismatch_info["category_match"] == "mismatch"

    duplicate_rows, duplicate_info = m._llm_rerank_candidates(
        target=good_target(),
        top_candidates=[good_row(unique_key="same"), good_row(unique_key="same")],
        context={},
        llm_settings={"provider": "ollama"},
    )
    assert duplicate_rows[0]["unique_key"] == "same"
    assert duplicate_info is None

    import builtins

    real_import = builtins.__import__
    fake_module = types.SimpleNamespace(
        chat_json=lambda **_kwargs: {"response": json.dumps({"chosen_unique_key": "chair model", "ordered_unique_keys": ["chair model"]})}
    )

    def fake_import(name, *args, **kwargs):
        if name in {"src.LLMModule.ollama_client", "LLMModule.ollama_client"}:
            return fake_module
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    ranked, info = m._llm_rerank_candidates(
        target=good_target(),
        top_candidates=[good_row(unique_key="chair"), good_row(unique_key="tail")],
        context={},
        llm_settings={"provider": "ollama", "top_n": 2},
    )
    assert info["status"] == "applied"
    assert ranked[0]["unique_key"] == "chair"

    assert m._luxury_ceiling_intent({"prompt_tokens": {"chandelier"}, "room_style_tokens": ["baroque"]})
    oversize = m._hard_dimension_reject_info(
        good_target(target_id="shelf", semantic_group="shelf", category="shelf", size_m=[0.5, 0.5, 0.5]),
        good_row(asset, semantic_group="shelf", category_norm="shelf", width_cm=200, depth_cm=210, height_cm=50),
        {},
    )
    assert oversize["hard_dimension_reject_reason"] == "rejected_oversized_for_target_aabb"
    assert oversize["fit_orientation"] in {"direct", "swapped_xy"}

    monkeypatch.chdir(tmp_path)
    sidecar_dir = tmp_path / "reports" / "supplier_image_colors"
    sidecar_dir.mkdir(parents=True)
    sidecar = sidecar_dir / "supplier_catalog_canonical.image_colors.jsonl"
    sidecar.write_text("{}", encoding="utf-8")
    original_read_text = m.Path.read_text
    monkeypatch.setattr(
        m.Path,
        "read_text",
        lambda self, *args, **kwargs: (_ for _ in ()).throw(OSError("denied"))
        if self == sidecar
        else original_read_text(self, *args, **kwargs),
    )
    assert m._load_image_color_feature_sidecar(Path("supplier_catalog_canonical.json")) == {}
    monkeypatch.undo()

    invalid_catalog = tmp_path / "invalid_catalog.json"
    invalid_catalog.write_text(json.dumps({"bad": True}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Некорректный supplier catalog JSON"):
        m.load_supplier_catalog_json([invalid_catalog])

    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "items": [
                    123,
                    good_row(asset, unique_key="skip-site", source_site="other"),
                    good_row(asset, unique_key="skip-title", title=""),
                    good_row(asset, unique_key="skip-rich", description="", price_value="", width_cm="", depth_cm="", height_cm=""),
                    good_row(asset, unique_key="keep", source_site="homeconcept"),
                ]
            }
        ),
        encoding="utf-8",
    )
    rows = m.load_supplier_catalog_json([catalog], sites={"homeconcept"}, rich_only=True)
    assert [row["unique_key"] for row in rows] == ["keep"]

    merged = m._merge_catalog_rows(
        [
            {"unique_key": "u", "title": "", "asset_status": ""},
            {"unique_key": "u", "title": "Title", "asset_status": "ready", "color": "brown"},
        ]
    )
    assert merged == [{"unique_key": "u", "title": "Title", "asset_status": "ready", "color": "brown"}]

    targets_path = tmp_path / "targets.json"
    targets_path.write_text(json.dumps({"targets": [good_target()], "room": {}}), encoding="utf-8")
    (tmp_path / "style_profile.json").write_text(json.dumps({"style_label": "japandi", "style_hint": "warm wood"}), encoding="utf-8")
    context = m._load_matcher_context(targets_path, {"targets": [], "room": {}}, selection_strategy="bad")
    assert context["style_label"] == "japandi"
    assert "warm" in context["room_style_tokens"]
    monkeypatch.setattr(m, "read_json", lambda _path: (_ for _ in ()).throw(RuntimeError("bad profile")))
    assert m._load_matcher_context(targets_path, {"targets": [], "room": {}}, selection_strategy="bad")["style_label"] in (None, "")
    monkeypatch.undo()

    source_json = tmp_path / "scene.json"
    source_json.write_text(json.dumps({"placements": {"bad": True}}), encoding="utf-8")
    assert m._enrich_targets_from_source_scene({"source_json": str(source_json)}, [good_target()])[0]["target_id"] == "bed_1"
    source_json.write_text(json.dumps({"items": [{"id": "bed_1", "constraints": {"style": "modern"}}]}), encoding="utf-8")
    enriched = m._enrich_targets_from_source_scene({"source_json": str(source_json)}, ["raw", good_target(constraints={})])
    assert enriched[0] == "raw"
    assert enriched[1]["constraints"]["style"] == "modern"

    pref_path = tmp_path / "prefs.json"
    pref_path.write_text(json.dumps([]), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["supplier_layout_matcher", "--targets", str(targets_path), "--supplier-json", str(catalog), "--user-preferences-json", str(pref_path), "--out", str(tmp_path / "out.json")])
    with pytest.raises(RuntimeError, match="user preferences JSON"):
        m.main()

    room_spec = tmp_path / "room_spec.json"
    room_spec.write_text(json.dumps([]), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["supplier_layout_matcher", "--targets", str(targets_path), "--supplier-json", str(catalog), "--room-design-spec", str(room_spec), "--out", str(tmp_path / "out.json")])
    with pytest.raises(RuntimeError, match="room design spec"):
        m.main()

    monkeypatch.setattr(sys, "argv", ["supplier_layout_matcher", "--targets", str(targets_path), "--out", str(tmp_path / "out.json")])
    with pytest.raises(RuntimeError, match="supplier-db"):
        m.main()
