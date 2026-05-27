import json
from pathlib import Path

import pytest

import src.supplier_replacement_report as rr
from src.supplier_replacement_report import write_supplier_replacement_reports


def test_supplier_replacement_reports_include_product_price_and_image(tmp_path: Path) -> None:
    bindings_path = tmp_path / "base_supplier_bindings.llm.assets.json"
    scene_path = tmp_path / "scene_supplier.v1.json"

    bindings_path.write_text(
        json.dumps(
            {
                "bindings": [
                    {
                        "target_id": "obj_0001",
                        "category": "BedFactory",
                        "semantic_group": "bed",
                        "selection_status": "llm_reranked_top1_selected",
                        "replacement_reason": "major_furniture_group",
                        "provenance": {"final_asset_source": "supplier_catalog"},
                        "chosen_candidate": {
                            "title": "OM Кровать Camilla",
                            "brand": "idealbeds",
                            "source_site": "idealbeds_yadisk",
                            "product_url": "https://example.test/product",
                            "model_download_url": "https://example.test/model.obj",
                            "price_value": 120000,
                            "price_currency": "RUB",
                            "images_json": json.dumps(["https://example.test/photo.jpg"]),
                            "asset_status": "archive_extracted_preferred",
                            "asset_format": "obj",
                            "asset_local_path": "/tmp/model.obj",
                            "width_cm": 174,
                            "depth_cm": 216,
                            "height_cm": 100,
                        },
                        "selection_notes": ["ok"],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    scene_path.write_text(
        json.dumps(
            {
                "placements": [
                    {
                        "id": "obj_0001",
                        "meta": {
                            "supplier_binding_applied": True,
                            "supplier_binding_target_id": "obj_0001",
                            "original_generated_item": {
                                "name": "BedFactory",
                                "category": "BedFactory",
                            },
                        },
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    info = write_supplier_replacement_reports(
        bindings_json_path=bindings_path,
        run_dir=tmp_path,
        supplier_scene_json_path=scene_path,
    )

    short_md = Path(info["short_md"]).read_text(encoding="utf-8")
    full_md = Path(info["extended_md"]).read_text(encoding="utf-8")

    assert info["replacement_count"] == 1
    assert "BedFactory" in short_md
    assert "[ссылка](https://example.test/product)" in short_md
    assert "120 000 руб." in short_md
    assert "![фото](https://example.test/photo.jpg)" in short_md
    assert "OM Кровать Camilla" in full_md
    assert "idealbeds" in full_md


def test_supplier_replacement_reports_include_surface_purchase_quantities(tmp_path: Path) -> None:
    bindings_path = tmp_path / "base_supplier_bindings.llm.assets.json"
    scene_path = tmp_path / "scene_supplier.v1.json"
    flooring_path = tmp_path / "flooring.selection.supplier.v1.json"
    pricing_path = tmp_path / "surface_materials.pricing.supplier.json"

    bindings_path.write_text(
        json.dumps(
            {
                "bindings": [
                    {
                        "target_id": "obj_0001",
                        "category": "BedFactory",
                        "semantic_group": "bed",
                        "selection_status": "heuristic_top1_selected",
                        "provenance": {"final_asset_source": "supplier_catalog"},
                        "chosen_candidate": {
                            "title": "Кровать",
                            "product_url": "https://example.test/bed",
                            "price_value": 120000,
                            "price_currency": "RUB",
                        },
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    scene_path.write_text(
        json.dumps(
            {
                "placements": [
                    {
                        "id": "obj_0001",
                        "meta": {
                            "supplier_binding_applied": True,
                            "supplier_binding_target_id": "obj_0001",
                        },
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    flooring_path.write_text(
        json.dumps(
            {
                "selected_material": {
                    "name": "Ламинат Swisskrono",
                    "brand": "SWISSKRONO",
                    "sku": "833024",
                    "product_url": "https://example.test/floor",
                    "price": 2636.51,
                    "price_currency": "RUB",
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    pricing_path.write_text(
        json.dumps(
            {
                "schema": "surface_materials_pricing/v1",
                "items": [
                    {
                        "target_id": "surface_floor",
                        "sku": "833024",
                        "product_url": "https://example.test/floor",
                        "price_status": "estimated",
                        "currency": "RUB",
                        "final_price_value": 23728.59,
                        "quantity": 9,
                        "quantity_unit": "package",
                        "unit_price_value": 2636.51,
                        "package_area_m2": 1.845,
                        "coverage_area_m2": 15.12,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    info = write_supplier_replacement_reports(
        bindings_json_path=bindings_path,
        run_dir=tmp_path,
        supplier_scene_json_path=scene_path,
    )

    short_md = Path(info["short_md"]).read_text(encoding="utf-8")
    html = Path(info["html"]).read_text(encoding="utf-8")

    assert info["surface_material_count"] == 1
    assert info["surface_material_total"] == "23 728.59 руб."
    assert info["estimate_total"] == "143 728.59 руб."
    assert "Ламинат Swisskrono" in short_md
    assert "15.12" in short_md
    assert "1.845" in short_md
    assert "9 уп." in short_md
    assert "143 728.59 руб." in html


def test_supplier_replacement_report_pure_helpers_cover_html_summary_and_edge_cases(tmp_path: Path) -> None:
    scene_path = tmp_path / "scene.json"
    scene_path.write_text(
        json.dumps(
            {
                "meta": {"supplier_binding_summary": {"replaced_count": 2, "local_asset_replaced_count": 1}},
                "items": [
                    {"id": "a", "meta": {"supplier_binding_applied": True, "supplier_binding_target_id": "target_a"}},
                    {"id": "ignored", "meta": {"supplier_binding_applied": False}},
                    "bad",
                ],
            }
        ),
        encoding="utf-8",
    )
    build_report = tmp_path / "build_report.json"
    build_report.write_text(json.dumps({"item_issues": {"target_a": ["missing texture", "", 3], "bad": "x"}}), encoding="utf-8")

    assert rr._applied_scene_items(scene_path)["target_a"]["id"] == "a"
    assert rr._scene_supplier_summary(scene_path)["replaced_count"] == 2
    assert rr._build_issues_by_target(build_report) == {"target_a": ["missing texture", "3"]}
    assert rr._applied_scene_items(None) == {}
    assert rr._scene_supplier_summary(tmp_path / "missing.json") == {}
    assert rr._build_issues_by_target(tmp_path / "missing.json") == {}

    candidate = {
        "unique_key": "cand1",
        "title": "Chair",
        "category_norm": "chair",
        "price_value": "12 345,67",
        "price_currency": "RUB",
        "asset_local_path": "/tmp/model.glb",
        "model_download_url": "https://example.test/model.zip",
        "score_breakdown": {
            "final_score": 0.91,
            "category_score": 0.8,
            "size_score": 0.7,
            "style_score": 0.6,
            "color_score": 0.5,
            "material_score": 0.4,
            "price_score": 0.3,
            "asset_availability_score": 1.0,
            "identity_required_hits": ["chair"],
        },
    }
    assert rr._parse_images('["a.jpg", ""]') == ["a.jpg"]
    assert rr._parse_images(["b.jpg", " "]) == ["b.jpg"]
    assert rr._parse_images("{bad") == ["{bad"]
    assert rr._format_price(None) == "цена не указана"
    assert rr._format_number(None) == "не указано"
    assert rr._float_or_none("bad") is None
    assert rr._fmt_score(0.12345) == "0.123"
    assert rr._json_safe({"x": (1, float("nan"))}) == {"x": [1, None]}
    assert rr._candidate_id(candidate) == "cand1"
    assert rr._candidate_title(candidate) == "Chair"
    assert rr._candidate_category(candidate) == "chair"
    assert rr._candidate_has_local_asset(candidate) is True
    assert rr._candidate_has_downloadable_asset(candidate) is True
    assert rr._candidate_model_url(candidate) == "https://example.test/model.zip"
    assert "score-table" in rr._score_table_html(candidate["score_breakdown"])
    assert "cand1" in rr._top_candidates_html([candidate])
    assert rr._top_candidates_html([]).startswith("<div")
    assert "фото 1" in rr._image_links_html({"images": ["https://example.test/a.jpg"]})
    assert rr._image_links_html({}) == "нет фото"

    binding = {
        "target_id": "target_a",
        "selection_status": "heuristic_top1_selected",
        "chosen_candidate": candidate,
        "selection_notes": ["scene_consistency_shared_candidate:cand1"],
    }
    assert rr._binding_consistency_info(binding, {})["shared_candidate"] == "cand1"
    group_info = rr._binding_consistency_info(
        {"target_id": "target_b"},
        {"scene_consistency": {"applied_groups": [{"target_ids": ["target_b"], "group_id": "chairs", "chosen_candidate_id": "cand2"}]}},
    )
    assert group_info["consistency_group_id"] == "chairs"
    assert rr._apply_status_for_row(binding, {"id": "a"}) == "applied"
    assert rr._apply_status_for_row(binding, None) == "not_applied"
    assert rr._apply_status_for_row({"selection_status": "unmatched"}, None) == "not_selected"

    rows = [
        {
            "target_id": "target_a",
            "target_category": "chair",
            "status": "heuristic_top1_selected",
            "is_selected": True,
            "candidate_id": "cand1",
            "new_title": "Chair",
            "chosen_candidate": candidate,
            "price_value": "12345.67",
            "final_score": 0.91,
            "score_breakdown": candidate["score_breakdown"],
            "asset_local_path": "/tmp/model.glb",
            "acquisition_status": "ok",
            "build_issues": ["missing texture"],
            "consistency_applied": True,
            "shared_candidate": "cand1",
            "selection_mode": "optimal",
            "replacement_policy": "replace_with_supplier",
            "apply_status": "applied",
            "has_local_asset": True,
            "used_alternative_candidate": True,
        },
        {
            "target_id": "target_b",
            "status": "no_candidates_found",
            "is_selected": False,
            "replacement_policy": "keep",
        },
    ]
    assert rr._price_value_sum(rows, "price_value") == 12345.67
    assert rr._average_scores(rows)["final_score"] == 0.91
    targets = rr._summary_targets(rows)
    assert targets[0]["chosen_candidate_id"] == "cand1"
    summary = rr._replacement_summary_json(
        rows,
        bindings_json_path=tmp_path / "bindings.json",
        scene_json_path=scene_path,
        mode="optimal",
        blender_build_report_path=build_report,
    )
    assert summary["counts"]["selected_count"] == 1
    assert summary["counts"]["applied_replacement_count"] == 2
    assert not summary["warnings"]


def test_supplier_replacement_report_remaining_edge_branches(tmp_path: Path) -> None:
    assert rr._format_price(12.5, "USD") == "12.50 USD"
    assert rr._format_price("not-a-number", "EUR") == "not-a-number EUR"
    assert rr._format_number("bad") == "bad"
    assert rr._fmt_score("bad") == "n/a"
    assert rr._candidate_id({"name": "Fallback Name"}) == "Fallback Name"
    assert rr._candidate_id("bad") == ""
    assert rr._candidate_title("bad") == ""
    assert rr._candidate_category("bad") == ""
    assert rr._score_breakdown({"score_breakdown": "bad"}) == {}
    assert rr._candidate_final_score({"score": "0.25"}) == 0.25
    assert rr._candidate_has_local_asset("bad") is False
    assert rr._candidate_has_downloadable_asset({"score_breakdown": {"has_model_url": True}}) is True
    assert rr._candidate_model_url({"model_page_url": "https://example.test/model"}) == "https://example.test/model"
    assert rr._product_link({}) == ""
    assert rr._selected_candidate({"selection_status": "no_candidates_found"}) is None
    assert (
        rr._selected_candidate(
            {
                "selection_status": "heuristic_top1_selected",
                "chosen_candidate": {"id": "x"},
                "provenance": {"final_asset_source": "generated"},
            }
        )
        is None
    )
    assert rr._status_badge_class("failed_to_import") == "bad"
    assert rr._status_badge_class("no_candidates_found") == "warn"
    assert rr._status_badge_class("generated_proxy") == "muted"
    assert rr._status_badge_class("pending") == "neutral"
    assert rr._candidate_bool(False) == "no"
    assert rr._price_value_sum([{"price": "bad"}, {"price": ""}, {"price": 2}], "price") == 2
    assert rr._price_value_sum([{"price": "bad"}], "price") is None
    assert rr._score_table_html({}) == '<div class="na">n/a</div>'
    assert "n/a" in rr._diagnostics_html({})
    assert "used_alternative_candidate" in rr._diagnostics_html({"used_alternative_candidate": True})
    assert rr._surface_quantity_text({}) == "не рассчитано"
    assert rr._surface_quantity_text({"quantity": 2, "quantity_unit": "roll"}) == "2 рул."
    assert rr._surface_quantity_text({"quantity": 3, "quantity_unit": "box"}) == "3 box"

    bad_bindings_path = tmp_path / "bad_bindings.json"
    bad_bindings_path.write_text(json.dumps({"meta": [], "bindings": {"bad": True}}), encoding="utf-8")
    with pytest.raises(RuntimeError):
        rr._replacement_rows(
            bindings_json_path=bad_bindings_path,
            supplier_scene_json_path=None,
        )

    build_report = tmp_path / "build_report.json"
    build_report.write_text(
        json.dumps({"item_issues": {"target_1": ["used_alternative_candidate:rank2"]}}),
        encoding="utf-8",
    )
    bindings_path = tmp_path / "bindings.json"
    bindings_path.write_text(
        json.dumps(
            {
                "meta": {
                    "selection_mode": "meta-mode",
                    "scene_consistency": {
                        "applied_groups": [
                            {
                                "target_ids": ["target_2"],
                                "semantic_group": "pair",
                                "shared_candidate": "shared-candidate",
                            }
                        ]
                    },
                },
                "bindings": [
                    "bad",
                    {
                        "target_id": "target_1",
                        "category": "lamp",
                        "semantic_group": "lamp",
                        "selection_status": "no_real_asset_after_acquisition",
                        "chosen_candidate": {
                            "name": "Raw Lamp",
                            "model_download_landing_url": "https://example.test/download",
                            "images": "https://example.test/photo.jpg",
                            "score_breakdown": "bad",
                        },
                        "selection_notes": ["asset_acquisition_selected_real_candidate_rank:2", "no asset"],
                        "top_candidates": [{"id": "top_1"}, "bad"],
                    },
                    {
                        "target_id": "target_2",
                        "category": "chair",
                        "semantic_group": "chair",
                        "supplier_selection_mode": "binding-mode",
                        "selection_status": "llm_reranked_top1_selected",
                        "provenance": {"final_asset_source": "supplier_catalog_pending"},
                        "chosen_candidate": {
                            "id": "candidate_2",
                            "title": "Chosen Chair",
                            "price_value": 5,
                            "price_currency": "EUR",
                            "score": 0.4,
                        },
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    rows = rr._replacement_rows(
        bindings_json_path=bindings_path,
        supplier_scene_json_path=None,
        blender_build_report_path=build_report,
    )
    assert [row["target_id"] for row in rows] == ["target_1", "target_2"]
    assert rows[0]["new_title"] == "Raw Lamp"
    assert rows[0]["image_url"] == "https://example.test/photo.jpg"
    assert rows[0]["has_downloadable_asset"] is True
    assert rows[0]["used_alternative_candidate"] is True
    assert rows[0]["apply_status"] == "not_selected"
    assert rows[0]["rejection_reason"]
    assert rows[0]["top_candidates"] == [{"id": "top_1"}]
    assert rows[1]["is_selected"] is True
    assert rows[1]["selection_mode"] == "binding-mode"
    assert rows[1]["consistency_group_id"] == "pair"
    assert rows[1]["shared_candidate"] == "shared-candidate"

    (tmp_path / "surface_materials.pricing.bad.json").write_text("{bad", encoding="utf-8")
    (tmp_path / "surface_materials.pricing.empty.json").write_text(json.dumps({"items": "bad"}), encoding="utf-8")
    (tmp_path / "surface_materials.pricing.ok.json").write_text(
        json.dumps(
            {
                "items": [
                    "bad",
                    {
                        "target_id": "surface_floor",
                        "sku": "floor-sku",
                        "quantity": 4,
                        "quantity_unit": "package",
                        "unit_price_value": 10,
                        "final_price_value": 40,
                        "coverage_area_m2": 7.5,
                    },
                    {
                        "target_id": "surface_walls",
                        "product_url": "https://example.test/wall",
                        "quantity": 2,
                        "quantity_unit": "roll",
                        "unit_price_value": 8,
                        "final_price_value": 16,
                        "coverage_area_m2": 11,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    local_image = tmp_path / "data/sourse/obi_floor_coverings_cards/rel/floor.jpg"
    local_image.parent.mkdir(parents=True, exist_ok=True)
    local_image.write_bytes(b"jpg")
    (tmp_path / "flooring.selection.supplier.v1.json").write_text(
        json.dumps(
            {
                "selected_material": {
                    "name": "Floor",
                    "sku": "floor-sku",
                    "price": 10,
                    "price_currency": "RUB",
                    "local_image_paths": ["rel/floor.jpg"],
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "wall_material.selection.supplier.v1.json").write_text(
        json.dumps({"selected_material": "bad"}),
        encoding="utf-8",
    )
    (tmp_path / "wall_material.selection.base.v1.json").write_text(
        json.dumps(
            {
                "selected_material": {
                    "name": "Wall",
                    "sku": "wall-sku",
                    "product_url": "https://example.test/wall",
                    "price": 8,
                    "image_urls": ["https://example.test/wall.jpg"],
                }
            }
        ),
        encoding="utf-8",
    )

    surface_rows = rr._surface_material_rows(tmp_path)
    assert [row["label"] for row in surface_rows] == ["Пол", "Обои"]
    assert surface_rows[0]["image_url"].endswith("floor.jpg")
    assert surface_rows[0]["final_price_value"] == 40
    assert surface_rows[1]["quantity_unit"] == "roll"
    assert "Материалы поверхностей" in rr._surface_materials_html(surface_rows)

    warning_summary = rr._replacement_summary_json(
        [
            {
                "target_id": "warn_target",
                "status": "heuristic_top1_selected",
                "is_selected": True,
                "selection_mode": "optimal",
                "candidate_id": "",
                "score_breakdown": {},
            }
        ],
        bindings_json_path=bindings_path,
        scene_json_path=tmp_path / "missing_scene.json",
        blender_build_report_path=None,
        mode=None,
    )
    assert warning_summary["warnings"]

    report_info = write_supplier_replacement_reports(
        bindings_json_path=bindings_path,
        run_dir=tmp_path,
        supplier_scene_json_path=None,
        blender_build_report_path=build_report,
        summary_filename="summary.json",
        mode="edge-mode",
    )
    summary_json = json.loads(Path(report_info["summary_json"]).read_text(encoding="utf-8"))
    assert report_info["counts"]["selected_count"] == 1
    assert summary_json["mode"] == "edge-mode"
