from __future__ import annotations

import math

import pytest

from src.suppliers import supplier_scoring as scoring
from src.suppliers.kitchen import kitchen_layout_solver as kitchen


def target(group: str = "chair", size=(0.5, 0.6, 0.9)) -> dict:
    return {
        "id": "target_1",
        "category": group,
        "semantic_group": group,
        "name": "modern blue wooden chair",
        "size_m": list(size),
    }


def candidate_row(**overrides) -> dict:
    row = {
        "unique_key": "ikea::id::1",
        "source_site": "ikea_de",
        "semantic_group": "chair",
        "category_norm": "chair",
        "category_raw": "chairs",
        "title": "Modern blue oak chair",
        "description": "Blue wooden contemporary modern chair, width 50 cm depth 60 cm height 90 cm",
        "width_cm": 50,
        "depth_cm": 60,
        "height_cm": 90,
        "price_value": 100.0,
        "brand": "IKEA",
        "style": "minimalist",
        "materials": "oak wood fabric",
        "color": "navy",
        "images_json": "[\"https://ikea.com/chair.jpg\"]",
        "image_color_features": {
            "color_tokens": ["navy", "blue"],
            "colors": {"top5": [{"basic_color": "blue"}, {"basic_color": "wood"}]},
        },
        "asset_local_path": "/tmp/model.glb",
        "asset_format": "glb",
        "asset_status": "downloaded_preferred",
        "model_download_url": "https://example.test/model.glb",
        "model_format": "glb",
        "vlm_description_text": "minimalism wood chair",
        "tags": ["minimal"],
    }
    row.update(overrides)
    return row


def design_spec() -> dict:
    return {
        "expanded_room_description": "modern minimalism blue wooden calm room",
        "object_requirements": {
            "chair": {
                "colors": ["blue", "wood_light"],
                "materials": ["wood", "fabric"],
                "style": ["minimalism"],
            },
            "bed": {
                "colors": ["white_warm"],
                "materials": ["fabric"],
                "style": ["soft_classic"],
            },
        },
        "color_palette": {
            "primary": "blue",
            "secondary": "oak",
            "forbidden_colors": ["red"],
        },
        "materials": {"preferred": ["wood", "fabric"], "forbidden": ["plastic"]},
        "style": {"primary": "minimalism", "allowed": ["modern"], "forbidden": ["baroque"]},
        "epoch": {"primary": "contemporary", "forbidden": ["baroque"]},
    }


def test_supplier_scoring_tokens_dimensions_and_price_stats():
    assert scoring._tokens({"a": ["Blue Chair", {"b": "oak-wood"}]}) >= {"blue", "chair", "oak-wood"}
    assert scoring._normalize_style_token("minimalist") == "minimalism"
    assert scoring._normalize_color_token("navy") == "blue"
    assert scoring._overlap_score({"blue", "wood"}, {"blue", "wood_light"})[0] == 1.0
    assert scoring._safe_float("12.5") == 12.5
    assert scoring._safe_float("bad") is None
    assert scoring._dimension_value_to_cm(2, "m") == 200.0
    assert scoring._dimension_value_to_cm(500, "мм") == 50.0

    inferred = scoring._infer_dimensions_cm_from_text({"description": "width: 50 cm depth: 60 cm height: 90 cm"})
    assert inferred == {"width": 50.0, "depth": 60.0, "height": 90.0}
    triple = scoring._infer_dimensions_cm_from_text({"title": "Size 50x60x90 cm"})
    assert triple == {"width": 50.0, "depth": 60.0, "height": 90.0}
    assert scoring._candidate_size(candidate_row(width_cm=None, depth_cm=None, height_cm=None)) == [0.5, 0.6, 0.9]
    assert scoring._dimension_weights("wardrobe")["height"] == 0.40

    stats = scoring.build_price_stats(
        [candidate_row(price_value=value, semantic_group="chair") for value in [100, 150, 200]]
        + [candidate_row(price_value=None, semantic_group="chair")]
    )
    assert stats["chair"]["min"] == 100.0
    assert stats["chair"]["p90"] == 200.0
    assert stats["chair"]["count"] == 3.0


def test_supplier_scoring_component_scores_and_final_modes():
    row = candidate_row()
    spec = design_spec()
    price_stats = scoring.build_price_stats([row, candidate_row(price_value=250)])

    category_score, category_info = scoring.compute_category_score(target(), row)
    assert category_score == 1.0
    assert category_info["category_match_v2"] == "exact_group"

    size_score, size_info = scoring.compute_size_score(target(), row)
    assert size_score > 0.95
    assert size_info["size_orientation"] == "direct"
    assert size_info["scale_policy"] == "preferred"

    assert scoring.compute_asset_availability_score(row)[1]["asset_availability"] == "ready_real_asset"
    assert scoring.compute_asset_availability_score(candidate_row(asset_local_path="", model_download_url="https://x/model.obj", model_format="obj"))[1]["asset_availability"] == "downloadable_asset"
    assert scoring.compute_asset_availability_score(candidate_row(asset_local_path="", model_download_url="", model_page_url="", model_download_landing_url=""))[0] == 0.0

    assert scoring.compute_color_score(target(), row, spec)[0] > 0.5
    assert scoring.compute_image_color_score(target(), row, spec)[0] > 0.5
    assert scoring.compute_material_score(target(), row, spec)[0] > 0.5
    assert scoring.compute_style_score(target(), row, spec)[0] > 0.5
    assert scoring.compute_epoch_score(target(), row, spec)[1]["matched_epoch"]
    assert scoring.compute_description_score(target(), row, spec)[0] > 0.5
    assert scoring.compute_source_quality_score(row)[1]["source_quality"]["trusted_product_catalog"] is True
    assert scoring.compute_price_score(row, price_stats, "balanced")[1]["price_known"] is True
    assert scoring.compute_design_similarity_score({"style_score": 1, "color_score": 1, "image_color_score": 1, "material_score": 1, "description_score": 1})[0] == 1.0

    result = scoring.score_candidate_for_mode(
        target=target(),
        row=row,
        room_design_spec=spec,
        mode="balanced",
        price_stats=price_stats,
    )
    assert result.acceptable
    assert result.total > 0.0
    assert result.breakdown["score_schema"] == "supplier_design_scores/v2"

    ranked, breakdown = scoring.rank_candidate_for_mode(
        target=target(),
        row=row,
        room_design_spec=spec,
        mode="balanced",
        price_stats=price_stats,
    )
    assert math.isclose(ranked, result.total)
    assert breakdown["candidate_score_acceptable"] is True


def test_supplier_scoring_rejects_bad_category_scale_and_missing_visual_reference():
    spec = design_spec()
    price_stats = {}

    mismatch = scoring.score_candidate_for_mode(
        target=target("chair"),
        row=candidate_row(semantic_group="bed", category_norm="bed", title="Bed"),
        room_design_spec=spec,
        mode="balanced",
        price_stats=price_stats,
    )
    assert not mismatch.acceptable
    assert mismatch.hard_reject_reason == "category_mismatch"

    huge = scoring.score_candidate_for_mode(
        target=target("chair", size=(0.5, 0.6, 0.9)),
        row=candidate_row(width_cm=500, depth_cm=600, height_cm=900),
        room_design_spec=spec,
        mode="balanced",
        price_stats=price_stats,
    )
    assert not huge.acceptable
    assert huge.hard_reject_reason == "unreasonable_scale"

    missing_visual = scoring.score_candidate_for_mode(
        target=target("bed", size=(1.8, 2.0, 0.6)),
        row=candidate_row(
            semantic_group="bed",
            category_norm="bed",
            width_cm=None,
            depth_cm=None,
            height_cm=None,
            description="",
            images_json="[]",
            image_color_features={},
            asset_local_path="",
            model_download_url="",
            model_page_url="",
            model_download_landing_url="",
            vlm_description_text="",
            vlm_description_summary="",
        ),
        room_design_spec=spec,
        mode="best_visual_reference",
        price_stats=price_stats,
    )
    assert not missing_visual.acceptable
    assert missing_visual.hard_reject_reason in {"missing_dimensions_for_bed", "missing_visual_reference_images"}


def test_supplier_scoring_remaining_edge_branches(monkeypatch):
    assert scoring._overlap_score({"blue"}, set()) == (0.55, [])
    assert scoring._dimension_value_to_cm(2.5) == 250.0
    assert scoring._infer_dimensions_cm_from_text({"description": "length: 80 cm"}) == {"depth": 80.0, "length": 80.0}
    assert scoring._dimension_weights("bed")["width"] == 0.40
    assert scoring._dimension_weights("desk")["height"] == 0.40
    assert scoring._scale_policy_info([0.7, 1.0, 1.0])["scale_policy"] == "moderate_with_penalty"
    assert scoring._target_size({"size_m": "bad"}) is None
    assert scoring._target_size({"size_m": ["bad", 1, 1]}) is None

    monkeypatch.setattr(scoring, "_row_dimension_cm", lambda row, axis: object())
    assert scoring._candidate_size({}) is None
    monkeypatch.undo()

    stats = scoring.build_price_stats([candidate_row(price_value=value, semantic_group="chair") for value in range(1, 12)])
    assert 9.0 <= stats["chair"]["p90"] <= 11.0

    same_family_score, same_family_info = scoring.compute_category_score(
        target("chair"),
        candidate_row(semantic_group="armchair", category_norm="armchair", title="Armchair"),
    )
    assert same_family_score == 0.78
    assert same_family_info["category_match_v2"] == "same_family"

    bed_size_score, bed_size_info = scoring.compute_size_score(
        target("bed", size=(1.8, 2.0, 0.6)),
        candidate_row(semantic_group="bed", category_norm="bed", width_cm=180, depth_cm=200, height_cm=60),
    )
    assert bed_size_score > 0.5
    assert bed_size_info["size_orientation"] == "direct"

    forbidden_row = candidate_row(
        color="red",
        image_color_features={"color_tokens": ["red"], "colors": {"top5": [{"basic_color": "red"}]}},
    )
    color_score, color_info = scoring.compute_color_score(target(), forbidden_row, design_spec())
    assert color_score < 0.2
    assert color_info["forbidden_color_hits"] == ["red"]
    assert color_info["forbidden_image_color_hits"] == ["red"]

    image_score, image_info = scoring.compute_image_color_score(target(), forbidden_row, design_spec())
    assert image_score < 0.2
    assert image_info["image_color_available"] is True
    assert scoring.compute_style_score(target(), candidate_row(style="plain"), {})[0] == 0.55

    quality_score, quality_info = scoring.compute_source_quality_score(
        candidate_row(images_json="{bad", unique_key="3ddd::x", source_site="3ddd", product_url="", model_page_url="", brand="")
    )
    assert quality_score < 1.0
    assert quality_info["source_quality"]["image_count"] == 0
    assert scoring.compute_price_score(candidate_row(price_value=None), {}, "cheapest")[1] == {"price_score": 0.2, "price_known": False}


def test_kitchen_layout_helpers_default_constrained_and_public_solver():
    assert kitchen._as_int("10.7", 0) == 11
    assert kitchen._as_int("bad", 5) == 5
    assert kitchen._as_bool("да") is True
    assert kitchen._as_bool("no") is False
    required = kitchen._normalize_required({"fridge": True, "dishwasher": True})
    assert required["sink"] is True
    assert required["microwave"] is True
    assert required["fridge"] is True

    module = kitchen._module("sink_cabinet", 600, cutouts=["sink"], role="sink")
    assert module["has_countertop"] is True
    assert kitchen._storage_module(450)["facade_layout"] == "one_door"
    assert kitchen._base_cabinet(600)["facade_layout"] == "two_doors"
    assert kitchen._split_storage_width(1370)[-1] >= 240
    assert kitchen._sum_width([{"width_mm": 100}, {"width_mm": 250}]) == 350
    assert kitchen._constraint_center({"sink": {"min_x_mm": 400, "max_x_mm": 800}}, "sink") == 600
    assert kitchen._constraint_is_hard({"sink": {"hard": "true"}}, "sink") is True

    sequential = kitchen._place_modules_sequentially([kitchen._base_cabinet(300), kitchen._base_cabinet(400)])
    assert [item["x_mm"] for item in sequential] == [0, 300]
    filled = kitchen._fill_gaps_with_storage([{"id": "fixed", "type": "sink_cabinet", "x_mm": 600, "width_mm": 600}], 1800)
    assert filled[0]["x_mm"] == 0
    assert filled[-1]["x_mm"] + filled[-1]["width_mm"] == 1800
    assert kitchen._overlaps({"x_mm": 0, "width_mm": 600}, {"x_mm": 500, "width_mm": 600})
    assert kitchen._gap_between({"x_mm": 0, "width_mm": 600}, {"x_mm": 800, "width_mm": 600}) == 200

    warnings: list[str] = []
    constrained = kitchen._build_constrained_modules(
        3200,
        kitchen._normalize_required({"fridge": True, "dishwasher": False}),
        {"sink": {"center_x_mm": 600}, "cooktop": {"center_x_mm": 1900}},
        warnings,
    )
    assert any(module["role"] == "sink" for module in constrained)
    assert any("sink_constraint_satisfied" in warning for warning in warnings)

    with pytest.raises(ValueError, match="sink_constraint_unmet"):
        kitchen._build_constrained_modules(
            1800,
            kitchen._normalize_required({}),
            {"sink": {"center_x_mm": 50, "hard": True}},
            [],
        )

    default_warnings: list[str] = []
    default_modules = kitchen._build_default_modules(4200, required, default_warnings)
    assert any(module["type"] == "fridge_slot" for module in default_modules)
    assert any(module["type"] == "dishwasher_slot" for module in default_modules)

    counter = kitchen._build_countertop_segments(default_modules)
    backsplash = kitchen._build_backsplash_segments(counter)
    upper, decor, upper_warnings = kitchen._build_upper_modules(default_modules, required)
    accessories = kitchen._build_countertop_accessories(default_modules, {"decor_accessories": True})
    zones = kitchen._build_functional_zones(default_modules, upper, decor + accessories)
    openings = kitchen._build_openings(default_modules, upper, counter)
    breakdown = kitchen._build_cabinet_breakdown(default_modules, upper)
    targets = kitchen._build_asset_targets(zones)
    assert counter and backsplash
    assert upper
    assert "facades" in breakdown
    assert any(target["role"] == "sink" for target in targets)
    assert any(opening["type"].endswith("_cutout") for opening in openings)
    assert isinstance(upper_warnings, list)

    result = kitchen.solve_kitchen_layout(
        {"available_width_mm": 3600, "layout_type": "u-shaped", "wall_id": "kitchen_wall"},
        plumbing_point={"x_mm": 500},
        entry_zone=None,
        required_appliances={"fridge": True, "dishwasher": True, "decor_accessories": True},
    )
    assert result["layout_type"] == "straight"
    assert result["wall_id"] == "kitchen_wall"
    assert any("layout_forced_to_straight" in warning for warning in result["warnings"])
    assert result["base_modules"]
    assert result["countertop_segments"]
