from __future__ import annotations

import csv
import json
import math
import types
from pathlib import Path

import pytest

from src.suppliers.kitchen import kitchen_appliance_matcher as app
from src.suppliers.kitchen import kitchen_assembly_builder as assembly
from src.suppliers.kitchen import kitchen_bom_estimator as bom
from src.suppliers.kitchen import kitchen_catalog_loader as loader
from src.suppliers.kitchen import kitchen_design_spec as design
from src.suppliers.kitchen import kitchen_layout_solver as layout_solver
from src.suppliers.kitchen import kitchen_material_matcher as matcher
from src.suppliers.kitchen import kitchen_pipeline as kitchen_pipeline
from src.suppliers.kitchen import kitchen_roles as roles
from src.suppliers.kitchen import kitchen_supplier_inventory as inventory
from src.suppliers.kitchen import kitchen_text_features as text


def material(role: str, sku: str, *, price: float = 1000.0, colors=None, pattern="plain", finish="matte") -> dict:
    return {
        "source": "unit",
        "name": f"{sku} {role} white oak stone matte",
        "sku": sku,
        "brand": "Unit",
        "price": price,
        "price_currency": "RUB",
        "availability": "in_stock",
        "kitchen_role": role,
        "unit": "piece",
        "dimensions": {"length_mm": 3000, "width_mm": 1200, "thickness_mm": 38},
        "visual": {
            "base_colors": colors or ["white", "wood"],
            "tone": "light",
            "pattern": pattern,
            "finish": finish,
            "style_tags": ["modern", "scandinavian"],
        },
        "flags": {"is_moisture_resistant": True, "is_accent_only": role == "accent_edge_band"},
    }


def test_text_features_roles_design_and_catalog_loader(tmp_path: Path) -> None:
    assert assembly._material_summary(None) is None
    assert text.normalize_text("Ёлка × ХДФ") == "елка x xдф"
    assert "oak-white" in text.tokenize("Oak-white 18mm")
    assert text.parse_json_maybe('{"a": 1}') == {"a": 1}
    assert text.parse_json_maybe("{bad", default=[]) == []
    assert text.safe_float("1 234,5 ₽") == 1234.5
    assert text.safe_int("10.6") == 11
    assert text.first_present({"a": "", "b": 2}, ["a", "b"]) == 2
    assert text.extract_size_triplet_mm("2800x2070x18") == (2800, 2070, 18)
    assert text.contains_any("матовая столешница", ["мат"])
    assert text.score_keyword_overlap("white oak matte", ["white", "stone"]) == 0.5
    assert {"wood", "light_wood"} <= text.detect_color_families("light oak")
    assert text.detect_pattern("white marble slab") == "marble"
    assert text.detect_finish("high gloss panel") == "gloss"
    assert text.detect_tone({"black"}, "") == "dark"
    assert "modern" in text.detect_style_tags("graphite concrete")
    assert text.normalize_color_request(["warm white", "dark oak"]) == ["white", "dark_wood", "wood"]
    assert text.clamp01(2.0) == 1.0

    raw_countertop = {
        "name": "Столешница PerfectSense дуб светлый 3000x600x38",
        "sku": "ct1",
        "price": "1234,5",
        "properties_json": json.dumps({"Категория": "Столешница", "Единица измерения": "шт"}),
        "normalized": {"style_tags": ["modern"]},
        "availability": "В наличии",
    }
    normalized = roles.normalize_material_record(raw_countertop)
    assert normalized["kitchen_role"] == "premium_countertop_slab"
    assert normalized["dimensions"] == {"length_mm": 3000, "width_mm": 600, "thickness_mm": 38}
    assert normalized["availability"] == "in_stock"
    assert roles.infer_unit({"properties_json": json.dumps({"Единица измерения": "м.п"})}, "edge_band") == "m"
    assert roles.infer_kitchen_role({"properties_json": json.dumps({"Категория": "Планка соединительная"})}) == "joint_profile"
    assert roles.infer_kitchen_role({"properties_json": json.dumps({"Категория": "Решетка вентиляционная"})}) == "ventilation_grille"

    assert design._infer_primary_style("скандинавская кухня") == "scandinavian"
    assert design._infer_primary_style("japandi warm wood") == "japandi"
    assert design._infer_primary_style("classic cream kitchen") == "classic"
    assert design._infer_primary_style("minimal white kitchen") == "minimalism"
    assert design._infer_primary_style("simple modern kitchen") == "modern"
    assert design._infer_palette_from_prompt("green kitchen", {"facades": "white"})["facades"] == ["white"]
    spec = design.build_kitchen_design_spec(
        "loft black kitchen",
        recommended_colors={"countertop": ["stone"]},
        appliances={"fridge": True, "microwave": True},
        room_meta={"id": "r1"},
    )
    assert spec["style"]["primary"] == "loft"
    assert spec["functional_requirements"]["fridge"] is True
    assert "stone" in spec["palette"]["countertop"]

    jsonl = tmp_path / "materials.jsonl"
    jsonl.write_text(json.dumps(raw_countertop, ensure_ascii=False) + "\n\n", encoding="utf-8")
    csv_path = tmp_path / "materials.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "sku", "properties_json"])
        writer.writeheader()
        writer.writerow({"name": "Кромка белая 3000x22", "sku": "edge1", "properties_json": json.dumps({"Категория": "Кромка"})})
    assert len(loader.load_jsonl(jsonl)) == 1
    assert len(loader.load_csv(csv_path)) == 1
    assert loader.load_raw_catalog(jsonl)[0]["sku"] == "ct1"
    assert loader.group_by_role(loader.load_kitchen_material_catalog(csv_path))["edge_band"]
    with pytest.raises(ValueError):
        loader.load_raw_catalog(tmp_path / "materials.txt")


def test_material_matcher_scores_and_selects_all_roles() -> None:
    materials = [
        material("facade_sheet", "facade", price=8000, colors=["white"], pattern="plain"),
        material("board_sheet", "body", price=5000, colors=["white"], pattern="plain"),
        material("countertop_slab", "counter", price=9000, colors=["gray", "stone"], pattern="stone"),
        material("backsplash_panel", "back", price=4500, colors=["gray", "stone"], pattern="stone"),
        material("edge_band", "edge", price=50, colors=["white"]),
        material("countertop_wall_plinth", "plinth", price=1500, colors=["gray"]),
        material("joint_profile", "joint", price=900, colors=["black"]),
        material("end_profile", "end", price=650, colors=["black"]),
        material("corner_profile", "corner", price=650, colors=["black"]),
        material("accent_edge_band", "accent", price=90, colors=["red"]),
    ]
    layout = {
        "countertop_segments": [{"width_mm": 1800}, {"width_mm": 1200}],
        "backsplash_segments": [{"width_mm": 3000}],
    }
    spec = {
        "style": {"primary": "scandinavian", "secondary": ["modern"]},
        "palette": {"facades": ["white"], "countertop": ["stone"], "backsplash": ["stone"], "accent": ["black"]},
        "materials_intent": {"facades": ["plain"], "countertop": ["stone"], "backsplash": ["stone"]},
    }
    stats = matcher._price_stats(materials)
    assert stats["facade_sheet"][0] == 8000
    assert matcher._price_score({"kitchen_role": "missing"}, {}, "cheapest") == 0.2
    assert matcher._availability_score({"availability": "под заказ"}) == 0.55
    assert matcher._desired_colors_for_role(spec, "edge_band")
    assert matcher._dimension_score(materials[2], "countertop", layout) > 0.8
    assert matcher._durability_score(materials[2], "countertop") == 1.0
    assert matcher._compatibility_score(materials[3], "backsplash", {"countertop": materials[2]}) > 0.5

    scored = matcher.score_material(materials[0], "facade", spec, layout, "optimal", stats)
    assert scored["final_score"] > 0.0
    with pytest.raises(ValueError):
        matcher.score_material(materials[0], "facade", spec, layout, "bad", stats)

    selected = matcher.select_kitchen_materials(materials, spec, layout, mode="optimal", top_n=2)
    assert {"facade", "body", "countertop", "backsplash", "edge_band", "wall_plinth", "joint_profile", "end_profile", "corner_profile"} <= set(
        selected["materials"]
    )
    assert selected["palette_consistency_score"] > 0.0
    assert matcher.estimate_palette_consistency({}) == 0.0
    with pytest.raises(ValueError):
        matcher.select_kitchen_materials(materials, spec, layout, mode="bad")


def test_kitchen_text_material_and_bom_edge_branches(monkeypatch) -> None:
    assert text.parse_json_maybe({"ready": True}) == {"ready": True}
    assert text.parse_json_maybe("", default={"d": 1}) == {"d": 1}
    assert text.parse_json_maybe(42, default=[]) == []
    assert text.safe_float(None, 1.5) == 1.5
    assert text.safe_float(True) == 1.0
    assert text.safe_float(3) == 3.0
    assert text.safe_float(float("nan"), 2.0) == 2.0
    assert text.safe_float("no-number", 7.0) == 7.0
    monkeypatch.setattr(text, "_FLOAT_RE", types.SimpleNamespace(search=lambda _value: types.SimpleNamespace(group=lambda _idx: "bad")))
    assert text.safe_float("123", 9.0) == 9.0
    monkeypatch.undo()
    assert text.extract_size_triplet_mm("plain text") == (None, None, None)
    assert text.score_keyword_overlap("anything", []) == 0.5
    assert text.score_keyword_overlap("warm oak panel", ["", "oak stone"]) == 0.25
    assert text.detect_finish("deep textured surface") == "textured"
    assert text.detect_tone(set(), "balanced gray") == "neutral"

    layout = {"countertop_segments": [{"width_mm": 5000}], "backsplash_segments": [{"width_mm": 2500}]}
    spec = {
        "style": {"primary": "classic"},
        "palette": {"facades": ["light oak"], "countertop": ["stone"], "backsplash": ["white"]},
        "materials_intent": {"backsplash": ["subtle tile"]},
    }
    facade = material("facade_sheet", "facade_edge", price=2000, colors=["light_wood"], pattern="wood")
    text_only = {"name": "white subtle tile", "kitchen_role": "backsplash_panel", "availability": "", "visual": {}, "price": None}
    small_board = {"name": "small board", "kitchen_role": "board_sheet", "dimensions": {"length_mm": 1300, "width_mm": 700}, "visual": {}, "price": 1000}
    accent = material("accent_edge_band", "accent", colors=["red"])
    wrong_role = material("board_sheet", "wrong", colors=["red"])

    assert matcher._availability_score({"availability": "unknown"}) == 0.45
    assert matcher._role_score(wrong_role, "countertop") == 0.0
    assert matcher._color_score({}, {}, "facade") == 0.5
    assert matcher._color_score(text_only, spec, "backsplash") > 0.0
    assert matcher._color_score(facade, spec, "facade") >= 0.85
    assert matcher._style_score(facade, {}) == 0.5
    assert matcher._pattern_score(text_only, spec, "backsplash") > 0.0
    assert matcher._finish_score({"visual": {}}, spec, "facade") == 0.5
    assert matcher._finish_score({"visual": {"finish": "matte"}}, {"style": {"primary": "modern"}}, "facade") == 0.9
    assert matcher._finish_score({"visual": {"finish": "matte"}}, {"style": {"primary": "classic"}}, "facade") == 0.7
    assert matcher._dimension_score(small_board, "facade", layout) == 0.75
    assert matcher._dimension_score({"dimensions": {}}, "facade", layout) == 0.5
    assert matcher._compatibility_score(facade, "backsplash", {"facade": facade}) == 0.5
    assert matcher.score_material(wrong_role, "countertop", spec, layout, "optimal", {})["final_score"] < 0.1
    assert matcher.score_material(material("countertop_slab", "bad_color", colors=["red"]), "countertop", spec, layout, "optimal", {})["final_score"] < 0.4
    assert matcher.score_material(accent, "facade", spec, layout, "optimal", {})["final_score"] < 0.1
    missing_selection = matcher.select_kitchen_materials([facade], spec, layout)
    assert any(warning.startswith("no_material_candidates_for_role:countertop") for warning in missing_selection["warnings"])
    assert matcher.estimate_palette_consistency({"facade": facade}) == 0.5

    assert bom._price({"price": 0, "kitchen_role": "facade_sheet"}, "facade_sheet") > 0
    assert bom._price(None, "unknown_role") == 1000.0
    assert bom._facade_panels(
        {
            "base_modules": [
                {"has_facade": False, "width_mm": 600, "height_mm": 720},
                {"width_mm": 500, "height_mm": 700},
            ],
            "upper_modules": [
                {"type": "hood_cabinet", "width_mm": 600, "height_mm": 800},
                {"width_mm": 800, "height_mm": 700},
                {"width_mm": 500, "height_mm": 700},
            ],
        }
    )[-1] == {"width_mm": 500, "height_mm": 700}
    items: list[dict] = []
    bom._add_item(items, "facade_sheet", None, 0, "sheet")
    assert items == []
    assert bom._asset_price({"price": 1234}, 10.0) == 1234.0
    assert bom._asset_price({"price": object()}, 77.0) == 77.0
    decor_total = bom._add_decor_items(
        items,
        {"decor_items": [{"type": "unknown"}, {"id": "vase", "type": "flowers_vase", "estimated_price": 3333}]},
        {"appliances": {"flowers_vase": {"chosen_asset": {"unique_key": "v", "title": "Vase", "price": "4 444,5"}}}},
    )
    assert decor_total == 4444.5
    assert items[-1]["sku"] == "v"

    selected_materials = {
        "materials": {
            "facade": {"chosen_material": {**facade, "dimensions": {"length_mm": 1200, "width_mm": 600}}},
            "body": {"chosen_material": material("board_sheet", "body", price=1500)},
            "countertop": {"chosen_material": material("countertop_slab", "counter", price=2500, pattern="stone")},
            "backsplash": {"chosen_material": material("backsplash_panel", "back", price=1200)},
            "edge_band": {"chosen_material": material("edge_band", "edge", price=100)},
            "wall_plinth": {"chosen_material": material("countertop_wall_plinth", "plinth", price=300)},
            "joint_profile": {"chosen_material": material("joint_profile", "joint", price=400)},
            "end_profile": {"chosen_material": material("end_profile", "end", price=500)},
        }
    }
    bom_report = bom.estimate_kitchen_bom(
        {
            "base_modules": [
                {"width_mm": 700, "height_mm": 720, "facade_layout": "three_drawers", "cutouts": ["sink", "cooktop"], "appliance": "oven"},
                {"width_mm": 500, "height_mm": 720, "facade_layout": "two_doors", "cutouts": ["entry_handwash"], "appliance": "dishwasher"},
                {"type": "fridge_slot", "width_mm": 600, "height_mm": 1800, "appliance": "fridge"},
            ],
            "upper_modules": [{"width_mm": 800, "height_mm": 700}],
            "countertop_segments": [{"width_mm": 6500}],
            "backsplash_segments": [{"width_mm": 3200}],
            "decor_items": [{"id": "small", "type": "small_kitchen_appliance"}],
        },
        selected_materials,
        include_appliance_estimate=True,
        appliance_assets={"appliances": {"small_kitchen_appliance": {"chosen_asset": {"price": "bad"}}}},
    )
    assert any(item["role"] == "joint_profile" for item in bom_report["items"])
    assert bom_report["estimates"]["appliance_estimate"] > 0
    assert bom_report["estimates"]["decor_accessory_estimate"] == 6500.0


def test_kitchen_layout_solver_edges(monkeypatch) -> None:
    monkeypatch.setitem(layout_solver.KITCHEN_DIMENSIONS_MM, "broken_dim", object())
    assert layout_solver._dim("broken_dim", 77) == 77
    assert layout_solver._as_int("bad", 9) == 9
    assert layout_solver._as_bool(None, True) is True
    assert layout_solver._as_bool(" да ") is True

    assert layout_solver._split_storage_width(250) == [250]
    assert layout_solver._split_storage_width(700)[-1] == 700
    assert layout_solver._split_storage_width(100) == [100]
    assert layout_solver._constraint_center({"sink": {"x_mm": "123"}}, "sink") == 123
    assert layout_solver._constraint_center({"sink": {}}, "sink") is None
    assert layout_solver._gap_between({"x_mm": 0, "width_mm": 100}, {"x_mm": 250, "width_mm": 100}) == 150
    assert layout_solver._gap_between({"x_mm": 300, "width_mm": 100}, {"x_mm": 50, "width_mm": 100}) == 150
    assert layout_solver._gap_between({"x_mm": 0, "width_mm": 200}, {"x_mm": 100, "width_mm": 200}) == -1

    warnings: list[str] = []
    constrained = layout_solver._build_constrained_modules(
        2400,
        {"sink": True, "cooktop": True, "oven": False, "fridge": False},
        {"sink": {"center_x_mm": 50}, "cooktop": {"center_x_mm": 2600}},
        warnings,
    )
    assert any(item.startswith("sink_constraint_ignored") for item in warnings)
    assert any(item.startswith("cooktop_constraint_ignored") for item in warnings)
    assert any(module["role"] == "sink" for module in constrained)
    assert any(module["role"] == "cooking" for module in constrained)

    fridge_warnings: list[str] = []
    layout_solver._build_constrained_modules(
        1600,
        {"sink": True, "cooktop": True, "oven": True, "fridge": True},
        {"sink": {"center_x_mm": 300}, "cooktop": {"center_x_mm": 1300}},
        fridge_warnings,
    )
    assert "removed_due_to_insufficient_width:fridge_slot" in fridge_warnings

    with pytest.raises(ValueError, match="sink_constraint_unmet"):
        layout_solver._build_constrained_modules(
            1500,
            {"sink": True, "cooktop": False},
            {"sink": {"center_x_mm": 50, "hard": True}},
            [],
        )
    with pytest.raises(ValueError, match="cooktop_constraint_unmet"):
        layout_solver._build_constrained_modules(
            1500,
            {"sink": True, "cooktop": True},
            {"cooktop": {"center_x_mm": 2200, "hard": True}},
            [],
        )
    with pytest.raises(ValueError, match="hard_functional_zone_overlap"):
        layout_solver._build_constrained_modules(
            2000,
            {"sink": True, "cooktop": True},
            {"sink": {"center_x_mm": 900}, "cooktop": {"center_x_mm": 1100}},
            [],
        )

    default_warnings: list[str] = []
    default_modules = layout_solver._build_default_modules(
        1500,
        {"sink": True, "cooktop": True, "fridge": True, "dishwasher": True},
        default_warnings,
    )
    assert {"removed_due_to_insufficient_width:fridge_slot", "removed_due_to_insufficient_width:dishwasher_slot", "removed_due_to_insufficient_width:cooktop"} <= set(default_warnings)
    assert any(module["type"] == "sink_cabinet" for module in default_modules)
    assert any(module["type"] == "filler" for module in layout_solver._build_default_modules(850, {"cooktop": False}, []))

    small_storage = {"id": "base_001", "type": "base_cabinet", "x_mm": 0, "width_mm": 400, "height_mm": 720, "depth_mm": 560, "cutouts": [], "has_countertop": True}
    sink = {"id": "base_002", "type": "sink_cabinet", "x_mm": 400, "width_mm": 600, "height_mm": 720, "depth_mm": 560, "cutouts": ["sink"], "has_countertop": True}
    upper, decor, upper_warnings = layout_solver._build_upper_modules([small_storage, sink], {"hood": True})
    assert all(module["type"] != "microwave_open_shelf" for module in upper)
    assert decor[0]["placement"] == "countertop"
    assert "microwave_placement:countertop" in upper_warnings
    assert layout_solver._build_countertop_accessories([small_storage], {"decor_accessories": False}) == []
    assert layout_solver._find_countertop_spot([sink], min_width_mm=260, preferred_y_mm=100, used_spots=[]) is None

    countertop_segments = layout_solver._build_countertop_segments([small_storage, sink])
    zones = layout_solver._build_functional_zones(
        [{"id": "fridge", "type": "fridge_slot", "role": "fridge", "x_mm": 0, "width_mm": 600, "depth_mm": 650, "height_mm": 1900}, sink],
        [{"id": "upper_001", "type": "microwave_open_shelf", "x_mm": 0, "width_mm": 600, "depth_mm": 320, "height_mm": 360}],
        [{"id": "decor_microwave_001", "type": "microwave", "placement": "countertop"}, {"id": "decor_flowers_vase_001", "type": "flowers_vase", "placement": "countertop"}],
    )
    assert {"fridge", "microwave", "flowers_vase"} <= {zone["role"] for zone in zones}
    openings = layout_solver._build_openings(
        [{"id": "fridge", "type": "fridge_slot", "x_mm": 0, "width_mm": 600, "depth_mm": 650, "height_mm": 1900}],
        [{"id": "upper_001", "type": "hood_wall_mounted", "x_mm": 600, "width_mm": 600, "depth_mm": 320, "height_mm": 360}, {"id": "upper_002", "type": "microwave_open_shelf", "x_mm": 0, "width_mm": 600, "depth_mm": 320, "height_mm": 360}],
        countertop_segments,
    )
    assert {"fridge_space", "hood_space", "microwave_niche", "sink_cutout"} <= {opening["type"] for opening in openings}
    breakdown = layout_solver._build_cabinet_breakdown([small_storage, {**sink, "has_facade": False}], upper)
    assert any(facade["tier"] == "upper" for facade in breakdown["facades"])
    targets = layout_solver._build_asset_targets(zones + [{"role": "cooking"}])
    assert {"fridge", "cooktop", "microwave", "flowers_vase"} <= {target["role"] for target in targets}

    solved = layout_solver.solve_kitchen_layout(
        {"layout_type": "corner", "available_width_mm": 2400, "constraints": {"sink": {"center_x_mm": 600}, "cooktop": {"center_x_mm": 1700}}},
        plumbing_point=None,
        entry_zone=None,
        required_appliances={"cooktop": True, "decor_accessories": False},
    )
    assert "layout_forced_to_straight:corner" in solved["warnings"]
    assert any(warning.startswith("sink_constraint_satisfied") for warning in solved["warnings"])
    microwave_only = layout_solver.solve_kitchen_layout(
        {"available_width_mm": 1500},
        plumbing_point=None,
        entry_zone=None,
        required_appliances={"cooktop": False, "decor_accessories": False},
    )
    assert "cooking_mode:microwave_only" in microwave_only["warnings"]


def test_appliance_matcher_catalog_helpers_and_selection(tmp_path: Path) -> None:
    fbx = tmp_path / "asset.fbx"
    fbx.write_bytes(b"Kaydara FBX Binary  \x00\x00")
    obj = tmp_path / "asset.obj"
    obj.write_text("o asset", encoding="utf-8")
    bad_fbx = tmp_path / "bad.fbx"
    bad_fbx.write_text("not binary", encoding="utf-8")

    assert app._is_importable_asset_path(fbx)
    assert app._is_importable_asset_path(obj)
    assert not app._is_importable_asset_path(bad_fbx)
    assert app._dims_cm({"dimensions_cm": {"width": 60, "depth": 50, "height": 20}}) == (60.0, 50.0, 20.0)
    assert app._asset_path({"asset_local_path": str(fbx)}) == str(fbx)

    items = [
        {"unique_key": "sink", "title": "Кухонная мойка черная", "category_norm": "kitchen_sink", "asset_local_path": str(fbx), "dimensions_cm": {"width": 56, "depth": 50, "height": 18}, "color": "черный", "price": 12000},
        {"unique_key": "faucet", "title": "Смеситель для кухни chrome", "category_norm": "kitchen_faucet", "asset_local_path": str(fbx), "dimensions_cm": {"width": 20, "depth": 25, "height": 35}, "price_value": 5000},
        {"unique_key": "cooktop", "title": "Индукционная варочная панель black", "category_norm": "cooktop_hob", "asset_local_path": str(obj), "dimensions_cm": {"width": 58, "depth": 52, "height": 5}},
        {"unique_key": "hood", "title": "Miele rangehood steel", "category_norm": "extractor_hood", "asset_local_path": str(fbx), "dimensions_cm": {"width": 60, "depth": 35, "height": 45}},
        {"unique_key": "fridge", "title": "Atlant refrigerator white", "category_norm": "refrigerator_freezer", "asset_local_path": str(fbx), "dimensions_cm": {"width": 60, "depth": 65, "height": 190}},
        {"unique_key": "microwave", "title": "Gorenje microwave", "category_norm": "microwave", "asset_local_path": str(fbx), "dimensions_cm": {"width": 45, "depth": 33, "height": 26}},
        {"unique_key": "small", "title": "Bosch coffee machine", "category_norm": "small_kitchen_appliance", "asset_local_path": str(fbx), "dimensions_cm": {"width": 24, "depth": 24, "height": 28}},
        {"unique_key": "vase", "title": "Flower vase bouquet", "category_norm": "plant_planter_vase", "asset_local_path": str(fbx), "dimensions_cm": {"width": 28, "depth": 28, "height": 48}},
        {"unique_key": "unavailable", "title": "Dishwasher white", "category_norm": "dishwasher", "dimensions_cm": {"width": 60, "depth": 56, "height": 85}},
    ]
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"items": items}), encoding="utf-8")
    assert len(app.load_supplier_catalog(catalog)) == len(items)
    assert app._category_score(items[0], app.APPLIANCE_TARGETS["sink"]) == 1.0
    assert app._prompt_preference_score(items[2], "cooktop", "индукционная панель") == 1.0
    assert app._dimension_score(items[0], app.APPLIANCE_TARGETS["sink"]["target_dims_cm"]) > 0.9
    assert not app._passes_min_dimensions({"dimensions_cm": {"width": 10, "depth": 10, "height": 10}}, app.APPLIANCE_TARGETS["fridge"])
    forbidden = {"unique_key": "combo", "title": "Мойка и смеситель", "category_norm": "kitchen_sink"}
    assert app._is_forbidden_for_role(forbidden, "faucet")

    layout = {
        "base_modules": [
            {"type": "fridge_slot", "appliance": "fridge"},
            {"type": "sink_cabinet", "cutouts": ["sink"]},
            {"type": "oven_cabinet", "cutouts": ["cooktop"], "appliance": "oven"},
        ],
        "decor_items": [{"type": "flowers_vase"}],
    }
    selected = app.select_kitchen_appliance_assets(
        catalog,
        layout,
        {"microwave": True, "hood": True},
        only_local_assets=True,
        user_prompt="white fridge induction cooktop",
    )
    assert {"sink", "faucet", "cooktop", "hood", "fridge", "microwave", "small_kitchen_appliance", "flowers_vase"} <= set(
        selected["appliances"]
    )
    assert "dishwasher" not in selected["appliances"]
    assert selected["unavailable_assets"].get("dishwasher") is None

    list_catalog = tmp_path / "catalog_list.json"
    list_catalog.write_text(json.dumps([items[0], "bad", items[1]]), encoding="utf-8")
    assert len(app.load_supplier_catalog(list_catalog)) == 2
    assert app._dims_cm({"width_cm": 10, "depth_cm": 0, "height_cm": "bad"}) == (10.0, None, None)
    assert app._asset_path({}) is None
    slashy_raw = str(obj).replace("/", "\\")
    assert app._asset_path({"asset_local_path": slashy_raw}) == str(obj)
    assert not app._is_importable_asset_path(tmp_path / "asset.txt")
    fbx_dir = tmp_path / "folder.fbx"
    fbx_dir.mkdir()
    assert not app._is_importable_asset_path(fbx_dir)
    assert app._color_score({"image_color_features": {"color_tokens": ["white"]}}, ("white",)) == 1.0
    assert app._prompt_preference_score(items[0], "sink", None) == 0.5
    assert app._prompt_preference_score({"unique_key": "hofmann_rf564cdbs", "title": "other"}, "fridge", "white") == 0.9
    assert app._prompt_preference_score({"unique_key": "aeg_s98392cmx2", "title": "other"}, "fridge", "white") == 0.42
    assert app._prompt_preference_score({"title": "Gaggenau refrigerator"}, "fridge", "dark") == 0.88
    assert app._prompt_preference_score({"title": "white refrigerator", "color": "white"}, "fridge", "white kitchen") == 1.0
    assert app._prompt_preference_score({"title": "stainless refrigerator", "color": "gray"}, "fridge", "white kitchen") == 0.82
    assert app._prompt_preference_score({"title": "black refrigerator", "color": "black"}, "fridge", "white kitchen") == 0.12
    assert app._prompt_preference_score({"title": "Teka hood"}, "hood", "modern") == 0.35
    assert app._prompt_preference_score({"title": "Florentina sink"}, "sink", "modern") == 0.95
    assert app._prompt_preference_score({"title": "Abber sink"}, "sink", "modern") == 0.72
    assert app._prompt_preference_score({"title": "Emar sink"}, "sink", "modern") == 0.55
    assert app._prompt_preference_score({"title": "gas hob"}, "cooktop", "induction") == 0.15
    assert app._prompt_preference_score({"title": "ceramic hob"}, "cooktop", "induction") == 0.45
    assert app._prompt_preference_score({"title": "gas hob"}, "cooktop", "gas") == 1.0
    assert app._prompt_preference_score({"title": "ceramic hob"}, "cooktop", "gas") == 0.35
    assert app._prompt_preference_score({"title": "electric hob"}, "cooktop", "electric") == 0.9
    assert app._prompt_preference_score({"title": "gas hob"}, "cooktop", "electric") == 0.3
    assert app._prompt_preference_score({"title": "plain hob"}, "cooktop", "modern") == 0.5
    assert app._dimension_score({"dimensions_cm": {"width": 0}}, (1, 1, 1)) == 0.45
    assert app._passes_min_dimensions({"dimensions_cm": {}}, app.APPLIANCE_TARGETS["fridge"]) is True
    assert app._is_forbidden_for_role({"unique_key": "zeelproject::id::2538", "title": "hood", "category_norm": "hood"}, "hood")
    assert app._is_forbidden_for_role(
        {"unique_key": "3ddd::url::https://3ddd.ru/3dmodels/show/moika_florentina_lipsi_460_chernyi_i_smesitel_florentina_vita_av", "title": "sink", "category_norm": "kitchen_sink"},
        "sink",
    )

    unavailable = app.select_kitchen_appliance_assets(
        [{"unique_key": "dish", "title": "Dishwasher white", "category_norm": "dishwasher", "dimensions_cm": {"width": 60, "depth": 56, "height": 85}}],
        {"base_modules": []},
        {"dishwasher": True},
        only_local_assets=True,
    )
    assert "dishwasher" in unavailable["unavailable_assets"]
    assert "no_appliance_asset_for_role:dishwasher" in unavailable["warnings"]

    no_local_needed = app.select_kitchen_appliance_assets(
        [{"unique_key": "dish", "title": "Dishwasher white", "category_norm": "dishwasher", "dimensions_cm": {"width": 60, "depth": 56, "height": 85}}],
        {"base_modules": []},
        {"dishwasher": True},
        only_local_assets=False,
    )
    assert no_local_needed["appliances"]["dishwasher"]["chosen_asset"]["asset_local_path"] is None

    prefer_fbx = app.select_kitchen_appliance_assets(
        [{"unique_key": "vase_obj", "title": "Flower vase bouquet", "category_norm": "plant_planter_vase", "asset_local_path": str(obj), "dimensions_cm": {"width": 28, "depth": 28, "height": 48}}],
        {"decor_items": [{"type": "flowers_vase"}]},
        {},
        only_local_assets=False,
    )
    assert "no_appliance_asset_for_role:flowers_vase" in prefer_fbx["warnings"]


def test_bom_and_assembly_json_convert_units_and_prices() -> None:
    selected = {
        "materials": {
            "facade": {"chosen_material": material("facade_sheet", "facade", price=8000), "top_candidates": [], "final_score": 0.8},
            "body": {"chosen_material": material("board_sheet", "body", price=5000), "top_candidates": []},
            "countertop": {"chosen_material": material("countertop_slab", "counter", price=9000), "top_candidates": []},
            "backsplash": {"chosen_material": material("backsplash_panel", "back", price=4500), "top_candidates": []},
            "edge_band": {"chosen_material": material("edge_band", "edge", price=50), "top_candidates": []},
            "wall_plinth": {"chosen_material": material("countertop_wall_plinth", "plinth", price=1500), "top_candidates": []},
            "joint_profile": {"chosen_material": material("joint_profile", "joint", price=900), "top_candidates": []},
            "end_profile": {"chosen_material": material("end_profile", "end", price=650), "top_candidates": []},
        },
        "palette_consistency_score": 0.9,
    }
    layout = {
        "layout_type": "straight",
        "layout_variant": "unit",
        "total_width_mm": 3000,
        "base_modules": [
            {"id": "sink", "type": "sink_cabinet", "x_mm": 0, "width_mm": 600, "height_mm": 720, "depth_mm": 560, "cutouts": ["sink"], "facade_layout": "two_doors"},
            {"id": "oven", "type": "oven_cabinet", "x_mm": 600, "width_mm": 600, "height_mm": 720, "depth_mm": 560, "cutouts": ["cooktop"], "appliance": "oven", "facade_layout": "oven_front"},
            {"id": "drawers", "type": "base_cabinet", "x_mm": 1200, "width_mm": 900, "height_mm": 720, "depth_mm": 560, "facade_layout": "three_drawers"},
        ],
        "upper_modules": [{"id": "hood", "type": "hood_cabinet", "x_mm": 600, "width_mm": 600, "height_mm": 360, "depth_mm": 320, "z_mm": 1500}],
        "countertop_segments": [{"x_mm": 0, "width_mm": 1500}, {"x_mm": 1500, "width_mm": 1500}],
        "backsplash_segments": [{"x_mm": 0, "width_mm": 3000}],
        "decor_items": [{"id": "decor1", "type": "flowers_vase", "placement": "countertop", "estimated_price": 1000}],
        "warnings": ["layout_warning"],
    }
    appliance_assets = {
        "appliances": {"flowers_vase": {"chosen_asset": {"unique_key": "vase1", "title": "Vase", "price": "2 500"}}},
        "warnings": ["asset_warning"],
    }

    assert bom._ceil_div(2.1, 1.0) == 3
    assert bom._asset_price({"price": "1 200,5"}, 10) == 1200.5
    bill = bom.estimate_kitchen_bom(layout, selected, mode="optimal", include_appliance_estimate=True, appliance_assets=appliance_assets)
    assert bill["computed_quantities"]["facade_panel_count"] >= 6
    assert bill["estimates"]["appliance_estimate"] > 0
    assert any(item["role"] == "flowers_vase" for item in bill["items"])

    converted = assembly._mm_to_m({"width_mm": 600, "nested": [{"height_mm": 720}]})
    assert converted == {"width_m": 0.6, "nested": [{"height_m": 0.72}]}
    assert assembly._material_summary(selected["materials"]["facade"]["chosen_material"])["sku"] == "facade"
    assembly_json = assembly.build_kitchen_assembly_json(
        target_id="kitchen",
        layout_plan=layout,
        selected_materials=selected,
        bill_of_materials=bill,
        design_spec={"style": {"primary": "modern"}},
        mode="optimal",
        appliance_assets=appliance_assets,
        position=[1, 2, 0],
        rotation=[0, 0, 90],
    )
    assert assembly_json["id"] == "kitchen"
    assert assembly_json["dimensions"]["width_m"] == 3.0
    assert assembly_json["base_modules"][0]["width_m"] == 0.6
    assert "layout_warning" in assembly_json["warnings"]
    assert "asset_warning" in assembly_json["warnings"]


def test_kitchen_supplier_inventory_buckets_summaries_and_cli(tmp_path: Path, capsys) -> None:
    local = tmp_path / "sink.glb"
    local.write_text("glb", encoding="utf-8")
    rows = [
        {
            "unique_key": "set",
            "title": "Modern kitchen set",
            "category_norm": "kitchen_set",
            "source_site": "unit",
            "asset_local_path": str(local),
            "dimensions_cm": {"width": 300, "depth": 60, "height": 220},
            "price": 100000,
        },
        {
            "unique_key": "sink",
            "title": "Кухонная мойка",
            "category_norm": "kitchen_sink",
            "source_site": "unit",
            "model_download_url": "https://example.test/sink.fbx",
            "width_cm": 55,
            "depth_cm": 45,
            "height_cm": 20,
            "price_value": 12000,
        },
        {
            "unique_key": "fruit",
            "title": "Fruit plate apples",
            "category_norm": "food_drink",
            "source_site": "decor",
        },
        {
            "unique_key": "chair",
            "title": "Dining chair",
            "category_norm": "chair",
            "source_site": "decor",
        },
        {
            "unique_key": "ordinary",
            "title": "Bedroom bed",
            "category_norm": "bed",
            "source_site": "decor",
        },
    ]
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"items": rows + ["bad"]}, ensure_ascii=False), encoding="utf-8")

    assert inventory._norm("Ёлка_big-chair") == "елка big chair"
    assert "kitchen_sinks" in inventory.classify_supplier_row(rows[1])
    assert "food_fruit" in inventory.classify_supplier_row(rows[2])
    assert inventory.supplier_row_key({"title": "Fallback"}) == "Fallback"
    assert inventory._has_local_asset(rows[0])
    assert inventory._has_downloadable_asset(rows[1])
    assert inventory._dimension_cm(rows[0], "width") == 300
    assert inventory._dimension_cm(rows[1], "width") == 55

    loaded = inventory.load_supplier_catalog(catalog)
    assert len(loaded) == len(rows)
    buckets = inventory.collect_kitchen_supplier_items(loaded)
    assert buckets["kitchen_sets"][0]["unique_key"] == "set"
    assert buckets["kitchen_sinks"][0]["unique_key"] == "sink"
    assert inventory.collect_by_category_norm(loaded)["bed"][0]["unique_key"] == "ordinary"

    index = inventory.build_kitchen_selection_index(loaded)
    assert len(index["ordinary_items"]) == 1
    assert index["kitchen_items"][0]["buckets"]

    compact = inventory.compact_item(rows[0])
    assert compact["width_cm"] == 300
    assert compact["price"] == 100000
    summary = inventory.build_inventory_summary(buckets)
    assert summary["buckets"]["kitchen_sets"]["local_asset_count"] == 1
    index_summary = inventory.build_selection_index_summary(index)
    assert index_summary["ordinary_count"] == 1
    inventory.print_summary(summary)
    inventory.print_selection_index_summary(index_summary, category_limit=1)
    printed = capsys.readouterr().out
    assert "kitchen_sets: count=1" in printed
    assert "ordinary=1" in printed

    out_json = tmp_path / "summary.json"
    out_csv = tmp_path / "summary.csv"
    code = inventory.main(
        [
            "--catalog",
            str(catalog),
            "--out-json",
            str(out_json),
            "--out-csv",
            str(out_csv),
            "--bucket",
            "kitchen_sets",
            "--limit",
            "1",
            "--ordinary-categories",
        ]
    )
    assert code == 0
    saved = json.loads(out_json.read_text(encoding="utf-8"))
    assert saved["source_row_count"] == len(rows)
    csv_text = out_csv.read_text(encoding="utf-8")
    assert "kitchen_sets,set" in csv_text
    assert "examples:kitchen_sets" in capsys.readouterr().out


def test_kitchen_pipeline_generate_variants_and_zone_helpers(monkeypatch) -> None:
    calls: dict[str, object] = {}
    materials = [material("facade_sheet", "facade")]
    layout = {
        "layout_type": "straight",
        "total_width_mm": 2400,
        "base_modules": [],
        "upper_modules": [],
        "countertop_segments": [],
        "backsplash_segments": [],
        "decor_items": [],
    }
    selected = {"materials": {"facade": {"chosen_material": materials[0], "top_candidates": []}}, "warnings": []}
    appliances = {"appliances": {"sink": {"chosen_asset": {"unique_key": "sink"}}}, "warnings": ["asset"]}
    bom_report = {"items": [], "totals_by_currency": {}, "computed_quantities": {}, "estimates": {}}

    monkeypatch.setattr(kitchen_pipeline, "load_kitchen_material_catalog", lambda catalog: materials)
    monkeypatch.setattr(kitchen_pipeline, "build_kitchen_design_spec", lambda **kwargs: {"style": {"primary": "modern"}, "functional_requirements": dict(kwargs["appliances"])})
    monkeypatch.setattr(kitchen_pipeline, "infer_prompt_preferences_with_llm", lambda **kwargs: {"colors": ["white"], "appliance_hints": {"sink": "steel"}})
    monkeypatch.setattr(kitchen_pipeline, "apply_llm_preferences_to_design_spec", lambda spec, prefs: {**spec, "llm_preferences": prefs})
    monkeypatch.setattr(kitchen_pipeline, "append_appliance_hints_to_prompt", lambda prompt, prefs: prompt + " sink steel")
    monkeypatch.setattr(kitchen_pipeline, "solve_kitchen_layout", lambda *args, **kwargs: layout)
    monkeypatch.setattr(kitchen_pipeline, "select_kitchen_appliance_assets", lambda **kwargs: appliances)
    monkeypatch.setattr(kitchen_pipeline, "rerank_appliance_assets_with_llm", lambda **kwargs: kwargs["appliance_assets"])
    monkeypatch.setattr(kitchen_pipeline, "select_kitchen_materials", lambda materials, design_spec, layout_plan, mode, top_n: {**selected, "mode": mode})
    monkeypatch.setattr(kitchen_pipeline, "rerank_material_bindings_with_llm", lambda **kwargs: kwargs["selected_materials"])
    monkeypatch.setattr(kitchen_pipeline, "estimate_kitchen_bom", lambda *args, **kwargs: bom_report)

    def fake_assembly(**kwargs):
        calls[kwargs["mode"]] = kwargs
        return {"id": kwargs["target_id"], "mode": kwargs["mode"], "position": kwargs["position"], "rotation": kwargs["rotation"]}

    monkeypatch.setattr(kitchen_pipeline, "build_kitchen_assembly_json", fake_assembly)
    variants = kitchen_pipeline.generate_kitchen_variants(
        material_catalog="catalog.json",
        user_prompt="white kitchen",
        room={"id": "room"},
        kitchen_zone={"available_width_mm": 2400},
        entry_zone={"has_entry_handwash": True},
        appliance_catalog=[{"unique_key": "sink"}],
        modes=["optimal", "cheapest"],
        target_id="kit",
        position=[1, 2, 0],
        rotation=[0, 0, 90],
        llm_settings={"provider": "none"},
    )
    assert set(variants) == {"optimal", "cheapest"}
    assert variants["optimal"]["id"] == "kit_optimal"
    assert calls["optimal"]["design_spec"]["functional_requirements"]["entry_handwash"] is True
    assert calls["optimal"]["appliance_assets"] is appliances
    no_appliance_catalog = kitchen_pipeline.generate_kitchen_variants(
        material_catalog=materials,
        user_prompt="white kitchen",
        room={"id": "room"},
        kitchen_zone={"available_width_mm": 2400},
        appliance_catalog=None,
        modes=["optimal"],
    )
    assert no_appliance_catalog["optimal"]["mode"] == "optimal"

    assert kitchen_pipeline.is_kitchen_target({"category": "kitchen_set"})
    assert kitchen_pipeline.is_kitchen_target({"type": "kitchen_assembly"})
    assert not kitchen_pipeline.is_kitchen_target({"category": "bed"})
    assert kitchen_pipeline.build_kitchen_zone_from_target({"size_m": [2.4, 0.6, 2.1]}, {"default_kitchen_wall_id": "w0"}) == {
        "layout_type": "straight",
        "wall_id": "w0",
        "available_width_mm": 2400,
        "depth_mm": 600,
        "start_x_mm": 0,
        "end_x_mm": 2400,
    }
    assert kitchen_pipeline.build_kitchen_zone_from_target({"kitchen_width_m": "bad"})["available_width_mm"] == 3000
