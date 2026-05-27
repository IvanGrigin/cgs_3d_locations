from __future__ import annotations

import json
import sqlite3
import sys
import types
from pathlib import Path

import pytest

from src.suppliers import room_design_spec_builder as spec_builder
from src.suppliers import runner
from src.suppliers import db as supplier_db
from src.suppliers import registry
from src.suppliers import supplier_identity_gates as gates
from src.suppliers import supplier_scene_consistency as consistency
from src.suppliers import supplier_variant_validator as validator
from src.suppliers import utils as supplier_utils
from src.suppliers.models import ProductRecord


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _product(**overrides) -> ProductRecord:
    data = {
        "unique_key": "site::1",
        "source_site": "unit",
        "source_url": "https://example.test/item",
        "parsed_at": "2026-01-01T00:00:00Z",
        "external_id": "1",
        "title": "Unit product",
    }
    data.update(overrides)
    return ProductRecord(**data)


def test_supplier_utils_json_sqlite_and_timestamp() -> None:
    assert supplier_utils.json_loads_or("", {"fallback": True}) == {"fallback": True}
    assert supplier_utils.json_loads_or("{bad", ["fallback"]) == ["fallback"]
    assert supplier_utils.json_loads_or('{"ok": true}', {}) == {"ok": True}
    assert supplier_utils.now_utc_iso().endswith("+00:00")

    con = sqlite3.connect(":memory:")
    try:
        assert supplier_utils.sqlite_table_exists(con, "items") is False
        con.execute("CREATE TABLE items(id INTEGER)")
        assert supplier_utils.sqlite_table_exists(con, "items") is True
    finally:
        con.close()


def test_supplier_db_migration_and_registry_dispatch(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.sqlite"
    with sqlite3.connect(db_path) as con:
        con.execute("CREATE TABLE supplier_product (unique_key TEXT NOT NULL UNIQUE)")
        supplier_db._ensure_table_columns(con, "supplier_product", {"volume_m3": "REAL", "scheme_url": "TEXT"})
        columns = {row[1] for row in con.execute("PRAGMA table_info(supplier_product)").fetchall()}
    assert {"volume_m3", "scheme_url"} <= columns

    adapters = registry.build_adapters()
    assert adapters
    assert registry.find_adapter("https://homeconcept.ru/catalog/unit").site_name
    with pytest.raises(ValueError):
        registry.get_adapter_for_url("https://unsupported.example.test/product")


def test_supplier_variant_validator_reports_errors_warnings_and_cross_mode(tmp_path, monkeypatch):
    optimal = _write_json(
        tmp_path / "bindings.optimal.assets.json",
        {
            "meta": {"supplier_selection_mode": "optimal", "room_design_spec_enabled": True, "asset_acquisition": True},
            "bindings": [
                {
                    "target_id": "bed",
                    "semantic_group": "bed",
                    "selection_status": "heuristic_top1_selected",
                    "chosen_candidate": {"unique_key": "bad-bed", "score_breakdown": []},
                    "top_candidates": [{"unique_key": "bad-bed"}],
                    "consistency_group_id": "bedroom_beds",
                },
                {
                    "target_id": "bed",
                    "selection_status": "no_real_asset_after_acquisition",
                    "chosen_candidate": {"unique_key": "still-present"},
                },
                "bad-binding",
            ],
        },
    )
    cheapest = _write_json(
        tmp_path / "bindings.cheapest.json",
        {
            "meta": {"supplier_selection_mode": "cheapest"},
            "bindings": [
                {
                    "target_id": "chair",
                    "semantic_group": "chair",
                    "selection_status": "heuristic_top1_selected",
                    "chosen_candidate": {"unique_key": "chair-ok", "asset_local_path": str(tmp_path / "chair.glb")},
                    "top_candidates": [{"unique_key": "chair-ok", "score_breakdown": {}}],
                }
            ],
        },
    )
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("[]", encoding="utf-8")

    monkeypatch.setattr(
        validator,
        "candidate_identity_gate",
        lambda binding, candidate: (
            False,
            {"identity_target_group": binding.get("semantic_group"), "identity_reject_reason": "unit_reject"},
        ),
    )

    item, data = validator._validate_binding_file(optimal)
    assert data is not None
    assert item["mode"] == "optimal"
    assert any("duplicate target_id" in error for error in item["errors"])
    assert any("no final_score" in warning for warning in item["warnings"])
    assert validator._mode_from_path(cheapest) == "cheapest"
    assert validator._candidate_id({"sku": "SKU1"}) == "SKU1"
    assert validator._has_local_asset({"asset_local_path": "x.glb"})

    bad_item, bad_data = validator._validate_binding_file(bad_json)
    assert bad_data is None
    assert bad_item["errors"] == ["cannot read JSON: root JSON is not an object"]

    out = tmp_path / "validation.json"
    code = validator.main(["--bindings", str(optimal), "--bindings", str(cheapest), "--out", str(out)])
    report = json.loads(out.read_text(encoding="utf-8"))
    assert code == 1
    assert report["ok"] is False
    assert report["cross_mode"]["target_ids_same"] is False
    assert "Target coverage differs across modes." in report["warnings"]


def test_room_design_spec_builder_cli_and_palette_helpers(tmp_path, monkeypatch, capsys):
    layout = {
        "room": {"id": "bedroom_1", "room_type": "bedroom", "area_m2": 18, "style_hint": "Japandi sage oak"},
        "targets": [
            {"target_id": "bed", "name": "Bed", "semantic_group": "bed", "size_m": [1.6, 2.0, 0.9], "replacement_policy": "replace_with_supplier"},
            {"target_id": "lamp", "name": "Lamp", "semantic_group": "lamp_table", "size_m": ["bad"], "replacement_policy": "keep_generated"},
        ],
    }
    layout_path = _write_json(tmp_path / "layout.json", layout)
    style_path = _write_json(tmp_path / "style.json", {"style_label": "scandi", "preferred_colors": ["cream"], "material_family": ["oak", "linen"]})
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("soft classic bedroom with green oak, avoid red and purple", encoding="utf-8")

    assert spec_builder._normalize_style("loft") == "loft_industrial"
    assert spec_builder._infer_style("слегка класс", {}, {}) == "soft_classic"
    assert spec_builder._infer_room_type("kitchen", {}, {}) == "kitchen"
    positive, negative = spec_builder._split_prompt_palette_segments("green sofa avoid red")
    assert "green sofa" in positive and "red" in negative
    palette = spec_builder._palette_from_inputs("green oak avoid red", {}, {}, spec_builder.STYLE_DEFAULTS["modern"])
    assert "green" in palette["preferred_colors"]
    assert "red" in palette["forbidden_colors"]
    assert spec_builder._target_summary({"target_id": "x", "size_m": ["bad"]})["size_m"] == [0.0, 0.0, 0.0]

    spec = spec_builder.build_room_design_spec(user_prompt=prompt_file.read_text(), layout_targets=layout, style_profile={"style_label": "japandi"})
    assert spec["schema"] == "room_design_spec/v1"
    assert spec["room_type"] == "bedroom"
    assert spec["object_requirements"]["bed"]["target_count"] == 1

    out = tmp_path / "spec.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "room_design_spec_builder",
            "--prompt-file",
            str(prompt_file),
            "--layout-targets",
            str(layout_path),
            "--style-profile",
            str(style_path),
            "--out",
            str(out),
        ],
    )
    assert spec_builder.main() == 0
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["style"]["primary"] == "soft_classic"
    assert "saved =" in capsys.readouterr().out


def test_suppliers_runner_coercion_main_success_skip_and_error(tmp_path, monkeypatch, capsys):
    class Adapter:
        site_name = "unit"
        empty_parse_is_skip = False

        def build_unique_key(self, url, _external_id):
            return f"unit::{url}"

        def now_utc_iso(self):
            return "2026-01-01T00:00:00Z"

        def filename_from_url(self, url):
            return Path(str(url or "")).name or None

        def ext_from_url(self, url):
            return Path(str(url or "")).suffix or None

        def fetch_html(self, url):
            return "<html>", url + "/final"

        def parse(self, source_url, html, final_url):
            assert source_url and html and final_url
            return [
                {
                    "title": "Chair",
                    "product_url": "https://example.test/chair",
                    "model_download_url": "https://example.test/chair.glb",
                    "price_value": 10,
                    "width_m": 0.5,
                    "depth_m": 0.6,
                    "height_m": 0.9,
                    "materials": ["wood", "fabric"],
                    "room_tags": ["bedroom", ""],
                    "preview_images": ["img.jpg"],
                    "attrs": {"old": 1},
                    "category": "chair",
                }
            ]

    adapter = Adapter()
    coerced = runner.coerce_product_record(adapter.parse("u", "h", "f")[0], adapter, "u", "f")
    assert coerced.width_cm == 50.0
    assert coerced.materials == "wood; fabric"
    assert json.loads(coerced.extra_json)["category"] == "chair"
    assert runner.coerce_product_record(coerced, adapter, "u", "f") is coerced
    with pytest.raises(TypeError):
        runner.coerce_product_record(object(), adapter, "u", "f")
    assert runner._metadata_slug(_product(title="Товар / test", external_id="id 1")) == "Товар_test__id_1"
    assert runner._metadata_slug(_product(title="!!!", unique_key="???", external_id="")) == "product"
    assert runner._to_cm("1.2") == 120.0
    assert runner._to_cm(None) is None
    assert runner._to_cm("bad") is None
    assert runner._coerce_materials([]) is None
    assert runner._coerce_materials(None) is None
    assert runner._coerce_materials("  ") is None
    assert runner._json_dumps(None, []) == "[]"

    calls = {"init": 0, "upsert": 0, "logs": []}
    monkeypatch.setattr(runner, "init_db", lambda db_path: calls.__setitem__("init", calls["init"] + 1))
    monkeypatch.setattr(runner, "upsert_products", lambda db_path, products: calls.__setitem__("upsert", len(products)))
    monkeypatch.setattr(runner, "insert_fetch_log", lambda **kwargs: calls["logs"].append(kwargs))
    monkeypatch.setattr(runner, "find_adapter", lambda url: adapter)
    monkeypatch.setattr(sys, "argv", ["runner", "--url", "https://example.test/chair", "--db", str(tmp_path / "db.sqlite"), "--out-dir", str(tmp_path / "items")])
    runner.main()
    assert calls["init"] == 1
    assert calls["upsert"] == 1
    assert calls["logs"][-1]["ok"] is True
    assert "records: 1" in capsys.readouterr().out

    class ManyAdapter(Adapter):
        def parse(self, source_url, html, final_url):
            base = Adapter.parse(self, source_url, html, final_url)[0]
            return [dict(base, title=f"Chair {idx}") for idx in range(11)]

    monkeypatch.setattr(runner, "find_adapter", lambda url: ManyAdapter())
    runner.main()
    assert calls["upsert"] == 11
    assert "more_records: 1" in capsys.readouterr().out

    class EmptyAdapter(Adapter):
        empty_parse_is_skip = True

        def parse(self, source_url, html, final_url):
            return []

    monkeypatch.setattr(runner, "find_adapter", lambda url: EmptyAdapter())
    runner.main()
    assert calls["logs"][-1]["error"] == "skip: empty adapter result"
    assert "status: skipped_empty_result" in capsys.readouterr().out

    class EmptyErrorAdapter(Adapter):
        empty_parse_is_skip = False

        def parse(self, source_url, html, final_url):
            return []

    monkeypatch.setattr(runner, "find_adapter", lambda url: EmptyErrorAdapter())
    with pytest.raises(ValueError, match="не вернул"):
        runner.main()

    class FailingAdapter(Adapter):
        def parse(self, source_url, html, final_url):
            raise ValueError("parse failed")

    monkeypatch.setattr(runner, "find_adapter", lambda url: FailingAdapter())
    with pytest.raises(ValueError, match="parse failed"):
        runner.main()
    assert calls["logs"][-1]["ok"] is False
    assert "ValueError: parse failed" in calls["logs"][-1]["error"]


def test_supplier_scene_consistency_groups_similar_repeats_and_cli(tmp_path, monkeypatch, capsys):
    first_chair = {
        "target_id": "chair_left",
        "semantic_group": "chair",
        "selection_status": "heuristic_top1_selected",
        "requested_size_m": [0.5, 0.55, 0.9],
        "chosen_candidate": {"unique_key": "chair-a", "final_score": 0.4},
        "top_candidates": [
            {"unique_key": "chair-b", "final_score": 0.7, "asset_local_path": "/tmp/chair.glb"},
            {"unique_key": "chair-c", "final_score": 0.9, "model_download_url": "https://example.test/c.fbx"},
        ],
    }
    second_chair = {
        "target_id": "chair_right",
        "semantic_group": "chair",
        "selection_status": "llm_reranked_top1_selected",
        "requested_size_m": [0.56, 0.57, 0.95],
        "chosen_candidate": {"unique_key": "chair-c", "final_score": 0.9, "model_download_url": "https://example.test/c.fbx"},
        "top_candidates": [{"unique_key": "chair-b", "final_score": 0.6, "asset_local_path": "/tmp/chair.glb"}],
    }
    oversized_chair = {
        "target_id": "chair_big",
        "semantic_group": "chair",
        "selection_status": "heuristic_top1_selected",
        "requested_size_m": [1.2, 1.2, 1.2],
        "chosen_candidate": {"unique_key": "chair-big", "asset_local_path": "/tmp/big.glb"},
    }
    untouched_bed = {
        "target_id": "bed",
        "semantic_group": "bed",
        "selection_status": "heuristic_top1_selected",
        "requested_size_m": [1.6, 2.0, 1.0],
        "chosen_candidate": {"unique_key": "bed-a", "asset_local_path": "/tmp/bed.glb"},
    }
    payload = {"meta": {"source": "unit"}, "bindings": [first_chair, second_chair, oversized_chair, untouched_bed, "bad"]}

    assert consistency._size_tuple({"requested_size_m": ["bad"]}) is None
    assert consistency._sizes_similar(None, (1.0, 1.0, 1.0))
    assert not consistency._sizes_similar((0.5, 0.5, 0.5), (0.8, 0.5, 0.5))
    assert consistency._candidate_asset_rank({"asset_local_path": "x.glb"}) == 0
    assert consistency._candidate_asset_rank({"model_page_url": "https://example.test"}) == 1
    assert consistency._candidate_score({"final_score": {"bad": True}, "score_breakdown": {"design_score": "0.75"}}) == 0.75
    assert consistency._group_key({"semantic_group": "bed"}) is None
    assert consistency._selected({"selection_status": "no_match", "chosen_candidate": {}}) is False

    result = consistency.apply_supplier_scene_consistency(payload)
    bindings = {item["target_id"]: item for item in result["bindings"] if isinstance(item, dict)}
    assert bindings["chair_left"]["chosen_candidate"]["unique_key"] == "chair-b"
    assert bindings["chair_right"]["chosen_candidate"]["unique_key"] == "chair-b"
    assert bindings["chair_big"]["chosen_candidate"]["unique_key"] == "chair-big"
    assert bindings["bed"]["chosen_candidate"]["unique_key"] == "bed-a"
    assert result["meta"]["scene_consistency"]["applied_group_count"] == 1
    assert "scene_consistency_shared_candidate:chair-b" in bindings["chair_left"]["selection_notes"]

    input_path = _write_json(tmp_path / "bindings.json", payload)
    out_path = tmp_path / "bindings.consistent.json"
    monkeypatch.setattr(sys, "argv", ["supplier_scene_consistency", "--bindings-json", str(input_path), "--out", str(out_path)])
    assert consistency.main() == 0
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["meta"]["scene_consistency"]["applied_groups"][0]["shared_unique_key"] == "chair-b"
    assert '"applied_group_count": 1' in capsys.readouterr().out


def test_supplier_identity_gates_allow_require_and_reject_terms():
    assert gates.normalize_text("Ёлка/Bed-Frame") == "елка bed frame"
    assert gates._tokens("Oak bed_frame") == ["oak", "bed", "frame"]
    assert gates._hit_phrases(" oak bed frame ", ["bed", "coffee table"]) == ["bed"]
    assert gates._target_group({"semantic_group": "nightstand"}) == "nightstand"
    assert gates._target_group({"name": "floor lamp"}) == "lamp_floor"
    assert gates._target_group({"category": "раковина"}) == "bathroom_sink"

    passed, info = gates.candidate_identity_gate({"semantic_group": "bed"}, {"title": "Solid oak bed frame with headboard"})
    assert passed is True
    assert info["identity_gate_checked"] is True
    assert "bed" in info["identity_required_hits"]

    passed, info = gates.candidate_identity_gate({"semantic_group": "bed"}, {"title": "Dining table cabinet"})
    assert passed is False
    assert info["identity_reject_reason"].startswith("identity_forbidden_terms")
    assert "table" in info["identity_forbidden_hits"]

    passed, info = gates.candidate_identity_gate({"semantic_group": "wardrobe"}, {"title": "Generic storage object"})
    assert passed is False
    assert info["identity_reject_reason"] == "identity_required_terms_missing"

    passed, info = gates.candidate_identity_gate({"semantic_group": "tv_projector_screen"}, {"title": "Computer monitor 27 inch"})
    assert passed is False
    assert info["identity_target_group"] == "tv"
    assert "monitor" in info["identity_forbidden_hits"]

    passed, info = gates.candidate_identity_gate({"semantic_group": "computer_monitor"}, {"title": "Smart TV display"})
    assert passed is False
    assert info["identity_target_group"] == "computer"
    assert "tv" in info["identity_forbidden_hits"]

    passed, info = gates.candidate_identity_gate({"semantic_group": "custom_decor"}, {"title": "anything"})
    assert passed is True
    assert info["identity_gate_checked"] is False
