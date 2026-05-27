from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from src import apply_supplier_bindings as apply_bindings
from src import supplier_layout_matcher as matcher
from src.tools import run_procedural_room_supplier as procedural_supplier
from tests.helpers.supplier_postprocess import patch_apply_postprocess


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _room_payload(room_name: str = "Bedroom", *, include_window: bool = True) -> dict:
    return {
        "version": "1.0",
        "units": "m",
        "room": {
            "id": "room_001",
            "name": room_name,
            "ceiling_height": 2.8,
            "floor_polygon": [
                {"x": 0.0, "y": 0.0},
                {"x": 4.8, "y": 0.0},
                {"x": 4.8, "y": 3.8},
                {"x": 0.0, "y": 3.8},
            ],
            "walls": [
                {"id": "w0", "from_vertex": 0, "to_vertex": 1},
                {"id": "w1", "from_vertex": 1, "to_vertex": 2},
                {"id": "w2", "from_vertex": 2, "to_vertex": 3},
                {"id": "w3", "from_vertex": 3, "to_vertex": 0},
            ],
            "doors": [
                {
                    "id": "door_0",
                    "wall_id": "w0",
                    "s": 0.8,
                    "width": 0.9,
                    "z0": 0.0,
                    "height": 2.05,
                }
            ],
            "windows": (
                [
                    {
                        "id": "win_0",
                        "wall_id": "w2",
                        "s": 1.6,
                        "width": 1.2,
                        "z0": 0.9,
                        "height": 1.1,
                    }
                ]
                if include_window
                else []
            ),
        },
    }


def _catalog_item(asset: Path, *, group: str = "chair", title: str = "Integration Chair") -> dict:
    return {
        "unique_key": f"integration::{group}",
        "source_site": "integration",
        "category_norm": group,
        "semantic_group": group,
        "title": title,
        "brand": "Test Brand",
        "product_url": "https://example.test/product",
        "model_page_url": "https://example.test/model",
        "price_value": 1234.0,
        "price_currency": "RUB",
        "width_cm": 50,
        "depth_cm": 55,
        "height_cm": 90,
        "asset_status": "local_supplier_asset",
        "asset_format": asset.suffix.lstrip("."),
        "asset_local_path": str(asset),
        "images": ["https://example.test/image.jpg"],
        "description": "A simple modern chair for integration tests.",
    }


@pytest.mark.integration
@pytest.mark.e2e
def test_matcher_to_apply_bindings_cli_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    patch_apply_postprocess(monkeypatch, apply_bindings)
    asset = tmp_path / "assets" / "chair.glb"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"glb")
    targets = _write_json(
        tmp_path / "layout_targets.json",
        {
            "schema": "layout_targets/v1",
            "source_json": str(tmp_path / "scene.input.json"),
            "targets": [
                {
                    "target_id": "chair_1",
                    "name": "Desk chair",
                    "category": "chair",
                    "semantic_group": "chair",
                    "size_m": [0.48, 0.52, 0.86],
                    "replacement_policy": "replace_with_supplier",
                    "replacement_reason": "large_furniture",
                    "constraints": {"mount_type": "floor"},
                }
            ],
        },
    )
    catalog = _write_json(tmp_path / "supplier_catalog.json", {"items": [_catalog_item(asset)]})
    bindings = tmp_path / "supplier_bindings.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "supplier_layout_matcher.py",
            "--targets",
            str(targets),
            "--supplier-json",
            str(catalog),
            "--selection-mode",
            "optimal",
            "--selection-strategy",
            "balanced",
            "--top-k",
            "3",
            "--llm-provider",
            "none",
            "--out",
            str(bindings),
        ],
    )
    matcher.main()
    matcher_stdout = capsys.readouterr().out
    assert "matched_target_count = 1" in matcher_stdout

    scene = _write_json(
        tmp_path / "scene.input.json",
        {
            "schema": "scene.v1",
            "room": _room_payload()["room"],
            "placements": [
                {
                    "id": "chair_1",
                    "name": "Generated chair",
                    "category": "chair",
                    "position_m": [2.0, 2.0, 0.43],
                    "size_m": [0.48, 0.52, 0.86],
                    "aabb": {"x_min": 1.76, "x_max": 2.24, "y_min": 1.74, "y_max": 2.26, "z_min": 0.0, "z_max": 0.86},
                    "source": {"placement_source": "procedural_room_stage"},
                    "meta": {"procedural": True},
                }
            ],
        },
    )
    output_scene = tmp_path / "scene.output.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "apply_supplier_bindings.py",
            "--input-json",
            str(scene),
            "--bindings-json",
            str(bindings),
            "--out",
            str(output_scene),
            "--require-local-asset",
        ],
    )
    apply_bindings.main()
    apply_stdout = capsys.readouterr().out
    assert "replaced = 1" in apply_stdout

    applied = json.loads(output_scene.read_text(encoding="utf-8"))
    [chair] = applied["placements"]
    assert chair["name"] == "Integration Chair"
    assert chair["asset"]["mesh_path"] == str(asset.resolve())
    assert chair["meta"]["supplier_binding_applied"] is True
    assert applied["meta"]["supplier_binding_summary"]["local_asset_replaced_count"] == 1


@pytest.mark.integration
def test_procedural_room_supplier_cli_report_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    patch_apply_postprocess(monkeypatch, apply_bindings)
    room = _write_json(tmp_path / "room.json", _room_payload(include_window=False))
    out_dir = tmp_path / "out"
    asset = tmp_path / "assets" / "selected.glb"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"glb")
    catalog = _write_json(tmp_path / "catalog.json", {"items": [_catalog_item(asset, group="bed", title="Integration Bed")]})

    def fake_load_supplier_catalog_json(_paths, sites=None, rich_only=False):
        return [_catalog_item(asset, group="bed", title="Integration Bed")]

    def fake_build_bindings_with_candidates(
        *,
        targets_json_path,
        catalog_rows,
        top_k,
        selection_strategy,
        user_preferences,
        llm_settings,
        room_design_spec,
        selection_mode,
    ):
        targets = json.loads(Path(targets_json_path).read_text(encoding="utf-8"))["targets"]
        replaceable = next(
            target
            for target in targets
            if target.get("replacement_policy") == "replace_with_supplier"
        )
        candidate = dict(catalog_rows[0])
        candidate["semantic_group"] = replaceable.get("semantic_group") or candidate["semantic_group"]
        candidate["category_norm"] = replaceable.get("semantic_group") or candidate["category_norm"]
        candidate["width_cm"] = 180
        candidate["depth_cm"] = 210
        candidate["height_cm"] = 90
        return {
            "schema": "supplier_bindings/v1",
            "meta": {
                "matched_target_count": 1,
                "target_count": len(targets),
                "selection_mode": selection_mode,
                "selection_strategy": selection_strategy,
            },
            "bindings": [
                {
                    "target_id": replaceable["target_id"],
                    "category": replaceable.get("category"),
                    "semantic_group": replaceable.get("semantic_group"),
                    "requested_size_m": replaceable.get("size_m"),
                    "replacement_policy": "replace_with_supplier",
                    "selection_status": "heuristic_top1_selected",
                    "provenance": {"final_asset_source": "supplier_catalog"},
                    "candidate_count": 1,
                    "top_candidates": [candidate],
                    "chosen_candidate": candidate,
                    "selection_notes": ["integration_smoke_selected"],
                }
            ],
        }

    monkeypatch.setattr(procedural_supplier, "load_supplier_catalog_json", fake_load_supplier_catalog_json)
    monkeypatch.setattr(procedural_supplier, "build_bindings_with_candidates", fake_build_bindings_with_candidates)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_procedural_room_supplier.py",
            "--room",
            str(room),
            "--out-dir",
            str(out_dir),
            "--prompt",
            "modern bedroom with bed and chair",
            "--density",
            "normal",
            "--policy",
            "always",
            "--seed",
            "7",
            "--supplier-catalog-json",
            str(catalog),
            "--supplier-selection-mode",
            "optimal",
            "--supplier-selection-strategy",
            "balanced",
            "--top-k",
            "3",
            "--supplier-llm-provider",
            "none",
            "--no-flooring",
            "--no-wall-material",
            "--no-curtains",
            "--no-stage-timings",
        ],
    )

    procedural_supplier.main()
    stdout = capsys.readouterr().out
    report = json.loads(stdout)

    assert report["schema"] == "procedural_room_supplier_report/v1"
    assert report["summary"]["target_count"] >= 1
    assert report["summary"]["matched_target_count"] == 1
    assert report["summary"]["local_asset_replaced"] >= 1
    assert report["validation"]["accessibility_ok"] is True
    assert Path(report["supplier_scene_json"]).is_file()
    assert Path(report["cost_report"]["json"]).is_file()
