from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src import acquire_supplier_bindings_assets as acquire
from src import layout_targets


def test_layout_targets_artifacts_cover_replace_keep_and_error_paths(tmp_path: Path) -> None:
    scene_path = tmp_path / "scene.json"
    scene = {
        "schema": "scene.v1",
        "room": {"id": "room1", "room_type": "bedroom"},
        "meta": {"placer": "unit", "mode": "test"},
        "placements": [
            {
                "id": "bed1",
                "name": "LargeBed",
                "category": "bed",
                "position_m": [1.0, 1.5, 0.3],
                "size_m": [1.8, 2.0, 0.6],
                "yaw_deg": 90,
                "meta": {"physical_role": "solid_floor"},
                "source": {"placement_source": "procedural_room_stage"},
            },
            {
                "id": "rug1",
                "name": "runner_rug",
                "category": "runner_rug",
                "aabb": {"x_min": 0, "x_max": 2, "y_min": 0, "y_max": 1, "z_min": 0, "z_max": 0.02},
                "meta": {"physical_role": "soft_floor"},
            },
            {
                "id": "lamp1",
                "name": "CeilingLight",
                "category": "ceiling_light",
                "constraints": {"mount_type": "ceiling"},
                "source": {"placeholder_bbox": True},
            },
            "ignored",
        ],
    }
    scene_path.write_text(json.dumps(scene), encoding="utf-8")

    artifacts = layout_targets.create_layout_selection_stub_artifacts(
        source_json_path=scene_path,
        run_dir=tmp_path,
        prefix="unit",
    )

    targets = json.loads(Path(artifacts["layout_targets_json"]).read_text(encoding="utf-8"))
    assert targets["meta"]["target_count"] == 3
    assert targets["meta"]["placeholder_bbox_count"] == 1
    by_id = {target["target_id"]: target for target in targets["targets"]}
    assert by_id["bed1"]["semantic_group"] == "bed"
    assert by_id["bed1"]["replacement_policy"] == "replace_with_supplier"
    assert by_id["rug1"]["replacement_policy"] == "keep_generated"
    assert by_id["lamp1"]["replacement_reason"] == "placeholder_bbox_requires_real_asset"
    assert by_id["rug1"]["position_m"] == [1.0, 0.5, 0.01]
    assert by_id["rug1"]["size_m"] == [2.0, 1.0, 0.02]

    bindings = json.loads(Path(artifacts["supplier_bindings_stub_json"]).read_text(encoding="utf-8"))
    assert [b["selection_status"] for b in bindings["bindings"]] == [
        "pending_candidate_search",
        "kept_generated_stub",
        "pending_candidate_search",
    ]
    pricing = json.loads(Path(artifacts["scene_pricing_stub_json"]).read_text(encoding="utf-8"))
    assert pricing["meta"]["supplier_catalog_count"] == 2

    bad_scene = tmp_path / "bad_scene.json"
    bad_scene.write_text(json.dumps({"placements": {"bad": True}}), encoding="utf-8")
    with pytest.raises(RuntimeError):
        layout_targets.extract_layout_targets(bad_scene, tmp_path / "bad_targets.json")

    bad_targets = tmp_path / "bad_targets.json"
    bad_targets.write_text(json.dumps({"targets": {"bad": True}}), encoding="utf-8")
    with pytest.raises(RuntimeError):
        layout_targets.build_supplier_bindings_stub(bad_targets, tmp_path / "bad_bindings.json")


def test_acquire_assets_for_bindings_uses_ready_cache_downloads_and_failures(tmp_path: Path, monkeypatch) -> None:
    ready = tmp_path / "ready.fbx"
    ready.write_text("fbx", encoding="utf-8")
    low_quality = tmp_path / "proxy.glb"
    low_quality.write_bytes(b"proxy")
    downloaded = tmp_path / "downloaded.glb"
    downloaded.write_bytes(b"glb")

    monkeypatch.setattr(acquire, "init_db", lambda db_path: None)
    monkeypatch.setattr(acquire, "upsert_product", lambda db_path, record: None)
    monkeypatch.setattr(acquire, "upsert_asset", lambda db_path, asset: None)

    acquired_records = []

    def fake_acquire_asset_for_record(record, *, db_path, out_dir, blender_bin=None):
        acquired_records.append((record.unique_key, db_path, out_dir, blender_bin))
        if record.unique_key == "fail":
            raise RuntimeError("download failed")
        return SimpleNamespace(
            asset_status="ready_downloaded_local_asset",
            asset_source_url=record.model_download_url,
            preview_local_path=None,
            blender_job_path=None,
            notes_json='{"note": true}',
            extra_json="{}",
            asset_format="glb",
            asset_local_path=str(downloaded),
        )

    monkeypatch.setattr(acquire, "acquire_asset_for_record", fake_acquire_asset_for_record)
    monkeypatch.setattr(acquire, "now_utc_iso", lambda: "2026-01-01T00:00:00Z")

    bindings = {
        "meta": {"source": "unit"},
        "bindings": [
            {
                "target_id": "ready",
                "selection_status": "heuristic_top1_selected",
                "chosen_candidate": {
                    "unique_key": "ready",
                    "source_site": "site",
                    "asset_local_path": str(ready),
                    "asset_format": "fbx",
                    "asset_status": "ready_existing_local_asset",
                },
                "top_candidates": [],
                "selection_notes": [],
            },
            {
                "target_id": "fallback",
                "selection_status": "heuristic_first_acceptable_selected",
                "chosen_candidate": {
                    "unique_key": "low",
                    "source_site": "site",
                    "asset_local_path": str(low_quality),
                    "asset_format": "glb",
                    "asset_status": "proxy_generated_with_blender",
                },
                "top_candidates": [
                    {
                        "unique_key": "download",
                        "source_site": "site",
                        "source_url": "https://product",
                        "model_download_url": "https://download/model.glb",
                        "acceptability": {"accepted": True},
                    }
                ],
                "selection_notes": [],
            },
            {
                "target_id": "failed",
                "selection_status": "llm_reranked_top1_selected",
                "chosen_candidate": {
                    "unique_key": "fail",
                    "source_site": "site",
                    "source_url": "https://bad",
                    "model_download_url": "https://bad/model.glb",
                },
                "top_candidates": [],
                "selection_notes": [],
                "provenance": {},
            },
            {
                "target_id": "ignored",
                "selection_status": "unmatched",
                "chosen_candidate": {"unique_key": "ignored", "source_site": "site"},
            },
        ],
    }

    out = acquire.acquire_assets_for_bindings_data(
        bindings,
        db_path=tmp_path / "assets.sqlite",
        out_dir=tmp_path / "assets",
        blender_bin="/bin/false",
    )

    meta = out["meta"]["asset_acquisition"]
    assert meta["selected_binding_count"] == 3
    assert meta["ready_before_count"] == 1
    assert meta["downloaded_ready_count"] == 1
    assert meta["unresolved_count"] == 1
    assert meta["failed_count"] == 1
    assert out["bindings"][0]["chosen_candidate"]["asset_local_path"].endswith("ready.fbx")
    assert out["bindings"][1]["chosen_candidate"]["unique_key"] == "download"
    assert out["bindings"][1]["chosen_candidate"]["asset_local_path"] == str(downloaded)
    assert out["bindings"][2]["selection_status"] == "no_real_asset_after_acquisition"
    assert out["bindings"][2]["provenance"]["final_asset_source"] == "generated"
    assert acquired_records == [
        ("download", tmp_path / "assets.sqlite", tmp_path / "assets", "/bin/false"),
        ("fail", tmp_path / "assets.sqlite", tmp_path / "assets", "/bin/false"),
    ]


def test_acquire_assets_json_loads_catalog_rows_and_keeps_unresolved(tmp_path: Path, monkeypatch) -> None:
    low_quality = tmp_path / "proxy.glb"
    low_quality.write_bytes(b"proxy")
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "unique_key": "cat1",
                        "source_site": "catalog_site",
                        "source_url": "https://catalog",
                        "title": "Catalog chair",
                        "dimensions_cm": {"width": 50, "depth": 60, "height": 90},
                        "asset_local_path": str(low_quality),
                        "asset_format": "glb",
                        "asset_status": "needs_blender_rebuild",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    bindings_path = tmp_path / "bindings.json"
    bindings_path.write_text(
        json.dumps(
            {
                "bindings": [
                    {
                        "selection_status": "heuristic_top1_selected",
                        "chosen_candidate": {"unique_key": "cat1"},
                        "top_candidates": [],
                        "selection_notes": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "out.json"
    monkeypatch.setattr(acquire, "init_db", lambda db_path: None)
    monkeypatch.setattr(acquire, "upsert_product", lambda db_path, record: None)
    monkeypatch.setattr(acquire, "upsert_asset", lambda db_path, asset: None)

    result_path = acquire.acquire_assets_for_bindings_json(
        bindings_json_path=bindings_path,
        output_json_path=output,
        db_path=tmp_path / "db.sqlite",
        out_dir=tmp_path / "assets",
        catalog_json_paths=[catalog],
        keep_unresolved_candidates=True,
    )

    data = json.loads(result_path.read_text(encoding="utf-8"))
    assert data["bindings"][0]["chosen_candidate"]["asset_low_quality_local_path"] == str(low_quality.resolve())
    assert "asset_acquisition_no_real_asset_found_keep_candidate_for_fallback" in data["bindings"][0]["selection_notes"]
    assert data["meta"]["asset_acquisition"]["keep_unresolved_candidates"] is True


def test_acquire_asset_helpers_cover_catalog_payload_and_candidate_edges(tmp_path: Path) -> None:
    local_obj = tmp_path / "chair.obj"
    local_obj.write_text("obj", encoding="utf-8")
    low_proxy = tmp_path / "proxy.glb"
    low_proxy.write_bytes(b"glb")

    assert acquire._infer_asset_format({"asset_local_path": str(local_obj)}) == "obj"
    assert acquire._infer_asset_format({}) is None
    assert acquire._candidate_has_ready_local_asset({}) is False
    assert acquire._candidate_has_ready_local_asset({"asset_local_path": str(tmp_path / "missing.glb")}) is False
    assert acquire._candidate_has_real_local_asset(
        {
            "asset_local_path": str(low_proxy),
            "asset_status": "needs_blender_rebuild",
        }
    ) is False

    low_payload = acquire._asset_payload_from_acquired_asset(
        SimpleNamespace(
            asset_status="proxy_generated_with_blender",
            asset_source_url="https://asset",
            preview_local_path=None,
            blender_job_path=None,
            notes_json="{}",
            extra_json="{}",
            asset_format="glb",
            asset_local_path=str(low_proxy),
        )
    )
    assert low_payload["asset_low_quality_local_path"] == str(low_proxy)
    assert "asset_local_path" not in low_payload

    binding = {
        "chosen_candidate": {"unique_key": "a", "source_site": "site"},
        "top_candidates": [
            {"unique_key": "a", "source_site": "site"},
            {"unique_key": "", "source_site": "site"},
            "bad",
            {"unique_key": "b", "source_site": "site"},
        ],
    }
    assert [candidate["unique_key"] for candidate in acquire._binding_candidate_pool(binding)] == ["a", "b"]

    acquire._apply_payload_to_binding(
        {
            "chosen_candidate": {"unique_key": "chosen"},
            "top_candidates": ["bad", {"unique_key": "other"}, {"unique_key": "match"}],
        },
        "match",
        {"asset_status": "ready_existing_local_asset", "asset_local_path": str(local_obj), "asset_format": "obj"},
    )

    catalog_dict = tmp_path / "catalog_dict.json"
    catalog_dict.write_text(
        json.dumps(
            {
                "unique_key": "cat-dict",
                "source_site": "catalog",
                "dimensions_cm": {"width": 10, "depth": 20, "height": 30, "weight_kg": 4},
                "images": [{"url": "https://img"}],
                "extra": {"raw": True},
            }
        ),
        encoding="utf-8",
    )
    catalog_bad = tmp_path / "catalog_bad.json"
    catalog_bad.write_text(json.dumps({"items": {"not": "a list"}}), encoding="utf-8")
    catalog_list = tmp_path / "catalog_list.json"
    catalog_list.write_text(
        json.dumps(
            [
                "bad",
                {"source_site": "no-key"},
                {"unique_key": "cat-list", "source_site": "catalog", "width_cm": 11},
            ]
        ),
        encoding="utf-8",
    )
    rows = acquire._catalog_row_by_unique_key([catalog_dict, catalog_bad, catalog_list])
    assert rows["cat-dict"]["width_cm"] == 10
    assert rows["cat-dict"]["weight_kg"] == 4
    assert json.loads(rows["cat-dict"]["images_json"])[0]["url"] == "https://img"
    assert rows["cat-list"]["width_cm"] == 11

    merged = acquire._merge_catalog_fields(
        {
            "unique_key": "same",
            "asset_status": "ready_existing_local_asset",
            "asset_local_path": "candidate.glb",
            "title": "Candidate title",
        },
        {
            "unique_key": "same",
            "asset_status": "trellis_generated_local_asset",
            "asset_local_path": "catalog.glb",
            "asset_format": "glb",
            "title": "Catalog title",
        },
    )
    assert merged["asset_local_path"] == "catalog.glb"
    assert merged["title"] == "Candidate title"

    with pytest.raises(RuntimeError, match="unique_key"):
        acquire._candidate_to_product_record({"source_site": "site"})

    for key in ("product_url", "model_page_url", "model_download_url", "model_download_landing_url"):
        record = acquire._candidate_to_product_record(
            {
                "unique_key": key,
                "source_site": "site",
                key: f"https://example.test/{key}",
            }
        )
        assert record.source_url == f"https://example.test/{key}"


def test_acquire_data_handles_invalid_skips_rejected_and_cached_candidates(tmp_path: Path, monkeypatch) -> None:
    ready = tmp_path / "ready.glb"
    ready.write_bytes(b"glb")
    downloaded = tmp_path / "downloaded.glb"
    downloaded.write_bytes(b"glb")

    monkeypatch.setattr(acquire, "init_db", lambda db_path: None)
    monkeypatch.setattr(acquire, "upsert_product", lambda db_path, record: None)
    monkeypatch.setattr(acquire, "upsert_asset", lambda db_path, asset: None)

    calls: list[str] = []

    def fake_acquire_asset_for_record(record, *, db_path, out_dir, blender_bin=None):
        calls.append(record.unique_key)
        return SimpleNamespace(
            asset_status="ready_downloaded_local_asset",
            asset_source_url=record.source_url,
            preview_local_path=None,
            blender_job_path=None,
            notes_json="{}",
            extra_json="{}",
            asset_format="glb",
            asset_local_path=str(downloaded),
        )

    monkeypatch.setattr(acquire, "acquire_asset_for_record", fake_acquire_asset_for_record)

    with pytest.raises(RuntimeError, match="bindings"):
        acquire.acquire_assets_for_bindings_data(
            {"bindings": {"bad": True}},
            db_path=tmp_path / "db.sqlite",
            out_dir=tmp_path / "assets",
        )

    data = {
        "bindings": [
            "bad",
            {"selection_status": "heuristic_top1_selected", "chosen_candidate": "bad"},
            {"selection_status": "unmatched", "chosen_candidate": {"unique_key": "skip", "source_site": "site"}},
            {
                "target_id": "choose-third",
                "selection_status": "heuristic_top1_selected",
                "chosen_candidate": {
                    "unique_key": "low",
                    "source_site": "site",
                    "asset_status": "needs_blender_rebuild",
                    "asset_local_path": str(ready),
                },
                "top_candidates": [
                    {"unique_key": "rejected", "source_site": "site", "acceptability": {"accepted": False}},
                    {"unique_key": "download", "source_site": "site", "source_url": "https://download"},
                ],
                "selection_notes": [],
            },
            {
                "target_id": "reuse-cache",
                "selection_status": "llm_reranked_first_acceptable_selected",
                "chosen_candidate": {"unique_key": "download", "source_site": "site", "source_url": "https://download"},
                "top_candidates": [],
                "selection_notes": [],
            },
        ]
    }

    out = acquire.acquire_assets_for_bindings_data(
        data,
        db_path=tmp_path / "db.sqlite",
        out_dir=tmp_path / "assets",
    )

    assert calls == ["download"]
    assert out["bindings"][3]["chosen_candidate"]["unique_key"] == "download"
    assert out["bindings"][4]["chosen_candidate"]["asset_local_path"] == str(downloaded)
    assert out["meta"]["asset_acquisition"]["selected_binding_count"] == 2
    assert out["meta"]["asset_acquisition"]["unresolved_count"] == 1


def test_acquire_cli_build_and_main_print_summary(tmp_path: Path, monkeypatch, capsys) -> None:
    parser = acquire.build_cli()
    args = parser.parse_args(
        [
            "--bindings-json",
            "bindings.json",
            "--out",
            "out.json",
            "--db",
            "db.sqlite",
            "--out-dir",
            "assets",
            "--blender",
            "blender",
            "--catalog-json",
            "catalog.json",
        ]
    )
    assert args.blender == "blender"
    assert args.catalog_json == ["catalog.json"]

    out_json = tmp_path / "out.json"
    out_json.write_text(json.dumps({"meta": {"asset_acquisition": {"selected_binding_count": 1}}}), encoding="utf-8")
    monkeypatch.setattr(acquire, "acquire_assets_for_bindings_json", lambda **kwargs: out_json)
    monkeypatch.setattr(
        acquire.sys,
        "argv",
        [
            "acquire",
            "--bindings-json",
            str(tmp_path / "bindings.json"),
            "--out",
            str(out_json),
            "--db",
            str(tmp_path / "db.sqlite"),
            "--out-dir",
            str(tmp_path / "assets"),
        ],
    )

    acquire.main()
    printed = capsys.readouterr().out
    assert f"saved = {out_json}" in printed
    assert '"selected_binding_count": 1' in printed
