from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import argparse
import importlib
import json
import math
import types
import sys

# Allow importing src modules when tests execute from repo root without PYTHONPATH=src
ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = str(ROOT / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import pytest

from src import run_pipeline as rp


def test_run_pipeline_can_import_through_top_level_fallback(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "src"))
    sys.modules.pop("run_pipeline", None)
    module = importlib.import_module("run_pipeline")
    try:
        assert module.PLACER_SPECS
        assert module.ASSET_FALLBACK_MODE_NONE == rp.ASSET_FALLBACK_MODE_NONE
    finally:
        sys.modules.pop("run_pipeline", None)


def test_append_pipeline_timing_records_timings(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    start = datetime.now()
    rp._append_pipeline_timing(
        run_dir,
        stage="a",
        started=start,
        duration_sec=1.23456,
        status="ok",
        extra="x",
    )
    rp._append_pipeline_timing(
        run_dir,
        stage="b",
        started=start + timedelta(seconds=2),
        duration_sec=0.5,
        status="failed",
        detail=None,
    )

    payload = json.loads((run_dir / "pipeline_stage_timings.json").read_text(encoding="utf-8"))
    stages = payload["stages"]
    assert len(stages) == 2
    assert stages[0]["stage"] == "a"
    assert stages[1]["stage"] == "b"
    assert stages[1]["status"] == "failed"
    assert payload["duration_sec"] == round(1.23456 + 0.5, 3)


def test_pipeline_stage_context_success_and_failures(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    with rp._pipeline_stage(run_dir, "ok"):
        pass

    with pytest.raises(RuntimeError):
        with rp._pipeline_stage(run_dir, "boom"):
            raise RuntimeError("x")

    payload = json.loads((run_dir / "pipeline_stage_timings.json").read_text(encoding="utf-8"))
    stages = payload["stages"]
    assert [row["stage"] for row in stages] == ["ok", "boom"]
    assert stages[1]["status"] == "failed"


def test_to_float_and_polygon_helpers():
    assert rp._to_float(12) == 12.0
    assert rp._to_float(1.5) == 1.5
    assert rp._to_float("3,5") == 3.5
    assert rp._to_float(" 7.2 ") == 7.2
    assert rp._to_float(None) is None
    assert rp._to_float(True) is None
    assert rp._to_float("abc") is None

    square = [
        {"x": 0, "z": 0},
        {"x": 2, "z": 0},
        {"x": 2, "z": 2},
        {"x": 0, "z": 2},
    ]
    assert rp._polygon_area(square) == pytest.approx(4.0)
    assert rp._polygon_perimeter(square) == pytest.approx(8.0)

    bad = [{"x": 1}]
    assert rp._polygon_area(bad) is None
    assert rp._polygon_perimeter([]) is None


def test_room_surface_metrics_and_price_helpers(tmp_path: Path):
    room_path = tmp_path / "room.json"
    room_path.write_text(
        json.dumps(
            {
                "room": {
                    "width_m": 4,
                    "depth_m": 3,
                    "ceiling_height": 2.5,
                    "doors": [{"width": 1.0, "height": 2.0}],
                }
            }
        ),
        encoding="utf-8",
    )

    metrics = rp._room_surface_metrics(room_path)
    assert metrics["floor_area_m2"] == 12.0
    assert metrics["perimeter_m"] == 14.0
    assert metrics["gross_wall_area_m2"] == 35.0
    assert metrics["opening_area_m2"] == 2.0
    assert metrics["wall_area_m2"] == 33.0
    assert metrics["ceiling_height_m"] == 2.5

    assert rp._raw_property({"raw_properties": {"A": 1, "B": 2}}, ("B", "C")) == 2
    assert rp._raw_property({}, ("A",)) is None

    assert rp._floor_package_area_m2({"package_area_m2": "4.5"}) == 4.5
    assert rp._floor_package_area_m2({"raw_properties": {"Площадь упаковки": "6"}}) == 6.0

    assert rp._wall_roll_area_m2({"raw_properties": {"Площадь рулона": "8"}}) == 8.0
    assert rp._wall_roll_area_m2(
        {
            "width_cm": 300,
            "length_m": 10,
        }
    ) == 30.0


def test_surface_pricing_item_and_merge(tmp_path: Path):
    assert rp._surface_pricing_item(
        target_id="x",
        category="c",
        semantic_group="g",
        material={},
        coverage_area_m2=10,
        package_area_m2=4,
        quantity_unit="package",
    ) is None

    item = rp._surface_pricing_item(
        target_id="x",
        category="c",
        semantic_group="g",
        material={
            "price": "12.345",
            "sku": "S1",
            "price_currency": "USD",
            "name": "N",
        },
        coverage_area_m2=10,
        package_area_m2=4,
        quantity_unit="package",
    )
    assert item is not None
    assert item["target_id"] == "x"
    assert item["quantity"] == 3
    assert item["final_price_value"] == 37.04
    assert item["currency"] == "USD"

    room = tmp_path / "room.json"
    room.write_text(
        json.dumps(
            {
                "room": {
                    "width_m": 2,
                    "depth_m": 1,
                    "ceiling_height_m": 2.0,
                    "windows": [{"width": 1, "height": 1}],
                }
            }
        ),
        encoding="utf-8",
    )

    floor_sel = {
        "selected_material": {
            "sku": "F",
            "price": 100,
            "package_area_m2": 1,
            "name": "Floor",
        }
    }
    wall_sel = {
        "selected_material": {
            "sku": "W",
            "price": 200,
            "width_m": 0.5,
            "length_m": 4,
            "name": "Wall",
        }
    }

    floor_info = {"selection_json": str((tmp_path / "floor.json").resolve())}
    wall_info = {"selection_json": str((tmp_path / "wall.json").resolve())}
    (tmp_path / "floor.json").write_text(json.dumps(floor_sel), encoding="utf-8")
    (tmp_path / "wall.json").write_text(json.dumps(wall_sel), encoding="utf-8")

    stub = tmp_path / "stub.json"
    stub.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "target_id": "surface_floor",
                        "final_price_value": 10,
                        "name": "old floor",
                    }
                ],
                "meta": {},
                "totals": {},
            }
        ),
        encoding="utf-8",
    )

    result = rp._write_surface_material_pricing(
        run_dir=tmp_path,
        room_path=room,
        flooring_info=floor_info,
        wall_info=wall_info,
        pricing_stub_json=str(stub),
        suffix=".base",
    )

    assert result is not None
    pricing = json.loads(Path(result["pricing_json"]).read_text(encoding="utf-8"))
    assert pricing["totals"]["surface_material_item_count"] == 2
    assert pricing["items"][0]["target_id"] == "surface_floor"
    assert pricing["items"][1]["target_id"] == "surface_walls"

    stub_after = json.loads(stub.read_text(encoding="utf-8"))
    assert any(item.get("target_id") == "surface_floor" for item in stub_after["items"])
    assert any(item.get("target_id") == "surface_walls" for item in stub_after["items"])
    assert stub_after["meta"]["surface_material_count"] == 2


def test_room_scene_and_curtain_utils_and_parsers(tmp_path: Path):
    scene = {
        "room": {
            "windows": [{"id": "w1"}],
        },
        "items": [
            {"name": "sofa", "id": "i1"},
            {"name": "window", "source": {"kind": "curtain"}, "id": "c1"},
        ],
    }

    assert rp._scene_windows(scene) == [{"id": "w1"}]
    assert rp._scene_has_curtain_items(scene)

    should, reason = rp._curtains_needed_for_scene(
        scene={"room": {"windows": []}},
        prompt_text="need nice curtains",
        style_profile={},
        policy="auto",
    )
    assert should is False and reason == "missing_windows"

    should, reason = rp._curtains_needed_for_scene(
        scene=scene,
        prompt_text="need nice curtains",
        style_profile={"room_type": "Bedroom"},
        policy="auto",
    )
    assert should is False
    assert reason == "existing_curtains"

    should, reason = rp._curtains_needed_for_scene(
        scene={"room": {"windows": [{"id": "w1"}]}},
        prompt_text="без штор в комнате",
        style_profile={"room_type": "Bedroom"},
        policy="auto",
    )
    assert should is False
    assert reason == "prompt_says_no_curtains"

    assert rp._parse_supplier_build_modes("balanced,cheapest", ["optimal", "cheapest"]) == ["optimal"]
    assert rp._parse_supplier_selection_modes("balanced,cheap_top20,unknown") == ["optimal", "cheapest_top20"]
    assert rp._parse_supplier_selection_modes(None) == []

    variants = {"a": {}, "b": {}}
    rp._mark_supplier_blender_skipped(variants, reason="x")
    assert variants["a"]["blender"]["blender_status"] == "skipped"
    assert variants["b"]["blender"]["blender_error"] == "x"

    assert rp._parse_elevations("") == [0.0, 30.0, 45.0]
    assert rp._parse_elevations("10,20, 30") == [10.0, 20.0, 30.0]
    assert rp._parse_supplier_gif_layers("interior, kitchen,unknown,interior") == ["interior", "kitchen"]

    assert rp._is_fatal_disk_full_error(RuntimeError("no space left on device"))
    assert not rp._is_fatal_disk_full_error(RuntimeError("ok"))


def test_run_pipeline_remaining_local_branch_edges(tmp_path: Path, monkeypatch):
    warnings: list[str] = []
    assert rp._variant_total_price(
        {
            "targets": [
                "bad",
                {"target_id": "skip", "price": 100},
                {"target_id": "bad_price", "chosen_candidate_id": "c1", "price": "not-number"},
                {"target_id": "ok", "chosen_candidate_id": "c2", "price": "12.5"},
            ]
        },
        warnings,
        "mode",
    ) == 12.5
    assert warnings
    assert rp._write_supplier_variants_comparison(tmp_path, {}) is None

    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "targets": [
                    "bad",
                    {"target_id": "", "chosen_candidate_id": "skip"},
                    {"target_id": "chair", "category": "chair", "chosen_candidate_id": "c1", "price": 10, "final_score": 0.2},
                ]
            }
        ),
        encoding="utf-8",
    )
    comparison = rp._write_supplier_variants_comparison(
        tmp_path,
        {"best_match": {"reports": {"summary_json": str(summary_path)}, "rebind": {"scene_v1": "scene.json"}}},
    )
    assert json.loads(comparison.read_text(encoding="utf-8"))["target_differences"][0]["target_id"] == "chair"

    room = tmp_path / "room.json"
    room.write_text("{}", encoding="utf-8")
    artifacts = types.SimpleNamespace(placement_v1=tmp_path / "placement.json", scene_v1=None)
    kitchen_calls = []
    monkeypatch.setattr(
        rp,
        "apply_kitchen_stage_to_artifacts",
        lambda **kwargs: kitchen_calls.append(kwargs) or (kwargs["artifacts"], {"skipped_reason": "not_needed"}),
    )
    returned, info = rp._maybe_apply_kitchen_stage(
        args=types.SimpleNamespace(
            kitchens="bad-policy",
            kitchen_material_catalog="relative_materials.json",
            kitchen_appliance_catalog="relative_appliances.json",
            kitchen_selection_mode="optimal",
            kitchen_dining="auto",
            kitchen_accessories="auto",
            kitchen_accessory_llm_provider="none",
            kitchen_llm_provider="none",
        ),
        artifacts=artifacts,
        run_dir=tmp_path,
        room_path=str(room),
        prompt_text="kitchen",
        suffix="edge",
    )
    assert returned is artifacts
    assert info == {"skipped_reason": "not_needed"}
    assert kitchen_calls[0]["policy"] == "auto"
    assert kitchen_calls[0]["appliance_catalog"].name == "relative_appliances.json"

    scene_json = tmp_path / "scene.json"
    scene_json.write_text(json.dumps({"items": [{"id": "chair1", "category": "chair"}]}), encoding="utf-8")
    topview_runs = []

    def fake_topview_run(cmd, check=True):
        topview_runs.append(cmd)
        if "--save-blend" in cmd:
            Path(cmd[cmd.index("--save-blend") + 1]).write_bytes(b"blend")
            Path(cmd[cmd.index("--build-report") + 1]).write_text("{}", encoding="utf-8")
        if "--out" in cmd:
            Path(cmd[cmd.index("--out") + 1]).write_bytes(b"png")
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(rp.subprocess, "run", fake_topview_run)
    monkeypatch.setattr(rp, "_resolve_blender_binary_for_topview", lambda _args: "Blender")
    real_unlink = Path.unlink

    def fake_unlink(path, *args, **kwargs):
        if str(path).endswith("topview_vlm.edge.blend"):
            raise OSError("locked")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fake_unlink)
    topview = rp._render_topview_vlm_image(
        cfg_runtime={"BLENDER_VIS_SCRIPT": "visualize.py"},
        args=types.SimpleNamespace(
            blender="Blender",
            no_bbox_fallback=True,
            topview_vlm_keep_inspection_blend=False,
            topview_vlm_elevation_deg=81,
            topview_vlm_radius_mult=0.6,
            topview_vlm_lens=35,
            topview_vlm_resolution_x=320,
            topview_vlm_resolution_y=240,
        ),
        run_dir=tmp_path,
        scene_json_path=scene_json,
        tag="edge",
        highlight_item_ids=["chair1"],
    )
    assert topview["highlight_item_ids"] == ["chair1"]
    assert "--no-bbox-fallback" in topview_runs[0]
    assert "--highlight-item-ids" in topview_runs[0]

    variants = {"optimal": {"rebind": {"scene_v1": str(tmp_path / "missing_scene.json")}}}
    rp._run_supplier_blender_variants(
        cfg_runtime={"BLENDER_VIS_SCRIPT": "visualize.py"},
        args=types.SimpleNamespace(supplier_build_modes="optimal", topview_vlm_orientation_repair=False),
        run_dir=tmp_path,
        layout_mode="lego_gen",
        effective_room_path=str(room),
        variants=variants,
    )
    assert variants["optimal"]["blender"]["blender_error"] == "scene_json_missing"

    build_report = tmp_path / "build.json"
    build_report.write_text("{}", encoding="utf-8")
    no_bindings = {"optimal": {"blender": {"build_report": str(build_report)}, "rebind": {"scene_v1": str(scene_json)}}}
    rp._refresh_supplier_reports_after_blender(run_dir=tmp_path, variants=no_bindings)
    assert "reports" not in no_bindings["optimal"]

    frame_dir = tmp_path / "_frames_supplier_interior.elev_00"
    frame_dir.mkdir()
    monkeypatch.setattr(rp, "_render_gif_from_frames", lambda _frame_dir, out_gif, _fps: Path(out_gif).write_bytes(b"gif"))
    gif_runs = []
    monkeypatch.setattr(
        rp.subprocess,
        "run",
        lambda cmd, check=True: gif_runs.append(cmd) or Path(cmd[cmd.index("--turntable-render-dir") + 1]).mkdir(parents=True, exist_ok=True),
    )
    gif_info = rp._render_supplier_room_gifs(
        cfg_runtime={"BLENDER_VIS_SCRIPT": "visualize.py"},
        args=types.SimpleNamespace(
            skip_supplier_gif=False,
            supplier_gif_elevations="0",
            supplier_gif_layers="interior,kitchen",
            supplier_gif_frames=2,
            supplier_gif_fps=4,
            keep_supplier_gif_frames=False,
            blender="Blender",
        ),
        run_dir=tmp_path,
        layout_mode="lego_gen",
        supplier_scene_json_path=scene_json,
        supplier_blend_path=tmp_path / "missing.blend",
    )
    assert gif_info["used_reference_blend"] is False
    assert any("--blender" in cmd for cmd in gif_runs)
    assert not frame_dir.exists()

    assert rp._scene_windows({"room": "bad"}) == []
    assert rp._scene_windows({"room": {"windows": "bad"}}) == []
    assert not rp._scene_has_curtain_items({"placements": ["bad"]})
    assert rp._scene_has_curtain_items({"items": [{"id": "x", "asset": {"kind": "window_covering"}}]})
    assert rp._curtains_needed_for_scene(
        scene={"room": {"windows": [{"id": "w"}]}},
        prompt_text="",
        style_profile={"room_type": "living room"},
        policy="auto",
    ) == (True, "default_for_room_type:living_room")

    assert rp._maybe_apply_flooring_to_scene(
        args=types.SimpleNamespace(no_flooring=False, flooring_materials="missing_materials", flooring_style_rules="missing_rules.json"),
        run_dir=tmp_path,
        scene_json_path=scene_json,
        prompt_text="",
        style_profile={},
        room_id="r1",
        suffix=".edge",
    )[1] is None
    assert rp._maybe_apply_wall_material_to_scene(
        args=types.SimpleNamespace(no_wall_material=False, wall_materials="missing_wall_materials"),
        run_dir=tmp_path,
        scene_json_path=scene_json,
        prompt_text="",
        style_profile={},
        room_id="r1",
        suffix=".edge",
    )[1] is None

    curtain_scene = tmp_path / "curtain_scene.json"
    curtain_scene.write_text(json.dumps({"room": {"windows": [{"id": "w"}]}, "items": []}), encoding="utf-8")
    assert rp._maybe_apply_curtains_to_scene(
        args=types.SimpleNamespace(no_curtains=True, curtains="always"),
        run_dir=tmp_path,
        scene_json_path=curtain_scene,
        prompt_text="curtains",
        style_profile={},
        suffix=".edge",
    )[1] is None
    skipped_path, skipped_info = rp._maybe_apply_curtains_to_scene(
        args=types.SimpleNamespace(no_curtains=False, curtains="unknown"),
        run_dir=tmp_path,
        scene_json_path=curtain_scene,
        prompt_text="",
        style_profile={"room_type": "office"},
        suffix=".edge",
    )
    assert skipped_path == curtain_scene
    assert skipped_info["skipped_reason"] == "auto_not_requested"
