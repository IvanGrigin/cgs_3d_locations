from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = str(ROOT / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src import run_pipeline as rp
from tests.helpers.scene_builders import scene_with_room


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _args(**overrides):
    defaults = {
        "blender": "",
        "no_bbox_fallback": False,
        "supplier_assets_dir": "",
        "supplier_assets_db": "",
        "supplier_assets_blender": "",
        "supplier_catalog_json": [],
        "supplier_site": [],
        "supplier_rich_only": False,
        "supplier_bindings_json": "",
        "supplier_llm_provider": "none",
        "supplier_ollama_url": "",
        "supplier_ollama_model": "",
        "supplier_ollama_timeout": 0,
        "supplier_ollama_temperature": 0.0,
        "supplier_llm_top_n": 0,
        "supplier_top_k": 3,
        "supplier_selection_strategy": "balanced",
        "supplier_selection_mode": "optimal",
        "supplier_selection_modes": "",
        "supplier_user_preferences_json": "",
        "supplier_require_local_asset": False,
        "supplier_asset_fallback_mode": "none",
        "ollama_url": "http://127.0.0.1:11434",
        "ollama_model": "gpt-oss:20b",
        "ollama_timeout": 180,
        "placer": "lego_gen",
        "validate_supplier_variants": False,
        "skip_supplier_gif": False,
        "supplier_gif_elevations": "0",
        "supplier_gif_layers": "interior",
        "supplier_gif_frames": 2,
        "supplier_gif_fps": 4,
        "keep_supplier_gif_frames": False,
        "topview_vlm_orientation_repair": False,
        "topview_vlm_orientation_provider": "none",
        "topview_vlm_orientation_review_json": "",
        "topview_vlm_orientation_scope": "chairs",
        "topview_vlm_include_armchairs": False,
        "topview_vlm_orientation_max_objects": 100,
        "topview_vlm_keep_inspection_blend": False,
        "topview_vlm_elevation_deg": 80.0,
        "topview_vlm_radius_mult": 0.55,
        "topview_vlm_lens": 32.0,
        "topview_vlm_resolution_x": 640,
        "topview_vlm_resolution_y": 480,
        "topview_vlm_orientation_model": "",
        "topview_vlm_orientation_min_confidence": 0.7,
        "topview_vlm_orientation_max_delta_deg": 180.0,
        "topview_vlm_orientation_snap_step_deg": 90.0,
        "topview_vlm_orientation_no_apply": False,
    }
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


def test_build_layout_stub_prefers_scene_and_falls_back_to_placement(tmp_path: Path, monkeypatch):
    scene_path = _write_json(tmp_path / "scene.json", {"items": []})
    placement_path = _write_json(tmp_path / "placement.json", {"placements": []})
    calls: list[Path] = []

    def fake_stub(*, source_json_path, run_dir, prefix):
        calls.append(Path(source_json_path))
        return {
            "layout_targets_json": str(run_dir / f"{prefix}.targets.json"),
            "supplier_bindings_stub_json": str(run_dir / f"{prefix}.bindings.json"),
            "scene_pricing_stub_json": str(run_dir / f"{prefix}.pricing.json"),
        }

    monkeypatch.setattr(rp, "create_layout_selection_stub_artifacts", fake_stub)

    artifacts = types.SimpleNamespace(scene_v1=scene_path, placement_v1=placement_path)
    result = rp._build_layout_selection_stub_for_artifacts(artifacts=artifacts, run_dir=tmp_path, prefix="p")
    assert calls[-1] == scene_path
    assert result["layout_targets_json"].endswith("p.targets.json")

    artifacts.scene_v1 = tmp_path / "missing.json"
    rp._build_layout_selection_stub_for_artifacts(artifacts=artifacts, run_dir=tmp_path, prefix="p")
    assert calls[-1] == placement_path


def test_layout_postprocess_skip_missing_and_success(tmp_path: Path, monkeypatch):
    args = _args(normalize_chandeliers=False, repair_furniture_overlaps=False)
    scene = tmp_path / "scene.json"
    unchanged, info = rp._maybe_apply_layout_postprocess(args=args, scene_json_path=scene, run_dir=tmp_path, tag="x")
    assert unchanged == scene
    assert info is None

    args = _args(normalize_chandeliers=True, repair_furniture_overlaps=False)
    unchanged, info = rp._maybe_apply_layout_postprocess(args=args, scene_json_path=scene, run_dir=tmp_path, tag="x")
    assert unchanged == scene.resolve()
    assert info["skipped_reason"] == "scene_json_missing"

    scene = _write_json(tmp_path / "scene.json", {"items": [{"id": "lamp"}]})

    def fake_chandeliers(data):
        data["chandelier_normalized"] = True
        return data, {"changed": 1}

    def fake_repairs(data):
        data["overlap_repaired"] = True
        return data, {"moved": 2}

    monkeypatch.setattr(rp, "normalize_chandelier_positions_in_scene", fake_chandeliers)
    monkeypatch.setattr(rp, "repair_furniture_intersections_in_scene", fake_repairs)

    args = _args(normalize_chandeliers=True, repair_furniture_overlaps=True)
    out_path, info = rp._maybe_apply_layout_postprocess(args=args, scene_json_path=scene, run_dir=tmp_path, tag="base")
    output = json.loads(out_path.read_text(encoding="utf-8"))
    assert output["chandelier_normalized"] is True
    assert output["overlap_repaired"] is True
    assert info["normalize_chandeliers"] == {"changed": 1}
    assert info["repair_furniture_overlaps"] == {"moved": 2}


def test_kitchen_stage_normalizes_policy_and_paths(tmp_path: Path, monkeypatch):
    room = _write_json(tmp_path / "room.json", {"room": {}})
    artifacts = types.SimpleNamespace(placement_v1=tmp_path / "placement.json", scene_v1=None)
    material_catalog = _write_json(tmp_path / "materials.json", [])
    captured = {}

    def fake_kitchen(**kwargs):
        captured.update(kwargs)
        return kwargs["artifacts"], {"replacement_count": 2}

    monkeypatch.setattr(rp, "apply_kitchen_stage_to_artifacts", fake_kitchen)
    args = _args(
        kitchens="on",
        kitchen_material_catalog=str(material_catalog),
        kitchen_appliance_catalog="",
        kitchen_selection_mode="best",
        kitchen_dining="always",
        kitchen_accessories="never",
        kitchen_accessory_llm_provider="none",
        kitchen_llm_provider="none",
    )

    returned, info = rp._maybe_apply_kitchen_stage(
        args=args,
        artifacts=artifacts,
        run_dir=tmp_path,
        room_path=str(room),
        prompt_text="modern kitchen",
        suffix="k",
    )
    assert returned is artifacts
    assert info == {"replacement_count": 2}
    assert captured["policy"] == "always"
    assert captured["material_catalog"] == material_catalog.resolve()
    assert captured["appliance_catalog"] is None
    assert captured["dining_policy"] == "always"
    assert captured["accessories_policy"] == "never"


def test_apply_supplier_bindings_and_reports_are_file_based(tmp_path: Path, monkeypatch):
    placement = _write_json(tmp_path / "placement.json", {"items": [{"id": "a"}]})
    scene = _write_json(tmp_path / "scene.json", {"items": [{"id": "a"}]})
    bindings = _write_json(tmp_path / "bindings.json", {"bindings": []})
    artifacts = types.SimpleNamespace(placement_v1=placement, scene_v1=scene)
    calls: list[dict[str, object]] = []

    def fake_apply(**kwargs):
        calls.append(kwargs)
        source = json.loads(Path(kwargs["input_json_path"]).read_text(encoding="utf-8"))
        source.setdefault("meta", {})["supplier_binding_summary"] = {"replaced": 1}
        _write_json(Path(kwargs["output_json_path"]), source)

    def fake_reports(**kwargs):
        assert kwargs["mode"] == "optimal"
        return {"summary_json": str(tmp_path / "summary.json"), "html": str(tmp_path / "report.html")}

    monkeypatch.setattr(rp, "apply_supplier_bindings_to_json", fake_apply)
    monkeypatch.setattr(rp, "write_supplier_replacement_reports", fake_reports)

    info = rp._apply_supplier_bindings_for_artifacts(
        artifacts=artifacts,
        run_dir=tmp_path,
        bindings_json_path=bindings,
        require_local_asset=True,
        supplier_asset_fallback_mode="proxy",
        variant_suffix="optimal",
    )
    assert len(calls) == 2
    assert info["summary"] == {"replaced": 1}
    assert Path(info["placement_v1"]).is_file()
    assert Path(info["scene_v1"]).is_file()

    reports = rp._write_supplier_replacement_reports_for_artifacts(
        run_dir=tmp_path,
        bindings_json_path=bindings,
        supplier_info=info,
        variant_suffix="optimal",
        blender_build_report_path=tmp_path / "build.json",
    )
    assert reports["html"].endswith("report.html")


def test_supplier_variant_comparison_manifest_and_surface_sync(tmp_path: Path):
    cheapest_summary = _write_json(
        tmp_path / "cheapest.summary.json",
        {
            "warnings": ["low confidence"],
            "counts": {"targets": 2},
            "targets": [
                {"target_id": "chair", "category": "chair", "chosen_candidate_id": "c1", "price": 10, "final_score": 0.7},
                {"target_id": "table", "category": "table", "chosen_candidate_id": "t1", "price": None},
            ],
        },
    )
    best_summary = _write_json(
        tmp_path / "best.summary.json",
        {
            "targets": [
                {"target_id": "chair", "category": "chair", "chosen_candidate_id": "c2", "price": 25, "final_score": 0.9},
                {"target_id": "table", "category": "table", "chosen_candidate_id": "t1", "price": 30},
            ]
        },
    )
    variants = {
        "cheapest": {
            "bindings": "cheap.json",
            "reports": {"summary_json": str(cheapest_summary), "html": "cheap.html"},
            "rebind": {"scene_v1": "cheap.scene.json"},
            "blender": {"blend_path": "cheap.blend", "blend_exists": True, "blender_status": "ok"},
        },
        "best_match": {
            "bindings": "best.json",
            "reports": {"summary_json": str(best_summary), "html": "best.html"},
            "rebind": {"scene_v1": "best.scene.json"},
            "blender": {"blend_path": "best.blend", "blend_exists": False},
        },
    }

    comparison = rp._write_supplier_variants_comparison(tmp_path, variants)
    data = json.loads(comparison.read_text(encoding="utf-8"))
    assert data["variants"]["cheapest"]["total_price_estimate"] == 10.0
    assert any("low confidence" in warning for warning in data["warnings"])
    chair_row = next(row for row in data["target_differences"] if row["target_id"] == "chair")
    assert chair_row["all_modes_same"] is False

    manifest = rp._write_supplier_variants_manifest(
        run_dir=tmp_path,
        modes=["cheapest", "best_match"],
        variants=variants,
        room_design_spec_path="spec.json",
        comparison_json=str(comparison),
        validation_json="validation.json",
        warnings=["warn"],
    )
    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_data["artifacts"]["cheapest"]["blend"] == "cheap.blend"
    assert manifest_data["validation_json"] == "validation.json"

    final_scene = _write_json(
        tmp_path / "final.scene.json",
        {"room": {"floor_material": {"sku": "F"}, "wall_material": {"sku": "W"}, "curtains": [{"id": "c"}]}},
    )
    variant_scene = _write_json(tmp_path / "variant.scene.json", {"room": {}, "rooms": [{"id": "r"}]})
    sync_variants = {"optimal": {"rebind": {"scene_v1": str(variant_scene)}}}
    rp._sync_supplier_variant_surface_scene_paths(
        run_dir=tmp_path,
        variants=sync_variants,
        final_supplier_scene_path=final_scene,
    )
    synced_path = Path(sync_variants["optimal"]["rebind"]["scene_v1"])
    synced = json.loads(synced_path.read_text(encoding="utf-8"))
    assert synced["room"]["floor_material"]["sku"] == "F"
    assert synced["rooms"][0]["curtains"][0]["id"] == "c"


def test_topview_render_and_orientation_repair_are_mockable(tmp_path: Path, monkeypatch):
    scene = _write_json(tmp_path / "scene.json", {"items": [{"id": "chair1"}]})
    args = _args(
        blender="/fake/blender",
        topview_vlm_orientation_provider="mock",
        topview_vlm_orientation_repair=True,
    )
    cfg = {"BLENDER_VIS_SCRIPT": str(tmp_path / "visualize.py")}
    calls: list[list[str]] = []

    def fake_run(cmd, check):
        calls.append(list(cmd))
        if "--save-blend" in cmd:
            _write_json(Path(cmd[cmd.index("--build-report") + 1]), {"ok": True})
            Path(cmd[cmd.index("--save-blend") + 1]).write_text("blend", encoding="utf-8")
        if "--out" in cmd:
            Path(cmd[cmd.index("--out") + 1]).write_bytes(b"png")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(rp.subprocess, "run", fake_run)
    monkeypatch.setattr(rp, "_resolve_blender_binary_for_topview", lambda args: "/fake/blender")

    render_info = rp._render_topview_vlm_image(
        cfg_runtime=cfg,
        args=args,
        run_dir=tmp_path,
        scene_json_path=scene,
        tag="t",
        highlight_item_ids=["chair1"],
    )
    assert Path(render_info["topview_image"]).is_file()
    assert render_info["inspection_blend"] is None
    assert calls[0][0] == sys.executable
    assert "--highlight-item-ids" in calls[0]

    monkeypatch.setattr(rp, "_topview_vlm_target_ids", lambda scene_json_path, args: ["chair1"])
    monkeypatch.setattr(
        rp,
        "_render_topview_vlm_image",
        lambda **kwargs: {
            "topview_image": str(tmp_path / "top.png"),
            "inspection_blend": None,
            "build_report": None,
            "highlight_item_ids": kwargs["highlight_item_ids"],
        },
    )

    def fake_repair(**kwargs):
        _write_json(kwargs["out_scene_path"], {"items": [{"id": "chair1", "rotation_y": 90}]})
        _write_json(kwargs["out_review_path"], {"review": []})
        _write_json(kwargs["out_report_path"], {"summary": {"applied": 1}, "counts": {"changed": 1}})
        return {"summary": {"applied": 1}, "counts": {"changed": 1}}

    monkeypatch.setattr(rp, "run_topview_vlm_orientation_repair", fake_repair)
    out_scene, info = rp._maybe_apply_topview_vlm_orientation_repair(
        cfg_runtime=cfg,
        args=args,
        run_dir=tmp_path,
        scene_json_path=scene,
        tag="supplier.optimal",
    )
    assert out_scene.is_file()
    assert info["summary"] == {"applied": 1}
    assert info["highlight_item_ids"] == ["chair1"]


def test_run_supplier_blender_variants_handles_skip_success_and_failure(tmp_path: Path, monkeypatch):
    ok_scene = _write_json(tmp_path / "ok.scene.json", {"items": []})
    fail_scene = _write_json(tmp_path / "fail.scene.json", {"items": []})
    variants = {
        "optimal": {"rebind": {"scene_v1": str(ok_scene)}},
        "cheapest": {"rebind": {"scene_v1": str(fail_scene)}},
        "best_match": {"rebind": {"scene_v1": str(tmp_path / "missing.json")}},
    }

    monkeypatch.setattr(
        rp,
        "_maybe_apply_topview_vlm_orientation_repair",
        lambda **kwargs: (kwargs["scene_json_path"], {"output_scene_json": str(kwargs["scene_json_path"])}),
    )

    def fake_blender(*, variant_suffix, **kwargs):
        if variant_suffix.endswith("cheapest"):
            raise RuntimeError("render failed")
        blend = _write_json(tmp_path / "ok.blend", {"blend": True})
        report = _write_json(tmp_path / "ok.build.json", {"build": True})
        return {"blend_path": str(blend), "build_report": str(report), "render_path": "render.png"}

    monkeypatch.setattr(rp, "run_blender_for_mode", fake_blender)
    monkeypatch.setattr(
        rp,
        "blender_outputs_for_mode",
        lambda args, run_dir, layout_mode, variant_suffix: (run_dir / f"{variant_suffix}.blend", None, None, None),
    )

    rp._run_supplier_blender_variants(
        cfg_runtime={"BLENDER_VIS_SCRIPT": "builder.py"},
        args=_args(supplier_build_modes="optimal,min_price"),
        run_dir=tmp_path,
        layout_mode="lego",
        effective_room_path=str(tmp_path / "room.json"),
        variants=variants,
    )
    assert variants["optimal"]["blender"]["blender_status"] == "ok"
    assert variants["optimal"]["topview_vlm_orientation_repair"]["output_scene_json"] == str(ok_scene)
    assert variants["cheapest"]["blender"]["blender_status"] == "failed"
    assert variants["best_match"]["blender"]["blender_error"] == "not_in_supplier_build_modes"


def test_refresh_validate_finalize_supplier_variants(tmp_path: Path, monkeypatch):
    bindings_a = _write_json(tmp_path / "a.bindings.json", {})
    bindings_b = _write_json(tmp_path / "b.bindings.json", {})
    build_report = _write_json(tmp_path / "build.json", {"ok": True})
    variants = {
        "a": {"bindings": str(bindings_a), "rebind": {"scene_v1": "a.scene.json"}, "blender": {"build_report": str(build_report)}},
        "b": {"bindings": str(bindings_b), "rebind": {"scene_v1": "b.scene.json"}, "blender": {"build_report": None}},
    }

    monkeypatch.setattr(
        rp,
        "_write_supplier_replacement_reports_for_artifacts",
        lambda **kwargs: {"summary_json": str(tmp_path / f"{kwargs['variant_suffix']}.summary.json")},
    )
    rp._refresh_supplier_reports_after_blender(run_dir=tmp_path, variants=variants)
    assert variants["a"]["reports"]["summary_json"].endswith("a.summary.json")
    assert "reports" not in variants["b"]

    def fake_validator(argv):
        out_path = Path(argv[argv.index("--out") + 1])
        _write_json(out_path, {"warnings": ["w"], "errors": []})
        return 0

    monkeypatch.setattr(rp, "supplier_variant_validator_main", fake_validator)
    validation_path, warnings = rp._validate_supplier_variants_if_requested(
        args=_args(validate_supplier_variants=True),
        run_dir=tmp_path,
        variants=variants,
    )
    assert Path(validation_path).is_file()
    assert warnings == ["w"]

    manifest = {"room_design_spec_json": "spec.json"}
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, manifest)
    monkeypatch.setattr(rp, "_write_supplier_variants_comparison", lambda run_dir, variants: _write_json(tmp_path / "comparison.json", {"warnings": ["cw"]}))
    monkeypatch.setattr(rp, "_validate_supplier_variants_if_requested", lambda **kwargs: ("validation.json", ["vw"]))
    out_manifest = rp._finalize_supplier_variant_artifacts(
        args=_args(validate_supplier_variants=True),
        run_dir=tmp_path,
        manifest=manifest,
        manifest_path=manifest_path,
        variants=variants,
    )
    assert out_manifest["supplier_variants_validation_json"] == "validation.json"
    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert saved["supplier_variants_manifest_json"].endswith("supplier_variants.manifest.json")


def test_gif_rendering_uses_mocked_subprocess_and_ffmpeg(tmp_path: Path, monkeypatch):
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    (frame_dir / "frame_000.png").write_bytes(b"png")
    commands: list[list[str]] = []

    def fake_run(cmd, check):
        commands.append(list(cmd))
        Path(cmd[-1]).write_bytes(b"out")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(rp.shutil, "which", lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None)
    monkeypatch.setattr(rp.subprocess, "run", fake_run)
    gif = tmp_path / "out.gif"
    rp._render_gif_from_frames(frame_dir, gif, fps=12)
    assert gif.is_file()
    assert len(commands) == 2

    scene = _write_json(tmp_path / "scene.json", {"items": []})
    blend = tmp_path / "scene.blend"
    blend.write_text("blend", encoding="utf-8")
    render_calls: list[list[str]] = []

    def fake_blender_run(cmd, check):
        render_calls.append(list(cmd))
        frame_out = Path(cmd[cmd.index("--turntable-render-dir") + 1])
        frame_out.mkdir(parents=True, exist_ok=True)
        (frame_out / "frame_000.png").write_bytes(b"png")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(rp.subprocess, "run", fake_blender_run)
    monkeypatch.setattr(rp, "_render_gif_from_frames", lambda frame_dir, out_gif, fps: out_gif.write_bytes(b"gif"))
    info = rp._render_supplier_room_gifs(
        cfg_runtime={"BLENDER_VIS_SCRIPT": "builder.py"},
        args=_args(supplier_gif_elevations="0,30", supplier_gif_layers="interior,kitchen", keep_supplier_gif_frames=True),
        run_dir=tmp_path,
        layout_mode="layout",
        supplier_scene_json_path=scene,
        supplier_blend_path=blend,
    )
    assert len(info["outputs"]) == 4
    assert all(Path(row["gif"]).is_file() for row in info["outputs"])
    assert render_calls[0][0] == sys.executable
    assert "--reference-blend" in render_calls[0]


def test_asset_acquisition_and_supplier_binding_resolution(tmp_path: Path, monkeypatch):
    bindings = _write_json(tmp_path / "bindings.json", {"bindings": []})

    def fake_acquire(**kwargs):
        out = Path(kwargs["output_json_path"])
        _write_json(out, {"meta": {"asset_acquisition": {"downloaded": 2}}})
        return out

    monkeypatch.setattr(rp, "acquire_assets_for_bindings_json", fake_acquire)
    out_path, info = rp._acquire_supplier_assets_for_bindings(
        args=_args(supplier_catalog_json=[str(tmp_path / "catalog.json")], blender="/bin/blender"),
        run_dir=tmp_path,
        bindings_json_path=bindings,
    )
    assert out_path.is_file()
    assert info["summary"] == {"downloaded": 2}
    assert info["out_dir"].endswith("supplier_assets")

    explicit = tmp_path / "explicit.json"
    args = _args(supplier_bindings_json=str(explicit))
    assert rp._resolve_supplier_bindings_json(args=args, run_dir=tmp_path, layout_targets_json_path="targets.json") == explicit.resolve()

    catalog = _write_json(tmp_path / "catalog.json", [{"id": "c"}])
    prefs = _write_json(tmp_path / "prefs.json", {"style": "modern"})
    targets = _write_json(tmp_path / "targets.json", {"targets": [{"id": "chair"}]})
    captured = {}
    monkeypatch.setattr(rp, "load_supplier_catalog_json", lambda paths, sites=None, rich_only=False: [{"id": "row"}])
    monkeypatch.setattr(rp, "read_supplier_matcher_json", lambda path: json.loads(Path(path).read_text(encoding="utf-8")))

    def fake_build(**kwargs):
        captured.update(kwargs)
        return {"bindings": [{"target_id": "chair"}]}

    monkeypatch.setattr(rp, "build_bindings_with_candidates", fake_build)
    args = _args(
        supplier_catalog_json=[str(catalog)],
        supplier_user_preferences_json=str(prefs),
        supplier_llm_provider="ollama",
        supplier_llm_top_n=2,
        supplier_site=["site-a"],
        supplier_rich_only=True,
    )
    generated = rp._resolve_supplier_bindings_json(args=args, run_dir=tmp_path, layout_targets_json_path=str(targets), selection_mode="best_match")
    assert generated.is_file()
    assert captured["user_preferences"] == {"style": "modern"}
    assert captured["llm_settings"]["provider"] == "ollama"
    assert captured["selection_mode"] == "best_match"


def test_room_design_spec_and_supplier_modes_flow(tmp_path: Path, monkeypatch):
    targets = _write_json(tmp_path / "targets.json", {"targets": [{"id": "chair"}]})
    monkeypatch.setattr(rp, "read_supplier_matcher_json", lambda path: json.loads(Path(path).read_text(encoding="utf-8")))
    monkeypatch.setattr(rp, "build_room_design_spec", lambda **kwargs: {"prompt": kwargs["user_prompt"], "count": len(kwargs["layout_targets"]["targets"])})
    spec_path, spec = rp._build_room_design_spec_for_targets(
        run_dir=tmp_path,
        prompt_text="modern room",
        layout_targets_json_path=str(targets),
        style_profile={"style": "modern"},
    )
    assert spec_path.is_file()
    assert spec["count"] == 1

    placement = _write_json(tmp_path / "placement.json", {"items": []})
    scene = _write_json(tmp_path / "scene.json", {"items": []})
    artifacts = types.SimpleNamespace(placement_v1=placement, scene_v1=scene)
    manifest = {}
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, manifest)

    monkeypatch.setattr(
        rp,
        "_build_room_design_spec_for_targets",
        lambda **kwargs: (_write_json(tmp_path / "room_design_spec.json", {"spec": True}), {"spec": True}),
    )

    def fake_resolve(*, selection_mode, **kwargs):
        return _write_json(tmp_path / f"{selection_mode}.bindings.json", {"bindings": [{"mode": selection_mode}]})

    monkeypatch.setattr(rp, "_resolve_supplier_bindings_json", fake_resolve)
    monkeypatch.setattr(rp, "apply_supplier_scene_consistency", lambda data: data)

    def fake_assets(*, bindings_json_path, **kwargs):
        assets = _write_json(tmp_path / f"{bindings_json_path.stem}.assets.json", json.loads(bindings_json_path.read_text(encoding="utf-8")))
        return assets, {"bindings_json": str(assets), "summary": {"assets": 1}}

    def fake_apply(*, variant_suffix, **kwargs):
        scene_out = _write_json(tmp_path / f"{variant_suffix}.scene.json", {"items": []})
        return {"scene_v1": str(scene_out), "placement_v1": str(placement), "summary": {"replaced": 1}}

    def fake_reports(*, variant_suffix, **kwargs):
        summary = _write_json(
            tmp_path / f"{variant_suffix}.summary.json",
            {"targets": [{"target_id": "chair", "chosen_candidate_id": variant_suffix, "price": 1.5}]},
        )
        return {"summary_json": str(summary), "html": str(tmp_path / f"{variant_suffix}.html")}

    monkeypatch.setattr(rp, "_acquire_supplier_assets_for_bindings", fake_assets)
    monkeypatch.setattr(rp, "_apply_supplier_bindings_for_artifacts", fake_apply)
    monkeypatch.setattr(rp, "_write_supplier_replacement_reports_for_artifacts", fake_reports)

    primary_scene, primary_info, assets_info, report_info, manifest = rp._run_supplier_modes_for_artifacts(
        args=_args(supplier_selection_modes="optimal,min_price", supplier_require_local_asset=True),
        run_dir=tmp_path,
        artifacts=artifacts,
        layout_targets_json_path=str(targets),
        prompt_text="prompt",
        style_profile={"style": "modern"},
        style_supplier_preferences_path=None,
        manifest=manifest,
        manifest_path=manifest_path,
    )
    assert primary_scene.name == "optimal.scene.json"
    assert primary_info["summary"]["replaced"] == 1
    assert assets_info["summary"] == {"assets": 1}
    assert report_info["summary_json"].endswith("optimal.summary.json")
    assert set(manifest["supplier_variants"]) == {"optimal", "cheapest"}
    assert Path(manifest["supplier_variants_manifest_json"]).is_file()


def test_floor_wall_and_curtain_application_are_mocked(tmp_path: Path, monkeypatch):
    scene = _write_json(tmp_path / "scene.json", {"room": {"room_type": "bedroom", "windows": [{"id": "w1"}]}, "items": []})
    materials = tmp_path / "materials"
    materials.mkdir()
    rules = _write_json(tmp_path / "rules.json", {})

    def fake_floor_select(**kwargs):
        selection = {
            "selected_material": {"sku": "F1", "name": "Floor"},
            "texture_candidate": {"texture_path": "tex.jpg", "usable_in_blender": True},
            "llm_rerank": {"status": "skipped"},
        }
        _write_json(kwargs["out_path"], selection)
        return selection

    monkeypatch.setattr(rp, "run_flooring_selection", fake_floor_select)
    monkeypatch.setattr(rp, "apply_flooring_to_scene", lambda scene, selection: {**scene, "floor_applied": selection["selected_material"]["sku"]})
    monkeypatch.setattr(rp, "write_flooring_json", lambda data, path: _write_json(path, data))
    floor_path, floor_info = rp._maybe_apply_flooring_to_scene(
        args=_args(flooring_materials=str(materials), flooring_style_rules=str(rules), flooring_llm_provider="none"),
        run_dir=tmp_path,
        scene_json_path=scene,
        prompt_text="prompt",
        style_profile={"style_label": "modern", "room_type": "Bedroom"},
        room_id="room_1",
        suffix=".test",
    )
    assert json.loads(floor_path.read_text(encoding="utf-8"))["floor_applied"] == "F1"
    assert floor_info["texture_usable_in_blender"] is True

    def fake_wall_select(**kwargs):
        selection = {
            "selected_material": {
                "sku": "W1",
                "name": "Wall",
                "average_hex": "#ffffff",
                "dominant_colors_hex": ["#ffffff"],
            },
            "llm_rerank": {"status": "skipped"},
        }
        _write_json(kwargs["out_path"], selection)
        return selection

    monkeypatch.setattr(rp, "run_wall_selection", fake_wall_select)
    monkeypatch.setattr(rp, "apply_wall_material_to_scene_with_catalog", lambda scene, selection, materials_path: {**scene, "wall_applied": selection["selected_material"]["sku"]})
    monkeypatch.setattr(rp, "write_wall_json", lambda data, path: _write_json(path, data))
    wall_path, wall_info = rp._maybe_apply_wall_material_to_scene(
        args=_args(wall_materials=str(materials), wall_llm_provider="none"),
        run_dir=tmp_path,
        scene_json_path=floor_path,
        prompt_text="prompt",
        style_profile={"style_label": "coastal", "room_type": "bedroom"},
        room_id="room_1",
        suffix=".test",
    )
    assert json.loads(wall_path.read_text(encoding="utf-8"))["wall_applied"] == "W1"
    assert wall_info["dominant_colors_hex"] == ["#ffffff"]

    monkeypatch.setattr(rp, "load_curtain_catalog", lambda path: ([{"sku": "C1"}], path))
    monkeypatch.setattr(rp, "discover_curtain_models", lambda path: [path / "curtain.fbx"])
    monkeypatch.setattr(rp, "discover_supplier_curtain_models", lambda **kwargs: [{"path": "supplier.fbx"}])
    monkeypatch.setattr(
        rp,
        "apply_curtains_to_scene",
        lambda scene, **kwargs: (
            {**scene, "items": scene.get("items", []) + [{"id": "curtain_win_0", "category": "curtain"}]},
            {"added_count": 1, "selected": [{"sku": "C1", "name": "Curtain", "texture_path": "curtain.jpg"}]},
        ),
    )
    monkeypatch.setattr(rp, "write_curtain_json", lambda path, data: _write_json(path, data))
    curtain_path, curtain_info = rp._maybe_apply_curtains_to_scene(
        args=_args(
            curtains="always",
            no_curtains=False,
            curtain_materials=str(materials),
            curtain_models_dir=str(tmp_path / "models"),
            curtain_supplier_catalog=str(tmp_path / "supplier_catalog.json"),
            curtain_seed=0,
            seed=123,
        ),
        run_dir=tmp_path,
        scene_json_path=wall_path,
        prompt_text="prompt",
        style_profile={"room_type": "bedroom"},
        suffix=".test",
    )
    assert curtain_path.is_file()
    assert curtain_info["added_count"] == 1
    assert curtain_info["needed_reason"] == "policy_always"


def test_run_pipeline_surface_metrics_topview_and_curtain_edge_paths(tmp_path: Path, monkeypatch):
    assert rp._to_float(None) is None
    assert rp._to_float(True) is None
    assert rp._to_float("bad") is None
    assert rp._to_float("12,5 m") == 12.5
    assert rp._polygon_area([{"x": 0, "y": "bad"}, {"x": 1, "y": 0}, {"x": 0, "y": 1}]) is None
    assert rp._polygon_perimeter([{"x": 0, "y": 0}, {"x": None, "y": 0}]) is None

    room_path = _write_json(
        tmp_path / "room_metrics.json",
        {
            "room": {
                "width_m": "4",
                "depth_m": "3",
                "ceiling_height": "2.5",
                "doors": [{"width": "0.8", "height": "2.0"}, "bad"],
                "windows": "not-list",
                "openings": [{"width": "1.2", "height": "1.0"}],
            }
        },
    )
    metrics = rp._room_surface_metrics(room_path)
    assert metrics["floor_area_m2"] == 12.0
    assert metrics["opening_area_m2"] == pytest.approx(2.8)
    assert metrics["wall_area_m2"] == pytest.approx(32.2)

    assert rp._floor_package_area_m2({"raw_properties": {"Площадь": "2,4"}}) == 2.4
    assert rp._wall_roll_area_m2({"width_cm": "106", "length_m": "10"}) == pytest.approx(10.6)
    assert rp._wall_roll_area_m2({"raw_properties": {"Площадь рулона": "5.3"}}) == 5.3
    assert rp._wall_roll_area_m2({}) is None

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{bad", encoding="utf-8")
    list_json = _write_json(tmp_path / "list.json", [])
    assert rp._load_json_if_file("") is None
    assert rp._load_json_if_file(tmp_path / "missing.json") is None
    assert rp._load_json_if_file(bad_json) is None
    assert rp._load_json_if_file(list_json) is None

    blender = tmp_path / "fake_blender"
    blender.write_text("#!/bin/sh\n", encoding="utf-8")
    blender.chmod(0o755)
    assert rp._resolve_blender_binary_for_topview(_args(blender=str(blender))) == str(blender.resolve())
    monkeypatch.setenv("BLENDER_PATH", "")
    monkeypatch.setattr(rp.os, "access", lambda *_args: False)
    monkeypatch.setattr(rp.shutil, "which", lambda candidate: "/usr/bin/blender" if candidate == "blender" else None)
    assert rp._resolve_blender_binary_for_topview(_args(blender="")) == "/usr/bin/blender"

    not_scene = _write_json(tmp_path / "not_scene.json", [])
    assert rp._topview_vlm_target_ids(not_scene, _args()) == []

    scene_with_window = scene_with_room(windows=[{"id": "w1"}]).build()
    assert rp._curtains_needed_for_scene(scene={}, prompt_text="", style_profile={}, policy="auto") == (False, "missing_windows")
    assert rp._curtains_needed_for_scene(
        scene={**scene_with_window, "items": [{"category": "curtain"}]},
        prompt_text="",
        style_profile={},
        policy="auto",
    ) == (False, "existing_curtains")
    assert rp._curtains_needed_for_scene(scene=scene_with_window, prompt_text="no curtains", style_profile={}, policy="auto") == (False, "prompt_says_no_curtains")
    assert rp._curtains_needed_for_scene(scene=scene_with_window, prompt_text="", style_profile={"needs_curtains": True}, policy="auto") == (True, "profile_needs_curtains")
    assert rp._curtains_needed_for_scene(scene=scene_with_window, prompt_text="linen drapes", style_profile={}, policy="auto") == (True, "prompt_mentions_curtains")
    assert rp._curtains_needed_for_scene(scene=scene_with_window, prompt_text="", style_profile={"room_type": "living room"}, policy="auto") == (True, "default_for_room_type:living_room")
    assert rp._curtains_needed_for_scene(scene=scene_with_window, prompt_text="", style_profile={"room_type": "office"}, policy="auto") == (False, "auto_not_requested")

    scene_path = _write_json(tmp_path / "curtain_scene.json", scene_with_window)
    assert rp._maybe_apply_curtains_to_scene(
        args=_args(curtains="off", no_curtains=False),
        run_dir=tmp_path,
        scene_json_path=scene_path,
        prompt_text="curtains",
        style_profile={},
        suffix=".x",
    ) == (scene_path, None)

    missing_path, missing_info = rp._maybe_apply_curtains_to_scene(
        args=_args(curtains="always", no_curtains=False, curtain_materials=str(tmp_path / "missing_catalog")),
        run_dir=tmp_path,
        scene_json_path=scene_path,
        prompt_text="",
        style_profile={},
        suffix=".x",
    )
    assert missing_path == scene_path
    assert missing_info is None

    materials = tmp_path / "curtain_materials"
    materials.mkdir()
    monkeypatch.setattr(rp, "load_curtain_catalog", lambda _path: ([], materials))
    empty_path, empty_info = rp._maybe_apply_curtains_to_scene(
        args=_args(curtains="always", no_curtains=False, curtain_materials=str(materials)),
        run_dir=tmp_path,
        scene_json_path=scene_path,
        prompt_text="",
        style_profile={},
        suffix=".x",
    )
    assert empty_path == scene_path
    assert empty_info is None

    monkeypatch.setattr(rp, "load_curtain_catalog", lambda _path: ([{"sku": "C"}], materials))
    monkeypatch.setattr(rp, "discover_curtain_models", lambda _path: [])
    monkeypatch.setattr(rp, "discover_supplier_curtain_models", lambda **_kwargs: [])
    monkeypatch.setattr(rp, "apply_curtains_to_scene", lambda scene, **_kwargs: (scene, {"added_count": 0, "skipped_reason": "no_window_fit"}))
    no_add_path, no_add_info = rp._maybe_apply_curtains_to_scene(
        args=_args(curtains="always", no_curtains=False, curtain_materials=str(materials), curtain_models_dir=str(tmp_path), curtain_supplier_catalog=str(tmp_path / "catalog.json")),
        run_dir=tmp_path,
        scene_json_path=scene_path,
        prompt_text="",
        style_profile={},
        suffix=".x",
    )
    assert no_add_path == scene_path
    assert no_add_info["skipped_reason"] == "no_window_fit"


def test_run_pipeline_remaining_error_and_skip_edges(tmp_path: Path, monkeypatch):
    timing_path = tmp_path / "pipeline_stage_timings.json"
    timing_path.write_text("{bad", encoding="utf-8")
    rp._append_pipeline_timing(tmp_path, stage="bad_json", started=rp.datetime.now(), duration_sec=0.2, status="ok")
    assert json.loads(timing_path.read_text(encoding="utf-8"))["stages"][0]["stage"] == "bad_json"
    timing_path.write_text("[]", encoding="utf-8")
    rp._append_pipeline_timing(tmp_path, stage="list_json", started=rp.datetime.now(), duration_sec=0.3, status="ok")
    assert json.loads(timing_path.read_text(encoding="utf-8"))["duration_sec"] == 0.3

    room_not_dict = _write_json(tmp_path / "room_not_dict.json", {"room": []})
    metrics = rp._room_surface_metrics(room_not_dict)
    assert metrics["floor_area_m2"] is None
    assert metrics["wall_area_m2"] is None
    room_bad_polygon = _write_json(tmp_path / "room_bad_polygon.json", {"room": {"floor_polygon": "bad", "doors": {}}})
    assert rp._room_surface_metrics(room_bad_polygon)["perimeter_m"] is None
    assert rp._raw_property({"raw_properties": {"known": "1"}}, ("missing",)) is None
    assert rp._surface_pricing_item(
        target_id="x",
        category="c",
        semantic_group="g",
        material={},
        coverage_area_m2=1.0,
        package_area_m2=1.0,
        quantity_unit="pkg",
    ) is None
    assert rp._write_surface_material_pricing(
        run_dir=tmp_path,
        room_path=room_not_dict,
        flooring_info=None,
        wall_info=None,
        pricing_stub_json=None,
        suffix=".none",
    ) is None
    rp._merge_surface_materials_into_pricing_stub(tmp_path / "missing_stub.json", {"items": []}, tmp_path / "surface.json")
    bad_stub = _write_json(tmp_path / "bad_stub.json", {"items": {}})
    rp._merge_surface_materials_into_pricing_stub(bad_stub, {"items": [{"target_id": "surface_floor"}]}, tmp_path / "surface.json")
    assert json.loads(bad_stub.read_text(encoding="utf-8"))["items"] == {}

    list_scene = _write_json(tmp_path / "list_scene.json", [])
    assert rp._apply_room_surface_payloads(list_scene, {"floor_material": {"sku": "F"}}, tmp_path / "out.json") == list_scene
    plain_scene = _write_json(tmp_path / "plain_scene.json", {"room": {}})
    assert rp._apply_room_surface_payloads(plain_scene, {}, tmp_path / "out.json") == plain_scene

    final_scene = _write_json(tmp_path / "final_scene.json", {"room": {"floor_material": {"sku": "F"}}})
    variants = {
        "norebind": {},
        "empty": {"rebind": {}},
        "no_scene": {"rebind": {"scene_v1": ""}},
        "same": {"rebind": {"scene_v1": str(final_scene)}},
        "missing": {"rebind": {"scene_v1": str(tmp_path / "missing_scene.json")}},
    }
    rp._sync_supplier_variant_surface_scene_paths(run_dir=tmp_path, variants=variants, final_supplier_scene_path=final_scene)
    assert variants["same"]["rebind"]["surface_materials_synced"] is True
    assert "surface_materials_synced" not in variants["missing"]["rebind"]
    assert rp._parse_supplier_build_modes(None, ["optimal", "cheapest"]) == ["optimal", "cheapest"]

    monkeypatch.setenv("BLENDER_PATH", "")
    monkeypatch.setattr(rp.os, "access", lambda *_args: False)
    monkeypatch.setattr(rp.shutil, "which", lambda _candidate: None)
    with pytest.raises(FileNotFoundError):
        rp._resolve_blender_binary_for_topview(_args(blender=""))

    target_scene = scene_with_room().item("chair1").write(tmp_path / "target_scene.json")
    monkeypatch.setattr(rp, "collect_topview_vlm_scene_objects", lambda scene, max_objects: ["raw_ref"])
    monkeypatch.setattr(rp, "filter_topview_vlm_target_objects", lambda refs, **kwargs: [types.SimpleNamespace(object_id="chair1")])
    assert rp._topview_vlm_target_ids(target_scene, _args()) == ["chair1"]
    unchanged, info = rp._maybe_apply_topview_vlm_orientation_repair(
        cfg_runtime={},
        args=_args(topview_vlm_orientation_repair=False),
        run_dir=tmp_path,
        scene_json_path=target_scene,
        tag="off",
    )
    assert unchanged == target_scene.resolve()
    assert info is None
    missing, info = rp._maybe_apply_topview_vlm_orientation_repair(
        cfg_runtime={},
        args=_args(topview_vlm_orientation_repair=True),
        run_dir=tmp_path,
        scene_json_path=tmp_path / "missing_topview.json",
        tag="missing",
    )
    assert info["skipped_reason"] == "scene_json_missing"
    monkeypatch.setattr(rp, "_topview_vlm_target_ids", lambda scene_json_path, args: [])
    unchanged, info = rp._maybe_apply_topview_vlm_orientation_repair(
        cfg_runtime={},
        args=_args(topview_vlm_orientation_repair=True),
        run_dir=tmp_path,
        scene_json_path=target_scene,
        tag="none",
    )
    assert unchanged == target_scene.resolve()
    assert info["skipped_reason"] == "no_target_objects"

    assert rp._validate_supplier_variants_if_requested(args=_args(validate_supplier_variants=False), run_dir=tmp_path, variants={}) == (None, [])
    one_binding = _write_json(tmp_path / "one_bindings.json", {"bindings": []})
    assert rp._validate_supplier_variants_if_requested(
        args=_args(validate_supplier_variants=True),
        run_dir=tmp_path,
        variants={"one": {"bindings": str(one_binding)}},
    )[1] == ["supplier variant validation skipped: less than two bindings files"]

    def fake_validator(argv):
        out_path = Path(argv[argv.index("--out") + 1])
        _write_json(out_path, {"warnings": ["warn"], "errors": ["bad"]})
        return 2

    monkeypatch.setattr(rp, "supplier_variant_validator_main", fake_validator)
    with pytest.raises(RuntimeError):
        rp._validate_supplier_variants_if_requested(
            args=_args(validate_supplier_variants=True),
            run_dir=tmp_path,
            variants={
                "a": {"bindings": str(_write_json(tmp_path / "a_bindings.json", {}))},
                "b": {"bindings": str(_write_json(tmp_path / "b_bindings.json", {}))},
            },
        )

    monkeypatch.setattr(rp.shutil, "which", lambda _name: None)
    with pytest.raises(FileNotFoundError):
        rp._render_gif_from_frames(tmp_path, tmp_path / "out.gif", fps=8)
    assert rp._render_supplier_room_gifs(
        cfg_runtime={"BLENDER_VIS_SCRIPT": "vis.py"},
        args=_args(skip_supplier_gif=True),
        run_dir=tmp_path,
        layout_mode="lego",
        supplier_scene_json_path=target_scene,
        supplier_blend_path=tmp_path / "scene.blend",
    ) is None
    assert rp._render_supplier_room_gifs(
        cfg_runtime={"BLENDER_VIS_SCRIPT": "vis.py"},
        args=_args(skip_supplier_gif=False),
        run_dir=tmp_path,
        layout_mode="lego",
        supplier_scene_json_path=tmp_path / "missing_scene.json",
        supplier_blend_path=tmp_path / "scene.blend",
    ) is None

    assert rp._resolve_supplier_bindings_json(args=_args(supplier_catalog_json=[]), run_dir=tmp_path, layout_targets_json_path="targets.json") is None
    monkeypatch.setattr(rp, "load_supplier_catalog_json", lambda *args, **kwargs: [{"id": "row"}])
    monkeypatch.setattr(rp, "read_supplier_matcher_json", lambda _path: [])
    with pytest.raises(RuntimeError):
        rp._resolve_supplier_bindings_json(
            args=_args(supplier_catalog_json=[str(tmp_path / "catalog.json")], supplier_user_preferences_json=str(tmp_path / "prefs.json")),
            run_dir=tmp_path,
            layout_targets_json_path="targets.json",
        )
    monkeypatch.setattr(rp, "read_supplier_matcher_json", lambda _path: {"targets": []})
    monkeypatch.setattr(rp, "build_bindings_with_candidates", lambda **kwargs: {"bindings": [], "selection_mode": kwargs.get("selection_mode")})
    bindings_path = rp._resolve_supplier_bindings_json(
        args=_args(supplier_catalog_json=[str(tmp_path / "catalog.json")], supplier_selection_strategy="price"),
        run_dir=tmp_path,
        layout_targets_json_path="targets.json",
    )
    assert bindings_path.name == "base_supplier_bindings.heuristic.price.json"

    monkeypatch.setattr(rp, "read_supplier_matcher_json", lambda _path: [])
    with pytest.raises(RuntimeError):
        rp._build_room_design_spec_for_targets(run_dir=tmp_path, prompt_text="", layout_targets_json_path="bad_targets.json", style_profile={})

    room_type_scene = _write_json(tmp_path / "room_type_scene.json", {"room": {"room_type": "dining_room"}})
    assert rp._flooring_room_type({}, room_type_scene) == "living_room"
    bad_room_type_scene = tmp_path / "bad_room_type.json"
    bad_room_type_scene.write_text("{bad", encoding="utf-8")
    assert rp._flooring_room_type({}, bad_room_type_scene) is None

    assert rp._maybe_apply_flooring_to_scene(
        args=_args(no_flooring=True),
        run_dir=tmp_path,
        scene_json_path=target_scene,
        prompt_text="",
        style_profile={},
        room_id="r1",
        suffix=".x",
    ) == (target_scene, None)
    missing_floor_path, missing_floor_info = rp._maybe_apply_flooring_to_scene(
        args=_args(no_flooring=False, flooring_materials=str(tmp_path / "missing_floor"), flooring_style_rules=str(tmp_path / "rules.json")),
        run_dir=tmp_path,
        scene_json_path=target_scene,
        prompt_text="",
        style_profile={},
        room_id="r1",
        suffix=".x",
    )
    assert missing_floor_path == target_scene
    assert missing_floor_info is None
    floor_catalog = tmp_path / "floor_catalog"
    floor_catalog.mkdir()
    assert rp._maybe_apply_flooring_to_scene(
        args=_args(no_flooring=False, flooring_materials=str(floor_catalog), flooring_style_rules=str(tmp_path / "missing_rules.json")),
        run_dir=tmp_path,
        scene_json_path=target_scene,
        prompt_text="",
        style_profile={},
        room_id="r1",
        suffix=".x",
    ) == (target_scene, None)

    assert rp._maybe_apply_wall_material_to_scene(
        args=_args(no_wall_material=True),
        run_dir=tmp_path,
        scene_json_path=target_scene,
        prompt_text="",
        style_profile={},
        room_id="r1",
        suffix=".x",
    ) == (target_scene, None)
    assert rp._maybe_apply_wall_material_to_scene(
        args=_args(no_wall_material=False, wall_materials=str(tmp_path / "missing_wall")),
        run_dir=tmp_path,
        scene_json_path=target_scene,
        prompt_text="",
        style_profile={},
        room_id="r1",
        suffix=".x",
    ) == (target_scene, None)


def test_prompt_and_fast_infinigen_profile_helpers(tmp_path: Path):
    (tmp_path / "infinigen_clean_meta.json").write_text(
        json.dumps({"style_label": "minimalism", "room_semantic": "bedroom"}),
        encoding="utf-8",
    )
    prompt = rp._flooring_prompt_for_selector(
        "plain prompt",
        {
            "expanded_prompt": "expanded",
            "style_hint": "soft palette",
            "surface_design_brief": "wood floor",
            "preferred_colors": ["white", "oak"],
            "wall_palette": ["cream"],
            "floor_palette": ["oak"],
            "furniture_palette": ["black"],
            "material_family": ["wood"],
        },
        tmp_path,
    )
    assert "expanded" in prompt
    assert "Infinigen generated scene context" in prompt
    assert "Preferred room colors: white, oak" in prompt

    style_profile: dict[str, object] = {}
    rp._maybe_apply_fast_infinigen_profile(
        _args(
            infinigen_fast_small=True,
            infinigen_no_pose_cameras=True,
            infinigen_solve_steps_large=12,
            infinigen_solve_steps_medium=0,
            infinigen_solve_steps_small=3,
        ),
        style_profile,
    )
    infinigen = style_profile["infinigen"]
    assert infinigen["monkeypatch_params"]["obj_interior_obj_pct"] == 0.0
    assert "compose_indoors.pose_cameras_enabled=False" in infinigen["overrides"]
    assert "compose_indoors.solve_steps_large=12" in infinigen["overrides"]

    with pytest.raises(RuntimeError):
        rp._maybe_apply_fast_infinigen_profile(_args(infinigen_fast_small=True), {"infinigen": []})


def test_build_cli_parses_modern_supplier_surface_and_infinigen_flags():
    parser = rp.build_cli()
    args = parser.parse_args(
        [
            "--prompt",
            "modern room",
            "--room",
            "room.json",
            "--placer",
            "infinigen_clean",
            "--modes",
            "infinigen_clean",
            "--save-blend",
            "scene.blend",
            "--render",
            "scene.png",
            "--blender-output",
            "both",
            "--keep-blend",
            "--skip-blender",
            "--no-bbox-fallback",
            "--supplier-catalog-json",
            "catalog_a.json",
            "--supplier-catalog-json",
            "catalog_b.json",
            "--supplier-selection-modes",
            "optimal,cheapest",
            "--supplier-build-modes",
            "optimal",
            "--build-supplier-blend",
            "--validate-supplier-variants",
            "--supplier-require-local-asset",
            "--kitchens",
            "always",
            "--kitchen-dining",
            "always",
            "--kitchen-accessories",
            "never",
            "--no-flooring",
            "--no-wall-material",
            "--curtains",
            "always",
            "--supplier-gif-layers",
            "interior,kitchen",
            "--topview-vlm-orientation-repair",
            "--topview-vlm-orientation-provider",
            "openrouter",
            "--infinigen-fast-small",
            "--infinigen-no-pose-cameras",
            "--lego-render-policy",
            "base_only",
        ]
    )
    assert args.prompt == "modern room"
    assert args.placer == "infinigen_clean"
    assert args.blender_output == "both"
    assert args.keep_blend is True
    assert args.supplier_catalog_json[-2:] == ["catalog_a.json", "catalog_b.json"]
    assert args.build_supplier_blend is True
    assert args.kitchens == "always"
    assert args.no_flooring is True
    assert args.curtains == "always"
    assert args.topview_vlm_orientation_provider == "openrouter"
    assert args.infinigen_fast_small is True


def test_run_pipeline_for_mode_infinigen_clean_stop_after_placement(tmp_path: Path, monkeypatch):
    room = _write_json(
        tmp_path / "room.json",
        {"room": {"id": "r1", "type": "Bedroom", "floor_polygon": [{"x": 0, "y": 0}, {"x": 2, "y": 0}, {"x": 2, "y": 2}]}},
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    args = _args(
        placer="infinigen_clean",
        max_attempts=2,
        skip_existing_placement=False,
        stop_after_placement=True,
        skip_blender=True,
        save_blend=None,
        render=None,
        no_flooring=True,
        no_wall_material=True,
        no_curtains=True,
        kitchens="never",
        normalize_chandeliers=False,
        repair_furniture_overlaps=False,
        ollama_models=["mock-json"],
        ollama_max_attempts=1,
        ollama_temperature=0.0,
        plan_models=["mock-plan"],
        plan_think="none",
        plan_temperature=0.0,
        critic_models=["mock-critic"],
        critic_think="none",
        critic_temperature=0.0,
        infinigen_fast_small=False,
        infinigen_no_pose_cameras=False,
        infinigen_solve_steps_large=None,
        infinigen_solve_steps_medium=None,
        infinigen_solve_steps_small=None,
    )

    monkeypatch.setattr(rp.secrets, "token_bytes", lambda n: b"\x01" * n)
    monkeypatch.setattr(rp, "maybe_run_semantic_room_planner_stage", lambda **_kwargs: None)
    monkeypatch.setattr(rp, "_maybe_apply_kitchen_stage", lambda **kwargs: (kwargs["artifacts"], None))
    monkeypatch.setattr(rp, "maybe_apply_procedural_room_stage", lambda **kwargs: kwargs["artifacts"])
    monkeypatch.setattr(
        rp,
        "_build_layout_selection_stub_for_artifacts",
        lambda **kwargs: {
            "layout_targets_json": str(_write_json(kwargs["run_dir"] / "targets.json", {"targets": []})),
            "supplier_bindings_stub_json": str(_write_json(kwargs["run_dir"] / "bindings_stub.json", {"bindings": []})),
            "scene_pricing_stub_json": str(_write_json(kwargs["run_dir"] / "pricing_stub.json", {"items": []})),
        },
    )

    def fake_execute_placer(**kwargs):
        _write_json(kwargs["out_path"], {"placements": [{"id": "bed"}]})

    def fake_build_artifacts(**kwargs):
        placement_v1 = _write_json(kwargs["run_dir"] / "placement_base.v1.json", {"items": [{"id": "bed"}]})
        scene_v1 = _write_json(kwargs["run_dir"] / "scene_base.v1.json", {"items": [{"id": "bed"}]})
        return rp.PlacementArtifacts(
            placement_legacy=kwargs["placement_out"],
            placement_v1=placement_v1,
            scene_v1=scene_v1,
            scene_legacy=None,
        )

    monkeypatch.setattr(rp, "execute_placer", fake_execute_placer)
    monkeypatch.setattr(rp, "build_scene_artifacts", fake_build_artifacts)

    outputs = rp.run_pipeline_for_mode(
        cfg_runtime={},
        args=args,
        room_path=str(room),
        run_dir=run_dir,
        layout_mode="infinigen_clean",
        prompt_text="prompt",
        style_profile_template={"style_label": "minimalism", "room_type": "Bedroom", "supplier_preferences": {"sites": ["mock"]}},
    )
    assert outputs.base_artifacts.scene_v1.name == "scene_base.v1.json"
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["placer"] == "infinigen_clean"
    assert manifest["supplier_preferences_json"].endswith("style_supplier_preferences.json")
    assert (run_dir / "prompt.styled.txt").read_text(encoding="utf-8") == "prompt"
    timings = json.loads((run_dir / "pipeline_stage_timings.json").read_text(encoding="utf-8"))
    assert any(stage["stage"] == "placement_execute" for stage in timings["stages"])


def _pipeline_mode_args(**overrides):
    defaults = dict(
        placer="infinigen_clean",
        max_attempts=1,
        skip_existing_placement=False,
        stop_after_placement=False,
        skip_blender=True,
        save_blend=None,
        render=None,
        no_flooring=False,
        no_wall_material=False,
        no_curtains=False,
        kitchens="never",
        normalize_chandeliers=False,
        repair_furniture_overlaps=False,
        build_supplier_blend=False,
        ollama_models=["mock-json"],
        ollama_max_attempts=1,
        ollama_temperature=0.0,
        plan_models=["mock-plan"],
        plan_think="none",
        plan_temperature=0.0,
        critic_models=["mock-critic"],
        critic_think="none",
        critic_temperature=0.0,
        infinigen_fast_small=False,
        infinigen_no_pose_cameras=False,
        infinigen_solve_steps_large=None,
        infinigen_solve_steps_medium=None,
        infinigen_solve_steps_small=None,
    )
    defaults.update(overrides)
    return _args(**defaults)


def _mock_common_pipeline_stages(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(rp.secrets, "token_bytes", lambda n: b"\x02" * n)
    monkeypatch.setattr(rp, "maybe_run_semantic_room_planner_stage", lambda **_kwargs: None)
    monkeypatch.setattr(rp, "_maybe_apply_kitchen_stage", lambda **kwargs: (kwargs["artifacts"], {"kitchen": True}))
    monkeypatch.setattr(rp, "maybe_apply_procedural_room_stage", lambda **kwargs: kwargs["artifacts"])
    monkeypatch.setattr(
        rp,
        "_build_layout_selection_stub_for_artifacts",
        lambda **kwargs: {
            "layout_targets_json": str(_write_json(kwargs["run_dir"] / "targets.json", {"targets": []})),
            "supplier_bindings_stub_json": str(_write_json(kwargs["run_dir"] / "bindings_stub.json", {"bindings": []})),
            "scene_pricing_stub_json": str(_write_json(kwargs["run_dir"] / "pricing_stub.json", {"items": []})),
        },
    )

    def fake_execute_placer(**kwargs):
        _write_json(kwargs["out_path"], {"placements": [{"id": "bed"}]})

    def fake_build_artifacts(**kwargs):
        placement_v1 = _write_json(kwargs["run_dir"] / "placement_base.v1.json", {"items": [{"id": "bed"}]})
        scene_v1 = _write_json(kwargs["run_dir"] / "scene_base.v1.json", {"items": [{"id": "bed"}]})
        return rp.PlacementArtifacts(
            placement_legacy=kwargs["placement_out"],
            placement_v1=placement_v1,
            scene_v1=scene_v1,
            scene_legacy=None,
        )

    monkeypatch.setattr(rp, "execute_placer", fake_execute_placer)
    monkeypatch.setattr(rp, "build_scene_artifacts", fake_build_artifacts)
    monkeypatch.setattr(rp, "maybe_repair_scene_json", lambda **kwargs: (kwargs["scene_json_path"], {"tag": kwargs["tag"]}))
    monkeypatch.setattr(rp, "_maybe_apply_layout_postprocess", lambda **kwargs: (kwargs["scene_json_path"], {"tag": kwargs["tag"]}))
    monkeypatch.setattr(rp, "_maybe_apply_flooring_to_scene", lambda **kwargs: (kwargs["scene_json_path"], {"scene_v1": str(kwargs["scene_json_path"])}))
    monkeypatch.setattr(rp, "_maybe_apply_wall_material_to_scene", lambda **kwargs: (kwargs["scene_json_path"], {"scene_v1": str(kwargs["scene_json_path"])}))
    monkeypatch.setattr(rp, "_maybe_apply_curtains_to_scene", lambda **kwargs: (kwargs["scene_json_path"], {"scene_v1": str(kwargs["scene_json_path"]), "added_count": 1}))
    monkeypatch.setattr(rp, "_write_surface_material_pricing", lambda **kwargs: {"suffix": kwargs["suffix"]})
    monkeypatch.setattr(rp, "_write_supplier_replacement_reports_for_artifacts", lambda **kwargs: {"summary_json": str(tmp_path / "supplier.summary.json")})


def test_run_pipeline_for_mode_full_branch_with_supplier_variants_and_skip_blender(tmp_path: Path, monkeypatch):
    room = _write_json(tmp_path / "room.json", {"room": {"id": "r1", "floor_polygon": [{"x": 0, "y": 0}, {"x": 2, "y": 0}, {"x": 2, "y": 2}]}})
    run_dir = tmp_path / "run_variants"
    run_dir.mkdir()
    _mock_common_pipeline_stages(monkeypatch, tmp_path)
    supplier_scene = _write_json(run_dir / "supplier.scene.json", {"items": [{"id": "bed"}]})
    finalized: dict[str, object] = {}

    def fake_supplier_modes(**kwargs):
        manifest = dict(kwargs["manifest"])
        manifest["supplier_variants"] = {
            "optimal": {
                "rebind": {"scene_v1": str(supplier_scene)},
                "reports": {"summary_json": str(tmp_path / "summary.json")},
            }
        }
        return supplier_scene, {"summary": {"replaced": 1}}, {"bindings_json": str(tmp_path / "bindings.json")}, {"summary_json": "summary.json"}, manifest

    monkeypatch.setattr(rp, "_run_supplier_modes_for_artifacts", fake_supplier_modes)
    monkeypatch.setattr(rp, "_mark_supplier_blender_skipped", lambda variants, reason: variants["optimal"].setdefault("blender", {}).update({"blender_status": reason}))
    monkeypatch.setattr(
        rp,
        "_finalize_supplier_variant_artifacts",
        lambda **kwargs: finalized.setdefault("manifest", {**kwargs["manifest"], "finalized": True}) or finalized["manifest"],
    )

    outputs = rp.run_pipeline_for_mode(
        cfg_runtime={},
        args=_pipeline_mode_args(skip_blender=True, build_supplier_blend=True),
        room_path=str(room),
        run_dir=run_dir,
        layout_mode="infinigen_clean",
        prompt_text="prompt",
        style_profile_template={"style_label": "minimalism", "room_type": "Bedroom"},
    )

    assert outputs.base_artifacts.scene_v1.name == "scene_base.v1.json"
    assert finalized["manifest"]["finalized"] is True
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["flooring_base"]["scene_v1"].endswith("scene_base.v1.json")
    assert manifest["supplier_rebind"]["summary"] == {"replaced": 1}


def test_run_pipeline_for_mode_full_branch_renders_supplier_without_variants(tmp_path: Path, monkeypatch):
    room = _write_json(tmp_path / "room.json", {"room": {"id": "r1", "floor_polygon": [{"x": 0, "y": 0}, {"x": 3, "y": 0}, {"x": 3, "y": 3}]}})
    run_dir = tmp_path / "run_render"
    run_dir.mkdir()
    _mock_common_pipeline_stages(monkeypatch, tmp_path)
    supplier_scene = _write_json(run_dir / "supplier.scene.json", {"items": [{"id": "chair"}]})
    calls: list[tuple[str, str]] = []

    def fake_supplier_modes(**kwargs):
        return supplier_scene, {"summary": {"replaced": 1}}, {"bindings_json": str(tmp_path / "bindings.json")}, {"summary_json": "summary.json"}, kwargs["manifest"]

    monkeypatch.setattr(rp, "_run_supplier_modes_for_artifacts", fake_supplier_modes)
    monkeypatch.setattr(rp, "_maybe_apply_topview_vlm_orientation_repair", lambda **kwargs: (kwargs["scene_json_path"], {"tag": kwargs["tag"]}))
    monkeypatch.setattr(rp, "run_blender_for_mode", lambda **kwargs: calls.append((kwargs["layout_mode"], kwargs["variant_suffix"])) or {"blend_path": "scene.blend"})
    monkeypatch.setattr(rp, "blender_outputs_for_mode", lambda args, run_dir, layout_mode, variant_suffix: (_write_json(run_dir / f"{variant_suffix}.blend", {"blend": True}), None))
    monkeypatch.setattr(rp, "_render_supplier_room_gifs", lambda **kwargs: {"outputs": [{"gif": str(tmp_path / "supplier.gif")}]})

    rp.run_pipeline_for_mode(
        cfg_runtime={},
        args=_pipeline_mode_args(skip_blender=False),
        room_path=str(room),
        run_dir=run_dir,
        layout_mode="infinigen_clean",
        prompt_text="prompt",
        style_profile_template={"style_label": "minimalism", "room_type": "Bedroom"},
    )

    assert calls == [("infinigen_clean", ""), ("infinigen_clean", "supplier")]
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["topview_vlm_orientation_repair_supplier"]["tag"] == "supplier"
    assert manifest["supplier_room_gifs"]["outputs"][0]["gif"].endswith("supplier.gif")


def test_run_pipeline_for_mode_lego_gen_branch_with_supplier_variants(tmp_path: Path, monkeypatch):
    room = _write_json(tmp_path / "room.json", {"room": {"id": "r1", "floor_polygon": [{"x": 0, "y": 0}, {"x": 4, "y": 0}, {"x": 4, "y": 3}]}})
    run_dir = tmp_path / "run_lego"
    run_dir.mkdir()

    monkeypatch.setattr(rp.secrets, "token_bytes", lambda n: b"\x03" * n)
    monkeypatch.setattr(rp, "maybe_run_semantic_room_planner_stage", lambda **_kwargs: None)
    monkeypatch.setattr(rp, "run_choose_stage", lambda **kwargs: _write_json(kwargs["run_dir"] / "objects.legacy.json", {"objects": [{"id": "chair"}]}))
    monkeypatch.setattr(rp, "normalize_json_artifact", lambda **kwargs: _write_json(kwargs["output_path"], {"objects": [{"id": "chair"}]}))

    def fake_lego(**kwargs):
        placement = _write_json(kwargs["run_dir"] / "lego.placement.json", {"placements": [{"id": "chair"}]})
        placement_v1 = _write_json(kwargs["run_dir"] / "lego.placement.v1.json", {"items": [{"id": "chair"}]})
        scene_v1 = _write_json(kwargs["run_dir"] / "lego.scene.v1.json", {"items": [{"id": "chair"}]})
        scene_legacy = _write_json(kwargs["run_dir"] / "lego.scene.json", {"placements": [{"id": "chair"}]})
        return rp.PlacementArtifacts(placement_legacy=placement, placement_v1=placement_v1, scene_v1=scene_v1, scene_legacy=scene_legacy)

    monkeypatch.setattr(rp, "run_lego_generate_from_scratch", fake_lego)
    monkeypatch.setattr(rp, "_maybe_apply_kitchen_stage", lambda **kwargs: (kwargs["artifacts"], {"kitchen": True}))
    monkeypatch.setattr(rp, "maybe_apply_procedural_room_stage", lambda **kwargs: kwargs["artifacts"])
    monkeypatch.setattr(
        rp,
        "_build_layout_selection_stub_for_artifacts",
        lambda **kwargs: {
            "layout_targets_json": str(_write_json(kwargs["run_dir"] / "lego.targets.json", {"targets": []})),
            "supplier_bindings_stub_json": str(_write_json(kwargs["run_dir"] / "lego.bindings_stub.json", {"bindings": []})),
            "scene_pricing_stub_json": str(_write_json(kwargs["run_dir"] / "lego.pricing_stub.json", {"items": []})),
        },
    )

    supplier_scene = _write_json(run_dir / "lego.supplier.scene.json", {"items": [{"id": "chair"}]})

    def fake_supplier_modes(**kwargs):
        manifest = dict(kwargs["manifest"])
        manifest["supplier_variants"] = {
            "optimal": {
                "rebind": {"scene_v1": str(supplier_scene)},
                "assets": {"bindings_json": str(tmp_path / "bindings.json")},
                "reports": {"summary_json": str(tmp_path / "summary.json")},
            }
        }
        return supplier_scene, {"summary": {"replaced": 1}}, {"bindings_json": str(tmp_path / "bindings.json")}, {"mode": "optimal"}, manifest

    monkeypatch.setattr(rp, "_run_supplier_modes_for_artifacts", fake_supplier_modes)
    monkeypatch.setattr(rp, "maybe_repair_scene_json", lambda **kwargs: (kwargs["scene_json_path"], {"tag": kwargs["tag"]}))
    monkeypatch.setattr(rp, "_maybe_apply_layout_postprocess", lambda **kwargs: (kwargs["scene_json_path"], {"tag": kwargs["tag"]}))
    monkeypatch.setattr(rp, "_maybe_apply_flooring_to_scene", lambda **kwargs: (kwargs["scene_json_path"], {"scene_v1": str(kwargs["scene_json_path"])}))
    monkeypatch.setattr(rp, "_maybe_apply_wall_material_to_scene", lambda **kwargs: (kwargs["scene_json_path"], {"scene_v1": str(kwargs["scene_json_path"])}))
    monkeypatch.setattr(rp, "_maybe_apply_curtains_to_scene", lambda **kwargs: (kwargs["scene_json_path"], {"scene_v1": str(kwargs["scene_json_path"])}))
    monkeypatch.setattr(rp, "_write_surface_material_pricing", lambda **kwargs: {"suffix": kwargs["suffix"]})
    monkeypatch.setattr(rp, "_write_supplier_replacement_reports_for_artifacts", lambda **kwargs: {"summary_json": str(tmp_path / "replacement.json")})
    monkeypatch.setattr(rp, "_mark_supplier_blender_skipped", lambda variants, reason: [info.setdefault("blender", {}).update({"blender_error": reason}) for info in variants.values()])
    monkeypatch.setattr(rp, "_finalize_supplier_variant_artifacts", lambda **kwargs: kwargs["manifest"])

    outputs = rp.run_pipeline_for_mode(
        cfg_runtime={},
        args=_pipeline_mode_args(placer="lego_gen", skip_blender=True, build_supplier_blend=True),
        room_path=str(room),
        run_dir=run_dir,
        layout_mode="lego_gen",
        prompt_text="prompt",
        style_profile_template={"style_label": "modern", "room_type": "Bedroom"},
    )

    assert outputs.base_artifacts.scene_v1.name == "lego.scene.v1.json"
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["lego_gen"]["scene_v1"].endswith("lego.scene.v1.json")
    assert manifest["kitchen_stage"] == {"kitchen": True}
    assert manifest["flooring_base"]["scene_v1"].endswith("lego.scene.v1.json")
    assert manifest["surface_materials_pricing_supplier"]["suffix"] == ".lego_gen_supplier"
    assert manifest["supplier_replacement_reports"]["summary_json"].endswith("replacement.json")


def test_run_pipeline_for_mode_lego_gen_builds_supplier_blender_variants(tmp_path: Path, monkeypatch):
    room = _write_json(tmp_path / "room.json", {"room": {"id": "r1", "floor_polygon": [{"x": 0, "y": 0}, {"x": 4, "y": 0}, {"x": 4, "y": 3}]}})
    run_dir = tmp_path / "run_lego_blender"
    run_dir.mkdir()

    monkeypatch.setattr(rp.secrets, "token_bytes", lambda n: b"\x04" * n)
    monkeypatch.setattr(rp, "maybe_run_semantic_room_planner_stage", lambda **_kwargs: None)
    monkeypatch.setattr(rp, "run_choose_stage", lambda **kwargs: _write_json(kwargs["run_dir"] / "objects.legacy.json", {"objects": [{"id": "chair"}]}))
    monkeypatch.setattr(rp, "normalize_json_artifact", lambda **kwargs: _write_json(kwargs["output_path"], {"objects": [{"id": "chair"}]}))

    def fake_lego(**kwargs):
        placement = _write_json(kwargs["run_dir"] / "lego.placement.json", {"placements": [{"id": "chair"}]})
        placement_v1 = _write_json(kwargs["run_dir"] / "lego.placement.v1.json", {"items": [{"id": "chair"}]})
        scene_v1 = _write_json(kwargs["run_dir"] / "lego.scene.v1.json", {"items": [{"id": "chair"}]})
        return rp.PlacementArtifacts(placement_legacy=placement, placement_v1=placement_v1, scene_v1=scene_v1, scene_legacy=None)

    supplier_scene = _write_json(run_dir / "lego.supplier.scene.json", {"items": [{"id": "chair"}]})
    calls: list[str] = []
    monkeypatch.setattr(rp, "run_lego_generate_from_scratch", fake_lego)
    monkeypatch.setattr(rp, "_maybe_apply_kitchen_stage", lambda **kwargs: (kwargs["artifacts"], None))
    monkeypatch.setattr(rp, "maybe_apply_procedural_room_stage", lambda **kwargs: kwargs["artifacts"])
    monkeypatch.setattr(
        rp,
        "_build_layout_selection_stub_for_artifacts",
        lambda **kwargs: {
            "layout_targets_json": str(_write_json(kwargs["run_dir"] / "lego.targets.json", {"targets": []})),
            "supplier_bindings_stub_json": str(_write_json(kwargs["run_dir"] / "lego.bindings_stub.json", {"bindings": []})),
            "scene_pricing_stub_json": str(_write_json(kwargs["run_dir"] / "lego.pricing_stub.json", {"items": []})),
        },
    )
    monkeypatch.setattr(
        rp,
        "_run_supplier_modes_for_artifacts",
        lambda **kwargs: (
            supplier_scene,
            {"summary": {"replaced": 1}},
            {"bindings_json": str(tmp_path / "bindings.json")},
            {"mode": "optimal"},
            {
                **kwargs["manifest"],
                "supplier_variants": {
                    "optimal": {
                        "rebind": {"scene_v1": str(supplier_scene)},
                        "reports": {"summary_json": str(tmp_path / "summary.json")},
                    }
                },
            },
        ),
    )
    monkeypatch.setattr(rp, "maybe_repair_scene_json", lambda **kwargs: (kwargs["scene_json_path"], None))
    monkeypatch.setattr(rp, "_maybe_apply_layout_postprocess", lambda **kwargs: (kwargs["scene_json_path"], None))
    monkeypatch.setattr(rp, "_maybe_apply_flooring_to_scene", lambda **kwargs: (kwargs["scene_json_path"], None))
    monkeypatch.setattr(rp, "_maybe_apply_wall_material_to_scene", lambda **kwargs: (kwargs["scene_json_path"], None))
    monkeypatch.setattr(rp, "_maybe_apply_curtains_to_scene", lambda **kwargs: (kwargs["scene_json_path"], None))
    monkeypatch.setattr(rp, "_write_surface_material_pricing", lambda **kwargs: None)
    monkeypatch.setattr(rp, "_write_supplier_replacement_reports_for_artifacts", lambda **kwargs: {"summary_json": str(tmp_path / "replacement.json")})
    monkeypatch.setattr(rp, "run_blender_for_mode", lambda **kwargs: calls.append(f"base:{kwargs['variant_suffix']}"))
    monkeypatch.setattr(rp, "_run_supplier_blender_variants", lambda **kwargs: calls.append("supplier_variants"))
    monkeypatch.setattr(rp, "_refresh_supplier_reports_after_blender", lambda **kwargs: calls.append("refresh_reports"))
    def fake_finalize(**kwargs):
        manifest = {**kwargs["manifest"], "finalized": True}
        rp.write_json(kwargs["manifest_path"], manifest)
        return manifest

    monkeypatch.setattr(rp, "_finalize_supplier_variant_artifacts", fake_finalize)

    rp.run_pipeline_for_mode(
        cfg_runtime={},
        args=_pipeline_mode_args(placer="lego_gen", skip_blender=False, build_supplier_blend=True),
        room_path=str(room),
        run_dir=run_dir,
        layout_mode="lego_gen",
        prompt_text="prompt",
        style_profile_template={"style_label": "modern", "room_type": "Bedroom"},
    )

    assert calls == ["base:lego_gen", "supplier_variants", "refresh_reports"]
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["finalized"] is True


def test_run_pipeline_for_mode_reuse_and_placement_failure_edges(tmp_path: Path, monkeypatch):
    room = _write_json(tmp_path / "room.json", {"room": {"id": "r1", "floor_polygon": [{"x": 0, "y": 0}, {"x": 2, "y": 0}, {"x": 2, "y": 2}]}})

    reuse_dir = tmp_path / "reuse"
    reuse_dir.mkdir()
    _write_json(reuse_dir / "placement_infinigen_clean.json", {"placements": [{"id": "cached"}]})
    (reuse_dir / "infinigen_clean_scene.blend").write_bytes(b"blend")
    _mock_common_pipeline_stages(monkeypatch, tmp_path)
    monkeypatch.setattr(rp, "execute_placer", lambda **_kwargs: pytest.fail("execute_placer should not run for cached placement"))
    outputs = rp.run_pipeline_for_mode(
        cfg_runtime={},
        args=_pipeline_mode_args(skip_existing_placement=True, stop_after_placement=True),
        room_path=str(room),
        run_dir=reuse_dir,
        layout_mode="infinigen_clean",
        prompt_text="prompt",
        style_profile_template={"style_label": "minimalism", "room_type": "Bedroom"},
    )
    assert outputs.base_artifacts.scene_v1.name == "scene_base.v1.json"

    fatal_dir = tmp_path / "fatal"
    fatal_dir.mkdir()
    _mock_common_pipeline_stages(monkeypatch, tmp_path)
    monkeypatch.setattr(rp, "execute_placer", lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("No space left on device")))
    with pytest.raises(RuntimeError, match="full disk"):
        rp.run_pipeline_for_mode(
            cfg_runtime={},
            args=_pipeline_mode_args(max_attempts=2),
            room_path=str(room),
            run_dir=fatal_dir,
            layout_mode="infinigen_clean",
            prompt_text="prompt",
            style_profile_template={"style_label": "minimalism", "room_type": "Bedroom"},
        )

    max_dir = tmp_path / "max"
    max_dir.mkdir()
    _mock_common_pipeline_stages(monkeypatch, tmp_path)
    monkeypatch.setattr(rp, "execute_placer", lambda **_kwargs: (_ for _ in ()).throw(ValueError("still bad")))
    with pytest.raises(ValueError, match="still bad"):
        rp.run_pipeline_for_mode(
            cfg_runtime={},
            args=_pipeline_mode_args(max_attempts=1),
            room_path=str(room),
            run_dir=max_dir,
            layout_mode="infinigen_clean",
            prompt_text="prompt",
            style_profile_template={"style_label": "minimalism", "room_type": "Bedroom"},
        )

    retry_dir = tmp_path / "retry"
    retry_dir.mkdir()
    attempts = {"count": 0}

    def flaky_execute(**kwargs):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise ValueError("transient")
        _write_json(kwargs["out_path"], {"placements": [{"id": "ok"}]})

    _mock_common_pipeline_stages(monkeypatch, tmp_path)
    monkeypatch.setattr(rp, "execute_placer", flaky_execute)
    retry_outputs = rp.run_pipeline_for_mode(
        cfg_runtime={},
        args=_pipeline_mode_args(max_attempts=2, stop_after_placement=True),
        room_path=str(room),
        run_dir=retry_dir,
        layout_mode="infinigen_clean",
        prompt_text="prompt",
        style_profile_template={"style_label": "minimalism", "room_type": "Bedroom"},
    )
    assert attempts["count"] == 2
    assert retry_outputs.base_artifacts.scene_v1.name == "scene_base.v1.json"

    none_dir = tmp_path / "none"
    none_dir.mkdir()
    _mock_common_pipeline_stages(monkeypatch, tmp_path)
    with pytest.raises(RuntimeError, match="base placement"):
        rp.run_pipeline_for_mode(
            cfg_runtime={},
            args=_pipeline_mode_args(max_attempts=0),
            room_path=str(room),
            run_dir=none_dir,
            layout_mode="infinigen_clean",
            prompt_text="prompt",
            style_profile_template={"style_label": "minimalism", "room_type": "Bedroom"},
        )


def test_run_pipeline_for_mode_lego_skipped_chooser_and_explicit_preferences(tmp_path: Path, monkeypatch):
    room = _write_json(tmp_path / "room.json", {"room": {"id": "r1"}})
    run_dir = tmp_path / "lego_error"
    run_dir.mkdir()
    prefs = _write_json(tmp_path / "prefs.json", {"sites": ["unit"]})
    monkeypatch.setitem(rp.PLACER_SPECS, "lego_gen", {"requires_object_selection": False})
    monkeypatch.setattr(rp.secrets, "token_bytes", lambda n: b"\x05" * n)
    monkeypatch.setattr(rp, "maybe_run_semantic_room_planner_stage", lambda **_kwargs: None)

    with pytest.raises(RuntimeError, match="objects.v1.json"):
        rp.run_pipeline_for_mode(
            cfg_runtime={},
            args=_pipeline_mode_args(placer="lego_gen", supplier_user_preferences_json=str(prefs)),
            room_path=str(room),
            run_dir=run_dir,
            layout_mode="lego_gen",
            prompt_text="prompt",
            style_profile_template={"style_label": "modern", "room_type": "Bedroom", "supplier_preferences": {"ignored": True}},
        )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["supplier_preferences_json"] == str(prefs.resolve())


def test_run_pipeline_for_mode_lego_supplier_render_without_variants(tmp_path: Path, monkeypatch):
    room = _write_json(tmp_path / "room.json", {"room": {"id": "r1", "floor_polygon": [{"x": 0, "y": 0}, {"x": 4, "y": 0}, {"x": 4, "y": 3}]}})
    run_dir = tmp_path / "run_lego_supplier_render"
    run_dir.mkdir()
    monkeypatch.setattr(rp.secrets, "token_bytes", lambda n: b"\x06" * n)
    monkeypatch.setattr(rp, "maybe_run_semantic_room_planner_stage", lambda **_kwargs: None)
    monkeypatch.setattr(rp, "run_choose_stage", lambda **kwargs: _write_json(kwargs["run_dir"] / "objects.legacy.json", {"objects": [{"id": "chair"}]}))
    monkeypatch.setattr(rp, "normalize_json_artifact", lambda **kwargs: _write_json(kwargs["output_path"], {"objects": [{"id": "chair"}]}))

    def fake_lego(**kwargs):
        placement = _write_json(kwargs["run_dir"] / "lego.placement.json", {"placements": [{"id": "chair"}]})
        placement_v1 = _write_json(kwargs["run_dir"] / "lego.placement.v1.json", {"items": [{"id": "chair"}]})
        scene_v1 = _write_json(kwargs["run_dir"] / "lego.scene.v1.json", {"items": [{"id": "chair"}]})
        return rp.PlacementArtifacts(placement_legacy=placement, placement_v1=placement_v1, scene_v1=scene_v1, scene_legacy=None)

    supplier_scene = _write_json(run_dir / "lego.supplier.scene.json", {"items": [{"id": "chair"}]})
    calls: list[str] = []
    monkeypatch.setattr(rp, "run_lego_generate_from_scratch", fake_lego)
    monkeypatch.setattr(rp, "_maybe_apply_kitchen_stage", lambda **kwargs: (kwargs["artifacts"], None))
    monkeypatch.setattr(rp, "maybe_apply_procedural_room_stage", lambda **kwargs: kwargs["artifacts"])
    monkeypatch.setattr(
        rp,
        "_build_layout_selection_stub_for_artifacts",
        lambda **kwargs: {
            "layout_targets_json": str(_write_json(kwargs["run_dir"] / "lego.targets.json", {"targets": []})),
            "supplier_bindings_stub_json": str(_write_json(kwargs["run_dir"] / "lego.bindings_stub.json", {"bindings": []})),
            "scene_pricing_stub_json": str(_write_json(kwargs["run_dir"] / "lego.pricing_stub.json", {"items": []})),
        },
    )
    monkeypatch.setattr(
        rp,
        "_run_supplier_modes_for_artifacts",
        lambda **kwargs: (
            supplier_scene,
            {"summary": {"replaced": 1}},
            {"bindings_json": str(tmp_path / "bindings.json")},
            {"summary_json": str(tmp_path / "summary.json")},
            kwargs["manifest"],
        ),
    )
    monkeypatch.setattr(rp, "maybe_repair_scene_json", lambda **kwargs: (kwargs["scene_json_path"], None))
    monkeypatch.setattr(rp, "_maybe_apply_layout_postprocess", lambda **kwargs: (kwargs["scene_json_path"], None))
    monkeypatch.setattr(rp, "_maybe_apply_flooring_to_scene", lambda **kwargs: (kwargs["scene_json_path"], None))
    monkeypatch.setattr(rp, "_maybe_apply_wall_material_to_scene", lambda **kwargs: (kwargs["scene_json_path"], None))
    monkeypatch.setattr(rp, "_maybe_apply_curtains_to_scene", lambda **kwargs: (kwargs["scene_json_path"], None))
    monkeypatch.setattr(rp, "_write_surface_material_pricing", lambda **kwargs: None)
    monkeypatch.setattr(rp, "_write_supplier_replacement_reports_for_artifacts", lambda **kwargs: {"summary_json": str(tmp_path / "replacement.json")})
    monkeypatch.setattr(rp, "_maybe_apply_topview_vlm_orientation_repair", lambda **kwargs: (kwargs["scene_json_path"], {"tag": kwargs["tag"]}))
    monkeypatch.setattr(rp, "run_blender_for_mode", lambda **kwargs: calls.append(kwargs["variant_suffix"]))
    monkeypatch.setattr(rp, "blender_outputs_for_mode", lambda args, run_dir, layout_mode, variant_suffix: (_write_json(run_dir / f"{variant_suffix}.blend", {"blend": True}), None))
    monkeypatch.setattr(rp, "_render_supplier_room_gifs", lambda **kwargs: {"outputs": [{"gif": str(tmp_path / "supplier.gif")}]})

    rp.run_pipeline_for_mode(
        cfg_runtime={},
        args=_pipeline_mode_args(placer="lego_gen", skip_blender=False, build_supplier_blend=False),
        room_path=str(room),
        run_dir=run_dir,
        layout_mode="lego_gen",
        prompt_text="prompt",
        style_profile_template={"style_label": "modern", "room_type": "Bedroom"},
    )

    assert calls == ["lego_gen", "lego_gen_supplier"]
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["topview_vlm_orientation_repair_supplier"]["tag"] == "lego_gen_supplier"
    assert manifest["supplier_room_gifs"]["outputs"][0]["gif"].endswith("supplier.gif")


def test_run_pipeline_for_mode_lego_supplier_variants_build_disabled(tmp_path: Path, monkeypatch):
    room = _write_json(tmp_path / "room.json", {"room": {"id": "r1", "floor_polygon": [{"x": 0, "y": 0}, {"x": 4, "y": 0}, {"x": 4, "y": 3}]}})
    run_dir = tmp_path / "run_lego_variant_disabled"
    run_dir.mkdir()
    monkeypatch.setattr(rp.secrets, "token_bytes", lambda n: b"\x07" * n)
    monkeypatch.setattr(rp, "maybe_run_semantic_room_planner_stage", lambda **_kwargs: None)
    monkeypatch.setattr(rp, "run_choose_stage", lambda **kwargs: _write_json(kwargs["run_dir"] / "objects.legacy.json", {"objects": [{"id": "chair"}]}))
    monkeypatch.setattr(rp, "normalize_json_artifact", lambda **kwargs: _write_json(kwargs["output_path"], {"objects": [{"id": "chair"}]}))
    monkeypatch.setattr(
        rp,
        "run_lego_generate_from_scratch",
        lambda **kwargs: rp.PlacementArtifacts(
            placement_legacy=_write_json(kwargs["run_dir"] / "lego.placement.json", {"placements": []}),
            placement_v1=_write_json(kwargs["run_dir"] / "lego.placement.v1.json", {"items": []}),
            scene_v1=_write_json(kwargs["run_dir"] / "lego.scene.v1.json", {"items": []}),
            scene_legacy=None,
        ),
    )
    monkeypatch.setattr(rp, "_maybe_apply_kitchen_stage", lambda **kwargs: (kwargs["artifacts"], None))
    monkeypatch.setattr(rp, "maybe_apply_procedural_room_stage", lambda **kwargs: kwargs["artifacts"])
    monkeypatch.setattr(
        rp,
        "_build_layout_selection_stub_for_artifacts",
        lambda **kwargs: {
            "layout_targets_json": str(_write_json(kwargs["run_dir"] / "lego.targets.json", {"targets": []})),
            "supplier_bindings_stub_json": str(_write_json(kwargs["run_dir"] / "lego.bindings_stub.json", {"bindings": []})),
            "scene_pricing_stub_json": str(_write_json(kwargs["run_dir"] / "lego.pricing_stub.json", {"items": []})),
        },
    )
    supplier_scene = _write_json(run_dir / "supplier.scene.json", {"items": []})
    monkeypatch.setattr(
        rp,
        "_run_supplier_modes_for_artifacts",
        lambda **kwargs: (
            supplier_scene,
            {"summary": {"replaced": 1}},
            {"bindings_json": str(tmp_path / "bindings.json")},
            {"summary_json": str(tmp_path / "summary.json")},
            {
                **kwargs["manifest"],
                "supplier_variants": {
                    "optimal": {
                        "rebind": {"scene_v1": str(supplier_scene)},
                        "reports": {"summary_json": str(tmp_path / "summary.json")},
                    }
                },
            },
        ),
    )
    monkeypatch.setattr(rp, "maybe_repair_scene_json", lambda **kwargs: (kwargs["scene_json_path"], None))
    monkeypatch.setattr(rp, "_maybe_apply_layout_postprocess", lambda **kwargs: (kwargs["scene_json_path"], None))
    monkeypatch.setattr(rp, "_maybe_apply_flooring_to_scene", lambda **kwargs: (kwargs["scene_json_path"], None))
    monkeypatch.setattr(rp, "_maybe_apply_wall_material_to_scene", lambda **kwargs: (kwargs["scene_json_path"], None))
    monkeypatch.setattr(rp, "_maybe_apply_curtains_to_scene", lambda **kwargs: (kwargs["scene_json_path"], None))
    monkeypatch.setattr(rp, "_write_surface_material_pricing", lambda **kwargs: None)
    monkeypatch.setattr(rp, "_write_supplier_replacement_reports_for_artifacts", lambda **kwargs: {"summary_json": str(tmp_path / "replacement.json")})
    monkeypatch.setattr(rp, "run_blender_for_mode", lambda **_kwargs: None)

    def fake_finalize(**kwargs):
        rp.write_json(kwargs["manifest_path"], kwargs["manifest"])
        return kwargs["manifest"]

    monkeypatch.setattr(rp, "_finalize_supplier_variant_artifacts", fake_finalize)

    rp.run_pipeline_for_mode(
        cfg_runtime={},
        args=_pipeline_mode_args(placer="lego_gen", skip_blender=False, build_supplier_blend=False),
        room_path=str(room),
        run_dir=run_dir,
        layout_mode="lego_gen",
        prompt_text="prompt",
        style_profile_template={"style_label": "modern", "room_type": "Bedroom"},
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["supplier_variants"]["optimal"]["blender"]["blender_error"] == "build_supplier_blend_disabled"


def test_run_pipeline_for_mode_supplier_variant_blender_on_off_edges(tmp_path: Path, monkeypatch):
    room = _write_json(tmp_path / "room.json", {"room": {"id": "r1", "floor_polygon": [{"x": 0, "y": 0}, {"x": 3, "y": 0}, {"x": 3, "y": 3}]}})

    for build_supplier_blend, expected_calls in ((True, ["base:", "supplier_variants", "refresh_reports"]), (False, ["base:"])):
        run_dir = tmp_path / f"run_variant_{build_supplier_blend}"
        run_dir.mkdir()
        _mock_common_pipeline_stages(monkeypatch, tmp_path)
        supplier_scene = _write_json(run_dir / "supplier.scene.json", {"items": [{"id": "chair"}]})
        calls: list[str] = []

        def fake_supplier_modes(**kwargs):
            manifest = dict(kwargs["manifest"])
            manifest["supplier_variants"] = {
                "optimal": {
                    "rebind": {"scene_v1": str(supplier_scene)},
                    "reports": {"summary_json": str(tmp_path / "summary.json")},
                }
            }
            return supplier_scene, {"summary": {"replaced": 1}}, {"bindings_json": str(tmp_path / "bindings.json")}, {"summary_json": "summary.json"}, manifest

        monkeypatch.setattr(rp, "_run_supplier_modes_for_artifacts", fake_supplier_modes)
        monkeypatch.setattr(rp, "run_blender_for_mode", lambda **kwargs: calls.append(f"base:{kwargs['variant_suffix']}"))
        monkeypatch.setattr(rp, "_run_supplier_blender_variants", lambda **kwargs: calls.append("supplier_variants"))
        monkeypatch.setattr(rp, "_refresh_supplier_reports_after_blender", lambda **kwargs: calls.append("refresh_reports"))

        def fake_finalize(**kwargs):
            manifest = {**kwargs["manifest"], "finalized": True}
            rp.write_json(kwargs["manifest_path"], manifest)
            return manifest

        monkeypatch.setattr(rp, "_finalize_supplier_variant_artifacts", fake_finalize)
        rp.run_pipeline_for_mode(
            cfg_runtime={},
            args=_pipeline_mode_args(skip_blender=False, build_supplier_blend=build_supplier_blend),
            room_path=str(room),
            run_dir=run_dir,
            layout_mode="infinigen_clean",
            prompt_text="prompt",
            style_profile_template={"style_label": "minimalism", "room_type": "Bedroom"},
        )

        assert calls == expected_calls
        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        assert manifest["finalized"] is True
        if not build_supplier_blend:
            assert manifest["supplier_variants"]["optimal"]["blender"]["blender_error"] == "build_supplier_blend_disabled"


def test_run_pipeline_small_remaining_argument_and_policy_edges(tmp_path: Path, monkeypatch):
    room = _write_json(tmp_path / "room.json", {"room": {"id": "r"}})
    artifacts = types.SimpleNamespace(placement_v1=tmp_path / "placement.json", scene_v1=None)
    kitchen_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        rp,
        "apply_kitchen_stage_to_artifacts",
        lambda **kwargs: kitchen_calls.append(kwargs) or (kwargs["artifacts"], {"replacement_count": 0, "skipped_reason": "unit"}),
    )
    returned, info = rp._maybe_apply_kitchen_stage(
        args=types.SimpleNamespace(
            kitchens="off",
            kitchen_material_catalog="materials.json",
            kitchen_appliance_catalog="",
            kitchen_selection_mode="optimal",
            kitchen_dining="auto",
            kitchen_accessories="auto",
            kitchen_accessory_llm_provider="none",
            kitchen_llm_provider="none",
        ),
        artifacts=artifacts,
        run_dir=tmp_path,
        room_path=str(room),
        prompt_text="prompt",
        suffix="unit",
    )
    assert returned is artifacts
    assert info["skipped_reason"] == "unit"
    assert kitchen_calls[-1]["policy"] == "never"
    assert kitchen_calls[-1]["appliance_catalog"] is None

    class RaisingFloat(float):
        def __new__(cls, value=0.0):
            raise ValueError("unit")

    monkeypatch.setattr(rp, "float", RaisingFloat, raising=False)
    assert rp._to_float("12") is None
    monkeypatch.delattr(rp, "float", raising=False)

    assert rp._flooring_room_type({}, _write_json(tmp_path / "list_scene.json", [])) is None
    assert rp._flooring_room_type({}, _write_json(tmp_path / "room_list_scene.json", {"room": []})) is None
    assert rp._flooring_room_type({}, tmp_path / "missing_scene.json") is None

    with pytest.raises(RuntimeError, match="monkeypatch_params"):
        rp._maybe_apply_fast_infinigen_profile(
            types.SimpleNamespace(
                infinigen_fast_small=True,
                infinigen_no_pose_cameras=False,
                infinigen_solve_steps_large=None,
                infinigen_solve_steps_medium=None,
                infinigen_solve_steps_small=None,
            ),
            {"infinigen": {"monkeypatch_params": []}},
        )
    with pytest.raises(RuntimeError, match="overrides"):
        rp._maybe_apply_fast_infinigen_profile(
            types.SimpleNamespace(
                infinigen_fast_small=False,
                infinigen_no_pose_cameras=True,
                infinigen_solve_steps_large=None,
                infinigen_solve_steps_medium=None,
                infinigen_solve_steps_small=None,
            ),
            {"infinigen": {"overrides": {}}},
        )


def test_run_pipeline_curtain_relative_success_and_flooring_prompt_meta_error(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    scene = _write_json(tmp_path / "scene.json", {"room": {"windows": [{"id": "w"}]}, "items": []})
    materials = tmp_path / "materials"
    models = tmp_path / "models"
    materials.mkdir()
    models.mkdir()
    supplier_catalog = _write_json(tmp_path / "supplier.json", {"items": []})
    calls: dict[str, object] = {}

    def fake_load_curtain_catalog(path):
        calls["materials_path"] = path
        return [{"sku": "C1"}], materials

    def fake_discover_curtain_models(path):
        calls["models_dir"] = path
        return [tmp_path / "curtain.fbx"]

    monkeypatch.setattr(rp, "load_curtain_catalog", fake_load_curtain_catalog)
    monkeypatch.setattr(rp, "discover_curtain_models", fake_discover_curtain_models)
    monkeypatch.setattr(
        rp,
        "discover_supplier_curtain_models",
        lambda **kwargs: calls.__setitem__("supplier_catalog_path", kwargs["supplier_catalog_path"]) or {"model": tmp_path / "manual.fbx"},
    )
    monkeypatch.setattr(
        rp,
        "apply_curtains_to_scene",
        lambda scene_payload, **kwargs: (
            {**scene_payload, "items": [{"id": "curtain"}]},
            {"added_count": 1, "selected": [{"sku": "C1", "name": "Curtain", "texture_path": "tex.jpg"}]},
        ),
    )

    out_path, info = rp._maybe_apply_curtains_to_scene(
        args=_args(
            curtains="on",
            no_curtains=False,
            curtain_materials="materials",
            curtain_models_dir="models",
            curtain_supplier_catalog=str(supplier_catalog.name),
            curtain_seed=0,
            seed=9,
        ),
        run_dir=tmp_path,
        scene_json_path=scene,
        prompt_text="",
        style_profile={},
        suffix=".unit",
    )
    assert out_path.name == "scene.curtains.v1.json"
    assert info["policy"] == "always"
    assert info["needed_reason"] == "policy_always"
    assert Path(info["catalog_path"]).name == "materials"
    assert Path(info["models_dir"]).name == "models"
    assert Path(info["supplier_catalog_path"]).name == "supplier.json"

    (tmp_path / "infinigen_clean_meta.json").write_text("{bad", encoding="utf-8")
    assert rp._flooring_prompt_for_selector("", {"expanded_prompt": ""}, tmp_path) == ""


def test_run_supplier_modes_for_artifacts_remaining_edges(tmp_path: Path, monkeypatch):
    artifacts = rp.PlacementArtifacts(
        placement_legacy=_write_json(tmp_path / "placement.json", {"placements": []}),
        placement_v1=_write_json(tmp_path / "placement.v1.json", {"items": []}),
        scene_v1=_write_json(tmp_path / "scene.v1.json", {"items": []}),
        scene_legacy=None,
    )
    targets = _write_json(tmp_path / "targets.json", {"targets": []})
    manifest_path = _write_json(tmp_path / "manifest.json", {})
    bindings = _write_json(tmp_path / "bindings.json", {"bindings": [{"target_id": "chair", "metadata": {"mutate": True}}]})
    calls: dict[str, object] = {"binding_calls": 0, "skip_first": False}

    monkeypatch.setattr(
        rp,
        "_build_room_design_spec_for_targets",
        lambda **kwargs: (_write_json(kwargs["run_dir"] / "room_design_spec.json", {"style": "unit"}), {"style": "unit"}),
    )

    def fake_resolve(**kwargs):
        calls["binding_calls"] = int(calls["binding_calls"]) + 1
        calls["fallback_user_preferences"] = kwargs["supplier_user_preferences_json"]
        if bool(calls.get("skip_first")) and calls["binding_calls"] == 1:
            return None
        return bindings

    monkeypatch.setattr(rp, "_resolve_supplier_bindings_json", fake_resolve)
    monkeypatch.setattr(rp, "apply_supplier_scene_consistency", lambda data: {**data, "consistent": True})
    monkeypatch.setattr(
        rp,
        "_acquire_supplier_assets_for_bindings",
        lambda **kwargs: (kwargs["bindings_json_path"], {"bindings_json": str(kwargs["bindings_json_path"])}),
    )
    monkeypatch.setattr(
        rp,
        "_apply_supplier_bindings_for_artifacts",
        lambda **kwargs: {
            "scene_v1": str(_write_json(kwargs["run_dir"] / f"scene_{kwargs['variant_suffix']}.json", {"items": []})),
            "fallback_mode": kwargs["supplier_asset_fallback_mode"],
        },
    )
    monkeypatch.setattr(rp, "_write_supplier_replacement_reports_for_artifacts", lambda **kwargs: {"summary_json": str(tmp_path / "summary.json")})
    monkeypatch.setattr(rp, "_write_supplier_variants_comparison", lambda run_dir, variants: _write_json(run_dir / "comparison.json", {"variants": list(variants)}))
    monkeypatch.setattr(rp, "_write_supplier_variants_manifest", lambda **kwargs: _write_json(tmp_path / "variants_manifest.json", {"modes": kwargs["modes"]}))

    primary_scene, primary_info, primary_assets, primary_report, manifest = rp._run_supplier_modes_for_artifacts(
        args=_args(
            placer="infinigen_clean",
            supplier_asset_fallback_mode="proxy",
            supplier_selection_modes="",
            supplier_selection_mode="",
            supplier_selection_strategy="balanced",
            supplier_user_preferences_json="",
            supplier_require_local_asset=True,
        ),
        run_dir=tmp_path,
        artifacts=artifacts,
        layout_targets_json_path=str(targets),
        prompt_text="prompt",
        style_profile={},
        style_supplier_preferences_path=tmp_path / "style_supplier_preferences.json",
        manifest={},
        manifest_path=manifest_path,
    )

    assert calls["binding_calls"] == 1
    assert str(calls["fallback_user_preferences"]).endswith("style_supplier_preferences.json")
    assert primary_scene and primary_scene.name == "scene_optimal.json"
    assert primary_info["fallback_mode"] == rp.ASSET_FALLBACK_MODE_NONE
    assert primary_assets["bindings_json"].endswith("bindings.consistent.json")
    assert primary_report["summary_json"].endswith("summary.json")
    assert manifest["supplier_variants_manifest_json"].endswith("variants_manifest.json")

    calls["binding_calls"] = 0
    calls["skip_first"] = True
    _, _, _, _, skipped_manifest = rp._run_supplier_modes_for_artifacts(
        args=_args(
            placer="lego_gen",
            supplier_asset_fallback_mode="proxy",
            supplier_selection_modes="optimal,min_price",
            supplier_selection_mode="",
            supplier_selection_strategy="balanced",
            supplier_user_preferences_json="explicit.json",
            supplier_require_local_asset=False,
        ),
        run_dir=tmp_path,
        artifacts=artifacts,
        layout_targets_json_path=str(targets),
        prompt_text="prompt",
        style_profile={},
        style_supplier_preferences_path=tmp_path / "style_supplier_preferences.json",
        manifest={},
        manifest_path=_write_json(tmp_path / "manifest_skip.json", {}),
    )
    assert calls["binding_calls"] == 2
    assert "cheapest" in skipped_manifest["supplier_variants"]


def test_main_wires_config_style_modes_and_cleanup_without_real_pipeline(tmp_path: Path, monkeypatch, capsys):
    room = _write_json(tmp_path / "room.json", {"room": {"id": "r"}})
    cfg_path = tmp_path / "paths.yaml"
    cfg_path.write_text("paths: {}\n", encoding="utf-8")

    parsed_args = _args(
        paths_config=str(cfg_path),
        room=str(room),
        ollama_models=["json-a", "json-b"],
        plan_models=["plan-a"],
        critic_models=["critic-a"],
        plan_think="low",
        critic_think="medium",
        llm_think="none",
        style_ollama_models=[],
        style_ollama_model="style-single",
        style_llm_provider="none",
        style_llm_think="unsupported",
        style_ollama_temperature=None,
        style_ollama_url="",
        style_ollama_timeout=None,
        style_llm_max_attempts=None,
        style_llm_debug_dir="",
        ollama_model="fallback-model",
        ollama_url="http://ollama",
        ollama_timeout=7,
        ollama_temperature=0.25,
        ollama_max_attempts=2,
        lego_postprocess=True,
        keep_tmp=False,
        run_dir="",
    )
    parser = types.SimpleNamespace(parse_args=lambda: parsed_args)
    run_dirs = [tmp_path / "run_a", tmp_path / "run_b"]
    for path in run_dirs:
        path.mkdir()
    calls = {"pipeline": [], "style": None, "lego": 0}

    monkeypatch.setattr(rp, "build_cli", lambda: parser)
    monkeypatch.setattr(rp, "load_yaml", lambda path: {"loaded": str(path)})
    monkeypatch.setattr(rp, "project_root_from_config", lambda cfg, cfg_path: tmp_path)
    monkeypatch.setattr(rp, "apply_config_defaults", lambda args, cfg, cfg_base_dir: setattr(args, "config_applied", True))
    monkeypatch.setattr(rp, "build_runtime_paths", lambda cfg, cfg_base_dir: {"DEFAULT_ROOM_JSON": str(room), "TMP_ROOT": tmp_path})
    monkeypatch.setattr(rp, "parse_modes", lambda args, cfg: ["mode_a", "mode_b"])
    monkeypatch.setattr(rp, "resolve_lego_generation_params", lambda args: calls.__setitem__("lego", calls["lego"] + 1) or {
        "preset": "unit",
        "method": "direct_map",
        "init_scene_mode": "perturb",
        "outer_passes": 1,
        "num_restarts": 1,
        "init_pos_noise_std": 0.0,
        "init_ang_noise_deg": 0.0,
    })
    monkeypatch.setattr(rp, "read_prompt_from_args", lambda args: "unit prompt")

    def fake_style(**kwargs):
        calls["style"] = kwargs
        return {"style_label": "minimalism", "room_type": "Bedroom", "confidence": 0.8}

    monkeypatch.setattr(rp, "analyze_prompt_to_style_profile", fake_style)
    monkeypatch.setattr(rp, "make_mode_run_dir", lambda tmp_root, layout_mode, run_dir: (run_dirs.pop(0), None))
    monkeypatch.setattr(
        rp,
        "run_pipeline_for_mode",
        lambda **kwargs: calls["pipeline"].append((kwargs["layout_mode"], Path(kwargs["run_dir"]).name, kwargs["style_profile_template"]["style_label"])),
    )

    rp.main()

    assert parsed_args.config_applied is True
    assert calls["style"]["think"] == "low"
    assert calls["style"]["ollama_models"] == ["json-a", "json-b"]
    assert calls["pipeline"] == [("mode_a", "run_a", "minimalism"), ("mode_b", "run_b", "minimalism")]
    assert calls["lego"] == 1
    assert not (tmp_path / "run_a").exists()
    assert not (tmp_path / "run_b").exists()
    out = capsys.readouterr().out
    assert "ВСЕ РЕЖИМЫ" in out


def test_main_uses_single_style_model_fallback(tmp_path: Path, monkeypatch):
    room = _write_json(tmp_path / "room.json", {"room": {"id": "r"}})
    cfg_path = tmp_path / "paths.yaml"
    cfg_path.write_text("paths: {}\n", encoding="utf-8")
    parsed_args = _args(
        paths_config=str(cfg_path),
        room=str(room),
        ollama_models=[],
        plan_models=["plan-a"],
        critic_models=["critic-a"],
        plan_think="low",
        critic_think="low",
        llm_think="none",
        style_ollama_models=[],
        style_ollama_model="",
        style_llm_provider="none",
        style_llm_think="low",
        style_ollama_temperature=0.2,
        style_ollama_url="",
        style_ollama_timeout=5,
        style_llm_max_attempts=1,
        style_llm_debug_dir="",
        ollama_model="fallback-style",
        ollama_url="http://ollama",
        ollama_timeout=7,
        ollama_temperature=0.25,
        ollama_max_attempts=2,
        lego_postprocess=False,
        keep_tmp=True,
        run_dir=str(tmp_path / "run"),
    )
    calls: dict[str, object] = {}
    monkeypatch.setattr(rp, "build_cli", lambda: types.SimpleNamespace(parse_args=lambda: parsed_args))
    monkeypatch.setattr(rp, "load_yaml", lambda path: {"loaded": str(path)})
    monkeypatch.setattr(rp, "project_root_from_config", lambda cfg, cfg_path: tmp_path)
    monkeypatch.setattr(rp, "apply_config_defaults", lambda args, cfg, cfg_base_dir: None)
    monkeypatch.setattr(rp, "build_runtime_paths", lambda cfg, cfg_base_dir: {"DEFAULT_ROOM_JSON": str(room), "TMP_ROOT": tmp_path})
    monkeypatch.setattr(rp, "parse_modes", lambda args, cfg: ["mode_a"])
    monkeypatch.setattr(rp, "read_prompt_from_args", lambda args: "unit prompt")
    def fake_style(**kwargs):
        calls["style_models"] = kwargs["ollama_models"]
        return {"style_label": "minimalism", "room_type": "Bedroom", "confidence": 0.8}

    monkeypatch.setattr(rp, "analyze_prompt_to_style_profile", fake_style)
    monkeypatch.setattr(rp, "make_mode_run_dir", lambda tmp_root, layout_mode, run_dir: (tmp_path / "run", None))
    monkeypatch.setattr(rp, "run_pipeline_for_mode", lambda **kwargs: calls.setdefault("mode", kwargs["layout_mode"]))

    rp.main()

    assert calls["style_models"] == ["fallback-style"]
    assert calls["mode"] == "mode_a"
