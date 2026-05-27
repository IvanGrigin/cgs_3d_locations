import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.tools import run_procedural_room_supplier as rprs
from tests.helpers.scene_builders import placement_scene_with_room, scene_with_room


def test_patch_block_wraps_existing_read_write_json_defs(tmp_path):
    module_path = Path(__file__).resolve().parents[1] / "src" / "tools" / "run_procedural_room_supplier.py"
    calls = []

    def fake_read_json(_path):
        calls.append("read")
        return {"placements": [{"id": "chair", "category": "chair", "meta": {"procedural": True}}]}

    def fake_write_json(path, payload, *args, **kwargs):
        calls.append(("write", Path(path), payload, args, kwargs))
        return "written"

    source = "\n".join(module_path.read_text(encoding="utf-8").splitlines()[:150])
    ns = {
        "__file__": str(module_path),
        "read_json": fake_read_json,
        "write_json": fake_write_json,
    }
    exec(compile(source, str(module_path), "exec"), ns)

    patched_read = ns["read_json"]
    patched_write = ns["write_json"]
    payload = patched_read(tmp_path / "scene.json")
    assert payload["placements"][0]["replace_with_supplier"] is True
    assert calls == ["read"]

    result = patched_write(tmp_path / "out.json", {"items": [{"id": "bed", "category": "bed", "source": {"generator": "bedroom_generator"}}]})
    assert result == "written"
    assert calls[-1][0] == "write"
    assert calls[-1][2]["items"][0]["replace_with_supplier"] is True


def _trellis_args(**overrides):
    values = {
        "trellis_remote_cuda_visible_devices": "0,1",
        "trellis_server_host": "gpu.example",
        "trellis_server_port": 28553,
        "trellis_server_user": "root",
        "trellis_ssh_key": "~/.ssh/id",
        "trellis_remote_root": "/workspace/jobs",
        "trellis_remote_trellis_root": "/workspace/TRELLIS.2",
        "trellis_remote_model_dir": "/workspace/models/TRELLIS.2-4B",
        "trellis_remote_text_model_dir": "",
        "trellis_remote_python": "/venv/trellis2/bin/python",
        "trellis_remote_worker_root": "/workspace/worker",
        "trellis_remote_worker_timeout_sec": 100.0,
        "trellis_remote_worker_poll_sec": 1.0,
        "trellis_remote_persistent_worker": True,
        "trellis_multi_mode": "stochastic",
        "trellis_max_images": 2,
        "trellis_oom_retry_max_images": 1,
        "trellis_seed": 11,
        "trellis_sparse_steps": 4,
        "trellis_slat_steps": 5,
        "trellis_texture_size": 256,
        "trellis_simplify": 0.98,
        "trellis_pipeline_type": 512,
        "trellis_ss_guidance_strength": 7.5,
        "trellis_slat_guidance_strength": 3.0,
        "trellis_decimation_target": 50000,
        "trellis_pre_export_simplify_target": 0,
        "trellis_no_remesh": False,
        "trellis_remesh_band": 1,
        "trellis_remesh_project": 0.0,
        "trellis_no_webp": True,
        "trellis_image_size": 336,
        "trellis_fill_holes_resolution": 256,
        "trellis_fill_holes_num_views": 120,
        "trellis_force_image_only": False,
        "trellis_max_candidate_pool": 0,
        "trellis_remote_runner_path": "",
        "trellis_vlm_single_object_filter": False,
        "trellis_vlm_provider": "ollama",
        "trellis_vlm_ollama_url": "http://127.0.0.1:11435",
        "trellis_vlm_model": "llama3.2-vision:11b",
        "trellis_vlm_timeout": 120,
        "trellis_vlm_unload_after_filter": True,
        "trellis_text_fallback_if_no_single_image": True,
        "trellis_allow_proxy_fallback": False,
        "trellis_ikea_mebelru_images_only": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_procedural_patch_forces_supplier_targets():
    payload = {
        "placements": [
            {"id": "bed", "category": "bed", "meta": {"procedural": True}},
            {"id": "pillow", "category": "pillow", "source": {"placement_source": "procedural_room_stage"}},
            {"id": "manual", "category": "chair"},
        ]
    }

    patched = rprs._cgs_trellis33_force_supplier_targets_payload(payload)

    assert patched["placements"][0]["replace_with_supplier"] is True
    assert patched["placements"][0]["meta"]["replace_with_supplier"] is True
    assert patched["placements"][1]["meta"]["replace_with_supplier"] is False
    assert "trellis33_force_supplier_targets_patch" in patched["meta"]
    assert rprs._cgs_trellis33_item_is_procedural({"source": {"generator": "bedroom_generator"}})
    assert rprs._cgs_trellis33_category({"type": "Desk"}) == "desk"


def test_candidate_assets_and_trellis_card_building(tmp_path):
    glb = tmp_path / "asset.glb"
    glb.write_bytes(b"glb")
    candidate = {
        "unique_key": "u1",
        "asset_local_path": str(glb),
        "extra": {"trellis_generated_asset": {"asset_local_path": str(glb)}},
    }

    assert rprs.candidate_asset_paths(candidate) == [glb.resolve()]
    assert rprs.candidate_has_supported_local_asset(candidate)
    assert not rprs.candidate_has_supported_local_asset({"asset_local_path": str(tmp_path / "missing.glb")})

    patched_card = {
        "asset_status": "trellis_generated_local_asset",
        "asset_format": "glb",
        "asset_local_path": str(glb),
        "asset_source_url": "https://example/model",
        "extra": {"trellis_generated_asset": {"asset_local_path": str(glb), "target_size_m": [1, 2, 3]}},
    }
    rprs.apply_trellis_card_to_candidate(candidate, patched_card)
    assert candidate["asset_local_path"] == str(glb)
    assert candidate["extra"]["trellis_generated_asset"]["target_size_m"] == [1, 2, 3]

    binding = {
        "target_id": "target1",
        "category": "chair",
        "semantic_group": "seat",
        "requested_size_m": [0.5, 0.6, 0.9],
        "candidate_pool": [{"unique_key": "u1"}, {"unique_key": "u2"}],
        "meta": {"top_candidates": [{"unique_key": "u2"}, {"unique_key": "u3"}]},
    }
    card = rprs.build_trellis_card_from_binding(binding, {"unique_key": "chosen"})
    assert card["target_id"] == "target1"
    assert card["target_size_m"] == [0.5, 0.6, 0.9]
    assert [c["unique_key"] for c in card["top_candidates"]] == ["u1", "u2", "u3"]
    assert card["binding"]["target_id"] == "target1"


def test_ikea_mebelru_filter_and_retry_helpers():
    candidate = {
        "unique_key": "c1",
        "product_url": "https://www.mebel.ru/product/1",
        "images": ["https://cdn.example/a.jpg", "https://ikea.com/b.jpg"],
    }
    filtered, info = rprs._trellis_filter_candidate_ikea_mebelru_images_only(candidate)
    assert filtered is not None
    assert info["source_allowed"] is True
    assert rprs._trellis_allowed_ikea_mebelru_host("shop.mebel.ru")
    assert rprs._trellis_allowed_ikea_mebelru_host("") is False
    assert rprs._trellis_allowed_ikea_mebelru_text("https://ikea.de/item")
    assert rprs._trellis_allowed_ikea_mebelru_text("s3://ikea.com/item")
    assert rprs._trellis_allowed_ikea_mebelru_text("www.mebel.ru")
    assert rprs._trellis_url_host("https://ikea.de/item") == "ikea.de"

    class BadStr:
        def __str__(self):
            raise RuntimeError("bad str")

    assert rprs._trellis_url_host(BadStr()) == ""
    assert rprs._trellis_image_item_allowed_ikea_mebelru({"url": "https://ikea.com/a.jpg"}, candidate_source_allowed=False)
    assert rprs._trellis_candidate_has_usable_image_source({"preview_path": "local.jpg"})
    assert rprs._trellis_candidate_has_usable_image_source({"image_color_features": {"source_image": {"value": "local.jpg"}}})

    bad, bad_info = rprs._trellis_filter_candidate_ikea_mebelru_images_only({"unique_key": "bad", "images": ["https://other/a.jpg"]})
    assert bad is None
    assert bad_info["kept_image_sources"] == 0

    rich_candidate, rich_info = rprs._trellis_filter_candidate_ikea_mebelru_images_only(
        {
            "unique_key": "rich",
            "preview_local_path": "https://ikea.com/a.jpg",
            "extra": {"preview_path": "https://mebel.ru/b.jpg"},
            "image_color_features": {"source_image": {"value": "https://ikea.com/c.jpg"}},
        }
    )
    assert rich_candidate is not None
    assert rich_info["kept_image_sources"] == 3
    assert rprs._trellis_candidate_source_allowed_ikea_mebelru({"extra": {"source_url": "https://ikea.com/product"}})

    preview_fallback, _ = rprs._trellis_filter_candidate_ikea_mebelru_images_only(
        {"unique_key": "preview", "preview_images": ["https://ikea.com/preview.jpg"]}
    )
    assert preview_fallback["images"] == ["https://ikea.com/preview.jpg"]
    photos_fallback, _ = rprs._trellis_filter_candidate_ikea_mebelru_images_only(
        {"unique_key": "photos", "images_json": "{bad", "photos": ["https://ikea.com/photo.jpg"]}
    )
    assert photos_fallback["images"] == ["https://ikea.com/photo.jpg"]

    card, card_info = rprs._trellis_filter_card_ikea_mebelru_images_only(
        {"target_id": "t1", "candidate_pool": [candidate, candidate, {"unique_key": "bad", "images": ["https://other/a.jpg"]}]}
    )
    assert card is not None
    assert card_info["kept_candidate_count"] == 1

    assert rprs.parse_trellis_gpu_ids("0, 2;3") == [0, 2, 3]
    assert rprs.parse_trellis_gpu_ids("0,,1") == [0, 1]
    assert rprs.trellis_image_count_attempts(argparse.Namespace(trellis_max_images=2, trellis_oom_retry_max_images=1)) == [2, 1]
    oom = subprocess.CalledProcessError(1, ["cmd"], output="CUDA out of memory")
    assert rprs.is_trellis_oom_error(oom)
    assert "CUDA out of memory" in rprs.exception_text(oom)
    err = subprocess.CalledProcessError(2, ["cmd"], stderr="stderr text")
    assert "stderr text" in str(err)

    with pytest.raises(SystemExit, match="Legacy TRELLIS"):
        rprs.validate_trellis2_only_cli_args(
            argparse.Namespace(
                trellis_remote_trellis_root="/workspace/TRELLIS",
                trellis_remote_model_dir="/workspace/trellis_models/old",
                trellis_remote_text_model_dir="",
                trellis_remote_python="/venv/trellis/bin/python",
                trellis_remote_runner_path="old.py",
            )
        )


def test_trellis_patch_filter_and_argument_edge_branches(tmp_path, monkeypatch):
    assert rprs._cgs_trellis33_item_is_procedural(None) is False
    assert rprs._cgs_trellis33_category(None) == ""
    assert rprs._cgs_trellis33_force_supplier_targets_payload(["not", "dict"]) == ["not", "dict"]
    assert rprs._cgs_trellis33_force_supplier_targets_payload({"placements": "bad"}) == {"placements": "bad"}
    assert rprs._cgs_trellis33_force_supplier_targets_payload({"items": [{"category": "chair"}]}) == {
        "items": [{"category": "chair"}]
    }

    monkeypatch.setattr(rprs, "_CGS_TRELLIS33_DISABLE", True)
    disabled_payload = {"placements": [{"category": "chair", "meta": {"procedural": True}}]}
    assert rprs._cgs_trellis33_force_supplier_targets_payload(disabled_payload) is disabled_payload
    monkeypatch.setattr(rprs, "_CGS_TRELLIS33_DISABLE", False)

    patched = rprs._cgs_trellis33_force_supplier_targets_payload(
        {
            "items": [
                {"id": "bad-meta", "category": "desk", "meta": "bad", "source": {"generator": "bedroom_generator"}},
                "skip-me",
            ]
        }
    )
    assert patched["items"][0]["meta"]["replace_with_supplier"] is True
    assert patched["meta"]["trellis33_force_supplier_targets_patch"]["enabled"] is True

    assert rprs.candidate_asset_paths(None) == []
    text = tmp_path / "asset.txt"
    glb = tmp_path / "asset.glb"
    text.write_text("txt", encoding="utf-8")
    glb.write_bytes(b"glb")
    assert rprs.candidate_asset_paths({"asset_local_path": [str(text), str(glb), str(glb)]}) == [glb.resolve()]
    nested = {"extra": {"trellis_generated_asset": {"asset_local_path": str(glb)}}}
    assert rprs.candidate_has_supported_local_asset(nested)

    binding = {
        "target_id": "t",
        "category": "chair",
        "binding": {"candidate_pool": [{"unique_key": "b1"}]},
        "source": {"alternatives": [{"unique_key": "s1"}]},
        "selected_candidates": [{"unique_key": "sel"}],
        "all_candidates": [{"unique_key": "b1"}, {"id": "anon"}],
    }
    card = rprs.build_trellis_card_from_binding(binding, {"unique_key": "chosen", "meta": {"top_candidates": ["bad"]}})
    assert [c.get("unique_key") or c.get("id") for c in card["top_candidates"]] == ["b1", "s1", "sel", "anon"]
    assert card["binding"]["top_candidates"] == card["top_candidates"]

    assert rprs._trellis_allowed_ikea_mebelru_text("") is False
    assert rprs._trellis_allowed_ikea_mebelru_text("mebel-ru") is True
    assert rprs._trellis_filter_image_list_ikea_mebelru("not-json", candidate_source_allowed=False) == ("not-json", 0, 0)
    filtered_json, kept, seen = rprs._trellis_filter_image_list_ikea_mebelru(
        json.dumps(["https://other.test/a.jpg", "https://ikea.com/a.jpg"]),
        candidate_source_allowed=False,
    )
    assert json.loads(filtered_json) == ["https://ikea.com/a.jpg"]
    assert (kept, seen) == (1, 2)

    filtered, info = rprs._trellis_filter_candidate_ikea_mebelru_images_only(
        {
            "unique_key": "x",
            "preview_local_path": "https://other.test/a.jpg",
            "images_json": json.dumps(["https://ikea.com/b.jpg"]),
            "extra": {"preview_path": "https://other.test/c.jpg", "photos": ["https://mebel.ru/d.jpg"]},
            "image_color_features": {"source_image": {"path": "https://other.test/e.jpg"}},
        }
    )
    assert filtered is not None
    assert filtered["images"] == ["https://ikea.com/b.jpg"]
    assert "preview_local_path" not in filtered
    assert "preview_path" not in filtered["extra"]
    assert info["seen_image_sources"] >= 5

    empty_card, empty_info = rprs._trellis_filter_card_ikea_mebelru_images_only(
        {"unique_key": "root", "images": ["https://other.test/a.jpg"], "candidate_pool": ["bad"]}
    )
    assert empty_card is None
    assert empty_info["status"] == "empty_after_filter"
    replacement_card, replacement_info = rprs._trellis_filter_card_ikea_mebelru_images_only(
        {
            "target_id": "t",
            "binding": {"category": "chair"},
            "images": ["https://other.test/root.jpg"],
            "candidate_pool": [{"unique_key": "ikea", "images": ["https://ikea.com/a.jpg"]}],
        }
    )
    assert replacement_card["unique_key"] == "ikea"
    assert replacement_info["root_replaced"] is True
    bound_card, bound_info = rprs._trellis_filter_card_ikea_mebelru_images_only(
        {"target_id": "bound", "binding": {"candidate_pool": [{"unique_key": "b", "images": ["https://ikea.com/b.jpg"]}]}}
    )
    assert bound_card is not None
    assert bound_info["candidate_count"] >= 1

    assert rprs.parse_trellis_gpu_ids("") == [0]
    assert rprs.parse_trellis_gpu_ids(None) == [0]
    assert rprs.trellis_image_count_attempts(argparse.Namespace(trellis_max_images=0, trellis_oom_retry_max_images=0)) == [2]
    with pytest.raises(SystemExit, match="remote-runner-path"):
        rprs.validate_trellis2_only_cli_args(
            argparse.Namespace(
                trellis_remote_trellis_root="/workspace/TRELLIS.2",
                trellis_remote_model_dir="/workspace/models/TRELLIS.2-4B",
                trellis_remote_text_model_dir="/workspace/trellis_models/old",
                trellis_remote_python="/venv/trellis2/bin/python",
                trellis_remote_runner_path="/tmp/runner.py",
            )
        )


def test_build_trellis_args_and_oom_retries(tmp_path, monkeypatch):
    args = _trellis_args()
    card_json = tmp_path / "card.json"
    card_json.write_text("{}", encoding="utf-8")

    built = rprs.build_trellis_args(args, card_json=card_json, job_id="job", out_dir=tmp_path, gpu_id=1, max_images=1)
    assert built.server_host == "gpu.example"
    assert built.remote_cuda_visible_devices == 1
    assert built.max_images == 1
    assert built.remote_trellis_root == "/workspace/TRELLIS.2"

    calls = []

    def fake_run_orchestration(trellis_args):
        calls.append((trellis_args.job_id, trellis_args.remote_cuda_visible_devices, trellis_args.max_images))
        if len(calls) == 1:
            raise subprocess.CalledProcessError(1, ["trellis"], output="torch.cuda.OutOfMemoryError")
        return {"ok": True, "card_with_trellis_asset_json": str(tmp_path / "patched.json"), "local_job_dir": str(tmp_path)}

    monkeypatch.setattr(rprs, "run_orchestration", fake_run_orchestration)
    attempts = []
    summary = rprs.run_trellis_with_oom_retries(args, card_json=card_json, job_id="job", out_dir=tmp_path, attempts=attempts)

    assert summary["ok"] is True
    assert attempts[0]["oom"] is True
    assert attempts[1]["status"] == "success"
    assert calls[:2] == [("job", 0, 2), ("job_retry2_gpu1_img2", 1, 2)]

    def non_oom_failure(_trellis_args):
        raise RuntimeError("network failed")

    monkeypatch.setattr(rprs, "run_orchestration", non_oom_failure)
    with pytest.raises(RuntimeError, match="network failed"):
        rprs.run_trellis_with_oom_retries(args, card_json=card_json, job_id="job", out_dir=tmp_path, attempts=[])

    def oom_failure(_trellis_args):
        raise subprocess.CalledProcessError(1, ["trellis"], output="CUDA out of memory")

    monkeypatch.setattr(rprs, "run_orchestration", oom_failure)
    with pytest.raises(RuntimeError, match="TRELLIS failed after CUDA OOM retries"):
        rprs.run_trellis_with_oom_retries(
            _trellis_args(trellis_remote_cuda_visible_devices="0", trellis_max_images=1, trellis_oom_retry_max_images=0),
            card_json=card_json,
            job_id="job",
            out_dir=tmp_path,
            attempts=[],
        )


def test_room_door_curtain_detection_and_blender_command(tmp_path, monkeypatch):
    room_payload = {
        "room": {
            "id": "r1",
            "floor_polygon": [[0, 0], [4, 0], [4, 3], [0, 3]],
            "windows": [{"id": "w0", "wall_id": "w1", "s": 1.0, "width": 1.2}],
        }
    }
    ensured, info = rprs.ensure_room_has_door(room_payload)
    assert info["changed"] is True
    assert ensured["room"]["doors"][0]["type"] == "door"
    assert rprs._room_has_door(ensured["room"])
    assert rprs._room_openings_from_anywhere(ensured["room"], "windows")[0]["id"] == "w0"
    assert rprs._room_wall_points(ensured["room"], ensured["room"]["walls"][0]) == ((0.0, 0.0), (4.0, 0.0))

    scene = {
        "room": ensured["room"],
        "items": [
            {
                "id": "curtain_placeholder",
                "category": "curtain",
                "asset": {"kind": "procedural_placeholder"},
                "meta": {"procedural": True},
            }
        ],
        "placements": [
            {
                "id": "curtain_placeholder",
                "category": "curtain",
                "asset": {"kind": "procedural_placeholder"},
                "meta": {"procedural": True},
            }
        ],
    }
    stripped, removed = rprs._strip_generated_placeholder_curtains(scene)
    assert removed == 2
    assert stripped["items"] == []
    assert rprs._curtains_needed_for_scene(scene=stripped, prompt_text="add curtains", style_profile={}, policy="auto") == (
        True,
        "prompt_mentions_curtains",
    )
    assert rprs._curtains_needed_for_scene(scene=stripped, prompt_text="no curtains", style_profile={}, policy="auto")[0] is False

    scene_path = tmp_path / "scene.json"
    scene_path.write_text(json.dumps(scene), encoding="utf-8")
    calls = []
    monkeypatch.setattr(rprs.subprocess, "run", lambda cmd, check=True: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0))
    args = argparse.Namespace(
        build_blend=True,
        scene_builder_script="src/Plasement/blender_scene_builder.py",
        out_blend="room.blend",
        out_png="room.png",
        blender="/Applications/Blender.app/Contents/MacOS/Blender",
    )
    artifacts = rprs.maybe_build_blend_scene(args, supplier_scene=scene_path, out_dir=tmp_path)
    assert artifacts["blend"].endswith("room.blend")
    assert "--background" in artifacts["command"]
    assert calls[0][0].endswith("Blender")

    args.build_blend = False
    assert rprs.maybe_build_blend_scene(args, supplier_scene=scene_path, out_dir=tmp_path) is None


def test_room_supplier_extra_skip_and_error_edges(tmp_path, monkeypatch):
    explicit = argparse.Namespace(blender="/custom/blender")
    assert rprs.blender_binary(explicit) == "/custom/blender"
    real_is_file = Path.is_file
    monkeypatch.setattr(Path, "is_file", lambda self: str(self) == "/Applications/Blender.app/Contents/MacOS/Blender")
    assert rprs.blender_binary(argparse.Namespace(blender="")).endswith("Blender")
    monkeypatch.setattr(Path, "is_file", lambda self: False)
    assert rprs.blender_binary(argparse.Namespace(blender="")) == "blender"
    monkeypatch.setattr(Path, "is_file", real_is_file)

    class WeirdRoom:
        def get(self, _key):
            return "not-a-room"

    missing_room, missing_info = rprs.ensure_room_has_door(WeirdRoom())
    assert isinstance(missing_room, WeirdRoom)
    assert missing_info["reason"] == "missing_room"
    normalized, normalized_info = rprs.ensure_room_has_door(
        {
            "room": {
                "openings": {
                    "doors": [{"id": "d", "wall_id": "w0", "s": 0.1}],
                    "windows": [{"id": "w", "wall_id": "w1", "s": 0.2}],
                }
            }
        }
    )
    assert normalized_info["reason"] == "normalized_existing_openings"
    assert normalized["room"]["doors"][0]["id"] == "d"
    with_openings_list, openings_info = rprs.ensure_room_has_door(
        {"room": {"floor_polygon": [[0, 0], [4, 0], [4, 3]], "openings": [{"type": "window", "id": "w"}]}}
    )
    assert openings_info["changed"] is True
    assert any(item.get("type") == "door" for item in with_openings_list["room"]["openings"])
    no_wall, no_wall_info = rprs.ensure_room_has_door({"room": {"floor_polygon": [[0, 0], [0.5, 0], [0.5, 0.5]]}})
    assert no_wall_info["reason"] == "no_wall_long_enough"

    scene_path = scene_with_room().write(tmp_path / "scene.json")
    args = argparse.Namespace(topview_vlm_orientation_repair=True, build_blend=False)
    unchanged, info, artifacts = rprs.maybe_apply_topview_vlm_orientation_repair(
        args,
        supplier_scene=scene_path,
        out_dir=tmp_path,
        blend_artifacts=None,
    )
    assert unchanged == scene_path
    assert info["skipped_reason"] == "requires_build_blend"
    assert artifacts is None

    args.build_blend = True
    missing_blend = {"blend": str(tmp_path / "missing.blend")}
    unchanged, info, _ = rprs.maybe_apply_topview_vlm_orientation_repair(
        args,
        supplier_scene=scene_path,
        out_dir=tmp_path,
        blend_artifacts=missing_blend,
    )
    assert info["skipped_reason"] == "blend_not_found"

    blend = tmp_path / "scene.blend"
    blend.write_bytes(b"blend")
    monkeypatch.setattr(rprs, "read_json", lambda _path: [])
    unchanged, info, _ = rprs.maybe_apply_topview_vlm_orientation_repair(
        args,
        supplier_scene=scene_path,
        out_dir=tmp_path,
        blend_artifacts={"blend": str(blend)},
    )
    assert info["skipped_reason"] == "invalid_scene_json"
    monkeypatch.setattr(rprs, "read_json", lambda _path: {"items": []})
    monkeypatch.setattr(rprs, "collect_topview_vlm_scene_objects", lambda scene, max_objects: [])
    monkeypatch.setattr(rprs, "filter_topview_vlm_target_objects", lambda refs, **kwargs: [])
    args.topview_vlm_target_scope = "all"
    args.topview_vlm_include_armchairs = True
    args.topview_vlm_max_objects = 10
    unchanged, info, _ = rprs.maybe_apply_topview_vlm_orientation_repair(
        args,
        supplier_scene=scene_path,
        out_dir=tmp_path,
        blend_artifacts={"blend": str(blend)},
    )
    assert info["skipped_reason"] == "no_target_objects"

    assert rprs._room_polygon_points({"floor_polygon": "bad"}) == []
    assert rprs._floor_area_m2({"room": {"floor_polygon": "bad"}}) is None
    assert rprs._wall_area_m2({"room": {"floor_polygon": "bad", "walls": []}}) is None

    report = rprs.build_cost_report(
        out_dir=tmp_path,
        bindings_data={
            "bindings": [
                "bad",
                {"selection_status": "rejected", "chosen_candidate": {"title": "skip"}},
                {"selection_status": "selected", "target_id": "missing_candidate"},
            ]
        },
        supplier_scene_data={"placements": ["bad", {"id": "generated"}]},
        surface_materials_info=None,
    )
    assert report["item_count"] == 0
    assert report["asset_source_counts"]["generated"] == 1


def test_stage_timings_surface_helpers_and_cost_report(tmp_path, monkeypatch, capsys):
    times = iter([1.0, 2.25, 3.0, 5.0])
    monkeypatch.setattr(rprs.time, "perf_counter", lambda: next(times, 30.0))
    timings = rprs.StageTimings(enabled=True)
    with timings.stage("ok_stage"):
        pass
    timings.finish()
    report = timings.report()
    assert report["stages"][0]["stage"] == "ok_stage"
    assert report["stages"][0]["status"] == "ok"
    assert report["total_seconds"] == 4.0
    assert "[TIMER]" in capsys.readouterr().err

    assert rprs._surface_style_from_design_spec({"style": {"primary": "soft classic"}}) == "classic"
    prompt = rprs._surface_prompt("", {"color_palette": {"preferred_colors": ["oak"]}, "materials": {"preferred": ["wood"]}})
    assert "Preferred surface colors" in prompt
    settings = rprs._llm_settings_from_args(argparse.Namespace(flooring_llm_provider="none"), "flooring")
    assert settings["provider"] == "none"
    assert settings["ollama_model"] == "gpt-oss:20b"

    assert rprs._num("1 234,50 RUB") == 1234.5
    assert rprs._money("99.999") == 100.0
    assert rprs._format_price(1234.5, "RUB") == "1 234.50 RUB"
    assert rprs._markdown_cell("a|b\nc") == "a\\|b c"
    assert rprs._proxy_like_asset_path("/tmp/built/proxy.glb")
    assert rprs._coverage_from_name("Laminate 2,13 м2") == 2.13
    assert rprs._material_coverage_m2({"width_cm": 106, "length_m": 10}, kind="wall_material") == 10.6

    floor_sel = tmp_path / "floor.selection.json"
    wall_sel = tmp_path / "wall.selection.json"
    floor_sel.write_text(json.dumps({"selected_material": {"sku": "f1", "name": "Floor 2 м2", "price": 100, "price_currency": "RUB"}}), encoding="utf-8")
    wall_sel.write_text(json.dumps({"selected_material": {"sku": "w1", "name": "Wall", "price": 200, "price_currency": "RUB", "width_cm": 100, "length_m": 10}}), encoding="utf-8")
    scene = (
        placement_scene_with_room(ceiling_height_m=3.0)
        .placement(
            "target1",
            source={"asset_source": "trellis_generated_local_asset"},
            asset={"mesh_path": "/models/chair.glb"},
        )
        .build()
    )
    bindings = {
        "bindings": [
            {
                "target_id": "target1",
                "category": "chair",
                "semantic_group": "seat",
                "selection_status": "heuristic_top1_selected",
                "chosen_candidate": {
                    "title": "Chair",
                    "unique_key": "u1",
                    "price_value": "1500",
                    "price_currency": "RUB",
                    "product_url": "https://product",
                    "width_cm": 50,
                    "depth_cm": 60,
                    "height_cm": 90,
                },
            }
        ]
    }

    result = rprs.build_cost_report(
        out_dir=tmp_path,
        bindings_data=bindings,
        supplier_scene_data=scene,
        surface_materials_info={
            "flooring": {"selection_json": str(floor_sel)},
            "wall_material": {"selection_json": str(wall_sel)},
        },
    )

    assert result["item_count"] == 1
    assert result["totals_by_currency"]["RUB"] > 1500
    assert Path(result["json"]).is_file()
    assert Path(result["markdown"]).read_text(encoding="utf-8").startswith("# Supplier Cost Report")


def test_surface_cost_geometry_and_curtain_skip_edge_matrix(tmp_path, monkeypatch):
    assert rprs._num(float("nan")) is None
    assert rprs._num("no digits") is None
    assert rprs._money("0") is None
    assert rprs._format_price(None) == ""
    assert rprs._material_from_selection(None) == {}
    assert rprs._material_from_selection({"selection_json": ""}) == {}
    monkeypatch.setattr(rprs, "read_json", lambda _path: (_ for _ in ()).throw(RuntimeError("bad json")))
    assert rprs._material_from_selection({"selection_json": str(tmp_path / "missing.json")}) == {}

    room = {
        "room": {
            "width_m": "4",
            "depth_m": "3",
            "height_m": "2.8",
            "walls": [
                {"length_m": "4"},
                {"from": {"x": 4, "y": 0}, "to": {"x": 4, "y": 3}},
                {"from": [4, 3], "to": [0, 3]},
                {"from": {"x": 0, "z": 3}, "to": {"x": 0, "z": 0}},
                "bad",
            ],
            "doors": [{"width": 0.8, "height": 2.05}],
            "windows": [{"w": 1.2, "h": 1.1}],
        }
    }
    assert rprs._room_polygon_points(room["room"]) == [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)]
    assert rprs._floor_area_m2(room) == 12.0
    assert rprs._wall_length_m(room["room"], room["room"]["walls"][1]) == pytest.approx(3.0)
    assert rprs._wall_area_m2(room) == pytest.approx(round((4 + 3 + 4) * 2.8 - 0.8 * 2.05 - 1.2 * 1.1, 3))
    assert rprs._wall_area_m2({"room": {"walls": [{"bad": True}]}}) is None
    assert rprs._polygon_area_m2([(0, 0), (1, 0)]) is None
    assert rprs._coverage_from_name("pack 60x120 м") == pytest.approx(7200.0)
    assert rprs._material_coverage_m2({"raw_properties": {"Площадь рулона": "5,5"}}, kind="flooring") == 5.5
    assert rprs._material_coverage_m2({"name": "Обои 10 м2"}, kind="wall_material") == 10.0
    assert rprs._surface_cost_line(kind="flooring", label="Floor", area_m2=12, material={"name": "No price"})["price_status"] == "missing_price_or_coverage"
    assert rprs._candidate_dimensions_m_for_report({"width_cm": "50", "depth_cm": None, "height_cm": "bad"}) == {
        "width": 0.5,
        "depth": None,
        "height": None,
    }
    md = rprs._cost_report_markdown(
        {
            "totals_by_currency": {},
            "items": [{"target_id": "t", "title": "A|B", "product_url": "", "unit_price": None}],
            "surface_materials": [{"label": "Wall", "name": "Paint", "area_m2": None, "unit_price": None}],
        }
    )
    assert "A\\|B" in md
    monkeypatch.setattr(rprs, "read_json", lambda path: json.loads(Path(path).read_text(encoding="utf-8")))

    scene_no_windows = {"room": {"floor_polygon": [[0, 0], [1, 0], [1, 1]]}, "items": []}
    assert rprs._curtains_needed_for_scene(scene=scene_no_windows, prompt_text="", style_profile={}, policy="always") == (
        False,
        "missing_windows",
    )
    scene_with_window = {"room": {"windows": [{"id": "w"}]}, "items": [{"id": "real_curtain", "category": "curtain", "asset": {"mesh_path": "curtain.fbx"}}]}
    assert rprs._scene_has_curtain_items(scene_with_window)
    assert rprs._curtains_needed_for_scene(scene=scene_with_window, prompt_text="", style_profile={}, policy="auto") == (
        False,
        "existing_curtains",
    )
    scene_needs = {"room": {"windows": [{"id": "w"}]}, "items": []}
    assert rprs._curtains_needed_for_scene(scene=scene_needs, prompt_text="", style_profile={"needs_curtains": True}, policy="auto") == (
        True,
        "profile_needs_curtains",
    )
    assert rprs._curtains_needed_for_scene(scene=scene_needs, prompt_text="", style_profile={"room_type": "office"}, policy="auto") == (
        False,
        "auto_not_requested",
    )
    assert rprs._curtains_needed_for_scene(scene=scene_needs, prompt_text="", style_profile={"room_type": "Bedroom"}, policy="auto") == (
        True,
        "default_for_room_type:bedroom",
    )

    scene_path = tmp_path / "curtain_scene.json"
    scene_path.write_text(json.dumps(scene_needs), encoding="utf-8")
    args = argparse.Namespace(
        curtains="always",
        no_curtains=False,
        curtain_materials=str(tmp_path / "missing_materials"),
        curtain_models_dir=str(tmp_path / "models"),
        curtain_supplier_catalog=str(tmp_path / "catalog.json"),
        curtain_seed=0,
        seed=0,
    )
    out_scene, info = rprs.maybe_apply_curtains_to_supplier_scene(args, out_dir=tmp_path, supplier_scene=scene_path, prompt="", room_design_spec={})
    assert out_scene == scene_path
    assert info["skipped_reason"] == "missing_curtain_materials"

    for raw_policy, expected in [("off", "never"), ("yes", "always"), ("bogus", "auto")]:
        args.curtains = raw_policy
        out_scene, info = rprs.maybe_apply_curtains_to_supplier_scene(args, out_dir=tmp_path, supplier_scene=scene_path, prompt="", room_design_spec={})
        assert info["policy"] == expected
        assert out_scene == scene_path
    args.curtains = "always"

    materials = tmp_path / "materials"
    materials.mkdir()
    monkeypatch.setattr(rprs, "load_curtain_catalog", lambda _path: ([], materials))
    args.curtain_materials = str(materials)
    out_scene, info = rprs.maybe_apply_curtains_to_supplier_scene(args, out_dir=tmp_path, supplier_scene=scene_path, prompt="", room_design_spec={})
    assert out_scene == scene_path
    assert info["skipped_reason"] == "empty_curtain_catalog"

    monkeypatch.setattr(rprs, "load_curtain_catalog", lambda _path: ([{"sku": "c"}], materials))
    monkeypatch.setattr(rprs, "discover_curtain_models", lambda _path: [])
    monkeypatch.setattr(rprs, "discover_supplier_curtain_models", lambda **_kwargs: [])
    monkeypatch.setattr(rprs, "apply_curtains_to_scene", lambda scene, **_kwargs: (scene, {"added_count": 0, "reason": "no_model"}))
    out_scene, info = rprs.maybe_apply_curtains_to_supplier_scene(args, out_dir=tmp_path, supplier_scene=scene_path, prompt="", room_design_spec={})
    assert out_scene == scene_path
    assert info["stripped_placeholder_curtain_count"] == 0
    assert info["needed_reason"] == "policy_always"


def test_surface_material_stage_numeric_and_skip_defensive_edges(tmp_path, monkeypatch):
    times = iter([10.0, 11.0, 12.5, 20.0])
    monkeypatch.setattr(rprs.time, "perf_counter", lambda: next(times, 30.0))
    timings = rprs.StageTimings(enabled=True)
    with pytest.raises(ValueError):
        with timings.stage("boom"):
            raise ValueError("bad stage")
    assert timings.records[0]["status"] == "failed"
    assert timings.report()["total_seconds"] == 10.0

    disabled = rprs.StageTimings(enabled=False)
    with disabled.stage("ignored"):
        pass
    disabled.finish()
    assert disabled.report() is None

    assert rprs._room_type_from_scene(tmp_path / "missing.json") is None
    invalid_room = tmp_path / "invalid_room.json"
    invalid_room.write_text(json.dumps({"room": "bad"}), encoding="utf-8")
    assert rprs._room_type_from_scene(invalid_room) is None
    valid_room = tmp_path / "valid_room.json"
    valid_room.write_text(json.dumps({"room": {"type_hint": "Bedroom"}}), encoding="utf-8")
    assert rprs._room_type_from_scene(valid_room) == "bedroom"

    fallback_prompt = rprs._surface_prompt("", {})
    assert "Fallback preference" in fallback_prompt
    assert rprs._surface_style_from_design_spec({"style": {"primary": "loft-industrial"}}) == "loft"
    assert rprs._num("") is None
    assert rprs._num("1,234.56 RUB") == pytest.approx(1234.56)

    original_search = rprs.re.search

    class BadMatch:
        def group(self, _index):
            return "1.2.3"

    monkeypatch.setattr(rprs.re, "search", lambda *_args, **_kwargs: BadMatch())
    assert rprs._num("123") is None
    monkeypatch.setattr(rprs.re, "search", original_search)

    assert rprs._floor_area_m2({"room": {"area_m2": "15,5"}}) == 15.5
    assert rprs._wall_area_m2({"room": {"floor_polygon": [[0, 0], [2, 0], [2, 2], [0, 2]], "height_m": 3}}) == 24.0
    assert rprs._material_coverage_m2(None, kind="flooring") is None
    assert rprs._material_coverage_m2({"package_area_m2": "2,4"}, kind="flooring") == 2.4
    assert rprs._material_coverage_m2({"roll_width_cm": "1.06", "roll_length_m": 10}, kind="wall_material") == 10.6
    priced = rprs._surface_cost_line(
        kind="flooring",
        label="Floor",
        area_m2=10.0,
        material={"sku": "f", "name": "Floor", "price": 100, "price_currency": "RUB", "package_area_m2": 3},
    )
    assert priced["units_needed"] == 4
    assert priced["estimated_total_price"] == 400.0

    scene_path = tmp_path / "scene.json"
    placement_path = tmp_path / "placement.json"
    scene_payload = {"room": {"room_type": "bedroom", "floor_polygon": [[0, 0], [1, 0], [1, 1]]}, "items": []}
    scene_path.write_text(json.dumps(scene_payload), encoding="utf-8")
    placement_path.write_text(json.dumps(scene_payload), encoding="utf-8")
    args = argparse.Namespace(
        no_flooring=False,
        flooring_materials="missing_floor_materials_for_test",
        flooring_style_rules="missing_floor_rules_for_test.json",
        flooring_top_k=5,
        flooring_llm_provider="none",
        flooring_ollama_url="http://127.0.0.1:9",
        flooring_ollama_model="none",
        flooring_ollama_timeout=1,
        flooring_ollama_temperature=0.0,
        flooring_ollama_num_ctx=1024,
        flooring_llm_top_n=2,
        no_wall_material=False,
        wall_materials="missing_wall_materials_for_test",
        wall_top_k=5,
        wall_llm_provider="none",
        wall_ollama_url="http://127.0.0.1:9",
        wall_ollama_model="none",
        wall_ollama_timeout=1,
        wall_ollama_temperature=0.0,
        wall_ollama_num_ctx=1024,
        wall_llm_top_n=2,
    )
    out_scene, out_placement, info = rprs.maybe_apply_surface_materials(
        args,
        out_dir=tmp_path,
        supplier_scene=scene_path,
        supplier_placement=placement_path,
        prompt="",
        room_design_spec={},
        timings=rprs.StageTimings(enabled=False),
    )
    assert out_scene == scene_path
    assert out_placement == placement_path
    assert info["flooring"]["reason"] == "materials_or_style_rules_missing"
    assert info["wall_material"]["reason"] == "materials_missing"


def test_enrich_missing_assets_with_trellis_success_reuse_skips_and_oom_disable(tmp_path, monkeypatch):
    glb = tmp_path / "generated.glb"
    glb.write_bytes(b"glb")
    existing = tmp_path / "existing.glb"
    existing.write_bytes(b"existing")
    bindings_path = tmp_path / "bindings.json"
    output_path = tmp_path / "bindings.with_trellis.json"
    bindings_payload = {
        "bindings": [
            {
                "target_id": "chair1",
                "category": "chair",
                "selection_status": "heuristic_top1_selected",
                "chosen_candidate": {"unique_key": "u1", "title": "Chair", "images": ["https://ikea.com/chair.jpg"]},
            },
            {
                "target_id": "chair2",
                "category": "chair",
                "selection_status": "heuristic_top1_selected",
                "chosen_candidate": {"unique_key": "u1", "title": "Chair duplicate", "images": ["https://ikea.com/chair2.jpg"]},
            },
            {
                "target_id": "pillow1",
                "category": "pillow",
                "selection_status": "heuristic_top1_selected",
                "chosen_candidate": {"unique_key": "pillow-key", "title": "Pillow"},
            },
            {
                "target_id": "local1",
                "category": "table",
                "selection_status": "heuristic_top1_selected",
                "chosen_candidate": {"unique_key": "local-key", "asset_local_path": str(existing)},
            },
            {
                "target_id": "",
                "category": "lamp",
                "selection_status": "heuristic_top1_selected",
                "chosen_candidate": {},
            },
            {
                "target_id": "bad1",
                "category": "wardrobe",
                "selection_status": "heuristic_top1_selected",
                "chosen_candidate": {"unique_key": "u_fail", "title": "Bad"},
            },
            {
                "target_id": "after_oom",
                "category": "desk",
                "selection_status": "heuristic_top1_selected",
                "chosen_candidate": {"unique_key": "u_after", "title": "After"},
            },
            {
                "target_id": "ignored",
                "category": "chair",
                "selection_status": "unmatched",
                "chosen_candidate": {"unique_key": "ignored"},
            },
        ]
    }
    bindings_path.write_text(json.dumps(bindings_payload), encoding="utf-8")
    args = _trellis_args(
        trellis_max_assets=0,
        trellis_skip_categories="pillow",
        trellis_disable_after_oom=True,
        trellis_vlm_provider="ollama",
        trellis_vlm_unload_after_filter=True,
    )
    monkeypatch.setattr(rprs, "unload_ollama_model", lambda **_kwargs: {"ok": True})

    def fake_run_trellis(args, *, card_json, job_id, out_dir, prepared_job_dir=None, attempts=None):
        del args, job_id, out_dir, prepared_job_dir
        card = json.loads(Path(card_json).read_text(encoding="utf-8"))
        if card.get("unique_key") == "u_fail":
            if attempts is not None:
                attempts.append({"oom": True, "status": "failed"})
            raise subprocess.CalledProcessError(1, ["trellis"], output="CUDA out of memory")
        patched = {
            **card,
            "asset_status": rprs.TRELLIS_ASSET_STATUS,
            "asset_local_path": str(glb),
            "asset_format": "glb",
            "extra": {"trellis_generated_asset": {"asset_local_path": str(glb), "target_size_m": [1, 1, 1]}},
        }
        patched_path = tmp_path / f"{card['unique_key']}.patched.json"
        patched_path.write_text(json.dumps(patched), encoding="utf-8")
        if attempts is not None:
            attempts.append({"oom": False, "status": "success"})
        return {"card_with_trellis_asset_json": str(patched_path), "local_job_dir": str(tmp_path)}

    monkeypatch.setattr(rprs, "run_trellis_with_oom_retries", fake_run_trellis)

    out_path, report = rprs.enrich_missing_assets_with_trellis(
        bindings_json_path=bindings_path,
        output_json_path=output_path,
        out_dir=tmp_path,
        args=args,
    )

    assert out_path == output_path
    assert report["prepared_count"] == 3
    assert report["generated_count"] == 2
    assert report["failed_count"] == 1
    assert report["skipped_count"] == 4
    statuses = [item["status"] for item in report["items"]]
    assert "generated" in statuses
    assert "reused_generated_asset" in statuses
    assert "skipped_trellis_category" in statuses
    assert "skipped_existing_local_asset" in statuses
    assert "skipped_trellis_disabled_after_oom" in statuses
    patched_data = json.loads(output_path.read_text(encoding="utf-8"))
    assert patched_data["meta"]["trellis_missing_asset_generation"]["disabled_after_oom"] is True
    assert patched_data["bindings"][0]["chosen_candidate"]["asset_local_path"] == str(glb)
    assert patched_data["bindings"][1]["chosen_candidate"]["asset_local_path"] == str(glb)


def test_maybe_apply_curtains_to_supplier_scene_full_path(tmp_path, monkeypatch):
    scene_path = tmp_path / "scene.json"
    scene = {
        "room": {
            "room_type": "bedroom",
            "windows": [{"id": "win1"}],
            "floor_polygon": [[0, 0], [3, 0], [3, 3], [0, 3]],
        },
        "items": [
            {
                "id": "placeholder",
                "category": "curtain",
                "asset": {"kind": "procedural_placeholder"},
                "meta": {"procedural": True},
            }
        ],
        "placements": [],
    }
    scene_path.write_text(json.dumps(scene), encoding="utf-8")
    materials_dir = tmp_path / "curtain_materials"
    models_dir = tmp_path / "curtain_models"
    catalog_path = tmp_path / "supplier_catalog.json"
    materials_dir.mkdir()
    models_dir.mkdir()
    catalog_path.write_text("[]", encoding="utf-8")
    model = models_dir / "curtain.fbx"
    model.write_text("fbx", encoding="utf-8")
    monkeypatch.setattr(rprs, "write_curtain_json", rprs.write_json)
    monkeypatch.setattr(rprs, "load_curtain_catalog", lambda _path: ([{"sku": "c1", "name": "Curtain"}], materials_dir))
    monkeypatch.setattr(rprs, "discover_curtain_models", lambda _path: [model])
    monkeypatch.setattr(rprs, "discover_supplier_curtain_models", lambda **_kwargs: [{"mesh_path": str(model)}])

    def fake_apply(scene_data, **kwargs):
        assert kwargs["catalog_base_dir"] == materials_dir
        updated = {**scene_data, "items": [{"id": "curtain_win1", "category": "curtain"}], "placements": []}
        return updated, {"added_count": 1, "selected": [{"sku": "c1", "name": "Curtain", "texture_path": "tex.jpg"}]}

    monkeypatch.setattr(rprs, "apply_curtains_to_scene", fake_apply)
    args = argparse.Namespace(
        curtains="always",
        no_curtains=False,
        curtain_materials=str(materials_dir),
        curtain_models_dir=str(models_dir),
        curtain_supplier_catalog=str(catalog_path),
        curtain_seed=7,
        seed=1,
    )

    out_scene, info = rprs.maybe_apply_curtains_to_supplier_scene(
        args,
        out_dir=tmp_path,
        supplier_scene=scene_path,
        prompt="",
        room_design_spec={},
    )

    assert out_scene.name.endswith(".curtains.v1.json")
    assert info["added_count"] == 1
    assert info["stripped_placeholder_curtain_count"] == 1
    assert json.loads(out_scene.read_text(encoding="utf-8"))["placements"][0]["id"] == "curtain_win1"

    args.curtains = "never"
    out_scene, info = rprs.maybe_apply_curtains_to_supplier_scene(
        args,
        out_dir=tmp_path,
        supplier_scene=scene_path,
        prompt="",
        room_design_spec={},
    )
    assert info["skipped_reason"] == "policy_never"
    assert out_scene.name.endswith(".no_curtains.v1.json")


def test_maybe_apply_surface_materials_with_mocked_selectors(tmp_path, monkeypatch):
    scene_path = tmp_path / "scene.json"
    placement_path = tmp_path / "placement.json"
    scene = {"room": {"room_type": "bedroom", "floor_polygon": [[0, 0], [2, 0], [2, 2], [0, 2]]}, "placements": []}
    scene_path.write_text(json.dumps(scene), encoding="utf-8")
    placement_path.write_text(json.dumps(scene), encoding="utf-8")
    materials_dir = tmp_path / "materials"
    materials_dir.mkdir()
    style_rules = tmp_path / "rules.json"
    style_rules.write_text("{}", encoding="utf-8")
    floor_calls = []
    wall_calls = []

    def fake_flooring_selection(**kwargs):
        floor_calls.append(kwargs)
        kwargs["out_path"].write_text(json.dumps({"selected_material": {"sku": "f1"}}), encoding="utf-8")
        return {
            "selected_material": {"sku": "f1", "name": "Oak", "price": 10, "price_currency": "RUB", "package_area_m2": 2.0},
            "texture_candidate": {"texture_abs_path": "/tmp/floor.jpg"},
            "llm_rerank": {"status": "skipped"},
        }

    def fake_wall_selection(**kwargs):
        wall_calls.append(kwargs)
        kwargs["out_path"].write_text(json.dumps({"selected_material": {"sku": "w1"}}), encoding="utf-8")
        return {
            "selected_material": {
                "sku": "w1",
                "name": "Wallpaper",
                "price": 20,
                "price_currency": "RUB",
                "width_cm": 100,
                "length_m": 10,
                "average_hex": "#ffffff",
                "dominant_colors_hex": ["#ffffff"],
            },
            "llm_rerank": {"status": "skipped"},
        }

    monkeypatch.setattr(rprs, "run_flooring_selection", fake_flooring_selection)
    monkeypatch.setattr(rprs, "apply_flooring_to_scene", lambda data, selection: {**data, "flooring_sku": selection["selected_material"]["sku"]})
    monkeypatch.setattr(rprs, "run_wall_selection", fake_wall_selection)
    monkeypatch.setattr(
        rprs,
        "apply_wall_material_to_scene_with_catalog",
        lambda data, selection, materials_path: {**data, "wall_sku": selection["selected_material"]["sku"], "materials_path": str(materials_path)},
    )
    args = argparse.Namespace(
        no_flooring=False,
        flooring_materials=str(materials_dir),
        flooring_style_rules=str(style_rules),
        flooring_top_k=3,
        flooring_llm_provider="none",
        no_wall_material=False,
        wall_materials=str(materials_dir),
        wall_top_k=4,
        wall_llm_provider="none",
    )

    out_scene, out_placement, info = rprs.maybe_apply_surface_materials(
        args,
        out_dir=tmp_path,
        supplier_scene=scene_path,
        supplier_placement=placement_path,
        prompt="warm oak bedroom",
        room_design_spec={"style": {"primary": "modern"}},
        timings=rprs.StageTimings(enabled=False),
    )

    assert floor_calls and floor_calls[0]["top_k"] == 3
    assert wall_calls and wall_calls[0]["top_k"] == 4
    assert out_scene.name.endswith(".flooring.v1.wall_material.v1.json")
    assert out_placement.name.endswith(".flooring.v1.wall_material.v1.json")
    assert info["flooring"]["selected_sku"] == "f1"
    assert info["wall_material"]["selected_sku"] == "w1"
    final_scene = json.loads(out_scene.read_text(encoding="utf-8"))
    assert final_scene["flooring_sku"] == "f1"
    assert final_scene["wall_sku"] == "w1"


def test_build_cli_and_main_full_flow_with_mocked_expensive_stages(tmp_path, monkeypatch, capsys):
    parser = rprs.build_cli()
    parsed = parser.parse_args(
        [
            "--room",
            "room.json",
            "--out-dir",
            "out",
            "--prompt",
            "bedroom",
            "--density",
            "very_high",
            "--policy",
            "always",
            "--supplier-selection-mode",
            "best_match",
            "--top-k",
            "7",
            "--build-blend",
            "--trellis-generate-missing-assets",
            "--trellis-server-host",
            "gpu.example",
            "--trellis-remote-cuda-visible-devices",
            "0,1",
            "--topview-vlm-orientation-repair",
        ]
    )
    assert parsed.density == "very_high"
    assert parsed.supplier_selection_mode == "best_match"
    assert parsed.trellis_generate_missing_assets is True
    assert parsed.topview_vlm_orientation_repair is True

    room_path = tmp_path / "room.json"
    room_path.write_text(
        json.dumps({"room": {"id": "room_1", "type": "bedroom", "floor_polygon": [[0, 0], [4, 0], [4, 3], [0, 3]]}}),
        encoding="utf-8",
    )
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps([{"id": "chair"}]), encoding="utf-8")
    out_dir = tmp_path / "out"

    def write(path, payload):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload), encoding="utf-8")
        return Path(path)

    procedural_scene = out_dir / "procedural.scene.json"
    procedural_placement = out_dir / "procedural.placement.json"
    targets = out_dir / "targets.json"
    bindings_with_trellis = out_dir / "bindings.with_trellis.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_procedural_room_supplier.py",
            "--room",
            str(room_path),
            "--out-dir",
            str(out_dir),
            "--prompt",
            "modern bedroom",
            "--policy",
            "always",
            "--density",
            "high",
            "--seed",
            "5",
            "--supplier-catalog-json",
            str(catalog),
            "--supplier-selection-mode",
            "optimal",
            "--top-k",
            "5",
            "--trellis-generate-missing-assets",
            "--trellis-server-host",
            "gpu.example",
            "--build-blend",
            "--no-stage-timings",
        ],
    )

    monkeypatch.setattr(
        rprs,
        "apply_procedural_room_stage",
        lambda **_kwargs: {
            "schema": "procedural_room_report/v1",
            "skipped": False,
            "output_scene_json": str(write(procedural_scene, {"room": {"floor_polygon": [[0, 0], [4, 0], [4, 3], [0, 3]]}, "placements": [{"id": "chair", "meta": {"supplier_binding_applied": True}, "source": {"asset_source": "trellis_generated_local_asset"}}]})),
            "output_placement_json": str(write(procedural_placement, {"placements": [{"id": "chair", "meta": {"supplier_binding_applied": True}, "source": {"asset_source": "trellis_generated_local_asset"}}]})),
            "validation": {"collisions": [], "accessibility_ok": True},
        },
    )
    monkeypatch.setattr(
        rprs,
        "create_layout_selection_stub_artifacts",
        lambda **_kwargs: {"layout_targets_json": str(write(targets, {"source_json": str(procedural_scene), "targets": [{"id": "chair"}]}))},
    )
    monkeypatch.setattr(rprs, "build_room_design_spec", lambda **_kwargs: {"style": {"primary": "modern"}})
    monkeypatch.setattr(rprs, "load_supplier_catalog", lambda *args, **kwargs: [])
    monkeypatch.setattr(rprs, "load_supplier_catalog_json", lambda *args, **kwargs: [{"id": "chair", "title": "Chair"}])
    monkeypatch.setattr(rprs, "_merge_catalog_rows", lambda rows: list(rows))
    monkeypatch.setattr(
        rprs,
        "build_bindings_with_candidates",
        lambda **_kwargs: {
            "bindings": [{"target_id": "chair", "chosen_candidate": {"unique_key": "chair"}}],
            "meta": {"matched_target_count": 1},
        },
    )
    monkeypatch.setattr(
        rprs,
        "enrich_missing_assets_with_trellis",
        lambda **_kwargs: (write(bindings_with_trellis, {"bindings": [{"target_id": "chair"}]}), {"generated_count": 1}),
    )

    def fake_apply_supplier(input_json_path, bindings_json_path, output_json_path, **_kwargs):
        del bindings_json_path
        data = json.loads(Path(input_json_path).read_text(encoding="utf-8"))
        data.setdefault("placements", [{"id": "chair"}])
        data["placements"][0].setdefault("meta", {})["supplier_binding_applied"] = True
        data["placements"][0].setdefault("source", {})["asset_source"] = "trellis_generated_local_asset"
        write(output_json_path, data)

    monkeypatch.setattr(rprs, "apply_supplier_bindings_to_json", fake_apply_supplier)
    monkeypatch.setattr(rprs, "maybe_apply_surface_materials", lambda args, out_dir, supplier_scene, supplier_placement, prompt, room_design_spec, timings: (supplier_scene, supplier_placement, {"flooring": None}))
    monkeypatch.setattr(rprs, "maybe_apply_curtains_to_supplier_scene", lambda *args, **kwargs: (kwargs["supplier_scene"] if "supplier_scene" in kwargs else args[2], {"added_count": 0}))
    monkeypatch.setattr(rprs, "maybe_build_blend_scene", lambda *args, **kwargs: {"blend": str(out_dir / "scene.blend"), "png": str(out_dir / "scene.png")})
    monkeypatch.setattr(rprs, "maybe_apply_topview_vlm_orientation_repair", lambda args, supplier_scene, out_dir, blend_artifacts: (supplier_scene, {"skipped_reason": "mock"}, blend_artifacts))
    monkeypatch.setattr(rprs, "build_room_context", lambda scene, prompt: {"room": scene.get("room", {})})
    monkeypatch.setattr(rprs, "validate_placements", lambda context, placements: {"collisions": [], "accessibility_ok": True})
    monkeypatch.setattr(rprs, "build_cost_report", lambda **_kwargs: {"json": str(out_dir / "cost.json"), "totals_by_currency": {"RUB": 1.0}})

    rprs.main()
    report_path = out_dir / "procedural_room_supplier_report.json"
    assert report_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["summary"]["matched_target_count"] == 1
    assert report["summary"]["trellis_asset_replaced"] == 1
    assert report["trellis_missing_asset_generation"] == {"generated_count": 1}
    assert "procedural_room_supplier_report/v1" in capsys.readouterr().out


def test_enrich_missing_assets_failure_dependents_and_filter_edges(tmp_path, monkeypatch, capsys):
    bindings_path = tmp_path / "bad_bindings.json"
    bindings_path.write_text(json.dumps({"bindings": {"bad": True}}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="нет bindings"):
        rprs.enrich_missing_assets_with_trellis(
            bindings_json_path=bindings_path,
            output_json_path=tmp_path / "out.json",
            out_dir=tmp_path,
            args=_trellis_args(trellis_max_assets=1),
        )

    payload = {
        "bindings": [
            {
                "target_id": "no_images",
                "category": "chair",
                "selection_status": "heuristic_top1_selected",
                "chosen_candidate": {"unique_key": "no_images", "images": ["https://other.test/a.jpg"]},
            },
            {
                "target_id": "fail_a",
                "category": "chair",
                "selection_status": "heuristic_top1_selected",
                "chosen_candidate": {"unique_key": "fail-key", "title": "Fail", "images": ["https://ikea.com/a.jpg"]},
            },
            {
                "target_id": "fail_b",
                "category": "chair",
                "selection_status": "heuristic_top1_selected",
                "chosen_candidate": {"unique_key": "fail-key", "title": "Fail duplicate", "images": ["https://ikea.com/b.jpg"]},
            },
            {
                "target_id": "over_limit",
                "category": "desk",
                "selection_status": "heuristic_top1_selected",
                "chosen_candidate": {"unique_key": "over-limit", "title": "Desk", "images": ["https://ikea.com/c.jpg"]},
            },
        ],
        "meta": "bad",
    }
    bindings_path.write_text(json.dumps(payload), encoding="utf-8")
    output_path = tmp_path / "with_trellis.json"
    args = _trellis_args(
        trellis_max_assets=1,
        trellis_ikea_mebelru_images_only=True,
        trellis_vlm_provider="ollama",
        trellis_vlm_unload_after_filter=True,
    )
    monkeypatch.setattr(rprs, "unload_ollama_model", lambda **_kwargs: {"ok": False, "error": "offline"})

    def fail_trellis(_args, *, attempts, **_kwargs):
        attempts.append({"oom": False, "status": "failed"})
        raise RuntimeError("unit trellis failure")

    monkeypatch.setattr(rprs, "run_trellis_with_oom_retries", fail_trellis)
    _out, report = rprs.enrich_missing_assets_with_trellis(
        bindings_json_path=bindings_path,
        output_json_path=output_path,
        out_dir=tmp_path,
        args=args,
    )

    statuses = [item["status"] for item in report["items"]]
    assert "skipped_no_ikea_mebelru_images" in statuses
    assert "failed" in statuses
    assert "skipped_same_unique_key_failed" in statuses
    assert "skipped_trellis_max_assets" in statuses
    assert report["vlm_unload_after_prepare_all"]["ok"] is False
    assert "[WARN] failed to unload Ollama model after prepare phase" in capsys.readouterr().err
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["meta"]["trellis_missing_asset_generation"]["failed_count"] == 1


def test_room_door_normalization_edge_branches():
    class NonDictRoom:
        def get(self, _key, _default=None):
            return None

    assert rprs.ensure_room_has_door(NonDictRoom())[1]["reason"] == "missing_room"
    assert rprs.ensure_room_has_door({"room": {"floor_polygon": [[0, 0], [1, 0]]}})[1]["reason"] == "missing_walls_and_floor_polygon"
    assert rprs.ensure_room_has_door({"room": {"floor_polygon": [[0, 0], [0.5, 0], [0.5, 0.5]], "walls": [{"bad": True}]}})[1]["reason"] == "no_wall_long_enough"

    normalized, info = rprs.ensure_room_has_door(
        {
            "room": {
                "floor_polygon": [[0, 0], [4, 0], [4, 3], [0, 3]],
                "openings": [
                    {"id": "door1", "type": "door", "wall_id": "w0", "s": 0.5},
                    {"id": "win1", "type": "window", "wall_id": "w1", "s": 0.5},
                ],
            }
        }
    )
    assert info["reason"] == "normalized_existing_openings"
    assert normalized["room"]["doors"][0]["id"] == "door1"
    assert normalized["room"]["windows"][0]["id"] == "win1"

    with_openings, info = rprs.ensure_room_has_door(
        {
            "room": {
                "floor_polygon": [[0, 0], [4, 0], [4, 3], [0, 3]],
                "openings": {"windows": [{"id": "w0", "wall_id": "w1", "s": 1.0}]},
            }
        }
    )
    assert info["changed"] is True
    assert with_openings["room"]["openings"]["doors"][0]["id"] == "door_main_0001"
    assert rprs._room_wall_points({"floor_polygon": [[0, 0], [1, 0]]}, {"from_vertex": "bad", "to_vertex": 1}) is None
    assert rprs._room_wall_points({}, {"start": {"x": 0, "y": 0}, "end": [1, 0]}) == ((0.0, 0.0), (1.0, 0.0))


def test_topview_vlm_orientation_repair_skip_and_success_paths(tmp_path, monkeypatch, capsys):
    scene_path = tmp_path / "scene.json"
    scene_path.write_text(json.dumps({"items": [{"id": "chair1", "category": "chair"}]}), encoding="utf-8")
    blend_path = tmp_path / "scene.blend"
    blend_path.write_bytes(b"blend")
    png_path = tmp_path / "scene.png"
    png_path.write_bytes(b"png")
    build_report = tmp_path / "scene.build_report.json"
    build_report.write_text("{}", encoding="utf-8")

    args = argparse.Namespace(
        topview_vlm_orientation_repair=True,
        build_blend=True,
        blender="Blender",
        topview_vlm_target_scope="chairs",
        topview_vlm_include_armchairs=True,
        topview_vlm_max_objects=10,
        topview_vlm_elevation_deg=82.0,
        topview_vlm_radius_mult=0.6,
        topview_vlm_lens=35.0,
        topview_vlm_resolution_x=800,
        topview_vlm_resolution_y=600,
        topview_vlm_provider="ollama",
        topview_vlm_model="vision",
        topview_vlm_min_confidence=0.7,
        topview_vlm_max_delta_deg=90.0,
        topview_vlm_snap_step_deg=45.0,
        topview_vlm_max_repairs_per_object=2,
        topview_vlm_visual_front_offset_deg=5.0,
    )

    out_scene, info, artifacts = rprs.maybe_apply_topview_vlm_orientation_repair(
        argparse.Namespace(topview_vlm_orientation_repair=False),
        supplier_scene=scene_path,
        out_dir=tmp_path,
        blend_artifacts={"blend": str(blend_path)},
    )
    assert out_scene == scene_path
    assert info is None
    assert artifacts == {"blend": str(blend_path)}

    out_scene, info, artifacts = rprs.maybe_apply_topview_vlm_orientation_repair(
        argparse.Namespace(topview_vlm_orientation_repair=True, build_blend=False),
        supplier_scene=scene_path,
        out_dir=tmp_path,
        blend_artifacts=None,
    )
    assert info == {"skipped_reason": "requires_build_blend"}

    Ref = type("Ref", (), {})
    ref = Ref()
    ref.object_id = "chair1"
    monkeypatch.setattr(rprs, "collect_topview_vlm_scene_objects", lambda scene, max_objects: [ref])
    monkeypatch.setattr(rprs, "filter_topview_vlm_target_objects", lambda refs, scope, include_armchairs: refs)
    monkeypatch.setattr(rprs, "blender_binary", lambda _args: "Blender")
    commands = []

    def fake_run(cmd, check=False):
        commands.append(cmd)
        topview_out = Path(cmd[cmd.index("--out") + 1])
        topview_out.write_bytes(b"png")
        return argparse.Namespace(returncode=134)

    monkeypatch.setattr(rprs.subprocess, "run", fake_run)

    def fake_repair(**kwargs):
        out = Path(kwargs["out_scene_path"])
        out.write_text(json.dumps({"items": [{"id": "chair1", "yaw_deg": 90}]}), encoding="utf-8")
        Path(kwargs["out_review_path"]).write_text("{}", encoding="utf-8")
        Path(kwargs["out_report_path"]).write_text("{}", encoding="utf-8")
        Path(kwargs["out_prompt_path"]).write_text("prompt", encoding="utf-8")
        return {"repaired_count": 1}

    monkeypatch.setattr(rprs, "run_topview_vlm_orientation_repair", fake_repair)
    monkeypatch.setattr(rprs, "maybe_build_blend_scene", lambda args, supplier_scene, out_dir: {"blend": str(out_dir / "rebuilt.blend"), "png": str(out_dir / "rebuilt.png")})

    out_scene, info, rebuilt = rprs.maybe_apply_topview_vlm_orientation_repair(
        args,
        supplier_scene=scene_path,
        out_dir=tmp_path,
        blend_artifacts={"blend": str(blend_path), "png": str(png_path), "build_report": str(build_report)},
    )

    assert out_scene.name.endswith(".topview_oriented.v1.json")
    assert info["target_count"] == 1
    assert info["report"] == {"repaired_count": 1}
    assert rebuilt["blend"].endswith("rebuilt.blend")
    assert json.loads((tmp_path / "topview_vlm_orientation.chairs.target_label_map.json").read_text(encoding="utf-8")) == {"C1": "chair1"}
    assert "--include-armchairs" in commands[0]
    assert "render-warning" in capsys.readouterr().err
    assert (tmp_path / "scene.before_topview_orientation.blend").is_file()
