import argparse
import json
import subprocess
import sys
import tarfile
import types
import zipfile
from pathlib import Path

import pytest

from src import trellis_supplier_asset_orchestrator as tso


def _args(tmp_path: Path, **overrides):
    card_path = tmp_path / "card.json"
    if not card_path.exists():
        card_path.write_text(
            json.dumps(
                {
                    "unique_key": "card-1",
                    "title": "Modern chair",
                    "category_norm": "chairs",
                    "dimensions_cm": {"width": 50, "depth": 60, "height": 90},
                    "images": [],
                }
            ),
            encoding="utf-8",
        )

    values = {
        "card_json": str(card_path),
        "catalog_json": "",
        "unique_key": "",
        "out_dir": str(tmp_path / "out"),
        "job_id": "job1",
        "prepared_job_dir": "",
        "prepare_only": False,
        "server_host": "gpu.example",
        "server_port": 2222,
        "server_user": "root",
        "ssh_key": str(tmp_path / "id_ed25519"),
        "remote_root": "/workspace/jobs",
        "remote_trellis_root": "/workspace/TRELLIS.2",
        "remote_model_dir": "/workspace/models/TRELLIS.2-4B",
        "remote_python": "/venv/trellis2/bin/python",
        "remote_worker_root": "/workspace/trellis2_worker",
        "remote_worker_timeout_sec": 5.0,
        "remote_worker_poll_sec": 0.1,
        "remote_persistent_worker": True,
        "remote_text_model_dir": "",
        "remote_cuda_visible_devices": 0,
        "mode": "multi_image",
        "multi_mode": "stochastic",
        "max_images": 2,
        "seed": 7,
        "sparse_steps": 4,
        "slat_steps": 5,
        "texture_size": 256,
        "simplify": 0.98,
        "pipeline_type": 512,
        "ss_guidance_strength": 7.5,
        "slat_guidance_strength": 3.0,
        "decimation_target": 50000,
        "pre_export_simplify_target": 0,
        "no_remesh": False,
        "remesh_band": 1,
        "remesh_project": 0.0,
        "no_webp": True,
        "image_size": 336,
        "fill_holes_resolution": 256,
        "fill_holes_num_views": 120,
        "remote_runner_path": "",
        "max_failures_per_candidate": 2,
        "progress_log": True,
        "allow_proxy_fallback": False,
        "force_trellis_image_only": False,
        "image_source_index": 0,
        "single_object_crop": False,
        "single_object_crop_component": "largest",
        "single_object_crop_padding": 0.16,
        "vlm_single_object_filter": False,
        "vlm_provider": "ollama",
        "vlm_ollama_url": "http://127.0.0.1:11435",
        "vlm_model": "llama3.2-vision:11b",
        "vlm_timeout": 5,
        "vlm_unload_after_filter": True,
        "text_fallback_if_no_single_image": True,
        "orientation_yaw_deg": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_validate_args_and_card_helpers(tmp_path):
    args = _args(tmp_path)
    tso.validate_trellis2_only_args(args)

    legacy = _args(tmp_path, remote_trellis_root="/workspace/TRELLIS", remote_python="/venv/trellis/bin/python")
    with pytest.raises(SystemExit, match="Legacy TRELLIS backend"):
        tso.validate_trellis2_only_args(legacy)

    card = {
        "unique_key": "u1",
        "source_site": "catalog",
        "title": "Dining table",
        "category_norm": "tables",
        "brand": "B",
        "style": "modern",
        "color": "oak",
        "materials": "wood",
        "description": "simple table",
        "dimensions_cm": {"width": "120", "depth": "80", "height": "75"},
        "preview_local_path": "a.jpg",
        "images": ["b.png", {"url": "c.webp"}, {"local_path": "d.jpg"}, "b.png"],
        "image_color_features": {"source_image": {"path": "e.jpg"}},
    }

    assert tso.safe_text("  x ") == "x"
    assert tso.slugify("Hello world/тест", max_len=10) == "hello_worl"
    assert len(tso.stable_hash(card, n=8)) == 8
    assert tso.dimensions_cm(card) == {"width": 120.0, "depth": 80.0, "height": 75.0}
    assert tso.target_size_m(card) == [1.2, 0.8, 0.75]
    assert tso.item_category_label(card) == "dining table"
    assert tso.item_category_label({"category_norm": "sofas"}) == "sofa"
    assert tso.item_category_label({"category_raw": "catalog > диваны"}) == "sofa"
    assert tso.candidate_image_sources(card) == ["a.jpg", "b.png", "c.webp", "d.jpg", "e.jpg"]
    assert tso.candidate_image_sources({"image_color_features": {"source_image": {"value": "v.jpg"}}}) == ["v.jpg"]
    assert "Dining table" in tso.build_text_prompt(card)
    assert "single standalone dining table" in tso.build_trellis_text_prompt(card)
    assert tso.has_text_description_for_trellis(card)

    normalized = tso.normalize_card_for_job(card, orientation_yaw_deg=90.0)
    assert normalized["schema"] == "trellis_supplier_card_job/v1"
    assert normalized["target_size_m"] == [1.2, 0.8, 0.75]
    assert normalized["orientation"]["yaw_deg"] == 90.0
    assert "raw_card_compact" in normalized

    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps({"items": [{"unique_key": "wanted", "title": "Lamp"}]}), encoding="utf-8")
    assert tso.load_card(_args(tmp_path, card_json="", catalog_json=str(catalog_path), unique_key="wanted"))["title"] == "Lamp"
    with pytest.raises(RuntimeError, match="unique_key not found"):
        tso.load_card(_args(tmp_path, card_json="", catalog_json=str(catalog_path), unique_key="missing"))
    bad_card = tmp_path / "bad_card.json"
    bad_card.write_text("[]", encoding="utf-8")
    with pytest.raises(RuntimeError, match="one JSON object"):
        tso.load_card(_args(tmp_path, card_json=str(bad_card)))
    with pytest.raises(RuntimeError, match="Provide either"):
        tso.load_card(_args(tmp_path, card_json="", catalog_json="", unique_key=""))


def test_build_cli_and_proxy_mesh_shapes_are_unit_tested(tmp_path, monkeypatch):
    parser = tso.build_cli()
    args = parser.parse_args(
        [
            "--card-json",
            "card.json",
            "--out-dir",
            str(tmp_path / "out"),
            "--server-host",
            "gpu.example",
            "--server-port",
            "32172",
            "--ssh-key",
            str(tmp_path / "id_ed25519"),
            "--remote-trellis-root",
            "/workspace/TRELLIS.2",
            "--remote-python",
            "/venv/trellis2/bin/python",
            "--remote-persistent-worker",
            "--mode",
            "single_image",
            "--max-images",
            "1",
            "--seed",
            "123",
            "--single-object-crop",
            "--vlm-single-object-filter",
            "--vlm-provider",
            "openrouter",
            "--orientation-yaw-deg",
            "90",
            "--allow-proxy-fallback",
        ]
    )
    assert args.server_host == "gpu.example"
    assert args.server_port == 32172
    assert args.mode == "single_image"
    assert args.single_object_crop is True
    assert args.vlm_provider == "openrouter"
    assert args.orientation_yaw_deg == 90.0

    created: list[tuple[str, object]] = []

    class FakeMesh:
        def __init__(self, kind, payload):
            self.kind = kind
            self.payload = payload
            self.metadata = {}
            self.translations = []
            created.append((kind, payload))

        def apply_translation(self, center):
            self.translations.append(center)

    class FakeScene:
        def __init__(self):
            self.parts = []

        def add_geometry(self, mesh, node_name):
            self.parts.append((mesh, node_name))

        def export(self, out_glb):
            Path(out_glb).write_text("|".join(name for _mesh, name in self.parts), encoding="utf-8")

    fake_trimesh = types.SimpleNamespace(
        creation=types.SimpleNamespace(
            box=lambda extents: FakeMesh("box", tuple(extents)),
            cylinder=lambda radius, height, sections=24: FakeMesh("cylinder", (radius, height, sections)),
        ),
        Scene=FakeScene,
    )
    monkeypatch.setitem(sys.modules, "trimesh", fake_trimesh)

    cases = [
        ("bed", "base|mattress|headboard|pillow_left|pillow_right"),
        ("nightstand", "case|drawer_1|handle_1|drawer_2|handle_2"),
        ("desk", "top|leg|leg|leg|leg"),
        ("chair", "seat|back|leg|leg|leg|leg"),
        ("floor lamp", "base|pole|shade"),
        ("plant", "pot|leaf|leaf|leaf|leaf|leaf|leaf|leaf"),
        ("decor box", "body"),
    ]
    for category, expected in cases:
        created.clear()
        out_glb = tmp_path / f"{category.replace(' ', '_')}.glb"
        tso._proxy_mesh_from_card(
            {
                "category_norm": category,
                "title": category,
                "target_size_m": [1.2, 0.8, 1.0],
            },
            out_glb,
        )
        assert out_glb.read_text(encoding="utf-8") == expected
        assert created


def test_direct_asset_resolution_and_patch_card(tmp_path):
    asset_dir = tmp_path / "asset"
    asset_dir.mkdir()
    model = asset_dir / "model.obj"
    model.write_text("o Cube\n", encoding="utf-8")
    mtl = asset_dir / "model.mtl"
    mtl.write_text("map_Kd tex.jpg\n", encoding="utf-8")
    (asset_dir / "tex.jpg").write_bytes(b"image-bytes")

    payload = tso.try_resolve_direct_model_asset({"asset_local_path": str(asset_dir)}, tmp_path / "job")

    assert payload is not None
    assert payload["ok"] is True
    assert payload["asset_format"] == "obj"
    assert Path(payload["asset_local_path"]).is_file()
    assert any(Path(p).name == "tex.jpg" for p in payload["asset_sidecar_paths"])

    patched = tso.patch_card_with_direct_asset({"title": "x"}, payload)
    assert patched["asset_status"] == "ready_existing_or_downloaded_model_asset"
    assert patched["extra"]["direct_supplier_model_asset"]["asset_format"] == "obj"


def test_direct_asset_archive_download_and_proxy_fallback(tmp_path, monkeypatch):
    assert tso._proxy_size_from_card({"target_size_m": ["0.01", "0.2", "bad"], "dimensions_cm": {}}) == (0.8, 0.45, 0.75)
    assert tso._proxy_size_from_card({"dimensions_cm": {"width": 120, "depth": 60, "height": 80}}) == (1.2, 0.6, 0.8)
    assert tso._url_or_path_suffix("https://example.test/model.tar.gz?x=1") == ".tar.gz"
    assert tso._looks_like_model_or_archive_ref("model.fbx")
    sources = tso._collect_direct_asset_sources(
        {
            "asset_local_path": ["a.obj", "a.obj"],
            "asset": {"model_download_url": {"url": "https://example.test/model.zip"}},
            "source": {"preview_url": "https://example.test/image.jpg"},
        }
    )
    assert [item["value"] for item in sources] == ["a.obj", "https://example.test/model.zip"]

    asset_dir = tmp_path / "asset"
    asset_dir.mkdir()
    small = asset_dir / "small.obj"
    big = asset_dir / "big.fbx"
    small.write_text("o small", encoding="utf-8")
    big.write_text("fbx" * 20, encoding="utf-8")
    mtl = asset_dir / "big.mtl"
    mtl.write_text("map_Kd textures/diffuse.jpg\n", encoding="utf-8")
    tex = asset_dir / "textures" / "diffuse.jpg"
    tex.parent.mkdir()
    tex.write_bytes(b"img")
    selected = tso._find_supported_model_file(asset_dir)
    assert selected == big.resolve()
    final_dir = tmp_path / "final"
    final_dir.mkdir()
    copied = tso._copy_direct_model_sidecars(selected, final_dir)
    assert any(path.endswith("big.mtl") for path in copied)
    assert any(path.endswith("textures/diffuse.jpg") for path in copied)

    zip_path = tmp_path / "model.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("nested/chair.obj", "o chair")
    zip_report = tso._extract_direct_archive(zip_path, tmp_path / "zip_extract")
    assert zip_report["method"] == "zipfile"

    tar_path = tmp_path / "model.tar.gz"
    tar_member = tmp_path / "table.obj"
    tar_member.write_text("o table", encoding="utf-8")
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(tar_member, arcname="table.obj")
    tar_report = tso._extract_direct_archive(tar_path, tmp_path / "tar_extract")
    assert tar_report["method"] == "tarfile"

    class FakeResponse:
        content = b"fbx"
        headers = {"content-type": "application/octet-stream"}

        def raise_for_status(self):
            return None

    monkeypatch.setattr(tso.requests, "get", lambda *args, **kwargs: FakeResponse())
    downloaded = tso._download_direct_asset("https://example.test/download/model.fbx", tmp_path / "downloads")
    assert downloaded.name == "model.fbx"
    assert downloaded.read_bytes() == b"fbx"

    payload = tso.try_resolve_direct_model_asset({"archive_url": str(zip_path)}, tmp_path / "job")
    assert payload and payload["asset_format"] == "obj"
    patched = tso.patch_card_with_direct_asset({"extra": {"old": True}}, payload)
    assert patched["extra"]["old"] is True
    assert patched["asset_local_path"].endswith(".obj")

    args = _args(tmp_path, job_id="proxyjob", allow_proxy_fallback=True)
    monkeypatch.setattr(tso, "_proxy_mesh_from_card", lambda card, out_glb: out_glb.write_bytes(b"glb"))
    summary = tso._run_local_proxy_fallback(tmp_path / "proxy_job", {"title": "Chair", "category_norm": "chair"}, args)
    assert summary["ok"] is True
    assert Path(summary["asset_glb"]).read_bytes() == b"glb"

    missing = tso.try_resolve_direct_model_asset(
        {"asset_local_path": str(tmp_path / "missing.obj"), "archive_url": "https://example.test/not-an-image.txt"},
        tmp_path / "failed_job",
    )
    assert missing is None
    attempts = tso.read_json(tmp_path / "failed_job" / "direct_model_asset" / "direct_model_asset_attempts.json")
    assert attempts["attempts"][0]["status"] == "skipped_local_path_not_found"

    with pytest.raises(RuntimeError, match="Unsupported archive type"):
        tso._extract_direct_archive(tmp_path / "model.unsupported", tmp_path / "unsupported_out")
    monkeypatch.setattr(tso.shutil, "which", lambda _name: None)
    seven = tmp_path / "model.7z"
    seven.write_bytes(b"7z")
    with pytest.raises(RuntimeError, match="requires 7z or unar"):
        tso._extract_direct_archive(seven, tmp_path / "seven_out")


def test_prepare_images_local_and_vlm_filter(tmp_path, monkeypatch):
    img1 = tmp_path / "one.jpg"
    img2 = tmp_path / "two.png"
    img1.write_bytes(b"img1")
    img2.write_bytes(b"img2")
    card = {"title": "chair", "category_norm": "chairs", "images": [str(img1), str(img2)]}

    args = _args(tmp_path, vlm_single_object_filter=True, max_images=1)
    reviews = []

    def fake_review(image, **kwargs):
        accepted = image.name.endswith(".jpg")
        row = {"accepted": accepted, "image": str(image), "parsed": {"reason": "unit"}}
        reviews.append(row)
        return row

    monkeypatch.setattr(tso, "ask_ollama_single_object_vlm", fake_review)
    monkeypatch.setattr(tso, "unload_ollama_model", lambda **_: {"ok": True})

    images, manifest = tso.prepare_images(card, tmp_path / "job", max_images=1, args=args)

    assert len(images) == 1
    assert images[0].name == "image_01.jpg"
    assert manifest["vlm_single_object_filter"] is True
    assert manifest["count"] == 1
    assert len(reviews) == 2

    with pytest.raises(RuntimeError, match="No images"):
        tso.prepare_images({"title": "empty"}, tmp_path / "empty", max_images=1, args=_args(tmp_path))

    with pytest.raises(RuntimeError, match="positive"):
        tso.prepare_images(card, tmp_path / "negative", max_images=1, image_source_index=-1, args=_args(tmp_path))
    with pytest.raises(RuntimeError, match="out of range"):
        tso.prepare_images(card, tmp_path / "range", max_images=1, image_source_index=3, args=_args(tmp_path))

    class FakeGetResponse:
        content = b"remote-image"
        headers = {"content-type": "image/png"}

        def raise_for_status(self):
            return None

    monkeypatch.setattr(tso.requests, "get", lambda *args, **kwargs: FakeGetResponse())
    images, manifest = tso.prepare_images(
        {"title": "remote", "preview_local_path": "https://example.test/image"},
        tmp_path / "remote",
        max_images=1,
        args=_args(tmp_path, vlm_single_object_filter=False),
    )
    assert images[0].suffix == ".png"
    assert manifest["prepared"][0]["source"] == "https://example.test/image"


def test_image_crop_and_vlm_api_helpers(tmp_path, monkeypatch):
    pytest.importorskip("PIL")
    from PIL import Image, ImageDraw

    image = tmp_path / "object.png"
    img = Image.new("RGBA", (80, 60), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle((20, 15, 50, 45), fill=(10, 20, 30, 255))
    img.save(image)
    bbox = tso._single_object_crop_bbox(image, component="largest")
    assert bbox is not None
    crop_report = tso.crop_single_object_image(image, tmp_path / "crop.jpg", component="largest", padding_ratio=0.1)
    assert Path(crop_report["cropped_image"]).is_file()

    assert tso.image_ext_from_response("https://x/a", "image/webp") == ".webp"
    assert tso._extract_json_object("prefix {\"single_object\": true} suffix") == {"single_object": True}
    encoded = tso._encode_image_base64(image)
    assert encoded
    assert "single chair" in tso._single_object_vlm_prompt("chair")

    class FakePostResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    posts = []
    monkeypatch.setattr(
        tso.requests,
        "post",
        lambda url, json, timeout: posts.append((url, json, timeout))
        or FakePostResponse({"message": {"content": "{\"single_object\": true, \"reason\": \"ok\"}"}}),
    )
    review = tso.ask_ollama_single_object_vlm(
        image,
        category_label="chair",
        ollama_url="http://ollama",
        model="vision",
        timeout_sec=3,
    )
    assert review["accepted"] is True
    unload = tso.unload_ollama_model(ollama_url="http://ollama", model="vision", timeout_sec=3)
    assert unload["ok"] is True
    assert posts

    fake_topview = types.ModuleType("topview_vlm_orientation_repair")
    fake_topview.call_openai_compatible_vlm = lambda **kwargs: {
        "choices": [{"message": {"content": "{\"single_object\": false, \"reason\": \"room scene\"}"}}]
    }
    monkeypatch.setitem(sys.modules, "topview_vlm_orientation_repair", fake_topview)
    openai_review = tso.ask_openai_compatible_single_object_vlm(
        image,
        provider="openrouter",
        category_label="chair",
        model="model",
        timeout_sec=3,
    )
    assert openai_review["accepted"] is False

    args = _args(tmp_path, vlm_single_object_filter=True, vlm_provider="openrouter", vlm_unload_after_filter=False)
    filtered, reviews = tso.filter_images_with_single_object_vlm([image], card={"category_norm": "chair"}, args=args)
    assert filtered == []
    assert reviews[0]["provider"] == "openrouter"

    monkeypatch.setenv("CGS_TRELLIS33_SKIP_VLM_FILTER", "1")
    skipped, skip_reviews = tso.filter_images_with_single_object_vlm([image], card={"category_norm": "chair"}, args=args)
    assert skipped == [image]
    assert skip_reviews[0]["stage"] == "trellis2_vlm_filter_skipped_by_env"


def test_ssh_scp_worker_queue_and_wait_are_mocked(tmp_path, monkeypatch):
    args = _args(tmp_path)
    calls = []

    def fake_stream(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "done\n", None)

    monkeypatch.setattr(tso, "run_cmd_stream", fake_stream)

    assert tso.ssh_base(args)[:3] == ["ssh", "-p", "2222"]
    assert tso.scp_base(args)[:4] == ["scp", "-P", "2222", "-r"]
    assert tso.ssh_run(args, "echo ok") == "done\n"
    tso.scp_to_remote(args, tmp_path / "x.json", "/remote/x.json")
    tso.scp_from_remote(args, "/remote/y.json", tmp_path / "local" / "y.json")
    assert any("root@gpu.example:/remote/x.json" in part for cmd in calls for part in cmd)
    assert (tmp_path / "local").is_dir()

    payload = tso.build_trellis2_worker_job_payload(args, "job", "/remote/job", "/remote/job/out.glb", "/remote/job/report.json")
    assert payload["schema"] == "trellis2_persistent_job/v1"
    assert payload["max_images"] == 2
    assert payload["slat_steps"] == 5

    ssh_scripts = []
    copied = []
    monkeypatch.setattr(tso, "ssh_run", lambda _args, script: ssh_scripts.append(script) or "done\n")
    monkeypatch.setattr(tso, "scp_to_remote", lambda _args, local, remote: copied.append((Path(local), remote)))
    remote_queue = tso.enqueue_remote_worker_job(args, payload, tmp_path)
    assert remote_queue.endswith("/queue/job.json")
    assert copied[0][1].endswith(".json.tmp")
    assert any("mkdir -p" in s for s in ssh_scripts)

    statuses = iter(["wait\n", "done\n"])
    monkeypatch.setattr(tso, "ssh_run", lambda *_: next(statuses))
    monkeypatch.setattr(tso.time, "sleep", lambda *_: None)
    assert tso.wait_remote_worker_job(args, "/remote/report.json", "/remote/out.glb") == "done"

    monkeypatch.setattr(tso, "ssh_run", lambda *_: "report_only\n")
    with pytest.raises(RuntimeError, match="without GLB"):
        tso.run_remote_trellis2_persistent(args, "job", "/remote/job", "/remote/glb", "/remote/report", tmp_path)

    monkeypatch.setattr(tso, "enqueue_remote_worker_job", lambda *_: "/queue/job.json")
    monkeypatch.setattr(tso, "ssh_run", lambda *_args: "worker stdout\n")
    monkeypatch.setattr(tso, "wait_remote_worker_job", lambda *_: "done")
    single = tso.run_remote_trellis2_single_run(args, "job", "/remote/job", "/remote/glb", "/remote/report", tmp_path)
    assert "[TRELLIS2][single-run]" in single
    assert "worker stdout" in single

    monkeypatch.setattr(tso, "wait_remote_worker_job", lambda *_: "report_only")
    with pytest.raises(RuntimeError, match="without GLB"):
        tso.run_remote_trellis2_single_run(args, "job", "/remote/job", "/remote/glb", "/remote/report", tmp_path)


def test_remote_runner_and_cache_paths_without_network(tmp_path, monkeypatch):
    args = _args(tmp_path)
    local_job_dir = tmp_path / "job"
    local_job_dir.mkdir()

    monkeypatch.setattr(tso, "ensure_remote_worker", lambda _args: None)
    monkeypatch.setattr(tso, "enqueue_remote_worker_job", lambda *_: "/worker/queue/job.json")
    monkeypatch.setattr(tso, "wait_remote_worker_job", lambda *_: "done")

    stdout = tso.run_remote_trellis2_persistent(args, "job", "/remote/job", "/remote/job/asset.glb", "/remote/job/report.json", local_job_dir)
    assert "wait_status=done" in stdout
    assert (local_job_dir / "trellis2_worker_queue_path.txt").read_text(encoding="utf-8") == "/worker/queue/job.json"

    monkeypatch.setattr(tso, "ssh_run", lambda *_: "done\n")
    assert tso.store_remote_trellis2_cache(args, {"unique_key": "u1"}, "/remote/a.glb", "/remote/r.json").endswith(
        tso.trellis2_generation_cache_key({"unique_key": "u1"}, args)
    )
    monkeypatch.setattr(tso, "ssh_run", lambda *_: (_ for _ in ()).throw(RuntimeError("offline")))
    assert tso.store_remote_trellis2_cache(args, {"unique_key": "u1"}, "/remote/a.glb", "/remote/r.json") is None

    assert "CUDA_VISIBLE_DEVICES=0" in tso.remote_env_prefix(args)
    assert "TrellisImageTo3DPipeline" in tso.remote_runner_code()
    assert "proxy_glb_fallback" in tso.remote_proxy_glb_code()
    assert tso.shell_quote("a'b") == "'a'\\''b'"


def test_run_command_wrappers_are_mocked(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        tso.subprocess,
        "run",
        lambda cmd, **kwargs: calls.append((cmd, kwargs)) or subprocess.CompletedProcess(cmd, 0, "ok\n", None),
    )
    completed = tso.run_cmd(["fake", "cmd"], cwd=tmp_path)
    assert completed.stdout == "ok\n"
    assert calls[0][1]["cwd"] == str(tmp_path)

    class FakeStdout:
        def __iter__(self):
            return iter(["a\n", "b\n"])

    class FakePopen:
        stdout = FakeStdout()

        def __init__(self, cmd, **kwargs):
            self.cmd = cmd
            self.kwargs = kwargs

        def wait(self):
            return 3

    monkeypatch.setattr(tso.subprocess, "Popen", FakePopen)
    with pytest.raises(subprocess.CalledProcessError) as exc:
        tso.run_cmd_stream(["fake"], check=True)
    assert exc.value.output == "a\nb\n"
    completed = tso.run_cmd_stream(["fake"], check=False)
    assert completed.returncode == 3


def test_reuse_and_enriched_result_without_remote_network(tmp_path, monkeypatch):
    args = _args(tmp_path)
    local_job_dir = tmp_path / "job"
    output_dir = local_job_dir / "output"
    output_dir.mkdir(parents=True)
    glb = output_dir / "asset.trellis.glb"
    glb.write_bytes(b"glb")
    report = output_dir / "trellis.report.json"
    report.write_text(json.dumps({"ok": True, "mode": "image", "total_sec": 1.2}), encoding="utf-8")

    summary = {
        "ok": True,
        "asset_glb": str(glb),
        "card_with_trellis_asset_json": str(local_job_dir / "card.with_trellis_asset.json"),
    }
    (local_job_dir / "card.with_trellis_asset.json").write_text("{}", encoding="utf-8")
    (local_job_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    assert tso.try_reuse_local_generated_asset(local_job_dir)["asset_path"] == str(glb.resolve())

    result = tso.build_enriched_result(
        card={"title": "Chair", "dimensions_cm": {"width": 50, "depth": 60, "height": 90}},
        normalized_card={"orientation": {"yaw_deg": None}},
        local_job_dir=local_job_dir,
        local_glb_path=glb,
        remote_report={"ok": True, "mode": "proxy_glb_fallback"},
        args=args,
    )
    assert result["asset_generation_method"] == "proxy_glb_fallback"
    assert result["asset_source"] == "supplier_catalog_procedural_proxy"
    assert tso.patch_card_with_asset({"extra": {}}, result)["extra"]["trellis_generated_asset"]["asset_format"] == "glb"

    monkeypatch.setattr(tso, "ssh_run", lambda *_: "done\n")

    def fake_scp(_args, remote_path, local_path):
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        if str(local_path).endswith(".glb"):
            Path(local_path).write_bytes(b"glb")
        else:
            Path(local_path).write_text(json.dumps({"ok": True, "mode": "image"}), encoding="utf-8")

    monkeypatch.setattr(tso, "scp_from_remote", fake_scp)
    reused = tso.try_reuse_remote_trellis2_asset(
        args=args,
        card={"unique_key": "u1"},
        normalized={"orientation": {"yaw_deg": None}},
        local_job_dir=tmp_path / "remote_reuse",
        job_id="job",
        remote_job_dir="/remote/job",
        candidate_cascade_report={},
    )
    assert reused["ok"] is True
    assert reused["reused_remote_trellis2_asset"] is True

    assert tso._summary_asset_path({"asset_glb": str(glb)}) == glb.resolve()
    assert tso._summary_asset_path({"asset_path": str(tmp_path / "missing.glb")}) is None
    assert tso._force_trellis_image_only(_args(tmp_path, force_trellis_image_only=True)) is True
    assert tso._remote_report_value({"total_job_sec": 3}, "total_sec") == 3
    assert tso._remote_report_value({"stages": {"export_sec": 2}}, "glb_export_sec") == 2
    text_result = tso.build_enriched_result(
        card={"unique_key": "u", "title": "Text item"},
        normalized_card={},
        local_job_dir=local_job_dir,
        local_glb_path=glb,
        remote_report={"mode": "text"},
        args=args,
    )
    assert text_result["asset_generation_method"] == "trellis2_text_to_3d"


def test_run_orchestration_candidate_fallback_uses_mocked_attempts(tmp_path, monkeypatch):
    card_path = tmp_path / "card.json"
    card_path.write_text(
        json.dumps(
            {
                "target_id": "target1",
                "unique_key": "base",
                "title": "Base chair",
                "candidate_pool": [
                    {"unique_key": "bad", "title": "Bad chair"},
                    {"unique_key": "good", "title": "Good chair"},
                ],
            }
        ),
        encoding="utf-8",
    )
    args = _args(tmp_path, card_json=str(card_path), out_dir=str(tmp_path / "out"), job_id="basejob", max_failures_per_candidate=1)
    attempts = []

    def fake_one(attempt_args):
        attempts.append(json.loads(Path(attempt_args.card_json).read_text(encoding="utf-8")))
        if len(attempts) == 1:
            raise RuntimeError("remote failed")
        asset = tmp_path / "ok.glb"
        asset.write_bytes(b"glb")
        return {"ok": True, "asset_glb": str(asset), "asset_path": str(asset)}

    monkeypatch.setattr(tso, "_run_orchestration_one", fake_one)
    monkeypatch.setattr(tso, "_cleanup_failed_job_dir", lambda *_: None)
    monkeypatch.setattr(
        tso,
        "_trellis_fallback_candidate_sequence",
        lambda **_: [
            {"unique_key": "bad", "title": "Bad chair"},
            {"unique_key": "good", "title": "Good chair"},
        ],
    )

    summary = tso.run_orchestration(args)

    assert summary["candidate_fallback"]["selected_candidate_index"] == 2
    assert summary["candidate_fallback"]["selected_unique_key"] == "good"
    assert [a["unique_key"] for a in attempts] == ["bad", "good"]


def test_run_orchestration_failure_and_proxy_fallback_paths(tmp_path, monkeypatch, capsys):
    card_path = tmp_path / "card.json"
    card_path.write_text(
        json.dumps(
            {
                "target_id": "target_proxy",
                "unique_key": "base",
                "title": "Base chair",
                "candidate_pool": [{"unique_key": "bad", "title": "Bad chair"}],
            }
        ),
        encoding="utf-8",
    )
    args = _args(tmp_path, card_json=str(card_path), out_dir=str(tmp_path / "out"), job_id="basejob", max_failures_per_candidate=1)
    monkeypatch.setattr(tso, "_run_orchestration_one", lambda _args: (_ for _ in ()).throw(RuntimeError("remote failed")))
    monkeypatch.setattr(tso, "_cleanup_failed_job_dir", lambda *_: None)
    monkeypatch.setattr(tso, "_trellis_fallback_candidate_sequence", lambda **_: [])

    with pytest.raises(RuntimeError, match="proxy fallback is disabled"):
        tso.run_orchestration(args)
    assert "fallback=disabled" in capsys.readouterr().out

    proxy_asset = tmp_path / "proxy.glb"

    def fake_proxy(local_job_dir, card, args):
        del local_job_dir, args
        proxy_asset.write_bytes(card.get("unique_key", "proxy").encode("utf-8"))
        return {"ok": True, "asset_glb": str(proxy_asset), "asset_generation_method": "proxy_glb_fallback"}

    proxy_args = _args(
        tmp_path,
        card_json=str(card_path),
        out_dir=str(tmp_path / "out_proxy"),
        job_id="basejob",
        max_failures_per_candidate=1,
        allow_proxy_fallback=True,
    )
    monkeypatch.setattr(tso, "_run_local_proxy_fallback", fake_proxy)
    summary = tso.run_orchestration(proxy_args)
    assert summary["asset_generation_method"] == "proxy_glb_fallback"
    assert proxy_asset.read_bytes() == b"bad"
    assert "proxy-fallback-start" in capsys.readouterr().out

    monkeypatch.setattr(tso, "_run_local_proxy_fallback", lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("proxy failed")))
    with pytest.raises(RuntimeError, match="proxy failed|remote failed"):
        tso.run_orchestration(proxy_args)


def test_run_orchestration_one_prepare_direct_and_error_paths(tmp_path, monkeypatch):
    image = tmp_path / "img.jpg"
    image.write_bytes(b"img")
    card_path = tmp_path / "prepare_card.json"
    card_path.write_text(json.dumps({"unique_key": "prep", "title": "Prep chair", "images": [str(image)]}), encoding="utf-8")
    args = _args(tmp_path, card_json=str(card_path), out_dir=str(tmp_path / "prep_out"), job_id="prepjob", prepare_only=True)
    summary = tso._run_orchestration_one(args)
    assert summary["prepare_only"] is True
    assert summary["accepted_image_count"] == 1

    model = tmp_path / "ready.glb"
    model.write_bytes(b"glb")
    direct_card = tmp_path / "direct_card.json"
    direct_card.write_text(json.dumps({"unique_key": "direct", "title": "Direct chair", "asset_local_path": str(model)}), encoding="utf-8")
    args = _args(tmp_path, card_json=str(direct_card), out_dir=str(tmp_path / "direct_out"), job_id="directjob")
    monkeypatch.setattr(tso, "try_reuse_remote_trellis2_asset", lambda **_kwargs: None)
    direct = tso._run_orchestration_one(args)
    assert direct["used_direct_supplier_asset"] is True
    assert direct["final_generation_mode"] == "direct_model"

    no_image_card = tmp_path / "no_image_card.json"
    no_image_card.write_text(json.dumps({"unique_key": "empty", "title": "No image"}), encoding="utf-8")
    args = _args(tmp_path, card_json=str(no_image_card), out_dir=str(tmp_path / "empty_out"), job_id="emptyjob")
    with pytest.raises(RuntimeError, match="requires product images"):
        tso._run_orchestration_one(args)


def test_run_orchestration_one_remote_success_is_fully_mocked(tmp_path, monkeypatch):
    image = tmp_path / "product.jpg"
    image.write_bytes(b"img")
    card_path = tmp_path / "remote_card.json"
    card_path.write_text(
        json.dumps(
            {
                "unique_key": "remote",
                "title": "Remote chair",
                "category_norm": "chair",
                "images": [str(image)],
                "dimensions_cm": {"width": 50, "depth": 60, "height": 90},
            }
        ),
        encoding="utf-8",
    )
    args = _args(
        tmp_path,
        card_json=str(card_path),
        out_dir=str(tmp_path / "remote_out"),
        job_id="remotejob",
        remote_persistent_worker=False,
        mode="multi_image",
    )
    ssh_scripts = []
    copied_to_remote = []

    monkeypatch.setattr(tso, "try_reuse_remote_trellis2_asset", lambda **_kwargs: None)
    monkeypatch.setattr(tso, "try_resolve_direct_model_asset", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(tso, "ssh_run", lambda _args, script: ssh_scripts.append(script) or "done\n")
    monkeypatch.setattr(tso, "scp_to_remote", lambda _args, local, remote: copied_to_remote.append((Path(local), remote)))
    monkeypatch.setattr(
        tso,
        "run_remote_trellis2_single_run",
        lambda **_kwargs: "[TRELLIS][remote-end] ok=True mode=image\nTRELLIS_FINAL_GENERATION_MODE=image\n",
    )

    def fake_scp_from_remote(_args, remote_path, local_path):
        local = Path(local_path)
        local.parent.mkdir(parents=True, exist_ok=True)
        if str(local).endswith(".glb"):
            local.write_bytes(b"glb")
        else:
            local.write_text(
                json.dumps(
                    {
                        "ok": True,
                        "mode": "image",
                        "total_job_sec": 3.0,
                        "stages": {"generation_sec": 2.0, "export_sec": 0.5},
                    }
                ),
                encoding="utf-8",
            )

    monkeypatch.setattr(tso, "scp_from_remote", fake_scp_from_remote)
    monkeypatch.setattr(tso, "store_remote_trellis2_cache", lambda *_args, **_kwargs: "/workspace/cache/remotejob")

    summary = tso._run_orchestration_one(args)

    assert summary["ok"] is True
    assert summary["remote_worker_mode"] == "single_run"
    assert summary["final_generation_mode"] == "image"
    assert summary["remote_trellis2_cache_dir"] == "/workspace/cache/remotejob"
    assert summary["candidate_cascade"]["image_trellis"]["ok"] is True
    assert any("rm -rf" in script for script in ssh_scripts)
    assert any(remote.endswith("/images") for _local, remote in copied_to_remote)
    assert Path(summary["asset_enriched_json"]).is_file()
    assert Path(summary["card_with_trellis_asset_json"]).is_file()


def test_run_orchestration_one_prepared_job_reuses_local_cache(tmp_path):
    job_dir = tmp_path / "prepared"
    output_dir = job_dir / "output"
    output_dir.mkdir(parents=True)
    glb = output_dir / "asset.trellis.glb"
    glb.write_bytes(b"glb")
    report = output_dir / "trellis.report.json"
    report.write_text(json.dumps({"ok": True, "mode": "image"}), encoding="utf-8")
    (job_dir / "card.raw.json").write_text(json.dumps({"unique_key": "prepared", "title": "Prepared chair"}), encoding="utf-8")
    (job_dir / "card.normalized.json").write_text(json.dumps({"unique_key": "prepared", "title": "Prepared chair"}), encoding="utf-8")
    (job_dir / "image_manifest.json").write_text(json.dumps({"images": [str(tmp_path / "missing.jpg")]}), encoding="utf-8")
    (job_dir / "card.with_trellis_asset.json").write_text(json.dumps({"asset_local_path": str(glb)}), encoding="utf-8")
    (job_dir / "summary.json").write_text(
        json.dumps(
            {
                "ok": True,
                "asset_glb": str(glb),
                "asset_path": str(glb),
                "asset_generation_mode": "image",
                "card_with_trellis_asset_json": str(job_dir / "card.with_trellis_asset.json"),
            }
        ),
        encoding="utf-8",
    )

    summary = tso._run_orchestration_one(
        _args(
            tmp_path,
            prepared_job_dir=str(job_dir),
            out_dir=str(tmp_path / "out"),
            job_id="",
        )
    )

    assert summary["asset_path"] == str(glb.resolve())


def test_run_orchestration_one_remote_text_recovery_and_proxy_modes(tmp_path, monkeypatch):
    image = tmp_path / "product.jpg"
    image.write_bytes(b"img")
    card_path = tmp_path / "remote_card.json"
    card_path.write_text(
        json.dumps(
            {
                "unique_key": "remote-text",
                "title": "Remote text chair",
                "category_norm": "chair",
                "images": [str(image)],
            }
        ),
        encoding="utf-8",
    )

    def install_common(mode: str, *, remote_stdout: str, report: dict):
        args = _args(
            tmp_path,
            card_json=str(card_path),
            out_dir=str(tmp_path / f"remote_{mode}"),
            job_id=f"remote_{mode}",
            remote_persistent_worker=False,
        )
        monkeypatch.setattr(tso, "try_reuse_remote_trellis2_asset", lambda **_kwargs: None)
        monkeypatch.setattr(tso, "try_resolve_direct_model_asset", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(tso, "scp_to_remote", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(tso, "ssh_run", lambda *_args, **_kwargs: "probe ok\n")
        monkeypatch.setattr(tso, "store_remote_trellis2_cache", lambda *_args, **_kwargs: "/remote/cache")

        def fake_scp_from_remote(_args, _remote_path, local_path):
            local = Path(local_path)
            local.parent.mkdir(parents=True, exist_ok=True)
            if str(local).endswith(".glb"):
                local.write_bytes(b"glb")
            else:
                local.write_text(json.dumps(report), encoding="utf-8")

        monkeypatch.setattr(tso, "scp_from_remote", fake_scp_from_remote)
        monkeypatch.setattr(tso, "run_remote_trellis2_single_run", lambda **_kwargs: remote_stdout)
        return args

    text_args = install_common(
        "text",
        remote_stdout="[TRELLIS][remote-end] ok=False mode=image trying text fallback\n[TRELLIS][remote-end] ok=True mode=text\n",
        report={"ok": True, "mode": "text", "total_sec": 1.0},
    )
    text_summary = tso._run_orchestration_one(text_args)
    assert text_summary["final_generation_mode"] == "text"
    assert text_summary["candidate_cascade"]["image_trellis"]["reason"] == "remote_image_failed"
    assert text_summary["candidate_cascade"]["text_trellis"]["ok"] is True

    proxy_args = install_common(
        "proxy",
        remote_stdout="TRELLIS_FINAL_GENERATION_MODE=image\n",
        report={"ok": True, "mode": "proxy_glb_fallback", "total_sec": 1.0},
    )
    proxy_summary = tso._run_orchestration_one(proxy_args)
    assert proxy_summary["final_generation_mode"] == "procedural_proxy"
    assert proxy_summary["remote_trellis2_cache_dir"] is None


def test_run_orchestration_one_remote_failure_handlers(tmp_path, monkeypatch):
    image = tmp_path / "product.jpg"
    image.write_bytes(b"img")
    card_path = tmp_path / "remote_card.json"
    card_path.write_text(json.dumps({"unique_key": "remote-fail", "title": "Remote fail", "images": [str(image)]}), encoding="utf-8")

    def make_args(label: str):
        return _args(
            tmp_path,
            card_json=str(card_path),
            out_dir=str(tmp_path / label),
            job_id=label,
            remote_persistent_worker=False,
        )

    monkeypatch.setattr(tso, "try_reuse_remote_trellis2_asset", lambda **_kwargs: None)
    monkeypatch.setattr(tso, "try_resolve_direct_model_asset", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(tso, "scp_to_remote", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(tso, "ssh_run", lambda *_args, **_kwargs: "done\n")
    monkeypatch.setattr(tso, "run_remote_trellis2_single_run", lambda **_kwargs: "TRELLIS_FINAL_GENERATION_MODE=image\n")

    def missing_glb_scp(_args, _remote_path, local_path):
        local = Path(local_path)
        local.parent.mkdir(parents=True, exist_ok=True)
        if str(local).endswith(".glb"):
            return
        local.write_text(json.dumps({"ok": True, "mode": "image"}), encoding="utf-8")

    monkeypatch.setattr(tso, "scp_from_remote", missing_glb_scp)
    with pytest.raises(RuntimeError, match="local GLB is missing"):
        tso._run_orchestration_one(make_args("missing_glb"))

    def bad_report_scp(_args, _remote_path, local_path):
        local = Path(local_path)
        local.parent.mkdir(parents=True, exist_ok=True)
        if str(local).endswith(".glb"):
            local.write_bytes(b"glb")
        else:
            local.write_text(json.dumps({"ok": False, "error": "bad asset", "mode": "image"}), encoding="utf-8")

    monkeypatch.setattr(tso, "scp_from_remote", bad_report_scp)
    with pytest.raises(RuntimeError, match="bad asset"):
        tso._run_orchestration_one(make_args("bad_report"))


def test_trellis_candidate_helpers_catalog_and_blacklist(tmp_path, monkeypatch, capsys):
    assert tso._trellis_fmt_duration(59.4).endswith("s")
    assert tso._trellis_fmt_duration(61) == "1m01s"
    assert tso._trellis_fmt_duration(3661) == "1h01m01s"
    assert tso._trellis_candidate_unique_key_v2({"title": "Only title"}) == "Only title"
    assert tso._trellis_candidate_unique_key_v2("bad") == "unknown"

    base = {
        "target_id": "bedroom_chair_1",
        "unique_key": "base",
        "size_m": [0.5, 0.5, 0.8],
        "candidate_pool": [{"unique_key": "c1", "title": "Chair one"}, {"candidate": {"unique_key": "c2", "title": "Chair two"}}],
        "extra": {"candidates": [{"unique_key": "c3", "title": "Chair three"}]},
    }
    pool = tso._trellis_extract_candidate_pool_v2(base)
    assert [row["unique_key"] for row in pool][:3] == ["base", "c1", "c2"]
    candidate_card = tso._trellis_make_candidate_card_v2(base, {"unique_key": "c4", "title": "Candidate"}, candidate_index=2, candidate_total=4)
    assert candidate_card["target_id"] == "bedroom_chair_1"
    assert candidate_card["extra"]["trellis_candidate_fallback"]["candidate_index"] == 2

    bad_blacklist = tmp_path / "blacklist.json"
    bad_blacklist.write_text("[]", encoding="utf-8")
    assert tso._trellis_blacklist_load_v2(bad_blacklist)["failures"] == {}
    assert tso._trellis_failure_count_v2(bad_blacklist, "target", "u") == 0
    count = tso._trellis_mark_failure_v2(bad_blacklist, "target", "u", RuntimeError("boom"), max_failures=2)
    assert count == 1
    assert tso._trellis_failure_count_v2(bad_blacklist, "target", "u") == 1

    assert tso._trellis_target_group_from_target_id("bedroom_wall_art_01") == "wall_art"
    assert "desk" in tso._trellis_group_aliases("desk")
    assert tso._trellis_candidate_matches_group({"category_norm": "chair", "title": "Dining chair"}, "chair")
    assert not tso._trellis_candidate_matches_group({"category_norm": "table", "title": "Table"}, "bed")
    assert tso._trellis_has_image_source({"extra": {"images": ["u"]}})
    assert tso._trellis_candidate_search_text({"source_site": "ikea", "images": ["https://ikea.com/img.jpg"]})
    assert tso._trellis_trusted_product_image_score({"source_site": "ikea", "images": ["https://ikea.com/img.jpg"]}) > 0
    assert tso._trellis_trusted_product_image_score({"source_site": "3ddd", "images": ["scene"]}) < 0
    assert tso._trellis_float_or_none("bad") is None
    assert tso._trellis_size_m_from_candidate({"width_cm": 50, "depth_cm": 60, "height_cm": 80}) == [0.5, 0.6, 0.8]
    assert tso._trellis_target_size_from_binding({"target": {"size_m": [1, 2, 3]}}) == [1.0, 2.0, 3.0]
    assert tso._trellis_size_score({"width": 1, "depth": 2, "height": 3}, [1, 2, 3]) == 30.0

    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "items": [
                    {"unique_key": "cat1", "title": "Dining chair", "category_norm": "chair", "images": ["https://ikea.com/chair.jpg"], "source_site": "ikea"},
                    {"unique_key": "cat2", "title": "Bed", "category_norm": "bed"},
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("TRELLIS_FALLBACK_CATALOG_JSON", str(catalog_path))
    monkeypatch.setattr(tso, "_TRELLIS_CATALOG_CACHE", None)
    assert tso._trellis_catalog_paths()[0] == catalog_path
    assert len(tso._trellis_collect_catalog_cards({"nested": {"row": {"unique_key": "x", "title": "T", "category_norm": "chair"}}})) == 1
    loaded = tso._trellis_load_catalog_cards()
    assert loaded["path"] == str(catalog_path)
    appended = tso._trellis_append_catalog_alternatives(
        {"target_size_m": [0.5, 0.6, 0.8]},
        "bedroom_chair_1",
        [],
        set(),
        max_catalog_candidates=2,
        require_images=True,
    )
    assert appended[0]["unique_key"] == "cat1"

    selected = tso._trellis_fallback_candidate_sequence(
        binding={"target_id": "bedroom_chair_1", "candidate_pool": [{"unique_key": "plain", "title": "Plain chair", "category_norm": "chair"}]},
        target_id="bedroom_chair_1",
        blacklist_path=tmp_path / "progress_blacklist.json",
        max_failures_per_candidate=2,
        max_candidate_pool=2,
    )
    assert selected
    assert tso._trellis_error_is_nonretryable(RuntimeError("No images found in product card"))
    assert not tso._trellis_error_is_nonretryable(RuntimeError("CUDA out of memory"))
    assert tso._trellis_mark_candidate_failure(tmp_path / "legacy_blacklist.json", "target", "bad", RuntimeError("only .max"), 2) == 2

    class Progress:
        def __init__(self):
            self.calls = []

        def update(self, **kwargs):
            self.calls.append(kwargs)

    progress = Progress()
    tso._trellis_progress_line(progress, target_id="t", status="ok", success=True)
    assert progress.calls[0]["success_delta"] == 1
    tso._trellis_progress_v2(
        t0=tso.time.monotonic(),
        done_attempts=0,
        total_attempts=1,
        target_id="t",
        candidate_index=1,
        candidate_total=1,
        attempt_index=1,
        max_failures=1,
        status="start",
        unique_key="u",
    )
    assert "[TRELLIS][progress]" in capsys.readouterr().out


def test_remaining_direct_asset_image_worker_and_candidate_edges(tmp_path, monkeypatch, capsys):
    legacy = _args(
        tmp_path,
        remote_model_dir="/workspace/trellis_models/TRELLIS-image-large",
        remote_text_model_dir="/workspace/trellis_models/TRELLIS-text-base",
        remote_runner_path="/workspace/legacy_runner.py",
    )
    with pytest.raises(SystemExit) as exc:
        tso.validate_trellis2_only_args(legacy)
    assert "remote_model_dir" in str(exc.value)
    assert "remote_text_model_dir" in str(exc.value)
    assert "--remote-runner-path" in str(exc.value)

    assert tso.now_stamp()
    doomed = tmp_path / "doomed"
    doomed.mkdir()
    monkeypatch.setattr(tso.shutil, "rmtree", lambda _path: (_ for _ in ()).throw(RuntimeError("locked")))
    tso._cleanup_failed_job_dir(doomed)
    assert doomed.exists()

    monkeypatch.setitem(sys.modules, "trimesh", None)
    with pytest.raises(RuntimeError, match="trimesh is required"):
        tso._proxy_mesh_from_card({"title": "chair"}, tmp_path / "proxy.glb")
    sys.modules.pop("trimesh", None)

    assert tso._proxy_size_from_card({"target_size_m": ["bad", 1, 1], "dimensions_cm": {"width": "bad"}}) == (0.8, 0.45, 0.75)
    assert tso.slugify("", max_len="bad") == "item"
    assert tso.slugify("abcdef", max_len="bad") == "abcdef"
    assert tso.item_category_label({}) == "object"
    assert tso.dimensions_cm({"dimensions_cm": {"width": "bad", "depth": 10, "height": None}})["width"] is None
    catalog_bad = tmp_path / "bad_catalog.json"
    catalog_bad.write_text(json.dumps({"items": {}}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="catalog-json"):
        tso.load_card(_args(tmp_path, card_json="", catalog_json=str(catalog_bad), unique_key="x"))

    asset_dir = tmp_path / "asset_bad_sidecars"
    asset_dir.mkdir()
    selected = asset_dir / "model.obj"
    selected.write_text("o model", encoding="utf-8")
    mtl = asset_dir / "model.mtl"
    mtl.write_text(
        "\n".join(
            [
                "map_Kd ../escape.jpg",
                "map_Kd missing/nested/diffuse.jpg",
                "map_Kd diffuse.jpg",
            ]
        ),
        encoding="utf-8",
    )
    (asset_dir / "sub").mkdir()
    nested_tex = asset_dir / "sub" / "diffuse.jpg"
    nested_tex.write_bytes(b"img")
    sidecar_dir = tmp_path / "sidecars"
    sidecar_dir.mkdir()
    copied = tso._copy_direct_model_sidecars(selected, sidecar_dir)
    assert any(path.endswith("diffuse.jpg") for path in copied)

    monkeypatch.setattr(tso.shutil, "which", lambda name: f"/usr/bin/{name}" if name in {"7z", "unar"} else None)
    calls = []

    def fake_extract(cmd, **_kwargs):
        calls.append(cmd[0])
        if cmd[0] == "7z":
            raise RuntimeError("7z failed")
        return subprocess.CompletedProcess(cmd, 0, "ok\n", None)

    monkeypatch.setattr(tso, "run_cmd_stream", fake_extract)
    rar = tmp_path / "archive.rar"
    rar.write_bytes(b"rar")
    report = tso._extract_direct_archive(rar, tmp_path / "rar_out")
    assert report["method"] == "unar"
    assert calls == ["7z", "unar"]

    def fake_get(url, **_kwargs):
        class Response:
            content = b"model-data"
            headers = {"content-type": "application/octet-stream"}

            def raise_for_status(self):
                if "fail" in url:
                    raise RuntimeError("download failed")

        return Response()

    monkeypatch.setattr(tso.requests, "get", fake_get)
    no_model_dir = tmp_path / "no_model_dir"
    no_model_dir.mkdir()
    (no_model_dir / "readme.txt").write_text("no model", encoding="utf-8")
    weird = tmp_path / "asset.txt"
    weird.write_text("not model", encoding="utf-8")
    empty_zip = tmp_path / "empty.zip"
    with zipfile.ZipFile(empty_zip, "w") as zf:
        zf.writestr("readme.txt", "no model")
    result = tso.try_resolve_direct_model_asset(
        {
            "asset_local_path": [str(no_model_dir), str(empty_zip), str(weird)],
            "asset": {"model_download_url": "https://example.test/fail.fbx"},
        },
        tmp_path / "direct_failures",
    )
    assert result is None
    attempts = tso.read_json(tmp_path / "direct_failures" / "direct_model_asset" / "direct_model_asset_attempts.json")["attempts"]
    assert {attempt["status"] for attempt in attempts} >= {
        "failed_no_supported_model_in_dir",
        "failed_no_supported_model_in_archive",
        "skipped_unsupported_suffix",
        "failed",
    }

    pytest.importorskip("PIL")
    from PIL import Image, ImageDraw

    multi = tmp_path / "multi.png"
    image = Image.new("RGB", (100, 120), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 5, 55, 20), fill=(10, 10, 10))
    draw.rectangle((35, 50, 60, 75), fill=(20, 20, 20))
    draw.rectangle((30, 95, 65, 115), fill=(30, 30, 30))
    image.save(multi)
    assert tso._single_object_crop_bbox(multi, component="top")[1] < 20
    assert 40 <= tso._single_object_crop_bbox(multi, component="middle")[1] <= 60
    assert tso._single_object_crop_bbox(multi, component="bottom")[1] >= 90
    blank = tmp_path / "blank.png"
    Image.new("RGB", (32, 32), (255, 255, 255)).save(blank)
    assert tso._single_object_crop_bbox(blank, component="largest") is None
    with pytest.raises(RuntimeError, match="Could not detect"):
        tso.crop_single_object_image(blank, tmp_path / "blank_crop.jpg")

    assert tso.image_ext_from_response("https://x/noext", "image/jpeg") == ".jpg"
    assert tso.image_ext_from_response("https://x/noext", None) == ".jpg"
    assert tso._extract_json_object("") == {}
    assert tso._extract_json_object("no json here") == {}
    assert tso._extract_json_object("{not-json}") == {}

    monkeypatch.setattr(tso.requests, "post", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("ollama down")))
    unload = tso.unload_ollama_model(ollama_url="http://ollama", model="vision", timeout_sec=1)
    assert unload["ok"] is False

    local_img = tmp_path / "local.jpg"
    local_img.write_bytes(b"img")
    missing_img = tmp_path / "missing.jpg"
    args = _args(tmp_path, vlm_single_object_filter=False)
    images, manifest = tso.prepare_images(
        {"title": "img", "images": [str(missing_img), str(local_img)]},
        tmp_path / "selected_image",
        max_images=1,
        image_source_index=2,
        single_object_crop=False,
        args=args,
    )
    assert images[0].name == "image_01.jpg"
    assert manifest["selected_sources"] == [str(local_img)]

    crop_src = tmp_path / "crop_src.png"
    Image.new("RGB", (40, 40), (255, 255, 255)).save(crop_src)
    crop_img = Image.open(crop_src)
    draw = ImageDraw.Draw(crop_img)
    draw.rectangle((10, 10, 30, 30), fill=(1, 2, 3))
    crop_img.save(crop_src)
    cropped_images, cropped_manifest = tso.prepare_images(
        {"title": "crop", "images": [str(crop_src)]},
        tmp_path / "crop_prepare",
        max_images=1,
        single_object_crop=True,
        single_object_crop_component="largest",
        single_object_crop_padding=0.0,
        args=_args(tmp_path, vlm_single_object_filter=False),
    )
    assert cropped_images[0].name.endswith(".single_object.png")
    assert "single_object_crop" in cropped_manifest["prepared"][0]

    monkeypatch.setattr(tso, "ssh_run", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("offline")))
    monotonic_values = iter([0.0, 1.0])
    monkeypatch.setattr(tso.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(tso.time, "sleep", lambda *_args: None)
    with pytest.raises(TimeoutError, match="probe_error"):
        tso.wait_remote_worker_job(_args(tmp_path, remote_worker_timeout_sec=0.1), "/remote/report", "/remote/glb")

    monkeypatch.setattr(tso, "ask_ollama_single_object_vlm", lambda *args, **kwargs: {"accepted": True, "parsed": {}})
    monkeypatch.setattr(tso, "unload_ollama_model", lambda **kwargs: {"ok": False, "error": "down"})
    tso.filter_images_with_single_object_vlm(
        [local_img],
        card={"category_norm": "chair"},
        args=_args(tmp_path, vlm_single_object_filter=True, vlm_provider="ollama", vlm_unload_after_filter=True),
    )
    assert "failed to unload" in capsys.readouterr().err

    nested_pool = tso._trellis_extract_candidate_pool_v2({"candidate_pool": {"items": [{"unique_key": "nested", "title": "Nested"}]}})
    assert any(row.get("unique_key") == "nested" for row in nested_pool)
    assert tso._trellis_candidate_matches_group({"category_norm": "bedside table", "title": "bedside table"}, "bed") is False
    assert tso._trellis_target_size_from_binding({"item": {"size_m": [1, 2, 3]}}) == [1.0, 2.0, 3.0]
    assert tso._trellis_size_score({"width": "bad"}, [1, 2, 3]) == 0.0
    assert tso._trellis_append_catalog_alternatives({}, "unknown_target", [], set()) == []


def test_more_trellis_orchestrator_edge_branches_without_remote_services(tmp_path, monkeypatch, capsys):
    class BadResponse:
        content = b"not-image"
        headers = {"content-type": "image/png"}

        def raise_for_status(self):
            raise RuntimeError("network failed")

    monkeypatch.setattr(tso.requests, "get", lambda *_args, **_kwargs: BadResponse())
    with pytest.raises(RuntimeError, match="All product images failed"):
        tso.prepare_images(
            {"title": "remote-only", "images": ["https://example.test/bad.png"]},
            tmp_path / "bad_remote_images",
            max_images=1,
            args=_args(tmp_path),
        )
    assert "Failed to fetch image" in capsys.readouterr().err

    with pytest.raises(RuntimeError, match="All product images failed"):
        tso.prepare_images(
            {"title": "missing-local", "images": [str(tmp_path / "missing.jpg")]},
            tmp_path / "missing_local_images",
            max_images=1,
            args=_args(tmp_path),
        )
    assert "Local image not found" in capsys.readouterr().err

    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    trellis_dir = tmp_path / "with_vlm" / "trellis_input_images"
    trellis_dir.mkdir(parents=True)
    (trellis_dir / "old.jpg").write_bytes(b"old")
    monkeypatch.setattr(
        tso,
        "filter_images_with_single_object_vlm",
        lambda images, **_kwargs: ([images[0]], [{"accepted": True}]),
    )
    images, manifest = tso.prepare_images(
        {"title": "two-local", "images": [str(first), str(second)]},
        tmp_path / "with_vlm",
        max_images=1,
        args=_args(tmp_path, vlm_single_object_filter=True, vlm_unload_after_filter=False),
    )
    assert images == [tmp_path / "with_vlm" / "trellis_input_images" / "image_01.jpg"]
    assert not (trellis_dir / "old.jpg").exists()
    assert manifest["vlm_reviews"][0]["accepted"] is True

    class GoodDownload:
        content = b"model-data"
        headers = {"content-type": "application/octet-stream"}

        def raise_for_status(self):
            return None

    monkeypatch.setattr(tso.requests, "get", lambda *_args, **_kwargs: GoodDownload())
    assert tso._download_direct_asset("https://example.test/download?file=model", tmp_path / "dl_no_suffix").name == "download"
    assert tso._download_direct_asset("https://example.test/", tmp_path / "dl_no_name").name == "downloaded_asset.bin"

    job_id = tso.build_job_id({}, _args(tmp_path, job_id="", seed=123))
    assert "__supplier_item__" in job_id
    fallback_identity = tso.stable_hash(tso.compact_card({}), n=16)
    cache_key = tso.trellis2_generation_cache_key({}, _args(tmp_path))
    assert cache_key.startswith(fallback_identity)
    glb = tmp_path / "generated.glb"
    glb.write_bytes(b"glb")
    enriched = tso.build_enriched_result(
        card={"title": "Proxy item"},
        normalized_card={},
        local_job_dir=tmp_path,
        local_glb_path=glb,
        remote_report={"ok": True, "generation_source": "proxy_glb_fallback"},
        args=_args(tmp_path),
    )
    assert enriched["asset_generation_method"] == "proxy_glb_fallback"
    assert tso._remote_report_value({"stages": {"load_model_sec": 4}}, "load_model_sec") == 4

    assert tso._trellis_candidate_key_safe("not-a-dict") == ""
    assert tso._trellis_candidate_group_text("not-a-dict") == ""
    assert not tso._trellis_candidate_matches_group({}, "chair")
    assert not tso._trellis_has_image_source("bad")
    assert tso._trellis_has_image_source({"extra": {"preview_path": "preview.jpg"}})
    assert tso._trellis_candidate_search_text({"extra": {"preview_images": ["img"], "source_site": "shop", "brand": "B"}})
    assert tso._trellis_trusted_product_image_score("bad") == 0.0
    assert tso._trellis_size_m_from_candidate("bad") is None
    assert tso._trellis_size_m_from_candidate({"width": 1, "depth": 2, "height": 3}) == [1.0, 2.0, 3.0]
    with pytest.raises(ValueError):
        tso._trellis_target_size_from_binding({"size_m": ["bad", 2, 3]})
    assert tso._trellis_size_score({"width": "bad"}, [1, 2, 3]) == 0.0

    list_catalog = tmp_path / "list_catalog.json"
    list_catalog.write_text(json.dumps([{"unique_key": "l1", "title": "Chair", "category_norm": "chair"}]), encoding="utf-8")
    monkeypatch.setenv("TRELLIS_FALLBACK_CATALOG_JSON", str(list_catalog))
    monkeypatch.setattr(tso, "_TRELLIS_CATALOG_CACHE", None)
    assert tso._trellis_collect_catalog_cards([{"unique_key": "x"}]) == [{"unique_key": "x"}]
    assert tso._trellis_load_catalog_cards()["cards"][0]["unique_key"] == "l1"

    monkeypatch.setattr(tso, "_TRELLIS_CATALOG_CACHE", {"path": "unit", "cards": []})
    unchanged = [{"unique_key": "base", "title": "Mystery object"}]
    assert tso._trellis_append_catalog_alternatives({}, "unknown_target", unchanged, set()) is unchanged

    bad_blacklist = tmp_path / "blacklist_bad.json"
    bad_blacklist.write_text("{bad", encoding="utf-8")
    assert tso._trellis_blacklist_load_v2(bad_blacklist)["failures"] == {}

    class FailingProgress:
        def update(self, **_kwargs):
            raise RuntimeError("progress broken")

    tso._trellis_progress_line(FailingProgress(), target_id="target", status="ok")
    assert "progress-warning" in capsys.readouterr().out


def test_trellis_remaining_direct_asset_cache_and_reuse_edges(tmp_path, monkeypatch):
    assert tso._url_or_path_suffix("") == ""
    assert tso._proxy_size_from_card({"target_size_m": [0.01, 0.02, 0.03]}) == (0.05, 0.05, 0.05)

    model = tmp_path / "direct.glb"
    model.write_bytes(b"glb")
    assert tso._find_supported_model_file(model) == model.resolve()
    assert tso._find_supported_model_file(tmp_path / "missing_dir") is None
    empty = tmp_path / "empty"
    empty.mkdir()
    assert tso._find_supported_model_file(empty) is None

    selected = tmp_path / "selected.obj"
    selected.write_text("o model", encoding="utf-8")

    def fail_iterdir(_self):
        raise OSError("no access")

    monkeypatch.setattr(tso.Path, "iterdir", fail_iterdir)
    assert tso._copy_direct_model_sidecars(selected, tmp_path / "final") == []

    monkeypatch.setattr(tso.shutil, "which", lambda name: f"/usr/bin/{name}" if name in {"7z", "unar"} else None)
    monkeypatch.setattr(tso, "run_cmd_stream", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("extract failed")))
    archive = tmp_path / "bad.7z"
    archive.write_bytes(b"7z")
    with pytest.raises(RuntimeError, match="Archive extraction failed"):
        tso._extract_direct_archive(archive, tmp_path / "bad_extract")

    missing_summary_dir = tmp_path / "missing_summary"
    assert tso.try_reuse_local_generated_asset(missing_summary_dir) is None

    bad_json_dir = tmp_path / "bad_json"
    bad_json_dir.mkdir()
    (bad_json_dir / "summary.json").write_text("{bad", encoding="utf-8")
    assert tso.try_reuse_local_generated_asset(bad_json_dir) is None

    list_summary_dir = tmp_path / "list_summary"
    list_summary_dir.mkdir()
    (list_summary_dir / "summary.json").write_text("[]", encoding="utf-8")
    assert tso.try_reuse_local_generated_asset(list_summary_dir) is None

    missing_asset_dir = tmp_path / "missing_asset"
    missing_asset_dir.mkdir()
    tso.write_json(missing_asset_dir / "summary.json", {"ok": True, "asset_glb": str(missing_asset_dir / "missing.glb")})
    assert tso.try_reuse_local_generated_asset(missing_asset_dir) is None

    bad_card_ref_dir = tmp_path / "bad_card_ref"
    bad_card_ref_dir.mkdir()
    reuse_glb = bad_card_ref_dir / "asset.glb"
    reuse_glb.write_bytes(b"glb")
    tso.write_json(
        bad_card_ref_dir / "summary.json",
        {"ok": True, "asset_glb": str(reuse_glb), "card_with_trellis_asset_json": str(bad_card_ref_dir / "missing_card.json")},
    )
    assert tso.try_reuse_local_generated_asset(bad_card_ref_dir) is None

    finalize_dir = tmp_path / "finalize_reuse"
    finalize_dir.mkdir()
    final_glb = finalize_dir / "asset.trellis.glb"
    final_glb.write_bytes(b"glb")
    report_path = finalize_dir / "trellis.report.json"
    report_path.write_text("[]", encoding="utf-8")
    cascade = {}
    summary = tso._finalize_reused_remote_trellis_asset(
        card={"title": "Reuse chair", "category_norm": "chair"},
        normalized={"title": "Reuse chair"},
        local_job_dir=finalize_dir,
        local_glb=final_glb,
        local_report=report_path,
        args=_args(tmp_path),
        job_id="reuse_job",
        remote_source_glb="/remote/asset.glb",
        remote_source_report="/remote/report.json",
        reuse_source="unit",
        candidate_cascade_report=cascade,
    )
    assert summary["reused_remote_trellis2_asset"] is True
    assert summary["remote_reuse_source"] == "unit"
    assert cascade["remote_trellis2_cache"]["ok"] is True
    assert tso.read_json(report_path)["reused_remote_trellis2_asset"] is True


def test_trellis_edge_branches_for_assets_vlm_and_catalog(tmp_path, monkeypatch):
    assert tso.image_ext_from_response("https://example.test/photo.png?x=1", None) == ".png"
    with pytest.raises(RuntimeError, match="Unsupported --vlm-provider"):
        tso.filter_images_with_single_object_vlm(
            [tmp_path / "image.jpg"],
            card={"category_norm": "chair"},
            args=_args(tmp_path, vlm_single_object_filter=True, vlm_provider="bad-provider", vlm_unload_after_filter=False),
        )

    out_glb = tmp_path / "proxy" / "output" / "asset.trellis.glb"
    out_glb.parent.mkdir(parents=True)
    out_glb.write_bytes(b"old")
    monkeypatch.setattr(tso, "_proxy_mesh_from_card", lambda _card, path: Path(path).write_bytes(b"new"))
    proxy = tso._run_local_proxy_fallback(tmp_path / "proxy", {"title": "Lamp", "category_norm": "lamp"}, _args(tmp_path))
    assert Path(proxy["asset_glb"]).read_bytes() == b"new"

    model_dir = tmp_path / "model_dir"
    model_dir.mkdir()
    obj = model_dir / "bad_stat.obj"
    obj.write_text("o bad", encoding="utf-8")
    real_stat = Path.stat

    def fake_stat(path, *args, **kwargs):
        if Path(path) == obj:
            raise OSError("stat blocked")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", fake_stat)
    assert tso._find_supported_model_file(model_dir) == obj.resolve()
    monkeypatch.setattr(Path, "stat", real_stat)

    sidecar_dir = tmp_path / "sidecars"
    sidecar_dir.mkdir()
    selected = sidecar_dir / "model.obj"
    selected.write_text("o model", encoding="utf-8")
    mtl = sidecar_dir / "model.mtl"
    mtl.write_text(
        "\n".join(
            [
                "",
                "# comment",
                "badline",
                "map_Kd",
                "map_Kd ../escape.jpg",
                "map_Kd missing.jpg",
                "map_Kd nested/tex.jpg",
            ]
        ),
        encoding="utf-8",
    )
    tex = sidecar_dir / "deep" / "tex.jpg"
    tex.parent.mkdir()
    tex.write_bytes(b"img")
    final_sidecars = tmp_path / "final_sidecars"
    final_sidecars.mkdir()
    copied = tso._copy_direct_model_sidecars(selected, final_sidecars)
    assert any(path.endswith("model.mtl") for path in copied)
    assert any(path.endswith("nested/tex.jpg") for path in copied)

    assert tso._trellis_extract_candidate_pool_v2(["bad"]) == []
    pool = tso._trellis_extract_candidate_pool_v2(
        {
            "unique_key": "root",
            "candidate_pool": [
                "bad",
                {"candidate": {"unique_key": "wrapped", "title": "Wrapped"}},
                {"candidate": "bad"},
                {"unique_key": "wrapped", "title": "Duplicate"},
            ],
            "extra": {"top_candidates": {"items": [{"unique_key": "extra", "title": "Extra"}]}},
        }
    )
    pool_keys = [item.get("unique_key") for item in pool]
    assert pool_keys[:2] == ["root", "wrapped"]
    assert "extra" in pool_keys

    bl = tmp_path / "blacklist.json"
    bl.write_text(json.dumps({"failures": "bad"}), encoding="utf-8")
    assert tso._trellis_failure_count_v2(bl, "t", "u") == 0
    bl.write_text(json.dumps({"failures": {"t||u": "bad-record"}}), encoding="utf-8")
    assert tso._trellis_failure_count_v2(bl, "t", "u") == 0
    count = tso._trellis_mark_failure_v2(bl, "t", "u", RuntimeError("broken"), max_failures=2)
    assert count == 1

    assert tso._trellis_candidate_key_safe(object()) == ""
    assert tso._trellis_candidate_matches_group({}, "chair") is False
    assert tso._trellis_candidate_matches_group({"category_norm": "table", "title": "Bedside table"}, "bed") is False
    assert tso._trellis_candidate_matches_group({"category_norm": "bed", "title": "Queen bed"}, "bed") is True
    assert tso._trellis_candidate_matches_group({"title": "Chair"}, "") is False
    assert tso._trellis_has_image_source({"preview_local_path": "x.jpg"}) is True
    assert tso._trellis_trusted_product_image_score({"source_site": "retailer", "title": "plain"}) == 60.0
    assert tso._trellis_target_size_from_binding(["bad"]) is None
    assert tso._trellis_size_score({"width_cm": "bad", "depth_cm": 1, "height_cm": 1}, [1, 1, 1]) == 0.0

    monkeypatch.setenv("TRELLIS_FALLBACK_CATALOG_JSON", str(tmp_path / "missing_catalog.json"))
    assert (tmp_path / "missing_catalog.json") not in tso._trellis_catalog_paths()
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "nested": {
                    "items": [
                        {"unique_key": "bad", "title": "No group"},
                        {
                            "unique_key": "chair1",
                            "title": "Good chair",
                            "category_norm": "chair",
                            "images": ["https://ikea.com/chair.jpg"],
                            "source_site": "retailer",
                            "product_url": "https://product",
                            "width_cm": 50,
                            "depth_cm": 60,
                            "height_cm": 90,
                        },
                        {
                            "unique_key": "chair_no_img",
                            "title": "No image chair",
                            "category_norm": "chair",
                        },
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("TRELLIS_FALLBACK_CATALOG_JSON", str(catalog_path))
    monkeypatch.setattr(tso, "_TRELLIS_CATALOG_CACHE", None)
    loaded = tso._trellis_load_catalog_cards()
    assert {item["unique_key"] for item in loaded["cards"]} == {"chair1", "chair_no_img"}
    pool = tso._trellis_append_catalog_alternatives(
        binding={"target_size_m": [0.5, 0.6, 0.9]},
        target_id="bedroom_chair_1",
        pool=[],
        seen=set(),
        max_catalog_candidates=2,
        require_images=True,
    )
    assert [item["unique_key"] for item in pool] == ["chair1"]
    assert tso._trellis_append_catalog_alternatives({}, "unknown", [], set()) == []


def test_trellis_run_orchestration_recovery_and_failure_edges(tmp_path, monkeypatch, capsys):
    image = tmp_path / "product.jpg"
    image.write_bytes(b"img")
    card_path = tmp_path / "remote_card.json"
    card_path.write_text(
        json.dumps({"unique_key": "remote", "title": "Remote chair", "category_norm": "chair", "images": [str(image)]}),
        encoding="utf-8",
    )

    args = _args(
        tmp_path,
        card_json=str(card_path),
        out_dir=str(tmp_path / "remote_recover"),
        job_id="remote_recover",
        remote_persistent_worker=False,
        mode="single_image",
        vlm_single_object_filter=True,
    )
    ssh_scripts = []
    monkeypatch.setattr(tso, "try_reuse_remote_trellis2_asset", lambda **_kwargs: None)
    monkeypatch.setattr(tso, "try_resolve_direct_model_asset", lambda *_args, **_kwargs: None)
    def fake_prepare_images(_card, local_job_dir, **_kwargs):
        images_dir = Path(local_job_dir) / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        local_image = images_dir / "product.jpg"
        local_image.write_bytes(b"img")
        return [local_image], {"images": [str(local_image)], "count": 1, "trellis_input_dir": str(images_dir)}

    monkeypatch.setattr(tso, "prepare_images", fake_prepare_images)
    monkeypatch.setattr(
        tso,
        "ask_ollama_single_object_vlm",
        lambda image, **_kwargs: {"accepted": True, "image": str(image), "parsed": {"reason": "unit"}},
    )
    monkeypatch.setattr(tso, "unload_ollama_model", lambda **_kwargs: {"ok": True})
    monkeypatch.setattr(tso, "ssh_run", lambda _args, script: ssh_scripts.append(script) or "done\n")
    monkeypatch.setattr(tso, "scp_to_remote", lambda *_args, **_kwargs: None)

    def raise_after_remote(**_kwargs):
        raise subprocess.CalledProcessError(
            2,
            ["remote"],
            output="[TRELLIS][remote-end] ok=False mode=image trying text fallback\n",
        )

    monkeypatch.setattr(tso, "run_remote_trellis2_single_run", raise_after_remote)

    def fake_scp_from_remote(_args, remote_path, local_path):
        local = Path(local_path)
        local.parent.mkdir(parents=True, exist_ok=True)
        if str(local).endswith(".glb"):
            local.write_bytes(b"glb")
        else:
            local.write_text(json.dumps({"ok": True, "mode": "text"}), encoding="utf-8")

    monkeypatch.setattr(tso, "scp_from_remote", fake_scp_from_remote)
    monkeypatch.setattr(tso, "store_remote_trellis2_cache", lambda *_args, **_kwargs: "/cache/recover")

    summary = tso._run_orchestration_one(args)
    assert summary["final_generation_mode"] == "text"
    assert summary["candidate_cascade"]["text_trellis"]["ok"] is True
    assert any("test -s" in script for script in ssh_scripts)
    assert "[TRELLIS][recover]" in capsys.readouterr().out

    persistent_args = _args(
        tmp_path,
        card_json=str(card_path),
        out_dir=str(tmp_path / "remote_persistent"),
        job_id="remote_persistent",
        remote_persistent_worker=True,
    )

    def fake_persistent(**kwargs):
        (Path(kwargs["local_job_dir"]) / "trellis2_worker_queue_path.txt").write_text("/queue/job.json", encoding="utf-8")
        return "[TRELLIS][remote-end] ok=True mode=image\n"

    monkeypatch.setattr(tso, "run_remote_trellis2_persistent", fake_persistent)
    monkeypatch.setattr(tso, "store_remote_trellis2_cache", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        tso,
        "scp_from_remote",
        lambda _args, remote_path, local_path: (
            Path(local_path).parent.mkdir(parents=True, exist_ok=True),
            Path(local_path).write_bytes(b"glb") if str(local_path).endswith(".glb") else Path(local_path).write_text(json.dumps({"ok": True, "mode": "proxy_glb_fallback"}), encoding="utf-8"),
        ),
    )
    persistent = tso._run_orchestration_one(persistent_args)
    assert persistent["remote_worker_mode"] == "persistent"
    assert persistent["remote_queue_job_json"] == "/queue/job.json"
    assert persistent["final_generation_mode"] == "procedural_proxy"

    direct_glb = tmp_path / "direct.glb"
    direct_glb.write_bytes(b"glb")
    prepared = tmp_path / "prepared_force"
    output = prepared / "output"
    output.mkdir(parents=True)
    (output / "asset.trellis.glb").write_bytes(b"glb")
    (output / "trellis.report.json").write_text(json.dumps({"ok": True, "mode": "direct_model"}), encoding="utf-8")
    (prepared / "card.raw.json").write_text(json.dumps({"unique_key": "prepared", "images": [str(image)]}), encoding="utf-8")
    (prepared / "card.normalized.json").write_text(json.dumps({"unique_key": "prepared"}), encoding="utf-8")
    (prepared / "image_manifest.json").write_text(json.dumps({"images": [str(image)], "trellis_input_dir": str(tmp_path)}), encoding="utf-8")
    (prepared / "summary.json").write_text(
        json.dumps({"ok": True, "asset_glb": str(output / "asset.trellis.glb"), "asset_generation_mode": "direct_model"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(tso, "try_reuse_remote_trellis2_asset", lambda **_kwargs: {"ok": True, "asset_glb": str(direct_glb)})
    forced = tso._run_orchestration_one(
        _args(tmp_path, prepared_job_dir=str(prepared), out_dir=str(tmp_path / "forced"), job_id="", force_trellis_image_only=True)
    )
    assert forced["ok"] is True

    monkeypatch.setattr(tso, "_trellis_fallback_candidate_sequence", lambda binding, **_kwargs: [binding])
    monkeypatch.setattr(tso, "_run_orchestration_one", lambda _args: {"prepare_only": True})
    with pytest.raises(RuntimeError, match="prepare-only result"):
        tso.run_orchestration(_args(tmp_path, card_json=str(card_path), out_dir=str(tmp_path / "bad_prepare")))

    bad_asset = tmp_path / "missing_result"
    monkeypatch.setattr(tso, "_run_orchestration_one", lambda _args: {"ok": False, "asset_glb": str(bad_asset)})
    with pytest.raises(RuntimeError, match="without a local GLB"):
        tso.run_orchestration(_args(tmp_path, card_json=str(card_path), out_dir=str(tmp_path / "bad_asset")))

    monkeypatch.setattr(tso, "_run_orchestration_one", lambda _args: {"ok": True})
    assert tso.run_orchestration(_args(tmp_path, card_json=str(card_path), out_dir=str(tmp_path / "prepare_only"), prepare_only=True)) == {"ok": True}


def test_trellis_remaining_candidate_catalog_blacklist_and_outer_fallback_edges(tmp_path, monkeypatch, capsys):
    original_candidate_unique_key = tso.candidate_unique_key
    monkeypatch.setattr(
        tso,
        "candidate_unique_key",
        lambda _cand: (_ for _ in ()).throw(RuntimeError("key failure")),
    )
    assert tso._trellis_candidate_key_safe({"product_url": "https://product"}) == "https://product"
    assert tso._trellis_candidate_key_safe({"model_page_url": "https://model"}) == "https://model"
    assert tso._trellis_candidate_key_safe({"title": "Only title"}) == "Only title"
    monkeypatch.setattr(tso, "candidate_unique_key", original_candidate_unique_key)

    assert tso._trellis_candidate_matches_group({"title": "Queen bed"}, "bed") is True
    assert tso._trellis_candidate_matches_group({"title": "Double bed"}, "bed") is True
    assert tso._trellis_size_score({"width": 1, "depth": 2, "height": 3}, ["bad", 2, 3]) == 0.0

    bad_catalog = tmp_path / "bad_catalog.json"
    bad_catalog.write_text("{bad", encoding="utf-8")
    monkeypatch.setattr(tso, "_trellis_catalog_paths", lambda: [bad_catalog])
    monkeypatch.setattr(tso, "_TRELLIS_CATALOG_CACHE", None)
    loaded = tso._trellis_load_catalog_cards()
    assert loaded["cards"] == []
    assert "catalog-warning" in capsys.readouterr().out

    monkeypatch.setattr(
        tso,
        "_TRELLIS_CATALOG_CACHE",
        {
            "path": "unit",
            "cards": [
                "bad",
                {"unique_key": "", "title": "No key chair", "category_norm": "chair"},
                {"unique_key": "seen", "title": "Seen chair", "category_norm": "chair", "images": ["seen.jpg"]},
                {"unique_key": "plain", "title": "Plain chair", "category_norm": "chair"},
                {"unique_key": "img", "title": "Image chair", "category_norm": "chair", "images": ["chair.jpg"], "source_site": "retailer"},
            ],
        },
    )
    pool = [{"unique_key": "base", "title": "Dining chair", "category_norm": "chair"}]
    appended = tso._trellis_append_catalog_alternatives(
        binding={"target_size_m": [0.5, 0.6, 0.9]},
        target_id="unknown_target",
        pool=pool,
        seen={"base", "seen"},
        max_catalog_candidates=3,
        require_images=False,
    )
    appended_keys = [candidate.get("unique_key") for candidate in appended]
    assert appended_keys[:2] == ["base", "img"]
    assert "plain" in appended_keys

    blacklist = tmp_path / "blacklist.json"
    blacklist.write_text(
        json.dumps(
            {
                "schema": "trellis_candidate_blacklist/v1",
                "targets": {"target": {"bad_unique_keys": {"skip": {"failures": 3, "blocked": True}}}},
            }
        ),
        encoding="utf-8",
    )
    selected = tso._trellis_fallback_candidate_sequence(
        binding={
            "target_id": "target",
            "candidate_pool": [
                "bad",
                {"unique_key": "skip", "title": "Skip chair", "category_norm": "chair", "images": ["skip.jpg"]},
                {"unique_key": "keep", "title": "Keep chair", "category_norm": "chair", "images": ["keep.jpg"]},
            ],
        },
        target_id="target",
        blacklist_path=blacklist,
        max_failures_per_candidate=2,
        max_candidate_pool=1,
    )
    assert [candidate["unique_key"] for candidate in selected] == ["keep"]
    assert "candidate-skip" in capsys.readouterr().out

    card = {
        "target_id": "target",
        "title": "Target chair",
        "category_norm": "chair",
        "candidate_pool": [{"unique_key": "fail", "title": "Fail chair", "category_norm": "chair"}],
    }
    card_path = tmp_path / "outer_card.json"
    card_path.write_text(json.dumps(card), encoding="utf-8")
    proxy_glb = tmp_path / "proxy.glb"
    proxy_glb.write_bytes(b"glb")
    attempts = []

    monkeypatch.setattr(tso, "_trellis_fallback_candidate_sequence", lambda **_kwargs: [{"unique_key": "fail", "title": "Fail chair"}])
    monkeypatch.setattr(
        tso,
        "_run_orchestration_one",
        lambda args: attempts.append(args.job_id) or (_ for _ in ()).throw(RuntimeError("remote failed")),
    )
    monkeypatch.setattr(tso, "_cleanup_failed_job_dir", lambda _path: None)
    monkeypatch.setattr(tso, "_run_local_proxy_fallback", lambda **_kwargs: {"ok": True, "asset_glb": str(proxy_glb)})

    summary = tso.run_orchestration(
        _args(
            tmp_path,
            card_json=str(card_path),
            out_dir=str(tmp_path / "outer"),
            job_id="outer_job",
            seed="bad-seed",
            allow_proxy_fallback=True,
        )
    )
    assert summary["asset_glb"] == str(proxy_glb)
    assert attempts == ["outer_job__cand001__try01"]
