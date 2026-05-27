from __future__ import annotations

import json
import subprocess
import zipfile
from pathlib import Path
from types import SimpleNamespace

from src.suppliers import acquire_site_assets as assets
from src.suppliers.models import ProductRecord
from src.suppliers.site_models import DownloadResult


def make_record(**overrides) -> ProductRecord:
    data = {
        "unique_key": "site::001",
        "source_site": "homeconcept",
        "source_url": "https://example.test/item",
        "parsed_at": "2026-01-01T00:00:00Z",
        "external_id": "Model 001",
        "title": "Modern Model 001 Chair",
        "product_url": "https://example.test/item",
        "category_raw": "chair",
        "brand": "Brand",
        "collection": "Line",
        "model_download_url": "https://example.test/model.glb",
        "model_download_filename": "model.glb",
        "model_format": "glb",
        "price_value": 100.0,
        "price_currency": "RUB",
        "width_cm": 50.0,
        "depth_cm": 60.0,
        "height_cm": 80.0,
        "style": "modern",
        "color": "gray",
        "materials": "wood",
        "description": "A modern chair",
        "images_json": json.dumps(["https://example.test/preview.jpg"]),
        "extra_json": json.dumps({"existing": True}),
    }
    data.update(overrides)
    return ProductRecord(**data)


def make_download(path: Path, *, ok: bool = True, final_url: str = "https://example.test/file") -> DownloadResult:
    return DownloadResult(
        final_url=final_url,
        local_path=str(path) if ok else None,
        filename=path.name,
        content_type="application/octet-stream",
        ok=ok,
        size_bytes=path.stat().st_size if path.exists() else 0,
        error=None if ok else "offline",
    )


def test_tokens_url_resolution_preview_and_archive_selection(tmp_path, monkeypatch):
    assert assets.slugify("  one / two  ") == "one_two"
    assert assets._normalize_model_name_token("Model № 12!") == "model_12"
    record = make_record(external_id="Chair 7788", title="Comfy Chair")
    assert assets._candidate_name_tokens(record)[:2] == ["chair_7788", "comfy_chair"]

    sancos = make_record(
        source_site="sancos",
        model_download_url="https://sancos.su/upload/3D/https://sancos.su/upload/3D/model.rar",
    )
    url, notes = assets.ensure_direct_model_url(sancos)
    assert url == "https://sancos.su/upload/3D/model.rar"
    assert notes == ["normalized_duplicate_sancos_3d_url"]

    loft = make_record(
        source_site="loftdesigne",
        model_download_url=None,
        model_download_landing_url="https://disk.yandex.test/public",
        model_download_filename=None,
        model_format=None,
    )
    monkeypatch.setattr(assets, "resolve_yadisk_public_download", lambda _url: ("https://download.test/model.fbx", "model.fbx"))
    url, notes = assets.ensure_direct_model_url(loft)
    assert url == "https://download.test/model.fbx"
    assert loft.model_format == ".fbx"
    assert notes == ["resolved_yadisk_public_download"]

    preview = tmp_path / "preview.jpg"
    preview.write_bytes(b"jpg")
    calls = []

    def fake_download(url, out_dir, filename_hint=None):
        calls.append((url, out_dir, filename_hint))
        return make_download(preview, ok="bad" not in url)

    monkeypatch.setattr(assets, "download_binary", fake_download)
    preview_path, preview_notes = assets.download_preview_image(["https://bad.test/img", "https://ok.test/img"], tmp_path)
    assert preview_path == str(preview)
    assert preview_notes == ["preview_download_failed:1", "preview_downloaded"]
    assert calls[0][2] == "preview_1.jpg"

    extracted = tmp_path / "extracted"
    (extracted / "nested").mkdir(parents=True)
    (extracted / "nested" / "Chair_7788.fbx").write_bytes(b"fbx")
    (extracted / "fallback.glb").write_bytes(b"glb")
    selected, ext, pick_notes = assets._pick_model_from_extracted_dir(extracted, record=record)
    assert selected.endswith("Chair_7788.fbx")
    assert ext == ".fbx"
    assert any(note.startswith("archive_selected_by_name:") for note in pick_notes)

    archive = tmp_path / "models.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("inner/model.glb", b"glb")
    selected, ext, notes = assets.inspect_archive(archive, tmp_path / "archive_out")
    assert selected.endswith("model.glb")
    assert ext == ".glb"
    assert "archive_extracted_with:zipfile" in notes

    bad_zip = tmp_path / "bad.zip"
    bad_zip.write_bytes(b"not a zip")
    ok, notes = assets._extract_archive_once(bad_zip, tmp_path / "bad_out")
    assert ok is False
    assert any(note.startswith("zip_extract_failed:") for note in notes)


def test_run_blender_helper_and_asset_record_helpers(tmp_path, monkeypatch):
    record = make_record()
    job_path = assets.build_blender_job_spec(record, tmp_path, "reason", "/tmp/source.obj", "/tmp/preview.jpg")
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    assert job["unique_key"] == record.unique_key
    assert job["dimensions_cm"]["width"] == 50.0

    monkeypatch.setattr(assets, "BLENDER_HELPER", tmp_path / "missing_helper.py")
    missing_ok, missing_notes = assets.run_blender_helper(
        blender_bin="/no/blender",
        mode="proxy",
        input_path=None,
        out_glb=tmp_path / "out.glb",
        out_fbx=tmp_path / "out.fbx",
    )
    assert missing_ok is False
    assert missing_notes[0].startswith("blender_helper_missing:")

    helper = tmp_path / "helper.py"
    helper.write_text("# helper\n", encoding="utf-8")
    monkeypatch.setattr(assets, "BLENDER_HELPER", helper)
    commands = []

    def fake_run(cmd, check=False, capture_output=True, text=True):
        commands.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(assets.subprocess, "run", fake_run)
    ok, notes = assets.run_blender_helper(
        blender_bin="/Applications/Blender.app/Contents/MacOS/Blender",
        mode="convert",
        input_path=str(tmp_path / "model.obj"),
        out_glb=tmp_path / "out.glb",
        out_fbx=tmp_path / "out.fbx",
        preview_path=str(tmp_path / "preview.jpg"),
        width_m=0.5,
        depth_m=0.6,
        height_m=0.8,
    )
    assert ok is True
    assert notes == ["blender_convert_ok"]
    assert "--mode" in commands[0] and "--width-m" in commands[0]

    asset = assets.build_asset_record(
        record=record,
        status="downloaded_preferred",
        asset_format="glb",
        asset_source_url="https://example.test/model.glb",
        asset_local_path="/tmp/model.glb",
        preview_local_path="/tmp/preview.jpg",
        blender_job_path=None,
        notes=["n"],
        extra={"x": 1},
    )
    assert assets.asset_is_ready(asset)
    assert json.loads(asset.notes_json) == ["n"]
    assert json.loads(asset.extra_json)["x"] == 1


def test_acquire_asset_for_record_branches(tmp_path, monkeypatch):
    db_path = tmp_path / "assets.sqlite"
    monkeypatch.setattr(assets, "insert_download", lambda **_kwargs: None)

    def fake_save_metadata(record, item_dir):
        path = item_dir / "metadata.json"
        path.write_text(json.dumps(record.to_dict(), ensure_ascii=False), encoding="utf-8")
        return path

    monkeypatch.setattr(assets, "save_metadata_json", fake_save_metadata)
    monkeypatch.setattr(assets, "download_preview_image", lambda _urls, item_dir: (str(item_dir / "preview.jpg"), ["preview_mocked"]))

    glb = tmp_path / "downloaded.glb"
    glb.write_bytes(b"glb")
    monkeypatch.setattr(assets, "download_binary_for_record", lambda *_args, **_kwargs: make_download(glb))
    direct = assets.acquire_asset_for_record(make_record(), db_path=db_path, out_dir=tmp_path / "out1", blender_bin=None)
    assert direct.asset_status == "downloaded_preferred"
    assert direct.asset_format == "glb"

    obj = tmp_path / "downloaded.obj"
    obj.write_bytes(b"obj")
    monkeypatch.setattr(assets, "download_binary_for_record", lambda *_args, **_kwargs: make_download(obj))

    def fake_blender(**kwargs):
        kwargs["out_glb"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["out_glb"].write_bytes(b"glb")
        return True, ["blender_convert_ok"]

    monkeypatch.setattr(assets, "run_blender_helper", fake_blender)
    converted = assets.acquire_asset_for_record(make_record(model_download_url="https://example.test/model.obj"), db_path=db_path, out_dir=tmp_path / "out2", blender_bin="/blender")
    assert converted.asset_status == "converted_with_blender"
    assert converted.asset_format == "glb"

    failed = tmp_path / "failed.rar"
    failed.write_bytes(b"rar")
    monkeypatch.setattr(assets, "download_binary_for_record", lambda *_args, **_kwargs: make_download(failed, ok=False))
    needs_rebuild = assets.acquire_asset_for_record(make_record(), db_path=db_path, out_dir=tmp_path / "out3", blender_bin=None)
    assert needs_rebuild.asset_status == "needs_blender_rebuild"
    assert needs_rebuild.blender_job_path and Path(needs_rebuild.blender_job_path).is_file()
    assert "download_failed:offline" in json.loads(needs_rebuild.notes_json)

    no_model = make_record(model_download_url=None, model_download_filename=None, model_format=None)
    proxy = assets.acquire_asset_for_record(no_model, db_path=db_path, out_dir=tmp_path / "out4", blender_bin="/blender")
    assert proxy.asset_status == "proxy_generated_with_blender"

    no_dims = make_record(model_download_url=None, model_download_filename=None, model_format=None, width_cm=None)
    queued = assets.acquire_asset_for_record(no_dims, db_path=db_path, out_dir=tmp_path / "out5", blender_bin="/blender")
    assert queued.asset_status == "needs_blender_rebuild"
    assert json.loads(queued.extra_json)["model_download_url"] is None


def test_extract_archive_subprocess_and_main_happy_path(tmp_path, monkeypatch, capsys):
    rar = tmp_path / "model.rar"
    rar.write_bytes(b"rar")
    calls = []

    def fake_run(cmd, check=False, capture_output=True, text=True):
        calls.append(cmd)
        if cmd[0] == "unar":
            return subprocess.CompletedProcess(cmd, 1, stderr="bad")
        return subprocess.CompletedProcess(cmd, 0, stderr="")

    monkeypatch.setattr(assets.subprocess, "run", fake_run)
    ok, notes = assets._extract_archive_once(rar, tmp_path / "rar_out")
    assert ok is True
    assert "archive_extract_failed:unar:1" in notes
    assert "archive_extracted_with:7z" in notes

    record = make_record()
    assets.log_product_card(record)
    assert "[card]" in capsys.readouterr().out

    adapter = SimpleNamespace(fetch_html=lambda url: ("<html>", url), parse=lambda url, html, final_url: [record])
    monkeypatch.setattr(assets, "init_db", lambda _path: None)
    monkeypatch.setattr(assets, "build_adapter_map", lambda: {"homeconcept": adapter})
    monkeypatch.setattr(assets, "build_site_fallback_map", lambda _adapter, _plan: {})
    monkeypatch.setattr(assets, "compute_discovery_limit", lambda count: count)
    monkeypatch.setattr(assets, "discover_product_urls", lambda **_kwargs: ["https://example.test/item"])
    monkeypatch.setattr(assets, "prioritize_product_urls", lambda _plan, urls: urls)
    monkeypatch.setattr(assets, "coerce_product_record", lambda item, _adapter, _source_url, _final_url: item)
    monkeypatch.setattr(assets, "apply_fallback_to_product", lambda _record, _fallback: None)
    monkeypatch.setattr(assets, "upsert_product", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(assets, "upsert_asset", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        assets,
        "acquire_asset_for_record",
        lambda *_args, **_kwargs: assets.build_asset_record(
            record=record,
            status="downloaded_preferred",
            asset_format="glb",
            asset_source_url="u",
            asset_local_path="/tmp/model.glb",
            preview_local_path=None,
            blender_job_path=None,
            notes=[],
            extra={},
        ),
    )
    monkeypatch.setattr(
        assets,
        "SITE_BATCH_PLANS",
        {"homeconcept": SimpleNamespace(categories=[])}
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "acquire_site_assets.py",
            "--sites",
            "homeconcept,missing",
            "--count-per-site",
            "1",
            "--db",
            str(tmp_path / "db.sqlite"),
            "--out-dir",
            str(tmp_path / "assets"),
        ],
    )
    assets.main()
    out = capsys.readouterr().out
    assert "[homeconcept] ready assets: 1/1" in out
    assert "[missing] skipped: no plan or adapter" in out


def test_acquire_site_assets_remaining_archive_blender_and_main_edges(tmp_path, monkeypatch, capsys):
    archive = tmp_path / "model.7z"
    archive.write_bytes(b"archive")
    calls = []

    def missing_then_error(cmd, check=False, capture_output=True, text=True):
        calls.append(cmd[0])
        if cmd[0] == "unar":
            raise FileNotFoundError(cmd[0])
        if cmd[0] == "7z":
            raise RuntimeError("runner failed")
        return subprocess.CompletedProcess(cmd, 1, stderr="")

    monkeypatch.setattr(assets.subprocess, "run", missing_then_error)
    ok, notes = assets._extract_archive_once(archive, tmp_path / "unsupported_out")
    assert ok is False
    assert "archive_extract_runner_failed:7z:RuntimeError:runner failed" in notes
    assert "archive_extract_unsupported:.7z" in notes

    unsupported = tmp_path / "model.tar"
    unsupported.write_bytes(b"tar")
    ok, notes = assets._extract_archive_once(unsupported, tmp_path / "tar_out")
    assert ok is False
    assert notes[-1] == "archive_extract_unsupported:.tar"

    inner = tmp_path / "inner.zip"
    with zipfile.ZipFile(inner, "w") as zf:
        zf.writestr("nested/model.obj", b"obj")
    outer = tmp_path / "outer.zip"
    with zipfile.ZipFile(outer, "w") as zf:
        zf.write(inner, "inner.zip")
    selected, ext, notes = assets.inspect_archive(outer, tmp_path / "outer_out")
    assert selected and selected.endswith("model.obj")
    assert ext == ".obj"
    assert any(note.startswith("nested_archive:inner.zip") for note in notes)

    helper = tmp_path / "helper.py"
    helper.write_text("# helper\n", encoding="utf-8")
    monkeypatch.setattr(assets, "BLENDER_HELPER", helper)

    def raise_blender(cmd, check=False, capture_output=True, text=True):
        raise RuntimeError("blender down")

    monkeypatch.setattr(assets.subprocess, "run", raise_blender)
    ok, notes = assets.run_blender_helper(
        blender_bin="/blender",
        mode="convert",
        input_path=str(tmp_path / "source.blend"),
        out_glb=tmp_path / "out.glb",
        out_fbx=tmp_path / "out.fbx",
    )
    assert ok is False
    assert notes[0].startswith("blender_run_failed:RuntimeError")

    def fail_blender(cmd, check=False, capture_output=True, text=True):
        return subprocess.CompletedProcess(cmd, 3, stderr="bad blender")

    monkeypatch.setattr(assets.subprocess, "run", fail_blender)
    ok, notes = assets.run_blender_helper(
        blender_bin="/blender",
        mode="convert",
        input_path=str(tmp_path / "source.blend"),
        out_glb=tmp_path / "out.glb",
        out_fbx=tmp_path / "out.fbx",
    )
    assert ok is False
    assert "blender_returncode:3" in notes
    assert any(note.startswith("blender_stderr:bad blender") for note in notes)

    db_path = tmp_path / "assets.sqlite"
    monkeypatch.setattr(assets, "insert_download", lambda **_kwargs: None)
    monkeypatch.setattr(assets, "download_preview_image", lambda _urls, item_dir: (None, []))
    monkeypatch.setattr(assets, "save_metadata_json", lambda record, item_dir: item_dir / "metadata.json")

    rar = tmp_path / "asset.rar"
    rar.write_bytes(b"rar")
    fbx = tmp_path / "preferred.fbx"
    fbx.write_bytes(b"fbx")
    monkeypatch.setattr(assets, "download_binary_for_record", lambda *_args, **_kwargs: make_download(rar))
    monkeypatch.setattr(assets, "inspect_archive", lambda *_args, **_kwargs: (str(fbx), ".fbx", ["archive_selected:.fbx"]))
    archived = assets.acquire_asset_for_record(make_record(model_download_url="https://example.test/model.rar"), db_path, tmp_path / "archive_asset", None)
    assert archived.asset_status == "archive_extracted_preferred"

    max_file = tmp_path / "asset.max"
    max_file.write_bytes(b"max")
    monkeypatch.setattr(assets, "download_binary_for_record", lambda *_args, **_kwargs: make_download(max_file))
    queued = assets.acquire_asset_for_record(make_record(model_download_url="https://example.test/model.max", extra_json="[]"), db_path, tmp_path / "queued_asset", None)
    assert queued.asset_status == "needs_blender_rebuild"
    assert queued.asset_format == "max"

    ready_asset = assets.build_asset_record(
        record=make_record(),
        status="downloaded_preferred",
        asset_format="glb",
        asset_source_url="u",
        asset_local_path="/tmp/model.glb",
        preview_local_path=None,
        blender_job_path=None,
        notes=[],
        extra={},
    )

    class EdgeAdapter:
        empty_parse_is_skip = True

        def fetch_html(self, url):
            if "error" in url:
                raise RuntimeError("offline")
            return "<html>", url

        def parse(self, url, html, final_url):
            if "skip" in url:
                return []
            return [make_record(product_url=final_url)]

    adapter = EdgeAdapter()
    monkeypatch.setattr(assets, "init_db", lambda _path: None)
    monkeypatch.setattr(assets, "build_adapter_map", lambda: {"homeconcept": adapter})
    monkeypatch.setattr(assets, "build_site_fallback_map", lambda _adapter, _plan: {})
    monkeypatch.setattr(assets, "compute_discovery_limit", lambda count: count + 3)
    monkeypatch.setattr(assets, "discover_product_urls", lambda **_kwargs: ["https://example.test/skip", "https://example.test/error", "https://example.test/ready", "https://example.test/unused"])
    monkeypatch.setattr(assets, "prioritize_product_urls", lambda _plan, urls: urls)
    monkeypatch.setattr(assets, "coerce_product_record", lambda item, _adapter, _source_url, _final_url: item)
    monkeypatch.setattr(assets, "apply_fallback_to_product", lambda _record, _fallback: None)
    monkeypatch.setattr(assets, "upsert_product", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(assets, "upsert_asset", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(assets, "log_product_card", lambda _record: None)
    monkeypatch.setattr(assets, "acquire_asset_for_record", lambda *_args, **_kwargs: ready_asset)
    monkeypatch.setattr(assets, "SITE_BATCH_PLANS", {"homeconcept": SimpleNamespace(categories=[])})
    monkeypatch.setattr(
        "sys.argv",
        [
            "acquire_site_assets.py",
            "--sites",
            "homeconcept",
            "--count-per-site",
            "1",
            "--db",
            str(tmp_path / "db.sqlite"),
            "--out-dir",
            str(tmp_path / "assets"),
        ],
    )
    assets.main()
    out = capsys.readouterr().out
    assert "asset skipped" in out
    assert "asset error" in out
    assert "ready assets: 1/1" in out
