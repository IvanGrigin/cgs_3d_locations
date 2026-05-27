from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import zipfile
from pathlib import Path

from src.suppliers import acquire_site_assets as assets
from src.suppliers import batch_runner as batch
from src.suppliers.models import ProductRecord
from src.suppliers.site_models import DownloadResult


class FakeAdapter:
    site_name = "fake"
    timeout = 3
    empty_parse_is_skip = False

    def __init__(self, pages=None, records=None, fail=False):
        self.pages = pages or {}
        self.records = records or []
        self.fail = fail

    def fetch_html(self, url):
        if self.fail:
            raise RuntimeError("offline")
        return self.pages.get(url, self.pages.get("default", "<html></html>")), url

    def parse(self, url, html, final_url):
        return list(self.records)

    def now_utc_iso(self):
        return "2026-01-01T00:00:00Z"


class FakeAPIResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def product(**overrides) -> ProductRecord:
    rec = ProductRecord(
        unique_key="site::id::1",
        source_site="homeconcept",
        source_url="https://supplier.test/item",
        parsed_at="2026-01-01T00:00:00Z",
        title="Test Chair",
        product_url="https://supplier.test/item",
        category_norm="chair",
        images_json=json.dumps(["https://supplier.test/image.jpg"]),
        width_cm=50,
        depth_cm=60,
        height_cm=90,
        extra_json="{}",
    )
    for key, value in overrides.items():
        setattr(rec, key, value)
    return rec


def test_batch_runner_url_extractors_and_site_specific_discovery(monkeypatch):
    plan = batch.SiteBatchPlan(
        site_name="test",
        root_url="https://example.test/",
        seed_urls=("https://example.test/catalog/",),
        product_path_markers=("/product/",),
        category_path_markers=("/catalog/",),
        deny_path_markers=("/blog/",),
    )

    assert batch.normalize_url("https://3ddd.ru/3dmodels?cat=1&bad=2&page=3#x") == "https://3ddd.ru/3dmodels?cat=1&page=3"
    assert batch.normalize_url("https://example.test/catalog/item/") == "https://example.test/catalog/item"
    assert batch.same_host("https://example.test/a", "https://example.test/root")
    assert batch.is_non_html_url("https://example.test/a/model.fbx")
    assert batch.is_product_url("https://example.test/product/1", plan)
    assert batch.is_category_url("https://example.test/catalog/chairs", plan)
    assert not batch.is_category_url("https://example.test/blog/post", plan)

    html = """
    <a href="/product/1">p1</a><a href="/catalog/chairs">cat</a>
    <a href="https://other.test/product/2">other</a><a href="/image.jpg">image</a>
    <a href="javascript:void(0)">bad</a>
    """
    links = batch.extract_links(html, "https://example.test/catalog/", "https://example.test/")
    assert links == {"https://example.test/product/1", "https://example.test/catalog/chairs"}
    assert batch.extract_sitemap_locs("<url><loc>https://example.test/product/1</loc></url>") == ["https://example.test/product/1"]

    home_plan = batch.SITE_BATCH_PLANS["homeconcept"]
    home_html = """
    <div class="items-list-3d-models">
      <div class="item"><div class="item-link-image"><a href="/catalog/product/chair/"></a></div><div class="item-name">Chair</div><div class="item-price"><a href="/upload/chair.rar"></a></div></div>
      <div class="item"><div class="item-link-image"><a href="https://other.test/catalog/product/bad/"></a></div></div>
    </div>
    """
    assert batch.extract_homeconcept_library_product_urls(home_html, "https://homeconcept.ru/3d-models/", home_plan) == [
        "https://homeconcept.ru/catalog/product/chair"
    ]
    fallback = batch.extract_homeconcept_library_fallback_map(home_html, "https://homeconcept.ru/3d-models/", home_plan)
    assert fallback["https://homeconcept.ru/catalog/product/chair"]["model_download_url"] == "https://homeconcept.ru/upload/chair.rar"
    assert batch.discover_product_urls_homeconcept_library(FakeAdapter({home_plan.seed_urls[0].rstrip("/"): home_html}), home_plan, 3) == [
        "https://homeconcept.ru/catalog/product/chair"
    ]
    assert batch.build_site_fallback_map(FakeAdapter({home_plan.seed_urls[0].rstrip("/"): home_html}), home_plan)

    timo_plan = batch.SITE_BATCH_PLANS["timotrader"]
    timo_html = """
    <div id="products"><div class="tm-product-item"><a class="tm-media-box" href="/katalog/item-1"></a></div></div>
    <ul class="uk-pagination"><li><a href="?page=2">2</a></li></ul>
    """
    assert batch.extract_timotrader_listing_product_urls(timo_html, timo_plan.seed_urls[0], timo_plan) == ["https://timotrader.ru/katalog/item-1"]
    assert batch.extract_timotrader_listing_page_numbers(timo_html) == {1, 2}
    assert batch.timotrader_listing_page_url("https://timotrader.ru/3d-modeli", 2).endswith("?page=2")
    timo_adapter = FakeAdapter(
        {
            timo_plan.seed_urls[0]: timo_html,
            batch.timotrader_listing_page_url(timo_plan.seed_urls[0], 2): timo_html.replace("item-1", "item-2").replace("?page=2", "?page=1"),
        }
    )
    assert len(batch.discover_product_urls_timotrader_3d_listing(timo_adapter, timo_plan, 3, 2)) == 2

    cersanit_plan = batch.SITE_BATCH_PLANS["cersanit"]
    cersanit_html = """
    <a class="catalog-list-item__info" href="/catalog/mito/3d-be/item-1/"></a>
    <a href="/catalog/mito/3d-be/?PAGEN_1=2&x=bad">next</a>
    """
    assert batch.extract_cersanit_collection_product_urls(cersanit_html, cersanit_plan.seed_urls[0], cersanit_plan) == [
        "https://cersanit.ru/catalog/mito/3d-be/item-1"
    ]
    assert batch.extract_cersanit_collection_page_urls(cersanit_html, cersanit_plan.seed_urls[0], cersanit_plan) == [
        "https://cersanit.ru/catalog/mito/3d-be?PAGEN_1=2"
    ]

    payloads = []
    monkeypatch.setattr(
        batch.requests,
        "post",
        lambda url, headers, json, timeout: payloads.append(json)
        or FakeAPIResponse({"data": {"models": [{"slug": "chair-1"}, {"slug": ""}], "per_page": 1, "total_value": 1}}),
    )
    three_plan = batch.SITE_BATCH_PLANS["3ddd"]
    assert batch.build_3ddd_listing_payload("https://3ddd.ru/3dmodels?subcat=12&types=free&page=bad")["page"] == 1
    assert batch.discover_product_urls_3ddd_api(FakeAdapter(), three_plan, 2, 1) == ["https://3ddd.ru/3dmodels/show/chair-1"]
    assert payloads


def test_batch_runner_generic_discovery_fallback_and_processing(monkeypatch, tmp_path):
    plan = batch.SiteBatchPlan(
        site_name="test",
        root_url="https://example.test/",
        seed_urls=("https://example.test/catalog/",),
        product_path_markers=("/product/",),
        category_path_markers=("/catalog/",),
    )
    pages = {
        "https://example.test/catalog": '<a href="/product/1">p1</a><a href="/catalog/page2">next</a>',
        "https://example.test/catalog/page2": '<a href="/product/2">p2</a>',
    }
    discovered = batch.discover_product_urls(FakeAdapter(pages), plan, limit=3, max_listing_pages=3, max_depth=1)
    assert discovered == ["https://example.test/product/1", "https://example.test/product/2"]
    assert batch.compute_discovery_limit(3) == 13
    assert batch.prioritize_product_urls(batch.SITE_BATCH_PLANS["imodern"], ["/product/zapchast-a", "/product/stul-a", "/product/divan-a"])[0] == "/product/stul-a"

    db_path = tmp_path / "products.db"
    with sqlite3.connect(db_path) as con:
        con.execute("CREATE TABLE supplier_product(source_site TEXT, product_url TEXT)")
        con.execute("INSERT INTO supplier_product VALUES('test', 'https://example.test/product/1?x=1#frag')")
    assert batch.load_existing_product_urls(db_path, "test") == {"https://example.test/product/1"}
    assert json.loads(batch.merge_extra_json("{bad", {"ok": True})) == {"ok": True}

    rec = product(title=None, model_download_url=None)
    batch.apply_fallback_to_product(rec, {"title": "Fallback title", "model_download_url": "https://example.test/model.fbx"})
    assert rec.title == "Fallback title"
    assert rec.model_download_filename == "model.fbx"
    assert rec.model_format == ".fbx"
    assert json.loads(rec.extra_json)["library_fallback_used"] is True

    saved_products = []
    fetch_logs = []
    metadata = []
    monkeypatch.setattr(batch, "upsert_products", lambda db_path, products: saved_products.extend(products))
    monkeypatch.setattr(batch, "insert_fetch_log", lambda **kwargs: fetch_logs.append(kwargs))
    monkeypatch.setattr(batch, "save_metadata_json", lambda product, out_dir: metadata.append(product.unique_key) or (out_dir / "meta.json"))
    adapter = FakeAdapter({"https://example.test/product/1": "<html>ok</html>"}, [product()])
    assert batch.process_single_product(adapter, "test", "https://example.test/product/1", tmp_path / "db.sqlite", tmp_path) == (1, 0)
    assert saved_products and fetch_logs[-1]["ok"] is True and metadata

    empty_adapter = FakeAdapter({"https://example.test/empty": "<html></html>"}, [])
    empty_adapter.empty_parse_is_skip = True
    assert batch.process_single_product(empty_adapter, "test", "https://example.test/empty", tmp_path / "db.sqlite", tmp_path) == (0, 0)

    failing_adapter = FakeAdapter(fail=True)
    assert batch.process_single_product(failing_adapter, "test", "https://example.test/fail", tmp_path / "db.sqlite", tmp_path) == (0, 1)
    assert fetch_logs[-1]["ok"] is False

    calls = []
    monkeypatch.setattr(batch, "process_single_product", lambda adapter, site_name, url, db_path, out_dir, fallback_map=None: calls.append(url) or (1, 0))
    assert batch.process_product_urls_parallel(adapter, "test", ["u1", "u2"], tmp_path / "db", tmp_path, workers=2, limit_success=1) == (1, 0)
    assert calls == ["u1"]


def test_batch_runner_remaining_discovery_parallel_and_main_edges(monkeypatch, tmp_path):
    assert batch.extract_sitemap_locs('<url><loc data-x="1">https://example.test/product/1</loc></url>') == [
        "https://example.test/product/1"
    ]

    home_plan = batch.SITE_BATCH_PLANS["homeconcept"]
    home_edge_html = """
    <div class="items-list-3d-models">
      <div class="item"><div class="item-link-image"></div></div>
      <div class="item"><div class="item-link-image"><a href="/not-product/1"></a></div></div>
      <div class="item"><div class="item-link-image"><a href="/catalog/product/chair/"></a></div></div>
      <div class="item"><div class="item-link-image"><a href="/catalog/product/chair/"></a></div></div>
    </div>
    """
    assert batch.extract_homeconcept_library_product_urls(home_edge_html, "https://homeconcept.ru/3d-models/", home_plan) == [
        "https://homeconcept.ru/catalog/product/chair"
    ]
    assert batch.build_site_fallback_map(FakeAdapter(fail=True), home_plan) == {}
    assert batch.discover_product_urls_homeconcept_library(FakeAdapter(fail=True), home_plan, 3) == []
    assert batch.build_site_fallback_map(FakeAdapter(), batch.SITE_BATCH_PLANS["imodern"]) == {}

    timo_plan = batch.SITE_BATCH_PLANS["timotrader"]
    timo_edge_html = """
    <div id="products">
      <div class="tm-product-item"></div>
      <div class="tm-product-item"><a class="tm-media-box" href="https://other.test/katalog/x"></a></div>
      <div class="tm-product-item"><a class="tm-media-box" href="/contacts/"></a></div>
      <div class="tm-product-item"><a class="tm-product-card-body" href="/katalog/item-1"></a></div>
      <div class="tm-product-item"><a class="tm-media-box" href="/katalog/item-1"></a></div>
    </div>
    """
    assert batch.extract_timotrader_listing_product_urls(timo_edge_html, timo_plan.seed_urls[0], timo_plan) == [
        "https://timotrader.ru/katalog/item-1"
    ]
    assert batch.discover_product_urls_timotrader_3d_listing(FakeAdapter(fail=True), timo_plan, 3, 1) == []

    cersanit_plan = batch.SITE_BATCH_PLANS["cersanit"]
    cersanit_edge_html = """
    <a class="catalog-list-item__info" href=""></a>
    <a class="catalog-list-item__info" href="https://other.test/catalog/mito/3d-be/item"></a>
    <a class="catalog-list-item__info" href="/catalog/other/item"></a>
    <a class="catalog-list-item__pic-area" href="/catalog/mito/3d-be/item-1/"></a>
    <a class="catalog-list-item__info" href="/catalog/mito/3d-be/item-1/"></a>
    <a href=""></a>
    <a href="https://other.test/catalog/mito/3d-be/?PAGEN_1=2"></a>
    <a href="/catalog/other/?PAGEN_1=2"></a>
    <a href="/catalog/mito/3d-be/?PAGEN_1=2"></a>
    <a href="/catalog/mito/3d-be/?PAGEN_1=2"></a>
    """
    assert batch.extract_cersanit_collection_product_urls(cersanit_edge_html, cersanit_plan.seed_urls[0], cersanit_plan) == [
        "https://cersanit.ru/catalog/mito/3d-be/item-1"
    ]
    assert batch.extract_cersanit_collection_page_urls(cersanit_edge_html, cersanit_plan.seed_urls[0], cersanit_plan) == [
        "https://cersanit.ru/catalog/mito/3d-be?PAGEN_1=2"
    ]
    assert batch.discover_product_urls_cersanit_collection(FakeAdapter(fail=True), cersanit_plan, 3, 1) == []

    sitemap_plan = batch.SiteBatchPlan(
        site_name="sitemap-test",
        root_url="https://example.test/",
        seed_urls=("https://example.test/sitemap.xml",),
        product_path_markers=("/product/",),
        category_path_markers=("/catalog/",),
        discovery_mode="sitemap",
    )
    sitemap_pages = {
        "https://example.test/sitemap.xml": """
        <loc>https://other.test/product/out</loc>
        <loc>https://example.test/child.xml</loc>
        <loc>https://example.test/catalog/category</loc>
        """,
        "https://example.test/child.xml": """
        <loc>https://example.test/product/a</loc>
        <loc>https://example.test/product/a</loc>
        <loc>https://example.test/product/b</loc>
        """,
    }
    assert batch.discover_product_urls_from_sitemap(FakeAdapter(sitemap_pages), sitemap_plan, 5, 3) == [
        "https://example.test/product/a",
        "https://example.test/product/b",
    ]

    generic_plan = batch.SiteBatchPlan(
        site_name="generic-test",
        root_url="https://example.test/",
        seed_urls=("https://example.test/catalog/",),
        product_path_markers=("/product/",),
        category_path_markers=("/catalog/",),
    )
    generic_pages = {
        "https://example.test/catalog": '<a href="/catalog/deep">deep</a>',
        "https://example.test/catalog/deep": '<a href="/catalog/too-deep">too deep</a><a href="/product/1">p1</a><a href="/product/2">p2</a>',
    }
    limited = batch.discover_product_urls(FakeAdapter(generic_pages), generic_plan, limit=1, max_listing_pages=3, max_depth=1)
    assert len(limited) == 1
    assert limited[0].startswith("https://example.test/product/")
    assert batch.discover_product_urls(FakeAdapter(fail=True), generic_plan, limit=2, max_listing_pages=1, max_depth=1) == []

    three_plan = batch.SiteBatchPlan(
        site_name="3ddd",
        root_url="https://3ddd.ru/",
        seed_urls=("https://3ddd.ru/3dmodels?query=chair&order=popular&page=1",),
        product_path_markers=("/3dmodels/show/",),
        category_path_markers=("/3dmodels",),
        discovery_mode="3ddd_api",
    )
    assert batch.build_3ddd_listing_payload(three_plan.seed_urls[0])["query"] == "chair"

    api_payloads = [
        {"data": {"models": [{"slug": "chair"}, {"slug": "chair"}, "bad", {"slug": ""}, {"slug": "table"}], "per_page": "bad"}},
        {"data": {"models": []}},
    ]

    def fake_post(url, headers, json, timeout):
        return FakeAPIResponse(api_payloads.pop(0))

    monkeypatch.setattr(batch.requests, "post", fake_post)
    assert batch.discover_product_urls_3ddd_api(FakeAdapter(), three_plan, 3, 2) == [
        "https://3ddd.ru/3dmodels/show/chair",
        "https://3ddd.ru/3dmodels/show/table",
    ]
    monkeypatch.setattr(batch.requests, "post", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("api down")))
    assert batch.discover_product_urls_3ddd_api(FakeAdapter(), three_plan, 3, 1) == []

    calls = []

    def fake_process(adapter, site_name, url, db_path, out_dir, fallback_map=None):
        calls.append(url)
        return (0, 1) if url == "bad" else (1, 0)

    monkeypatch.setattr(batch, "process_single_product", fake_process)
    assert batch.process_product_urls_parallel(FakeAdapter(), "test", ["u1", "bad", "u2"], tmp_path / "db", tmp_path, workers=1) == (2, 1)
    assert calls == ["u1", "bad", "u2"]

    monkeypatch.setattr(batch, "build_adapters", lambda: [type("Adapter", (), {"site_name": "one"})()])
    assert set(batch.build_adapter_map()) == {"one"}

    main_plan = batch.SiteBatchPlan(
        site_name="test",
        root_url="https://example.test/",
        seed_urls=("https://example.test/catalog/",),
        product_path_markers=("/product/",),
        category_path_markers=("/catalog/",),
    )
    main_adapter = FakeAdapter()
    monkeypatch.setattr(batch, "SITE_BATCH_PLANS", {"test": main_plan})
    monkeypatch.setattr(batch, "build_adapter_map", lambda: {"test": main_adapter})
    monkeypatch.setattr(batch, "init_db", lambda db_path: None)
    monkeypatch.setattr(batch, "build_site_fallback_map", lambda adapter, plan: {})

    monkeypatch.setattr(batch, "discover_product_urls", lambda **kwargs: [])
    monkeypatch.setattr(sys, "argv", ["batch_runner", "--sites", "test,missing", "--db", str(tmp_path / "db1.sqlite"), "--out-dir", str(tmp_path)])
    batch.main()

    monkeypatch.setattr(batch, "discover_product_urls", lambda **kwargs: ["https://example.test/product/1"])
    monkeypatch.setattr(batch, "load_existing_product_urls", lambda db_path, site_name: {"https://example.test/product/1"})
    monkeypatch.setattr(
        sys,
        "argv",
        ["batch_runner", "--sites", "test", "--skip-existing-products", "--db", str(tmp_path / "db2.sqlite"), "--out-dir", str(tmp_path)],
    )
    batch.main()

    monkeypatch.setattr(batch, "build_site_fallback_map", lambda adapter, plan: (_ for _ in ()).throw(RuntimeError("fatal")))
    monkeypatch.setattr(sys, "argv", ["batch_runner", "--sites", "test", "--db", str(tmp_path / "db3.sqlite"), "--out-dir", str(tmp_path)])
    batch.main()


def test_acquire_site_assets_archive_helpers_and_blender_jobs(monkeypatch, tmp_path):
    assert assets.slugify(" Test / chair ? ") == "Test_chair"
    assert assets._normalize_model_name_token("Chair Model 01!") == "chair_model_01"
    rec = product(external_id="AB 123", title="Chair AB 123")
    assert assets._candidate_name_tokens(rec)[:2] == ["ab_123", "chair_ab_123"]

    direct = product(source_site="sancos", model_download_url="https://sancos.su/upload/3D/https://sancos.su/upload/3D/model.fbx")
    model_url, notes = assets.ensure_direct_model_url(direct)
    assert model_url == "https://sancos.su/upload/3D/model.fbx"
    assert notes == ["normalized_duplicate_sancos_3d_url"]

    loft = product(source_site="loftdesigne", model_download_url=None, model_download_landing_url="https://disk.yandex.ru/d/file")
    monkeypatch.setattr(assets, "resolve_yadisk_public_download", lambda url: ("https://downloader/model.zip", "model.zip"))
    assert assets.ensure_direct_model_url(loft)[0] == "https://downloader/model.zip"

    monkeypatch.setattr(
        assets,
        "download_binary",
        lambda url, target_dir, filename_hint=None: DownloadResult(url, str(target_dir / filename_hint), filename_hint, "image/jpeg", True, 10),
    )
    preview, preview_notes = assets.download_preview_image(["https://example.test/a.jpg"], tmp_path)
    assert preview.endswith("preview_1.jpg")
    assert preview_notes == ["preview_downloaded"]

    archive = tmp_path / "models.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("chair_ab_123.obj", "obj")
        zf.writestr("nested.zip", b"not a real zip")
    selected, ext, archive_notes = assets.inspect_archive(archive, tmp_path / "extract", record=rec)
    assert selected.endswith("chair_ab_123.obj")
    assert ext == ".obj"
    assert "archive_extracted_with:zipfile" in archive_notes
    assert any(note.startswith("archive_selected_by_name") for note in archive_notes)

    helper_path = tmp_path / "helper.py"
    helper_path.write_text("print('ok')", encoding="utf-8")
    monkeypatch.setattr(assets, "BLENDER_HELPER", helper_path)

    def fake_run(cmd, check, capture_output, text):
        assert "--mode" in cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(assets.subprocess, "run", fake_run)
    ok, blender_notes = assets.run_blender_helper(
        blender_bin="/bin/echo",
        mode="convert",
        input_path=str(tmp_path / "input.obj"),
        out_glb=tmp_path / "out.glb",
        out_fbx=tmp_path / "out.fbx",
        preview_path=preview,
        width_m=0.5,
        depth_m=0.6,
        height_m=0.9,
    )
    assert ok
    assert blender_notes == ["blender_convert_ok"]

    job_path = assets.build_blender_job_spec(rec, tmp_path, "reason", selected, preview)
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    assert job["unique_key"] == rec.unique_key
    assert job["reason"] == "reason"

    asset = assets.build_asset_record(rec, "downloaded_preferred", "glb", "https://x/model.glb", "/tmp/model.glb", preview, None, ["ok"], {"x": 1})
    assert assets.asset_is_ready(asset)
    assert json.loads(asset.notes_json) == ["ok"]


def test_acquire_asset_for_record_uses_mocks_for_download_and_proxy(monkeypatch, tmp_path):
    calls = {"downloads": [], "assets": []}
    monkeypatch.setattr(assets, "save_metadata_json", lambda record, item_dir: item_dir / "metadata.json")
    monkeypatch.setattr(assets, "download_preview_image", lambda preview_urls, item_dir: (str(item_dir / "preview.jpg"), ["preview_downloaded"]))
    monkeypatch.setattr(assets, "insert_download", lambda **kwargs: calls["downloads"].append(kwargs))
    monkeypatch.setattr(assets, "run_blender_helper", lambda **kwargs: (False, ["blender_disabled"]))

    glb_path = tmp_path / "downloaded.glb"
    glb_path.write_text("glb", encoding="utf-8")
    monkeypatch.setattr(
        assets,
        "download_binary_for_record",
        lambda record, target_dir, filename_hint=None: DownloadResult("https://example.test/model.glb", str(glb_path), "model.glb", "model/gltf-binary", True, 3),
    )
    rec = product(model_download_url="https://example.test/model.glb", model_download_filename="model.glb", model_format=".glb")
    acquired = assets.acquire_asset_for_record(rec, tmp_path / "db.sqlite", tmp_path / "assets", blender_bin=None)
    assert acquired.asset_status == "downloaded_preferred"
    assert acquired.asset_format == "glb"
    assert calls["downloads"]

    no_model = product(unique_key="site::id::2", model_download_url=None, images_json="[]")
    proxy = assets.acquire_asset_for_record(no_model, tmp_path / "db.sqlite", tmp_path / "assets", blender_bin=None)
    assert proxy.asset_status == "needs_blender_rebuild"
    assert proxy.blender_job_path and Path(proxy.blender_job_path).is_file()
