from __future__ import annotations

from pathlib import Path
import sqlite3
from types import SimpleNamespace

from src.suppliers import batch_runner as br
from src.suppliers.models import ProductRecord


def _record(**overrides) -> ProductRecord:
    data = {
        "unique_key": "site::1",
        "source_site": "unit",
        "source_url": "https://example.com/item",
        "parsed_at": "2026-01-01T00:00:00Z",
        "external_id": "1",
        "title": "Title",
        "product_url": "https://example.com/item",
        "extra_json": "{}",
    }
    data.update(overrides)
    return ProductRecord(**data)


class FakeAdapter:
    site_name = "unit"
    timeout = 1
    empty_parse_is_skip = False

    def __init__(self, pages=None, parsed=None, fail=False):
        self.pages = dict(pages or {})
        self.parsed = list([_record()] if parsed is None else parsed)
        self.fail = fail
        self.fetches = []

    def fetch_html(self, url):
        self.fetches.append(url)
        if self.fail:
            raise RuntimeError("fetch failed")
        return self.pages.get(url, self.pages.get("*", "")), url

    def parse(self, url, html, final_url):
        return list(self.parsed)

    def now_utc_iso(self):
        return "2026-01-01T00:00:00Z"


def test_url_helpers_and_listing_extractors() -> None:
    plan = br.SiteBatchPlan(
        site_name="unit",
        root_url="https://example.com/",
        seed_urls=("https://example.com/catalog/?bad=1",),
        product_path_markers=("/product/",),
        category_path_markers=("/catalog/",),
        deny_path_markers=("/blog/",),
    )
    assert br.normalize_url("https://3ddd.ru/3dmodels?cat=x&bad=1&page=2#frag") == "https://3ddd.ru/3dmodels?cat=x&page=2"
    assert br.normalize_url("https://cersanit.ru/catalog/?PAGEN_1=2&bad=1") == "https://cersanit.ru/catalog?PAGEN_1=2"
    assert br.same_host("https://example.com/a", plan.root_url)
    assert br.is_non_html_url("https://example.com/model.fbx")
    assert br.is_product_url("https://example.com/product/1", plan)
    assert br.is_category_url("https://example.com/catalog/chairs", plan)
    assert not br.is_category_url("https://example.com/blog/post", plan)

    html = """
    <a href="/catalog/chairs/">cat</a>
    <a href="/product/one">one</a>
    <a href="https://other.com/product/two">two</a>
    <a href="/file.zip">zip</a>
    <a href="mailto:test">mail</a>
    """
    assert br.extract_links(html, "https://example.com/catalog/", plan.root_url) == {
        "https://example.com/catalog/chairs",
        "https://example.com/product/one",
    }
    sitemap = "<url><loc>https://example.com/product/one</loc></url><loc>https://example.com/product/one</loc>"
    assert br.extract_sitemap_locs(sitemap) == ["https://example.com/product/one"]


def test_site_specific_discovery_helpers(monkeypatch) -> None:
    home = br.SITE_BATCH_PLANS["homeconcept"]
    home_html = """
    <div class="items-list-3d-models">
      <div class="item"><div class="item-link-image"><a href="/catalog/product/a/"></a></div>
      <div class="item-name">Chair</div><div class="item-price"><a href="/download/a.rar"></a></div></div>
    </div>
    """
    assert br.extract_homeconcept_library_product_urls(home_html, home.root_url, home) == ["https://homeconcept.ru/catalog/product/a"]
    fallback = br.extract_homeconcept_library_fallback_map(home_html, home.root_url, home)
    assert fallback["https://homeconcept.ru/catalog/product/a"]["title"] == "Chair"
    assert br.discover_product_urls_homeconcept_library(FakeAdapter({"*": home_html}), home, 10) == ["https://homeconcept.ru/catalog/product/a"]
    assert br.build_site_fallback_map(FakeAdapter({"*": home_html}), home)["https://homeconcept.ru/catalog/product/a"]["model_download_url"].endswith(".rar")
    assert br.build_site_fallback_map(FakeAdapter(fail=True), home) == {}

    timo = br.SITE_BATCH_PLANS["timotrader"]
    timo_html = """
    <div id="products"><div class="tm-product-item"><a class="tm-media-box" href="/katalog/stul"></a></div></div>
    <ul class="uk-pagination"><a href="?page=2">2</a><a href="?page=x">x</a></ul>
    """
    assert br.extract_timotrader_listing_product_urls(timo_html, timo.root_url, timo)[0].endswith("/katalog/stul")
    assert br.extract_timotrader_listing_page_numbers(timo_html) == {1, 2}
    assert br.timotrader_listing_page_url("https://timotrader.ru/3d-modeli", 2).endswith("?page=2")
    adapter = FakeAdapter({timo.seed_urls[0]: timo_html, br.timotrader_listing_page_url(timo.seed_urls[0], 2): ""})
    assert br.discover_product_urls_timotrader_3d_listing(adapter, timo, 5, 2) == ["https://timotrader.ru/katalog/stul"]

    cersanit = br.SITE_BATCH_PLANS["cersanit"]
    cersanit_html = '<a class="catalog-list-item__info" href="/catalog/mito/3d-be/item/"></a><a href="/catalog/mito/3d-be/?PAGEN_1=2">2</a>'
    assert br.extract_cersanit_collection_product_urls(cersanit_html, cersanit.root_url, cersanit)[0].endswith("/item")
    assert br.extract_cersanit_collection_page_urls(cersanit_html, cersanit.root_url, cersanit)[0].endswith("PAGEN_1=2")
    assert br.discover_product_urls_cersanit_collection(FakeAdapter({br.normalize_url(cersanit.seed_urls[0]): cersanit_html}), cersanit, 5, 1)[0].endswith("/item")

    api_payload = {"data": {"models": [{"slug": "chair"}, {}, {"slug": "chair"}], "per_page": 1, "total_value": 1}}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return api_payload

    monkeypatch.setattr(br.requests, "post", lambda *args, **kwargs: Response())
    assert br.build_3ddd_listing_payload("https://3ddd.ru/3dmodels?subcat=1&types=free&page=bad")["page"] == 1
    assert br.discover_product_urls_3ddd_api(SimpleNamespace(timeout=1), br.SITE_BATCH_PLANS["3ddd"], 2, 2) == ["https://3ddd.ru/3dmodels/show/chair"]


def test_generic_discovery_prioritization_existing_and_fallback_processing(tmp_path: Path, monkeypatch) -> None:
    plan = br.SiteBatchPlan(
        site_name="unit",
        root_url="https://example.com/",
        seed_urls=("https://example.com/catalog",),
        product_path_markers=("/product/",),
        category_path_markers=("/catalog",),
    )
    pages = {
        "https://example.com/catalog": '<a href="/product/a"></a><a href="/catalog/more"></a>',
        "https://example.com/catalog/more": '<a href="/product/b"></a>',
    }
    assert br.discover_product_urls(FakeAdapter(pages), plan, 5, 5, 2) == ["https://example.com/product/a", "https://example.com/product/b"]
    assert br.compute_discovery_limit(3) == 13
    assert br.prioritize_product_urls(
        br.SITE_BATCH_PLANS["imodern"],
        ["https://imodern.ru/product/zapchast-x", "https://imodern.ru/product/stul-a", "https://imodern.ru/product/divan-a"],
    )[0].endswith("stul-a")

    db = tmp_path / "existing.db"
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE supplier_product (source_site text, product_url text)")
        con.execute("INSERT INTO supplier_product VALUES (?, ?)", ("unit", "https://example.com/product/a"))
    assert br.load_existing_product_urls(db, "unit") == {"https://example.com/product/a"}
    assert br.load_existing_product_urls(tmp_path / "missing.db", "unit") == set()

    product = _record(title="", model_download_url="")
    br.apply_fallback_to_product(product, {"title": "Fallback", "model_download_url": "https://cdn/model.fbx"})
    assert product.title == "Fallback"
    assert product.model_format == ".fbx"
    assert "library_fallback" in product.model_extraction_method
    assert br.merge_extra_json("{bad", {"x": 1}) == '{"x": 1}'
    assert br.adapter_filename_from_url("https://cdn/model.fbx") == "model.fbx"
    assert br.adapter_ext_from_url("https://cdn/model.fbx") == ".fbx"

    saved = []
    logs = []
    monkeypatch.setattr(br, "upsert_products", lambda db_path, products: saved.extend(products))
    monkeypatch.setattr(br, "insert_fetch_log", lambda **kwargs: logs.append(kwargs))
    monkeypatch.setattr(br, "save_metadata_json", lambda product, out_dir: out_dir / f"{product.external_id}.json")
    count, failed = br.process_single_product(FakeAdapter(parsed=[_record(title="")]), "unit", "https://example.com/product/a", tmp_path / "db.sqlite", tmp_path, {br.normalize_url("https://example.com/product/a"): {"title": "From fallback"}})
    assert (count, failed) == (1, 0)
    assert saved[0].title == "From fallback"
    assert logs[-1]["ok"] is True

    empty_adapter = FakeAdapter(parsed=[])
    empty_adapter.empty_parse_is_skip = True
    assert br.process_single_product(empty_adapter, "unit", "https://example.com/product/empty", tmp_path / "db.sqlite", tmp_path) == (0, 0)

    fail_adapter = FakeAdapter(fail=True)
    assert br.process_single_product(fail_adapter, "unit", "https://example.com/product/fail", tmp_path / "db.sqlite", tmp_path) == (0, 1)
    assert logs[-1]["ok"] is False


def test_parallel_processing_and_main_smoke(tmp_path: Path, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(br, "process_single_product", lambda adapter, site_name, url, db_path, out_dir, fallback_map=None: calls.append(url) or (1, 0))
    saved, failed = br.process_product_urls_parallel(FakeAdapter(), "unit", ["u1", "u2", "u3"], tmp_path / "db", tmp_path, workers=2, limit_success=2)
    assert (saved, failed) == (2, 0)
    assert calls == ["u1", "u2"]

    calls.clear()
    saved, failed = br.process_product_urls_parallel(FakeAdapter(), "unit", ["u1", "u2"], tmp_path / "db", tmp_path, workers=2)
    assert (saved, failed) == (2, 0)
    assert sorted(calls) == ["u1", "u2"]

    adapter = FakeAdapter()
    monkeypatch.setattr(br, "init_db", lambda db_path: None)
    monkeypatch.setattr(br, "build_adapter_map", lambda: {"unit": adapter})
    monkeypatch.setitem(br.SITE_BATCH_PLANS, "unit", br.SiteBatchPlan("unit", "https://example.com/", ("https://example.com/catalog",), ("/product/",), ("/catalog/",)))
    monkeypatch.setattr(br, "discover_product_urls", lambda **kwargs: ["https://example.com/product/a"])
    monkeypatch.setattr(br, "process_product_urls_parallel", lambda **kwargs: (1, 0))
    monkeypatch.setattr("sys.argv", ["batch_runner.py", "--sites", "unit,missing", "--limit-per-site", "1", "--db", str(tmp_path / "db.sqlite"), "--out-dir", str(tmp_path)])
    br.main()


def test_sitemap_dispatch_empty_parse_and_main_skip_edges(tmp_path: Path, monkeypatch) -> None:
    plan = br.SiteBatchPlan(
        site_name="unit",
        root_url="https://example.com/",
        seed_urls=("https://example.com/sitemap.xml",),
        product_path_markers=("/product/",),
        category_path_markers=("/catalog/",),
        discovery_mode="sitemap",
    )
    assert not br.is_category_url("https://example.com/catalog/file.zip", plan)
    assert not br.is_category_url("https://example.com/product/a", plan)
    assert br.is_category_url("https://example.com/", plan)

    pages = {
        "https://example.com/sitemap.xml": """
          <loc>https://example.com/nested.xml</loc>
          <loc>https://other.com/product/x</loc>
          <loc>https://example.com/catalog/not-product</loc>
        """,
        "https://example.com/nested.xml": """
          <loc>https://example.com/product/a</loc>
          <loc>https://example.com/product/a</loc>
          <loc>https://example.com/product/b</loc>
        """,
    }
    assert br.discover_product_urls_from_sitemap(FakeAdapter(pages), plan, limit=2, max_listing_pages=4) == [
        "https://example.com/product/a",
        "https://example.com/product/b",
    ]
    assert br.discover_product_urls_from_sitemap(FakeAdapter(fail=True), plan, limit=2, max_listing_pages=1) == []

    dispatch_calls = []
    monkeypatch.setattr(br, "discover_product_urls_homeconcept_library", lambda **kwargs: dispatch_calls.append("home") or ["home"])
    monkeypatch.setattr(br, "discover_product_urls_timotrader_3d_listing", lambda **kwargs: dispatch_calls.append("timo") or ["timo"])
    monkeypatch.setattr(br, "discover_product_urls_cersanit_collection", lambda **kwargs: dispatch_calls.append("cersanit") or ["cersanit"])
    monkeypatch.setattr(br, "discover_product_urls_from_sitemap", lambda **kwargs: dispatch_calls.append("sitemap") or ["sitemap"])
    monkeypatch.setattr(br, "discover_product_urls_3ddd_api", lambda **kwargs: dispatch_calls.append("3ddd") or ["3ddd"])
    for mode in ["homeconcept_library", "timotrader_3d_listing", "cersanit_collection", "sitemap", "3ddd_api"]:
        br.discover_product_urls(
            FakeAdapter(),
            br.SiteBatchPlan("unit", "https://example.com/", ("https://example.com/",), ("/product/",), ("/catalog/",), discovery_mode=mode),
            limit=1,
            max_listing_pages=1,
            max_depth=1,
        )
    assert dispatch_calls == ["home", "timo", "cersanit", "sitemap", "3ddd"]

    logs = []
    monkeypatch.setattr(br, "insert_fetch_log", lambda **kwargs: logs.append(kwargs))
    empty_fail_adapter = FakeAdapter(parsed=[])
    empty_fail_adapter.empty_parse_is_skip = False
    assert br.process_single_product(empty_fail_adapter, "unit", "https://example.com/product/empty", tmp_path / "db.sqlite", tmp_path) == (0, 1)
    assert "zero records" in logs[-1]["error"]

    monkeypatch.setattr(br, "init_db", lambda db_path: None)
    adapter = FakeAdapter()
    monkeypatch.setattr(br, "build_adapter_map", lambda: {"unit": adapter, "bad": adapter})
    monkeypatch.setitem(br.SITE_BATCH_PLANS, "unit", br.SiteBatchPlan("unit", "https://example.com/", ("https://example.com/catalog",), ("/product/",), ("/catalog/",)))
    monkeypatch.setitem(br.SITE_BATCH_PLANS, "bad", br.SiteBatchPlan("bad", "https://bad.example/", ("https://bad.example/catalog",), ("/product/",), ("/catalog/",)))
    discovered_by_site = {
        "unit": ["https://example.com/product/a"],
        "bad": ["https://bad.example/product/a"],
    }
    monkeypatch.setattr(br, "discover_product_urls", lambda **kwargs: discovered_by_site[kwargs["plan"].site_name])
    monkeypatch.setattr(br, "build_site_fallback_map", lambda adapter, plan: {})
    monkeypatch.setattr(br, "load_existing_product_urls", lambda db_path, site_name: {"https://example.com/product/a"} if site_name == "unit" else set())
    monkeypatch.setattr(br, "process_product_urls_parallel", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("fatal")))
    monkeypatch.setattr(
        "sys.argv",
        [
            "batch_runner.py",
            "--sites",
            "unit,bad",
            "--skip-existing-products",
            "--limit-per-site",
            "1",
            "--db",
            str(tmp_path / "db.sqlite"),
            "--out-dir",
            str(tmp_path),
        ],
    )
    br.main()
