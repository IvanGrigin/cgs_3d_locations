# -*- coding: utf-8 -*-
import json
import subprocess
import sys

import pytest
import requests
from bs4 import BeautifulSoup

from src.suppliers.adapters.base import SupplierAdapter
from src.suppliers.adapters.cersanit import CersanitAdapter
from src.suppliers.adapters.homeconcept import HomeConceptAdapter
from src.suppliers.adapters.imodern import IModernAdapter
from src.suppliers.adapters.loftdesigne import LoftDesigneAdapter
from src.suppliers.adapters.sancos import SancosAdapter
from src.suppliers.adapters.three_ddd import ThreeDDDAdapter
from src.suppliers.adapters.timotrader import TimoTraderAdapter
from src.suppliers.adapters.zeelproject import ZeelProjectAdapter


class DummyAdapter(SupplierAdapter):
    site_name = "dummy"

    def can_handle(self, url: str) -> bool:
        return "example.test" in url

    def parse(self, url: str, html: str, final_url: str):
        return []


class FakeResponse:
    def __init__(self, body: bytes | str, url: str = "https://example.test/final", status_code: int = 200):
        self.body = body.encode("utf-8") if isinstance(body, str) else body
        self.url = url
        self.status_code = status_code
        self.encoding = "utf-8"
        self.headers = {"Content-Type": "text/html; charset=utf-8"}
        self.history = []
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    @property
    def text(self):
        return self.body.decode("utf-8", errors="replace")

    def iter_content(self, chunk_size=65536):
        yield self.body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status={self.status_code}")

    def close(self):
        self.closed = True


def test_base_adapter_fetch_fallback_and_common_extractors(monkeypatch):
    adapter = DummyAdapter(timeout=7)
    html = "<!doctype html><html><body><div class='page-catalog-product'>ok</div>" + ("x" * 2100) + "</body></html>"

    monkeypatch.setattr("src.suppliers.adapters.base.requests.get", lambda *a, **k: FakeResponse(html))
    fetched, final_url = adapter.fetch_html("https://example.test/item")
    assert "page-catalog-product" in fetched
    assert final_url == "https://example.test/final"

    def fake_run(command, capture_output, check):
        assert command[0] == "curl"
        stdout = (html + "\n__CODEX_EFFECTIVE_URL__:https://example.test/curl-final").encode("utf-8")
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr=b"")

    monkeypatch.setattr(adapter, "_fetch_html_via_requests", lambda url: (_ for _ in ()).throw(RuntimeError("requests failed")))
    monkeypatch.setattr("src.suppliers.adapters.base.subprocess.run", fake_run)
    fetched, final_url = adapter.fetch_html("https://example.test/item")
    assert "page-catalog-product" in fetched
    assert final_url == "https://example.test/curl-final"

    soup = BeautifulSoup(
        """
        <meta property="og:title" content=" Sofa  title ">
        <meta name="description" content=" Nice item ">
        <script type="application/ld+json">
        {"@type": ["Product"], "name": "JSON sofa", "brand": {"name": "Brand X"}}
        </script>
        <img src="/a.jpg"><img data-src="/b.jpg"><img src="/a.jpg">
        <a onclick="window.location.href='/download/model.fbx'">download</a>
        """,
        "html.parser",
    )
    assert adapter.extract_meta_content(soup, "og:title") == "Sofa title"
    assert adapter.extract_meta_content(soup, "description") == "Nice item"
    assert adapter.extract_name_from_jsonld(soup) == "JSON sofa"
    assert adapter.extract_brand_from_jsonld(soup) == "Brand X"
    assert adapter.extract_images(soup, "https://example.test/base/") == [
        "https://example.test/a.jpg",
        "https://example.test/b.jpg",
    ]
    assert adapter.extract_onclick_download(soup.find("a"), "https://example.test/base/").endswith("/download/model.fbx")
    assert adapter.parse_price_rub("old 12 000 ₽ new 9 500 ₽") == (9500.0, 12000.0, "RUB")
    assert adapter.parse_dimension_cm("Ширина 12,5 см", "Ширина") == 12.5
    assert adapter.parse_weight_kg("Вес в упаковке 3,2 кг") == 3.2
    assert adapter.ext_from_url("https://x.test/file?filename=model.rar") == ".rar"
    assert adapter.filename_from_url("https://x.test/file?filename=model%20one.rar") == "model one.rar"
    assert adapter.classify_category("подвесной светильник") == "lamp"
    assert adapter.build_unique_key("https://x.test/item", "42") == "dummy::id::42"


def test_homeconcept_product_and_library_parsing(monkeypatch):
    adapter = HomeConceptAdapter()
    product_html = """
    <html><head>
      <meta property="og:description" content="Meta description">
      <script type="application/ld+json">
      {"@type": "Product", "brand": "JsonBrand",
       "offers": {"price": "12345", "priceCurrency": "RUB"},
       "additionalProperty": [
         {"name": "Вес в упаковке", "value": "8 кг"},
         {"name": "Диаметр", "value": "50 см"}
       ]}
      </script>
    </head><body>
      <div class="catalog-product__offer-name">Стул Lounge | Home Concept</div>
      <div class="catalog-product__name-brand">Home Brand</div>
      <span class="js-item-info-current-check-offer-code">HC-77</span>
      <div class="item-info-current-check-offer-price">30 000 ₽ 25 000 ₽</div>
      <div class="item-info-current-check-offer-status-available">В наличии</div>
      <div class="catalog-product__material-text--active">Blue fabric</div>
      <table class="product-characteristics">
        <tr><td>Ширина</td><td>80 см</td></tr>
        <tr><td>Глубина</td><td>70 см</td></tr>
        <tr><td>Высота</td><td>90 см</td></tr>
        <tr><td>Материал</td><td>Ткань</td></tr>
        <tr><td>Стиль</td><td>modern</td></tr>
        <tr><td>Помещение</td><td>bedroom</td></tr>
      </table>
      <div class="bx_breadcrumbs"><a>Мебель</a><a>Стулья</a></div>
      <a class="catalog-product__3d-model__link-download" href="/models/lounge.rar">3d</a>
      <div class="catalog-product__image-slider"><img src="/img1.jpg"><img data-src="/img2.jpg"></div>
      <div class="catalog-product__tags"><a>tag1</a></div>
      <div class="collections-list"><div class="item"><a class="item-name" href="/rel">Related</a><div class="item-price">10 ₽</div><img src="/rel.jpg"></div></div>
    </body></html>
    """
    records = adapter.parse("https://homeconcept.ru/product/lounge/", product_html, "https://homeconcept.ru/product/lounge/")
    rec = records[0]
    assert rec.unique_key == "homeconcept::id::HC-77"
    assert rec.title == "Стул Lounge"
    assert rec.category_norm == "chair"
    assert rec.model_download_url == "https://homeconcept.ru/models/lounge.rar"
    assert rec.model_download_filename == "lounge.rar"
    assert rec.width_cm == 80.0
    assert rec.depth_cm == 70.0
    assert rec.height_cm == 90.0
    assert rec.weight_kg == 8.0
    assert json.loads(rec.tags_json) == ["tag1"]
    assert json.loads(rec.related_json)[0]["url"] == "https://homeconcept.ru/rel"

    library_html = """
    <div class="items-list-3d-models">
      <div class="item">
        <div class="item-link-image"><a href="/product/lounge/"><img src="/thumb.jpg"></a></div>
        <div class="item-name">Стул Lounge</div>
        <div class="item-price"><a href="/upload/lounge.rar">download</a></div>
      </div>
      <div class="item"><div class="item-name"></div></div>
    </div>
    """
    monkeypatch.setattr(adapter, "fetch_html", lambda url: (product_html, "https://homeconcept.ru/product/lounge/"))
    enriched = adapter.parse("https://homeconcept.ru/3d-models/", library_html, "https://homeconcept.ru/3d-models/")
    assert len(enriched) == 1
    assert json.loads(enriched[0].extra_json)["library_page_url"] == "https://homeconcept.ru/3d-models/"

    monkeypatch.setattr(adapter, "fetch_html", lambda url: (_ for _ in ()).throw(RuntimeError("offline")))
    fallback = adapter.parse("https://homeconcept.ru/3d-models/", library_html, "https://homeconcept.ru/3d-models/")
    assert json.loads(fallback[0].extra_json)["enriched_from_product_page"] is False
    assert "product_fetch_error" in json.loads(fallback[0].extra_json)


def test_three_ddd_json_and_html_parsing(monkeypatch):
    adapter = ThreeDDDAdapter()
    payload = {
        "data": {
            "slug": "chair-123",
            "title": "Chair Model",
            "category": {"title": "Furniture"},
            "subcategory": {"title": "Chairs"},
            "user": {"username": "artist"},
            "typeText": "om",
            "style": "modern",
            "description": "Source https://vendor.test/item\nСтрана производства: Italy\nВысота сиденья: 45 см",
            "length": "80",
            "width": "70,5",
            "height": "90",
            "size_kb": 2048,
            "platform": {"title": "3ds Max"},
            "render": {"title": "Corona"},
            "form": {"form": "round"},
            "formats": [{"title": "FBX"}, {"title": "OBJ"}],
            "images": [{"webPath": "chair/a.jpg"}, {"webPath": "chair/a.jpg"}],
            "materials": [{"material": "wood"}, {"material": "fabric"}],
            "colors": [{"title": "blue"}],
            "created": "2025-05-12 10:11:12",
        }
    }
    rec = adapter.parse("https://3ddd.ru/3dmodels/show/chair-123", json.dumps(payload), "https://3ddd.ru/3dmodels/show/chair-123")[0]
    assert rec.external_id == "chair-123"
    assert rec.availability == "FREE"
    assert rec.width_cm == 80.0
    assert rec.depth_cm == 70.5
    assert rec.height_cm == 90.0
    assert rec.materials == "wood, fabric"
    assert rec.color == "blue"
    extra = json.loads(rec.extra_json)
    assert extra["archive_size_mb"] == 2.0
    assert extra["published_date"] == "2025-05-12"
    assert extra["archive_formats"] == ["fbx", "obj"]

    class ApiResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return payload

    monkeypatch.setattr("src.suppliers.adapters.three_ddd.requests.post", lambda *a, **k: ApiResponse())
    html, final_url = adapter.fetch_html("https://3ddd.ru/3dmodels/show/chair-123")
    assert json.loads(html)["data"]["slug"] == "chair-123"
    assert final_url.endswith("chair-123")

    html_page = """
    <span class="plate-title" title="Кресло Lounge"></span>
    <div class="category"><span>Мебель</span></div><div class="subcategory"><span>Кресла</span></div>
    <div class="model-user-name">artist</div><div class="status"><span>PRO</span></div>
    <div class="royalty-free"><span>Royalty Free</span></div>
    <div class="model-info-block"><table>
      <tr><td>Платформа:</td><td>3ds Max</td></tr><tr><td>Рендер:</td><td>VRay</td></tr>
      <tr><td>Размер:</td><td>12,5 МБ</td></tr><tr><td>Стиль:</td><td>classic</td></tr>
      <tr><td>Материалы:</td><td>leather</td></tr><tr><td>Длина:</td><td>100 см</td></tr>
      <tr><td>Ширина:</td><td>60 см</td></tr><tr><td>Высота:</td><td>80 см</td></tr>
    </table></div>
    <div class="description"><div><a href="https://vendor.test/source">src</a>
      Какие форматы в архиве: fbx, obj
      Высота посадки: 44 см
    </div></div>
    <div class="big-view"><picture><source srcset="/big.jpg"></picture></div>
    <div class="preview"><img src="/thumb.jpg"></div>
    <div class="publication-date">12 мая 2025</div>
    """
    html_rec = adapter.parse("https://3ddd.ru/3dmodels/show/chair-html", html_page, "https://3ddd.ru/3dmodels/show/chair-html")[0]
    assert html_rec.title == "Кресло Lounge"
    assert html_rec.category_norm == "armchair"
    assert html_rec.width_cm == 100.0
    assert html_rec.depth_cm == 60.0
    assert html_rec.height_cm == 80.0
    assert json.loads(html_rec.extra_json)["royalty_free"] is True


def test_three_ddd_remaining_helper_and_fallback_branches(monkeypatch):
    adapter = ThreeDDDAdapter()
    assert adapter.can_handle("https://3ddd.ru/3dmodels/show/chair")
    assert not adapter.can_handle("https://example.test/3dmodels/show/chair")

    class BadApiResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": []}

    monkeypatch.setattr("src.suppliers.adapters.three_ddd.requests.post", lambda *a, **k: BadApiResponse())
    with pytest.raises(RuntimeError, match="unexpected payload"):
        adapter._fetch_product_json("https://3ddd.ru/3dmodels/show/chair", "chair")

    monkeypatch.setattr(SupplierAdapter, "fetch_html", lambda self, url: ("html", url + "?fallback"))
    assert adapter.fetch_html("https://3ddd.ru/3dmodels/show/chair") == ("html", "https://3ddd.ru/3dmodels/show/chair?fallback")
    assert adapter.fetch_html("https://3ddd.ru/no-slug") == ("html", "https://3ddd.ru/no-slug?fallback")

    assert adapter._load_json_payload("not json") is None
    assert adapter._load_json_payload("{bad") is None
    assert adapter._load_json_payload("[]") is None
    assert adapter._load_json_payload('{"data": []}') is None

    soup = BeautifulSoup(
        """
        <html><head><title>Title fallback</title></head><body>
          <h1 class="title">Heading fallback</h1>
          <div class="category"><span>Furniture</span></div>
          <div class="subcategory"><span>Chair</span></div>
          <div class="model-user-name">Author</div>
          <div class="status"><span>FREE</span></div>
          <div class="description"><div>Source href=&quot;https://vendor.test/x&quot;\nКакие форматы в архиве: FBX, OBJ\nВысота сиденья: 44 см</div></div>
          <div class="publication-date">33 неизвестно 2025</div>
          <div class="big-view"><picture><source srcset=""></picture></div>
          <div class="preview"><img src="/a.jpg"><img src="/a.jpg"></div>
        </body></html>
        """,
        "html.parser",
    )
    assert adapter._extract_title(soup) == "Heading fallback"
    assert adapter._extract_category(soup) == "Furniture > Chair"
    assert adapter._extract_author(soup) == "Author"
    assert adapter._extract_status(soup) == "FREE"
    assert adapter._extract_royalty_free(BeautifulSoup("<html></html>", "html.parser")) is False
    assert adapter._extract_info_table(BeautifulSoup("<table><tr><td>bad</td></tr></table>", "html.parser")) == {}
    html, text = adapter._extract_description(soup)
    assert "Высота сиденья" in text
    assert adapter._extract_source_link_from_description(BeautifulSoup("<html></html>", "html.parser"), html, "") == "https://vendor.test/x"
    assert adapter._extract_source_link_from_text("no url") is None
    assert adapter._extract_archive_formats("no formats") == []
    assert adapter._extract_images(soup, "https://3ddd.ru/model") == ["https://3ddd.ru/a.jpg"]
    assert adapter._extract_published_date(soup) is None
    assert adapter._extract_published_date(BeautifulSoup("<div class='publication-date'>12 мая 2025</div>", "html.parser")) == "2025-05-12"
    assert adapter._extract_slug_from_url("https://3ddd.ru/bad") is None

    data = {
        "category": "bad",
        "subcategory": {"title": "Sub"},
        "user": {"slug": "user-slug"},
        "formats": ["bad", {"title": "MAX"}],
        "images": ["bad", {"webPath": ""}, {"webPath": "img/a.jpg"}, {"webPath": "img/a.jpg"}],
        "materials": ["bad", {"material": "Wood"}],
        "colors": ["bad", {"title": "Red"}],
        "created": "2026-01-02 03:04:05",
        "size_kb": "bad",
    }
    assert adapter._build_category_raw_from_json(data) == "Sub"
    assert adapter._extract_author_from_json(data) == "user-slug"
    assert adapter._extract_author_from_json({}) is None
    assert adapter._map_status_from_json({"typeText": "om"}) == "FREE"
    assert adapter._map_status_from_json({"typeText": ""}) is None
    assert adapter._extract_archive_formats_from_json(data) == ["max"]
    assert adapter._extract_images_from_json(data) == [
        "https://b5.3ddd.ru/media/cache/tuk_model_custom_filter_ang_ru/img/a.jpg"
    ]
    assert adapter._extract_published_date_from_json(data) == "2026-01-02"
    assert adapter._extract_published_date_from_json({}) is None
    assert adapter._extract_materials_from_json(data) == "Wood"
    assert adapter._extract_materials_from_json({}) is None
    assert adapter._extract_color_from_json(data) == "Red"
    assert adapter._extract_color_from_json({}) is None
    assert adapter._get_nested_text({"nested": "bad"}, "nested", "title") is None
    assert adapter._size_kb_to_mb("bad") is None
    assert adapter._to_float(None) is None
    assert adapter._to_float("bad") is None
    assert adapter._parse_named_text("none", "Страна производства") is None


def test_loftdesigne_and_imodern_product_parsing():
    loft = LoftDesigneAdapter()
    loft_html = """
    <html><head><meta property="og:title" content="Desk - Купить"></head><body>
    <div class="catalog__breadcrumbs"><a>Главная</a><a>Столы</a></div>
    <div class="product-modal" data-product-card-full data-product-card
         data-product-category="Столы" data-product-name="Loft Desk" data-product-price-rubles="12300">
      <div class="product-modal__name">Стол Loft - Loft Designe</div>
      <div class="product-modal__id">LD-42</div>
      <div class="product-modal__price">15 000 ₽ 12 300 ₽</div>
      <div class="product-modal__status">В наличии</div>
      <div class="product-modal__description">Oak desk description</div>
      <div class="product-modal__main-slider"><img src="/desk1.jpg"></div>
      <div class="product-modal__details-item"><span class="product-modal__details-label">Ширина:</span><span class="product-modal__details-value">120 см</span></div>
      <div class="product-modal__details-item"><span class="product-modal__details-label">Глубина:</span><span class="product-modal__details-value">60 см</span></div>
      <div class="product-modal__details-item"><span class="product-modal__details-label">Высота:</span><span class="product-modal__details-value">75 см</span></div>
      <div class="product-modal__details-item"><span class="product-modal__details-label">Материал:</span><span class="product-modal__details-value">Oak</span></div>
      <div class="product-modal__details-item"><span class="product-modal__details-label">Цвет:</span><span class="product-modal__details-value">Brown</span></div>
      <div class="product-modal__details-item"><span class="product-modal__details-label">Коллекция:</span><span class="product-modal__details-value">Work</span></div>
      <a href="https://disk.yandex.ru/d/model">Скачать 3D-модель</a>
    </div></body></html>
    """
    rec = loft.parse("https://loftdesigne.ru/item", loft_html, "https://loftdesigne.ru/item")[0]
    assert rec.unique_key == "loftdesigne::id::LD-42"
    assert rec.model_link_type == "landing_page"
    assert rec.model_download_landing_url == "https://disk.yandex.ru/d/model"
    assert rec.description == "Oak desk description"
    assert rec.width_cm == 120.0
    assert rec.materials == "Oak"
    assert json.loads(rec.tags_json) == ["Столы", "Brown", "Oak"]
    assert loft.parse("https://loftdesigne.ru/no-model", "<h1>No model</h1>", "https://loftdesigne.ru/no-model") == []

    imodern = IModernAdapter()
    imodern_html = """
    <html><head>
      <meta property="product:price:amount" content="9876.5">
      <meta property="og:image" content="https://imodern.ru/og.jpg">
      <meta property="og:description" content="Meta desc">
    </head><body>
      <h1>Стул Jenny</h1>
      <div itemprop="description">Chair description</div>
      <p>Ширина: 50 см\nГлубина: 55 см\nВысота: 82 см\nВес 7 кг\nОбивка: шенилл</p>
      <button onclick="location.href='/download/jenny.zip'">Скачать 3D модель</button>
      <div class="slider-pro"><div class="sp-slides"><img src="/upload/jenny.jpg"><img src="/static/logo.png"></div></div>
      <div class="sim_prod" data-name="Related chair" data-price="100" data-product='{"url": "/rel", "name": "Related JSON"}'></div>
    </body></html>
    """
    rec = imodern.parse("https://imodern.ru/product/stul/", imodern_html, "https://imodern.ru/product/stul/#model")[0]
    assert rec.product_url == "https://imodern.ru/product/stul/"
    assert rec.price_value == 9876.5
    assert rec.model_download_url == "https://imodern.ru/download/jenny.zip"
    assert rec.weight_kg == 7.0
    assert "Обивка: шенилл" in rec.materials
    assert json.loads(rec.images_json) == ["https://imodern.ru/upload/jenny.jpg", "https://imodern.ru/og.jpg"]


def test_loftdesigne_adapter_remaining_fallback_branches():
    loft = LoftDesigneAdapter()
    assert loft.can_handle("https://shop.loftdesigne.ru/card")
    assert not loft.can_handle("https://example.test/card")

    direct_html = """
    <html><head>
      <meta property="og:title" content="Direct Chair - цена">
      <meta property="og:description" content="OG description">
      <script type="application/ld+json">{"@type": "Product", "brand": {"name": "Json Brand"}, "offers": {"lowPrice": "123,45", "priceCurrency": "RUB"}}</script>
    </head><body>
    <div class="product-modal" data-product-card-full data-product-card
         data-product-category="" data-product-name="bad-model model" data-product-price-rubles="bad">
      <div class="product-modal__price">bad price</div>
      <div class="product-modal__details-item"><span class="product-modal__details-label">Диаметр:</span><span class="product-modal__details-value">42,5 см</span></div>
      <div class="product-modal__details-item"><span class="product-modal__details-label">Высота:</span><span class="product-modal__details-value">70 см</span></div>
      <div class="product-modal__main-slider"><img src="/a.jpg"><img src="/a.jpg"><img data-src="/b.jpg"></div>
      <a href="/files/model.fbx">Скачать модель</a>
    </div></body></html>
    """
    rec = loft.parse("https://loftdesigne.ru/direct", direct_html, "https://loftdesigne.ru/direct")[0]
    assert rec.title == "Direct Chair"
    assert rec.brand == "Json Brand"
    assert rec.model_link_type == "direct_file"
    assert rec.model_download_url == "https://loftdesigne.ru/files/model.fbx"
    assert rec.model_download_filename == "model.fbx"
    assert rec.model_format == ".fbx"
    assert rec.price_value == 123.45
    assert rec.width_cm == 42.5
    assert rec.depth_cm == 42.5
    assert rec.height_cm == 70.0
    assert rec.description == "OG description"
    assert json.loads(rec.images_json) == ["https://loftdesigne.ru/a.jpg", "https://loftdesigne.ru/b.jpg"]

    soup = BeautifulSoup(
        """
        <html><head>
          <script type="application/ld+json">{"@type": "Product", "name": "Json Name", "description": "Json desc", "offers": {"price": "bad"}}</script>
        </head><body><h1>Heading Title - Loft Designe</h1></body></html>
        """,
        "html.parser",
    )
    assert loft.extract_product_title(None, soup) == "Json Name"
    assert loft.extract_price(None, soup)[0] is None
    assert loft.extract_description(
        product=None,
        soup=soup,
        title=None,
        category_raw=None,
        materials=None,
        color=None,
        width_cm=None,
        depth_cm=None,
        height_cm=None,
        availability=None,
    ) == ("Json desc", "jsonld")
    assert loft.extract_description(
        product=None,
        soup=BeautifulSoup("<html></html>", "html.parser"),
        title="Title",
        category_raw="Category",
        materials="Oak",
        color="Brown",
        width_cm=10.0,
        depth_cm=20.5,
        height_cm=30.0,
        availability="In stock",
    ) == ("Title. Категория: Category. Материал: Oak. Цвет: Brown. Размеры: 10 x 20.5 x 30 см. Наличие: In stock", "structured_fields_fallback")
    assert loft.extract_description(
        product=None,
        soup=BeautifulSoup("<html></html>", "html.parser"),
        title=None,
        category_raw=None,
        materials=None,
        color=None,
        width_cm=None,
        depth_cm=None,
        height_cm=12.25,
        availability=None,
    ) == ("Высота: 12.25 см", "structured_fields_fallback")
    assert loft.extract_description(
        product=None,
        soup=BeautifulSoup("<html></html>", "html.parser"),
        title=None,
        category_raw=None,
        materials=None,
        color=None,
        width_cm=None,
        depth_cm=None,
        height_cm=None,
        availability=None,
    ) == (None, "missing")
    assert loft.extract_availability(None) is None
    assert loft.extract_detail_value(None, "Материал") is None
    assert loft.parse_numeric_value("нет числа") is None
    assert loft.parse_numeric_value("12,75 см") == 12.75
    assert loft._normalize_label("Вес (в упаковке):") == "вес"
    assert loft._format_measure(10.0) == "10"
    assert loft._format_measure(10.50) == "10.5"


def test_bathroom_supplier_adapters_parse_products_and_helpers():
    cersanit = CersanitAdapter()
    cersanit_html = """
    <html><head><meta name="description" content="Cersanit desc"><meta property="og:image" content="/upload/bath.jpg"></head>
    <body>
      <div class="bx-breadcrumb"><a href="/catalog/mito/">MITO</a><a>Ванны</a></div>
      <h1>Каркас ванны MITO</h1>
      <div class="product-detail-info__text">Артикул: MITO-1</div>
      <div class="specs__element"><span class="specs__title">Тип продукта:</span><span class="specs__value">Каркас</span></div>
      <div class="specs__element"><span class="specs__title">Ширина, см:</span><span class="specs__value">170</span></div>
      <div class="specs__element"><span class="specs__title">Длина, см:</span><span class="specs__value">75</span></div>
      <div class="specs__element"><span class="specs__title">Высота, см:</span><span class="specs__value">55</span></div>
      <div class="specs__element"><span class="specs__title">Вес (в упаковке), кг:</span><span class="specs__value">20,5</span></div>
      <span>Описание</span><div class="description-section__text">Product text</div>
      <a href="/files/model.fbx">Скачать FBX</a><a href="/files/scheme.pdf">Схема</a>
      <img src="/upload/bath.jpg" alt="Каркас ванны MITO">
    </body></html>
    """
    rec = cersanit.parse("https://cersanit.ru/catalog/mito/3d-be/item/", cersanit_html, "https://cersanit.ru/catalog/mito/3d-be/item/")[0]
    assert rec.model_download_url.endswith("/files/model.fbx")
    assert rec.scheme_url.endswith("/files/scheme.pdf")
    assert rec.category_norm == "bath_fixture"
    assert rec.width_cm == 170.0
    assert rec.depth_cm == 75.0
    assert rec.packed_weight_kg == 20.5
    assert cersanit.normalize_url("https://cersanit.ru/catalog/mito/3d-be/?x=1&PAGEN_1=2#frag").endswith("?PAGEN_1=2")
    assert cersanit._looks_like_js_challenge("gorizontal-vertikal construct_utm_uri __jhash_")

    sancos = SancosAdapter()
    sancos_html = """
    <div class="breadcrumbs"><span class="bx-breadcrumb-item"><a><span>Каталог</span></a></span><span class="bx-breadcrumb-item"><a><span>Мойки</span></a></span></div>
    <h1 class="product__info__title">Кухонная мойка Sancos</h1>
    <div class="product__info__text">Sink description</div>
    <div class="product__info__colors__item active"><div class="product__info__colors__item__title">Steel</div></div>
    <div class="product-tab__param"><span class="product-tab__key">Артикул:</span><span class="product-tab__value">SN-1</span></div>
    <div class="product-tab__param"><span class="product-tab__key">Коллекция:</span><span class="product-tab__value">Kitchen</span></div>
    <div class="product-tab__param"><span class="product-tab__key">Ширина (мм):</span><span class="product-tab__value">600</span></div>
    <div class="product-tab__param"><span class="product-tab__key">Глубина (мм):</span><span class="product-tab__value">500</span></div>
    <div class="product-tab__param"><span class="product-tab__key">Высота (мм):</span><span class="product-tab__value">200</span></div>
    <div class="product-tab__param"><span class="product-tab__key">Материал корпуса:</span><span class="product-tab__value">Steel</span></div>
    <div class="product__info__links"><a href="/model.zip">3D модель</a><a href="/scheme.pdf">Схема</a></div>
    <div class="product__gallery"><a href="/gallery/1.jpg"></a><a href="/gallery/1.jpg"></a></div>
    """
    rec = sancos.parse("https://sancos.su/sink", sancos_html, "https://sancos.su/sink")[0]
    assert rec.room == "kitchen"
    assert rec.width_cm == 60.0
    assert rec.depth_cm == 50.0
    assert rec.height_cm == 20.0
    assert rec.model_format == ".zip"
    assert json.loads(rec.extra_json)["model_kind"] == "archive"
    assert SancosAdapter.is_real_product_record(rec)
    assert SancosAdapter.infer_room_bucket(None, "зеркало", None, None) == "bathroom"


def test_timotrader_and_zeelproject_parsing(monkeypatch):
    timo = TimoTraderAdapter()
    library_html = """
    <div id="products">
      <div class="tm-product-item">
        <a class="tm-media-box" href="/product/saona"></a>
        <input name="shk-name" value="Saona MIX-123/01 Хром">
        <input name="shk-id" value="T-1">
        <input name="shk-category" value="Душевые системы">
        <div class="tm-product-price">10 000 ₽</div>
        <div class="tm-product-stock">Есть в наличии</div>
        <img src="/saona.jpg">
      </div>
    </div>
    """
    library_records = timo.parse("https://timotrader.ru/3d-modeli", library_html, "https://timotrader.ru/3d-modeli")
    assert library_records[0].category_norm == "bath_fixture"
    assert library_records[0].collection == "Saona"
    assert library_records[0].color == "Хром"

    product_html = """
    <h1>Saona MIX-123/01 Хром | Timo</h1>
    <div class="uk-breadcrumb"><a>Каталог</a><a>Смесители</a></div>
    <dl class="tm-deflist"><dt>Артикул:</dt><dd>MIX-123</dd></dl>
    <dl class="tm-deflist"><dt>Ширина:</dt><dd>500 мм</dd></dl>
    <dl class="tm-deflist"><dt>Высота:</dt><dd>80 см</dd></dl>
    <dl class="tm-deflist"><dt>Материал корпуса:</dt><dd>Brass</dd></dl>
    <div class="uk-panel"><p>Product description</p></div>
    <div class="tm-product-price">12 000 ₽</div><div class="tm-product-stock">Есть в наличии</div>
    <a href="/files/model.fbx.rar" download>FBX</a>
    <div class="tm-product-gallery"><img src="/img.jpg"></div>
    """
    rec = timo.parse("https://timotrader.ru/product/saona", product_html, "https://timotrader.ru/product/saona")[0]
    assert rec.external_id == "MIX-123"
    assert rec.model_download_url == "https://timotrader.ru/files/model.fbx.rar"
    assert rec.width_cm == 50.0
    assert rec.height_cm == 80.0
    assert rec.materials == "Материал корпуса: Brass"

    zeel = ZeelProjectAdapter()
    zeel_html = """
    <h1 class="model_title">Modern Chair</h1>
    <div class="info_names"><a class="author_link" href="/author">Designer</a></div>
    <div class="speedbar"><span itemprop="name">3D Models</span><span itemprop="name">Furniture</span><span itemprop="name">Chair</span></div>
    <div class="model_options">
      <div class="option"><span class="option_bold">Material</span><span class="option_name"><a>Wood</a><a>Fabric</a></span></div>
      <div class="option"><span class="option_bold">Style</span><span class="option_name">Modern</span></div>
    </div>
    <div class="full_description"><p class="full_story">Useful description\nWidth - 50 cm\nDepth - 60 cm\nHeight - 90 cm</p></div>
    <div class="down_bottons"><a href="https://accounts.zeelproject.com/download/1"><span class="dttl">Download 3D Model</span></a><a href="/skp"><span class="dttl">Download SketchUp</span></a></div>
    <div class="down_container"><div class="down_status">2 credit required</div></div>
    <div class="product_price"><span class="price"><span class="nmbr">$ 3.5</span></span></div>
    <ul id="slider"><li><img data-original="/chair.jpg"></li></ul>
    """
    rec = zeel.parse("https://zeelproject.com/123-modern-chair.html", zeel_html, "https://zeelproject.com/123-modern-chair.html")[0]
    assert rec.external_id == "123"
    assert rec.model_link_type == "button_requires_auth"
    assert rec.price_value == 3.5
    assert rec.price_currency == "USD"
    assert rec.width_cm == 50.0
    assert rec.depth_cm == 60.0
    assert rec.height_cm == 90.0
    assert rec.materials == "Wood, Fabric"
    assert rec.availability == "2 credit required"
    assert ZeelProjectAdapter.parse_dim("Diameter 600 mm", "Diameter") == 60.0
    assert ZeelProjectAdapter.clean_og_title("Chair, free - Download the 3D Model") == "Chair"

    class VerifiedResponse(FakeResponse):
        pass

    monkeypatch.setattr(
        "src.suppliers.adapters.zeelproject.requests.get",
        lambda *a, **k: VerifiedResponse("<html>ok</html>", "https://zeelproject.com/final"),
    )
    assert zeel.fetch_html("https://zeelproject.com/item") == ("<html>ok</html>", "https://zeelproject.com/final")


def test_base_adapter_error_and_edge_branches(monkeypatch):
    adapter = DummyAdapter(timeout=1)

    with pytest.raises(NotImplementedError):
        SupplierAdapter.can_handle(adapter, "https://example.test")
    with pytest.raises(NotImplementedError):
        SupplierAdapter.parse(adapter, "https://example.test", "", "https://example.test")

    monkeypatch.setattr(adapter, "_fetch_html_via_requests", lambda _url: (_ for _ in ()).throw(RuntimeError("requests down")))
    monkeypatch.setattr(adapter, "_fetch_html_via_curl", lambda _url: (_ for _ in ()).throw(RuntimeError("curl down")))
    with pytest.raises(RuntimeError, match="requests down"):
        adapter.fetch_html("https://example.test")
    adapter = DummyAdapter(timeout=1)

    class ChunkyResponse(FakeResponse):
        encoding = "utf-8"

        def iter_content(self, chunk_size=65536):
            yield b""
            yield b"<!doctype html><html><body>"
            yield b"class=\"page-catalog-product\""

    monkeypatch.setattr("src.suppliers.adapters.base.requests.get", lambda *a, **k: ChunkyResponse(b"", "https://example.test/chunky"))
    assert "page-catalog-product" in adapter._fetch_html_via_requests("https://example.test/chunky")[0]

    class TimeoutAfterPartial(FakeResponse):
        encoding = "utf-8"

        def iter_content(self, chunk_size=65536):
            yield (b"<!doctype html><html><body><div class=\"page-catalog-product\">" + b"x" * 2200)
            raise requests.exceptions.Timeout("slow")

    monkeypatch.setattr("src.suppliers.adapters.base.requests.get", lambda *a, **k: TimeoutAfterPartial(b"", "https://example.test/partial"))
    assert "page-catalog-product" in adapter._fetch_html_via_requests("https://example.test/partial")[0]

    class EmptyResponse(FakeResponse):
        def iter_content(self, chunk_size=65536):
            return iter([])

    monkeypatch.setattr("src.suppliers.adapters.base.requests.get", lambda *a, **k: EmptyResponse(b"", "https://example.test/empty"))
    with pytest.raises(RuntimeError, match="Пустой ответ"):
        adapter._fetch_html_via_requests("https://example.test/empty")

    monkeypatch.setattr(
        "src.suppliers.adapters.base.subprocess.run",
        lambda *a, **k: subprocess.CompletedProcess(a[0], 7, stdout=b"not html", stderr=b"bad"),
    )
    with pytest.raises(RuntimeError, match="curl не смог"):
        adapter._fetch_html_via_curl("https://example.test/curl")

    soup = BeautifulSoup(
        """
        <script type="application/ld+json"></script>
        <script type="application/ld+json">not-json</script>
        <script type="application/ld+json">[{"@type": "Product", "name": "List product", "brand": "StringBrand"}, "bad"]</script>
        <a onclick="location.href='/fallback.fbx'">go</a>
        """,
        "html.parser",
    )
    assert adapter.extract_jsonld_objects(soup)[0]["name"] == "List product"
    assert adapter.extract_name_from_jsonld(soup) == "List product"
    assert adapter.extract_brand_from_jsonld(soup) == "StringBrand"
    assert adapter.extract_onclick_download(soup.find("a"), "https://example.test") == "https://example.test/fallback.fbx"
    assert adapter.parse_price_rub("no price") == (None, None, None)
    assert adapter.parse_dimension_cm("Width nope", "Width") is None
    assert adapter.parse_weight_kg("Weight nope") is None
    assert adapter.ext_from_url(None) is None
    assert adapter.filename_from_url(None) is None

    for title, expected in [
        ("large sofa", "sofa"),
        ("queen bed", "bed"),
        ("coffee table", "coffee_table"),
        ("wall cabinet", "cabinet"),
        ("wide sideboard", "sideboard"),
        ("tall wardrobe", "bookcase"),
        ("big mirror", "mirror"),
        ("other soft seating", "ottoman"),
        ("green plant", "plant"),
        ("ceramic planter", "plant"),
        ("unknown", None),
    ]:
        assert adapter.classify_category(title) == expected


def test_cersanit_collection_playwright_and_fallback_extractors(monkeypatch):
    adapter = CersanitAdapter(timeout=1)
    assert adapter.can_handle("https://www.cersanit.ru/catalog/mito/3d-be/")
    assert not adapter.can_handle("https://example.test/")

    ok_html = "<html><title>Cersanit</title><div class='catalog-list-item__title'>ok</div></html>"
    assert adapter._looks_like_js_challenge(ok_html) is False

    class FakeRoute:
        def __init__(self, resource_type):
            self.request = type("Req", (), {"resource_type": resource_type})()
            self.action = None

        def abort(self):
            self.action = "abort"

        def continue_(self):
            self.action = "continue"

    class FakePage:
        url = "https://cersanit.ru/final"

        def route(self, _pattern, callback):
            image_route = FakeRoute("image")
            html_route = FakeRoute("document")
            callback(image_route)
            callback(html_route)
            assert image_route.action == "abort"
            assert html_route.action == "continue"

        def goto(self, *args, **kwargs):
            return None

        def wait_for_timeout(self, *_args):
            return None

        def content(self):
            return "<html><body><div class='product-detail-info__text'>Артикул: A</div><a href='/m.fbx'>fbx</a></body></html>"

    class FakeBrowser:
        closed = False

        def new_page(self):
            return FakePage()

        def close(self):
            self.closed = True

    class FakeChromium:
        def launch(self, headless=True):
            return FakeBrowser()

    class FakePlaywright:
        chromium = FakeChromium()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    import types

    monkeypatch.setitem(sys.modules, "playwright", types.ModuleType("playwright"))
    sync_mod = types.ModuleType("playwright.sync_api")
    sync_mod.sync_playwright = lambda: FakePlaywright()
    monkeypatch.setitem(sys.modules, "playwright.sync_api", sync_mod)
    html, final_url = adapter.fetch_html("https://cersanit.ru/item")
    assert final_url == "https://cersanit.ru/final"
    assert "product-detail-info" in html

    collection_html = """
    <a class="catalog-list-item__info" href="/catalog/mito/3d-be/product-a/"></a>
    <a class="catalog-list-item__pic-area" href="/other/skip/"></a>
    <a href="/catalog/mito/3d-be/?PAGEN_1=2&x=1">page</a>
    """
    product_html = """
    <h1>Сифон MITO</h1>
    <div class="product-detail-info__text">Артикул: S-1</div>
    <div class="specs__element"><span class="specs__title">Тип продукта:</span><span class="specs__value">Сифон</span></div>
    <a href="/files/siphon.fbx">FBX</a>
    """
    fetched = {
        "https://cersanit.ru/catalog/mito/3d-be/?PAGEN_1=2": collection_html,
        "https://cersanit.ru/catalog/mito/3d-be/product-a": product_html,
    }
    monkeypatch.setattr(adapter, "fetch_html", lambda url: (fetched[url], url))
    soup = BeautifulSoup(collection_html, "html.parser")
    records = adapter.parse_collection_page("https://cersanit.ru/catalog/mito/3d-be/", soup, "https://cersanit.ru/catalog/mito/3d-be/")
    assert records[0].category_norm == "bath_accessory"
    assert adapter.parse("https://cersanit.ru/x", "<html>plain</html>", "https://cersanit.ru/x") == []

    fallback_soup = BeautifulSoup(
        """
        <meta property="og:title" content="Раковина купить / цена">
        <meta name="description" content="Fallback desc">
        <div class="bx-breadcrumb"><a href="/catalog/mito/">Mito</a></div>
        <img src="/upload/wrong.jpg" alt="other">
        <img src="/upload/sink.jpg" alt="Раковина купить">
        """,
        "html.parser",
    )
    assert adapter.extract_title(fallback_soup) == "Раковина"
    assert adapter.extract_collection(fallback_soup, "https://cersanit.ru/catalog/mito/sinks/item/") == "mito".title()
    assert adapter.extract_description(fallback_soup) == "Fallback desc"
    assert adapter.extract_breadcrumb_category(fallback_soup) == "Mito"
    assert adapter.extract_product_images(fallback_soup, "https://cersanit.ru", "Раковина купить") == ["https://cersanit.ru/upload/sink.jpg"]
    assert adapter.clean_og_title(None) is None
    assert adapter.parse_float("bad") is None
    assert adapter.classify_cersanit_category("сиденье", None) == "toilet_seat"
    assert adapter.classify_cersanit_category("компакт", None) == "toilet"
    assert adapter.classify_cersanit_category("раковина", None) == "bath_sink"


def test_homeconcept_extractor_fallbacks_and_json_edges():
    adapter = HomeConceptAdapter()
    assert adapter.can_handle("https://homeconcept.ru/item")
    assert adapter.clean_title(None) is None

    soup = BeautifulSoup(
        """
        <meta property="og:title" content="OG Chair - Home Concept">
        <meta property="og:description" content="OG desc">
        <script type="application/ld+json">{"@type": "Product", "name": "Json chair", "description": "Json desc", "offers": {"lowPrice": "10,5"}}</script>
        <table class="product-characteristics">
          <tr><td>Бренд</td><td>TableBrand</td></tr>
          <tr><td>Материал каркаса</td><td>Metal</td></tr>
          <tr><td>Диаметр</td><td>45 см</td></tr>
          <tr><td>Вес в упаковке</td><td>6 кг</td></tr>
        </table>
        <div class="content-block collections-list"><div class="title"><a>Collection A</a></div></div>
        <a href="/download/model.rar">скачать 3d-модель</a>
        <div class="catalog-product__3d-model"><img data-src="/model-preview.jpg"></div>
        <div class="catalog-top-items"><div class="item"><a class="item-link-image" href="/generic">Generic</a><img data-src="/g.jpg"></div></div>
        """,
        "html.parser",
    )
    assert adapter.extract_product_title(soup) == "OG Chair"
    assert adapter.extract_brand(soup) == "TableBrand"
    assert adapter.extract_description(soup) == "OG desc"
    assert adapter.extract_price(soup) == (10.5, None, "RUB")
    assert adapter.extract_availability(BeautifulSoup("<div></div>", "html.parser")) is None
    assert adapter.extract_color(BeautifulSoup("<div class='catalog-product__current-check-offer-material'>Walnut</div>", "html.parser")) == "Walnut"
    assert adapter.extract_materials(soup) == "Материал каркаса: Metal"
    assert adapter.extract_collection(soup, "") == "Collection A"
    assert adapter.extract_dimensions(soup) == (45.0, 45.0, None)
    assert adapter.extract_weight_from_table(soup) == 6.0
    assert adapter.extract_model_download_url(soup, "https://homeconcept.ru/base/") == "https://homeconcept.ru/download/model.rar"
    assert adapter.extract_product_images(soup, "https://homeconcept.ru") == ["https://homeconcept.ru/model-preview.jpg"]
    assert adapter.extract_images_from_card(None, "https://homeconcept.ru") == []
    assert adapter.extract_images_from_card(BeautifulSoup("<img>", "html.parser").img, "https://homeconcept.ru") == []
    assert adapter.extract_related_items(soup, "https://homeconcept.ru")[0]["relation"] == "generic"
    assert adapter.extract_related_from_tile(BeautifulSoup("<div></div>", "html.parser"), "https://homeconcept.ru", "x") is None
    assert json.loads(adapter._merge_extra_json("not-json", {"ok": True})) == {"ok": True}
    assert adapter._json_load_list("not-json") == []

    json_only = BeautifulSoup(
        '<script type="application/ld+json">{"@type":"Product","name":"JSON only","description":"Json description","offers":{"price":"bad"}}</script>',
        "html.parser",
    )
    assert adapter.extract_product_title(json_only) == "JSON only"
    assert adapter.extract_description(BeautifulSoup('<script type="application/ld+json">{"description":"Json description"}</script>', "html.parser")) == "Json description"
    assert adapter.extract_price(json_only) == (None, None, None)
    assert adapter.parse_numeric_value(None) is None
    assert adapter.parse_numeric_value("no number") is None


def test_zeelproject_network_fallback_and_edge_helpers(monkeypatch):
    adapter = ZeelProjectAdapter(timeout=1)
    assert adapter.can_handle("https://www.zeelproject.com/model")
    assert not adapter.can_handle("https://example.test/model")
    assert adapter.parse("https://zeelproject.com/x", "<html></html>", "https://zeelproject.com/x") == []

    monkeypatch.setattr(adapter, "_fetch_html_verified_via_requests", lambda _url: (_ for _ in ()).throw(RuntimeError("blocked")))
    monkeypatch.setattr(adapter, "_fetch_html_verified_via_curl", lambda url: ("<html>curl</html>", url + "?ok=1"))
    assert adapter.fetch_html("https://zeelproject.com/model")[1].endswith("?ok=1")
    adapter = ZeelProjectAdapter(timeout=1)

    class AntiBotResponse(FakeResponse):
        text = "ZEEL PROJECT - 3D Models & Interior Design navigator.webdriver"

    monkeypatch.setattr("src.suppliers.adapters.zeelproject.requests.get", lambda *a, **k: AntiBotResponse("", "https://zeelproject.com/x"))
    with pytest.raises(RuntimeError, match="anti-bot"):
        adapter._fetch_html_verified_via_requests("https://zeelproject.com/x")

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a[0], 1, stdout=b"", stderr=b"curl fail"))
    with pytest.raises(RuntimeError, match="curl failed"):
        adapter._fetch_html_verified_via_curl("https://zeelproject.com/x")

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a[0], 0, stdout=b"<html></html>", stderr=b""))
    with pytest.raises(RuntimeError, match="effective URL marker"):
        adapter._fetch_html_verified_via_curl("https://zeelproject.com/x")

    soup = BeautifulSoup(
        """
        <h1>Plain Title</h1>
        <div class="brand_name"><a href="/brands/b">Brand B</a></div>
        <div class="model_options"><div class="option"><span class="option_bold">Category</span><span class="option_name">Decor</span></div></div>
        <div class="full_description"><p class="full_story">Diameter 600 mm\nHeight bad cm</p></div>
        <meta property="og:image" content="/og.jpg">
        """,
        "html.parser",
    )
    assert adapter.extract_title(soup) == "Plain Title"
    assert adapter.extract_brand(soup) == "Brand B"
    assert adapter.extract_category(soup) == "Decor"
    assert adapter.extract_description(soup) == "Diameter 600 mm\nHeight bad cm"
    assert adapter.extract_description(BeautifulSoup("<div class='full_description'><p class='full_story'>Height - 40 cm</p></div>", "html.parser")) is None
    assert adapter.extract_dimensions(soup) == (60.0, 60.0, None)
    assert adapter.extract_download_landing_url(soup, "https://zeelproject.com") is None
    assert adapter.extract_sketchup_landing_url(soup, "https://zeelproject.com") is None
    assert adapter.extract_credit_requirement(BeautifulSoup("<div class='down_container'><div class='down_status'>бесплатно</div></div>", "html.parser")) == 0
    assert adapter.extract_credit_requirement(BeautifulSoup("<div>требуется 3 кредит</div>", "html.parser")) == 3
    assert adapter.extract_availability(BeautifulSoup("<div></div>", "html.parser")) is None
    assert adapter.extract_price(BeautifulSoup("<span class='nmbr'>bad</span>", "html.parser")) == (None, None)
    assert adapter.extract_images(soup, "https://zeelproject.com") == ["https://zeelproject.com/og.jpg"]
    assert adapter.extract_external_id("https://zeelproject.com/no-id.html") is None
    assert adapter.extract_option_value(BeautifulSoup("<div></div>", "html.parser"), "Material") is None
    assert ZeelProjectAdapter.parse_dim("Height broken cm", "Height") is None


def test_supplier_adapter_remaining_small_edge_branches(monkeypatch):
    base = DummyAdapter(timeout=1)
    assert base._has_enough_html_signals(b'<div class="page-3d-card">x</div>')
    assert base._has_enough_html_signals(b"plain download link")
    assert base._is_usable_partial_html(b"short html") is False
    assert base._is_usable_partial_html(b"x" * 3000) is False
    assert base._is_usable_partial_html(b"<!doctype html><html>" + b"x" * 3000 + b"</html>")
    assert base.parse_dimension_cm("Ширина bad см", "Ширина") is None
    assert base.parse_weight_kg("Вес в упаковке bad кг") is None
    assert base.extract_onclick_download(BeautifulSoup("<a></a>", "html.parser").a, "https://example.test") is None
    assert base.ext_from_url("https://example.test/noext") is None
    assert base.filename_from_url("https://example.test/") is None
    assert base.classify_category("кашпо ceramic") == "planter"

    class BrokenPartial(FakeResponse):
        def iter_content(self, chunk_size=65536):
            yield b"<html>tiny"
            raise requests.exceptions.ConnectionError("broken")

    monkeypatch.setattr("src.suppliers.adapters.base.requests.get", lambda *a, **k: BrokenPartial(b"", "https://example.test/bad"))
    with pytest.raises(requests.exceptions.ConnectionError):
        base._fetch_html_via_requests("https://example.test/bad")

    timo = TimoTraderAdapter()
    assert timo.can_handle("https://www.timotrader.ru/3d-modeli")
    assert not timo.can_handle("https://example.test/3d-modeli")
    assert timo.parse_library_card(BeautifulSoup("<div class='tm-product-item'></div>", "html.parser").div, "https://timotrader.ru") is None
    empty_title_card = BeautifulSoup(
        "<div><a class='tm-media-box' href='/p'></a><div class='tm-product-card-body'><p><a></a></p></div></div>",
        "html.parser",
    ).div
    assert timo.parse_library_card(empty_title_card, "https://timotrader.ru") is None
    fallback_card = BeautifulSoup(
        """
        <div>
          <a class="tm-product-card-body" href="/ignored"></a>
          <div class="tm-product-card-body"><a href="/p">Tetra T-100 Никель</a><p><a>Tetra T-100 Никель</a></p></div>
          <input name="shk-id" value="">
          <img data-src="/a.jpg"><img data-src="/a.jpg">
        </div>
        """,
        "html.parser",
    ).div
    parsed_card = timo.parse_library_card(fallback_card, "https://timotrader.ru")
    assert parsed_card is not None
    assert parsed_card.title == "Tetra T-100 Никель"
    assert json.loads(parsed_card.images_json) == ["https://timotrader.ru/a.jpg"]

    timo_soup = BeautifulSoup(
        """
        <html><head>
          <meta property="og:title" content="OG Timo | Timo">
          <meta name="description" content="Meta desc">
          <script type="application/ld+json">{"@type":"Product","name":"Json Timo"}</script>
        </head><body>
          <dl class="tm-deflist"><dt></dt><dd>skip</dd></dl>
          <dl class="tm-deflist"><dt>Диаметр:</dt><dd>250 мм</dd></dl>
          <a href="/no-model">plain</a>
          <span class="tm-product-price">Цена 7 000 ₽</span>
          <p>скоро в продаже</p>
        </body></html>
        """,
        "html.parser",
    )
    assert timo.extract_title(timo_soup) == "OG Timo"
    assert timo.extract_description(timo_soup) == "Meta desc"
    assert timo.extract_fbx_download_url(timo_soup, "https://timotrader.ru") is None
    assert timo.extract_price_from_node(timo_soup) == (7000.0, None, "RUB")
    assert timo.extract_availability(timo_soup) == "Скоро в продаже"
    assert timo.extract_card_availability(BeautifulSoup("<div></div>", "html.parser")) is None
    assert timo.extract_hidden_value(BeautifulSoup("<input name='x' value=''>", "html.parser"), "x") is None
    assert timo.extract_collection_from_title("Unknown") is None
    assert timo.extract_article_from_title("MIX-123/01A Хром") == "MIX-123/01A"
    assert timo.extract_color_from_title("Saona /01A Золото матовое") == "Золото матовое"
    assert timo.extract_color({"Цвет корпуса": "White"}, "ignored") == "White"
    assert timo.extract_dimensions({"Диаметр": "250 мм"})[:2] == (25.0, 25.0)
    assert timo.parse_dimension_value(None) is None
    assert timo.parse_dimension_value("no number") is None
    monkeypatch.setattr("src.suppliers.adapters.timotrader.re.search", lambda *a, **k: type("M", (), {"group": lambda self, _i: "bad"})())
    assert TimoTraderAdapter.parse_dimension_value("123 см") is None
    monkeypatch.undo()
    assert timo.classify_timotrader_category("бумагодержатель", None) == "bath_accessory"
    assert timo.classify_timotrader_category("трап", None) == "drain"

    home = HomeConceptAdapter()
    assert home.extract_product_title(BeautifulSoup("<div class='catalog-product__name-text'> Name Text | Home Concept</div>", "html.parser")) == "Name Text"
    assert home.extract_product_title(BeautifulSoup("<h1 class='catalog-product__name'> H1 Name - Home Concept</h1>", "html.parser")) == "H1 Name"
    assert home.extract_code(BeautifulSoup("<div></div>", "html.parser")) is None
    assert home.extract_description(BeautifulSoup("<div class='item-info-detail-text'>Line 1<br>Line 2</div>", "html.parser")) == "Line 1 Line 2"
    assert home.extract_price(BeautifulSoup('<script type="application/ld+json">{"@type":"Thing"}</script>', "html.parser")) == (None, None, None)
    assert home.extract_color(BeautifulSoup("<div></div>", "html.parser")) is None
    assert home.extract_collection(BeautifulSoup("<div></div>", "html.parser"), "My Collection другие модели коллекции") == "My Collection"
    assert home.extract_breadcrumb_category(BeautifulSoup("<div></div>", "html.parser")) is None
    assert home.extract_numeric_dimension(BeautifulSoup('<script type="application/ld+json">{"@type":"Product","additionalProperty":[{"name":"Ширина","value":"88 см"}]}</script>', "html.parser"), ["Ширина"]) == 88.0
    assert home.extract_weight_from_table(BeautifulSoup('<script type="application/ld+json">{"@type":"Product","additionalProperty":[{"name":"Вес в упаковке","value":"9 кг"}]}</script>', "html.parser")) == 9.0
    assert home.extract_characteristic(BeautifulSoup("<table class='product-characteristics'><tr><td>bad</td></tr></table>", "html.parser"), "bad") is None
    assert home.extract_model_download_url(BeautifulSoup("<a href=''></a><a href='/m.rar'>скачать 3d</a>", "html.parser"), "https://homeconcept.ru") == "https://homeconcept.ru/m.rar"
    assert home.extract_product_images(BeautifulSoup("<div class='catalog-product__image-slider'><img src='/a.jpg'><img src='/a.jpg'></div>", "html.parser"), "https://homeconcept.ru") == ["https://homeconcept.ru/a.jpg"]
    assert home._json_load_list('[1, 2]') == [1, 2]

    sancos = SancosAdapter()
    assert sancos.parse("https://sancos.su/empty", "<h1>Продукция по типологии</h1>", "https://sancos.su/empty") == []
    assert not SancosAdapter.is_real_product_record(sancos.parse_product_page("u", BeautifulSoup("<h1>Продукция по типологии</h1>", "html.parser"), "u"))
    assert SancosAdapter.is_real_product_record(sancos.parse_product_page("u", BeautifulSoup("<h1>Title</h1><div class='product__info__text'>Desc</div><div class='product-tab__param'><span class='product-tab__key'>Коллекция:</span><span class='product-tab__value'>C</span></div>", "html.parser"), "u"))
    assert SancosAdapter.infer_room_bucket(None, "other", None, None) is None
    assert sancos.extract_title(BeautifulSoup("<title>Title fallback</title>", "html.parser")) == "Title fallback"
    assert sancos.extract_description(BeautifulSoup("<meta name='description' content='Meta desc'>", "html.parser")) == "Meta desc"
    assert sancos.extract_color(BeautifulSoup("<div class='product__info__colors__item__title'>Any color</div>", "html.parser")) == "Any color"
    assert sancos.extract_labeled_link(BeautifulSoup("<div class='product__info__links'><a href=''>3D модель</a></div>", "html.parser"), "https://sancos.su", "3D модель") is None
    assert sancos.extract_gallery_images(BeautifulSoup("<div class='product__gallery'><a href=''></a></div>", "html.parser"), "https://sancos.su") == []
    assert sancos.extract_breadcrumb_category(BeautifulSoup("<div></div>", "html.parser")) is None
    assert SancosAdapter.extract_materials({"Покрытие корпуса": "matte", "Покрытие фасада": "gloss"}) == "Покрытие корпуса: matte; Покрытие фасада: gloss"
    assert SancosAdapter._to_float("no number") is None
    assert SancosAdapter._mm_to_cm(None) is None
    assert SancosAdapter._infer_model_kind(None) is None
    assert SancosAdapter._infer_model_kind("https://x.test/model.fbx") == "direct_model"
    assert SancosAdapter._infer_model_kind("https://x.test/file.txt") == "file"
