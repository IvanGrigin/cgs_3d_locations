# -*- coding: utf-8 -*-
"""
Cersanit adapter for the MITO collection catalog.

The site serves a short JS anti-bot page to plain HTTP clients. For this
supplier we use Playwright and wait for the timed redirect to finish before
reading the DOM. Product pages expose downloadable materials where only FBX
files are acceptable for ingestion; pages without FBX are skipped.
"""
from __future__ import annotations

import json
import re
from typing import Optional
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

from bs4 import BeautifulSoup

from src.suppliers.models import ProductRecord

from .base import SupplierAdapter


class CersanitAdapter(SupplierAdapter):
    site_name = "cersanit"
    empty_parse_is_skip = True
    collection_root = "https://cersanit.ru/catalog/mito/3d-be/"

    def can_handle(self, url: str) -> bool:
        host = urlparse(url).netloc.lower()
        return host == "cersanit.ru" or host == "www.cersanit.ru"

    def fetch_html(self, url: str) -> tuple[str, str]:
        return self._fetch_html_via_playwright(url)

    def _fetch_html_via_playwright(self, url: str) -> tuple[str, str]:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                page.route(
                    "**/*",
                    lambda route: route.abort()
                    if route.request.resource_type in {"image", "media", "font"}
                    else route.continue_(),
                )
                page.goto(url, wait_until="domcontentloaded", timeout=max(60000, self.timeout * 1000))
                page.wait_for_timeout(4000)
                html = page.content()
                final_url = page.url or url
            finally:
                browser.close()

        if self._looks_like_js_challenge(html):
            raise RuntimeError("Cersanit anti-bot challenge did not resolve to product HTML")
        return html, final_url

    @staticmethod
    def _looks_like_js_challenge(html: str) -> bool:
        low = html.lower()
        if "<title>" in low and "cersanit" in low and "catalog-list-item__title" in low:
            return False
        return (
            "gorizontal-vertikal" in low
            and "construct_utm_uri" in low
            and "__jhash_" in low
            and "catalog-list-item__title" not in low
            and "product-detail-info__text" not in low
        )

    def parse(self, url: str, html: str, final_url: str) -> list[ProductRecord]:
        soup = BeautifulSoup(html, "html.parser")
        if self.looks_like_product_page(soup):
            product = self.parse_product_page(url, soup, final_url)
            return [product] if product else []
        if self.looks_like_collection_page(soup):
            return self.parse_collection_page(url, soup, final_url)
        product = self.parse_product_page(url, soup, final_url)
        return [product] if product else []

    @staticmethod
    def looks_like_collection_page(soup: BeautifulSoup) -> bool:
        return bool(soup.select_one("a.catalog-list-item__info[href], a.catalog-list-item__pic-area[href]"))

    @staticmethod
    def looks_like_product_page(soup: BeautifulSoup) -> bool:
        return bool(
            soup.select_one(
                ".product-detail-info__text, .specs__element, .description-section__text, a[href*='/catalog/pdf/?ID=']"
            )
        )

    def parse_collection_page(self, source_url: str, soup: BeautifulSoup, final_url: str) -> list[ProductRecord]:
        product_urls = self.extract_collection_product_urls(soup, final_url)
        page_urls = self.extract_collection_page_urls(soup, final_url)
        records: list[ProductRecord] = []
        seen_product_urls = set()

        for listing_url in [final_url, *page_urls]:
            if listing_url != final_url:
                html, listing_final_url = self.fetch_html(listing_url)
                listing_soup = BeautifulSoup(html, "html.parser")
                current_product_urls = self.extract_collection_product_urls(listing_soup, listing_final_url)
            else:
                current_product_urls = product_urls

            for product_url in current_product_urls:
                if product_url in seen_product_urls:
                    continue
                seen_product_urls.add(product_url)
                html, product_final_url = self.fetch_html(product_url)
                product_soup = BeautifulSoup(html, "html.parser")
                product = self.parse_product_page(source_url, product_soup, product_final_url)
                if product:
                    records.append(product)

        return records

    def parse_product_page(self, source_url: str, soup: BeautifulSoup, final_url: str) -> Optional[ProductRecord]:
        title = self.extract_title(soup)
        specs = self.extract_specs(soup)
        model_download_url = self.extract_fbx_download_url(soup, final_url)
        if not model_download_url:
            return None

        category_raw = specs.get("Тип продукта") or self.extract_breadcrumb_category(soup)
        description = self.extract_description(soup)
        collection = self.extract_collection(soup, final_url)
        external_id = self.extract_article(soup)
        color = specs.get("Цвет")
        materials = specs.get("Материал")
        width_cm, depth_cm, height_cm = self.extract_dimensions(specs)
        weight_kg = self.parse_float(specs.get("Вес (без упаковки), кг"))
        packed_weight_kg = self.parse_float(specs.get("Вес (в упаковке), кг"))
        tags = [x for x in [category_raw, collection, color, specs.get("Дизайн сегмент")] if x]
        images = self.extract_product_images(soup, final_url, title)

        extra = {
            "parse_stage": "product",
            "download_links": self.extract_download_links(soup, final_url),
            "specs": specs,
        }

        return ProductRecord(
            unique_key=self.build_unique_key(final_url, external_id),
            source_site=self.site_name,
            source_url=source_url,
            parsed_at=self.now_utc_iso(),
            external_id=external_id,
            category_raw=category_raw,
            category_norm=self.classify_cersanit_category(title, category_raw),
            title=title,
            brand="Cersanit",
            collection=collection,
            product_url=final_url,
            model_link_type="direct_file",
            model_page_url=final_url,
            model_download_url=model_download_url,
            model_download_landing_url=None,
            model_vendor_url=final_url,
            model_extraction_method="cersanit_downloads_fbx",
            model_download_filename=self.filename_from_url(model_download_url),
            model_format=self.ext_from_url(model_download_url),
            price_value=None,
            price_currency=None,
            old_price_value=None,
            style=specs.get("Дизайн сегмент"),
            color=color,
            description=description,
            width_cm=width_cm,
            depth_cm=depth_cm,
            height_cm=height_cm,
            weight_kg=weight_kg,
            volume_m3=None,
            package_width_cm=None,
            package_depth_cm=None,
            package_height_cm=None,
            packed_weight_kg=packed_weight_kg,
            scheme_url=self.extract_scheme_url(soup, final_url),
            room="bathroom",
            materials=materials,
            availability=None,
            country_brand=None,
            production_country=None,
            tags_json=json.dumps(tags, ensure_ascii=False),
            images_json=json.dumps(images, ensure_ascii=False),
            related_json="[]",
            extra_json=json.dumps(extra, ensure_ascii=False),
            raw_html=str(soup),
        )

    def extract_collection_product_urls(self, soup: BeautifulSoup, base_url: str) -> list[str]:
        urls: list[str] = []
        for a in soup.select("a.catalog-list-item__info[href], a.catalog-list-item__pic-area[href]"):
            href = str(a.get("href") or "").strip()
            if not href:
                continue
            absolute = urljoin(base_url, href)
            if not absolute.startswith(self.collection_root):
                continue
            urls.append(self.normalize_url(absolute))
        return self.unique_keep_order(urls)

    def extract_collection_page_urls(self, soup: BeautifulSoup, base_url: str) -> list[str]:
        urls: list[str] = []
        for a in soup.select("a[href*='PAGEN_1=']"):
            href = str(a.get("href") or "").strip()
            if not href:
                continue
            absolute = self.normalize_url(urljoin(base_url, href))
            if absolute.startswith(self.collection_root):
                urls.append(absolute)
        return self.unique_keep_order(urls)

    def extract_title(self, soup: BeautifulSoup) -> Optional[str]:
        node = soup.select_one("h1")
        if node:
            return self.norm_space(node.get_text(" ", strip=True))
        og = self.extract_meta_content(soup, "og:title")
        return self.clean_og_title(og)

    def extract_collection(self, soup: BeautifulSoup, final_url: str) -> Optional[str]:
        crumbs = []
        for a in soup.select(".bx-breadcrumb a[href], .breadcrumb a[href], .breadcrumbs a[href]"):
            text = self.norm_space(a.get_text(" ", strip=True))
            href = str(a.get("href") or "")
            if "/catalog/mito/" in href and text.lower() != "продукты":
                crumbs.append(text)
        if crumbs:
            return crumbs[0]

        parts = [x for x in urlparse(final_url).path.split("/") if x]
        if len(parts) >= 2:
            return parts[1].replace("-", " ").title()
        return None

    def extract_article(self, soup: BeautifulSoup) -> Optional[str]:
        node = soup.select_one(".product-detail-info__text")
        text = self.norm_space(node.get_text(" ", strip=True)) if node else ""
        m = re.search(r"Артикул:\s*(.+)$", text, flags=re.IGNORECASE)
        return self.norm_space(m.group(1)) if m else None

    def extract_specs(self, soup: BeautifulSoup) -> dict[str, str]:
        specs: dict[str, str] = {}
        for item in soup.select(".specs__element"):
            key_node = item.select_one(".specs__title")
            value_node = item.select_one(".specs__value")
            if not key_node or not value_node:
                continue
            key = self.norm_space(key_node.get_text(" ", strip=True)).rstrip(":")
            value = self.norm_space(value_node.get_text(" ", strip=True))
            if key and value and key not in specs:
                specs[key] = value
        return specs

    def extract_description(self, soup: BeautifulSoup) -> Optional[str]:
        heading = soup.find(lambda tag: tag.name in {"span", "div", "h2"} and "Описание" in tag.get_text(" ", strip=True))
        if heading:
            section = heading.find_next(class_="description-section__text")
            if section:
                text = self.norm_space(section.get_text(" ", strip=True))
                if text:
                    return text
        for node in soup.select(".description-section__text"):
            text = self.norm_space(node.get_text(" ", strip=True))
            if text:
                return text
        return self.extract_meta_content(soup, "description")

    def extract_download_links(self, soup: BeautifulSoup, base_url: str) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        for a in soup.select("a[href]"):
            text = self.norm_space(a.get_text(" ", strip=True))
            href = str(a.get("href") or "").strip()
            if not href:
                continue
            low = f"{text} {href}".lower()
            if "скач" not in low and not any(ext in low for ext in (".fbx", ".obj", ".max", ".pdf", ".dwg", ".stp", ".mtl", ".docx")):
                continue
            out.append(
                {
                    "label": text,
                    "url": urljoin(base_url, href),
                }
            )
        return out

    def extract_fbx_download_url(self, soup: BeautifulSoup, base_url: str) -> Optional[str]:
        for item in self.extract_download_links(soup, base_url):
            low = f"{item['label']} {item['url']}".lower()
            if ".fbx" in low or re.search(r"\bfbx\b", low):
                return item["url"]
        return None

    def extract_breadcrumb_category(self, soup: BeautifulSoup) -> Optional[str]:
        parts: list[str] = []
        for a in soup.select(".bx-breadcrumb a[href], .breadcrumb a[href], .breadcrumbs a[href]"):
            text = self.norm_space(a.get_text(" ", strip=True))
            if text and text.lower() not in {"главная", "продукты"}:
                parts.append(text)
        return " > ".join(parts) if parts else None

    def extract_scheme_url(self, soup: BeautifulSoup, base_url: str) -> Optional[str]:
        for item in self.extract_download_links(soup, base_url):
            low = item["label"].lower()
            if "чертеж" in low or "схем" in low:
                return item["url"]
        return None

    def extract_dimensions(self, specs: dict[str, str]) -> tuple[Optional[float], Optional[float], Optional[float]]:
        width = self.parse_float(specs.get("Ширина, см"))
        depth = self.parse_float(specs.get("Глубина, см"))
        height = self.parse_float(specs.get("Высота, см"))
        if depth is None:
            depth = self.parse_float(specs.get("Длина, см"))
        return width, depth, height

    def extract_product_images(self, soup: BeautifulSoup, base_url: str, title: Optional[str]) -> list[str]:
        urls: list[str] = []
        og_image = self.extract_meta_content(soup, "og:image")
        if og_image:
            urls.append(urljoin(base_url, og_image))

        title_l = (title or "").lower()
        for img in soup.select("img[src], img[data-src]"):
            src = img.get("data-src") or img.get("src")
            if not src:
                continue
            alt = self.norm_space(img.get("alt", "")).lower()
            absolute = urljoin(base_url, src)
            if "/upload/" not in absolute and "/resize_cache/" not in absolute:
                continue
            if title_l and alt and title_l[:20] not in alt and alt not in title_l:
                continue
            urls.append(absolute)

        return self.unique_keep_order(urls)[:20]

    @staticmethod
    def parse_float(value: Optional[str]) -> Optional[float]:
        text = str(value or "").strip()
        if not text:
            return None
        m = re.search(r"(-?\d+(?:[.,]\d+)?)", text)
        if not m:
            return None
        try:
            return float(m.group(1).replace(",", "."))
        except Exception:
            return None

    @staticmethod
    def clean_og_title(title: Optional[str]) -> Optional[str]:
        if not title:
            return None
        text = SupplierAdapter.norm_space(title)
        text = re.sub(r"\s+купить\s*/\s*цена.*$", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*/\s*Официальный сайт.*$", "", text, flags=re.IGNORECASE)
        return text or None

    @staticmethod
    def normalize_url(url: str) -> str:
        parsed = urlparse(url.strip())
        query = dict(parse_qsl(parsed.query, keep_blank_values=False))
        query = {k: v for k, v in query.items() if k == "PAGEN_1"}
        path = parsed.path.rstrip("/") or "/"
        return urlunparse(parsed._replace(path=path, query=urlencode(query), fragment=""))

    @staticmethod
    def unique_keep_order(values: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for value in values:
            if not value or value in seen:
                continue
            seen.add(value)
            out.append(value)
        return out

    def classify_cersanit_category(self, title: Optional[str], category_raw: Optional[str]) -> Optional[str]:
        text = f"{title or ''} {category_raw or ''}".lower()
        if "каркас" in text or "рама" in text or "ванн" in text:
            return "bath_fixture"
        if "панел" in text:
            return "bath_accessory"
        if "раковин" in text:
            return "bath_sink"
        if "сиденье" in text or "крышка" in text:
            return "toilet_seat"
        if "компакт" in text or "унитаз" in text:
            return "toilet"
        if "сифон" in text:
            return "bath_accessory"
        return self.classify_category(title or category_raw)
