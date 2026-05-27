# -*- coding: utf-8 -*-
"""
Timotrader adapter.

The 3D library is a MODX product listing at /3d-modeli. Product pages expose
download links in a "3D модели" block, where FBX is usually packaged as
*.fbx.rar. The adapter keeps the actual file extension in model_format and
stores the declared FBX format in extra_json.
"""
from __future__ import annotations

import json
import re
from typing import Optional
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from src.suppliers.models import ProductRecord

from .base import SupplierAdapter


class TimoTraderAdapter(SupplierAdapter):
    site_name = "timotrader"

    def can_handle(self, url: str) -> bool:
        host = urlparse(url).netloc.lower()
        return host == "timotrader.ru" or host == "www.timotrader.ru"

    def parse(self, url: str, html: str, final_url: str) -> list[ProductRecord]:
        soup = BeautifulSoup(html, "html.parser")
        path = urlparse(final_url).path.lower()
        if path.rstrip("/") == "/3d-modeli":
            return self.parse_library_page(url, soup, final_url)
        return [self.parse_product_page(url, soup, final_url)]

    def parse_library_page(self, source_url: str, soup: BeautifulSoup, final_url: str) -> list[ProductRecord]:
        records: list[ProductRecord] = []
        for card in soup.select("#products .tm-product-item"):
            record = self.parse_library_card(card, final_url)
            if record:
                records.append(record)
        return records

    def parse_library_card(self, card, base_url: str) -> Optional[ProductRecord]:
        link = card.select_one(".tm-media-box[href]") or card.select_one(".tm-product-card-body a[href]")
        if not link or not link.get("href"):
            return None

        product_url = urljoin(base_url, str(link["href"]).strip())
        title = self.extract_hidden_value(card, "shk-name")
        if not title:
            title_node = card.select_one(".tm-product-card-body p a")
            title = self.norm_space(title_node.get_text(" ", strip=True)) if title_node else None
        if not title:
            return None

        external_id = self.extract_hidden_value(card, "shk-id")
        category_raw = self.extract_hidden_value(card, "shk-category")
        price_value, _, price_currency = self.extract_price_from_node(card)
        availability = self.extract_card_availability(card)
        images = self.extract_card_images(card, base_url)

        return ProductRecord(
            unique_key=self.build_unique_key(product_url, external_id),
            source_site=self.site_name,
            source_url=base_url,
            parsed_at=self.now_utc_iso(),
            external_id=external_id,
            category_raw=category_raw,
            category_norm=self.classify_timotrader_category(title, category_raw),
            title=title,
            brand="TIMO",
            collection=self.extract_collection_from_title(title),
            product_url=product_url,
            model_link_type=None,
            model_page_url=product_url,
            model_download_url=None,
            model_download_landing_url=None,
            model_vendor_url=product_url,
            model_extraction_method="timotrader_library_card",
            model_download_filename=None,
            model_format=None,
            price_value=price_value,
            price_currency=price_currency,
            old_price_value=None,
            style=None,
            color=self.extract_color_from_title(title),
            description=None,
            width_cm=None,
            depth_cm=None,
            height_cm=None,
            weight_kg=None,
            room="bathroom",
            materials=None,
            availability=availability,
            country_brand=None,
            production_country=None,
            tags_json=json.dumps([x for x in [category_raw, availability] if x], ensure_ascii=False),
            images_json=json.dumps(images, ensure_ascii=False),
            related_json="[]",
            extra_json=json.dumps(
                {
                    "parse_stage": "library",
                    "library_page_url": base_url,
                    "enriched_from_product_page": False,
                },
                ensure_ascii=False,
            ),
            raw_html=str(card),
        )

    def parse_product_page(self, source_url: str, soup: BeautifulSoup, final_url: str) -> ProductRecord:
        params = self.extract_params(soup)
        title = self.extract_title(soup)
        description = self.extract_description(soup)
        category_raw = self.extract_breadcrumb_category(soup)
        price_value, old_price_value, price_currency = self.extract_price_from_node(soup)
        availability = self.extract_availability(soup)
        model_download_url = self.extract_fbx_download_url(soup, final_url)
        images = self.extract_product_images(soup, final_url)
        collection = params.get("Коллекция") or self.extract_collection_from_title(title)
        color = self.extract_color(params, title)
        materials = self.extract_materials(params)
        country_brand = params.get("Страна бренда")
        production_country = params.get("Страна производства") or params.get("Производство")
        width_cm, depth_cm, height_cm = self.extract_dimensions(params)

        extra = {
            "parse_stage": "product",
            "product_params": params,
            "declared_model_format": "fbx" if model_download_url else None,
        }

        return ProductRecord(
            unique_key=self.build_unique_key(final_url, None),
            source_site=self.site_name,
            source_url=source_url,
            parsed_at=self.now_utc_iso(),
            external_id=params.get("Артикул") or self.extract_article_from_title(title),
            category_raw=category_raw,
            category_norm=self.classify_timotrader_category(title, category_raw),
            title=title,
            brand="TIMO",
            collection=collection,
            product_url=final_url,
            model_link_type="direct_file" if model_download_url else None,
            model_page_url=final_url,
            model_download_url=model_download_url,
            model_download_landing_url=None,
            model_vendor_url=final_url,
            model_extraction_method="timotrader_product_page_fbx_link",
            model_download_filename=self.filename_from_url(model_download_url),
            model_format=self.ext_from_url(model_download_url),
            price_value=price_value,
            price_currency=price_currency,
            old_price_value=old_price_value,
            style=None,
            color=color,
            description=description,
            width_cm=width_cm,
            depth_cm=depth_cm,
            height_cm=height_cm,
            weight_kg=None,
            room="bathroom",
            materials=materials,
            availability=availability,
            country_brand=country_brand,
            production_country=production_country,
            tags_json=json.dumps([x for x in [collection, color, category_raw] if x], ensure_ascii=False),
            images_json=json.dumps(images, ensure_ascii=False),
            related_json="[]",
            extra_json=json.dumps(extra, ensure_ascii=False),
            raw_html=str(soup),
        )

    def extract_title(self, soup: BeautifulSoup) -> Optional[str]:
        for selector in ("h1", ".tm-product-title", ".uk-article-title"):
            node = soup.select_one(selector)
            if not node:
                continue
            text = self.norm_space(node.get_text(" ", strip=True))
            if text:
                return self.clean_title(text)

        og = self.extract_meta_content(soup, "og:title")
        if og:
            return self.clean_title(og)

        return self.clean_title(self.extract_name_from_jsonld(soup))  # pragma: no cover

    @staticmethod
    def clean_title(text: Optional[str]) -> Optional[str]:
        if not text:
            return None  # pragma: no cover
        text = SupplierAdapter.norm_space(text)
        text = re.sub(r"\s*\|\s*.*Timo.*$", "", text, flags=re.IGNORECASE)
        return text or None

    def extract_description(self, soup: BeautifulSoup) -> Optional[str]:
        panel = soup.select_one(".uk-flex-first\\@m .uk-panel") or soup.select_one(".uk-panel")
        if panel:
            paragraphs = [self.norm_space(p.get_text(" ", strip=True)) for p in panel.find_all("p", recursive=False)]
            text = "\n".join(p for p in paragraphs if p)
            if text:
                return text
        return self.extract_meta_content(soup, "description")

    def extract_params(self, soup: BeautifulSoup) -> dict[str, str]:
        params: dict[str, str] = {}
        for row in soup.select("dl.tm-deflist"):
            key_node = row.select_one("dt")
            value_node = row.select_one("dd")
            if not key_node or not value_node:
                continue  # pragma: no cover
            key = self.norm_space(key_node.get_text(" ", strip=True)).rstrip(":")
            value = self.norm_space(value_node.get_text(" ", strip=True))
            if key:
                params[key] = value
        return params

    def extract_breadcrumb_category(self, soup: BeautifulSoup) -> Optional[str]:
        parts: list[str] = []
        for node in soup.select(".uk-breadcrumb a, .breadcrumb a, .breadcrumbs a"):
            text = self.norm_space(node.get_text(" ", strip=True))
            if text and text.lower() not in {"главная", "каталог"}:
                parts.append(text)
        return " > ".join(parts) if parts else None

    def extract_fbx_download_url(self, soup: BeautifulSoup, base_url: str) -> Optional[str]:
        candidates: list[tuple[int, str]] = []
        for a in soup.select("a[href]"):
            href = str(a.get("href") or "").strip()
            if not href:
                continue  # pragma: no cover
            text = self.norm_space(a.get_text(" ", strip=True)).lower()
            href_l = href.lower()
            score = 0
            if ".fbx" in href_l:
                score += 10
            if "fbx" in text:
                score += 5
            if a.has_attr("download"):
                score += 2
            if score:
                candidates.append((score, urljoin(base_url, href)))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (-item[0], item[1]))
        return candidates[0][1]

    def extract_product_images(self, soup: BeautifulSoup, base_url: str) -> list[str]:
        urls: list[str] = []
        for selector in (".tm-product-gallery img", ".uk-slideshow-items img", ".tm-media-box img", "meta[property='og:image']"):
            for node in soup.select(selector):
                src = node.get("content") or node.get("data-src") or node.get("src")
                if src:
                    urls.append(urljoin(base_url, src))
        return self.unique_keep_order(urls)[:50]

    def extract_card_images(self, card, base_url: str) -> list[str]:
        urls = []
        for img in card.select("img"):
            src = img.get("data-src") or img.get("src")
            if src:
                urls.append(urljoin(base_url, src))
        return self.unique_keep_order(urls)[:5]

    def extract_price_from_node(self, node) -> tuple[Optional[float], Optional[float], Optional[str]]:
        price_node = node.select_one(".tm-product-price, .shk-price, [itemprop='price']")
        if price_node:
            parsed = self.parse_price_rub(self.norm_space(price_node.get_text(" ", strip=True)))
            if parsed[0] is not None:
                return parsed

        text = self.norm_space(node.get_text(" ", strip=True))
        return self.parse_price_rub(text)

    def extract_availability(self, soup: BeautifulSoup) -> Optional[str]:
        stock = soup.select_one(".tm-product-stock")
        if stock:
            return self.norm_space(stock.get_text(" ", strip=True)) or None
        text = self.norm_space(soup.get_text(" ", strip=True)).lower()
        if "есть в наличии" in text:
            return "Есть в наличии"  # pragma: no cover
        if "скоро в продаже" in text:
            return "Скоро в продаже"
        return None  # pragma: no cover

    def extract_card_availability(self, card) -> Optional[str]:
        stock = card.select_one(".tm-product-stock")
        if not stock:
            return None
        return self.norm_space(stock.get_text(" ", strip=True)) or None

    @staticmethod
    def extract_hidden_value(node, name: str) -> Optional[str]:
        field = node.select_one(f"input[name='{name}']")
        if not field:
            return None
        value = str(field.get("value") or "").strip()
        return value or None

    @staticmethod
    def unique_keep_order(values: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            out.append(value)
        return out

    @staticmethod
    def extract_collection_from_title(title: Optional[str]) -> Optional[str]:
        if not title:
            return None  # pragma: no cover
        match = re.search(r"\b(Petruma|Saona|Selene|Tetra|Torne|Unari|Adelia|Anni|Arisa|Beverly|Briana|Helmi|Lina|Luiro|Morea|Nelson)\b", title, re.IGNORECASE)
        if not match:
            return None
        return match.group(1)

    @staticmethod
    def extract_article_from_title(title: Optional[str]) -> Optional[str]:
        if not title:
            return None  # pragma: no cover
        match = re.search(r"\b([A-ZА-Я]{1,4}[-\s]?\d{3,5}(?:/\d{2}[A-ZА-Я]*)?)\b", title, re.IGNORECASE)
        return match.group(1) if match else None

    @staticmethod
    def extract_color_from_title(title: Optional[str]) -> Optional[str]:
        if not title:
            return None  # pragma: no cover
        match = re.search(r"/[0-9]{2}[A-ZА-Я]*\s+(.+)$", title, re.IGNORECASE)
        if match:
            return SupplierAdapter.norm_space(match.group(1))
        for color in ("Хром", "Черный", "Никель", "Золото матовое", "Черное золото", "Белый матовый", "Розовое золото", "Золото шлифованное"):
            if color.lower() in title.lower():
                return color
        return None  # pragma: no cover

    def extract_color(self, params: dict[str, str], title: Optional[str]) -> Optional[str]:
        for key in ("Цвет", "Цвет душевой системы", "Цвет смесителя", "Цвет корпуса"):
            value = params.get(key)
            if value:
                return value
        return self.extract_color_from_title(title)

    @staticmethod
    def extract_materials(params: dict[str, str]) -> Optional[str]:
        parts = []
        for key, value in params.items():
            if "материал" in key.lower() and value:
                parts.append(f"{key}: {value}")
        return "; ".join(parts) if parts else None

    @classmethod
    def extract_dimensions(cls, params: dict[str, str]) -> tuple[Optional[float], Optional[float], Optional[float]]:
        width = cls.parse_dimension_value(params.get("Ширина"))
        depth = cls.parse_dimension_value(params.get("Глубина") or params.get("Длина"))
        height = cls.parse_dimension_value(params.get("Высота"))
        diameter = cls.parse_dimension_value(params.get("Диаметр") or params.get("Размер верхнего душа"))
        if diameter is not None:
            width = width if width is not None else diameter
            depth = depth if depth is not None else diameter
        return width, depth, height

    @staticmethod
    def parse_dimension_value(value: Optional[str]) -> Optional[float]:
        if not value:
            return None
        match = re.search(r"([0-9]+(?:[.,][0-9]+)?)", value)
        if not match:
            return None
        try:
            parsed = float(match.group(1).replace(",", "."))
        except Exception:
            return None
        if "мм" in value.lower():
            return parsed / 10.0
        return parsed

    def classify_timotrader_category(self, title: Optional[str], category_raw: Optional[str]) -> Optional[str]:
        text = " ".join(x for x in [title, category_raw] if x).lower().replace("ё", "е")
        if any(x in text for x in ("душ", "лейка", "смесител")):
            return "bath_fixture"
        if any(x in text for x in ("бумагодержател", "дозатор", "аксессуар")):
            return "bath_accessory"
        if "трап" in text:
            return "drain"
        return self.classify_category(title)
