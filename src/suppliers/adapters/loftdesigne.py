# -*- coding: utf-8 -*-
"""
This adapter parses Loft Designe product pages into normalized records.
It extracts structured details from modal-like product markup and metadata.
The parser also resolves whether a model link is direct or a Yandex landing page.
Loft Designe often requires careful fallback logic for sparse titles.
Keep model-link classification conservative.
"""
from __future__ import annotations

import json
import re
from typing import Optional
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from src.suppliers.adapters.base import SupplierAdapter
from src.suppliers.models import ProductRecord


class LoftDesigneAdapter(SupplierAdapter):
    site_name = "loftdesigne"
    empty_parse_is_skip = True

    def can_handle(self, url: str) -> bool:
        host = urlparse(url).netloc.lower()
        return "loftdesigne.ru" in host

    def parse(self, url: str, html: str, final_url: str) -> list[ProductRecord]:
        record = self.parse_product(url, html, final_url)
        if not (record.model_download_landing_url or record.model_download_url):
            return []
        return [record]

    def parse_product(self, source_url: str, html: str, final_url: str) -> ProductRecord:
        soup = BeautifulSoup(html, "html.parser")
        product = soup.select_one("[data-product-card-full][data-product-card]") or soup.select_one(".product-modal")

        title = self.extract_product_title(product, soup)
        external_id = self.extract_external_id(product)
        category_raw = self.extract_category(product, soup)
        category_norm = self.classify_category(title or category_raw)

        price_value, old_price_value, price_currency = self.extract_price(product, soup)
        width_cm, depth_cm, height_cm = self.extract_dimensions(product)
        materials = self.extract_detail_value(product, "Материал")
        color = self.extract_detail_value(product, "Цвет")
        style = self.extract_detail_value(product, "Стиль")
        availability = self.extract_availability(product)
        images = self.extract_product_images(product, final_url)
        related: list[dict] = []
        tags = self.extract_tags(category_raw, color, materials)

        model_href = self.extract_model_href(product, soup, final_url)
        model_link_type = None
        model_download_url = None
        model_download_landing_url = None
        model_download_filename = None
        model_format = None
        model_extraction_method = None

        if model_href:
            lower_href = model_href.lower()
            if "disk.yandex.ru" in lower_href:
                model_link_type = "landing_page"
                model_download_landing_url = model_href
                model_extraction_method = "vendor_page_to_yadisk"
            else:
                model_link_type = "direct_file"
                model_download_url = model_href
                model_download_filename = self.filename_from_url(model_href)
                model_format = self.ext_from_url(model_href)
                model_extraction_method = "vendor_page_direct_model_link"

        description, description_source = self.extract_description(
            product=product,
            soup=soup,
            title=title,
            category_raw=category_raw,
            materials=materials,
            color=color,
            width_cm=width_cm,
            depth_cm=depth_cm,
            height_cm=height_cm,
            availability=availability,
        )

        brand = self.extract_brand(product, soup)
        collection = self.extract_detail_value(product, "Коллекция")
        country_brand = self.extract_detail_value(product, "Страна бренда")
        production_country = self.extract_detail_value(product, "Страна производства")
        weight_kg = self.extract_numeric_detail(product, ["Вес", "Вес в упаковке"])
        room = None

        extra = {
            "parse_stage": "product",
            "description_source": description_source,
            "data_product_category": product.get("data-product-category") if product else None,
            "data_product_name": product.get("data-product-name") if product else None,
            "data_product_price_rubles": product.get("data-product-price-rubles") if product else None,
        }

        return ProductRecord(
            unique_key=self.build_unique_key(final_url, external_id),
            source_site=self.site_name,
            source_url=source_url,
            parsed_at=self.now_utc_iso(),
            external_id=external_id,
            category_raw=category_raw,
            category_norm=category_norm,
            title=title,
            brand=brand,
            collection=collection,
            product_url=final_url,
            model_link_type=model_link_type,
            model_page_url=final_url if model_href else None,
            model_download_url=model_download_url,
            model_download_landing_url=model_download_landing_url,
            model_vendor_url=final_url if model_href else None,
            model_extraction_method=model_extraction_method,
            model_download_filename=model_download_filename,
            model_format=model_format,
            price_value=price_value,
            price_currency=price_currency,
            old_price_value=old_price_value,
            style=style,
            color=color,
            description=description,
            width_cm=width_cm,
            depth_cm=depth_cm,
            height_cm=height_cm,
            weight_kg=weight_kg,
            room=room,
            materials=materials,
            availability=availability,
            country_brand=country_brand,
            production_country=production_country,
            tags_json=json.dumps(tags, ensure_ascii=False),
            images_json=json.dumps(images, ensure_ascii=False),
            related_json=json.dumps(related, ensure_ascii=False),
            extra_json=json.dumps(extra, ensure_ascii=False),
            raw_html=html,
        )

    @staticmethod
    def _clean_title(text: str | None) -> str | None:
        if not text:
            return None
        value = SupplierAdapter.norm_space(text)
        value = re.sub(r"\s*[-|–—]\s*Loft Designe.*$", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\s*[-|–—]\s*Купить.*$", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\s*[-|–—]\s*цена.*$", "", value, flags=re.IGNORECASE)
        return value or None

    def extract_product_title(self, product, soup: BeautifulSoup) -> Optional[str]:
        if product:
            node = product.select_one(".product-modal__name")
            if node:
                text = self._clean_title(node.get_text(" ", strip=True))
                if text:
                    return text

            data_name = self.norm_space(product.get("data-product-name", ""))
            if data_name:
                cleaned = self._clean_title(data_name)
                if cleaned and not re.fullmatch(r"[\w\-]+\s+model", cleaned, flags=re.IGNORECASE):
                    return cleaned

        og_title = self.extract_meta_content(soup, "og:title")
        if og_title:
            cleaned = self._clean_title(og_title)
            if cleaned:
                return cleaned

        name = self.extract_name_from_jsonld(soup)
        if name:
            cleaned = self._clean_title(name)
            if cleaned:
                return cleaned

        h1 = soup.find("h1")
        if h1:
            cleaned = self._clean_title(h1.get_text(" ", strip=True))
            if cleaned:
                return cleaned

        return None

    def extract_external_id(self, product) -> Optional[str]:
        if product:
            node = product.select_one(".product-modal__id")
            if node:
                text = self.norm_space(node.get_text(" ", strip=True))
                if text:
                    return text

            data_name = self.norm_space(product.get("data-product-name", ""))
            if data_name:
                return data_name

        return None

    def extract_brand(self, product, soup: BeautifulSoup) -> Optional[str]:
        brand = self.extract_detail_value(product, "Бренд")
        if brand:
            return brand

        brand = self.extract_brand_from_jsonld(soup)
        if brand:
            return brand

        return "Loft Designe"

    def extract_category(self, product, soup: BeautifulSoup) -> Optional[str]:
        if product:
            category = self.norm_space(product.get("data-product-category", ""))
            if category:
                return category

        crumbs = [
            self.norm_space(node.get_text(" ", strip=True))
            for node in soup.select(".catalog__breadcrumbs a, .catalog__breadcrumbs span")
            if self.norm_space(node.get_text(" ", strip=True))
        ]
        if crumbs:
            return " > ".join(crumbs)

        return None

    def extract_price(self, product, soup: BeautifulSoup) -> tuple[Optional[float], Optional[float], Optional[str]]:
        if product:
            raw_price = self.norm_space(product.get("data-product-price-rubles", ""))
            if raw_price:
                try:
                    return float(raw_price.replace(" ", "").replace(",", ".")), None, "RUB"
                except Exception:
                    pass

            node = product.select_one(".product-modal__price")
            if node:
                parsed = self.parse_price_rub(node.get_text(" ", strip=True))
                if parsed[0] is not None:
                    return parsed

        for item in self.extract_jsonld_objects(soup):
            offers = item.get("offers")
            if not isinstance(offers, dict):
                continue
            price = offers.get("price") or offers.get("lowPrice")
            if price is None:
                continue
            try:
                return float(str(price).replace(",", ".")), None, str(offers.get("priceCurrency") or "RUB")
            except Exception:
                continue

        return self.parse_price_rub(soup.get_text(" ", strip=True))

    def extract_description(
        self,
        *,
        product,
        soup: BeautifulSoup,
        title: Optional[str],
        category_raw: Optional[str],
        materials: Optional[str],
        color: Optional[str],
        width_cm: Optional[float],
        depth_cm: Optional[float],
        height_cm: Optional[float],
        availability: Optional[str],
    ) -> tuple[Optional[str], str]:
        selectors = [
            "[itemprop='description']",
            ".product-modal__description",
            ".product-modal__text",
            ".product-modal__content",
            ".product-description",
            ".catalog-product__description",
            ".seo-text",
        ]
        scope = product or soup
        for selector in selectors:
            node = scope.select_one(selector)
            if not node:
                continue
            text = self.norm_space(node.get_text(" ", strip=True))
            if text:
                return text, "dom"

        og_description = self.extract_meta_content(soup, "og:description")
        if og_description:
            return og_description, "meta_og"

        for item in self.extract_jsonld_objects(soup):
            description = item.get("description")
            if isinstance(description, str):
                text = self.norm_space(description)
                if text:
                    return text, "jsonld"

        parts = []
        if title:
            parts.append(title)
        if category_raw:
            parts.append(f"Категория: {category_raw}")
        if materials:
            parts.append(f"Материал: {materials}")
        if color:
            parts.append(f"Цвет: {color}")
        if width_cm is not None and depth_cm is not None and height_cm is not None:
            parts.append(
                f"Размеры: {self._format_measure(width_cm)} x {self._format_measure(depth_cm)} x {self._format_measure(height_cm)} см"
            )
        elif height_cm is not None:
            parts.append(f"Высота: {self._format_measure(height_cm)} см")
        if availability:
            parts.append(f"Наличие: {availability}")

        if parts:
            return ". ".join(parts), "structured_fields_fallback"

        return None, "missing"

    def extract_availability(self, product) -> Optional[str]:
        if not product:
            return None

        parts = []
        for node in product.select(".product-modal__status"):
            text = self.norm_space(node.get_text(" ", strip=True))
            if text:
                parts.append(text)

        if not parts:
            return None
        return "; ".join(parts)

    def extract_dimensions(self, product) -> tuple[Optional[float], Optional[float], Optional[float]]:
        width_cm = self.extract_numeric_detail(product, ["Ширина"])
        depth_cm = self.extract_numeric_detail(product, ["Глубина", "Длина"])
        height_cm = self.extract_numeric_detail(product, ["Высота"])
        diameter_cm = self.extract_numeric_detail(product, ["Диаметр"])

        if diameter_cm is not None:
            if width_cm is None:
                width_cm = diameter_cm
            if depth_cm is None:
                depth_cm = diameter_cm

        return width_cm, depth_cm, height_cm

    def extract_numeric_detail(self, product, labels: list[str]) -> Optional[float]:
        value = None
        for label in labels:
            value = self.extract_detail_value(product, label)
            parsed = self.parse_numeric_value(value)
            if parsed is not None:
                return parsed
        return None

    def extract_detail_value(self, product, label: str) -> Optional[str]:
        if not product:
            return None

        label_norm = self._normalize_label(label)
        for item in product.select(".product-modal__details-item"):
            label_node = item.select_one(".product-modal__details-label")
            value_node = item.select_one(".product-modal__details-value")
            if not label_node or not value_node:
                continue

            current_label = self._normalize_label(label_node.get_text(" ", strip=True))
            if current_label != label_norm:
                continue

            text = self.norm_space(value_node.get_text(" ", strip=True))
            if text:
                return text

        return None

    def extract_model_href(self, product, soup: BeautifulSoup, base_url: str) -> Optional[str]:
        scope = product or soup
        for a in scope.select("a[href]"):
            href = a.get("href")
            if not href:
                continue

            text = self.norm_space(a.get_text(" ", strip=True)).lower()
            if "скачать 3d-модель" in text or "скачать 3d модель" in text or "скачать модель" in text:
                return urljoin(base_url, href)

        return None

    def extract_product_images(self, product, base_url: str) -> list[str]:
        if not product:
            return []

        out: list[str] = []
        for img in product.select(".product-modal__main-slider img, .product-modal__preview-slider img"):
            src = img.get("src") or img.get("data-src")
            if not src:
                continue
            out.append(urljoin(base_url, src))

        seen = set()
        uniq = []
        for image_url in out:
            if image_url in seen:
                continue
            seen.add(image_url)
            uniq.append(image_url)
        return uniq[:50]

    def extract_tags(self, category_raw: Optional[str], color: Optional[str], materials: Optional[str]) -> list[str]:
        out = []
        for value in (category_raw, color, materials):
            text = self.norm_space(value or "")
            if text and text not in out:
                out.append(text)
        return out

    def parse_numeric_value(self, value: Optional[str]) -> Optional[float]:
        if not value:
            return None
        match = re.search(r"([0-9]+(?:[.,][0-9]+)?)", value)
        if not match:
            return None
        try:
            return float(match.group(1).replace(",", "."))
        except Exception:
            return None

    @staticmethod
    def _normalize_label(text: str) -> str:
        value = SupplierAdapter.norm_space(text).lower().replace("\xa0", " ")
        value = re.sub(r"\s*\([^)]*\)", "", value)
        value = re.sub(r"\s+", " ", value)
        return value.strip(" :")

    @staticmethod
    def _format_measure(value: float) -> str:
        if float(value).is_integer():
            return str(int(value))
        return str(round(float(value), 2)).rstrip("0").rstrip(".")
