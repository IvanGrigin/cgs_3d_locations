# -*- coding: utf-8 -*-
"""
Sancos product adapter.

The site exposes direct model links from the product card itself. In practice
the model can be either a direct FBX file or a ZIP archive with FBX/MAX inside.
The product page also contains a structured parameter block with dimensions,
article, package dimensions, delivery volume and both net/gross weights.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from src.suppliers.models import ProductRecord

from .base import SupplierAdapter


class SancosAdapter(SupplierAdapter):
    site_name = "sancos"
    empty_parse_is_skip = True

    def can_handle(self, url: str) -> bool:
        host = urlparse(url).netloc.lower()
        return host == "sancos.su" or host == "www.sancos.su"

    def parse(self, url: str, html: str, final_url: str) -> list[ProductRecord]:
        soup = BeautifulSoup(html, "html.parser")
        product = self.parse_product_page(url, soup, final_url)
        if not self.is_real_product_record(product):
            return []
        return [product]

    def parse_product_page(self, source_url: str, soup: BeautifulSoup, final_url: str) -> ProductRecord:
        params = self.extract_params(soup)

        title = self.extract_title(soup)
        description = self.extract_description(soup)
        collection = params.get("Коллекция")
        external_id = params.get("Артикул")
        category_raw = self.extract_breadcrumb_category(soup)
        category_norm = self.classify_category(title or category_raw)
        color = self.extract_color(soup)
        model_download_url = self.extract_labeled_link(soup, final_url, "3D модель")
        scheme_url = self.extract_labeled_link(soup, final_url, "Схема")
        brochure_url = self.extract_labeled_link(soup, final_url, "Брошюра")
        images = self.extract_gallery_images(soup, final_url)
        width_cm = self._mm_to_cm(params.get("Ширина (мм)"))
        depth_cm = self._mm_to_cm(params.get("Глубина (мм)"))
        height_cm = self._mm_to_cm(params.get("Высота (мм)"))
        weight_kg = self._to_float(params.get("Вес товара без упаковки (кг)"))
        packed_weight_kg = self._to_float(params.get("Вес товара в упаковке (кг)"))
        volume_m3 = self._to_float(params.get("Объем для доставки (м3)"))
        package_width_cm = self._mm_to_cm(params.get("Ширина упаковки (мм)"))
        package_depth_cm = self._mm_to_cm(params.get("Глубина упаковки (мм)"))
        package_height_cm = self._mm_to_cm(params.get("Высота упаковки (мм)"))
        country_brand = params.get("Страна бренда")
        availability = "available" if model_download_url else None
        materials = self.extract_materials(params)

        model_filename = self.filename_from_url(model_download_url)
        model_format = self.ext_from_url(model_download_url)
        model_kind = self._infer_model_kind(model_download_url)
        collection_slug = self._normalize_token(collection)
        model_stem = self._normalize_token(Path(model_filename).stem if model_filename else "")
        collection_level_archive = bool(
            model_format == ".zip"
            and collection_slug
            and model_stem
            and collection_slug in model_stem
            and (not external_id or self._normalize_token(external_id) not in model_stem)
        )

        extra = {
            "parse_stage": "product",
            "product_params": params,
            "brochure_url": brochure_url,
            "model_kind": model_kind,
            "collection_level_archive": collection_level_archive,
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
            brand="Sancos",
            collection=collection,
            product_url=final_url,
            model_link_type="direct_file" if model_download_url else None,
            model_page_url=final_url,
            model_download_url=model_download_url,
            model_download_landing_url=None,
            model_vendor_url=final_url,
            model_extraction_method="sancos_product_page_direct_link",
            model_download_filename=model_filename,
            model_format=model_format,
            price_value=None,
            price_currency=None,
            old_price_value=None,
            style=None,
            color=color,
            description=description,
            width_cm=width_cm,
            depth_cm=depth_cm,
            height_cm=height_cm,
            weight_kg=weight_kg,
            volume_m3=volume_m3,
            package_width_cm=package_width_cm,
            package_depth_cm=package_depth_cm,
            package_height_cm=package_height_cm,
            packed_weight_kg=packed_weight_kg,
            scheme_url=scheme_url,
            room=self.infer_room_bucket(category_raw, title, description, final_url),
            materials=materials,
            availability=availability,
            country_brand=country_brand,
            production_country=None,
            tags_json=json.dumps(
                [x for x in [collection, color, params.get("Материал корпуса"), params.get("Материал фасада")] if x],
                ensure_ascii=False,
            ),
            images_json=json.dumps(images, ensure_ascii=False),
            related_json="[]",
            extra_json=json.dumps(extra, ensure_ascii=False),
            raw_html=str(soup),
        )

    @staticmethod
    def is_real_product_record(product: ProductRecord) -> bool:
        title = str(product.title or "").strip().lower()
        if title == "продукция по типологии":
            return False
        if product.model_download_url:
            return True
        if product.external_id:
            return True
        if product.width_cm is not None and product.depth_cm is not None and product.height_cm is not None:
            return True
        if product.collection and product.description:
            return True
        return False

    @staticmethod
    def infer_room_bucket(
        category_raw: str | None,
        title: str | None,
        description: str | None,
        final_url: str | None,
    ) -> str | None:
        text = " ".join(x for x in [category_raw, title, description, final_url] if x)
        text = text.lower().replace("ё", "е")

        def has_any(*needles: str) -> bool:
            return any(needle in text for needle in needles)

        if has_any("кухн", "kitchen", "мойк", "sink", "дозатор для кухонной мойки"):
            return "kitchen"
        if has_any("ванн", "bathroom", "душ", "shower", "зеркал", "mirror", "смесител", "аксессуар", "bath", "toilet"):
            return "bathroom"
        return None

    def extract_title(self, soup: BeautifulSoup) -> Optional[str]:
        for selector in (".product__info__title", "h1.main-title", "title"):
            node = soup.select_one(selector)
            if not node:
                continue
            text = self.norm_space(node.get_text(" ", strip=True))
            if text:
                return text
        return None

    def extract_description(self, soup: BeautifulSoup) -> Optional[str]:
        node = soup.select_one(".product__info__text")
        if not node:
            return self.extract_meta_content(soup, "description")
        text = self.norm_space(node.get_text("\n", strip=True))
        return text or None

    def extract_params(self, soup: BeautifulSoup) -> dict[str, str]:
        out: dict[str, str] = {}
        for row in soup.select(".product-tab__param"):
            key_node = row.select_one(".product-tab__key")
            value_node = row.select_one(".product-tab__value")
            if not key_node or not value_node:
                continue
            key = self.norm_space(key_node.get_text(" ", strip=True)).rstrip(":")
            value = self.norm_space(value_node.get_text(" ", strip=True))
            if key:
                out[key] = value
        return out

    def extract_color(self, soup: BeautifulSoup) -> Optional[str]:
        active = soup.select_one(".product__info__colors__item.active .product__info__colors__item__title")
        if active:
            text = self.norm_space(active.get_text(" ", strip=True))
            if text:
                return text
        any_color = soup.select_one(".product__info__colors__item__title")
        if any_color:
            text = self.norm_space(any_color.get_text(" ", strip=True))
            if text:
                return text
        return None

    def extract_labeled_link(self, soup: BeautifulSoup, base_url: str, label_text: str) -> Optional[str]:
        for link in soup.select(".product__info__links a[href]"):
            text = self.norm_space(link.get_text(" ", strip=True))
            if label_text.lower() not in text.lower():
                continue
            href = str(link.get("href") or "").strip()
            if href:
                return urljoin(base_url, href)
        return None

    def extract_gallery_images(self, soup: BeautifulSoup, base_url: str) -> list[str]:
        urls: list[str] = []
        seen: set[str] = set()
        for link in soup.select(".product__gallery a[href]"):
            href = str(link.get("href") or "").strip()
            if not href:
                continue
            absolute = urljoin(base_url, href)
            if absolute in seen:
                continue
            seen.add(absolute)
            urls.append(absolute)
        return urls

    def extract_breadcrumb_category(self, soup: BeautifulSoup) -> Optional[str]:
        parts: list[str] = []
        for node in soup.select(".breadcrumbs .bx-breadcrumb-item a span"):
            text = self.norm_space(node.get_text(" ", strip=True))
            if text and text.lower() not in {"главная", "каталог"}:
                parts.append(text)
        return " > ".join(parts) if parts else None

    @staticmethod
    def extract_materials(params: dict[str, str]) -> Optional[str]:
        parts: list[str] = []
        for key in ("Материал корпуса", "Материал фасада", "Покрытие корпуса", "Покрытие фасада"):
            value = str(params.get(key) or "").strip()
            if value:
                parts.append(f"{key}: {value}")
        return "; ".join(parts) if parts else None

    @staticmethod
    def _to_float(value: str | None) -> Optional[float]:
        text = str(value or "").strip().replace(",", ".")
        if not text:
            return None
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        if not match:
            return None
        try:
            return float(match.group(0))
        except Exception:
            return None

    @classmethod
    def _mm_to_cm(cls, value: str | None) -> Optional[float]:
        parsed = cls._to_float(value)
        if parsed is None:
            return None
        return parsed / 10.0

    @staticmethod
    def _normalize_token(value: str | None) -> str:
        text = str(value or "").strip().lower()
        text = re.sub(r"[^a-z0-9а-яё]+", "_", text, flags=re.IGNORECASE)
        text = re.sub(r"_+", "_", text).strip("_")
        return text

    @staticmethod
    def _infer_model_kind(model_download_url: str | None) -> str | None:
        ext = Path(str(model_download_url or "")).suffix.lower()
        if not ext:
            return None
        if ext == ".zip":
            return "archive"
        if ext in {".fbx", ".obj", ".glb", ".gltf", ".blend", ".max"}:
            return "direct_model"
        return "file"
