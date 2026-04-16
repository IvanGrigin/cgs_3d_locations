# -*- coding: utf-8 -*-
"""
This adapter parses 3ddd product pages and the product JSON API payload.
It prefers the API path when available and falls back to HTML extraction.
The records intentionally mark model downloads as auth-gated instead of direct.
This keeps metadata collection separate from the later authenticated download flow.
Keep API and HTML field mapping aligned.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from src.suppliers.adapters.base import DEFAULT_HEADERS, SupplierAdapter
from src.suppliers.models import ProductRecord


class ThreeDDDAdapter(SupplierAdapter):
    site_name = "3ddd"
    product_api_url = "https://models.3ddd.ru/api/models/show"
    image_host = "https://b5.3ddd.ru/"

    def can_handle(self, url: str) -> bool:
        return "3ddd.ru" in url.lower()

    def fetch_html(self, url: str) -> tuple[str, str]:
        slug = self._extract_slug_from_url(url)
        if slug:
            try:
                return self._fetch_product_json(url, slug)
            except Exception:
                pass
        return super().fetch_html(url)

    def parse(self, url: str, html: str, final_url: str) -> list[ProductRecord]:
        payload = self._load_json_payload(html)
        if payload:
            return [self._build_record_from_json(url, final_url, html, payload)]

        soup = BeautifulSoup(html, "html.parser")

        title = self._extract_title(soup)
        category_raw = self._extract_category(soup)
        author = self._extract_author(soup)
        status = self._extract_status(soup)
        royalty_free = self._extract_royalty_free(soup)

        table = self._extract_info_table(soup)

        platform = table.get("Платформа")
        render = table.get("Рендер")
        archive_size_mb = self._parse_size_mb(table.get("Размер"))
        style = table.get("Стиль")
        materials_table = table.get("Материалы")
        form_factor = table.get("Форма")

        description_html, description_text = self._extract_description(soup)

        length_cm = self._parse_cm_value(table.get("Длина")) or self._parse_named_cm(description_text, "Длина")
        table_width_cm = self._parse_cm_value(table.get("Ширина")) or self._parse_named_cm(description_text, "Ширина")
        depth_cm = self._parse_cm_value(table.get("Глубина")) or self._parse_named_cm(description_text, "Глубина")
        height_cm = self._parse_cm_value(table.get("Высота")) or self._parse_named_cm(description_text, "Высота")

        width_cm = length_cm or table_width_cm
        if depth_cm is None and length_cm is not None:
            depth_cm = table_width_cm
        if width_cm is None:
            width_cm = table_width_cm

        seat_height_cm = self._parse_named_cm(description_text, "Высота посадки")
        seat_depth_cm = self._parse_named_cm(description_text, "Глубина посадки")

        source_product_url = self._extract_source_link_from_description(soup, description_html, description_text)
        archive_formats = self._extract_archive_formats(description_text)
        images = self._extract_images(soup, final_url)
        published_at = self._extract_published_date(soup)

        category_norm = self.classify_category(title)

        extra = {
            "author": author,
            "status": status,
            "royalty_free": royalty_free,
            "platform": platform,
            "render": render,
            "archive_size_mb": archive_size_mb,
            "form_factor": form_factor,
            "seat_height_cm": seat_height_cm,
            "seat_depth_cm": seat_depth_cm,
            "archive_formats": archive_formats,
            "published_date": published_at,
            "source_product_url": source_product_url,
            "parse_stage": "product",
            "download_requires_auth": True,
        }

        record = ProductRecord(
            unique_key=self.build_unique_key(final_url, None),
            source_site=self.site_name,
            source_url=url,
            parsed_at=self.now_utc_iso(),
            external_id=self._extract_external_id(final_url),
            category_raw=category_raw,
            category_norm=category_norm,
            title=title,
            brand=author,
            collection=None,
            product_url=final_url,
            model_link_type="button_requires_auth",
            model_page_url=final_url,
            model_download_url=None,
            model_download_landing_url=None,
            model_vendor_url=source_product_url or final_url,
            model_extraction_method="3ddd_product_page",
            model_download_filename=None,
            model_format=None,
            price_value=None,
            price_currency=None,
            old_price_value=None,
            style=style,
            color=None,
            description=description_text or None,
            width_cm=width_cm,
            depth_cm=depth_cm,
            height_cm=height_cm,
            weight_kg=None,
            room=None,
            materials=materials_table,
            availability=status,
            country_brand=None,
            production_country=None,
            tags_json=json.dumps(
                [x for x in [status, style, *(archive_formats or [])] if x],
                ensure_ascii=False,
            ),
            images_json=json.dumps(images, ensure_ascii=False),
            related_json="[]",
            extra_json=json.dumps(extra, ensure_ascii=False),
            raw_html=html,
        )

        return [record]

    def _fetch_product_json(self, url: str, slug: str) -> tuple[str, str]:
        response = requests.post(
            self.product_api_url,
            headers={
                **DEFAULT_HEADERS,
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            json={"slug": slug},
            timeout=(10, self.timeout),
        )
        response.raise_for_status()

        payload = response.json()
        data = payload.get("data")
        if not isinstance(data, dict) or not data:
            raise RuntimeError(f"3ddd product API returned unexpected payload for slug={slug}")

        return json.dumps(payload, ensure_ascii=False), url

    def _load_json_payload(self, raw: str) -> Optional[dict[str, Any]]:
        text = raw.lstrip()
        if not text.startswith("{"):
            return None

        try:
            payload = json.loads(raw)
        except Exception:
            return None

        if not isinstance(payload, dict):
            return None

        data = payload.get("data")
        if not isinstance(data, dict):
            return None

        return payload

    def _build_record_from_json(
        self,
        url: str,
        final_url: str,
        raw_payload: str,
        payload: dict[str, Any],
    ) -> ProductRecord:
        data = payload.get("data") or {}

        title = self.norm_space(str(data.get("title") or data.get("titleEn") or "")) or None
        category_raw = self._build_category_raw_from_json(data)
        author = self._extract_author_from_json(data)
        status = self._map_status_from_json(data)
        style = self.norm_space(str(data.get("style") or "")) or None
        description_text = self.norm_space(str(data.get("description") or "")) or None

        width_cm = self._to_float(data.get("length"))
        depth_cm = self._to_float(data.get("width"))
        height_cm = self._to_float(data.get("height"))

        seat_height_cm = self._parse_named_cm(description_text or "", "Высота сиденья")
        seat_depth_cm = self._parse_named_cm(description_text or "", "Глубина сиденья")
        archive_size_mb = self._size_kb_to_mb(data.get("size_kb"))
        source_product_url = self._extract_source_link_from_text(description_text or "")
        archive_formats = self._extract_archive_formats_from_json(data)
        images = self._extract_images_from_json(data)
        published_at = self._extract_published_date_from_json(data)
        materials = self._extract_materials_from_json(data)
        platform = self._get_nested_text(data, "platform", "title")
        render = self._get_nested_text(data, "render", "title")
        form_factor = self._get_nested_text(data, "form", "form")
        color = self._extract_color_from_json(data)
        category_norm = self.classify_category(title or category_raw)

        extra = {
            "author": author,
            "status": status,
            "royalty_free": False,
            "platform": platform,
            "render": render,
            "archive_size_mb": archive_size_mb,
            "form_factor": form_factor,
            "seat_height_cm": seat_height_cm,
            "seat_depth_cm": seat_depth_cm,
            "archive_formats": archive_formats,
            "published_date": published_at,
            "source_product_url": source_product_url,
            "parse_stage": "product_api",
            "download_requires_auth": True,
            "api_slug": data.get("slug"),
            "api_type": data.get("typeText"),
        }

        return ProductRecord(
            unique_key=self.build_unique_key(final_url, None),
            source_site=self.site_name,
            source_url=url,
            parsed_at=self.now_utc_iso(),
            external_id=str(data.get("slug") or self._extract_external_id(final_url) or ""),
            category_raw=category_raw,
            category_norm=category_norm,
            title=title,
            brand=author,
            collection=None,
            product_url=final_url,
            model_link_type="button_requires_auth",
            model_page_url=final_url,
            model_download_url=None,
            model_download_landing_url=None,
            model_vendor_url=source_product_url or final_url,
            model_extraction_method="3ddd_product_api",
            model_download_filename=None,
            model_format=None,
            price_value=None,
            price_currency=None,
            old_price_value=None,
            style=style,
            color=color,
            description=description_text,
            width_cm=width_cm,
            depth_cm=depth_cm,
            height_cm=height_cm,
            weight_kg=None,
            room=None,
            materials=materials,
            availability=status,
            country_brand=None,
            production_country=self._parse_named_text(description_text or "", "Страна производства"),
            tags_json=json.dumps(
                [x for x in [status, style, *(archive_formats or [])] if x],
                ensure_ascii=False,
            ),
            images_json=json.dumps(images, ensure_ascii=False),
            related_json="[]",
            extra_json=json.dumps(extra, ensure_ascii=False),
            raw_html=raw_payload,
        )

    def _extract_title(self, soup: BeautifulSoup) -> Optional[str]:
        node = soup.select_one("span.plate-title[title]")
        if node and node.get("title"):
            return self.norm_space(node["title"])

        node = soup.select_one("h1.title")
        if node:
            return self.norm_space(node.get_text(" ", strip=True))

        if soup.title:
            return self.norm_space(soup.title.get_text(" ", strip=True))

        return None

    def _extract_category(self, soup: BeautifulSoup) -> Optional[str]:
        parts: list[str] = []

        category = soup.select_one(".category span")
        subcategory = soup.select_one(".subcategory span")

        if category:
            parts.append(self.norm_space(category.get_text(" ", strip=True)))
        if subcategory:
            parts.append(self.norm_space(subcategory.get_text(" ", strip=True)))

        return " > ".join(parts) if parts else None

    def _extract_author(self, soup: BeautifulSoup) -> Optional[str]:
        node = soup.select_one(".model-user-name")
        if node:
            return self.norm_space(node.get_text(" ", strip=True))
        return None

    def _extract_status(self, soup: BeautifulSoup) -> Optional[str]:
        for node in soup.select(".status span"):
            text = self.norm_space(node.get_text(" ", strip=True))
            if text:
                return text
        return None

    def _extract_royalty_free(self, soup: BeautifulSoup) -> bool:
        node = soup.select_one(".royalty-free span")
        if not node:
            return False
        return "royalty free" in node.get_text(" ", strip=True).lower()

    def _extract_info_table(self, soup: BeautifulSoup) -> dict[str, str]:
        result: dict[str, str] = {}

        for row in soup.select(".model-info-block table tr"):
            cells = row.find_all("td")
            if len(cells) != 2:
                continue

            key = self.norm_space(cells[0].get_text(" ", strip=True)).rstrip(":")
            value = self.norm_space(cells[1].get_text(" ", strip=True))

            if key:
                result[key] = value

        return result

    def _extract_description(self, soup: BeautifulSoup) -> tuple[str, str]:
        node = soup.select_one(".description > div")
        if not node:
            return "", ""

        html = str(node)
        text = self.norm_space(node.get_text("\n", strip=True))
        return html, text

    def _parse_cm_value(self, value: Optional[str]) -> Optional[float]:
        if not value:
            return None

        m = re.search(r"([0-9]+(?:[.,][0-9]+)?)", value)
        if not m:
            return None

        try:
            return float(m.group(1).replace(",", "."))
        except Exception:
            return None

    def _parse_named_cm(self, text: str, label: str) -> Optional[float]:
        m = re.search(
            rf"{re.escape(label)}\s*:\s*([0-9]+(?:[.,][0-9]+)?)\s*см",
            text,
            flags=re.IGNORECASE,
        )
        if not m:
            return None
        try:
            return float(m.group(1).replace(",", "."))
        except Exception:
            return None

    def _parse_size_mb(self, text: Optional[str]) -> Optional[float]:
        if not text:
            return None
        m = re.search(r"([0-9]+(?:[.,][0-9]+)?)\s*МБ", text, flags=re.IGNORECASE)
        if not m:
            return None
        try:
            return float(m.group(1).replace(",", "."))
        except Exception:
            return None

    def _extract_source_link_from_description(
        self,
        soup: BeautifulSoup,
        description_html: str,
        description_text: str,
    ) -> Optional[str]:
        node = soup.select_one(".description a[href]")
        if node and node.get("href"):
            return node["href"].strip()

        m = re.search(r'href="([^"]+)"', description_html)
        if m:
            return m.group(1).strip()

        return self._extract_source_link_from_text(description_text)

    def _extract_source_link_from_text(self, text: str) -> Optional[str]:
        m = re.search(r"https?://[^\s<]+", text, flags=re.IGNORECASE)
        if not m:
            return None
        return m.group(0).strip()

    def _extract_archive_formats(self, text: str) -> list[str]:
        m = re.search(
            r"Какие форматы в архиве\s*:\s*([^\n\r]+)",
            text,
            flags=re.IGNORECASE,
        )
        if not m:
            return []

        raw = m.group(1)
        parts = [self.norm_space(x).lower() for x in raw.split(",")]
        return [x for x in parts if x]

    def _extract_images(self, soup: BeautifulSoup, base_url: str) -> list[str]:
        out: list[str] = []

        for img in soup.select(".big-view picture source[srcset], .preview img[src]"):
            src = img.get("srcset") or img.get("src")
            if not src:
                continue
            out.append(urljoin(base_url, src.strip()))

        seen = set()
        uniq: list[str] = []
        for item in out:
            if item not in seen:
                seen.add(item)
                uniq.append(item)

        return uniq

    def _extract_published_date(self, soup: BeautifulSoup) -> Optional[str]:
        node = soup.select_one(".publication-date")
        if not node:
            return None

        text = self.norm_space(node.get_text(" ", strip=True))
        m = re.search(
            r"(\d{1,2})\s+([а-яё]+)\s+(\d{4})",
            text,
            flags=re.IGNORECASE,
        )
        if not m:
            return None

        day = int(m.group(1))
        month_name = m.group(2).lower()
        year = int(m.group(3))

        months = {
            "января": 1,
            "февраля": 2,
            "марта": 3,
            "апреля": 4,
            "мая": 5,
            "июня": 6,
            "июля": 7,
            "августа": 8,
            "сентября": 9,
            "октября": 10,
            "ноября": 11,
            "декабря": 12,
        }
        month = months.get(month_name)
        if not month:
            return None

        return f"{year:04d}-{month:02d}-{day:02d}"

    def _extract_external_id(self, final_url: str) -> Optional[str]:
        return self._extract_slug_from_url(final_url)

    def _extract_slug_from_url(self, url: str) -> Optional[str]:
        path = urlparse(url).path
        m = re.search(r"/3dmodels/show/([^/?#]+)", path)
        if not m:
            return None
        return m.group(1)

    def _build_category_raw_from_json(self, data: dict[str, Any]) -> Optional[str]:
        parts: list[str] = []

        category = data.get("category") or {}
        subcategory = data.get("subcategory") or {}

        if isinstance(category, dict):
            title = self.norm_space(str(category.get("title") or ""))
            if title:
                parts.append(title)

        if isinstance(subcategory, dict):
            title = self.norm_space(str(subcategory.get("title") or ""))
            if title:
                parts.append(title)

        return " > ".join(parts) if parts else None

    def _extract_author_from_json(self, data: dict[str, Any]) -> Optional[str]:
        user = data.get("user")
        if not isinstance(user, dict):
            return None

        username = self.norm_space(str(user.get("username") or ""))
        if username:
            return username

        slug = self.norm_space(str(user.get("slug") or ""))
        return slug or None

    def _map_status_from_json(self, data: dict[str, Any]) -> Optional[str]:
        raw = self.norm_space(str(data.get("typeText") or ""))
        if not raw:
            return None

        low = raw.lower()
        if low == "om":
            return "FREE"
        return raw.upper()

    def _extract_archive_formats_from_json(self, data: dict[str, Any]) -> list[str]:
        formats = data.get("formats")
        if not isinstance(formats, list):
            return []

        out: list[str] = []
        for item in formats:
            if not isinstance(item, dict):
                continue
            title = self.norm_space(str(item.get("title") or "")).lower()
            if title:
                out.append(title)
        return out

    def _extract_images_from_json(self, data: dict[str, Any]) -> list[str]:
        images = data.get("images")
        if not isinstance(images, list):
            return []

        out: list[str] = []
        seen: set[str] = set()

        for item in images:
            if not isinstance(item, dict):
                continue

            web_path = str(item.get("webPath") or "").strip().lstrip("/")
            if not web_path:
                continue

            image_url = urljoin(
                self.image_host,
                f"media/cache/tuk_model_custom_filter_ang_ru/{web_path}",
            )
            if image_url in seen:
                continue
            seen.add(image_url)
            out.append(image_url)

        return out

    def _extract_published_date_from_json(self, data: dict[str, Any]) -> Optional[str]:
        created = str(data.get("created") or "").strip()
        if not created:
            return None
        return created.split(" ", 1)[0] or None

    def _extract_materials_from_json(self, data: dict[str, Any]) -> Optional[str]:
        materials = data.get("materials")
        if not isinstance(materials, list):
            return None

        parts: list[str] = []
        for item in materials:
            if not isinstance(item, dict):
                continue
            title = self.norm_space(str(item.get("material") or ""))
            if title:
                parts.append(title)

        return ", ".join(parts) if parts else None

    def _extract_color_from_json(self, data: dict[str, Any]) -> Optional[str]:
        colors = data.get("colors")
        if not isinstance(colors, list):
            return None

        for item in colors:
            if not isinstance(item, dict):
                continue
            title = self.norm_space(str(item.get("title") or ""))
            if title:
                return title
        return None

    def _get_nested_text(self, data: dict[str, Any], key: str, nested_key: str) -> Optional[str]:
        nested = data.get(key)
        if not isinstance(nested, dict):
            return None
        text = self.norm_space(str(nested.get(nested_key) or ""))
        return text or None

    def _size_kb_to_mb(self, value: Any) -> Optional[float]:
        try:
            return round(float(value) / 1024.0, 3)
        except Exception:
            return None

    def _to_float(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(str(value).replace(",", "."))
        except Exception:
            return None

    def _parse_named_text(self, text: str, label: str) -> Optional[str]:
        m = re.search(
            rf"{re.escape(label)}\s*:\s*([^\n\r]+)",
            text,
            flags=re.IGNORECASE,
        )
        if not m:
            return None
        value = self.norm_space(m.group(1))
        return value or None
