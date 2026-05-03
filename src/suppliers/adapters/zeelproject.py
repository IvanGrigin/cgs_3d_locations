# -*- coding: utf-8 -*-
"""
Zeelproject product-page adapter.

The site exposes metadata on a public product page, but download buttons point
to the authenticated accounts domain and require credits. We therefore store
the download target as a gated landing URL instead of a direct file link.
"""
from __future__ import annotations

import json
import re
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from src.suppliers.models import ProductRecord
from src.suppliers.utils import DEFAULT_HEADERS

from .base import SupplierAdapter


class ZeelProjectAdapter(SupplierAdapter):
    site_name = "zeelproject"
    empty_parse_is_skip = True

    def can_handle(self, url: str) -> bool:
        host = urlparse(url).netloc.lower()
        return host == "zeelproject.com" or host == "www.zeelproject.com"

    def fetch_html(self, url: str) -> tuple[str, str]:
        try:
            return self._fetch_html_verified_via_requests(url)
        except Exception:
            return self._fetch_html_verified_via_curl(url)

    def parse(self, url: str, html: str, final_url: str) -> list[ProductRecord]:
        soup = BeautifulSoup(html, "html.parser")
        product = self.parse_product_page(url, soup, final_url)
        if product is None:
            return []
        return [product]

    def parse_product_page(self, source_url: str, soup: BeautifulSoup, final_url: str) -> ProductRecord | None:
        title = self.extract_title(soup)
        brand = self.extract_brand(soup)
        category_raw = self.extract_category(soup)
        materials = self.extract_materials(soup)
        style = self.extract_style(soup)
        description = self.extract_description(soup)
        width_cm, depth_cm, height_cm = self.extract_dimensions(soup)
        model_download_landing_url = self.extract_download_landing_url(soup, final_url)
        sketchup_landing_url = self.extract_sketchup_landing_url(soup, final_url)
        images = self.extract_images(soup, final_url)
        credits_required = self.extract_credit_requirement(soup)
        availability = self.extract_availability(soup)
        price_value, price_currency = self.extract_price(soup)
        if price_value is None and credits_required is not None:
            price_value = float(credits_required)
            price_currency = "CREDIT"
        external_id = self.extract_external_id(final_url)

        if not title or not model_download_landing_url:
            return None

        extra = {
            "parse_stage": "product",
            "download_requires_auth": True,
            "credits_required": credits_required,
            "sketchup_download_landing_url": sketchup_landing_url,
        }

        return ProductRecord(
            unique_key=self.build_unique_key(final_url, external_id),
            source_site=self.site_name,
            source_url=source_url,
            parsed_at=self.now_utc_iso(),
            external_id=external_id,
            category_raw=category_raw,
            category_norm=self.classify_category(title or category_raw),
            title=title,
            brand=brand,
            collection=None,
            product_url=final_url,
            model_link_type="button_requires_auth",
            model_page_url=final_url,
            model_download_url=None,
            model_download_landing_url=model_download_landing_url,
            model_vendor_url=final_url,
            model_extraction_method="zeelproject_product_page",
            model_download_filename=None,
            model_format=None,
            price_value=price_value,
            price_currency=price_currency,
            old_price_value=None,
            style=style,
            color=None,
            description=description,
            width_cm=width_cm,
            depth_cm=depth_cm,
            height_cm=height_cm,
            weight_kg=None,
            room=None,
            materials=materials,
            availability=availability or (f"{credits_required} credit required" if credits_required is not None else None),
            country_brand=None,
            production_country=None,
            tags_json=json.dumps([x for x in [style, *(materials.split(", ") if materials else [])] if x], ensure_ascii=False),
            images_json=json.dumps(images, ensure_ascii=False),
            related_json="[]",
            extra_json=json.dumps(extra, ensure_ascii=False),
            raw_html=str(soup),
        )

    def extract_title(self, soup: BeautifulSoup) -> Optional[str]:
        node = soup.select_one("h1.model_title")
        if node:
            text = self.norm_space(node.get_text(" ", strip=True))
            if text:
                return text
        h1 = soup.select_one("h1")
        if h1:
            text = self.norm_space(h1.get_text(" ", strip=True))
            if text:
                return text
        og = self.extract_meta_content(soup, "og:title")
        if og:
            return self.clean_og_title(og)
        return None

    def extract_brand(self, soup: BeautifulSoup) -> Optional[str]:
        node = soup.select_one(".info_names a.author_link[href]")
        if node:
            text = self.norm_space(node.get_text(" ", strip=True))
            if text:
                return text

        node = soup.select_one(".brand_name a[href*='/brands/']")
        if node:
            text = self.norm_space(node.get_text(" ", strip=True))
            if text:
                return text
        return None

    def extract_category(self, soup: BeautifulSoup) -> Optional[str]:
        parts: list[str] = []
        for node in soup.select(".speedbar [itemprop='name']"):
            text = self.norm_space(node.get_text(" ", strip=True))
            if text and text not in {"zeelproject.com", "3D Models"}:
                parts.append(text)
        if parts:
            return " > ".join(parts[-2:])

        option = self.extract_option_value(soup, "Category", "Категория")
        return option

    def extract_materials(self, soup: BeautifulSoup) -> Optional[str]:
        return self.extract_option_value(soup, "Material", "Материал")

    def extract_style(self, soup: BeautifulSoup) -> Optional[str]:
        return self.extract_option_value(soup, "Style", "Стиль")

    def extract_description(self, soup: BeautifulSoup) -> Optional[str]:
        node = soup.select_one(".full_description p.full_story")
        if not node:
            return None
        text = node.get_text("\n", strip=True)
        lines = [self.norm_space(x) for x in text.splitlines() if self.norm_space(x)]
        description_lines = [
            line
            for line in lines
            if not re.match(r"^(Diameter|Height|Cable height|Ширина|Глубина|Высота)\s*[-:]", line, flags=re.IGNORECASE)
        ]
        if not description_lines:
            return None
        return "\n".join(description_lines)

    def extract_dimensions(self, soup: BeautifulSoup) -> tuple[Optional[float], Optional[float], Optional[float]]:
        node = soup.select_one(".full_description p.full_story")
        text = node.get_text("\n", strip=True) if node else soup.get_text("\n", strip=True)
        diameter = self.parse_dim(text, "Diameter", "Диаметр")
        width = self.parse_dim(text, "Width", "Ширина") or diameter
        depth = self.parse_dim(text, "Depth", "Глубина", "Length", "Длина") or diameter
        height = self.parse_dim(text, "Height", "Высота")
        return width, depth, height

    def extract_download_landing_url(self, soup: BeautifulSoup, base_url: str) -> Optional[str]:
        for a in soup.select(".down_bottons a[href]"):
            title = self.norm_space(a.select_one(".dttl").get_text(" ", strip=True)) if a.select_one(".dttl") else ""
            href = str(a.get("href") or "").strip()
            title_key = title.casefold()
            if href and ("download 3d model" in title_key or "скачать 3d модель" in title_key):
                return urljoin(base_url, href)
        return None

    def extract_sketchup_landing_url(self, soup: BeautifulSoup, base_url: str) -> Optional[str]:
        for a in soup.select(".down_bottons a[href]"):
            title = self.norm_space(a.select_one(".dttl").get_text(" ", strip=True)) if a.select_one(".dttl") else ""
            href = str(a.get("href") or "").strip()
            title_key = title.casefold()
            if href and ("download sketchup" in title_key or "скачать sketchup" in title_key):
                return urljoin(base_url, href)
        return None

    def extract_credit_requirement(self, soup: BeautifulSoup) -> Optional[int]:
        status = self.extract_availability(soup) or ""
        if re.search(r"\bfree\b|бесплат", status, flags=re.IGNORECASE):
            return 0
        text = status or soup.get_text(" ", strip=True)
        m = re.search(r"(\d+)\s+credit\s+required", text, flags=re.IGNORECASE)
        if not m:
            m = re.search(r"требуется\s+(\d+)\s+кредит", text, flags=re.IGNORECASE)
        if not m:
            return None
        try:
            return int(m.group(1))
        except Exception:
            return None

    def extract_availability(self, soup: BeautifulSoup) -> Optional[str]:
        node = soup.select_one(".down_container .down_status")
        if not node:
            return None
        text = self.norm_space(node.get_text(" ", strip=True))
        return text or None

    def extract_price(self, soup: BeautifulSoup) -> tuple[Optional[float], Optional[str]]:
        node = soup.select_one(".product_price .price .nmbr")
        if not node:
            return None, None
        text = self.norm_space(node.get_text(" ", strip=True))
        m = re.search(r"([$€£₽])\s*([0-9]+(?:[.,][0-9]+)?)", text)
        if not m:
            return None, None
        currency_map = {"$": "USD", "€": "EUR", "£": "GBP", "₽": "RUB"}
        try:
            value = float(m.group(2).replace(",", "."))
        except Exception:
            return None, None
        return value, currency_map.get(m.group(1))

    def extract_images(self, soup: BeautifulSoup, base_url: str, limit: int = 50) -> list[str]:
        out: list[str] = []
        for node in soup.select("#slider li img, #slide-pagination li a[href]"):
            src = node.get("href") or node.get("data-original") or node.get("data-src") or node.get("src")
            if not src:
                continue
            out.append(urljoin(base_url, str(src)))
        if not out:
            og_image = self.extract_meta_content(soup, "og:image")
            if og_image:
                out.append(urljoin(base_url, og_image))
        return self.unique_keep_order(out)[:limit]

    @staticmethod
    def extract_external_id(final_url: str) -> Optional[str]:
        m = re.search(r"/(\d+)-", urlparse(final_url).path)
        return m.group(1) if m else None

    def _fetch_html_verified_via_requests(self, url: str) -> tuple[str, str]:
        headers = dict(DEFAULT_HEADERS)
        with requests.get(
            url,
            headers=headers,
            cookies={"zp_verified": "1"},
            timeout=(10, self.timeout),
            allow_redirects=True,
        ) as response:
            response.raise_for_status()
            html = response.text
            if "ZEEL PROJECT - 3D Models & Interior Design" in html and "navigator.webdriver" in html:
                raise RuntimeError("zeelproject anti-bot stub returned instead of product page")
            return html, response.url

    def _fetch_html_verified_via_curl(self, url: str) -> tuple[str, str]:
        import subprocess

        marker = "__CODEX_EFFECTIVE_URL__:"
        command = [
            "curl",
            "-L",
            "--compressed",
            "--connect-timeout",
            "10",
            "--max-time",
            str(self.timeout),
            "--retry",
            "2",
            "--retry-delay",
            "1",
            "--retry-all-errors",
            "-A",
            DEFAULT_HEADERS["User-Agent"],
            "-H",
            f"Accept-Language: {DEFAULT_HEADERS['Accept-Language']}",
            "-H",
            "Cookie: zp_verified=1",
            "-o",
            "-",
            "-w",
            f"\n{marker}%{{url_effective}}",
            url,
        ]
        result = subprocess.run(command, capture_output=True, check=False)
        if result.returncode != 0:
            stderr = (result.stderr or b"").decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"curl failed: {stderr[:500]}")

        stdout = result.stdout or b""
        marker_bytes = f"\n{marker}".encode("utf-8")
        idx = stdout.rfind(marker_bytes)
        if idx == -1:
            raise RuntimeError("curl did not return effective URL marker")

        raw = stdout[:idx]
        final_url = stdout[idx + len(marker_bytes):].decode("utf-8", errors="replace").strip() or url
        html = raw.decode("utf-8", errors="replace")
        if "ZEEL PROJECT - 3D Models & Interior Design" in html and "navigator.webdriver" in html:
            raise RuntimeError("zeelproject anti-bot stub returned instead of product page")
        return html, final_url

    @staticmethod
    def clean_og_title(text: str) -> str:
        value = SupplierAdapter.norm_space(text)
        value = re.sub(r"\s*,\s*.*?- Download the 3D Model.*$", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\s*\(\d+\)\s*\|\s*zeelproject\.com$", "", value, flags=re.IGNORECASE)
        return value

    @staticmethod
    def extract_option_value(soup: BeautifulSoup, *labels: str) -> Optional[str]:
        label_set = {SupplierAdapter.norm_space(label).casefold() for label in labels if label}
        for option in soup.select(".model_options .option"):
            label_node = option.select_one(".option_bold")
            value_node = option.select_one(".option_name")
            if not label_node or not value_node:
                continue
            option_label = SupplierAdapter.norm_space(label_node.get_text(" ", strip=True))
            if option_label.casefold() not in label_set:
                continue
            texts = [
                SupplierAdapter.norm_space(node.get_text(" ", strip=True))
                for node in value_node.select("a")
            ]
            texts = [text for text in texts if text]
            if texts:
                return ", ".join(texts)
            text = SupplierAdapter.norm_space(value_node.get_text(" ", strip=True))
            return text or None
        return None

    @staticmethod
    def parse_dim(text: str, *labels: str) -> Optional[float]:
        units = r"(cm|см|mm|мм)"
        for label in labels:
            m = re.search(
                rf"{re.escape(label)}[^0-9\n]*([0-9]+(?:[.,][0-9]+)?(?:\s*/\s*[0-9]+(?:[.,][0-9]+)?)*)\s*{units}",
                text,
                flags=re.IGNORECASE,
            )
            if not m:
                continue
            try:
                values = [
                    float(part.replace(",", "."))
                    for part in re.split(r"\s*/\s*", m.group(1))
                    if part.strip()
                ]
                value = max(values)
                if m.group(2).lower() in {"mm", "мм"}:
                    value = value / 10.0
                return value
            except Exception:
                return None
        return None

    @staticmethod
    def unique_keep_order(values: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for value in values:
            if not value or value in seen:
                continue
            seen.add(value)
            out.append(value)
        return out
