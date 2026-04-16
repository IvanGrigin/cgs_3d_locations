# -*- coding: utf-8 -*-
"""
This adapter parses iModern product pages into normalized product records.
It extracts dimensions, price, descriptive text, related items, and 3D links.
The parser relies mostly on HTML content rather than a supplier API.
It is intentionally simple because page structure is fairly regular.
Keep direct download-link extraction explicit.
"""
from __future__ import annotations

import json
import re
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from src.suppliers.adapters.base import SupplierAdapter
from src.suppliers.models import ProductRecord


class IModernAdapter(SupplierAdapter):
    site_name = "imodern"

    def can_handle(self, url: str) -> bool:
        return "imodern.ru" in url.lower()

    def parse(self, url: str, html: str, final_url: str) -> list[ProductRecord]:
        return [self.parse_product(final_url, html, url)]

    def parse_product(self, final_url: str, html: str, source_url: str) -> ProductRecord:
        soup = BeautifulSoup(html, "html.parser")
        page_text = soup.get_text("\n", strip=True)

        title = None
        h1 = soup.find("h1")
        if h1:
            title = self.norm_space(h1.get_text(" ", strip=True))
        if not title:
            title = self.extract_meta_content(soup, "og:title")
        if not title:
            title = self.extract_name_from_jsonld(soup)

        price_value, old_price_value, price_currency = self.extract_price(soup, page_text, html)

        width_cm = self.parse_dimension_cm(page_text, "Ширина:")
        depth_cm = self.parse_dimension_cm(page_text, "Глубина:")
        height_cm = self.parse_dimension_cm(page_text, "Высота:")
        weight_kg = self.parse_weight_kg(page_text)

        description = None
        desc = soup.find(attrs={"itemprop": "description"})
        if desc:
            description = self.norm_space(desc.get_text(" ", strip=True))
        if not description:
            description = self.extract_meta_content(soup, "og:description")

        materials = []
        for label in ["Обивка", "Сиденье и спинка", "Каркас", "Материал", "Корпус", "Ножки"]:
            m = re.search(rf"{re.escape(label)}:\s*([^\n]+)", page_text, re.IGNORECASE)
            if m:
                materials.append(f"{label}: {self.norm_space(m.group(1))}")

        model_download_url = None
        for tag in soup.find_all(True):
            txt = self.norm_space(tag.get_text(" ", strip=True))
            if txt == "Скачать 3D модель":
                found = self.extract_onclick_download(tag, final_url)
                if found:
                    model_download_url = found
                    break

        related = self.extract_related_items(soup, final_url)

        base_product_url = final_url.split("#")[0]

        return ProductRecord(
            unique_key=self.build_unique_key(base_product_url, None),
            source_site=self.site_name,
            source_url=source_url,
            parsed_at=self.now_utc_iso(),
            external_id=None,
            category_raw=None,
            category_norm=self.classify_category(title),
            title=title,
            brand="Imodern",
            collection=None,
            product_url=base_product_url,
            model_link_type="page_anchor" if model_download_url else None,
            model_page_url=base_product_url + "#model",
            model_download_url=model_download_url,
            model_download_landing_url=None,
            model_vendor_url=base_product_url,
            model_extraction_method="onclick_from_product_page",
            model_download_filename=self.filename_from_url(model_download_url),
            model_format=self.ext_from_url(model_download_url),
            price_value=price_value,
            price_currency=price_currency,
            old_price_value=old_price_value,
            style=None,
            color=None,
            description=description,
            width_cm=width_cm,
            depth_cm=depth_cm,
            height_cm=height_cm,
            weight_kg=weight_kg,
            room=None,
            materials="; ".join(materials) if materials else None,
            availability=None,
            country_brand=None,
            production_country=None,
            tags_json="[]",
            images_json=json.dumps(self.extract_product_images(soup, final_url), ensure_ascii=False),
            related_json=json.dumps(related, ensure_ascii=False),
            extra_json=json.dumps({}, ensure_ascii=False),
            raw_html=html,
        )

    def extract_price(
        self,
        soup: BeautifulSoup,
        page_text: str,
        html: str,
    ) -> tuple[float | None, float | None, str | None]:
        price_meta = soup.find("meta", attrs={"property": "product:price:amount"})
        if price_meta and price_meta.get("content"):
            try:
                price_value = float(str(price_meta["content"]).replace(",", "."))
                return price_value, None, "RUB"
            except Exception:
                pass

        for pattern in (
            r"""price\s*=\s*['"]([0-9]+(?:[.,][0-9]+)?)['"]""",
            r"""['"]price['"]\s*:\s*['"]([0-9]+(?:[.,][0-9]+)?)['"]""",
        ):
            match = re.search(pattern, html, re.IGNORECASE)
            if not match:
                continue
            try:
                return float(match.group(1).replace(",", ".")), None, "RUB"
            except Exception:
                pass

        return self.parse_price_rub(page_text)

    def extract_product_images(self, soup: BeautifulSoup, base_url: str) -> list[str]:
        out: list[str] = []

        for img in soup.select(".slider-pro .sp-slides img"):
            src = img.get("data-src") or img.get("src")
            if not src:
                continue
            absolute = urljoin(base_url, src)
            if "/upload/" not in absolute:
                continue
            out.append(absolute)

        og_image = self.extract_meta_content(soup, "og:image")
        if og_image:
            out.append(og_image)

        seen = set()
        uniq = []
        for url in out:
            if url not in seen:
                seen.add(url)
                uniq.append(url)

        return uniq[:30]

    def extract_related_items(self, soup: BeautifulSoup, base_url: str) -> list[dict]:
        out: list[dict] = []
        seen: set[tuple[str | None, str | None]] = set()

        for node in soup.select(".sim_prod"):
            title = node.get("data-name")
            href = None
            a = node.find("a", href=True)
            if a:
                href = urljoin(base_url, a["href"])

            data_product = node.get("data-product")
            if data_product:
                try:
                    payload = json.loads(data_product)
                except Exception:
                    payload = None
                if isinstance(payload, dict):
                    title = title or payload.get("name")
                    href = href or urljoin(base_url, payload.get("url", "")) if payload.get("url") else href

            if not title:
                continue

            key = (title, href)
            if key in seen:
                continue
            seen.add(key)

            price = node.get("data-price")
            out.append(
                {
                    "title": title,
                    "url": href,
                    "price": price,
                }
            )

        return out[:50]
