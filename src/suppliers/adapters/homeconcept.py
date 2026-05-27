# -*- coding: utf-8 -*-
"""
This adapter parses HomeConcept product pages and 3D library listings.
It can enrich library cards by following the linked product page.
The parser extracts prices, dimensions, materials, and direct model links.
HomeConcept is currently the richest supplier source in this codebase.
Keep library-to-product enrichment behavior stable.
"""
from __future__ import annotations

import json
import re
from typing import Optional
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from src.suppliers.models import ProductRecord

from .base import SupplierAdapter


class HomeConceptAdapter(SupplierAdapter):
    site_name = "homeconcept"

    def can_handle(self, url: str) -> bool:
        host = urlparse(url).netloc.lower()
        return "homeconcept.ru" in host

    def parse(self, url: str, html: str, final_url: str) -> list[ProductRecord]:
        soup = BeautifulSoup(html, "html.parser")
        path = urlparse(final_url).path.lower()

        if "/3d-models" in path:
            return self.parse_library_page(url, soup, final_url)

        return [self.parse_product_page(url, soup, final_url)]

    # =========================
    # Library page
    # =========================

    def parse_library_page(
        self,
        source_url: str,
        soup: BeautifulSoup,
        final_url: str,
    ) -> list[ProductRecord]:
        cards = soup.select(".items-list-3d-models .item")
        out: list[ProductRecord] = []

        for card in cards:
            rec = self.parse_library_card(card, final_url)
            if rec is None:
                continue

            if rec.product_url:
                try:
                    product_html, product_final_url = self.fetch_html(rec.product_url)
                    product_soup = BeautifulSoup(product_html, "html.parser")
                    enriched = self.parse_product_page(
                        rec.product_url,
                        product_soup,
                        product_final_url,
                        fallback_record=rec,
                        source_url_override=source_url,
                        library_url=final_url,
                    )
                    out.append(enriched)
                    continue
                except Exception as e:
                    rec.extra_json = self._merge_extra_json(
                        rec.extra_json,
                        {
                            "parse_stage": "library",
                            "enriched_from_product_page": False,
                            "product_fetch_error": str(e),
                        },
                    )

            out.append(rec)

        return out

    def parse_library_card(self, card, base_url: str) -> Optional[ProductRecord]:
        product_a = card.select_one(".item-link-image a[href]")
        download_a = card.select_one(".item-price a[href]")
        img = card.select_one(".item-link-image img")
        title_div = card.select_one(".item-name")

        title = self.norm_space(title_div.get_text(" ", strip=True)) if title_div else None
        if not title:
            return None

        product_url = urljoin(base_url, product_a["href"]) if product_a and product_a.get("href") else None
        model_download_url = urljoin(base_url, download_a["href"]) if download_a and download_a.get("href") else None
        images = self.extract_images_from_card(img, base_url)

        unique_key = self.build_unique_key(product_url or model_download_url or base_url, None)

        return ProductRecord(
            unique_key=unique_key,
            source_site=self.site_name,
            source_url=base_url,
            parsed_at=self.now_utc_iso(),
            external_id=None,
            category_raw=None,
            category_norm=self.classify_category(title),
            title=title,
            brand=None,
            collection=None,
            product_url=product_url,
            model_link_type="direct_file" if model_download_url else None,
            model_page_url=product_url or base_url,
            model_download_url=model_download_url,
            model_download_landing_url=None,
            model_vendor_url=product_url or base_url,
            model_extraction_method="homeconcept_library_card",
            model_download_filename=self.filename_from_url(model_download_url),
            model_format=self.ext_from_url(model_download_url),
            price_value=None,
            price_currency=None,
            old_price_value=None,
            style=None,
            color=None,
            description=None,
            width_cm=None,
            depth_cm=None,
            height_cm=None,
            weight_kg=None,
            room=None,
            materials=None,
            availability=None,
            country_brand=None,
            production_country=None,
            tags_json=json.dumps([], ensure_ascii=False),
            images_json=json.dumps(images, ensure_ascii=False),
            related_json=json.dumps([], ensure_ascii=False),
            extra_json=json.dumps(
                {
                    "title_from_library": title,
                    "library_page_url": base_url,
                    "parse_stage": "library",
                    "enriched_from_product_page": False,
                    "category": self.classify_category(title),
                },
                ensure_ascii=False,
            ),
            raw_html=str(card),
        )

    # =========================
    # Product page
    # =========================

    def parse_product_page(
        self,
        source_url: str,
        soup: BeautifulSoup,
        final_url: str,
        fallback_record: Optional[ProductRecord] = None,
        source_url_override: Optional[str] = None,
        library_url: Optional[str] = None,
    ) -> ProductRecord:
        page_text = self.norm_space(soup.get_text(" ", strip=True))

        title = self.extract_product_title(soup)
        brand = self.extract_brand(soup) or self.extract_brand_from_jsonld(soup)
        collection = self.extract_collection(soup, page_text)
        external_id = self.extract_code(soup)
        description = self.extract_description(soup)
        price_value, old_price_value, price_currency = self.extract_price(soup)
        availability = self.extract_availability(soup)
        style = self.extract_characteristic(soup, "Стиль")
        room = self.extract_characteristic(soup, "Помещение")
        materials = self.extract_materials(soup)
        color = self.extract_color(soup)
        country_brand = self.extract_characteristic(soup, "Страна бренда")
        production_country = self.extract_characteristic(soup, "Производство")
        width_cm, depth_cm, height_cm = self.extract_dimensions(soup)
        weight_kg = self.extract_weight_from_table(soup)
        model_download_url = self.extract_model_download_url(soup, final_url)
        images = self.extract_product_images(soup, final_url)
        related = self.extract_related_items(soup, final_url)
        tags = self.extract_tags(soup)

        category_raw = self.extract_breadcrumb_category(soup)
        fallback_title = fallback_record.title if fallback_record else None
        category_norm = self.classify_category(title or fallback_title)

        product_url = final_url

        merged_title = self.clean_title(title or fallback_title)
        merged_brand = brand or (fallback_record.brand if fallback_record else None)
        merged_collection = collection or (fallback_record.collection if fallback_record else None)
        merged_model_download_url = model_download_url or (fallback_record.model_download_url if fallback_record else None)
        merged_images = images or self._json_load_list(fallback_record.images_json if fallback_record else None)

        unique_key = self.build_unique_key(product_url, external_id)

        extra = {
            "parse_stage": "product",
            "enriched_from_product_page": True,
            "library_page_url": library_url,
        }
        if fallback_title:
            extra["title_from_library"] = fallback_title

        return ProductRecord(
            unique_key=unique_key,
            source_site=self.site_name,
            source_url=source_url_override or source_url,
            parsed_at=self.now_utc_iso(),
            external_id=external_id,
            category_raw=category_raw,
            category_norm=category_norm,
            title=merged_title,
            brand=merged_brand,
            collection=merged_collection,
            product_url=product_url,
            model_link_type="direct_file" if merged_model_download_url else None,
            model_page_url=product_url,
            model_download_url=merged_model_download_url,
            model_download_landing_url=None,
            model_vendor_url=product_url,
            model_extraction_method="homeconcept_product_page",
            model_download_filename=self.filename_from_url(merged_model_download_url),
            model_format=self.ext_from_url(merged_model_download_url),
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
            images_json=json.dumps(merged_images, ensure_ascii=False),
            related_json=json.dumps(related, ensure_ascii=False),
            extra_json=json.dumps(extra, ensure_ascii=False),
            raw_html=str(soup),
        )

    # =========================
    # Extractors
    # =========================

    @staticmethod
    def clean_title(text: Optional[str]) -> Optional[str]:
        if not text:
            return None

        cleaned = SupplierAdapter.norm_space(text)
        cleaned = re.sub(r"\s*\|\s*Home Concept\s*$", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*[-|–—]\s*Home Concept\s*$", "", cleaned, flags=re.IGNORECASE)
        return cleaned or None

    def extract_product_title(self, soup: BeautifulSoup) -> Optional[str]:
        offer = soup.select_one(".catalog-product__offer-name")
        if offer:
            txt = self.norm_space(offer.get_text(" ", strip=True))
            if txt:
                return self.clean_title(txt)

        h1_text = soup.select_one(".catalog-product__name-text")
        if h1_text:
            txt = self.norm_space(h1_text.get_text(" ", strip=True))
            if txt:
                return self.clean_title(txt)

        h1 = soup.select_one("h1.catalog-product__name")
        if h1:
            txt = self.norm_space(h1.get_text(" ", strip=True))
            if txt:
                return self.clean_title(txt)

        og = self.extract_meta_content(soup, "og:title")
        if og:
            return self.clean_title(og)

        return self.clean_title(self.extract_name_from_jsonld(soup))

    def extract_brand(self, soup: BeautifulSoup) -> Optional[str]:
        x = soup.select_one(".catalog-product__name-brand")
        if x:
            return self.norm_space(x.get_text(" ", strip=True))

        x = soup.select_one(".product-characteristics")
        if x:
            v = self.extract_characteristic(soup, "Бренд")
            if v:
                return v

        return None  # pragma: no cover

    def extract_code(self, soup: BeautifulSoup) -> Optional[str]:
        x = soup.select_one(".js-item-info-current-check-offer-code")
        if x:
            txt = self.norm_space(x.get_text(" ", strip=True))
            if txt:
                return txt
        return None

    def extract_description(self, soup: BeautifulSoup) -> Optional[str]:
        x = soup.select_one(".item-info-detail-text")
        if x:
            txt = self.norm_space(x.get_text("\n", strip=True))
            if txt:
                return txt

        og_description = self.extract_meta_content(soup, "og:description")
        if og_description:
            return og_description

        for item in self.extract_jsonld_objects(soup):
            description = item.get("description")
            if isinstance(description, str):
                txt = self.norm_space(description)
                if txt:
                    return txt

        return None  # pragma: no cover

    def extract_price(self, soup: BeautifulSoup) -> tuple[Optional[float], Optional[float], Optional[str]]:
        price_block = soup.select_one(".item-info-current-check-offer-price")
        if price_block:
            txt = self.norm_space(price_block.get_text(" ", strip=True))
            parsed_price, parsed_old_price, parsed_currency = self.parse_price_rub(txt)
            if parsed_price is not None:
                return parsed_price, parsed_old_price, parsed_currency

        for item in self.extract_jsonld_objects(soup):
            offers = item.get("offers")
            if not isinstance(offers, dict):
                continue

            low_price = offers.get("lowPrice") or offers.get("price")
            if low_price is None:
                continue  # pragma: no cover

            try:
                price_value = float(str(low_price).replace(",", "."))
            except Exception:
                continue

            currency = offers.get("priceCurrency")
            currency_text = str(currency).strip() if currency else "RUB"
            return price_value, None, currency_text

        return None, None, None

    def extract_availability(self, soup: BeautifulSoup) -> Optional[str]:
        x = soup.select_one(".item-info-current-check-offer-status-available")
        if x:
            txt = self.norm_space(x.get_text(" ", strip=True))
            if txt:
                return txt
        return None

    def extract_color(self, soup: BeautifulSoup) -> Optional[str]:
        x = soup.select_one(".catalog-product__material-text--active")
        if x:
            txt = self.norm_space(x.get_text(" ", strip=True))
            if txt:
                return txt

        x = soup.select_one(".catalog-product__current-check-offer-material")
        if x:
            txt = self.norm_space(x.get_text(" ", strip=True))
            if txt:
                return txt

        return None

    def extract_materials(self, soup: BeautifulSoup) -> Optional[str]:
        vals = []
        for label in [
            "Материал",
            "Материал каркаса",
            "Материал наполнителя",
            "Материал ножек",
        ]:
            v = self.extract_characteristic(soup, label)
            if v:
                vals.append(f"{label}: {v}")

        return "; ".join(vals) if vals else None

    def extract_collection(self, soup: BeautifulSoup, page_text: str) -> Optional[str]:
        h2 = soup.select_one(".content-block.collections-list .title > a")
        if h2:
            txt = self.norm_space(h2.get_text(" ", strip=True))
            if txt:
                return txt

        m = re.search(r"([A-Za-z0-9][A-Za-z0-9\s&\-]+?)\s+другие модели коллекции", page_text, re.IGNORECASE)
        if m:
            return self.norm_space(m.group(1))

        return None

    def extract_breadcrumb_category(self, soup: BeautifulSoup) -> Optional[str]:
        crumbs = [
            self.norm_space(a.get_text(" ", strip=True))
            for a in soup.select(".bx_breadcrumbs a")
            if a.get_text(strip=True)
        ]
        if not crumbs:
            return None
        return " > ".join(crumbs)

    def extract_dimensions(self, soup: BeautifulSoup) -> tuple[Optional[float], Optional[float], Optional[float]]:
        width_cm = self.extract_numeric_dimension(
            soup,
            ["Ширина"],
        )
        depth_cm = self.extract_numeric_dimension(
            soup,
            ["Глубина", "Длина"],
        )
        height_cm = self.extract_numeric_dimension(
            soup,
            [
                "Высота",
                "Высота абажура",
                "Высота основания",
                "Общая высота",
                "Высота изделия",
            ],
        )

        diameter_cm = self.extract_numeric_dimension(
            soup,
            ["Диаметр", "Диаметр абажура", "Диаметр основания", "Диаметр технического горшка"],
        )

        if diameter_cm is not None:
            if width_cm is None:
                width_cm = diameter_cm
            if depth_cm is None:
                depth_cm = diameter_cm

        return width_cm, depth_cm, height_cm

    def extract_numeric_dimension(self, soup: BeautifulSoup, labels: list[str]) -> Optional[float]:
        for label in labels:
            value = self.extract_characteristic(soup, label)
            parsed = self.parse_numeric_value(value)
            if parsed is not None:
                return parsed

        label_set = {self.norm_space(label).lower() for label in labels}
        for item in self.extract_jsonld_objects(soup):
            properties = item.get("additionalProperty")
            if not isinstance(properties, list):
                continue

            for prop in properties:
                if not isinstance(prop, dict):
                    continue  # pragma: no cover

                name = prop.get("name")
                value = prop.get("value")
                if not isinstance(name, str):
                    continue  # pragma: no cover

                if self.norm_space(name).lower() not in label_set:
                    continue

                parsed = self.parse_numeric_value(str(value) if value is not None else None)
                if parsed is not None:
                    return parsed

        return None

    def parse_numeric_value(self, value: Optional[str]) -> Optional[float]:
        if not value:
            return None

        m = re.search(r"([0-9]+(?:[.,][0-9]+)?)", value, re.IGNORECASE)
        if not m:
            return None

        try:
            return float(m.group(1).replace(",", "."))
        except Exception:  # pragma: no cover
            return None  # pragma: no cover

    def extract_weight_from_table(self, soup: BeautifulSoup) -> Optional[float]:
        value = self.extract_characteristic(soup, "Вес в упаковке")
        parsed = self.parse_numeric_value(value)
        if parsed is not None:
            return parsed

        for item in self.extract_jsonld_objects(soup):
            properties = item.get("additionalProperty")
            if not isinstance(properties, list):
                continue  # pragma: no cover

            for prop in properties:
                if not isinstance(prop, dict):
                    continue  # pragma: no cover
                name = self.norm_space(str(prop.get("name", ""))).lower()
                if name != "вес в упаковке":
                    continue  # pragma: no cover
                parsed = self.parse_numeric_value(str(prop.get("value", "")))
                if parsed is not None:
                    return parsed

        return None  # pragma: no cover

    def extract_characteristic(self, soup: BeautifulSoup, label: str) -> Optional[str]:
        for row in soup.select(".product-characteristics tr"):
            cells = row.find_all("td")
            if len(cells) < 2:
                continue
            key = self.norm_space(cells[0].get_text(" ", strip=True))
            val = self.norm_space(cells[1].get_text(" ", strip=True))
            if key.lower() == label.lower():
                return val or None
        return None

    def extract_model_download_url(self, soup: BeautifulSoup, base_url: str) -> Optional[str]:
        selectors = [
            ".catalog-product__3d-model__link-download[href]",
            "a[href][download]",
        ]
        for sel in selectors:
            a = soup.select_one(sel)
            if a and a.get("href"):
                return urljoin(base_url, a["href"])

        for a in soup.select("a[href]"):
            txt = self.norm_space(a.get_text(" ", strip=True)).lower()
            href = a.get("href")
            if not href:
                continue
            if "скачать 3d" in txt or "скачать 3d-модель" in txt:
                return urljoin(base_url, href)

        return None  # pragma: no cover

    def extract_product_images(self, soup: BeautifulSoup, base_url: str) -> list[str]:
        out: list[str] = []

        for img in soup.select(".catalog-product__image-slider img"):
            src = img.get("src") or img.get("data-src")
            if src:
                out.append(urljoin(base_url, src))

        for img in soup.select(".catalog-product__3d-model img"):
            src = img.get("src") or img.get("data-src")
            if src:
                out.append(urljoin(base_url, src))

        seen = set()
        uniq = []
        for x in out:
            if x not in seen:
                seen.add(x)
                uniq.append(x)
        return uniq[:50]

    def extract_images_from_card(self, img, base_url: str) -> list[str]:
        if not img:
            return []
        src = img.get("data-src") or img.get("src")
        if not src:
            return []
        return [urljoin(base_url, src)]

    def extract_tags(self, soup: BeautifulSoup) -> list[str]:
        out = []
        for a in soup.select(".catalog-product__tags a"):
            txt = self.norm_space(a.get_text(" ", strip=True))
            if txt:
                out.append(txt)
        return out

    def extract_related_items(self, soup: BeautifulSoup, base_url: str) -> list[dict]:
        out: list[dict] = []

        for block in soup.select(".collections-list .item"):
            item = self.extract_related_from_tile(block, base_url, relation="collection")
            if item:
                out.append(item)

        for block in soup.select(".catalog-top-items .item"):
            item = self.extract_related_from_tile(block, base_url, relation="generic")
            if item:
                out.append(item)

        seen = set()
        uniq = []
        for x in out:
            key = (x.get("title"), x.get("url"), x.get("relation"))
            if key not in seen:
                seen.add(key)
                uniq.append(x)
        return uniq[:100]

    def extract_related_from_tile(self, block, base_url: str, relation: str) -> Optional[dict]:
        a = block.select_one(".item-name[href], .item-link-image[href]")
        if not a or not a.get("href"):
            return None

        title = self.norm_space(a.get_text(" ", strip=True))
        url = urljoin(base_url, a["href"])

        price_div = block.select_one(".item-price")
        price_text = self.norm_space(price_div.get_text(" ", strip=True)) if price_div else None

        img = block.select_one("img")
        image = None
        if img:
            src = img.get("src") or img.get("data-src")
            if src:
                image = urljoin(base_url, src)

        return {
            "relation": relation,
            "title": title,
            "url": url,
            "price_text": price_text,
            "image": image,
        }

    # =========================
    # Helpers
    # =========================

    def _merge_extra_json(self, old_json: Optional[str], patch: dict) -> str:
        base = {}
        if old_json:
            try:
                base = json.loads(old_json)
            except Exception:
                base = {}
        base.update(patch)
        return json.dumps(base, ensure_ascii=False)

    def _json_load_list(self, raw: Optional[str]) -> list:
        if not raw:
            return []  # pragma: no cover
        try:
            x = json.loads(raw)
            if isinstance(x, list):
                return x
        except Exception:
            pass
        return []
