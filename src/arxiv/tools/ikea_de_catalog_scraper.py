#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Scrape IKEA Germany product cards into supplier-catalog compatible JSON.

The IKEA product page server-renders the data needed for a supplier card in
`type="text/hydrate"` and JSON-LD scripts. This scraper keeps only the first
main product image URL by default, parses EUR prices, measurements, packaging,
materials, descriptions and optional IKEA-hosted 3D model URLs.

Examples:
    # One product smoke test.
    python3 -m src.tools.ikea_de_catalog_scraper \
      --product-url https://www.ikea.com/de/de/p/smalom-blockkerze-led-batteriebetrieben-weiss-90619785/ \
      --out out/supplier_ingest/ikea_de/single

    # Category crawl.
    python3 -m src.tools.ikea_de_catalog_scraper \
      --category-url https://www.ikea.com/de/de/cat/sofas-fu003/ \
      --max-pages 0 \
      --out out/supplier_ingest/ikea_de/catalog
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://www.ikea.com"
SIK_BASE_URL = "https://sik.search.blue.cdtapps.com/de/de/product-list-page"
DEFAULT_OUT_DIR = "out/supplier_ingest/ikea_de/catalog"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/121.0.0.0 Safari/537.36"
)

DEFAULT_CATEGORY_URLS = [
    "https://www.ikea.com/de/de/cat/sofas-fu003/",
    "https://www.ikea.com/de/de/cat/2er-sofas-10668/",
    "https://www.ikea.com/de/de/cat/3er-sofas-10670/",
    "https://www.ikea.com/de/de/cat/sessel-16239/",
    "https://www.ikea.com/de/de/cat/regale-10382/",
    "https://www.ikea.com/de/de/cat/kleiderschraenke-19053/",
    "https://www.ikea.com/de/de/cat/kommoden-10451/",
    "https://www.ikea.com/de/de/cat/esstische-21825/",
    "https://www.ikea.com/de/de/cat/stuehle-20652/",
    "https://www.ikea.com/de/de/cat/betten-bm003/",
    "https://www.ikea.com/de/de/cat/leuchten-li001/",
]
DEFAULT_SITEMAP_INDEX_URL = "https://www.ikea.com/sitemaps/sitemap.xml"


@dataclass
class IkeaCard:
    source_page_url: str = ""
    source_category_url: str = ""
    source_page_number: int = 0
    product_url: str = ""
    item_no: str = ""
    visible_item_no: str = ""
    title: str = ""
    name: str = ""
    type_name: str = ""
    description: str = ""
    category_raw: str = ""
    category_norm: str = "furniture"
    price: float | None = None
    price_currency: str = "EUR"
    availability: str = "unknown"
    rating: float | None = None
    reviews_count: int | None = None
    color: str = ""
    materials: str = ""
    dimensions_cm: dict[str, float | None] = field(default_factory=dict)
    measurements_raw: list[dict[str, Any]] = field(default_factory=list)
    package_measurements: list[dict[str, Any]] = field(default_factory=list)
    image_url: str = ""
    image_alt: str = ""
    all_product_images: list[str] = field(default_factory=list)
    model_urls: list[str] = field(default_factory=list)
    raw_status: str = "ok"
    error: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def norm_text(value: Any) -> str:
    text = str(value or "").replace("\xa0", " ").replace("&nbsp;", " ")
    return re.sub(r"\s+", " ", text).strip()


def lower(value: Any) -> str:
    return norm_text(value).lower().replace("ё", "е")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def unique_list(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        value = norm_text(value)
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def empty_dimensions() -> dict[str, float | None]:
    return {
        "width": None,
        "depth": None,
        "height": None,
        "weight_kg": None,
        "package_width": None,
        "package_depth": None,
        "package_height": None,
        "packed_weight_kg": None,
        "volume_m3": None,
    }


def normalize_url(url: str, *, keep_query: bool = False) -> str:
    if not url:
        return ""
    parsed = urlparse(urljoin(BASE_URL, url))
    query = parsed.query if keep_query else ""
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path or "/", "", query, ""))


def session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "de-DE,de;q=0.9,en;q=0.7",
        }
    )
    return s


def fetch(sess: requests.Session, url: str, *, retries: int = 3, sleep: float = 0.4) -> str:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = sess.get(url, timeout=30)
            resp.raise_for_status()
            return resp.text
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(sleep * attempt)
    try:
        proc = subprocess.run(
            ["curl", "-L", "--retry", str(max(1, retries)), "--retry-delay", str(max(1, int(sleep))), url],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=90,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout
        last_error = RuntimeError((proc.stderr or "").strip() or f"curl exited {proc.returncode}")
    except Exception as exc:
        last_error = exc
    raise RuntimeError(f"fetch failed for {url}: {last_error}")


def fetch_json(sess: requests.Session, url: str, *, retries: int = 3, sleep: float = 0.4) -> Any:
    text = fetch(sess, url, retries=retries, sleep=sleep)
    try:
        return json.loads(text)
    except Exception as exc:
        raise RuntimeError(f"json parse failed for {url}: {exc}") from exc


def json_loads_script(text: str) -> Any | None:
    text = html.unescape(text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def iter_jsonld(soup: BeautifulSoup) -> list[Any]:
    out: list[Any] = []
    for script in soup.find_all("script", {"type": "application/ld+json"}):
        data = json_loads_script(script.get_text())
        if data is not None:
            out.append(data)
    return out


def find_product_hydrate(soup: BeautifulSoup) -> tuple[dict[str, Any], dict[str, Any]]:
    compact: dict[str, Any] = {}
    page_props: dict[str, Any] = {}
    for script in soup.find_all("script", {"type": "text/hydrate"}):
        data = json_loads_script(script.get_text())
        if not isinstance(data, dict):
            continue
        if isinstance(data.get("product"), dict) and data["product"].get("itemNo"):
            compact = data
        if isinstance(data.get("pageProps"), dict) and isinstance(data["pageProps"].get("product"), dict):
            page_props = data["pageProps"]
    return compact, page_props


def parse_item_no_from_url(url: str) -> str:
    path = urlparse(url).path.rstrip("/")
    tail = path.rsplit("-", 1)[-1]
    return tail[1:] if tail.startswith("s") and tail[1:].isdigit() else tail


def parse_item_list(soup: BeautifulSoup, page_url: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for data in iter_jsonld(soup):
        candidates = data if isinstance(data, list) else [data]
        for item in candidates:
            if not isinstance(item, dict) or item.get("@type") != "ItemList":
                continue
            for entry in item.get("itemListElement") or []:
                if not isinstance(entry, dict):
                    continue
                url = normalize_url(entry.get("url") or "")
                if "/p/" not in url:
                    continue
                image = entry.get("image") or {}
                rows.append(
                    {
                        "url": url,
                        "name": norm_text(entry.get("name")),
                        "image": normalize_url(image.get("url") or "", keep_query=True) if isinstance(image, dict) else "",
                        "position": entry.get("position"),
                        "source_page_url": page_url,
                    }
                )
    if rows:
        return rows
    seen: set[str] = set()
    for match in re.finditer(r"https://www\.ikea\.com/de/de/p/[^\"<\s]+", str(soup)):
        url = normalize_url(match.group(0))
        if url and url not in seen:
            seen.add(url)
            rows.append({"url": url, "name": "", "image": "", "position": len(rows) + 1, "source_page_url": page_url})
    return rows


def category_id_from_url(category_url: str) -> str:
    path = urlparse(category_url).path.rstrip("/")
    slug = path.rsplit("/", 1)[-1]
    match = re.search(r"([a-z]{2}\d{3}|\d{4,6})$", slug)
    return match.group(1) if match else ""


def rows_from_sik_api_item(item: dict[str, Any], category_url: str, position: int) -> dict[str, Any] | None:
    url = normalize_url(item.get("pipUrl") or item.get("url") or "")
    if not url or "/p/" not in url:
        return None
    images = item.get("allProductImage") or []
    main = next((x for x in images if isinstance(x, dict) and x.get("type") == "MAIN_PRODUCT_IMAGE"), None)
    if not main and images and isinstance(images[0], dict):
        main = images[0]
    price = item.get("salesPrice") or {}
    category_path = item.get("categoryPath") or []
    return {
        "url": url,
        "name": norm_text(item.get("name")),
        "image": normalize_url((main or {}).get("url") or item.get("mainImageUrl") or "", keep_query=True),
        "position": position,
        "source_page_url": SIK_BASE_URL,
        "source_category_url": category_url,
        "source_page_number": 1,
        "listing_item_no": norm_text(item.get("itemNo") or item.get("id")),
        "listing_type_name": norm_text(item.get("typeName")),
        "listing_valid_design_text": norm_text(item.get("validDesignText")),
        "listing_price": price.get("numeral") if isinstance(price, dict) else None,
        "listing_price_currency": price.get("currencyCode") if isinstance(price, dict) else None,
        "listing_rating": item.get("ratingValue"),
        "listing_reviews_count": item.get("ratingCount"),
        "listing_category_path": category_path,
        "listing_category_raw": " / ".join(norm_text(x.get("name")) for x in category_path if isinstance(x, dict) and x.get("name")),
        "listing_colors": item.get("colors") or [],
    }


def collect_product_urls_from_sik_api(
    sess: requests.Session,
    category_url: str,
    *,
    limit: int,
    sleep: float,
) -> list[dict[str, Any]]:
    category_id = category_id_from_url(category_url)
    if not category_id:
        return []
    size = limit if limit > 0 else 1000
    query = urlencode({"category": category_id, "size": max(1, size)})
    api_url = f"{SIK_BASE_URL}?{query}"
    eprint(f"[list:api] {api_url}")
    data = fetch_json(sess, api_url, sleep=sleep)
    product_window = ((data or {}).get("productListPage") or {}).get("productWindow") or []
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in product_window:
        if not isinstance(item, dict):
            continue
        row = rows_from_sik_api_item(item, category_url, len(rows) + 1)
        if not row or row["url"] in seen:
            continue
        seen.add(row["url"])
        rows.append(row)
        if limit > 0 and len(rows) >= limit:
            break
    total = ((data or {}).get("productListPage") or {}).get("productCount")
    eprint(f"[list:api] category={category_id} total={total} rows={len(rows)}")
    return rows


def parse_sitemap_locs(xml_text: str) -> list[str]:
    return [html.unescape(x.strip()) for x in re.findall(r"<loc>\s*([^<]+)\s*</loc>", xml_text) if x.strip()]


def collect_product_urls_from_sitemaps(
    sess: requests.Session,
    sitemap_index_url: str,
    *,
    locale_marker: str = "prod-de-DE_",
    limit: int = 0,
    sleep: float = 0.25,
) -> list[dict[str, Any]]:
    eprint(f"[sitemap:index] {sitemap_index_url}")
    index_xml = fetch(sess, sitemap_index_url, sleep=sleep)
    sitemap_urls = [url for url in parse_sitemap_locs(index_xml) if locale_marker in url]
    if not sitemap_urls and locale_marker.endswith("_"):
        sitemap_urls = [url for url in parse_sitemap_locs(index_xml) if f"/{locale_marker.rstrip('_')}" in url]
    eprint(f"[sitemap:index] files={len(sitemap_urls)} marker={locale_marker}")

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for sitemap_url in sitemap_urls:
        eprint(f"[sitemap] {sitemap_url}")
        sitemap_xml = fetch(sess, sitemap_url, sleep=sleep)
        locs = parse_sitemap_locs(sitemap_xml)
        new_count = 0
        for url in locs:
            normalized = normalize_url(url)
            if "/de/de/p/" not in normalized or normalized in seen:
                continue
            seen.add(normalized)
            new_count += 1
            rows.append(
                {
                    "url": normalized,
                    "name": "",
                    "image": "",
                    "position": len(rows) + 1,
                    "source_page_url": sitemap_url,
                    "source_category_url": "ikea_de_product_sitemap",
                    "source_page_number": len(rows) // 50000 + 1,
                }
            )
            if limit > 0 and len(rows) >= limit:
                break
        eprint(f"[sitemap] urls={len(locs)} new={new_count} collected={len(rows)}")
        if limit > 0 and len(rows) >= limit:
            break
        time.sleep(sleep)
    return rows


def next_page_url(soup: BeautifulSoup) -> str:
    link = soup.find("link", rel=lambda value: value and "next" in value)
    if link and link.get("href"):
        return normalize_url(str(link["href"]), keep_query=True)
    a = soup.find("a", attrs={"rel": lambda value: value and "next" in value})
    if a and a.get("href"):
        return normalize_url(str(a["href"]), keep_query=True)
    return ""


def collect_product_urls(sess: requests.Session, category_urls: list[str], max_pages: int, sleep: float, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for category_url in category_urls:
        try:
            api_cards = collect_product_urls_from_sik_api(sess, category_url, limit=limit, sleep=sleep)
        except Exception as exc:
            eprint(f"[list:api] failed {category_url}: {type(exc).__name__}: {exc}")
            api_cards = []
        if api_cards:
            new_count = 0
            for card in api_cards:
                url = card["url"]
                if not url or url in seen:
                    continue
                seen.add(url)
                new_count += 1
                rows.append(card)
                if limit > 0 and len(rows) >= limit:
                    break
            eprint(f"[list:api] new={new_count}")
            if limit > 0 and len(rows) >= limit:
                break
            continue
        page_url = normalize_url(category_url, keep_query=True)
        page_number = 1
        while page_url:
            eprint(f"[list] {page_url}")
            soup = BeautifulSoup(fetch(sess, page_url, sleep=sleep), "html.parser")
            cards = parse_item_list(soup, page_url)
            new_count = 0
            for card in cards:
                url = card["url"]
                if not url or url in seen:
                    continue
                seen.add(url)
                new_count += 1
                card["source_category_url"] = category_url
                card["source_page_number"] = page_number
                rows.append(card)
                if limit > 0 and len(rows) >= limit:
                    break
            eprint(f"[list] cards={len(cards)} new={new_count}")
            if limit > 0 and len(rows) >= limit:
                break
            if max_pages and page_number >= max_pages:
                break
            page_number += 1
            nxt = next_page_url(soup)
            if not nxt or nxt == page_url:
                break
            page_url = nxt
            time.sleep(sleep)
    return rows


def measurement_value_cm(measure: dict[str, Any]) -> float | None:
    value = measure.get("value")
    if isinstance(value, (int, float)):
        raw = float(value)
    else:
        text = norm_text(measure.get("measure") or measure.get("text"))
        m = re.search(r"(\d+(?:[.,]\d+)?)", text)
        if not m:
            return None
        raw = float(m.group(1).replace(",", "."))
    unit_text = lower(measure.get("measure") or measure.get("text") or "")
    if "mm" in unit_text:
        return raw / 10.0
    if "m" in unit_text and "cm" not in unit_text and "mm" not in unit_text:
        return raw * 100.0
    return raw


def parse_dimensions(page_props: dict[str, Any], compact_product: dict[str, Any]) -> tuple[dict[str, float | None], list[dict[str, Any]], list[dict[str, Any]]]:
    dims = empty_dimensions()
    measurements = (
        page_props.get("productInformationSectionProps", {})
        .get("measurementsProps", {})
        .get("measurements")
        or []
    )
    raw_measurements: list[dict[str, Any]] = []
    for measure in measurements:
        if not isinstance(measure, dict):
            continue
        raw_measurements.append(measure)
        name = lower(measure.get("name"))
        value = measurement_value_cm(measure)
        if value is None:
            continue
        if "breite" in name or name == "width":
            dims["width"] = value
        elif "tiefe" in name or "länge" in name or "laenge" in name or "length" in name or "depth" in name:
            dims["depth"] = value
        elif "höhe" in name or "hoehe" in name or "height" in name:
            dims["height"] = value
        elif "durchmesser" in name or "diameter" in name:
            dims["width"] = dims["width"] or value
            dims["depth"] = dims["depth"] or value

    package_rows: list[dict[str, Any]] = []
    packaging = page_props.get("product", {}).get("packaging") or compact_product.get("packaging") or {}
    packages = packaging.get("packages") or []
    if not packages:
        packages = [{"measurementGroups": [{"measurements": compact_product.get("packageMeasurements") or []}]}]
    for package in packages:
        for group in package.get("measurementGroups") or []:
            for measure in group.get("measurements") or []:
                if not isinstance(measure, dict):
                    continue
                package_rows.append(measure)
                typ = lower(measure.get("type") or measure.get("label"))
                value = measure.get("value")
                if not isinstance(value, (int, float)):
                    continue
                if "width" in typ or "breite" in typ:
                    dims["package_width"] = float(value)
                elif "length" in typ or "länge" in typ or "laenge" in typ or "depth" in typ:
                    dims["package_depth"] = float(value)
                elif "height" in typ or "höhe" in typ or "hoehe" in typ:
                    dims["package_height"] = float(value)
                elif "weight" in typ or "gewicht" in typ:
                    dims["packed_weight_kg"] = float(value)
    return dims, raw_measurements, package_rows


def parse_materials(page_props: dict[str, Any]) -> str:
    node = (
        page_props.get("productInformationSectionProps", {})
        .get("productDetailsProps", {})
        .get("accordionObject", {})
        .get("materialsAndCare", {})
        .get("contentProps", {})
    )
    values: list[str] = []
    for group in node.get("materials") or []:
        for item in group.get("materials") or []:
            values.append(norm_text(item.get("material")))
    return ", ".join(unique_list(values))


def parse_info_texts(page_props: dict[str, Any]) -> list[str]:
    acc = (
        page_props.get("productInformationSectionProps", {})
        .get("productDetailsProps", {})
        .get("accordionObject", {})
    )
    texts: list[str] = []
    for key in ("goodToKnow",):
        node = acc.get(key, {}).get("contentProps", {})
        for item in node.get(key) or []:
            texts.append(norm_text(item.get("text")))
    return unique_list(texts)


def product_jsonld(soup: BeautifulSoup) -> dict[str, Any]:
    for data in iter_jsonld(soup):
        candidates = data if isinstance(data, list) else [data]
        for item in candidates:
            if isinstance(item, dict) and item.get("@type") == "Product":
                return item
    return {}


def model_jsonld_urls(soup: BeautifulSoup) -> list[str]:
    urls: list[str] = []
    for data in iter_jsonld(soup):
        candidates = data if isinstance(data, list) else [data]
        for item in candidates:
            if not isinstance(item, dict) or item.get("@type") != "3DModel":
                continue
            for enc in item.get("encoding") or []:
                if isinstance(enc, dict) and enc.get("contentUrl"):
                    urls.append(normalize_url(enc["contentUrl"], keep_query=True))
    return unique_list(urls)


def category_from_url_or_jsonld(url: str, jsonld: dict[str, Any]) -> tuple[str, str]:
    raw = norm_text(jsonld.get("category"))
    text = lower(raw + " " + url)
    if any(x in text for x in ["sofa", "couch", "sessel", "recamiere", "récamiere"]):
        return raw or "Sofas & Sessel", "sofa" if "sofa" in text or "couch" in text else "armchair"
    if any(x in text for x in ["stuhl", "chair"]):
        return raw or "Stühle", "chair"
    if any(x in text for x in ["tisch", "table"]):
        return raw or "Tische", "table"
    if any(x in text for x in ["bett", "matratze", "bed"]):
        return raw or "Betten", "bed"
    if any(x in text for x in ["schrank", "regal", "kommode", "cabinet", "shelf"]):
        return raw or "Aufbewahrung", "cabinet"
    if any(x in text for x in ["leuchte", "lampe", "lighting", "led"]):
        return raw or "Leuchten", "lighting"
    return raw or "IKEA", "furniture"


def parse_product(url: str, html_text: str, source: dict[str, Any] | None = None) -> IkeaCard:
    source = source or {}
    soup = BeautifulSoup(html_text, "html.parser")
    compact, page_props = find_product_hydrate(soup)
    compact_product = compact.get("product") or {}
    page_product = page_props.get("product") or {}
    jsonld = product_jsonld(soup)

    item_no = norm_text(page_product.get("itemNo") or compact_product.get("itemNo") or jsonld.get("sku") or parse_item_no_from_url(url))
    visible_item_no = norm_text(page_product.get("visibleItemNo") or compact_product.get("visibleItemNo") or jsonld.get("mpn"))
    name = norm_text(page_product.get("name") or compact_product.get("name") or jsonld.get("name"))
    type_name = norm_text(compact_product.get("typeName") or "")
    description_short = norm_text(page_product.get("description") or compact_product.get("description"))
    title = norm_text(jsonld.get("name") or " ".join(x for x in [name, description_short] if x))
    description = norm_text(jsonld.get("description") or description_short or title)
    category_raw, category_norm = category_from_url_or_jsonld(url, jsonld)
    if not category_raw and source.get("listing_category_raw"):
        category_raw = norm_text(source.get("listing_category_raw"))
    dims, measurements_raw, package_measurements = parse_dimensions(page_props, compact_product)
    materials = parse_materials(page_props)
    info_texts = parse_info_texts(page_props)
    if info_texts:
        description = norm_text(description + " " + " ".join(info_texts[:3]))

    media = page_product.get("mediaList") or compact_product.get("mediaList") or []
    image_items: list[dict[str, Any]] = []
    for item in media:
        content = item.get("content") if isinstance(item, dict) else None
        if isinstance(content, dict) and content.get("url"):
            image_items.append(content)
    main_image = next((x for x in image_items if x.get("type") == "MAIN_PRODUCT_IMAGE"), None) or (image_items[0] if image_items else None)
    image_url = normalize_url(main_image.get("url"), keep_query=True) if main_image else ""
    image_alt = norm_text(main_image.get("alt")) if main_image else ""
    all_images = unique_list([normalize_url(x.get("url"), keep_query=True) for x in image_items if x.get("url")])
    if not image_url:
        image_node = jsonld.get("image")
        if isinstance(image_node, list) and image_node:
            first = image_node[0]
            image_url = normalize_url(first.get("contentUrl") if isinstance(first, dict) else str(first), keep_query=True)
        elif isinstance(image_node, dict):
            image_url = normalize_url(image_node.get("contentUrl") or image_node.get("url") or "", keep_query=True)
        elif isinstance(image_node, str):
            image_url = normalize_url(image_node, keep_query=True)
    if not image_url and source.get("image"):
        image_url = normalize_url(source.get("image"), keep_query=True)

    offers = jsonld.get("offers") if isinstance(jsonld.get("offers"), dict) else {}
    price = page_product.get("price") or compact_product.get("price") or offers.get("price")
    if price is None and source.get("listing_price") is not None:
        price = source.get("listing_price")
    try:
        price_value = float(price) if price is not None else None
    except Exception:
        price_value = None
    availability = norm_text(offers.get("availability") or ("online_sellable" if page_product.get("isServerOnlineSellable") else "unknown"))
    rating = page_product.get("rating") or (jsonld.get("aggregateRating") or {}).get("ratingValue")
    reviews = page_product.get("reviewCount") or (jsonld.get("aggregateRating") or {}).get("reviewCount")
    try:
        rating_value = float(rating) if rating is not None else None
    except Exception:
        rating_value = None
    try:
        reviews_count = int(reviews) if reviews is not None else None
    except Exception:
        reviews_count = None

    return IkeaCard(
        source_page_url=source.get("source_page_url", ""),
        source_category_url=source.get("source_category_url", ""),
        source_page_number=int(source.get("source_page_number") or 0),
        product_url=normalize_url(url),
        item_no=item_no,
        visible_item_no=visible_item_no,
        title=title or name or f"IKEA product {item_no}",
        name=name,
        type_name=type_name,
        description=description,
        category_raw=category_raw,
        category_norm=category_norm,
        price=price_value,
        price_currency=norm_text(page_product.get("currencyCode") or compact_product.get("currencyCode") or offers.get("priceCurrency") or source.get("listing_price_currency") or "EUR"),
        availability=availability,
        rating=rating_value,
        reviews_count=reviews_count,
        color=norm_text(jsonld.get("color")),
        materials=materials,
        dimensions_cm=dims,
        measurements_raw=measurements_raw,
        package_measurements=package_measurements,
        image_url=image_url,
        image_alt=image_alt,
        all_product_images=all_images,
        model_urls=model_jsonld_urls(soup),
        extra={
            "commercial_label": page_product.get("commercialLabel"),
            "item_measure_reference_text": page_product.get("itemMeasureReferenceText") or compact_product.get("itemMeasureReferenceText"),
            "info_texts": info_texts,
            "listing_name": source.get("name"),
            "listing_image": source.get("image"),
            "position": source.get("position"),
            "listing_item_no": source.get("listing_item_no"),
            "listing_type_name": source.get("listing_type_name"),
            "listing_valid_design_text": source.get("listing_valid_design_text"),
            "listing_colors": source.get("listing_colors"),
            "listing_category_path": source.get("listing_category_path"),
        },
    )


def fetch_product_card(sess: requests.Session, source: dict[str, Any], sleep: float) -> IkeaCard:
    url = source["url"] if isinstance(source, dict) else str(source)
    try:
        html_text = fetch(sess, url, sleep=sleep)
        return parse_product(url, html_text, source if isinstance(source, dict) else {})
    except Exception as exc:
        return IkeaCard(product_url=normalize_url(url), item_no=parse_item_no_from_url(url), raw_status="error", error=f"{type(exc).__name__}: {exc}")


def room_for_category(category_norm: str) -> str | None:
    if category_norm in {"sofa", "armchair", "cabinet", "chair", "table", "lighting"}:
        return "living_room"
    if category_norm == "bed":
        return "bedroom"
    return None


def to_canonical(row: IkeaCard, parsed_at: str, *, promote_model_urls: bool = False) -> dict[str, Any]:
    title = row.title or f"IKEA product {row.item_no}"
    tags = unique_list(["IKEA", "ikea.de", row.category_norm, row.category_raw, row.type_name, row.color, row.materials])
    model_url = next((u for u in row.model_urls if ".glb" in u.lower()), None) if promote_model_urls else None
    return {
        "unique_key": f"ikea_de::item::{row.item_no}",
        "source_site": "ikea.com/de/de",
        "source_db": None,
        "source_url": row.product_url,
        "parsed_at": parsed_at,
        "external_id": row.item_no,
        "title": title,
        "brand": "IKEA",
        "collection": row.name or None,
        "category_raw": row.category_raw or row.type_name,
        "category_norm": row.category_norm or "furniture",
        "product_url": row.product_url,
        "model_link_type": "direct_glb" if model_url else None,
        "model_page_url": row.product_url if model_url else None,
        "model_download_url": model_url,
        "model_download_landing_url": row.product_url if model_url else None,
        "model_vendor_url": row.product_url,
        "model_extraction_method": "ikea_de_product_hydrate_jsonld",
        "model_download_filename": Path(urlparse(model_url or "").path).name if model_url else None,
        "model_format": "glb" if model_url else None,
        "asset_status": "direct_model_url_available" if model_url else "product_card_no_3d_asset",
        "asset_format": "glb" if model_url else None,
        "asset_local_path": None,
        "preview_local_path": None,
        "asset_source_url": model_url,
        "price_value": row.price,
        "price_currency": row.price_currency or "EUR",
        "old_price_value": None,
        "style": "not_specified",
        "color": row.color or "not_specified",
        "description": row.description or title,
        "dimensions_cm": row.dimensions_cm or empty_dimensions(),
        "scheme_url": None,
        "room": room_for_category(row.category_norm),
        "materials": row.materials or None,
        "availability": row.availability or "unknown",
        "country_brand": "Sweden",
        "production_country": None,
        "tags": tags,
        "images": [row.image_url] if row.image_url else [],
        "related": [],
        "extra": {
            "ikea_de": {
                "visible_item_no": row.visible_item_no,
                "name": row.name,
                "type_name": row.type_name,
                "source_category_url": row.source_category_url,
                "source_page_url": row.source_page_url,
                "source_page_number": row.source_page_number,
                "rating": row.rating,
                "reviews_count": row.reviews_count,
                "image_alt": row.image_alt,
                "all_product_images": row.all_product_images,
                "model_urls": row.model_urls,
                "model_urls_promoted_to_asset": promote_model_urls,
                "measurements_raw": row.measurements_raw,
                "package_measurements": row.package_measurements,
                **row.extra,
            }
        },
        "completeness": {
            "has_title": bool(title),
            "has_price": row.price is not None,
            "has_full_dimensions": bool((row.dimensions_cm or {}).get("width") and (row.dimensions_cm or {}).get("depth") and (row.dimensions_cm or {}).get("height")),
            "has_description": bool(row.description),
            "has_category": bool(row.category_norm),
            "has_brand": True,
            "has_model_link": bool(model_url),
            "rich_card": bool(title and row.price is not None and row.category_norm and row.image_url),
        },
    }


def write_json(path: Path, data: Any) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[IkeaCard]) -> None:
    fields = ["item_no", "title", "category_norm", "price", "price_currency", "product_url", "image_url", "dimensions_cm", "raw_status", "error"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            item = asdict(row)
            item["dimensions_cm"] = json.dumps(row.dimensions_cm, ensure_ascii=False)
            writer.writerow({k: item.get(k) for k in fields})


def write_image_manifest(path: Path, rows: list[IkeaCard]) -> None:
    manifest = [
        {
            "unique_key": f"ikea_de::item::{row.item_no}",
            "item_no": row.item_no,
            "title": row.title,
            "product_url": row.product_url,
            "image_rank": 1,
            "image_source": "main_product_image",
            "image_url": row.image_url,
        }
        for row in rows
        if row.image_url
    ]
    write_jsonl(path.with_suffix(".jsonl"), manifest)
    with path.with_suffix(".csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["unique_key", "item_no", "title", "product_url", "image_rank", "image_source", "image_url"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest)


def merge_catalog(catalog_path: Path, cards: list[dict[str, Any]], parsed_at: str) -> tuple[int, int]:
    data = json.loads(catalog_path.read_text(encoding="utf-8"))
    items = data.setdefault("items", [])
    index = {item.get("unique_key"): i for i, item in enumerate(items) if isinstance(item, dict)}
    added = 0
    updated = 0
    for card in cards:
        key = card.get("unique_key")
        if key in index:
            items[index[key]] = card
            updated += 1
        else:
            items.append(card)
            added += 1
    meta = data.setdefault("meta", {})
    meta["item_count"] = len(items)
    meta.setdefault("manual_merges", []).append(
        {
            "source": "ikea_de_catalog_scraper",
            "updated_at": parsed_at,
            "added": added,
            "updated": updated,
            "item_count_after": len(items),
        }
    )
    write_json(catalog_path, data)
    return added, updated


def load_catalog_keys(catalog_path: Path) -> set[str]:
    data = json.loads(catalog_path.read_text(encoding="utf-8"))
    return {str(item.get("unique_key")) for item in data.get("items", []) if isinstance(item, dict) and item.get("unique_key")}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scrape IKEA Germany product cards.")
    parser.add_argument("--product-url", action="append", default=[], help="Product URL. Can be passed multiple times.")
    parser.add_argument("--product-url-file", default="", help="Text file with one product URL per line.")
    parser.add_argument("--category-url", action="append", default=[], help="IKEA category URL. Can be passed multiple times.")
    parser.add_argument("--use-default-categories", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--all-products-de", action="store_true", help="Collect all IKEA Germany product URLs from prod-de-DE sitemaps.")
    parser.add_argument("--sitemap-index-url", default=DEFAULT_SITEMAP_INDEX_URL)
    parser.add_argument("--sitemap-locale-marker", default="prod-de-DE_", help="Sitemap filename marker, default prod-de-DE_.")
    parser.add_argument("--out", default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-pages", type=int, default=1, help="Pages per category. Use 0 for all pages linked by rel=next.")
    parser.add_argument("--limit", type=int, default=0, help="Limit product detail cards after URL collection. 0 means no limit.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sleep", type=float, default=0.25)
    parser.add_argument("--merge-catalog", default="", help="Optional path to supplier_catalog_canonical.json to upsert scraped cards.")
    parser.add_argument("--merge-every", type=int, default=0, help="When --merge-catalog is set, upsert every N successful cards during the run. 0 means only at the end.")
    parser.add_argument("--skip-existing-in-merge-catalog", action="store_true", help="Skip IKEA item ids already present in --merge-catalog.")
    parser.add_argument("--progress-every", type=int, default=100, help="Print collected progress every N successful cards.")
    parser.add_argument(
        "--promote-model-urls",
        action="store_true",
        help="Put IKEA JSON-LD GLB URLs into canonical model_download_url. By default they are kept only in extra.ikea_de.model_urls.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out)
    ensure_dir(out_dir)
    parsed_at = now_iso()
    sess = session()

    sources: list[dict[str, Any]] = []
    for url in args.product_url:
        sources.append({"url": normalize_url(url)})
    if args.product_url_file:
        for line in Path(args.product_url_file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                sources.append({"url": normalize_url(line)})

    if args.all_products_de:
        sources.extend(
            collect_product_urls_from_sitemaps(
                sess,
                args.sitemap_index_url,
                locale_marker=args.sitemap_locale_marker,
                limit=args.limit,
                sleep=args.sleep,
            )
        )

    category_urls = list(args.category_url)
    if args.use_default_categories and not category_urls:
        category_urls = DEFAULT_CATEGORY_URLS
    if category_urls:
        sources.extend(collect_product_urls(sess, category_urls, args.max_pages, args.sleep, args.limit))

    seen: set[str] = set()
    unique_sources: list[dict[str, Any]] = []
    for source in sources:
        url = normalize_url(source.get("url") or source.get("product_url") or "")
        if not url or url in seen:
            continue
        source["url"] = url
        seen.add(url)
        unique_sources.append(source)
    if args.limit > 0:
        unique_sources = unique_sources[: args.limit]
    if args.skip_existing_in_merge_catalog and args.merge_catalog:
        existing_keys = load_catalog_keys(Path(args.merge_catalog))
        before_skip = len(unique_sources)
        unique_sources = [
            source
            for source in unique_sources
            if f"ikea_de::item::{source.get('listing_item_no') or parse_item_no_from_url(source.get('url') or '')}" not in existing_keys
        ]
        eprint(f"[resume] skipped_existing={before_skip - len(unique_sources)} remaining={len(unique_sources)}")
    eprint(f"[detail] urls={len(unique_sources)}")

    rows: list[IkeaCard] = []
    ok_count = 0
    incremental_merge_buffer: list[IkeaCard] = []
    incrementally_merged_keys: set[str] = set()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(fetch_product_card, sess, source, args.sleep) for source in unique_sources]
        for i, future in enumerate(as_completed(futures), 1):
            row = future.result()
            rows.append(row)
            if row.raw_status == "ok":
                ok_count += 1
                incremental_merge_buffer.append(row)
                progress_every = max(1, args.progress_every)
                if ok_count % progress_every == 0:
                    eprint(f"[progress] collected={ok_count}")
                if args.merge_catalog and args.merge_every > 0 and len(incremental_merge_buffer) >= args.merge_every:
                    batch_cards = [to_canonical(item, parsed_at, promote_model_urls=args.promote_model_urls) for item in incremental_merge_buffer]
                    added, updated = merge_catalog(Path(args.merge_catalog), batch_cards, parsed_at)
                    incrementally_merged_keys.update(card["unique_key"] for card in batch_cards)
                    eprint(f"[merge:batch] {args.merge_catalog}: batch={len(batch_cards)} added={added} updated={updated} collected={ok_count}")
                    incremental_merge_buffer.clear()
            if i % 20 == 0 or i == len(futures):
                eprint(f"[detail] {i}/{len(futures)}")
            time.sleep(args.sleep)

    rows.sort(key=lambda r: (r.source_category_url, r.source_page_number, r.product_url))
    canonical_cards = [to_canonical(row, parsed_at, promote_model_urls=args.promote_model_urls) for row in rows if row.raw_status == "ok"]
    raw_rows = [asdict(row) for row in rows]
    export = {
        "schema": "ikea_de_catalog_scrape/v1",
        "meta": {
            "source_site": "ikea.com/de/de",
            "parsed_at": parsed_at,
            "product_url_count": len(unique_sources),
            "raw_item_count": len(rows),
            "canonical_item_count": len(canonical_cards),
            "category_urls": category_urls,
        },
        "items": raw_rows,
        "canonical_items": canonical_cards,
    }
    write_json(out_dir / "ikea_de_catalog_scrape.json", export)
    write_json(out_dir / "ikea_de_supplier_catalog.json", {"schema": "supplier_catalog_export/v1", "meta": export["meta"], "items": canonical_cards})
    write_jsonl(out_dir / "ikea_de_supplier_catalog.jsonl", canonical_cards)
    write_csv(out_dir / "ikea_de_catalog_cards.csv", rows)
    write_image_manifest(out_dir / "ikea_de_image_urls", rows)
    eprint(f"[out] {out_dir / 'ikea_de_catalog_scrape.json'}")
    eprint(f"[out] {out_dir / 'ikea_de_supplier_catalog.json'}")
    eprint(f"[out] {out_dir / 'ikea_de_image_urls.jsonl'}")
    eprint(f"[out] {out_dir / 'ikea_de_image_urls.csv'}")
    if args.merge_catalog:
        if args.merge_every > 0:
            remaining_cards = [card for card in canonical_cards if card["unique_key"] not in incrementally_merged_keys]
            added, updated = merge_catalog(Path(args.merge_catalog), remaining_cards, parsed_at) if remaining_cards else (0, 0)
        else:
            added, updated = merge_catalog(Path(args.merge_catalog), canonical_cards, parsed_at)
        eprint(f"[merge] {args.merge_catalog}: added={added} updated={updated}")
    eprint(f"[done] items={len(rows)} canonical={len(canonical_cards)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
