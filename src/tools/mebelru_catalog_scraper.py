#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Scrape mebel.ru catalog cards into supplier-catalog compatible JSON.

The mebel.ru listing HTML contains rich Vue component attributes on every card:
product id, title, price, detail page URL, affiliate/away URL and often the
gallery URLs. This scraper starts from listing pages, optionally opens the
mebel.ru detail page and the final seller page, and exports both raw rows and
canonical supplier catalog cards.

Examples:
    python3 -m src.tools.mebelru_catalog_scraper \
      --out out/supplier_ingest/mebelru/catalog --max-pages 1

    # Keep all collected image URLs instead of canonical first two.
    python3 -m src.tools.mebelru_catalog_scraper --max-images 0

    # Merge scraped canonical cards into the main supplier catalog.
    python3 -m src.tools.mebelru_catalog_scraper \
      --merge-catalog data/sourse/suppliers/supplier_catalog_canonical.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, unquote, urlencode, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup, Tag


BASE_URL = "https://mebel.ru"
REST_BASE_URL = "https://mebel.ru/rest"
DEFAULT_CATEGORY_URLS = [
    "https://mebel.ru/catalog/sofas/?PAGEN_1=8",
    "https://mebel.ru/catalog/armchairs/",
    "https://mebel.ru/catalog/cabinets/",
    "https://mebel.ru/catalog/kitchens/",
]
DEFAULT_OUT_DIR = "out/supplier_ingest/mebelru/catalog"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/121.0.0.0 Safari/537.36"
)

CATEGORY_NORM_BY_PATH = {
    "sofas": "sofa",
    "armchairs": "armchair",
    "cabinets": "cabinet",
    "kitchens": "kitchen_set",
}


@dataclass
class MebelCard:
    source_category_url: str
    source_page_url: str
    source_page_number: int
    mebel_id: str = ""
    title: str = ""
    type_raw: str = ""
    product_type: str = ""
    category_path: str = ""
    category_norm: str = ""
    listing_url: str = ""
    detail_page_url: str = ""
    away_url: str = ""
    affiliate_url: str = ""
    vendor_url: str = ""
    seller_id: str = ""
    seller_code: str = ""
    seller_name: str = ""
    brand: str = ""
    price: float | None = None
    old_price: float | None = None
    price_currency: str = "RUB"
    rating: float | None = None
    reviews_count: int | None = None
    description: str = ""
    materials: str = ""
    color: str = ""
    style: str = ""
    availability: str = ""
    dimensions_cm: dict[str, float | None] = field(default_factory=dict)
    properties: dict[str, str] = field(default_factory=dict)
    images_listing: list[str] = field(default_factory=list)
    images_detail: list[str] = field(default_factory=list)
    images_vendor: list[str] = field(default_factory=list)
    canonical_images: list[str] = field(default_factory=list)
    parse_status: str = "ok"
    detail_status: str = "not_fetched"
    vendor_status: str = "not_fetched"
    error: str = ""


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def norm_text(value: Any) -> str:
    text = str(value or "").replace("\xa0", " ").replace("&nbsp;", " ")
    return re.sub(r"\s+", " ", text).strip()


def lower(value: Any) -> str:
    return norm_text(value).lower().replace("ё", "е")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def now_unix() -> float:
    return time.time()


def absolutize(url: str, base_url: str = BASE_URL) -> str:
    return urljoin(base_url, url or "")


def normalize_url(url: str, *, base_url: str = BASE_URL, keep_query: bool = False) -> str:
    if not url:
        return ""
    parsed = urlparse(absolutize(url, base_url))
    query = parsed.query if keep_query else ""
    path = parsed.path or "/"
    return urlunparse((parsed.scheme, parsed.netloc, path, "", query, ""))


def normalize_image_url(url: str, *, base_url: str = BASE_URL) -> str:
    if not url:
        return ""
    if url.startswith("data:"):
        return ""
    return normalize_url(url, base_url=base_url, keep_query=True)


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


def parse_price(value: Any) -> float | None:
    text = norm_text(value)
    if not text:
        return None
    text = text.replace(" ", "").replace(",", ".")
    match = re.search(r"\d+(?:\.\d+)?", text)
    if not match:
        return None
    return float(match.group(0))


def parse_int(value: Any) -> int | None:
    digits = re.sub(r"\D+", "", norm_text(value))
    return int(digits) if digits else None


def parse_rating(value: Any) -> float | None:
    text = norm_text(value).replace(",", ".")
    match = re.search(r"\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else None


def parse_dimension_triplet(text: str) -> dict[str, float | None]:
    text = norm_text(text).lower().replace(",", ".")
    match = re.search(r"(\d+(?:\.\d+)?)\s*[xх×]\s*(\d+(?:\.\d+)?)\s*[xх×]\s*(\d+(?:\.\d+)?)\s*(?:см|cm)?", text)
    if not match:
        return empty_dimensions()
    width, depth, height = (float(match.group(i)) for i in range(1, 4))
    return dimensions(width=width, depth=depth, height=height)


def empty_dimensions() -> dict[str, float | None]:
    return dimensions()


def dimensions(
    *,
    width: float | None = None,
    depth: float | None = None,
    height: float | None = None,
    weight_kg: float | None = None,
    package_width: float | None = None,
    package_depth: float | None = None,
    package_height: float | None = None,
    packed_weight_kg: float | None = None,
    volume_m3: float | None = None,
) -> dict[str, float | None]:
    return {
        "width": width,
        "depth": depth,
        "height": height,
        "weight_kg": weight_kg,
        "package_width": package_width,
        "package_depth": package_depth,
        "package_height": package_height,
        "packed_weight_kg": packed_weight_kg,
        "volume_m3": volume_m3,
    }


def category_path_from_url(url: str) -> str:
    parts = [p for p in urlparse(url).path.split("/") if p]
    if len(parts) >= 2 and parts[0] == "catalog":
        return parts[1]
    return ""


def category_norm_from_url(url: str, raw: str = "") -> str:
    path = category_path_from_url(url)
    if path in CATEGORY_NORM_BY_PATH:
        return CATEGORY_NORM_BY_PATH[path]
    text = lower(raw)
    if "диван" in text:
        return "sofa"
    if "крес" in text or "оттоман" in text:
        return "armchair"
    if "шкаф" in text or "стеллаж" in text:
        return "cabinet"
    if "кух" in text:
        return "kitchen_set"
    return path or "furniture"


def build_page_url(category_url: str, page_number: int) -> str:
    parsed = urlparse(category_url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    if page_number <= 1:
        query.pop("PAGEN_1", None)
    else:
        query["PAGEN_1"] = str(page_number)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", urlencode(query), ""))


def starting_page(category_url: str) -> int:
    query = dict(parse_qsl(urlparse(category_url).query, keep_blank_values=True))
    return parse_int(query.get("PAGEN_1")) or 1


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
        }
    )
    return session


def fetch_html(session: requests.Session, url: str, timeout: tuple[int, int] = (20, 70)) -> tuple[str | None, str, str]:
    try:
        response = session.get(url, timeout=timeout, allow_redirects=True)
        response.raise_for_status()
        response.encoding = response.encoding or "utf-8"
        return response.text, response.url, ""
    except Exception as exc:  # noqa: BLE001 - per-card errors are exported.
        return None, url, f"{type(exc).__name__}: {exc}"


def fetch_json(session: requests.Session, url: str, timeout: tuple[int, int] = (20, 70)) -> tuple[dict[str, Any] | None, str, str]:
    try:
        response = session.get(
            url,
            timeout=timeout,
            allow_redirects=True,
            headers={"Accept": "application/json, text/plain, */*"},
        )
        response.raise_for_status()
        return response.json(), response.url, ""
    except Exception as exc:  # noqa: BLE001 - per-page errors are exported.
        return None, url, f"{type(exc).__name__}: {exc}"


def decode_away_url(href_or_url: str) -> tuple[str, str]:
    """Return (affiliate_url, vendor_url) from mebel.ru away or direct attr URL."""
    if not href_or_url:
        return "", ""
    url = absolutize(href_or_url)
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    affiliate = unquote(query.get("to") or href_or_url)
    vendor = extract_ulp_url(affiliate)
    return affiliate, vendor


def extract_ulp_url(url: str) -> str:
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    ulp = query.get("ulp") or query.get("url") or query.get("u")
    if ulp:
        return unquote(ulp)
    return url if parsed.scheme in {"http", "https"} and "mebel.ru" not in parsed.netloc else ""


def attr(node: Tag | None, name: str) -> str:
    if not node:
        return ""
    value = node.get(name)
    if isinstance(value, list):
        return " ".join(str(x) for x in value)
    return str(value or "")


def extract_images_from_attrs(node: Tag | None, *, base_url: str = BASE_URL) -> list[str]:
    if not node:
        return []
    images: list[str] = []
    for key in ["pictures", "picture", "src", "data-src", "data-original", "srcset"]:
        raw = attr(node, key)
        if not raw:
            continue
        for part in raw.split(","):
            candidate = part.strip().split(" ")[0]
            image = normalize_image_url(candidate, base_url=base_url)
            if image:
                images.append(image)
    return unique_list(images)


def extract_card_images(card: Tag) -> list[str]:
    images: list[str] = []
    price = card.select_one(".price")
    slider = card.select_one(".slider")
    images.extend(extract_images_from_attrs(price))
    images.extend(extract_images_from_attrs(slider))
    for img in card.select(".slider img[src], .slider img[data-src], .slider source[srcset]"):
        images.extend(extract_images_from_attrs(img))
    return unique_list(images)


def extract_seller(card: Tag) -> tuple[str, str]:
    partner = card.select_one(".partner")
    seller_code = ""
    seller_name = ""
    if partner:
        seller_code = attr(partner, "code")
        logo = partner.select_one("img[alt]")
        seller_name = attr(logo, "alt")
    if not seller_name:
        link = card.select_one(".partner__link[href]")
        if link:
            seller_code = seller_code or urlparse(attr(link, "href")).path.strip("/").split("/")[-1]
    return seller_code, seller_name


def extract_card_rows(html_text: str, page_url: str, category_url: str, page_number: int) -> list[MebelCard]:
    soup = BeautifulSoup(html_text, "html.parser")
    cards = soup.select("li.item.item_theme_grid")
    rows: list[MebelCard] = []
    seen: set[str] = set()
    for card in cards:
        price_node = card.select_one(".price")
        slider_node = card.select_one(".slider")
        data_node = price_node or slider_node
        if not data_node:
            continue
        mebel_id = attr(data_node, "id") or attr(slider_node, "id")
        if not mebel_id or mebel_id in seen:
            continue
        seen.add(mebel_id)

        raw_detail = attr(data_node, "detailpageurl") or attr(slider_node, "detailpageurl")
        detail_url = normalize_url(raw_detail) if raw_detail else ""
        away_anchor = card.select_one('a[href*="/away/"]')
        affiliate_url, vendor_url = decode_away_url(attr(data_node, "url") or attr(away_anchor, "href"))
        seller_code, seller_name = extract_seller(card)
        title = norm_text(attr(data_node, "name")) or norm_text(card.select_one(".link_theme_grid").get_text(" ", strip=True) if card.select_one(".link_theme_grid") else "")
        category_norm = category_norm_from_url(detail_url or page_url, attr(data_node, "type") or attr(data_node, "producttype"))
        category_path = category_path_from_url(detail_url or page_url)
        images = extract_card_images(card)
        price = parse_price(attr(data_node, "price"))
        old_price = parse_price(attr(data_node, "oldprice"))
        if price is None:
            price = parse_price(card.select_one(".price__cost").get_text(" ", strip=True) if card.select_one(".price__cost") else "")
        if old_price is None:
            old_price = parse_price(card.select_one(".price__old-cost").get_text(" ", strip=True) if card.select_one(".price__old-cost") else "")

        rating = parse_rating(card.select_one(".rating-block__number").get_text(" ", strip=True) if card.select_one(".rating-block__number") else "")
        reviews_count = parse_int(card.select_one(".feedback__link").get_text(" ", strip=True) if card.select_one(".feedback__link") else "")
        dims = parse_dimension_triplet(title)

        rows.append(
            MebelCard(
                source_category_url=category_url,
                source_page_url=page_url,
                source_page_number=page_number,
                mebel_id=mebel_id,
                title=title,
                type_raw=attr(data_node, "type"),
                product_type=attr(data_node, "producttype"),
                category_path=category_path,
                category_norm=category_norm,
                listing_url=page_url,
                detail_page_url=detail_url,
                away_url=normalize_url(attr(away_anchor, "href"), keep_query=True) if away_anchor else "",
                affiliate_url=affiliate_url,
                vendor_url=vendor_url,
                seller_id=attr(data_node, "sellerid"),
                seller_code=seller_code,
                seller_name=seller_name,
                brand=seller_name or seller_code or None,
                price=price,
                old_price=old_price,
                rating=rating,
                reviews_count=reviews_count,
                description=norm_text(attr(data_node, "description")),
                dimensions_cm=dims,
                images_listing=images,
                canonical_images=images,
            )
        )
    return rows


def api_category_from_url(category_url: str) -> str:
    return category_path_from_url(category_url)


def build_api_products_url(category_url: str, page_number: int) -> str:
    parsed = urlparse(category_url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    category = api_category_from_url(category_url)
    api_query: dict[str, str] = {"category": category or "root", "page": str(page_number)}
    sort_by_typical = {
        "deshevye": "price",
        "dorogie": "pricedesc",
        "novinki": "date",
    }
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) >= 3:
        typical = parts[2]
        if typical in sort_by_typical:
            api_query["sort"] = sort_by_typical[typical]
        else:
            api_query["typical"] = typical
    for key, value in query.items():
        if key == "PAGEN_1":
            continue
        api_query[key] = value
    return f"{REST_BASE_URL}/catalog/products/?{urlencode(api_query, doseq=True)}"


def product_to_card(
    product: dict[str, Any],
    sellers: dict[str, Any],
    *,
    category_url: str,
    page_url: str,
    page_number: int,
) -> MebelCard | None:
    mebel_id = norm_text(product.get("id"))
    if not mebel_id:
        return None
    raw_detail = norm_text(product.get("detailPageUrl"))
    detail_url = normalize_url(raw_detail) if raw_detail else ""
    affiliate_url, vendor_url = decode_away_url(norm_text(product.get("url")))
    seller_id = norm_text(product.get("sellerId"))
    seller = sellers.get(seller_id) or sellers.get(str(product.get("sellerId"))) or {}
    reviews = seller.get("reviews") if isinstance(seller, dict) else {}
    pictures = product.get("pictures")
    images: list[str] = []
    if isinstance(pictures, list):
        images.extend(normalize_image_url(str(url)) for url in pictures)
    images.append(normalize_image_url(norm_text(product.get("picture"))))
    images = unique_list([url for url in images if url])
    category_path = norm_text(product.get("category")) or category_path_from_url(detail_url or page_url)
    category_norm = category_norm_from_url(f"{BASE_URL}/catalog/{category_path}/", norm_text(product.get("type") or product.get("productType")))
    title = norm_text(product.get("name"))
    return MebelCard(
        source_category_url=category_url,
        source_page_url=page_url,
        source_page_number=page_number,
        mebel_id=mebel_id,
        title=title,
        type_raw=norm_text(product.get("type")),
        product_type=norm_text(product.get("productType")),
        category_path=category_path,
        category_norm=category_norm,
        listing_url=page_url,
        detail_page_url=detail_url,
        away_url="",
        affiliate_url=affiliate_url,
        vendor_url=vendor_url,
        seller_id=seller_id,
        seller_code=norm_text(seller.get("code") if isinstance(seller, dict) else ""),
        seller_name=norm_text(seller.get("name") if isinstance(seller, dict) else ""),
        brand=norm_text(seller.get("name") if isinstance(seller, dict) else ""),
        price=parse_price(product.get("price")),
        old_price=parse_price(product.get("oldPrice")),
        rating=parse_rating(reviews.get("avgRating") if isinstance(reviews, dict) else ""),
        reviews_count=parse_int(reviews.get("cnt") if isinstance(reviews, dict) else ""),
        description=norm_text(product.get("description")),
        dimensions_cm=parse_dimension_triplet(title),
        images_listing=images,
        canonical_images=images,
    )


def extract_api_rows(data: dict[str, Any], page_url: str, category_url: str, page_number: int) -> tuple[list[MebelCard], dict[str, Any]]:
    payload = data.get("data") if isinstance(data.get("data"), dict) else data
    products = payload.get("products") if isinstance(payload, dict) else []
    sellers = payload.get("sellers") if isinstance(payload, dict) else {}
    pagination = payload.get("pagination") if isinstance(payload, dict) else {}
    if not isinstance(products, list):
        products = []
    if not isinstance(sellers, dict):
        sellers = {}
    rows = [
        row
        for product in products
        if isinstance(product, dict)
        for row in [product_to_card(product, sellers, category_url=category_url, page_url=page_url, page_number=page_number)]
        if row is not None
    ]
    return rows, pagination if isinstance(pagination, dict) else {}


def extract_jsonld_objects(soup: BeautifulSoup) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = (script.string or script.get_text() or "").strip()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        queue = data if isinstance(data, list) else [data]
        for item in queue:
            if isinstance(item, dict):
                objects.append(item)
                graph = item.get("@graph")
                if isinstance(graph, list):
                    objects.extend(x for x in graph if isinstance(x, dict))
    return objects


def extract_jsonld_product(soup: BeautifulSoup) -> dict[str, Any] | None:
    for obj in extract_jsonld_objects(soup):
        types = obj.get("@type")
        type_list = types if isinstance(types, list) else [types]
        if "Product" in type_list:
            return obj
    return None


def extract_detail_images(soup: BeautifulSoup, page_url: str) -> list[str]:
    images: list[str] = []
    for node in soup.select('[itemprop="image"]'):
        images.extend(extract_images_from_attrs(node, base_url=page_url))
        images.extend(extract_images_from_attrs(node if node.name in {"img", "source"} else node.select_one("img, source"), base_url=page_url))
        if node.name == "meta":
            images.append(normalize_image_url(attr(node, "content"), base_url=page_url))
        if node.name == "a":
            images.append(normalize_image_url(attr(node, "href"), base_url=page_url))
    for node in soup.select(".product img[src], .product source[srcset], .slider__image, .detail-preview img, .detail-preview source"):
        images.extend(extract_images_from_attrs(node, base_url=page_url))
    return unique_list([x for x in images if is_product_like_image(x)])


def is_product_like_image(url: str) -> bool:
    if not url:
        return False
    parsed = urlparse(url)
    path = parsed.path.lower()
    if path.endswith(".svg"):
        return False
    deny = ["/upload/iblock/891/", "/upload/iblock/d60/", "/upload/iblock/78f/", "/_nuxt/"]
    return not any(part in path for part in deny)


def extract_properties_from_pairs(soup: BeautifulSoup) -> dict[str, str]:
    props: dict[str, str] = {}
    for li in soup.select(".detail-properties__list li, .property-list li, .features li, .characteristics li"):
        spans = [norm_text(x.get_text(" ", strip=True)) for x in li.select("span, div")]
        spans = [x for x in spans if x]
        if len(spans) >= 2:
            props.setdefault(spans[0], spans[-1])
    for row in soup.select("tr"):
        cells = [norm_text(x.get_text(" ", strip=True)) for x in row.select("th, td")]
        if len(cells) >= 2 and len(cells[0]) <= 80:
            props.setdefault(cells[0], cells[-1])
    return props


def extract_dimensions_from_properties(props: dict[str, str], fallback_text: str = "") -> dict[str, float | None]:
    candidates = []
    for key, value in props.items():
        low = lower(key)
        if "размер" in low or "габарит" in low:
            candidates.append(value)
    candidates.append(fallback_text)
    for text in candidates:
        dims = parse_dimension_triplet(text)
        if dims.get("width") and dims.get("depth") and dims.get("height"):
            return dims
    return empty_dimensions()


def merge_detail(row: MebelCard, html_text: str, final_url: str) -> None:
    soup = BeautifulSoup(html_text, "html.parser")
    product = extract_jsonld_product(soup)
    title_node = soup.select_one("h1")
    if title_node:
        row.title = norm_text(title_node.get_text(" ", strip=True)) or row.title
    if product:
        row.title = norm_text(product.get("name")) or row.title
        row.description = norm_text(product.get("description")) or row.description
        row.brand = norm_text(product.get("brand") if isinstance(product.get("brand"), str) else (product.get("brand") or {}).get("name")) or row.brand
        offers = product.get("offers")
        offer = offers[0] if isinstance(offers, list) and offers else offers if isinstance(offers, dict) else {}
        row.price = parse_price(offer.get("price")) or row.price
        row.price_currency = norm_text(offer.get("priceCurrency")) or row.price_currency
        image = product.get("image")
        if isinstance(image, str):
            row.images_detail.append(normalize_image_url(image, base_url=final_url))
        elif isinstance(image, list):
            row.images_detail.extend(normalize_image_url(str(x), base_url=final_url) for x in image)
    props = extract_properties_from_pairs(soup)
    row.properties.update(props)
    dims = extract_dimensions_from_properties(props, row.title)
    if dims.get("width") and not row.dimensions_cm.get("width"):
        row.dimensions_cm = dims
    row.images_detail.extend(extract_detail_images(soup, final_url))
    row.images_detail = unique_list(row.images_detail)


def merge_vendor(row: MebelCard, html_text: str, final_url: str) -> None:
    soup = BeautifulSoup(html_text, "html.parser")
    if not looks_like_product_page(soup, final_url):
        row.vendor_status = f"ignored_non_product_page: {final_url}"
        return
    product = extract_jsonld_product(soup)
    title_node = soup.select_one("h1")
    if title_node:
        row.title = norm_text(title_node.get_text(" ", strip=True)) or row.title
    if product:
        row.description = norm_text(product.get("description")) or row.description
        offers = product.get("offers")
        offer = offers[0] if isinstance(offers, list) and offers else offers if isinstance(offers, dict) else {}
        row.price = parse_price(offer.get("price")) or row.price
        row.price_currency = norm_text(offer.get("priceCurrency")) or row.price_currency
        image = product.get("image")
        if isinstance(image, str):
            row.images_vendor.append(normalize_image_url(image, base_url=final_url))
        elif isinstance(image, list):
            row.images_vendor.extend(normalize_image_url(str(x), base_url=final_url) for x in image)
    props = extract_properties_from_pairs(soup)
    row.properties.update({k: v for k, v in props.items() if k not in row.properties})
    dims = extract_dimensions_from_properties(props, row.title)
    if dims.get("width"):
        row.dimensions_cm = dims
    row.images_vendor.extend(extract_detail_images(soup, final_url))
    row.images_vendor = unique_list(row.images_vendor)

    color = props.get("Цвет") or props.get("Цвет товара") or props.get("Основной цвет")
    material = props.get("Материал") or props.get("Тип материала") or props.get("Обивка")
    if color:
        row.color = color
    material_parts = [props.get(k) for k in ["Тип материала", "Обивка", "Материал", "Наполнитель"] if props.get(k)]
    row.materials = "; ".join(unique_list([*(row.materials.split("; ") if row.materials else []), *(material_parts or ([material] if material else []))]))


def looks_like_product_page(soup: BeautifulSoup, final_url: str) -> bool:
    parsed = urlparse(final_url)
    path = parsed.path.rstrip("/") + "/"
    if re.fullmatch(r"/catalog/[^/]+/", path):
        return False
    product_signals = [
        soup.select_one(".detail-preview"),
        soup.select_one(".detail-properties__list"),
        soup.select_one("[itemtype*='schema.org/Product'] [itemprop='offers']"),
        soup.select_one("[itemprop='sku']"),
    ]
    if any(product_signals):
        return True
    product = extract_jsonld_product(soup)
    if not product:
        return False
    offers = product.get("offers")
    return bool(product.get("name") and offers)


def enrich_row(
    row: MebelCard,
    *,
    fetch_detail: bool,
    fetch_vendor: bool,
    sleep_sec: float,
    max_images: int,
) -> MebelCard:
    session = make_session()
    if fetch_detail and row.detail_page_url:
        if sleep_sec:
            time.sleep(sleep_sec)
        html, final_url, error = fetch_html(session, row.detail_page_url)
        if html:
            row.detail_status = "ok"
            merge_detail(row, html, final_url)
        else:
            row.detail_status = f"error: {error}"
    if fetch_vendor and row.vendor_url:
        if sleep_sec:
            time.sleep(sleep_sec)
        html, final_url, error = fetch_html(session, row.vendor_url)
        if html:
            row.vendor_status = "ok"
            merge_vendor(row, html, final_url)
            if row.vendor_status == "ok":
                row.vendor_url = final_url
        else:
            row.vendor_status = f"error: {error}"
    images = unique_list([*row.images_vendor, *row.images_detail, *row.images_listing])
    row.canonical_images = images if max_images <= 0 else images[:max_images]
    if not row.dimensions_cm:
        row.dimensions_cm = parse_dimension_triplet(row.title)
    return row


def scrape_listing_pages(args: argparse.Namespace) -> list[MebelCard]:
    session = make_session()
    rows: list[MebelCard] = []
    seen_ids: set[str] = set()
    for category_url in args.category_url:
        start = starting_page(category_url)
        if args.listing_source == "api" and not api_category_from_url(category_url):
            eprint(f"[list:warn] cannot infer API category from {category_url}; falling back to html")
            listing_source = "html"
        else:
            listing_source = args.listing_source
        page_number = start
        pages_seen = 0
        while args.max_pages <= 0 or pages_seen < args.max_pages:
            page_url = build_page_url(category_url, page_number)
            if listing_source == "api":
                api_url = build_api_products_url(category_url, page_number)
                eprint(f"[list:api] {api_url}")
                data, final_url, error = fetch_json(session, api_url)
                if not data:
                    eprint(f"[list:error] {api_url} {error}")
                    break
                page_rows, pagination = extract_api_rows(data, page_url, category_url, page_number)
                total_pages = parse_int(pagination.get("pages")) if pagination else None
            else:
                eprint(f"[list:html] {page_url}")
                html, final_url, error = fetch_html(session, page_url)
                if not html:
                    eprint(f"[list:error] {page_url} {error}")
                    break
                page_rows = extract_card_rows(html, final_url, category_url, page_number)
                total_pages = None
            new_count = 0
            duplicate_count = 0
            for row in page_rows:
                if row.mebel_id in seen_ids:
                    duplicate_count += 1
                    continue
                seen_ids.add(row.mebel_id)
                rows.append(row)
                new_count += 1
            eprint(f"[list] cards={len(page_rows)} new={new_count} duplicates={duplicate_count}")
            if not page_rows and args.stop_on_empty:
                break
            pages_seen += 1
            if total_pages is not None and page_number >= total_pages:
                break
            if args.sleep:
                time.sleep(args.sleep)
            page_number += 1
    return rows


def enrich_rows(args: argparse.Namespace, rows: list[MebelCard]) -> list[MebelCard]:
    if not args.fetch_detail and not args.fetch_vendor:
        for row in rows:
            images = unique_list(row.images_listing)
            row.canonical_images = images if args.max_images <= 0 else images[: args.max_images]
        return rows
    out: list[MebelCard] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                enrich_row,
                row,
                fetch_detail=args.fetch_detail,
                fetch_vendor=args.fetch_vendor,
                sleep_sec=args.sleep,
                max_images=args.max_images,
            )
            for row in rows
        ]
        for i, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            out.append(row)
            if i % 20 == 0 or i == len(futures):
                eprint(f"[detail] {i}/{len(futures)}")
    out.sort(key=lambda x: (x.source_category_url, x.source_page_number, int(x.mebel_id) if x.mebel_id.isdigit() else 0))
    return out


def to_canonical(row: MebelCard, parsed_at: str) -> dict[str, Any]:
    title = row.title or f"mebel.ru product {row.mebel_id}"
    brand = row.brand or row.seller_name or row.seller_code or None
    description = row.description or f"{title}. Карточка товара собрана из каталога mebel.ru."
    source_url = row.vendor_url or row.detail_page_url or row.listing_url
    materials = row.materials or None
    tags = unique_list(
        [
            "mebel.ru",
            row.category_norm,
            row.type_raw,
            row.product_type,
            row.seller_name,
            row.seller_code,
            row.color,
            row.style,
        ]
    )
    return {
        "unique_key": f"mebelru::id::{row.mebel_id}",
        "source_site": "mebel.ru",
        "source_db": None,
        "source_url": source_url,
        "parsed_at": parsed_at,
        "external_id": row.mebel_id,
        "title": title,
        "brand": brand,
        "collection": None,
        "category_raw": row.type_raw or row.product_type or row.category_path,
        "category_norm": row.category_norm or "furniture",
        "product_url": source_url,
        "model_link_type": None,
        "model_page_url": None,
        "model_download_url": None,
        "model_download_landing_url": None,
        "model_vendor_url": row.vendor_url or None,
        "model_extraction_method": "mebelru_catalog_api_or_html",
        "model_download_filename": None,
        "model_format": None,
        "asset_status": "product_card_no_3d_asset",
        "asset_format": None,
        "asset_local_path": None,
        "preview_local_path": None,
        "asset_source_url": None,
        "price_value": row.price,
        "price_currency": row.price_currency or "RUB",
        "old_price_value": row.old_price,
        "style": row.style or "not_specified",
        "color": row.color or "not_specified",
        "description": description,
        "dimensions_cm": row.dimensions_cm or empty_dimensions(),
        "scheme_url": None,
        "room": room_for_category(row.category_norm),
        "materials": materials,
        "availability": row.availability or "unknown",
        "country_brand": "Россия" if row.seller_code == "tsvet-divanov" else None,
        "production_country": None,
        "tags": tags,
        "images": row.canonical_images,
        "related": [],
        "extra": {
            "mebelru": {
                "listing_url": row.listing_url,
                "detail_page_url": row.detail_page_url,
                "away_url": row.away_url,
                "affiliate_url": row.affiliate_url,
                "vendor_url": row.vendor_url,
                "seller_id": row.seller_id,
                "seller_code": row.seller_code,
                "seller_name": row.seller_name,
                "source_category_url": row.source_category_url,
                "source_page_url": row.source_page_url,
                "source_page_number": row.source_page_number,
                "rating": row.rating,
                "reviews_count": row.reviews_count,
                "detail_status": row.detail_status,
                "vendor_status": row.vendor_status,
                "properties": row.properties,
                "images_listing": row.images_listing,
                "images_detail": row.images_detail,
                "images_vendor": row.images_vendor,
            }
        },
        "completeness": {
            "has_title": bool(title),
            "has_price": row.price is not None,
            "has_full_dimensions": bool((row.dimensions_cm or {}).get("width") and (row.dimensions_cm or {}).get("depth") and (row.dimensions_cm or {}).get("height")),
            "has_description": bool(description),
            "has_category": bool(row.category_norm),
            "has_brand": bool(brand),
            "has_model_link": False,
            "rich_card": bool(title and row.price is not None and row.category_norm and row.canonical_images),
        },
    }


def room_for_category(category_norm: str) -> str | None:
    if category_norm in {"sofa", "armchair", "cabinet"}:
        return "living_room"
    if category_norm == "kitchen_set":
        return "kitchen"
    return None


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[MebelCard]) -> None:
    fieldnames = [
        "mebel_id",
        "title",
        "category_norm",
        "price",
        "old_price",
        "price_currency",
        "seller_code",
        "seller_name",
        "detail_page_url",
        "vendor_url",
        "canonical_images",
        "dimensions_cm",
        "detail_status",
        "vendor_status",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            item = asdict(row)
            item["canonical_images"] = "|".join(row.canonical_images)
            item["dimensions_cm"] = json.dumps(row.dimensions_cm, ensure_ascii=False)
            writer.writerow({k: item.get(k) for k in fieldnames})


def build_image_url_manifest(rows: list[MebelCard]) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for row in rows:
        seen: set[str] = set()
        rank = 0
        buckets = [
            ("vendor", row.images_vendor),
            ("detail", row.images_detail),
            ("listing", row.images_listing),
        ]
        for source, urls in buckets:
            for url in urls:
                if not url or url in seen:
                    continue
                seen.add(url)
                rank += 1
                manifest.append(
                    {
                        "mebel_id": row.mebel_id,
                        "unique_key": f"mebelru::id::{row.mebel_id}",
                        "title": row.title,
                        "category_norm": row.category_norm,
                        "seller_code": row.seller_code,
                        "seller_name": row.seller_name,
                        "product_url": row.vendor_url or row.detail_page_url or row.listing_url,
                        "detail_page_url": row.detail_page_url,
                        "vendor_url": row.vendor_url,
                        "image_rank": rank,
                        "image_source": source,
                        "image_url": url,
                    }
                )
    return manifest


def write_image_manifest_csv(path: Path, manifest: list[dict[str, Any]]) -> None:
    fieldnames = [
        "mebel_id",
        "unique_key",
        "title",
        "category_norm",
        "seller_code",
        "seller_name",
        "product_url",
        "detail_page_url",
        "vendor_url",
        "image_rank",
        "image_source",
        "image_url",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in manifest:
            writer.writerow({k: row.get(k) for k in fieldnames})


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
            "source": "mebelru_catalog_scraper",
            "updated_at": parsed_at,
            "added": added,
            "updated": updated,
            "item_count_after": len(items),
        }
    )
    write_json(catalog_path, data)
    return added, updated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scrape mebel.ru catalog cards.")
    parser.add_argument("--category-url", action="append", default=[], help="Category/listing URL. Can be passed multiple times.")
    parser.add_argument("--out", default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-pages", type=int, default=1, help="Pages per category, starting from PAGEN_1 in URL if present. Use 0 for all API pages.")
    parser.add_argument("--listing-source", choices=["api", "html"], default="api", help="Use mebel.ru REST listing API or static HTML cards.")
    parser.add_argument("--max-images", type=int, default=2, help="Images per canonical card. Use 0 for all collected images.")
    parser.add_argument("--fetch-detail", action=argparse.BooleanOptionalAction, default=True, help="Open mebel.ru product detail pages.")
    parser.add_argument("--fetch-vendor", action=argparse.BooleanOptionalAction, default=False, help="Open decoded vendor pages from away/ulp URLs.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--stop-on-empty", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--merge-catalog", default="", help="Optional path to supplier_catalog_canonical.json to upsert scraped cards.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.category_url:
        args.category_url = DEFAULT_CATEGORY_URLS
    out_dir = Path(args.out)
    ensure_dir(out_dir)
    parsed_at = now_iso()

    rows = scrape_listing_pages(args)
    rows = enrich_rows(args, rows)
    raw_rows = [asdict(row) for row in rows]
    canonical_cards = [to_canonical(row, parsed_at) for row in rows]
    image_manifest = build_image_url_manifest(rows)
    export = {
        "schema": "mebelru_catalog_scrape/v1",
        "meta": {
            "parsed_at": parsed_at,
            "source": "mebel.ru",
            "category_urls": args.category_url,
            "max_pages": args.max_pages,
            "listing_source": args.listing_source,
            "max_images": args.max_images,
            "fetch_detail": args.fetch_detail,
            "fetch_vendor": args.fetch_vendor,
            "item_count": len(rows),
            "canonical_item_count": len(canonical_cards),
            "image_url_count": len(image_manifest),
        },
        "items": raw_rows,
        "canonical_items": canonical_cards,
        "image_urls": image_manifest,
    }
    write_json(out_dir / "mebelru_catalog_scrape.json", export)
    write_json(out_dir / "mebelru_supplier_catalog.json", {"schema": "supplier_catalog_export/v1", "meta": export["meta"], "items": canonical_cards})
    write_jsonl(out_dir / "mebelru_supplier_catalog.jsonl", canonical_cards)
    write_jsonl(out_dir / "mebelru_image_urls.jsonl", image_manifest)
    write_image_manifest_csv(out_dir / "mebelru_image_urls.csv", image_manifest)
    write_csv(out_dir / "mebelru_catalog_scrape.csv", rows)
    eprint(f"[out] {out_dir / 'mebelru_catalog_scrape.json'}")
    eprint(f"[out] {out_dir / 'mebelru_supplier_catalog.json'}")
    eprint(f"[out] {out_dir / 'mebelru_image_urls.jsonl'}")
    eprint(f"[out] {out_dir / 'mebelru_image_urls.csv'}")

    if args.merge_catalog:
        added, updated = merge_catalog(Path(args.merge_catalog), canonical_cards, parsed_at)
        eprint(f"[merge] {args.merge_catalog}: added={added} updated={updated}")
    eprint(f"[done] items={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
