#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Scrape Piterra wallpaper products, collection links, and material records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from src.ChooseObject.wall_material_normalizer import (
    WallMaterial,
    analyze_wallpaper_colors,
    normalize_base_material,
    normalize_color_and_tone,
    normalize_pattern,
    normalize_wall_material_type,
)


BASE_URL = "https://www.piterra.ru"
DEFAULT_START_URL = "https://www.piterra.ru/catalog/oboi/"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/121.0.0.0 Safari/537.36"
)


@dataclass
class ProductCard:
    url: str
    final_url: str = ""
    name: str = ""
    sku: str = ""
    brand: str = ""
    collection: str = ""
    collection_url: str = ""
    country: str = ""
    length_m: float | None = None
    width_m: float | None = None
    width_cm: float | None = None
    weight: str = ""
    base_material: str = ""
    coating_material: str = ""
    price: int | None = None
    old_price: int | None = None
    price_per_m2: int | None = None
    price_currency: str = "RUB"
    availability: str = ""
    stock_count: int | None = None
    stock_text: str = ""
    rating: float | None = None
    rating_count: int | None = None
    model: str = ""
    description: str = ""
    breadcrumbs: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)
    images: list[str] = field(default_factory=list)
    local_image_paths: list[str] = field(default_factory=list)
    asset_links: list[dict[str, str]] = field(default_factory=list)
    source_page_url: str = ""
    source_page_number: int | None = None
    parse_status: str = "ok"
    error: str = ""


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("\xa0", " ")).strip()


def lower(value: Any) -> str:
    return norm(value).lower().replace("ё", "е")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def stable_hash(value: str, n: int = 12) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()[:n]


def clean_filename(value: str, max_len: int = 120) -> str:
    value = lower(value)
    value = re.sub(r"https?://", "", value)
    value = re.sub(r"[^\wа-яА-ЯёЁ\-\. ]+", "_", value, flags=re.UNICODE)
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"_+", "_", value).strip("._- ")
    return (value or "item")[:max_len].strip("._- ")


def absolutize(url: str) -> str:
    return urljoin(BASE_URL, url or "")


def normalize_url(url: str) -> str:
    parsed = urlparse(absolutize(url))
    path = parsed.path
    if path and not path.endswith("/") and "/catalog/" in path:
        path += "/"
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def page_url(start_url: str, page_number: int) -> str:
    start = normalize_url(start_url)
    if page_number <= 1:
        return start
    return urljoin(start.rstrip("/") + "/", f"page/{page_number}/")


def fetch(session: requests.Session, url: str, cache_path: Path | None = None, refresh: bool = False) -> str:
    if cache_path and cache_path.exists() and not refresh:
        return cache_path.read_text(encoding="utf-8", errors="ignore")
    response = session.get(url, timeout=40)
    response.raise_for_status()
    response.encoding = response.encoding or "utf-8"
    text = response.text
    if cache_path:
        ensure_dir(cache_path.parent)
        cache_path.write_text(text, encoding="utf-8")
    return text


def parse_int(text: str) -> int | None:
    digits = re.sub(r"\D+", "", norm(text))
    return int(digits) if digits else None


def parse_money_candidates(text: str) -> list[int]:
    values: list[int] = []
    for match in re.finditer(r"(?<!\d)(\d[\d\s\xa0]{1,10})\s*(?:P|₽|руб)", norm(text), flags=re.I):
        value = parse_int(match.group(1))
        if value is not None and 0 < value < 1_000_000:
            values.append(value)
    return values


def parse_float(text: str) -> float | None:
    match = re.search(r"\d+(?:[.,]\d+)?", norm(text).replace(" ", ""))
    return float(match.group(0).replace(",", ".")) if match else None


def parse_size(text: str) -> tuple[float | None, float | None, float | None]:
    match = re.search(r"(\d+(?:[.,]\d+)?)\s*[*xх×]\s*(\d+(?:[.,]\d+)?)\s*м?", lower(text))
    if not match:
        return None, None, None
    width_m = float(match.group(1).replace(",", "."))
    length_m = float(match.group(2).replace(",", "."))
    return length_m, width_m, round(width_m * 100, 3)


def slug_from_url(url: str) -> str:
    return urlparse(url).path.rstrip("/").split("/")[-1]


def extract_image_url(node: Any) -> str:
    if not node:
        return ""
    raw = node.get("data-download") or node.get("src") or node.get("content") or ""
    if not raw and node.get("style"):
        match = re.search(r"url\((?:['\"])?(.+?)(?:['\"])?\)", node.get("style", ""))
        raw = match.group(1) if match else ""
    return absolutize(raw) if raw else ""


def extract_price_from_node(node: Any) -> int | None:
    if not node:
        return None
    meta = node.select_one('[itemprop="price"][content], meta[itemprop="price"][content], meta[property="product:price:amount"][content]')
    if meta:
        parsed = parse_int(meta.get("content", ""))
        if parsed is not None and parsed < 1_000_000:
            return parsed
    offer = node.select_one('[itemprop="price"]')
    if offer:
        value = offer.get("content") or offer.get_text(" ", strip=True)
        parsed = parse_int(value)
        if parsed is not None and parsed < 1_000_000:
            return parsed
    candidates = parse_money_candidates(node.get_text(" ", strip=True))
    return candidates[-1] if candidates else None


def parse_listing(html_text: str, page_url_value: str, page_number: int) -> list[ProductCard]:
    soup = BeautifulSoup(html_text, "html.parser")
    cards: list[ProductCard] = []
    for node in soup.select(".card-box[itemtype*='Product'], .card-box.news_itemN"):
        link = node.select_one("a.card-content__title[href], a[itemprop='url'][href], .card-img a[href]")
        if not link:
            continue
        url = normalize_url(link.get("href", ""))
        name_node = node.select_one("[itemprop='name']")
        name = norm(name_node.get_text(" ", strip=True) if name_node else link.get_text(" ", strip=True))
        if not name:
            continue
        brand_node = node.select_one("[itemprop='brand'] [itemprop='name'], .card-content__brand")
        brand = norm(brand_node.get_text(" ", strip=True)) if brand_node else ""
        model_node = node.select_one("meta[itemprop='model']")
        model = norm(model_node.get("content", "")) if model_node else ""
        desc_node = node.select_one("[itemprop='description']")
        description = norm(desc_node.get("content", "") or desc_node.get_text(" ", strip=True)) if desc_node else ""
        image_node = node.select_one("meta[itemprop='image'], .card-img img")
        image_url = extract_image_url(image_node)
        rating_value = parse_float((node.select_one("[itemprop='ratingValue']") or "").get_text(" ", strip=True) if node.select_one("[itemprop='ratingValue']") else "")
        rating_count = parse_int((node.select_one("[itemprop='reviewCount']") or "").get_text(" ", strip=True) if node.select_one("[itemprop='reviewCount']") else "")
        size_text = norm((node.select_one(".card-content__size") or "").get_text(" ", strip=True))
        length_m, width_m, width_cm = parse_size(size_text)
        price = extract_price_from_node(node)
        old_node = node.select_one(".price-item-old")
        stock_text = norm((node.select_one(".card-price__quantity") or node.select_one(".card-content__status") or "").get_text(" ", strip=True))
        card = ProductCard(
            url=url,
            final_url=url,
            name=name,
            sku=slug_from_url(url),
            brand=brand,
            model=model,
            description=description,
            price=price,
            old_price=parse_int(old_node.get_text(" ", strip=True)) if old_node else None,
            availability="in_stock" if "налич" in lower(stock_text) else "unknown",
            stock_count=parse_int(stock_text),
            stock_text=stock_text,
            rating=rating_value,
            rating_count=rating_count,
            length_m=length_m,
            width_m=width_m,
            width_cm=width_cm,
            images=[image_url] if image_url else [],
            source_page_url=page_url_value,
            source_page_number=page_number,
            categories=["wallpaper", "piterra"],
        )
        card.properties = build_properties(card)
        cards.append(card)
    return cards


def parse_table_properties(soup: BeautifulSoup) -> dict[str, str]:
    props: dict[str, str] = {}
    for row in soup.select("tr"):
        cells = [norm(x.get_text(" ", strip=True)) for x in row.select("th,td")]
        cells = [x for x in cells if x]
        if len(cells) >= 2 and len(cells[0]) <= 90:
            props[cells[0]] = cells[1]
    for hidden in soup.select('input[name="CHARACTER"][value]'):
        fragment = BeautifulSoup(html.unescape(hidden.get("value", "")), "html.parser")
        for row in fragment.select("tr"):
            cells = [norm(x.get_text(" ", strip=True)) for x in row.select("th,td")]
            cells = [x for x in cells if x]
            if len(cells) >= 2 and len(cells[0]) <= 90:
                props[cells[0]] = cells[1]
    return props


def parse_product_page(card: ProductCard, html_text: str) -> ProductCard:
    soup = BeautifulSoup(html_text, "html.parser")
    h1 = soup.select_one("h1[itemprop='name'], h1.product-info__title, h1")
    if h1 and norm(h1.get_text(" ", strip=True)):
        card.name = norm(h1.get_text(" ", strip=True))
    brand_node = soup.select_one("[itemprop='brand'][content], .product-info__head-brand a")
    if brand_node:
        card.brand = norm(brand_node.get("content", "") or brand_node.get_text(" ", strip=True)) or card.brand
    model_node = soup.select_one("meta[itemprop='model']")
    if model_node:
        card.model = norm(model_node.get("content", "")) or card.model
    desc_node = soup.select_one("[itemprop='description'], meta[name='description']")
    if desc_node:
        card.description = norm(desc_node.get("content", "") or desc_node.get_text(" ", strip=True)) or card.description
    collection_node = soup.select_one("a.product-info__show-collection[href]")
    if collection_node:
        card.collection_url = normalize_url(collection_node.get("href", ""))
        card.collection = infer_collection_from_url(card.collection_url)
    price_node = soup.select_one(".product-info__price-box")
    card.price = extract_price_from_node(price_node) or card.price
    old_node = soup.select_one(".product-info__price-box .price-item-old")
    card.old_price = parse_int(old_node.get_text(" ", strip=True)) if old_node else card.old_price
    for item in soup.select(".product-info__price-item"):
        text = norm(item.get_text(" ", strip=True))
        if "кв" in lower(text) and "метр" in lower(text):
            card.price_per_m2 = parse_int(text)
    for image_node in soup.select(".product-card-slider__item-img, img[itemprop='image'], [itemprop='contentUrl']"):
        image_url = extract_image_url(image_node)
        if image_url and image_url not in card.images:
            card.images.append(image_url)
    for a in soup.select("a[href]"):
        text = norm(a.get_text(" ", strip=True))
        href = a.get("href", "")
        if text in {"Скачать в 3D", "Инструкция по монтажу", "Презентация коллекции"}:
            card.asset_links.append({"title": text, "url": absolutize(href)})
    props = parse_table_properties(soup)
    card.properties = {**build_properties(card), **props}
    apply_properties_to_card(card)
    return card


def infer_collection_from_url(url: str) -> str:
    match = re.search(r"kollektsiyakod-is-([^/]+)/", url)
    return match.group(1) if match else ""


def build_properties(card: ProductCard) -> dict[str, Any]:
    props: dict[str, Any] = {
        "Бренд": card.brand,
        "Коллекция": card.collection,
        "Ссылка на коллекцию": card.collection_url,
        "Модель": card.model,
        "Страна": card.country,
        "Размер": f"{card.width_m:g}*{card.length_m:g}" if card.width_m and card.length_m else "",
        "Длина рулона": card.length_m,
        "Ширина рулона": card.width_cm,
        "Ширина рулона, м": card.width_m,
        "Материал основы": card.base_material,
        "Материал покрытия": card.coating_material,
        "Старая цена": card.old_price,
        "Цена за кв. метр": card.price_per_m2,
        "Наличие": card.stock_text,
        "Ссылки": card.asset_links,
    }
    return {k: v for k, v in props.items() if v not in ("", None, [])}


def apply_properties_to_card(card: ProductCard) -> None:
    props_l = {lower(k): norm(v) for k, v in card.properties.items()}
    size = props_l.get("размер") or props_l.get("размер рулона")
    if size:
        card.length_m, card.width_m, card.width_cm = parse_size(size)
    card.country = props_l.get("страна", card.country)
    material_text = " ".join([props_l.get("материал", ""), props_l.get("основа", ""), props_l.get("тип основы", "")])
    if "флизелин" in lower(material_text):
        card.base_material = "Флизелин"
    elif "бумаг" in lower(material_text):
        card.base_material = "Бумага"
    if "винил" in lower(material_text):
        card.coating_material = "Винил"


def download_image(session: requests.Session, url: str, out_dir: Path, index: int) -> str:
    ensure_dir(out_dir)
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        suffix = ".jpg"
    path = out_dir / f"{index:02d}_{stable_hash(url)}{suffix}"
    if path.exists() and path.stat().st_size > 0:
        return str(path)
    response = session.get(url, timeout=40)
    response.raise_for_status()
    path.write_bytes(response.content)
    return str(path)


def enrich_card(
    idx_card: tuple[int, ProductCard],
    out_root: Path,
    fetch_product_pages: bool,
    download_images: bool,
    max_images_per_product: int,
    refresh: bool,
) -> ProductCard:
    idx, card = idx_card
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    if fetch_product_pages:
        html_path = out_root / "product_html" / f"{idx:05d}_{clean_filename(card.sku)}.html"
        try:
            parse_product_page(card, fetch(session, card.url, html_path, refresh=refresh))
        except Exception as exc:
            card.parse_status = "partial"
            card.error = norm(exc)
    if download_images and card.images:
        product_dir = out_root / "material_images" / f"{idx:05d}_{clean_filename(card.sku or card.name)}"
        for image_idx, image_url in enumerate(card.images[:max_images_per_product], start=1):
            try:
                card.local_image_paths.append(download_image(session, image_url, product_dir, image_idx))
            except Exception as exc:
                card.error = norm(exc)
    card.properties = {**build_properties(card), **card.properties}
    return card


def make_wall_material(card: ProductCard, out_root: Path, analyze_image_colors: bool = False) -> WallMaterial:
    local_paths: list[str] = []
    for raw in card.local_image_paths:
        path = Path(raw)
        try:
            local_paths.append(str(path.relative_to(out_root)))
        except ValueError:
            local_paths.append(str(path))
    text = " ".join([card.name, card.description, card.brand, card.collection, " ".join(f"{k} {v}" for k, v in card.properties.items())])
    color, tone = normalize_color_and_tone(text)
    pattern = normalize_pattern(text)
    material_type = normalize_wall_material_type("обои " + text)
    base_material = normalize_base_material(" ".join([card.base_material, card.coating_material, text]))
    colors = analyze_wallpaper_colors(out_root, local_paths) if analyze_image_colors and local_paths else {}
    material = WallMaterial(
        source="piterra",
        sku=card.sku,
        name=card.name,
        brand=card.brand,
        product_url=card.url,
        price=float(card.price) if card.price is not None else None,
        price_currency=card.price_currency,
        availability=card.availability,
        material_type=material_type,
        color=color,
        tone=tone,
        pattern=pattern,
        base_material=base_material,
        width_cm=card.width_cm,
        length_m=card.length_m,
        country=card.country or None,
        description=card.description,
        raw_properties=card.properties,
        image_urls=card.images,
        local_image_paths=local_paths,
        average_rgb=colors.get("average_rgb"),
        average_hex=colors.get("average_hex"),
        dominant_colors_rgb=colors.get("dominant_colors_rgb") or [],
        dominant_colors_hex=colors.get("dominant_colors_hex") or [],
        room_suitability=["bedroom", "living_room", "office", "children", "hallway"],
        parse_status=card.parse_status,
    )
    tags = {"wallpaper", "wall_covering", material_type}
    for value in [color, tone, pattern]:
        if value:
            tags.add(value)
    material.style_tags = sorted(tags)
    material.search_text = lower(text)
    return material


def make_surface_material(card: ProductCard, wall: WallMaterial) -> dict[str, Any]:
    image_path = wall.local_image_paths[0] if wall.local_image_paths else ""
    return {
        "version": "surface_material.v1",
        "source": "piterra",
        "url": card.url,
        "name": card.name,
        "sku": card.sku,
        "brand": card.brand,
        "collection": card.collection,
        "collection_url": card.collection_url,
        "price": wall.price,
        "price_currency": card.price_currency,
        "availability": card.availability,
        "normalized": {
            "material_type": "wallpaper",
            "material_role": "field_tile",
            "is_selectable_floor": False,
            "is_selectable_wall": True,
            "is_accent_only": False,
            "exclude_reason": "",
            "usage": ["wall"],
            "rooms": wall.room_suitability,
            "style_tags": wall.style_tags,
            "visual_pattern": wall.pattern,
            "base_color": wall.color,
            "tone": wall.tone,
            "width_cm": wall.width_cm,
            "length_m": wall.length_m,
            "roll_area_m2": round((card.length_m or 0) * (card.width_m or 0), 3) if card.length_m and card.width_m else None,
        },
        "text_facts": {
            "model": card.model,
            "country": card.country,
            "base_material": card.base_material,
            "coating_material": card.coating_material,
            "stock": card.stock_text,
            "old_price": card.old_price,
            "price_per_m2": card.price_per_m2,
            "asset_links": card.asset_links,
        },
        "text_description_ru": card.description,
        "material_image": {
            "path": image_path,
            "source_path": image_path,
            "image_url": card.images[0] if card.images else "",
            "status": "ok" if image_path else "missing",
        },
        "vlm": {"status": "not_run", "model": "", "description_ru": "", "color": "", "pattern": "", "style": ""},
        "raw_properties": card.properties,
    }


def write_jsonl(path: Path, records: list[Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            if hasattr(record, "to_dict"):
                record = record.to_dict()
            elif hasattr(record, "__dataclass_fields__"):
                record = asdict(record)
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=False) + "\n")


def read_product_cards(path: Path) -> list[ProductCard]:
    cards: list[ProductCard] = []
    valid_fields = set(ProductCard.__dataclass_fields__)
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            cards.append(ProductCard(**{key: value for key, value in data.items() if key in valid_fields}))
    return cards


def write_normalized_outputs(out_root: Path, cards: list[ProductCard], analyze_image_colors: bool = False) -> None:
    walls = [make_wall_material(card, out_root, analyze_image_colors=analyze_image_colors) for card in cards]
    surfaces = [make_surface_material(card, wall) for card, wall in zip(cards, walls)]
    write_jsonl(out_root / "normalized_wall_materials.jsonl", walls)
    write_jsonl(out_root / "piterra_surface_materials.jsonl", surfaces)
    write_csv(out_root / "piterra_surface_materials.csv", [flatten_surface(record) for record in surfaces])
    write_analytics(out_root, cards, walls, surfaces)


def card_to_row(card: ProductCard, out_root: Path) -> dict[str, str]:
    local_paths: list[str] = []
    for raw in card.local_image_paths:
        try:
            local_paths.append(str(Path(raw).relative_to(out_root)))
        except ValueError:
            local_paths.append(str(raw))
    return {
        "url": card.url,
        "final_url": card.final_url,
        "name": card.name,
        "sku": card.sku,
        "brand": card.brand,
        "collection": card.collection,
        "collection_url": card.collection_url,
        "price": "" if card.price is None else str(card.price),
        "old_price": "" if card.old_price is None else str(card.old_price),
        "price_per_m2": "" if card.price_per_m2 is None else str(card.price_per_m2),
        "price_currency": card.price_currency,
        "availability": card.availability,
        "description": card.description,
        "breadcrumbs": " > ".join(card.breadcrumbs),
        "categories": " > ".join(card.categories),
        "properties_json": json.dumps(card.properties, ensure_ascii=False, sort_keys=True),
        "images_json": json.dumps(card.images, ensure_ascii=False),
        "local_image_paths_json": json.dumps(local_paths, ensure_ascii=False),
        "asset_links_json": json.dumps(card.asset_links, ensure_ascii=False),
        "parse_status": card.parse_status,
        "error": card.error,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_auxiliary(out_root: Path, cards: list[ProductCard]) -> None:
    write_csv(out_root / "products.csv", [card_to_row(card, out_root) for card in cards])
    with (out_root / "product_urls.txt").open("w", encoding="utf-8") as f:
        for card in cards:
            f.write(card.url + "\n")
    collection_rows = []
    for url, count in Counter(card.collection_url for card in cards if card.collection_url).most_common():
        collection_rows.append({"collection_url": url, "item_count": count})
    write_csv(out_root / "collection_urls.csv", collection_rows)
    write_jsonl(out_root / "collection_urls.jsonl", collection_rows)


def write_analytics(out_root: Path, cards: list[ProductCard], walls: list[WallMaterial], surfaces: list[dict[str, Any]]) -> None:
    prices = [card.price for card in cards if card.price is not None]
    analytics = {
        "source": "piterra",
        "products_total": len(cards),
        "normalized_wall_materials_total": len(walls),
        "surface_materials_total": len(surfaces),
        "with_price": sum(card.price is not None for card in cards),
        "with_dimensions": sum(bool(card.length_m and card.width_cm) for card in cards),
        "with_images": sum(bool(card.images) for card in cards),
        "with_local_images": sum(bool(card.local_image_paths) for card in cards),
        "with_collection_url": sum(bool(card.collection_url) for card in cards),
        "unique_collection_urls": len({card.collection_url for card in cards if card.collection_url}),
        "with_3d_link": sum(any(link.get("title") == "Скачать в 3D" for link in card.asset_links) for card in cards),
        "price_min": min(prices) if prices else None,
        "price_max": max(prices) if prices else None,
        "price_avg": round(sum(prices) / len(prices), 2) if prices else None,
        "brands": Counter(card.brand or "unknown" for card in cards).most_common(),
        "countries": Counter(card.country or "unknown" for card in cards).most_common(),
        "patterns": Counter(wall.pattern or "unknown" for wall in walls).most_common(),
        "colors": Counter(wall.color or "unknown" for wall in walls).most_common(),
    }
    (out_root / "analytics_current.json").write_text(json.dumps(analytics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_root / "piterra_surface_materials_analytics.json").write_text(json.dumps(analytics, ensure_ascii=False, indent=2), encoding="utf-8")


def scrape(args: argparse.Namespace) -> None:
    out_root = Path(args.out)
    for subdir in ["listing_html", "product_html", "material_images"]:
        ensure_dir(out_root / subdir)
    if args.normalize_existing:
        products_path = out_root / "products.jsonl"
        cards = read_product_cards(products_path)
        write_normalized_outputs(out_root, cards, analyze_image_colors=args.analyze_image_colors)
        eprint((out_root / "analytics_current.json").read_text(encoding="utf-8"))
        return

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    seen: dict[str, ProductCard] = {}
    empty = 0
    for page_number in range(1, args.max_pages + 1):
        url = page_url(args.start_url, page_number)
        html = fetch(session, url, out_root / "listing_html" / f"page_{page_number:04d}.html", refresh=args.refresh)
        cards = parse_listing(html, url, page_number)
        new_count = 0
        for card in cards:
            if card.url not in seen:
                seen[card.url] = card
                new_count += 1
        eprint(f"page {page_number}: cards={len(cards)} new={new_count} total={len(seen)}")
        empty = empty + 1 if new_count == 0 else 0
        if empty >= args.stop_after_empty:
            break
        if args.limit and len(seen) >= args.limit:
            break
        if args.delay:
            time.sleep(args.delay)

    cards = list(seen.values())[: args.limit or None]
    if args.fetch_product_pages or args.download_images:
        enriched: list[ProductCard | None] = [None] * len(cards)
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {
                executor.submit(enrich_card, (idx, card), out_root, args.fetch_product_pages, args.download_images, args.max_images_per_product, args.refresh): idx
                for idx, card in enumerate(cards, start=1)
            }
            for done_count, future in enumerate(as_completed(futures), start=1):
                idx = futures[future]
                try:
                    enriched[idx - 1] = future.result()
                except Exception as exc:
                    cards[idx - 1].parse_status = "partial"
                    cards[idx - 1].error = norm(exc)
                    enriched[idx - 1] = cards[idx - 1]
                if done_count % 100 == 0 or done_count == len(cards):
                    eprint(f"enriched {done_count}/{len(cards)}")
        cards = [card for card in enriched if card is not None]

    write_jsonl(out_root / "products.jsonl", cards)
    write_auxiliary(out_root, cards)
    write_normalized_outputs(out_root, cards, analyze_image_colors=args.analyze_image_colors)
    eprint((out_root / "analytics_current.json").read_text(encoding="utf-8"))


def flatten_surface(record: dict[str, Any]) -> dict[str, Any]:
    n = record.get("normalized", {})
    facts = record.get("text_facts", {})
    return {
        "name": record.get("name"),
        "sku": record.get("sku"),
        "url": record.get("url"),
        "brand": record.get("brand"),
        "collection": record.get("collection"),
        "collection_url": record.get("collection_url"),
        "price": record.get("price"),
        "availability": record.get("availability"),
        "is_selectable_wall": n.get("is_selectable_wall"),
        "width_cm": n.get("width_cm"),
        "length_m": n.get("length_m"),
        "roll_area_m2": n.get("roll_area_m2"),
        "visual_pattern": n.get("visual_pattern"),
        "base_color": n.get("base_color"),
        "country": facts.get("country"),
        "price_per_m2": facts.get("price_per_m2"),
        "image_path": record.get("material_image", {}).get("path"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-url", default=DEFAULT_START_URL)
    parser.add_argument("--out", default="data/floor_materials/piterra")
    parser.add_argument("--max-pages", type=int, default=300)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--delay", type=float, default=0.05)
    parser.add_argument("--stop-after-empty", type=int, default=3)
    parser.add_argument("--fetch-product-pages", action="store_true")
    parser.add_argument("--download-images", action="store_true")
    parser.add_argument("--max-images-per-product", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--analyze-image-colors", action="store_true")
    parser.add_argument("--normalize-existing", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    scrape(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
