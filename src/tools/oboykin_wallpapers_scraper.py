#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Scrape Oboykin wallpaper catalog into the wall/surface material formats."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

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


BASE_URL = "https://www.oboykin.ru"
DEFAULT_START_URL = "https://www.oboykin.ru/catalog/oboi/?city=Санкт-Петербург&begin=0&end=32"
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
    country: str = ""
    length_m: float | None = None
    width_m: float | None = None
    width_cm: float | None = None
    weight: str = ""
    base_material: str = ""
    coating_material: str = ""
    price: int | None = None
    old_price: int | None = None
    price_currency: str = "RUB"
    price_note: str = ""
    availability: str = ""
    stock_count: int | None = None
    stock_text: str = ""
    badge: str = ""
    rating: float | None = None
    rating_count: int | None = None
    spec: str = ""
    description: str = ""
    breadcrumbs: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)
    images: list[str] = field(default_factory=list)
    local_image_paths: list[str] = field(default_factory=list)
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


def norm_lines(value: Any) -> str:
    lines = [norm(x) for x in str(value or "").replace("\xa0", " ").splitlines()]
    return "\n".join(x for x in lines if x)


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


def absolutize(url: str, base_url: str = BASE_URL) -> str:
    return urljoin(base_url, url or "")


def normalize_url(url: str, keep_query: bool = False) -> str:
    parsed = urlparse(absolutize(url))
    path = parsed.path
    if path and not path.endswith("/") and "/catalog/" in path:
        path += "/"
    query = parsed.query if keep_query else ""
    return urlunparse((parsed.scheme, parsed.netloc, path, "", query, ""))


def catalog_page_url(start_url: str, page_number: int) -> str:
    start = normalize_url(start_url, keep_query=True)
    parsed = urlparse(start)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    per_page = 32
    if "begin" in query and "end" in query:
        try:
            per_page = max(1, int(query["end"]) - int(query["begin"]))
        except ValueError:
            per_page = 32
        begin = (page_number - 1) * per_page
        query["begin"] = str(begin)
        query["end"] = str(begin + per_page)
    else:
        try:
            per_page = int(query.get("end", "32"))
        except ValueError:
            per_page = 32
        query["end"] = str(page_number * per_page)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", urlencode(query, doseq=True), ""))


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


def parse_price(text: str) -> int | None:
    if "запрос" in lower(text):
        return None
    digits = re.sub(r"\D+", "", text)
    return int(digits) if digits else None


def parse_float(text: str) -> float | None:
    match = re.search(r"\d+(?:[.,]\d+)?", norm(text).replace(" ", ""))
    if not match:
        return None
    return float(match.group(0).replace(",", "."))


def parse_stock(text: str) -> int | None:
    match = re.search(r"\d+", norm(text).replace(" ", ""))
    return int(match.group(0)) if match else None


def parse_spec(spec: str) -> dict[str, Any]:
    lines = [norm(x) for x in re.split(r"[\n\r]+", spec) if norm(x)]
    first = lines[0] if lines else ""
    second = lines[1] if len(lines) > 1 else ""
    parts = [norm(x) for x in first.split(",") if norm(x)]
    country = parts[0] if parts else ""
    dim_text = next((x for x in parts if re.search(r"\d+(?:[.,]\d+)?\s*[xх×]\s*\d+(?:[.,]\d+)?\s*м", x, re.I)), "")
    weight = next((x for x in parts if re.search(r"(?:кг|г/м)", x, re.I)), "")
    length_m: float | None = None
    width_m: float | None = None
    if dim_text:
        match = re.search(r"(\d+(?:[.,]\d+)?)\s*[xх×]\s*(\d+(?:[.,]\d+)?)\s*м", dim_text, re.I)
        if match:
            length_m = float(match.group(1).replace(",", "."))
            width_m = float(match.group(2).replace(",", "."))
    materials = [norm(x) for x in second.split(",") if norm(x)]
    return {
        "country": country,
        "length_m": length_m,
        "width_m": width_m,
        "width_cm": round(width_m * 100, 3) if width_m is not None else None,
        "weight": weight,
        "base_material": materials[0] if materials else "",
        "coating_material": materials[1] if len(materials) > 1 else "",
    }


def parse_css_image(style: str) -> str:
    match = re.search(r"url\((?:&quot;|['\"])?(.+?)(?:&quot;|['\"])?\)", style or "")
    return absolutize(match.group(1)) if match else ""


def slug_from_url(url: str) -> str:
    return urlparse(url).path.rstrip("/").split("/")[-1]


def infer_brand(name: str, product_url: str) -> str:
    title = re.sub(r"^(обои|малярный флизелин|флизелин)\s+", "", norm(name), flags=re.I)
    tokens = title.split()
    brand_tokens: list[str] = []
    for token in tokens:
        if re.search(r"\d", token) and brand_tokens:
            break
        brand_tokens.append(token)
        if len(brand_tokens) >= 3:
            break
    brand = " ".join(brand_tokens).strip()
    path_parts = [p for p in urlparse(product_url).path.split("/") if p]
    if not brand and len(path_parts) >= 4:
        brand = path_parts[-3].replace("-", " ").title()
    return brand


def collection_from_url(product_url: str) -> str:
    parts = [p for p in urlparse(product_url).path.split("/") if p]
    if len(parts) >= 2:
        return parts[-2].replace("-", " ")
    return ""


def parse_listing(html_text: str, page_url: str, page_number: int) -> list[ProductCard]:
    soup = BeautifulSoup(html_text, "html.parser")
    cards: list[ProductCard] = []
    for node in soup.select("div.tovar"):
        link = node.select_one("a[href]")
        if not link:
            continue
        product_url = normalize_url(link.get("href", ""))
        if "/catalog/" not in product_url:
            continue
        name = norm((node.select_one(".h3") or link).get_text(" ", strip=True))
        if not name:
            continue
        spec_node = node.select_one(".spec")
        spec = norm_lines(spec_node.get_text("\n", strip=True)) if spec_node else ""
        parsed_spec = parse_spec(spec)
        price_node = node.select_one(".price")
        old_price_node = node.select_one(".oldprice")
        price_text = norm(price_node.get_text(" ", strip=True)) if price_node else ""
        image_node = node.select_one(".image")
        image_url = parse_css_image(image_node.get("style", "")) if image_node else ""
        stock_text = norm((node.select_one(".count .avl") or node.select_one(".count") or "").get_text(" ", strip=True))
        rating_node = node.select_one(".raiting__stars")
        rating_count_node = node.select_one(".raiting")
        rating_count_text = norm(rating_count_node.get_text(" ", strip=True)) if rating_count_node else ""
        rating_count = parse_stock(re.sub(r"^\s*\d+(?:[.,]\d+)?", "", rating_count_text))
        rating = parse_float(rating_node.get_text(" ", strip=True)) if rating_node else None
        badge_node = node.select_one(".badge")
        badge = norm(badge_node.get_text(" ", strip=True)) if badge_node else ""
        card = ProductCard(
            url=product_url,
            final_url=product_url,
            name=name,
            sku=slug_from_url(product_url),
            brand=infer_brand(name, product_url),
            collection=collection_from_url(product_url),
            price=parse_price(price_text),
            old_price=parse_price(old_price_node.get_text(" ", strip=True)) if old_price_node else None,
            price_note="Цена по запросу" if "запрос" in lower(price_text) else "",
            availability="in_stock" if parse_stock(stock_text) else "unknown",
            stock_count=parse_stock(stock_text),
            stock_text=stock_text,
            badge=badge,
            rating=rating,
            rating_count=rating_count,
            spec=spec,
            images=[image_url] if image_url else [],
            source_page_url=page_url,
            source_page_number=page_number,
            **parsed_spec,
        )
        card.categories = ["wallpaper", "oboykin"]
        card.properties = build_properties(card)
        cards.append(card)
    return cards


def build_properties(card: ProductCard) -> dict[str, Any]:
    props: dict[str, Any] = {
        "Бренд": card.brand,
        "Коллекция": card.collection,
        "Страна": card.country,
        "Длина рулона": card.length_m,
        "Ширина рулона": card.width_cm,
        "Ширина рулона, м": card.width_m,
        "Вес": card.weight,
        "Материал основы": card.base_material,
        "Материал покрытия": card.coating_material,
        "Наличие": card.stock_text,
        "Старая цена": card.old_price,
        "Бейдж": card.badge,
        "Спецификация": card.spec,
    }
    return {k: v for k, v in props.items() if v not in ("", None, [])}


def parse_product_page(card: ProductCard, html_text: str) -> ProductCard:
    soup = BeautifulSoup(html_text, "html.parser")
    breadcrumbs = [norm(x.get_text(" ", strip=True)) for x in soup.select(".breadcrumbs a, .bread a, nav a")]
    if breadcrumbs:
        card.breadcrumbs = [x for x in breadcrumbs if x]
    h1 = soup.select_one("h1")
    if h1 and norm(h1.get_text(" ", strip=True)):
        card.name = norm(h1.get_text(" ", strip=True))
    desc_candidates = soup.select(".description, .descr, .content_text, .text, .detail_text")
    for node in desc_candidates:
        text = norm(node.get_text(" ", strip=True))
        if len(text) > len(card.description):
            card.description = text
    images = list(card.images)
    for img in soup.select("img[src], img[data-src], img[data-lazy]"):
        raw = img.get("data-src") or img.get("data-lazy") or img.get("src") or ""
        full = absolutize(raw)
        if "/assets/images/products/" in full and full not in images:
            images.append(full)
    for node in soup.select("[style]"):
        full = parse_css_image(node.get("style", ""))
        if "/assets/images/products/" in full and full not in images:
            images.append(full)
    for row in soup.select("tr, .chars .item, .properties .item, .param, .property"):
        cells = [norm(x.get_text(" ", strip=True)) for x in row.select("th,td,.name,.key,.value")]
        cells = [x for x in cells if x]
        if len(cells) >= 2 and len(cells[0]) <= 80:
            card.properties.setdefault(cells[0], cells[1])
    card.images = images
    card.brand = card.brand or infer_brand(card.name, card.url)
    card.properties = {**build_properties(card), **card.properties}
    return card


def download_image(session: requests.Session, url: str, out_dir: Path, index: int) -> str:
    ensure_dir(out_dir)
    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        suffix = ".jpg"
    filename = f"{index:02d}_{stable_hash(url)}{suffix}"
    path = out_dir / filename
    if path.exists() and path.stat().st_size > 0:
        return str(path)
    response = session.get(url, timeout=40)
    response.raise_for_status()
    path.write_bytes(response.content)
    return str(path)


def enrich_card(
    index_and_card: tuple[int, ProductCard],
    out_root: Path,
    fetch_product_pages: bool,
    download_images: bool,
    max_images_per_product: int,
    refresh: bool,
) -> ProductCard:
    idx, card = index_and_card
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8"})
    if fetch_product_pages:
        html_path = out_root / "product_html" / f"{idx:05d}_{clean_filename(card.sku or card.name)}.html"
        try:
            html_text = fetch(session, card.url, html_path, refresh=refresh)
            parse_product_page(card, html_text)
        except Exception as exc:
            card.parse_status = "partial"
            card.error = norm(exc)
    if download_images and card.images:
        product_dir = out_root / "material_images" / f"{idx:05d}_{clean_filename(card.sku or card.name)}"
        max_images = min(max_images_per_product, len(card.images))
        for image_idx, image_url in enumerate(card.images[:max_images], start=1):
            try:
                card.local_image_paths.append(download_image(session, image_url, product_dir, image_idx))
            except Exception as exc:
                card.error = norm(exc)
    card.properties = {**build_properties(card), **card.properties}
    return card


def product_to_csv_row(card: ProductCard, out_root: Path) -> dict[str, str]:
    local_paths = []
    for raw in card.local_image_paths:
        path = Path(raw)
        try:
            local_paths.append(str(path.relative_to(out_root)))
        except ValueError:
            local_paths.append(str(path))
    return {
        "url": card.url,
        "final_url": card.final_url,
        "name": card.name,
        "sku": card.sku,
        "brand": card.brand,
        "price": "" if card.price is None else str(card.price),
        "price_currency": card.price_currency,
        "availability": card.availability,
        "description": card.description,
        "breadcrumbs": " > ".join(card.breadcrumbs),
        "categories": " > ".join(card.categories),
        "properties_json": json.dumps(card.properties, ensure_ascii=False, sort_keys=True),
        "images_json": json.dumps(card.images, ensure_ascii=False),
        "local_image_paths_json": json.dumps(local_paths, ensure_ascii=False),
        "parse_status": card.parse_status,
        "error": card.error,
        "collection": card.collection,
        "old_price": "" if card.old_price is None else str(card.old_price),
        "price_note": card.price_note,
        "stock_count": "" if card.stock_count is None else str(card.stock_count),
        "rating": "" if card.rating is None else str(card.rating),
        "rating_count": "" if card.rating_count is None else str(card.rating_count),
    }


def style_tags(material_type: str, color: str | None, tone: str | None, pattern: str | None) -> list[str]:
    tags = {material_type, "wall_covering"}
    for value in [color, tone, pattern]:
        if value:
            tags.add(value)
    if pattern in {"floral", "botanical"}:
        tags.update({"classic", "contemporary"})
    if pattern in {"concrete", "brick", "plaster"}:
        tags.update({"loft", "industrial"})
    if pattern in {"plain", None}:
        tags.update({"minimalism", "contemporary"})
    return sorted(tags)


def make_wall_material(card: ProductCard, out_root: Path) -> WallMaterial:
    local_paths = []
    for raw in card.local_image_paths:
        path = Path(raw)
        try:
            local_paths.append(str(path.relative_to(out_root)))
        except ValueError:
            local_paths.append(str(path))
    all_text = " ".join([card.name, card.description, card.spec, " ".join(f"{k} {v}" for k, v in card.properties.items())])
    color, tone = normalize_color_and_tone(all_text)
    pattern = normalize_pattern(all_text)
    material_type = normalize_wall_material_type("обои " + all_text)
    base_material = normalize_base_material(" ".join([card.base_material, card.coating_material, all_text]))
    colors = analyze_wallpaper_colors(out_root, local_paths) if local_paths else {}
    material = WallMaterial(
        source="oboykin",
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
        finish=None,
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
        style_tags=style_tags(material_type, color, tone, pattern),
        room_suitability=["bedroom", "living_room", "children", "office", "hallway"],
        parse_status=card.parse_status,
    )
    material.search_text = lower(" ".join([card.name, card.brand, card.collection, all_text]))
    return material


def make_surface_material(card: ProductCard, wall: WallMaterial) -> dict[str, Any]:
    image_path = wall.local_image_paths[0] if wall.local_image_paths else ""
    return {
        "version": "surface_material.v1",
        "source": "oboykin",
        "url": card.url,
        "name": card.name,
        "sku": card.sku,
        "brand": card.brand,
        "collection": card.collection,
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
            "precise_color_ru": None,
            "tone": wall.tone,
            "surface_finish": wall.finish,
            "edge": None,
            "anti_slip": False,
            "frost_resistant": False,
            "rectified": False,
            "thickness_mm": None,
            "width_cm": wall.width_cm,
            "height_cm": None,
            "length_m": wall.length_m,
            "roll_area_m2": round((card.length_m or 0) * (card.width_m or 0), 3) if card.length_m and card.width_m else None,
        },
        "text_facts": {
            "type": "Обои",
            "country": card.country,
            "base_material": card.base_material,
            "coating_material": card.coating_material,
            "weight": card.weight,
            "stock": card.stock_text,
            "badge": card.badge,
            "old_price": card.old_price,
        },
        "text_description_ru": ", ".join(x for x in [card.country, card.spec, card.description] if x),
        "material_image": {
            "path": image_path,
            "source_path": image_path,
            "image_url": card.images[0] if card.images else "",
            "product_dir": str(Path(image_path).parent) if image_path else "",
            "image_index": 1 if image_path else None,
            "width": None,
            "height": None,
            "aspect": None,
            "status": "ok" if image_path else "missing",
            "reason": "",
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


def write_products_csv(path: Path, cards: list[ProductCard], out_root: Path) -> None:
    rows = [product_to_csv_row(card, out_root) for card in cards]
    fields = list(rows[0].keys()) if rows else []
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_surface_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "name",
        "sku",
        "url",
        "brand",
        "collection",
        "price",
        "availability",
        "material_type",
        "is_selectable_wall",
        "width_cm",
        "length_m",
        "roll_area_m2",
        "base_color",
        "tone",
        "visual_pattern",
        "country",
        "base_material",
        "coating_material",
        "image_path",
    ]
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            n = record.get("normalized", {})
            facts = record.get("text_facts", {})
            writer.writerow(
                {
                    "name": record.get("name"),
                    "sku": record.get("sku"),
                    "url": record.get("url"),
                    "brand": record.get("brand"),
                    "collection": record.get("collection"),
                    "price": record.get("price"),
                    "availability": record.get("availability"),
                    "material_type": n.get("material_type"),
                    "is_selectable_wall": n.get("is_selectable_wall"),
                    "width_cm": n.get("width_cm"),
                    "length_m": n.get("length_m"),
                    "roll_area_m2": n.get("roll_area_m2"),
                    "base_color": n.get("base_color"),
                    "tone": n.get("tone"),
                    "visual_pattern": n.get("visual_pattern"),
                    "country": facts.get("country"),
                    "base_material": facts.get("base_material"),
                    "coating_material": facts.get("coating_material"),
                    "image_path": record.get("material_image", {}).get("path"),
                }
            )


def write_auxiliary_tables(out_root: Path, cards: list[ProductCard]) -> None:
    with (out_root / "product_urls.txt").open("w", encoding="utf-8") as f:
        for card in cards:
            f.write(card.url + "\n")
    with (out_root / "product_urls.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "name", "sku", "source_page_number", "source_page_url"])
        writer.writeheader()
        for card in cards:
            writer.writerow(
                {
                    "url": card.url,
                    "name": card.name,
                    "sku": card.sku,
                    "source_page_number": card.source_page_number,
                    "source_page_url": card.source_page_url,
                }
            )
    with (out_root / "product_images.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "sku", "image_url", "local_image_path", "image_index"])
        writer.writeheader()
        for card in cards:
            for idx, image_url in enumerate(card.images, start=1):
                local = card.local_image_paths[idx - 1] if idx - 1 < len(card.local_image_paths) else ""
                try:
                    local = str(Path(local).relative_to(out_root)) if local else ""
                except ValueError:
                    pass
                writer.writerow({"url": card.url, "sku": card.sku, "image_url": image_url, "local_image_path": local, "image_index": idx})
    with (out_root / "product_properties.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "sku", "property", "value"])
        writer.writeheader()
        for card in cards:
            for key, value in card.properties.items():
                writer.writerow({"url": card.url, "sku": card.sku, "property": key, "value": value})


def write_analytics(out_root: Path, cards: list[ProductCard], walls: list[WallMaterial], surfaces: list[dict[str, Any]]) -> None:
    prices = [card.price for card in cards if card.price is not None]
    analytics = {
        "source": "oboykin",
        "products_total": len(cards),
        "normalized_wall_materials_total": len(walls),
        "surface_materials_total": len(surfaces),
        "with_price": sum(1 for card in cards if card.price is not None),
        "with_old_price": sum(1 for card in cards if card.old_price is not None),
        "with_images": sum(1 for card in cards if card.images),
        "with_local_images": sum(1 for card in cards if card.local_image_paths),
        "with_dimensions": sum(1 for card in cards if card.length_m and card.width_cm),
        "price_min": min(prices) if prices else None,
        "price_max": max(prices) if prices else None,
        "price_avg": round(sum(prices) / len(prices), 2) if prices else None,
        "countries": Counter(card.country or "unknown" for card in cards).most_common(),
        "base_materials": Counter(card.base_material or "unknown" for card in cards).most_common(),
        "brands": Counter(card.brand or "unknown" for card in cards).most_common(),
        "patterns": Counter(wall.pattern or "unknown" for wall in walls).most_common(),
        "colors": Counter(wall.color or "unknown" for wall in walls).most_common(),
    }
    (out_root / "analytics_current.json").write_text(json.dumps(analytics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_root / "oboykin_surface_materials_analytics.json").write_text(json.dumps(analytics, ensure_ascii=False, indent=2), encoding="utf-8")


def write_readme(out_root: Path, analytics: dict[str, Any] | None = None) -> None:
    if analytics is None:
        analytics_path = out_root / "analytics_current.json"
        analytics = json.loads(analytics_path.read_text(encoding="utf-8")) if analytics_path.exists() else {}
    text = "\n".join(
        [
            "# Oboykin Wallpapers",
            "",
            "Catalog bundle scraped from https://www.oboykin.ru/catalog/oboi/.",
            "",
            "Files:",
            "- `products.csv` / `products.jsonl`: raw product cards with prices, roll dimensions, materials, stock, URLs and images.",
            "- `normalized_wall_materials.jsonl`: wall selector `wall_material.v1` records.",
            "- `oboykin_surface_materials.jsonl` / `.csv`: Mosplitka-like `surface_material.v1` records.",
            "- `product_urls.*`, `product_images.csv`, `product_properties.csv`: auxiliary source tables.",
            "- `material_images/`: downloaded wallpaper images referenced by normalized records.",
            "",
            f"Products: {analytics.get('products_total', 0)}",
            f"With price: {analytics.get('with_price', 0)}",
            f"With dimensions: {analytics.get('with_dimensions', 0)}",
            f"With local images: {analytics.get('with_local_images', 0)}",
            "",
        ]
    )
    (out_root / "README.md").write_text(text, encoding="utf-8")


def scrape(args: argparse.Namespace) -> None:
    out_root = Path(args.out)
    ensure_dir(out_root)
    ensure_dir(out_root / "listing_html")
    ensure_dir(out_root / "product_html")
    ensure_dir(out_root / "material_images")
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8"})

    seen: dict[str, ProductCard] = {}
    stagnant_pages = 0
    for page_number in range(1, args.max_pages + 1):
        page_url = catalog_page_url(args.start_url, page_number)
        html_path = out_root / "listing_html" / f"page_{page_number:04d}.html"
        try:
            html_text = fetch(session, page_url, html_path, refresh=args.refresh)
        except Exception as exc:
            eprint(f"listing fetch failed page={page_number} url={page_url}: {exc}")
            break
        cards = parse_listing(html_text, page_url, page_number)
        new_count = 0
        for card in cards:
            if card.url not in seen:
                seen[card.url] = card
                new_count += 1
        eprint(f"page {page_number}: cards={len(cards)} new={new_count} total={len(seen)}")
        stagnant_pages = stagnant_pages + 1 if new_count == 0 else 0
        if stagnant_pages >= args.stop_after_empty:
            break
        if args.limit and len(seen) >= args.limit:
            break
        if args.delay:
            time.sleep(args.delay)

    cards = list(seen.values())[: args.limit or None]
    needs_enrichment = args.fetch_product_pages or args.download_images
    if needs_enrichment and args.workers > 1:
        enriched: list[ProductCard | None] = [None] * len(cards)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    enrich_card,
                    (idx, card),
                    out_root,
                    args.fetch_product_pages,
                    args.download_images,
                    args.max_images_per_product,
                    args.refresh,
                ): idx
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
    else:
        for idx, card in enumerate(cards, start=1):
            card = enrich_card(
                (idx, card),
                out_root,
                args.fetch_product_pages,
                args.download_images,
                args.max_images_per_product,
                args.refresh,
            )
            cards[idx - 1] = card
            if args.delay and needs_enrichment:
                time.sleep(args.delay)

    products_jsonl = out_root / "products.jsonl"
    products_csv = out_root / "products.csv"
    write_jsonl(products_jsonl, cards)
    write_products_csv(products_csv, cards, out_root)
    write_auxiliary_tables(out_root, cards)

    walls = [make_wall_material(card, out_root) for card in cards]
    write_jsonl(out_root / "normalized_wall_materials.jsonl", walls)
    surfaces = [make_surface_material(card, wall) for card, wall in zip(cards, walls)]
    write_jsonl(out_root / "oboykin_surface_materials.jsonl", surfaces)
    write_surface_csv(out_root / "oboykin_surface_materials.csv", surfaces)
    write_analytics(out_root, cards, walls, surfaces)
    analytics = json.loads((out_root / "analytics_current.json").read_text(encoding="utf-8"))
    write_readme(out_root, analytics)
    eprint(json.dumps(analytics, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-url", default=DEFAULT_START_URL)
    parser.add_argument("--out", default="data/floor_materials/oboykin")
    parser.add_argument("--max-pages", type=int, default=300)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--delay", type=float, default=0.05)
    parser.add_argument("--stop-after-empty", type=int, default=3)
    parser.add_argument("--fetch-product-pages", action="store_true")
    parser.add_argument("--download-images", action="store_true")
    parser.add_argument("--max-images-per-product", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--refresh", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scrape(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
