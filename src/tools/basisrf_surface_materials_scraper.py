#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import mimetypes
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


BASE_URL = "https://basisrf.ru"
DEFAULT_CATEGORY_URLS = [
    "https://basisrf.ru/catalog/mdf/",
    "https://basisrf.ru/catalog/stoleshnitsy/",
    "https://basisrf.ru/catalog/fasadnye-polotna/",
    "https://basisrf.ru/catalog/kromochnaya-produktsiya/",
]
DEFAULT_PRODUCT_URLS = [
    "https://basisrf.ru/catalog/dekory-imitatsiya-9/f-108-st9-mramor-san-luka/",
]
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/121.0.0.0 Safari/537.36"
)


@dataclass
class ProductUrlRow:
    source_url: str
    page_url: str
    product_url: str
    anchor_text: str = ""


@dataclass
class ProductRow:
    url: str
    final_url: str = ""
    name: str = ""
    sku: str = ""
    brand: str = ""
    price: str = ""
    price_currency: str = "RUB"
    availability: str = ""
    description: str = ""
    breadcrumbs: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    properties: dict[str, str] = field(default_factory=dict)
    images: list[str] = field(default_factory=list)
    local_image_paths: list[str] = field(default_factory=list)
    parse_status: str = "ok"
    error: str = ""


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def norm_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def lower_text(value: Any) -> str:
    return norm_text(value).lower().replace("ё", "е")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def stable_hash(value: str, n: int = 12) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()[:n]


def clean_filename(value: str, max_len: int = 120) -> str:
    value = lower_text(value)
    value = re.sub(r"https?://", "", value)
    value = re.sub(r"[^\wа-яА-ЯёЁ\-\. ]+", "_", value, flags=re.UNICODE)
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"_+", "_", value)
    value = value.strip("._- ")
    return (value or "item")[:max_len].strip("._- ")


def normalize_url(url: str, *, keep_query: bool = False) -> str:
    parsed = urlparse(urljoin(BASE_URL, url))
    path = parsed.path or "/"
    if path.startswith("/catalog/") and not path.endswith("/") and "." not in Path(path).name:
        path += "/"
    query = parsed.query if keep_query else ""
    return urlunparse((parsed.scheme, parsed.netloc, path, "", query, ""))


def product_slug_from_url(url: str) -> str:
    return urlparse(url).path.rstrip("/").split("/")[-1]


def fetch(session: requests.Session, url: str, timeout: int = 45) -> requests.Response:
    resp = session.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp


def is_basis_catalog_url(url: str) -> bool:
    parsed = urlparse(urljoin(BASE_URL, url))
    return parsed.netloc.endswith("basisrf.ru") and parsed.path.startswith("/catalog/")


def is_probably_file_url(url: str) -> bool:
    path = urlparse(urljoin(BASE_URL, url)).path.lower()
    return bool(re.search(r"\.(?:jpg|jpeg|png|webp|gif|pdf|doc|docx|xls|xlsx|zip|rar)$", path))


def looks_like_product_html(html_text: str) -> bool:
    soup = BeautifulSoup(html_text, "html.parser")
    product_scope = soup.select_one('[itemscope][itemtype*="schema.org/Product"]')
    if product_scope and soup.select_one("h1"):
        return True
    return bool(soup.select_one("#main-photo") and soup.select_one(".bxr-props-table"))


def extract_links(html_text: str, page_url: str) -> list[tuple[str, str]]:
    soup = BeautifulSoup(html_text, "html.parser")
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for a in soup.select("a[href]"):
        href = str(a.get("href") or "").strip()
        if not href or href.startswith(("javascript:", "mailto:", "tel:", "#")):
            continue
        url = normalize_url(urljoin(page_url, href), keep_query=True)
        if not is_basis_catalog_url(url) or is_probably_file_url(url):
            continue
        text = norm_text(a.get_text(" ", strip=True))
        if url not in seen:
            seen.add(url)
            out.append((url, text))
    return out


def candidate_page_variants(category_url: str, page_number: int) -> list[str]:
    base = normalize_url(category_url)
    if page_number <= 1:
        return [base]
    parsed = urlparse(base)
    path = re.sub(r"/page/\d+/?$", "/", parsed.path).rstrip("/")
    variants = [
        urlunparse((parsed.scheme, parsed.netloc, f"{path}/page/{page_number}/", "", "", "")),
        urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", f"PAGEN_1={page_number}", "")),
        urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", f"PAGEN_2={page_number}", "")),
        urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", f"page={page_number}", "")),
    ]
    out: list[str] = []
    for url in variants:
        if url not in out:
            out.append(url)
    return out


def extract_listing_product_links(html_text: str, page_url: str) -> list[tuple[str, str]]:
    soup = BeautifulSoup(html_text, "html.parser")
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    selectors = [
        ".bxr-element-container .bxr-element-name a[href]",
        ".bxr-element-container .bxr-element-image a[href]",
        ".bxr-element-action [data-trade-link]",
    ]
    for selector in selectors:
        for node in soup.select(selector):
            raw = str(node.get("data-trade-link") or node.get("href") or "").strip()
            if not raw:
                continue
            url = normalize_url(urljoin(page_url, raw))
            if not is_basis_catalog_url(url) or is_probably_file_url(url):
                continue
            text = norm_text(node.get_text(" ", strip=True) or node.get("title") or "")
            if url not in seen:
                seen.add(url)
                out.append((url, text))
    return out


def collect_product_urls(
    session: requests.Session,
    category_urls: list[str],
    explicit_product_urls: list[str],
    out_dir: Path,
    max_pages_per_category: int,
    max_discovery_pages: int,
    save_html: bool,
    sleep_sec: float,
) -> list[ProductUrlRow]:
    listing_dir = out_dir / "listing_html"
    product_rows: list[ProductUrlRow] = []
    product_urls: dict[str, ProductUrlRow] = {}
    queued_category_pages: list[tuple[str, str]] = []
    seen_category_pages: set[str] = set()

    for category_url in category_urls:
        for page_number in range(1, max(1, max_pages_per_category) + 1):
            for page_url in candidate_page_variants(category_url, page_number):
                queued_category_pages.append((category_url, page_url))

    for product_url in explicit_product_urls:
        url = normalize_url(product_url)
        product_urls[url] = ProductUrlRow(source_url="explicit", page_url=url, product_url=url, anchor_text="")

    page_budget = max_discovery_pages if max_discovery_pages > 0 else len(queued_category_pages)
    pbar = tqdm(total=min(len(queued_category_pages), page_budget), desc="BasisRF listing pages")
    while queued_category_pages and len(seen_category_pages) < page_budget:
        source_url, page_url = queued_category_pages.pop(0)
        page_url = normalize_url(page_url, keep_query=True)
        if page_url in seen_category_pages:
            continue
        seen_category_pages.add(page_url)
        pbar.update(1)
        try:
            resp = fetch(session, page_url)
        except Exception as exc:
            eprint(f"[WARN] listing fetch failed: {page_url}: {exc}")
            continue
        html_text = resp.text
        if save_html:
            ensure_dir(listing_dir)
            name = f"{stable_hash(page_url)}_{clean_filename(product_slug_from_url(page_url) or 'listing')}.html"
            (listing_dir / name).write_text(html_text, encoding="utf-8")

        for link_url, anchor_text in extract_listing_product_links(html_text, resp.url):
            product_urls.setdefault(
                link_url,
                ProductUrlRow(
                    source_url=source_url,
                    page_url=page_url,
                    product_url=link_url,
                    anchor_text=anchor_text,
                ),
            )

        for link_url, anchor_text in extract_links(html_text, resp.url):
            if link_url in seen_category_pages:
                continue
            path = urlparse(link_url).path
            source_path = urlparse(normalize_url(source_url)).path.rstrip("/")
            is_under_source = path.startswith(source_path + "/")
            if link_url in product_urls:
                continue
            if is_under_source and len(seen_category_pages) + len(queued_category_pages) < page_budget:
                queued_category_pages.append((source_url, link_url))
        if sleep_sec:
            time.sleep(sleep_sec)
    pbar.close()

    product_rows = list(product_urls.values())
    product_rows.sort(key=lambda row: row.product_url)
    return product_rows


def extract_breadcrumbs(soup: BeautifulSoup) -> list[str]:
    candidates = soup.select(".breadcrumb a, .breadcrumb span, .bx-breadcrumb a, .bx-breadcrumb span")
    out: list[str] = []
    for node in candidates:
        text = norm_text(node.get_text(" ", strip=True))
        if text and text not in out:
            out.append(text)
    return out


def extract_properties(soup: BeautifulSoup) -> dict[str, str]:
    props: dict[str, str] = {}
    for row in soup.select("table.bxr-props-table tr, .bxr-detail-text table tr"):
        cells = row.find_all(["td", "th"])
        if len(cells) < 2:
            continue
        key = norm_text(cells[0].get_text(" ", strip=True))
        value = norm_text(cells[1].get_text(" ", strip=True))
        if key and value and key not in props:
            props[key] = value
    for hidden in soup.select("input.basket-prop-artrix"):
        key = norm_text(hidden.get("data-name") or hidden.get("data-code"))
        value = norm_text(hidden.get("value"))
        if key and value and key not in props:
            props[key] = value
    return props


def extract_images(soup: BeautifulSoup, page_url: str) -> list[str]:
    images: list[str] = []
    seen: set[str] = set()
    selectors = [
        "#main-photo",
        "#main-photo img",
        ".ax-element-slider-main a.fancybox[href]",
        ".ax-element-slider-main img.zoom-img",
        '[itemprop="image"]',
    ]
    for selector in selectors:
        for node in soup.select(selector):
            raw = ""
            if node.name == "a":
                raw = str(node.get("href") or "")
            else:
                raw = str(node.get("data-large") or node.get("src") or node.get("href") or "")
            if not raw:
                continue
            url = normalize_url(urljoin(page_url, raw), keep_query=True)
            path = urlparse(url).path.lower()
            if "no-image" in path or "/resize_cache/uf/" in path:
                continue
            if url not in seen and is_probably_file_url(url):
                seen.add(url)
                images.append(url)
    return images


def extract_description(soup: BeautifulSoup) -> str:
    desc_meta = soup.select_one('[itemprop="description"]')
    if desc_meta and desc_meta.get("content"):
        meta_text = norm_text(desc_meta.get("content"))
    else:
        meta_text = ""
    detail = soup.select_one('.bxr-detail-text[data-tab="detail"], .bxr-detail-text')
    detail_text = norm_text(detail.get_text(" ", strip=True)) if detail else ""
    return detail_text or meta_text


def parse_price(text: str) -> str:
    match = re.search(r"\d[\d\s]*(?:[,.]\d+)?", text or "")
    return match.group(0).replace(" ", "").replace(",", ".") if match else ""


def parse_product_html(html_text: str, url: str, final_url: str = "") -> ProductRow:
    soup = BeautifulSoup(html_text, "html.parser")
    name = ""
    h1 = soup.select_one('h1[itemprop="name"], h1')
    if h1:
        name = norm_text(h1.get_text(" ", strip=True))
    props = extract_properties(soup)
    brand = props.get("Производитель") or props.get("Бренд")
    brand_img = soup.select_one(".brand-detail img")
    if not brand and brand_img:
        brand = norm_text(Path(str(brand_img.get("src") or brand_img.get("alt") or "")).stem)
    price_node = soup.select_one('[itemprop="lowPrice"], [itemprop="price"], .bxr-market-current-price')
    price = ""
    if price_node:
        price = parse_price(str(price_node.get("content") or price_node.get_text(" ", strip=True)))
    availability_node = soup.select_one(".bxr-instock-wrap")
    availability = norm_text(availability_node.get_text(" ", strip=True)) if availability_node else ""
    sku = props.get("Артикул") or product_slug_from_url(url)
    breadcrumbs = extract_breadcrumbs(soup)
    categories = [x for x in breadcrumbs if x and x.lower() not in {"главная", "каталог"}]
    return ProductRow(
        url=normalize_url(url),
        final_url=normalize_url(final_url or url, keep_query=True),
        name=name,
        sku=sku,
        brand=brand,
        price=price,
        price_currency="RUB",
        availability=availability,
        description=extract_description(soup),
        breadcrumbs=breadcrumbs,
        categories=categories,
        properties=props,
        images=extract_images(soup, final_url or url),
        parse_status="ok" if name else "error",
        error="" if name else "missing_name",
    )


def download_images(
    session: requests.Session,
    row: ProductRow,
    out_dir: Path,
    sleep_sec: float,
    max_images: int,
) -> list[str]:
    image_dir = out_dir / "material_images" / f"{stable_hash(row.url)}_{clean_filename(row.name or row.sku)}"
    local_paths: list[str] = []
    for idx, image_url in enumerate(row.images[: max(1, max_images)], start=1):
        try:
            resp = fetch(session, image_url, timeout=60)
            content_type = resp.headers.get("content-type", "").split(";")[0].strip().lower()
            ext = mimetypes.guess_extension(content_type) or Path(urlparse(image_url).path).suffix or ".jpg"
            if ext == ".jpe":
                ext = ".jpg"
            ensure_dir(image_dir)
            path = image_dir / f"{idx:02d}_{stable_hash(image_url)}{ext}"
            path.write_bytes(resp.content)
            local_paths.append(str(path.relative_to(out_dir)))
        except Exception as exc:
            eprint(f"[WARN] image download failed: {image_url}: {exc}")
        if sleep_sec:
            time.sleep(sleep_sec)
    return local_paths


def parse_products(
    session: requests.Session,
    product_urls: list[str],
    out_dir: Path,
    save_html: bool,
    download: bool,
    sleep_sec: float,
    max_images_per_product: int,
) -> list[ProductRow]:
    product_dir = out_dir / "product_html"
    rows: list[ProductRow] = []
    for url in tqdm(product_urls, desc="BasisRF products"):
        try:
            resp = fetch(session, url)
            html_text = resp.text
            if not looks_like_product_html(html_text):
                rows.append(ProductRow(url=url, final_url=resp.url, parse_status="skipped", error="not_product_page"))
                continue
            if save_html:
                ensure_dir(product_dir)
                name = f"{stable_hash(url)}_{clean_filename(product_slug_from_url(url))}.html"
                (product_dir / name).write_text(html_text, encoding="utf-8")
            row = parse_product_html(html_text, url, resp.url)
            if download and row.images:
                row.local_image_paths = download_images(session, row, out_dir, sleep_sec, max_images_per_product)
            rows.append(row)
        except Exception as exc:
            rows.append(ProductRow(url=url, parse_status="error", error=f"{type(exc).__name__}: {exc}"))
        if sleep_sec:
            time.sleep(sleep_sec)
    return rows


def json_field(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def write_product_urls(rows: list[ProductUrlRow], out_dir: Path) -> None:
    ensure_dir(out_dir)
    csv_path = out_dir / "product_urls.csv"
    jsonl_path = out_dir / "product_urls.jsonl"
    txt_path = out_dir / "product_urls.txt"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(ProductUrlRow("", "", "")).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
    txt_path.write_text("\n".join(row.product_url for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def write_products(rows: list[ProductRow], out_dir: Path) -> None:
    ensure_dir(out_dir)
    fieldnames = [
        "url",
        "final_url",
        "name",
        "sku",
        "brand",
        "price",
        "price_currency",
        "availability",
        "description",
        "breadcrumbs",
        "categories",
        "properties_json",
        "images_json",
        "local_image_paths_json",
        "parse_status",
        "error",
    ]
    with (out_dir / "products.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "url": row.url,
                    "final_url": row.final_url,
                    "name": row.name,
                    "sku": row.sku,
                    "brand": row.brand,
                    "price": row.price,
                    "price_currency": row.price_currency,
                    "availability": row.availability,
                    "description": row.description,
                    "breadcrumbs": json_field(row.breadcrumbs),
                    "categories": json_field(row.categories),
                    "properties_json": json_field(row.properties),
                    "images_json": json_field(row.images),
                    "local_image_paths_json": json_field(row.local_image_paths),
                    "parse_status": row.parse_status,
                    "error": row.error,
                }
            )


def infer_material_type(row: ProductRow) -> str:
    text = lower_text(" ".join([row.name, " ".join(row.categories), " ".join(f"{k} {v}" for k, v in row.properties.items())]))
    if "столешниц" in text:
        return "countertop_laminate"
    if "кром" in text:
        return "edge_banding"
    if "мдф" in text or "mdf" in text:
        return "mdf_panel"
    if "фасад" in text:
        return "facade_panel"
    if "пластик" in text or "дсп" in text:
        return "laminated_board"
    return "decor_surface"


def infer_usage(material_type: str) -> list[str]:
    if material_type == "countertop_laminate":
        return ["countertop", "worktop", "furniture_surface"]
    if material_type == "edge_banding":
        return ["edge", "furniture_edge"]
    if material_type in {"mdf_panel", "facade_panel", "laminated_board"}:
        return ["furniture_panel", "facade", "wall_panel"]
    return ["furniture_surface"]


def infer_visual_pattern(row: ProductRow) -> str:
    text = lower_text(" ".join([row.name, row.description, " ".join(row.properties.values())]))
    if "мрамор" in text:
        return "marble"
    if "камень" in text:
        return "stone"
    if "бетон" in text:
        return "concrete"
    if "дуб" in text or "дерев" in text or "ясень" in text or "орех" in text:
        return "wood"
    if "однотон" in text:
        return "plain"
    return "decor"


def infer_base_color(row: ProductRow) -> str:
    text = lower_text(" ".join([row.name, row.description, " ".join(row.properties.values())]))
    for needle, color in [
        ("бел", "white"),
        ("черн", "black"),
        ("сер", "gray"),
        ("беж", "beige"),
        ("корич", "brown"),
        ("зелен", "green"),
        ("син", "blue"),
        ("красн", "red"),
        ("желт", "yellow"),
    ]:
        if needle in text:
            return color
    if "мрамор" in text:
        return "white"
    return "neutral"


def infer_tone(base_color: str) -> str:
    if base_color in {"white", "beige"}:
        return "light"
    if base_color in {"black", "brown"}:
        return "dark"
    if base_color == "gray":
        return "neutral"
    return "neutral"


def to_float(value: Any) -> float | None:
    text = norm_text(value).replace(",", ".")
    match = re.search(r"-?\d+(?:\.\d+)?", text.replace(" ", ""))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def product_to_surface_material(row: ProductRow) -> dict[str, Any] | None:
    if row.parse_status != "ok":
        return None
    material_type = infer_material_type(row)
    usage = infer_usage(material_type)
    pattern = infer_visual_pattern(row)
    color = infer_base_color(row)
    local_image = row.local_image_paths[0] if row.local_image_paths else ""
    image_url = row.images[0] if row.images else ""
    image_info = {
        "path": local_image,
        "source_path": str(Path("data/floor_materials/basisrf") / local_image) if local_image else "",
        "image_url": image_url,
        "product_dir": str(Path(local_image).parent) if local_image else "",
        "image_index": 1 if (local_image or image_url) else None,
        "status": "exists" if (local_image or image_url) else "missing",
        "reason": "primary_gallery_image" if (local_image or image_url) else "no_image",
    }
    description_parts = [color, pattern, row.properties.get("Cтруктура") or row.properties.get("Структура поверхности") or ""]
    return {
        "version": "surface_material.v1",
        "source": "basisrf",
        "url": row.url,
        "name": row.name,
        "sku": row.sku,
        "brand": row.brand,
        "collection": None,
        "price": to_float(row.price),
        "price_currency": row.price_currency or "RUB",
        "availability": "in_stock" if "налич" in lower_text(row.availability) else "unknown",
        "normalized": {
            "material_type": material_type,
            "material_role": "decor_surface",
            "is_selectable_floor": False,
            "is_selectable_wall": material_type in {"mdf_panel", "facade_panel", "laminated_board"},
            "is_accent_only": material_type in {"edge_banding"},
            "exclude_reason": "basisrf_decor_surface_not_floor_covering",
            "usage": usage,
            "rooms": ["kitchen", "living_room", "hallway", "office"],
            "style_tags": sorted({pattern, color, "decor_surface", "contemporary"}),
            "visual_pattern": pattern,
            "base_color": color,
            "precise_color_ru": "",
            "tone": infer_tone(color),
            "surface_finish": row.properties.get("Cтруктура") or row.properties.get("Структура поверхности") or None,
            "thickness_mm": to_float(row.properties.get("Толщина, мм")),
            "length_mm": to_float(row.properties.get("Длина, мм")),
            "width_mm": to_float(row.properties.get("Глубина, мм") or row.properties.get("Ширина, мм")),
        },
        "text_facts": {
            "type": row.properties.get("Категория") or material_type,
            "material_type": material_type,
            "role": "декоративное покрытие / мебельная поверхность",
            "usage": ", ".join(usage),
            "pattern": pattern,
            "color": color,
            "surface": row.properties.get("Cтруктура") or row.properties.get("Структура поверхности"),
        },
        "text_description_ru": ", ".join(x for x in description_parts if x),
        "material_image": image_info,
        "image_paths": row.local_image_paths,
        "vlm": {"status": "not_run", "model": "", "description_ru": "", "color": ""},
        "raw_properties": row.properties,
    }


def write_surface_materials(rows: list[ProductRow], out_dir: Path) -> list[dict[str, Any]]:
    materials = [m for row in rows if (m := product_to_surface_material(row)) is not None]
    out_jsonl = out_dir / "basisrf_surface_materials.jsonl"
    out_csv = out_dir / "basisrf_surface_materials.csv"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for item in materials:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["url", "name", "sku", "brand", "price", "availability", "material_type", "usage", "image_url", "local_image"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in materials:
            writer.writerow(
                {
                    "url": item["url"],
                    "name": item["name"],
                    "sku": item["sku"],
                    "brand": item["brand"],
                    "price": item["price"],
                    "availability": item["availability"],
                    "material_type": item["normalized"]["material_type"],
                    "usage": json.dumps(item["normalized"]["usage"], ensure_ascii=False),
                    "image_url": item["material_image"]["image_url"],
                    "local_image": item["material_image"]["path"],
                }
            )
    return materials


def write_readme(out_dir: Path, products: list[ProductRow], materials: list[dict[str, Any]]) -> None:
    ok = [row for row in products if row.parse_status == "ok"]
    with (out_dir / "README.md").open("w", encoding="utf-8") as f:
        f.write("# Basis RF Surface Materials\n\n")
        f.write("Autonomous surface/decor material bundle scraped from `basisrf.ru`.\n\n")
        f.write("Seed categories:\n")
        for url in DEFAULT_CATEGORY_URLS:
            f.write(f"- {url}\n")
        f.write("\nFiles:\n")
        f.write("- `product_urls.*`: discovered product URL list.\n")
        f.write("- `products.csv`: raw parsed product cards.\n")
        f.write("- `basisrf_surface_materials.jsonl`: `surface_material.v1` records.\n")
        f.write("- `material_images/`: downloaded primary/gallery images.\n")
        f.write("- `listing_html/`, `product_html/`: saved HTML when `--save-html` is used.\n\n")
        f.write("Counts:\n")
        f.write(f"- products_total: {len(products)}\n")
        f.write(f"- products_ok: {len(ok)}\n")
        f.write(f"- surface_materials_total: {len(materials)}\n")
        f.write(f"- with_material_image: {sum(1 for item in materials if item.get('material_image', {}).get('status') == 'exists')}\n")


def write_analytics(out_dir: Path, products: list[ProductRow], materials: list[dict[str, Any]]) -> None:
    material_types: dict[str, int] = {}
    availability: dict[str, int] = {}
    brands: dict[str, int] = {}
    for item in materials:
        normalized = item.get("normalized") if isinstance(item.get("normalized"), dict) else {}
        material_type = str(normalized.get("material_type") or "unknown")
        material_types[material_type] = material_types.get(material_type, 0) + 1
        av = str(item.get("availability") or "unknown")
        availability[av] = availability.get(av, 0) + 1
        brand = str(item.get("brand") or "unknown")
        brands[brand] = brands.get(brand, 0) + 1
    analytics = {
        "source": "basisrf",
        "products_total": len(products),
        "products_ok": sum(1 for row in products if row.parse_status == "ok"),
        "surface_materials_total": len(materials),
        "with_material_image": sum(1 for item in materials if item.get("material_image", {}).get("status") == "exists"),
        "material_types": dict(sorted(material_types.items())),
        "availability": dict(sorted(availability.items())),
        "top_brands": dict(sorted(brands.items(), key=lambda kv: (-kv[1], kv[0]))[:25]),
        "seed_categories": list(DEFAULT_CATEGORY_URLS),
        "explicit_product_urls": list(DEFAULT_PRODUCT_URLS),
    }
    (out_dir / "analytics_current.json").write_text(
        json.dumps(analytics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": DEFAULT_USER_AGENT, "Accept-Language": "ru,en;q=0.8"})
    return session


def load_urls_file(path: Path) -> list[str]:
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(normalize_url(line))
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scrape Basis RF surface/decor materials.")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-o", "--out", default="data/floor_materials/basisrf")
    common.add_argument("--category-url", action="append", default=None)
    common.add_argument("--product-url", action="append", default=None)
    common.add_argument("--urls-file", default=None)
    common.add_argument("--max-pages-per-category", type=int, default=12)
    common.add_argument("--max-discovery-pages", type=int, default=200)
    common.add_argument("--sleep", type=float, default=0.15)
    common.add_argument("--save-html", action="store_true")
    common.add_argument("--download-images", action="store_true")
    common.add_argument("--max-images-per-product", type=int, default=1)

    sub.add_parser("all", parents=[common], help="Collect URLs, parse cards, write products and surface materials.")
    sub.add_parser("collect-urls", parents=[common], help="Only collect product URLs.")
    parse = sub.add_parser("parse-products", parents=[common], help="Parse existing URL list.")
    parse.add_argument("--product-urls", default=None, help="Path to product_urls.txt/csv-like plain list.")

    return parser


def main() -> int:
    args = build_parser().parse_args()
    out_dir = Path(args.out).expanduser()
    ensure_dir(out_dir)
    session = build_session()
    category_urls = [normalize_url(x) for x in (args.category_url or DEFAULT_CATEGORY_URLS)]
    product_urls = [normalize_url(x) for x in (args.product_url or DEFAULT_PRODUCT_URLS)]

    if args.command in {"all", "collect-urls"}:
        rows = collect_product_urls(
            session=session,
            category_urls=category_urls,
            explicit_product_urls=product_urls,
            out_dir=out_dir,
            max_pages_per_category=args.max_pages_per_category,
            max_discovery_pages=args.max_discovery_pages,
            save_html=args.save_html,
            sleep_sec=args.sleep,
        )
        write_product_urls(rows, out_dir)
        print(f"Product URLs: {len(rows)}")
        if args.command == "collect-urls":
            return 0
        product_urls = [row.product_url for row in rows]

    if args.command == "parse-products":
        url_path = Path(args.product_urls or args.urls_file or (out_dir / "product_urls.txt"))
        if url_path.is_file():
            product_urls = load_urls_file(url_path)
        elif not product_urls:
            raise SystemExit(f"URL list not found: {url_path}")

    products = parse_products(
        session=session,
        product_urls=product_urls,
        out_dir=out_dir,
        save_html=args.save_html,
        download=args.download_images,
        sleep_sec=args.sleep,
        max_images_per_product=args.max_images_per_product,
    )
    write_products(products, out_dir)
    materials = write_surface_materials(products, out_dir)
    write_readme(out_dir, products, materials)
    write_analytics(out_dir, products, materials)
    print(f"Products parsed: {len(products)}")
    print(f"Products ok: {sum(1 for row in products if row.parse_status == 'ok')}")
    print(f"Surface materials: {len(materials)}")
    print(f"Saved: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
