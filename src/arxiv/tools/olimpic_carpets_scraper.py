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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

try:
    from src.ChooseObject.floor_material_normalizer import analyze_floor_material_colors
except Exception:
    analyze_floor_material_colors = None


BASE_URL = "https://www.olimpic.ru"
DEFAULT_CATEGORY_URL = "https://www.olimpic.ru/kovry/kovry-s-sovremennym-risunkom/"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/121.0.0.0 Safari/537.36"
)


@dataclass
class ProductUrlRow:
    page_number: int
    page_url: str
    product_url: str
    title: str = ""
    short_text: str = ""
    listing_price: str = ""
    listing_image: str = ""
    brand_badge: str = ""


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
    properties: dict[str, Any] = field(default_factory=dict)
    variants: list[dict[str, Any]] = field(default_factory=list)
    sizes: list[str] = field(default_factory=list)
    images: list[str] = field(default_factory=list)
    local_image_paths: list[str] = field(default_factory=list)
    listing_price: str = ""
    listing_image: str = ""
    listing_short_text: str = ""
    parse_status: str = "ok"
    error: str = ""


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def norm_text(value: Any) -> str:
    text = str(value or "").replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_attr_title(node: Any) -> str:
    if node is None:
        return ""
    clone = BeautifulSoup(str(node), "html.parser")
    for extra in clone.select(".question-with-tooltip, script, style"):
        extra.decompose()
    return norm_text(clone.get_text(" ", strip=True)).rstrip(":! ")


def clean_filename(value: str, max_len: int = 120) -> str:
    value = norm_text(value).lower()
    value = re.sub(r"https?://", "", value)
    value = re.sub(r"[^\wа-яА-ЯёЁ\-\. ]+", "_", value, flags=re.UNICODE)
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"_+", "_", value)
    value = value.strip("._- ")
    return (value or "item")[:max_len].strip("._- ")


def stable_hash(value: str, n: int = 12) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()[:n]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def strip_query(url: str) -> str:
    parsed = urlparse(urljoin(BASE_URL, url))
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))


def parse_price(value: Any) -> float | None:
    text = norm_text(value)
    if not text:
        return None
    match = re.search(r"\d+(?:[ \u00a0]\d{3})*(?:[,.]\d+)?|\d+", text)
    if not match:
        return None
    try:
        return float(match.group(0).replace(" ", "").replace("\xa0", "").replace(",", "."))
    except ValueError:
        return None


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "ru,en;q=0.8",
        }
    )
    return session


def fetch(session: requests.Session, url: str, timeout: int = 45, retries: int = 6) -> requests.Response:
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = session.get(url, timeout=timeout)
            if response.status_code == 429 and attempt < retries:
                retry_after = parse_price(response.headers.get("Retry-After")) or 0
                time.sleep(max(retry_after, 10.0 + attempt * 10.0))
                continue
            response.raise_for_status()
            return response
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(0.5 + attempt)
    raise last_exc or RuntimeError(f"fetch failed: {url}")


def category_page_url(category_url: str, page_number: int) -> str:
    base = strip_query(category_url).rstrip("/")
    if page_number <= 1:
        return f"{base}/"
    return f"{base}/{page_number}/"


def extract_max_page(html_text: str, base_category_url: str) -> int:
    soup = BeautifulSoup(html_text, "html.parser")
    max_page = 1
    base_path = urlparse(strip_query(base_category_url)).path.rstrip("/")
    for a in soup.select("ol.pagination a[href]"):
        href = strip_query(a.get("href", ""))
        parsed = urlparse(href)
        if not parsed.path.rstrip("/").startswith(base_path):
            continue
        text = norm_text(a.get_text(" ", strip=True))
        if text.isdigit():
            max_page = max(max_page, int(text))
        match = re.search(r"/(\d+)/?$", parsed.path)
        if match:
            max_page = max(max_page, int(match.group(1)))
    return max_page


def extract_listing_rows(html_text: str, page_url: str, page_number: int) -> list[ProductUrlRow]:
    soup = BeautifulSoup(html_text, "html.parser")
    rows: list[ProductUrlRow] = []
    seen: set[str] = set()
    for card in soup.select("#productList .productData, .productData.productBox"):
        title_link = card.select_one(".item__title__link[href]")
        link = title_link or card.select_one(".item-photo__link[href], a.item-order[href]")
        if not link:
            continue
        product_url = strip_query(urljoin(page_url, link.get("href", "")))
        if "/kovry/" not in product_url or not product_url.endswith(".html"):
            continue
        if product_url in seen:
            continue
        seen.add(product_url)
        img = card.select_one(".item-photo__img")
        image_url = ""
        if img:
            image_url = str(img.get("data-src") or img.get("src") or "").strip()
            if "tail-spin.svg" in image_url:
                image_url = str(img.get("data-src") or "").strip()
            image_url = urljoin(page_url, image_url) if image_url else ""
        rows.append(
            ProductUrlRow(
                page_number=page_number,
                page_url=page_url,
                product_url=product_url,
                title=norm_text((title_link.get_text(" ", strip=True) if title_link else "") or link.get("title") or ""),
                short_text=norm_text(card.select_one(".item__text__link").get_text(" ", strip=True) if card.select_one(".item__text__link") else ""),
                listing_price=norm_text(card.select_one(".item-price__value").get_text(" ", strip=True) if card.select_one(".item-price__value") else ""),
                listing_image=image_url,
                brand_badge=norm_text(card.select_one(".item-photo__type").get_text(" ", strip=True) if card.select_one(".item-photo__type") else ""),
            )
        )
    return rows


def collect_product_urls(
    category_url: str,
    out_dir: Path,
    max_pages: int | None,
    save_html: bool,
    sleep_sec: float,
) -> list[ProductUrlRow]:
    session = make_session()
    first_url = category_page_url(category_url, 1)
    first = fetch(session, first_url)
    first_html = first.text
    discovered_max = extract_max_page(first_html, category_url)
    page_count = max_pages or discovered_max
    page_count = max(1, min(page_count, discovered_max if max_pages is None else page_count))
    listing_dir = out_dir / "listing_html"
    if save_html:
        ensure_dir(listing_dir)
        (listing_dir / "page_001.html").write_text(first_html, encoding="utf-8")

    all_rows: list[ProductUrlRow] = []
    seen_products: set[str] = set()
    for page_number in tqdm(range(1, page_count + 1), desc="Olimpic listing pages"):
        page_url = category_page_url(category_url, page_number)
        try:
            html_text = first_html if page_number == 1 else fetch(session, page_url).text
        except Exception as exc:
            eprint(f"[WARN] listing fetch failed: {page_url}: {exc}")
            continue
        if save_html and page_number != 1:
            ensure_dir(listing_dir)
            (listing_dir / f"page_{page_number:03d}.html").write_text(html_text, encoding="utf-8")
        rows = extract_listing_rows(html_text, page_url, page_number)
        if not rows and page_number > discovered_max:
            break
        for row in rows:
            if row.product_url not in seen_products:
                seen_products.add(row.product_url)
                all_rows.append(row)
        partial_path = out_dir / "product_urls.partial.jsonl"
        partial_path.write_text(
            "\n".join(json.dumps(asdict(row), ensure_ascii=False) for row in all_rows) + ("\n" if all_rows else ""),
            encoding="utf-8",
        )
        if sleep_sec > 0:
            time.sleep(sleep_sec)
    return all_rows


def extract_properties(soup: BeautifulSoup) -> dict[str, str]:
    props: dict[str, str] = {}
    root = soup.select_one("#attributes") or soup
    for dl in root.select("dl.attr-wrap"):
        title = clean_attr_title(dl.select_one("dt.attr__title") or dl.select_one("dt"))
        value = norm_text(dl.select_one("dd.attr__value").get_text(" ", strip=True) if dl.select_one("dd.attr__value") else "")
        if title and value and title not in props:
            props[title] = value
    return props


def extract_variants(soup: BeautifulSoup) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    for label in soup.select(".item-variant-value"):
        size = norm_text(label.select_one(".item-title").get_text(" ", strip=True) if label.select_one(".item-title") else "")
        info = norm_text(label.get("data-variant-tocartinfo") or "")
        if not size and info:
            size = re.sub(r"^размер\s+", "", info, flags=re.I)
        variants.append(
            {
                "size": size,
                "price": parse_price(label.get("data-variant-price")),
                "price_raw": norm_text(label.get("data-variant-price") or ""),
                "sku": norm_text(label.get("data-variant-code") or ""),
                "stock": norm_text(label.get("data-variant-stock") or ""),
                "stock_info": norm_text(str(label.get("data-variant-stockinfo") or "").replace("$#span#$", "(").replace("$#/span#$", ")").replace("$#br#$", " ")),
                "variant_id": norm_text(label.get("data-variant-id") or ""),
                "max_to_cart": parse_price(label.get("data-variant-max-tocart")),
            }
        )
    return variants


def extract_images(soup: BeautifulSoup, page_url: str) -> list[str]:
    images: list[str] = []
    seen: set[str] = set()
    for a in soup.select('.gallery-top a[data-fancybox="gallery"][href], .gallery-thumbs a[data-fancybox="gallery1"][href]'):
        url = urljoin(page_url, str(a.get("href") or "").strip())
        if url and url not in seen:
            seen.add(url)
            images.append(url)
    if not images:
        for meta in soup.select('meta[property="og:image"][content]'):
            url = urljoin(page_url, str(meta.get("content") or "").strip())
            if url and url not in seen:
                seen.add(url)
                images.append(url)
    return images


def parse_product(session: requests.Session, row: ProductUrlRow) -> ProductRow:
    try:
        response = fetch(session, row.product_url)
        soup = BeautifulSoup(response.text, "html.parser")
        title = norm_text(soup.select_one("#productTitle").get_text(" ", strip=True) if soup.select_one("#productTitle") else "")
        short_desc = norm_text(soup.select_one("#productShortdesc").get_text(" ", strip=True) if soup.select_one("#productShortdesc") else "")
        description = norm_text(soup.select_one("#description").get_text(" ", strip=True) if soup.select_one("#description") else "")
        props = extract_properties(soup)
        variants = extract_variants(soup)
        sizes = [v["size"] for v in variants if v.get("size")]
        prices = [v.get("price") for v in variants if v.get("price") is not None]
        sku = variants[0].get("sku", "") if variants else ""
        brand = row.brand_badge or props.get("Производитель", "") or props.get("Бренд", "")
        availability = norm_text(soup.select_one(".product-card-fullstockinfo").get_text(" ", strip=True) if soup.select_one(".product-card-fullstockinfo") else "")
        breadcrumbs = [norm_text(x.get_text(" ", strip=True)) for x in soup.select("#breadcrumb span[itemprop='title'], #breadcrumb a")]
        breadcrumbs = [x for x in breadcrumbs if x]
        return ProductRow(
            url=row.product_url,
            final_url=strip_query(response.url),
            name=title or row.title,
            sku=str(sku or props.get("Код производителя", "") or stable_hash(row.product_url)),
            brand=brand,
            price=str(min(prices)) if prices else str(parse_price(row.listing_price) or ""),
            price_currency="RUB",
            availability=availability,
            description=description or short_desc,
            breadcrumbs=breadcrumbs,
            categories=breadcrumbs,
            properties=props,
            variants=variants,
            sizes=sizes,
            images=extract_images(soup, response.url) or ([row.listing_image] if row.listing_image else []),
            listing_price=row.listing_price,
            listing_image=row.listing_image,
            listing_short_text=row.short_text,
        )
    except Exception as exc:
        return ProductRow(
            url=row.product_url,
            name=row.title,
            brand=row.brand_badge,
            price=str(parse_price(row.listing_price) or ""),
            availability="unknown",
            images=[row.listing_image] if row.listing_image else [],
            listing_price=row.listing_price,
            listing_image=row.listing_image,
            listing_short_text=row.short_text,
            parse_status="error",
            error=str(exc),
        )


def parse_products(rows: list[ProductUrlRow], workers: int, sleep_sec: float) -> list[ProductRow]:
    indexed_products: dict[str, ProductRow] = {}

    def task(row: ProductUrlRow) -> ProductRow:
        product = parse_product(make_session(), row)
        if sleep_sec > 0:
            time.sleep(sleep_sec)
        return product

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(task, row) for row in rows]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Olimpic product pages"):
            product = future.result()
            indexed_products[product.url] = product
    return [indexed_products[row.product_url] for row in rows if row.product_url in indexed_products]


def image_extension(response: requests.Response, url: str) -> str:
    ctype = response.headers.get("content-type", "").split(";")[0].strip().lower()
    ext = mimetypes.guess_extension(ctype) if ctype else ""
    if ext == ".jpe":
        ext = ".jpg"
    if ext:
        return ext
    suffix = Path(urlparse(url).path).suffix.lower()
    return suffix if suffix in {".jpg", ".jpeg", ".png", ".webp"} else ".jpg"


def download_first_images(products: list[ProductRow], out_dir: Path, workers: int) -> None:
    image_root = out_dir / "material_images"
    ensure_dir(image_root)

    def one(idx_product: tuple[int, ProductRow]) -> tuple[int, str]:
        idx, product = idx_product
        if not product.images:
            return idx, ""
        url = product.images[0]
        session = make_session()
        response = fetch(session, url, timeout=60, retries=2)
        folder = image_root / f"{idx + 1:05d}_{clean_filename(product.sku or product.name or stable_hash(product.url), 80)}"
        ensure_dir(folder)
        ext = image_extension(response, url)
        path = folder / f"01_{stable_hash(url)}{ext}"
        path.write_bytes(response.content)
        return idx, str(path.relative_to(out_dir))

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(one, item) for item in enumerate(products)]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Olimpic first images"):
            idx, rel_path = future.result()
            if rel_path:
                products[idx].local_image_paths = [rel_path]


def normalize_product(product: ProductRow, out_dir: Path) -> dict[str, Any]:
    props = dict(product.properties)
    first_variant = product.variants[0] if product.variants else {}
    price = parse_price(product.price) or parse_price(first_variant.get("price"))
    material = props.get("Состав ворса") or props.get("Качество ворса") or ""
    design = props.get("Дизайн") or product.listing_short_text
    shape = props.get("Форма") or props.get("Форма!") or ""
    search_parts = [
        product.name,
        product.brand,
        design,
        shape,
        material,
        props.get("Коллекция", ""),
        product.description,
        " ".join(f"{k} {v}" for k, v in props.items()),
    ]
    data = {
        "version": "floor_material.v1",
        "source": "olimpic",
        "sku": product.sku,
        "name": product.name,
        "brand": product.brand,
        "product_url": product.final_url or product.url,
        "price": price,
        "price_currency": product.price_currency or "RUB",
        "availability": "in_stock" if "налич" in product.availability.lower() or "под заказ" in product.availability.lower() else "unknown",
        "material_type": "carpet",
        "surface_group": "floor_covering",
        "decor": None,
        "decor_name": design or None,
        "design": design or None,
        "tone": props.get("Цвет") or None,
        "tone_family": props.get("Цвет") or None,
        "gloss": None,
        "thickness_mm": None,
        "plank_width_mm": None,
        "plank_length_mm": None,
        "package_area_m2": None,
        "chamfer": None,
        "water_resistant": False,
        "warm_floor_compatible": None,
        "country": props.get("Страна") or None,
        "description": product.description,
        "raw_properties": {
            **props,
            "variants": product.variants,
            "sizes": product.sizes,
            "listing_price": product.listing_price,
            "listing_short_text": product.listing_short_text,
        },
        "image_urls": product.images,
        "local_image_paths": product.local_image_paths,
        "style_tags": ["carpet", "rug", "floor_covering"],
        "room_suitability": ["bedroom", "living_room", "children", "office", "hallway"],
        "bad_for": ["bathroom", "wet_zone"],
        "search_text": norm_text(" ".join(search_parts)).lower(),
        "parse_status": product.parse_status,
    }
    if analyze_floor_material_colors is not None:
        data.update(analyze_floor_material_colors(out_dir, product.local_image_paths))
    return data


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = {}
            for key, value in row.items():
                if isinstance(value, (list, dict)):
                    flat[key] = json.dumps(value, ensure_ascii=False)
                else:
                    flat[key] = value
            writer.writerow(flat)


def write_outputs(out_dir: Path, url_rows: list[ProductUrlRow], products: list[ProductRow], normalized: list[dict[str, Any]]) -> None:
    ensure_dir(out_dir)
    url_dicts = [asdict(row) for row in url_rows]
    product_dicts = [asdict(product) for product in products]
    write_csv(out_dir / "product_urls.csv", url_dicts)
    write_csv(out_dir / "products.csv", product_dicts)
    write_csv(out_dir / "olimpic_surface_materials.csv", normalized)
    (out_dir / "product_urls.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in url_dicts) + "\n", encoding="utf-8")
    (out_dir / "products.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in product_dicts) + "\n", encoding="utf-8")
    (out_dir / "normalized_floor_materials.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in normalized) + "\n", encoding="utf-8")
    analytics = {
        "root": str(out_dir),
        "schema": "olimpic_carpets_bundle/v1",
        "products": len(products),
        "ok": sum(1 for p in products if p.parse_status == "ok"),
        "errors": sum(1 for p in products if p.parse_status != "ok"),
        "with_price": sum(1 for row in normalized if row.get("price") is not None),
        "with_first_image": sum(1 for p in products if p.local_image_paths),
    }
    (out_dir / "analytics_current.json").write_text(json.dumps(analytics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "olimpic_surface_materials_analytics.json").write_text(json.dumps(analytics, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Scrape Olimpic carpets into data/floor_materials/olimpic.")
    parser.add_argument("--category-url", default=DEFAULT_CATEGORY_URL)
    parser.add_argument("-o", "--out-dir", type=Path, default=Path("data/floor_materials/olimpic"))
    parser.add_argument("--max-pages", type=int, default=None, help="Default: auto-discover all category pages.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--image-workers", type=int, default=8)
    parser.add_argument("--sleep", type=float, default=0.0, help="Delay after each product page request.")
    parser.add_argument("--listing-sleep", type=float, default=0.0, help="Delay after each listing page request.")
    parser.add_argument("--no-download-images", action="store_true")
    parser.add_argument("--save-html", action="store_true")
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    url_rows = collect_product_urls(args.category_url, args.out_dir, args.max_pages, args.save_html, args.listing_sleep)
    eprint(f"[INFO] collected product urls: {len(url_rows)}")
    products = parse_products(url_rows, args.workers, args.sleep)
    if not args.no_download_images:
        download_first_images(products, args.out_dir, args.image_workers)
    normalized = [normalize_product(product, args.out_dir) for product in products]
    write_outputs(args.out_dir, url_rows, products, normalized)
    eprint(f"[INFO] wrote {len(normalized)} normalized rows to {args.out_dir / 'normalized_floor_materials.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
