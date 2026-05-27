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
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

try:
    from src.ChooseObject.floor_material_normalizer import analyze_floor_material_colors
except Exception:
    analyze_floor_material_colors = None


BASE_URL = "https://spb.shtorystore.ru"
DEFAULT_CATEGORY_URLS = [
    "https://spb.shtorystore.ru/shtory/",
    "https://spb.shtorystore.ru/tyuli/",
]
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/121.0.0.0 Safari/537.36"
)


@dataclass
class ProductUrlRow:
    category_url: str
    category_name: str
    page_url: str
    page_number: int
    product_url: str
    title: str = ""
    listing_price: str = ""
    listing_old_price: str = ""
    listing_image: str = ""


@dataclass
class ProductRow:
    url: str
    final_url: str = ""
    name: str = ""
    sku: str = ""
    brand: str = "Chernogorov"
    category: str = ""
    price: str = ""
    old_price: str = ""
    price_currency: str = "RUB"
    description: str = ""
    selected_material: str = ""
    materials: dict[str, dict[str, str]] = field(default_factory=dict)
    attachments: list[str] = field(default_factory=list)
    widths_cm: list[float] = field(default_factory=list)
    heights_cm: list[float] = field(default_factory=list)
    selected_width_cm: float | None = None
    selected_height_cm: float | None = None
    raw_properties: dict[str, Any] = field(default_factory=dict)
    images: list[str] = field(default_factory=list)
    selected_image_url: str = ""
    selected_image_index: int | None = None
    image_selection_note: str = ""
    local_image_paths: list[str] = field(default_factory=list)
    parse_status: str = "ok"
    error: str = ""


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def norm_text(value: Any) -> str:
    text = str(value or "").replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def stable_hash(value: str, n: int = 12) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()[:n]


def clean_filename(value: str, max_len: int = 120) -> str:
    value = norm_text(value).lower()
    value = re.sub(r"https?://", "", value)
    value = re.sub(r"[^\wа-яА-ЯёЁ\-\. ]+", "_", value, flags=re.UNICODE)
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"_+", "_", value)
    value = value.strip("._- ")
    return (value or "item")[:max_len].strip("._- ")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def absolutize(url: str, base_url: str = BASE_URL) -> str:
    return urljoin(base_url, str(url or "").strip())


def strip_fragment(url: str) -> str:
    parsed = urlparse(absolutize(url))
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", parsed.query, ""))


def strip_query(url: str) -> str:
    parsed = urlparse(absolutize(url))
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


def parse_number(value: Any) -> float | None:
    text = norm_text(value).replace(",", ".")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
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


def fetch(session: requests.Session, url: str, timeout: int = 45, retries: int = 3) -> requests.Response:
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = session.get(url, timeout=timeout)
            if response.status_code in {429, 500, 502, 503, 504} and attempt < retries:
                time.sleep(2.0 + attempt * 3.0)
                continue
            response.raise_for_status()
            return response
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(1.0 + attempt * 2.0)
    raise last_exc or RuntimeError(f"fetch failed: {url}")


def category_name_from_url(url: str) -> str:
    path = urlparse(url).path.strip("/")
    if path.startswith("tyuli"):
        return "Тюль"
    if path.startswith("shtory"):
        return "Шторы"
    return path or "Каталог"


def listing_page_url(category_url: str, cur_pos: int) -> str:
    parsed = urlparse(strip_query(category_url))
    if cur_pos <= 0:
        return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))
    query = {"cur_cc": "10", "q": parsed.path.strip("/") + "/", "curPos": str(cur_pos)}
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", urlencode(query), ""))


def extract_listing_max_cur_pos(html_text: str, category_url: str) -> int:
    soup = BeautifulSoup(html_text, "html.parser")
    values = [0]
    for a in soup.select(".navig-block a[href], .all-but-nav a[href]"):
        href = absolutize(a.get("href", ""), category_url)
        parsed = urlparse(href)
        qs = parse_qs(parsed.query)
        if "curPos" in qs:
            value = parse_number(qs["curPos"][0])
            if value is not None:
                values.append(int(value))
    return max(values)


def extract_listing_rows(html_text: str, category_url: str, page_url: str, page_number: int) -> list[ProductUrlRow]:
    soup = BeautifulSoup(html_text, "html.parser")
    rows: list[ProductUrlRow] = []
    seen: set[str] = set()
    category_name = category_name_from_url(category_url)
    for card in soup.select("a.one-catal-shtor[href]"):
        href = str(card.get("href") or "").strip()
        if not href:
            continue
        product_url = strip_query(urljoin(page_url, href))
        if product_url in seen:
            continue
        seen.add(product_url)
        title = norm_text(card.select_one(".art-catal-shtor").get_text(" ", strip=True) if card.select_one(".art-catal-shtor") else "")
        image_node = card.select_one(".pic-catal-shtor img[src]")
        rows.append(
            ProductUrlRow(
                category_url=strip_query(category_url),
                category_name=category_name,
                page_url=page_url,
                page_number=page_number,
                product_url=product_url,
                title=title,
                listing_price=norm_text(card.select_one(".price-catalog").get_text(" ", strip=True) if card.select_one(".price-catalog") else ""),
                listing_old_price=norm_text(card.select_one(".old-price-catalog").get_text(" ", strip=True) if card.select_one(".old-price-catalog") else ""),
                listing_image=absolutize(image_node.get("src"), page_url) if image_node else "",
            )
        )
    return rows


def collect_product_urls(category_urls: list[str], out_dir: Path, max_pages: int | None, sleep_sec: float, save_html: bool) -> list[ProductUrlRow]:
    ensure_dir(out_dir)
    session = make_session()
    all_rows: list[ProductUrlRow] = []
    seen_products: set[str] = set()
    listing_dir = out_dir / "listing_html"
    for category_url in category_urls:
        first_url = listing_page_url(category_url, 0)
        first_html = fetch(session, first_url).text
        max_cur_pos = extract_listing_max_cur_pos(first_html, category_url)
        cur_positions = list(range(0, max_cur_pos + 1, 100)) if max_cur_pos else [0]
        if max_pages is not None:
            cur_positions = cur_positions[: max(1, max_pages)]
        for page_idx, cur_pos in enumerate(tqdm(cur_positions, desc=f"Shtorystore {category_name_from_url(category_url)} listings"), start=1):
            page_url = listing_page_url(category_url, cur_pos)
            try:
                html_text = first_html if cur_pos == 0 else fetch(session, page_url).text
            except Exception as exc:
                eprint(f"[WARN] listing fetch failed: {page_url}: {exc}")
                continue
            if save_html:
                ensure_dir(listing_dir)
                name = f"{clean_filename(category_name_from_url(category_url))}_{page_idx:03d}.html"
                (listing_dir / name).write_text(html_text, encoding="utf-8")
            for row in extract_listing_rows(html_text, category_url, page_url, page_idx):
                if row.product_url not in seen_products:
                    seen_products.add(row.product_url)
                    all_rows.append(row)
            (out_dir / "product_urls.partial.jsonl").write_text(
                "\n".join(json.dumps(asdict(row), ensure_ascii=False) for row in all_rows) + ("\n" if all_rows else ""),
                encoding="utf-8",
            )
            if sleep_sec > 0:
                time.sleep(sleep_sec)
    return all_rows


def option_values(soup: BeautifulSoup, selector: str) -> tuple[list[float], float | None]:
    values: list[float] = []
    selected: float | None = None
    for option in soup.select(f"{selector} option"):
        if option.get("id"):
            continue
        value = parse_number(option.get("value") or option.get_text(" ", strip=True))
        if value is None:
            continue
        values.append(value)
        if option.has_attr("selected"):
            selected = value
    return values, selected


def option_texts(soup: BeautifulSoup, selector: str) -> list[str]:
    out: list[str] = []
    for option in soup.select(f"{selector} option"):
        text = norm_text(option.get_text(" ", strip=True))
        if text and text.lower() != "свое":
            out.append(text)
    return out


def selected_option_text(soup: BeautifulSoup, selector: str) -> str:
    selected = soup.select_one(f"{selector} option[selected]")
    if selected:
        return norm_text(selected.get_text(" ", strip=True))
    first = soup.select_one(f"{selector} option:not([id])")
    return norm_text(first.get_text(" ", strip=True) if first else "")


def parse_material_blocks(soup: BeautifulSoup) -> dict[str, dict[str, str]]:
    root = soup.select_one(".material-yakor")
    if not root:
        return {}
    materials: dict[str, dict[str, str]] = {}
    current_name = ""
    for child in root.find_all(["div"], recursive=False):
        classes = child.get("class") or []
        if "top-list-krep" in classes:
            current_name = norm_text(child.get_text(" ", strip=True))
            materials.setdefault(current_name, {})
        elif "bot-list-krep" in classes and current_name:
            props: dict[str, str] = {}
            for item in child.select(".v_table"):
                key = norm_text(item.select_one("strong").get_text(" ", strip=True) if item.select_one("strong") else "")
                value_nodes = item.select(".text")
                value = norm_text(" ".join(v.get_text(" ", strip=True) for v in value_nodes) or item.get_text(" ", strip=True).replace(key, ""))
                if key and value:
                    props[key] = value
            if not props:
                props["Описание"] = norm_text(child.get_text(" ", strip=True))
            materials[current_name] = props
    return materials


def parse_description(soup: BeautifulSoup) -> str:
    root = soup.select_one(".descri-yakor")
    if not root:
        return ""
    parts: list[str] = []
    left = root.select_one(".left-opisanie")
    if left:
        parts.append(norm_text(left.get_text(" ", strip=True)))
    care = [norm_text(x.get_text(" ", strip=True)) for x in root.select(".anons-rekomend-yhod")]
    if care:
        parts.append("Уход: " + "; ".join(care))
    note = root.select_one(".text-illustr-client")
    if note:
        parts.append(norm_text(note.get_text(" ", strip=True)))
    return norm_text(" ".join(parts))


def extract_images(soup: BeautifulSoup, page_url: str) -> list[str]:
    images: list[str] = []
    seen: set[str] = set()
    for slide in soup.select(".pic-gallery-big .one-gallery-big, .width-gallery-big .one-gallery-big"):
        img = slide.select_one("img")
        link = slide.select_one("a.pic-cart[href]")
        srcset = str(img.get("srcset") or "").strip() if img else ""
        raw = ""
        if srcset:
            raw = srcset.split(",", 1)[0].strip().split(" ", 1)[0]
        if not raw and img:
            raw = str(img.get("src") or "").strip()
        if not raw and link:
            raw = str(link.get("href") or "").strip()
        url = absolutize(raw, page_url) if raw else ""
        if url and url not in seen:
            seen.add(url)
            images.append(url)
    return images


def parse_product(session: requests.Session, row: ProductUrlRow) -> ProductRow:
    try:
        response = fetch(session, row.product_url)
        soup = BeautifulSoup(response.text, "html.parser")
        images = extract_images(soup, response.url)
        selected_image_index: int | None = None
        selected_image_url = ""
        note = ""
        if len(images) >= 2:
            selected_image_index = 2
            selected_image_url = images[1]
            note = "second_gallery_image"
        elif images:
            selected_image_index = 1
            selected_image_url = images[0]
            note = "fallback_only_one_gallery_image"
        elif row.listing_image:
            selected_image_index = None
            selected_image_url = row.listing_image
            note = "fallback_listing_image_no_gallery"
        widths, selected_width = option_values(soup, ".left-sel-tov")
        heights, selected_height = option_values(soup, ".right-sel-tov")
        materials = parse_material_blocks(soup)
        selected_material = selected_option_text(soup, "#select-material")
        title = norm_text(soup.select_one("h1").get_text(" ", strip=True) if soup.select_one("h1") else row.title)
        product_id = norm_text(soup.select_one("input[name='rowID']").get("value") if soup.select_one("input[name='rowID']") else "")
        category = norm_text(soup.select_one("input[name='categoryItem']").get("value") if soup.select_one("input[name='categoryItem']") else row.category_name)
        return ProductRow(
            url=row.product_url,
            final_url=strip_query(response.url),
            name=title,
            sku=product_id or urlparse(row.product_url).path.rstrip("/").split("/")[-1],
            category=category or row.category_name,
            price=str(parse_price(soup.select_one(".price-buy-cart").get_text(" ", strip=True) if soup.select_one(".price-buy-cart") else row.listing_price) or ""),
            old_price=norm_text(soup.select_one(".old-buy-cart").get_text(" ", strip=True) if soup.select_one(".old-buy-cart") else row.listing_old_price),
            description=parse_description(soup),
            selected_material=selected_material,
            materials=materials,
            attachments=option_texts(soup, "#select-kreplenie"),
            widths_cm=widths,
            heights_cm=heights,
            selected_width_cm=selected_width,
            selected_height_cm=selected_height,
            raw_properties={
                "category_url": row.category_url,
                "listing_title": row.title,
                "material_options": option_texts(soup, "#select-material"),
                "attachment_options": option_texts(soup, "#select-kreplenie"),
                "selected_material_properties": materials.get(selected_material, {}),
            },
            images=images,
            selected_image_url=selected_image_url,
            selected_image_index=selected_image_index,
            image_selection_note=note,
        )
    except Exception as exc:
        return ProductRow(
            url=row.product_url,
            name=row.title,
            sku=urlparse(row.product_url).path.rstrip("/").split("/")[-1],
            category=row.category_name,
            price=str(parse_price(row.listing_price) or ""),
            old_price=row.listing_old_price,
            images=[row.listing_image] if row.listing_image else [],
            selected_image_url=row.listing_image,
            image_selection_note="error_fallback_listing_image",
            parse_status="error",
            error=str(exc),
        )


def parse_products(rows: list[ProductUrlRow], workers: int, sleep_sec: float) -> list[ProductRow]:
    indexed: dict[str, ProductRow] = {}

    def task(row: ProductUrlRow) -> ProductRow:
        product = parse_product(make_session(), row)
        if sleep_sec > 0:
            time.sleep(sleep_sec)
        return product

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(task, row) for row in rows]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Shtorystore product pages"):
            product = future.result()
            indexed[product.url] = product
    return [indexed[row.product_url] for row in rows if row.product_url in indexed]


def image_extension(response: requests.Response, url: str) -> str:
    ctype = response.headers.get("content-type", "").split(";")[0].strip().lower()
    ext = mimetypes.guess_extension(ctype) if ctype else ""
    if ext == ".jpe":
        ext = ".jpg"
    if ext:
        return ext
    suffix = Path(urlparse(url).path).suffix.lower()
    return suffix if suffix in {".jpg", ".jpeg", ".png", ".webp"} else ".jpg"


def download_selected_images(products: list[ProductRow], out_dir: Path, workers: int) -> None:
    image_root = out_dir / "material_images"
    ensure_dir(image_root)

    def one(item: tuple[int, ProductRow]) -> tuple[int, str]:
        idx, product = item
        if not product.selected_image_url:
            return idx, ""
        try:
            response = fetch(make_session(), product.selected_image_url, timeout=60)
            folder = image_root / f"{idx + 1:05d}_{clean_filename(product.sku or product.name, 80)}"
            ensure_dir(folder)
            ext = image_extension(response, product.selected_image_url)
            path = folder / f"02_{stable_hash(product.selected_image_url)}{ext}"
            path.write_bytes(response.content)
            return idx, str(path.relative_to(out_dir))
        except Exception as exc:
            product.error = norm_text(f"{product.error}; image_download_error: {exc}".strip("; "))
            if product.parse_status == "ok":
                product.parse_status = "image_error"
            return idx, ""

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(one, item) for item in enumerate(products)]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Shtorystore selected images"):
            idx, rel_path = future.result()
            if rel_path:
                products[idx].local_image_paths = [rel_path]


def normalized_row(product: ProductRow, out_dir: Path) -> dict[str, Any]:
    selected_material_props = product.materials.get(product.selected_material, {})
    search_text = norm_text(
        " ".join(
            [
                product.name,
                product.category,
                product.selected_material,
                product.description,
                " ".join(product.attachments),
                " ".join(f"{k} {v}" for k, v in selected_material_props.items()),
            ]
        )
    ).lower()
    row = {
        "version": "window_material.v1",
        "source": "shtorystore",
        "sku": product.sku,
        "name": product.name,
        "brand": product.brand,
        "product_url": product.final_url or product.url,
        "price": parse_price(product.price),
        "old_price": parse_price(product.old_price),
        "price_currency": product.price_currency,
        "availability": "made_to_order",
        "material_type": "curtain" if "штор" in product.category.lower() else "tulle",
        "surface_group": "window_covering",
        "category": product.category,
        "selected_material": product.selected_material,
        "selected_material_properties": selected_material_props,
        "all_materials": product.materials,
        "attachments": product.attachments,
        "widths_cm": product.widths_cm,
        "heights_cm": product.heights_cm,
        "selected_width_cm": product.selected_width_cm,
        "selected_height_cm": product.selected_height_cm,
        "description": product.description,
        "raw_properties": product.raw_properties,
        "image_urls": product.images,
        "selected_image_url": product.selected_image_url,
        "selected_image_index": product.selected_image_index,
        "image_selection_note": product.image_selection_note,
        "local_image_paths": product.local_image_paths,
        "style_tags": ["curtain", "window_covering", product.selected_material.lower()] if product.selected_material else ["curtain", "window_covering"],
        "room_suitability": ["bedroom", "living_room", "children", "office", "kitchen"],
        "search_text": search_text,
        "parse_status": product.parse_status,
    }
    if analyze_floor_material_colors is not None:
        row.update(analyze_floor_material_colors(out_dir, product.local_image_paths))
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def write_outputs(out_dir: Path, url_rows: list[ProductUrlRow], products: list[ProductRow], normalized: list[dict[str, Any]]) -> None:
    ensure_dir(out_dir)
    url_dicts = [asdict(row) for row in url_rows]
    product_dicts = [asdict(product) for product in products]
    write_csv(out_dir / "product_urls.csv", url_dicts)
    write_csv(out_dir / "products.csv", product_dicts)
    write_csv(out_dir / "shtorystore_curtains.csv", normalized)
    (out_dir / "product_urls.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in url_dicts) + "\n", encoding="utf-8")
    (out_dir / "products.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in product_dicts) + "\n", encoding="utf-8")
    (out_dir / "shtorystore_curtains.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in normalized) + "\n", encoding="utf-8")
    analytics = {
        "root": str(out_dir),
        "schema": "shtorystore_curtains_bundle/v1",
        "products": len(products),
        "ok": sum(1 for p in products if p.parse_status == "ok"),
        "errors": sum(1 for p in products if p.parse_status != "ok"),
        "with_price": sum(1 for row in normalized if row.get("price") is not None),
        "with_selected_image": sum(1 for p in products if p.local_image_paths),
        "second_image_downloaded": sum(1 for p in products if p.image_selection_note == "second_gallery_image" and p.local_image_paths),
        "fallback_image_downloaded": sum(1 for p in products if p.image_selection_note != "second_gallery_image" and p.local_image_paths),
    }
    (out_dir / "analytics_current.json").write_text(json.dumps(analytics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "shtorystore_curtains_analytics.json").write_text(json.dumps(analytics, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Scrape Shtorystore curtains/tulles and download the second gallery image.")
    parser.add_argument("--category-url", action="append", default=None, help="Can be passed multiple times. Defaults to /shtory/ and /tyuli/.")
    parser.add_argument("-o", "--out-dir", type=Path, default=Path("data/floor_materials/shtorystore_curtains"))
    parser.add_argument("--max-pages", type=int, default=None, help="Limit listing pages per category.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--image-workers", type=int, default=4)
    parser.add_argument("--listing-sleep", type=float, default=0.0)
    parser.add_argument("--sleep", type=float, default=0.0, help="Delay after each product page request.")
    parser.add_argument("--no-download-images", action="store_true")
    parser.add_argument("--save-html", action="store_true")
    args = parser.parse_args()

    category_urls = args.category_url or DEFAULT_CATEGORY_URLS
    url_rows = collect_product_urls(category_urls, args.out_dir, args.max_pages, args.listing_sleep, args.save_html)
    eprint(f"[INFO] collected product urls: {len(url_rows)}")
    products = parse_products(url_rows, args.workers, args.sleep)
    if not args.no_download_images:
        download_selected_images(products, args.out_dir, args.image_workers)
    normalized = [normalized_row(product, args.out_dir) for product in products]
    write_outputs(args.out_dir, url_rows, products, normalized)
    eprint(f"[INFO] wrote {len(normalized)} rows to {args.out_dir / 'shtorystore_curtains.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
