#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Collect Tony Roaks products that have a "Скачать 3D модель" link.

The site is a Tilda store. Category pages load product cards from the Tilda
Store API, which already returns the same product object used in product cards:
description HTML, price, gallery, options, editions, dimensions and links.
This scraper records model links but intentionally does not download archives.

Examples:
    python3 -m src.tools.tonyroaks_3d_model_catalog_scraper all
    python3 -m src.tools.tonyroaks_3d_model_catalog_scraper all --discover-pages
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://tonyroaks.ru"
TILDA_API_BASE = "https://store.tildaapi.com/api/getproductslist/"
DEFAULT_CATEGORY_URLS = [
    "https://tonyroaks.ru/tables",
    "https://tonyroaks.ru/chairs",
    "https://tonyroaks.ru/storagesystems",
    "https://tonyroaks.ru/accessories",
]
SITEMAP_URLS = [
    "https://tonyroaks.ru/sitemap.xml",
    "https://tonyroaks.ru/sitemap-store.xml",
]
LEGACY_CATEGORY_ROOTS = {
    "tables": [("tables", "486877244"), ("tables_old", "486877244")],
    "storagesystems": [("storagesystems", "499966435"), ("storagesystems_old", "499966435")],
}
DEFAULT_OUT_DIR = "out/supplier_ingest/tonyroaks/catalog"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/121.0.0.0 Safari/537.36"
)


@dataclass(frozen=True)
class StoreBlock:
    category_url: str
    category_path: str
    category_title: str
    recid: str
    storepartuid: str
    section_title: str = ""


@dataclass
class ProductUrlRow:
    category_url: str
    category_title: str
    section_title: str
    recid: str
    storepartuid: str
    product_url: str
    product_uid: str
    product_title: str


@dataclass
class ProductRow:
    url: str
    name: str = ""
    product_uid: str = ""
    external_id: str = ""
    sku: str = ""
    price: int | None = None
    old_price: int | None = None
    price_currency: str = "RUB"
    description_html: str = ""
    description: str = ""
    category_url: str = ""
    category_path: str = ""
    category_title: str = ""
    section_title: str = ""
    recid: str = ""
    storepartuid: str = ""
    categories: list[str] = field(default_factory=list)
    properties: dict[str, str] = field(default_factory=dict)
    dimensions: dict[str, Any] = field(default_factory=dict)
    images: list[str] = field(default_factory=list)
    options: list[dict[str, Any]] = field(default_factory=list)
    variants: list[dict[str, Any]] = field(default_factory=list)
    model_links: list[dict[str, str]] = field(default_factory=list)
    raw_product: dict[str, Any] = field(default_factory=dict)


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def norm_text(value: Any) -> str:
    text = str(value or "").replace("\xa0", " ").replace("\u2800", " ")
    return re.sub(r"\s+", " ", text).strip()


def lower(value: Any) -> str:
    return norm_text(value).lower().replace("ё", "е")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.7",
        }
    )
    return session


def absolutize(url: str, base_url: str = BASE_URL) -> str:
    return urljoin(base_url, url or "")


def normalize_url(url: str, keep_query: bool = False) -> str:
    parsed = urlparse(absolutize(url))
    query = parsed.query if keep_query else ""
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path.rstrip("/") or "/", "", query, ""))


def with_trailing_slash(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path if parsed.path.endswith("/") else parsed.path + "/"
    return urlunparse((parsed.scheme, parsed.netloc, path, "", parsed.query, ""))


def parse_int(value: Any) -> int | None:
    text = norm_text(value)
    if not text:
        return None
    digits = re.sub(r"[^\d]", "", text)
    return int(digits) if digits else None


def parse_price(value: Any) -> int | None:
    text = norm_text(value)
    if not text:
        return None
    text = text.replace(" ", "").replace(",", ".")
    match = re.search(r"\d+(?:\.\d+)?", text)
    if not match:
        return None
    return int(round(float(match.group(0))))


def fetch_text(session: requests.Session, url: str, *, referer: str | None = None, timeout: tuple[int, int] = (20, 70)) -> str:
    headers = {"Referer": referer} if referer else None
    response = session.get(url, headers=headers, timeout=timeout, allow_redirects=True)
    if response.status_code == 403 and not urlparse(url).path.endswith("/"):
        response = session.get(with_trailing_slash(url), headers=headers, timeout=timeout, allow_redirects=True)
    response.raise_for_status()
    response.encoding = response.encoding or "utf-8"
    return response.text


def title_from_html(html_text: str) -> str:
    soup = BeautifulSoup(html_text, "html.parser")
    for selector in ["h1", "title"]:
        node = soup.select_one(selector)
        title = norm_text(node.get_text(" ", strip=True) if node else "")
        if title:
            return title
    return ""


def discover_category_urls(session: requests.Session, seed_urls: list[str]) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()

    def add(url: str) -> None:
        parsed = urlparse(normalize_url(url))
        if parsed.netloc != urlparse(BASE_URL).netloc:
            return
        path = parsed.path.strip("/")
        if not path or "/tproduct/" in path:
            return
        if path in {"contacts", "partners", "policy", "help", "rulers", "trpremium", "oferta"}:
            return
        normalized = normalize_url(url)
        if normalized not in seen:
            seen.add(normalized)
            urls.append(normalized)

    for url in seed_urls:
        add(url)

    try:
        html_text = fetch_text(session, BASE_URL)
    except Exception as exc:  # noqa: BLE001 - discovery fallback uses defaults.
        eprint(f"[warn] cannot fetch homepage for discovery: {exc}")
        return urls

    soup = BeautifulSoup(html_text, "html.parser")
    for anchor in soup.find_all("a", href=True):
        add(anchor.get("href", ""))
    return urls


def urls_from_sitemap(session: requests.Session, sitemap_url: str) -> list[str]:
    try:
        text = fetch_text(session, sitemap_url, referer=BASE_URL)
    except Exception as exc:  # noqa: BLE001 - sitemap discovery is best effort.
        eprint(f"[warn] cannot fetch sitemap {sitemap_url}: {exc}")
        return []
    urls: list[str] = []
    try:
        root = ET.fromstring(text)
        for node in root.findall(".//{*}loc"):
            if node.text:
                urls.append(norm_text(node.text))
    except ET.ParseError:
        urls.extend(re.findall(r"<loc>(.*?)</loc>", text))
    return urls


def discover_site_urls(session: requests.Session, seed_urls: list[str]) -> tuple[list[str], list[str]]:
    page_urls: list[str] = []
    product_urls: list[str] = []
    seen_pages: set[str] = set()
    seen_products: set[str] = set()

    def add_page(url: str) -> None:
        normalized = normalize_url(url)
        parsed = urlparse(normalized)
        if parsed.netloc != urlparse(BASE_URL).netloc:
            return
        path = parsed.path.strip("/")
        if not path:
            normalized = BASE_URL
        if "/tproduct/" in parsed.path:
            add_product(normalized)
            return
        if normalized not in seen_pages:
            seen_pages.add(normalized)
            page_urls.append(normalized)

    def add_product(url: str) -> None:
        normalized = normalize_url(url)
        parsed = urlparse(normalized)
        if parsed.netloc != urlparse(BASE_URL).netloc or "/tproduct/" not in parsed.path:
            return
        if normalized not in seen_products:
            seen_products.add(normalized)
            product_urls.append(normalized)

    for url in seed_urls:
        add_page(url)
    add_page(BASE_URL)

    for sitemap_url in SITEMAP_URLS:
        for url in urls_from_sitemap(session, sitemap_url):
            if "/tproduct/" in url:
                add_product(url)
            else:
                add_page(url)

    return page_urls, product_urls


def extract_product_urls_from_html(html_text: str) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    soup = BeautifulSoup(html_text, "html.parser")
    for anchor in soup.find_all("a", href=True):
        href = normalize_url(anchor.get("href", ""))
        if "/tproduct/" in urlparse(href).path and href not in seen:
            seen.add(href)
            urls.append(href)
    for raw in re.findall(r"https?://tonyroaks\.ru/[^\"'<> ]+/tproduct/[^\"'<> )]+", html_text):
        href = normalize_url(raw.replace("\\/", "/"))
        if href not in seen:
            seen.add(href)
            urls.append(href)
    return urls


def section_titles_from_tabs(html_text: str) -> dict[str, str]:
    titles: dict[str, str] = {}
    for recids, title in re.findall(r'<option value="([0-9,]+)">([^<]+)</option>', html_text):
        clean_title = norm_text(BeautifulSoup(title, "html.parser").get_text(" ", strip=True))
        for recid in recids.split(","):
            if recid and clean_title and clean_title.lower() != "все изделия":
                titles.setdefault(recid, clean_title)
    return titles


def extract_store_blocks_from_html(html_text: str, category_url: str) -> list[StoreBlock]:
    category_title = title_from_html(html_text)
    category_path = urlparse(normalize_url(category_url)).path.strip("/")
    tabs = section_titles_from_tabs(html_text)
    blocks: list[StoreBlock] = []
    seen: set[tuple[str, str]] = set()
    pattern = re.compile(r"recid:'(\d+)'.{0,900}?storepart:'(\d+)'", re.DOTALL)
    for recid, storepartuid in pattern.findall(html_text):
        key = (recid, storepartuid)
        if key in seen:
            continue
        seen.add(key)
        blocks.append(
            StoreBlock(
                category_url=normalize_url(category_url),
                category_path=category_path,
                category_title=category_title,
                recid=recid,
                storepartuid=storepartuid,
                section_title=tabs.get(recid, ""),
            )
        )
    return blocks


def extract_store_blocks(session: requests.Session, category_url: str) -> list[StoreBlock]:
    html_text = fetch_text(session, category_url, referer=BASE_URL)
    return extract_store_blocks_from_html(html_text, category_url)


def extract_js_object_after(text: str, marker: str = "var product = ") -> dict[str, Any] | None:
    idx = text.find(marker)
    if idx < 0:
        return None
    start = text.find("{", idx)
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for pos in range(start, len(text)):
        char = text[pos]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                raw = text[start : pos + 1]
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    return None
    return None


def block_from_product_url(product_url: str, product: dict[str, Any] | None = None) -> StoreBlock:
    parsed = urlparse(normalize_url(product_url))
    parts = [part for part in parsed.path.strip("/").split("/") if part]
    category_path = parts[0] if parts else ""
    recid = parts[2].split("-", 1)[0] if len(parts) >= 3 and parts[1] == "tproduct" else ""
    storepartuid = recid
    if product:
        partuids = parse_json_field(product.get("partuids"), [])
        if isinstance(partuids, list) and partuids:
            storepartuid = str(partuids[0])
    return StoreBlock(
        category_url=f"{BASE_URL}/{category_path}" if category_path else BASE_URL,
        category_path=category_path,
        category_title=category_path,
        recid=recid,
        storepartuid=storepartuid,
        section_title="",
    )


def product_url_parts(product_url: str) -> tuple[str, str, str] | None:
    parsed = urlparse(normalize_url(product_url))
    parts = [part for part in parsed.path.strip("/").split("/") if part]
    if len(parts) < 3 or parts[1] != "tproduct":
        return None
    product_id = parts[2]
    match = re.match(r"(\d+)-(\d+)-(.+)", product_id)
    if not match:
        return None
    return parts[0], match.group(2), match.group(3)


def legacy_alias_urls(product_url: str) -> list[str]:
    parsed = product_url_parts(product_url)
    if not parsed:
        return []
    category_path, uid, slug = parsed
    candidates: list[tuple[str, str]] = []
    candidates.extend(LEGACY_CATEGORY_ROOTS.get(category_path, []))
    if category_path in {"chairs", "test_chair"}:
        if "skamya" in slug or "banketka" in slug:
            candidates.extend([("chairs", "1385624671"), ("test_chair", "1385624671")])
        if "stul" in slug:
            candidates.extend([("chairs", "1385625151"), ("test_chair", "1385625151")])
    if category_path in {"accessories", "accessories_test"}:
        if "zerkalo" in slug:
            candidates.extend([("accessories", "1385641101"), ("accessories_test", "1385641101")])
        if "torsher" in slug or "svetil" in slug or "chaplet" in slug or "chaplit" in slug:
            candidates.extend([("accessories", "1385641121"), ("accessories_test", "1385641121")])
    urls: list[str] = []
    seen: set[str] = set()
    for alias_path, rootpart in candidates:
        url = normalize_url(f"{BASE_URL}/{alias_path}/tproduct/{rootpart}-{uid}-{slug}")
        if url != normalize_url(product_url) and url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def parse_product_page(session: requests.Session, product_url: str) -> ProductRow | None:
    html_text = fetch_text(session, product_url, referer=BASE_URL)
    product = extract_js_object_after(html_text)
    if not product:
        return None
    block = block_from_product_url(product_url, product)
    row = normalize_product(product, block)
    row.url = normalize_url(product_url)
    return row


def tilda_api_url(block: StoreBlock, slice_number: int, size: int) -> str:
    params = {
        "storepartuid": block.storepartuid,
        "recid": block.recid,
        "c": int(time.time() * 1000),
        "getparts": "true",
        "getoptions": "true",
        "size": size,
        "slice": slice_number,
        "flag_root": "withroot",
    }
    return f"{TILDA_API_BASE}?{urlencode(params)}"


def load_products_for_block(session: requests.Session, block: StoreBlock, *, size: int = 100) -> list[dict[str, Any]]:
    products: list[dict[str, Any]] = []
    seen: set[str] = set()
    total: int | None = None
    for slice_number in range(1, 200):
        url = tilda_api_url(block, slice_number, size)
        response = session.get(url, headers={"Referer": block.category_url}, timeout=(20, 70))
        response.raise_for_status()
        data = response.json()
        if total is None:
            total = parse_int(data.get("total"))
        batch = data.get("products") or []
        if not isinstance(batch, list) or not batch:
            break
        new_count = 0
        for product in batch:
            if not isinstance(product, dict):
                continue
            uid = str(product.get("uid") or "")
            key = uid or str(product.get("url") or "")
            if key in seen:
                continue
            seen.add(key)
            products.append(product)
            new_count += 1
        if total is not None and len(products) >= total:
            break
        if new_count == 0 or len(batch) < size:
            break
    return products


def parse_json_field(value: Any, fallback: Any) -> Any:
    if isinstance(value, (list, dict)):
        return value
    if not isinstance(value, str) or not value.strip() or value.strip() == "null":
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return fallback


def image_list(product: dict[str, Any]) -> list[str]:
    gallery = parse_json_field(product.get("gallery"), [])
    images: list[str] = []
    if isinstance(gallery, list):
        for item in gallery:
            if isinstance(item, dict):
                img = norm_text(item.get("img"))
            else:
                img = norm_text(item)
            if img and img not in images:
                images.append(img)
    img = norm_text(product.get("img"))
    if img and img not in images:
        images.insert(0, img)
    return images


def extract_model_links(description_html: str) -> list[dict[str, str]]:
    soup = BeautifulSoup(description_html or "", "html.parser")
    links: list[dict[str, str]] = []
    seen: set[str] = set()
    for anchor in soup.find_all("a", href=True):
        text = norm_text(anchor.get_text(" ", strip=True))
        text_l = lower(text)
        href = absolutize(anchor.get("href", ""))
        href_l = href.lower()
        is_model = "скачать" in text_l and ("3d" in text_l or "3д" in text_l) and "модел" in text_l
        is_disk_model = "disk.yandex" in href_l and ("модел" in text_l or "3d" in text_l or "3д" in text_l)
        if not is_model and not is_disk_model:
            continue
        if href in seen:
            continue
        seen.add(href)
        links.append({"title": text or "Скачать 3D модель", "url": href, "source": "description"})
    return links


def description_text(description_html: str) -> str:
    soup = BeautifulSoup(description_html or "", "html.parser")
    return norm_text(soup.get_text("\n", strip=True))


def extract_properties(description: str) -> dict[str, str]:
    props: dict[str, str] = {}
    for line in description.split("\n"):
        line = norm_text(line)
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = norm_text(key)
        value = norm_text(value)
        if key and value and len(key) <= 80:
            props[key] = value
    return props


def product_dimensions(product: dict[str, Any], variants: list[dict[str, Any]]) -> dict[str, Any]:
    dims: dict[str, Any] = {}
    for key in ["pack_label", "pack_x", "pack_y", "pack_z", "pack_m"]:
        value = product.get(key)
        if value not in (None, "", 0, "0"):
            dims[key] = value
    variant_dims: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any, Any]] = set()
    for variant in variants:
        triple = (variant.get("pack_x"), variant.get("pack_y"), variant.get("pack_z"))
        if not any(triple) or triple in seen:
            continue
        seen.add(triple)
        variant_dims.append(
            {
                "pack_x": variant.get("pack_x"),
                "pack_y": variant.get("pack_y"),
                "pack_z": variant.get("pack_z"),
                "pack_label": product.get("pack_label") or "lwh",
            }
        )
    if variant_dims:
        dims["variant_dimensions"] = variant_dims
    return dims


def normalize_product(product: dict[str, Any], block: StoreBlock) -> ProductRow:
    description_html = str(product.get("text") or "")
    descr = description_text(description_html)
    variants = product.get("editions") if isinstance(product.get("editions"), list) else []
    options = parse_json_field(product.get("json_options"), [])
    if not isinstance(options, list):
        options = []
    url = normalize_url(str(product.get("url") or ""))
    return ProductRow(
        url=url,
        name=norm_text(product.get("title")),
        product_uid=str(product.get("uid") or ""),
        external_id=norm_text(product.get("externalid")),
        sku=norm_text(product.get("sku")),
        price=parse_price(product.get("price")),
        old_price=parse_price(product.get("priceold")),
        description_html=description_html,
        description=descr,
        category_url=block.category_url,
        category_path=block.category_path,
        category_title=block.category_title,
        section_title=block.section_title,
        recid=block.recid,
        storepartuid=block.storepartuid,
        categories=[x for x in [block.category_title, block.section_title] if x],
        properties=extract_properties(descr),
        dimensions=product_dimensions(product, variants),
        images=image_list(product),
        options=options,
        variants=variants,
        model_links=extract_model_links(description_html),
        raw_product=product,
    )


def write_jsonl(path: Path, rows: list[Any]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            data = asdict(row) if hasattr(row, "__dataclass_fields__") else row
            fh.write(json.dumps(data, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[Any]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    dict_rows = [asdict(row) if hasattr(row, "__dataclass_fields__") else dict(row) for row in rows]
    fieldnames = list(dict_rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in dict_rows:
            flat = {
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            }
            writer.writerow(flat)


def collect(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out)
    ensure_dir(out_dir)
    session = make_session()

    category_urls = list(args.category_url or DEFAULT_CATEGORY_URLS)
    direct_product_urls: list[str] = []
    if args.discover_pages:
        page_urls, direct_product_urls = discover_site_urls(session, category_urls)
        category_urls = page_urls

    blocks: list[StoreBlock] = []
    seen_blocks: set[tuple[str, str]] = set()
    for category_url in category_urls:
        try:
            html_text = fetch_text(session, category_url, referer=BASE_URL)
            found = extract_store_blocks_from_html(html_text, category_url)
            for product_url in extract_product_urls_from_html(html_text):
                if product_url not in direct_product_urls:
                    direct_product_urls.append(product_url)
        except Exception as exc:  # noqa: BLE001 - keep scraping other sections.
            eprint(f"[warn] cannot parse category {category_url}: {exc}")
            continue
        if not found and args.verbose:
            eprint(f"[info] no store blocks: {category_url}")
        for block in found:
            key = (block.recid, block.storepartuid)
            if key not in seen_blocks:
                seen_blocks.add(key)
                blocks.append(block)

    all_products: list[ProductRow] = []
    product_urls: list[ProductUrlRow] = []
    seen_products: set[str] = set()
    for index, block in enumerate(blocks, 1):
        if args.verbose:
            eprint(
                f"[{index}/{len(blocks)}] {block.category_path} {block.section_title or block.category_title} "
                f"recid={block.recid} storepart={block.storepartuid}"
            )
        products = load_products_for_block(session, block, size=args.page_size)
        for product in products:
            row = normalize_product(product, block)
            key = row.url
            if key in seen_products:
                continue
            seen_products.add(key)
            all_products.append(row)
            product_urls.append(
                ProductUrlRow(
                    category_url=block.category_url,
                    category_title=block.category_title,
                    section_title=block.section_title,
                    recid=block.recid,
                    storepartuid=block.storepartuid,
                    product_url=row.url,
                    product_uid=row.product_uid,
                    product_title=row.name,
                )
            )

    if args.legacy_aliases:
        alias_sources = [row.url for row in all_products] + direct_product_urls
        for source_url in alias_sources:
            for alias_url in legacy_alias_urls(source_url):
                if alias_url not in direct_product_urls:
                    direct_product_urls.append(alias_url)

    for index, product_url in enumerate(direct_product_urls, 1):
        normalized_url = normalize_url(product_url)
        if normalized_url in seen_products:
            continue
        if args.verbose:
            eprint(f"[direct {index}/{len(direct_product_urls)}] {normalized_url}")
        try:
            row = parse_product_page(session, normalized_url)
        except Exception as exc:  # noqa: BLE001 - keep scraping other product pages.
            eprint(f"[warn] cannot parse product {normalized_url}: {exc}")
            continue
        if not row:
            continue
        seen_products.add(row.url)
        all_products.append(row)
        product_urls.append(
            ProductUrlRow(
                category_url=row.category_url,
                category_title=row.category_title,
                section_title=row.section_title,
                recid=row.recid,
                storepartuid=row.storepartuid,
                product_url=row.url,
                product_uid=row.product_uid,
                product_title=row.name,
            )
        )

    with_models = [row for row in all_products if row.model_links]
    without_models = [row for row in all_products if not row.model_links]

    write_jsonl(out_dir / "store_blocks.jsonl", blocks)
    write_csv(out_dir / "store_blocks.csv", blocks)
    write_jsonl(out_dir / "product_urls.jsonl", product_urls)
    write_csv(out_dir / "product_urls.csv", product_urls)
    write_jsonl(out_dir / "products_with_3d_models.jsonl", with_models)
    write_csv(out_dir / "products_with_3d_models.csv", with_models)
    write_jsonl(out_dir / "products_without_3d_models.jsonl", without_models)

    summary = {
        "category_urls": category_urls,
        "direct_product_urls": len(direct_product_urls),
        "store_blocks": len(blocks),
        "products_total": len(all_products),
        "products_with_3d_models": len(with_models),
        "products_without_3d_models": len(without_models),
        "out_dir": str(out_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    all_parser = subparsers.add_parser("all", help="Collect Tony Roaks products with 3D model links")
    all_parser.add_argument("--out", default=DEFAULT_OUT_DIR, help="Output directory")
    all_parser.add_argument("--category-url", action="append", help="Category URL to scan; can be repeated")
    all_parser.add_argument("--discover-pages", action="store_true", help="Discover category pages from homepage links")
    all_parser.add_argument(
        "--no-legacy-aliases",
        action="store_false",
        dest="legacy_aliases",
        help="Do not probe old/test Tilda product URL aliases",
    )
    all_parser.add_argument("--page-size", type=int, default=100, help="Tilda API page size")
    all_parser.add_argument("--verbose", action="store_true")
    all_parser.set_defaults(legacy_aliases=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "all":
        summary = collect(args)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    parser.error(f"unknown command: {args.cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
