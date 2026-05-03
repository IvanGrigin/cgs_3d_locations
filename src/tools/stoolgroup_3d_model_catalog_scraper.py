#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Scrape Stool Group products that expose a 3D model attachment.

The scraper visits category pages, opens product cards, and writes full product
metadata only when the product page has a "3D-модель" download item. It records
the attachment URL and file label, but intentionally does not download zip files.

Examples:
    python3 -m src.tools.stoolgroup_3d_model_catalog_scraper all \
      --out out/supplier_ingest/stoolgroup/catalog \
      --workers 4

    python3 -m src.tools.stoolgroup_3d_model_catalog_scraper parse-saved-html \
      --html /tmp/stool_product.html \
      --url https://stoolgroup.ru/divan-stoun-tkan-bukle-molochnyy/
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
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://stoolgroup.ru"
DEFAULT_CATEGORY_URLS = [
    "https://stoolgroup.ru/stulya/",
    "https://stoolgroup.ru/stoly/",
    "https://stoolgroup.ru/divany-i-kresla/",
    "https://stoolgroup.ru/kompyuternye-kresla/",
    "https://stoolgroup.ru/komplekty/",
    "https://stoolgroup.ru/banketnaya-mebel/",
    "https://stoolgroup.ru/sadovaya-mebel/",
    "https://stoolgroup.ru/stellazhi/",
    "https://stoolgroup.ru/home-light/",
]
DEFAULT_OUT_DIR = "out/supplier_ingest/stoolgroup/catalog"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/121.0.0.0 Safari/537.36"
)


@dataclass
class ProductUrlRow:
    category_url: str
    category_title: str
    page_number: int
    page_url: str
    product_url: str
    product_slug: str
    anchor_text: str = ""


@dataclass
class ProductRow:
    url: str
    final_url: str = ""
    name: str = ""
    sku: str = ""
    product_id: str = ""
    brand: str = ""
    price: int | None = None
    old_price: int | None = None
    price_currency: str = "RUB"
    availability: str = ""
    stock_text: str = ""
    description: str = ""
    breadcrumbs: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    properties: dict[str, str] = field(default_factory=dict)
    dimensions: dict[str, str] = field(default_factory=dict)
    colors: dict[str, str] = field(default_factory=dict)
    images: list[str] = field(default_factory=list)
    model_links: list[dict[str, str]] = field(default_factory=list)
    download_links: list[dict[str, str]] = field(default_factory=list)
    source_page_url: str = ""
    source_page_number: int | None = None
    parse_status: str = "ok"
    error: str = ""


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def norm_text(value: Any) -> str:
    text = str(value or "").replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def lower(value: Any) -> str:
    return norm_text(value).lower().replace("ё", "е")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def absolutize(url: str, base_url: str = BASE_URL) -> str:
    return urljoin(base_url, url or "")


def normalize_url(url: str, keep_query: bool = False) -> str:
    parsed = urlparse(absolutize(url))
    query = parsed.query if keep_query else ""
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", query, ""))


def product_slug_from_url(url: str) -> str:
    return urlparse(url).path.rstrip("/").split("/")[-1]


def category_page_url(category_url: str, page_number: int) -> str:
    category_url = normalize_url(category_url, keep_query=True)
    if page_number <= 1:
        return category_url
    parsed = urlparse(category_url)
    path = re.sub(r"/page-\d+/?$", "", parsed.path.rstrip("/"))
    return urlunparse((parsed.scheme, parsed.netloc, f"{path}/page-{page_number}/", "", parsed.query, ""))


def parse_int(text: Any) -> int | None:
    digits = re.sub(r"\D+", "", norm_text(text))
    return int(digits) if digits else None


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


def fetch_html(session: requests.Session, url: str, timeout: tuple[int, int] = (20, 70)) -> tuple[str | None, str, str]:
    try:
        response = session.get(url, timeout=timeout, allow_redirects=True)
        response.raise_for_status()
        response.encoding = response.encoding or "utf-8"
        return response.text, response.url, ""
    except Exception as exc:  # noqa: BLE001 - report per-URL parse errors in output.
        return None, url, f"{type(exc).__name__}: {exc}"


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
        obj_type = obj.get("@type")
        types = obj_type if isinstance(obj_type, list) else [obj_type]
        if "Product" in types:
            return obj
    return None


def extract_category_title(html_text: str) -> str:
    soup = BeautifulSoup(html_text, "html.parser")
    for selector in ["h1", "title"]:
        node = soup.select_one(selector)
        if node:
            title = norm_text(node.get_text(" ", strip=True))
            if title:
                return title
    return ""


def extract_product_links_from_listing(
    html_text: str,
    page_url: str,
    category_url: str,
    category_title: str,
    page_number: int,
) -> list[ProductUrlRow]:
    soup = BeautifulSoup(html_text, "html.parser")
    rows: list[ProductUrlRow] = []
    seen: set[str] = set()
    for card in soup.select(".product-card, form[name^='product_form']"):
        link = card.select_one("a.product-card__wrapper[href]")
        if not link:
            continue
        product_url = normalize_url(link.get("href", ""))
        parsed = urlparse(product_url)
        if parsed.netloc != urlparse(BASE_URL).netloc or not parsed.path.strip("/"):
            continue
        if product_url in seen:
            continue
        seen.add(product_url)
        title_node = card.select_one(".product-card__title")
        anchor_text = norm_text(title_node.get_text(" ", strip=True) if title_node else link.get_text(" ", strip=True))
        rows.append(
            ProductUrlRow(
                category_url=category_url,
                category_title=category_title,
                page_number=page_number,
                page_url=page_url,
                product_url=product_url,
                product_slug=product_slug_from_url(product_url),
                anchor_text=anchor_text,
            )
        )
    return rows


def extract_max_page_from_listing(html_text: str, page_url: str) -> int:
    soup = BeautifulSoup(html_text, "html.parser")
    pages = [1]
    for a in soup.select(".pagination a[href]"):
        href = absolutize(a.get("href", ""), page_url)
        match = re.search(r"/page-(\d+)/?$", urlparse(href).path)
        if match:
            pages.append(int(match.group(1)))
        text = norm_text(a.get_text(" ", strip=True))
        if text.isdigit():
            pages.append(int(text))
    return max(pages)


def extract_breadcrumbs(soup: BeautifulSoup) -> list[str]:
    values: list[str] = []
    for selector in [".breadcrumbs a, .breadcrumbs span", ".ty-breadcrumbs a, .ty-breadcrumbs span", '[itemtype*="BreadcrumbList"] [itemprop="name"]']:
        for node in soup.select(selector):
            text = norm_text(node.get_text(" ", strip=True) or node.get("content", ""))
            if text and text not in values:
                values.append(text)
    return values


def extract_price_from_product(soup: BeautifulSoup, product: dict[str, Any] | None) -> tuple[int | None, str]:
    if product:
        offers = product.get("offers")
        if isinstance(offers, dict):
            price = parse_int(offers.get("price"))
            currency = norm_text(offers.get("priceCurrency")) or "RUB"
            if price is not None:
                return price, currency
    node = soup.select_one(".product-interface__price .ty-price-num, [itemprop='price']")
    return parse_int(node.get_text(" ", strip=True) if node else ""), "RUB"


def extract_properties(soup: BeautifulSoup) -> dict[str, str]:
    props: dict[str, str] = {}
    for table in soup.select("table.table-product, table"):
        for row in table.select("tr"):
            cells = [norm_text(cell.get_text(" ", strip=True)) for cell in row.select("th,td")]
            cells = [cell for cell in cells if cell]
            if len(cells) >= 2 and 1 <= len(cells[0]) <= 100:
                props[cells[0]] = cells[1]
    return props


def extract_images(soup: BeautifulSoup, product: dict[str, Any] | None, base_url: str) -> list[str]:
    urls: list[str] = []

    def add(raw: Any) -> None:
        if not raw:
            return
        url = absolutize(str(raw), base_url)
        if url and url not in urls and not url.startswith("data:"):
            urls.append(url)

    if product:
        image = product.get("image")
        if isinstance(image, list):
            for item in image:
                add(item)
        else:
            add(image)
    for meta in soup.select('meta[property="og:image"][content], meta[itemprop="image"][content]'):
        add(meta.get("content"))
    gallery_selectors = [
        ".product-images img",
        ".product-gallery img",
        ".product-page__images img",
        ".product-interface__image img",
        ".product-detail img",
        "[data-ca-gallery-large-id] img",
        "[id^='det_img_link'] img",
        ".cm-image-previewer img",
    ]
    for img in soup.select(", ".join(gallery_selectors)):
        for attr in ["data-src", "data-srcset", "src", "content"]:
            value = img.get(attr)
            if not value:
                continue
            if attr.endswith("srcset"):
                value = str(value).split(",")[0].strip().split(" ")[0]
            add(value)
    return urls[:80]


def extract_download_links(soup: BeautifulSoup, base_url: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    downloads: list[dict[str, str]] = []
    model_links: list[dict[str, str]] = []
    for item in soup.select("a.download-list__item[href], a[href*='attachments.getfile']"):
        title_node = item.select_one(".download-list__item-title")
        file_node = item.select_one(".download-list__item-file")
        title = norm_text(title_node.get_text(" ", strip=True) if title_node else item.get_text(" ", strip=True))
        file_text = norm_text(file_node.get_text(" ", strip=True) if file_node else "")
        text = norm_text(item.get_text(" ", strip=True))
        link = {
            "title": title,
            "file_text": file_text,
            "text": text,
            "url": absolutize(item.get("href", ""), base_url),
        }
        if link["url"] not in {x["url"] for x in downloads}:
            downloads.append(link)
        haystack = lower(" ".join([title, file_text, text, link["url"]]))
        is_model = ("3d" in haystack and "модел" in haystack) or ("3d-модель" in haystack)
        if is_model and link["url"] not in {x["url"] for x in model_links}:
            model_links.append(link)
    return downloads, model_links


def select_prefixed(props: dict[str, str], keywords: Iterable[str]) -> dict[str, str]:
    keys = [lower(x) for x in keywords]
    return {key: value for key, value in props.items() if any(token in lower(key) for token in keys)}


def parse_product_html(html_text: str, url: str, final_url: str = "") -> ProductRow:
    soup = BeautifulSoup(html_text, "html.parser")
    product = extract_jsonld_product(soup)
    final = normalize_url(final_url or url, keep_query=True)
    name_node = soup.select_one("h1.product-interface__title, h1")
    name = norm_text(name_node.get_text(" ", strip=True) if name_node else "")
    if product:
        name = norm_text(product.get("name")) or name
    price, currency = extract_price_from_product(soup, product)
    brand = ""
    sku = ""
    description = ""
    availability = ""
    if product:
        sku = norm_text(product.get("sku"))
        brand_obj = product.get("brand")
        brand = norm_text(brand_obj.get("name") if isinstance(brand_obj, dict) else brand_obj)
        description = norm_text(BeautifulSoup(str(product.get("description") or ""), "html.parser").get_text(" ", strip=True))
        offers = product.get("offers")
        if isinstance(offers, dict):
            availability = norm_text(offers.get("availability")).split("/")[-1]
    if not description:
        meta_desc = soup.select_one('meta[name="description"][content], meta[property="og:description"][content]')
        description = norm_text(meta_desc.get("content", "") if meta_desc else "")
    if not name:
        title = soup.select_one("title")
        name = norm_text(title.get_text(" ", strip=True) if title else "")

    props = extract_properties(soup)
    downloads, model_links = extract_download_links(soup, final)
    stock_node = soup.select_one(".product-list-field.sw--amounts, .ty-control-group.product-list-field")
    product_id_node = soup.select_one('input[name="metrics[product_id]"], input[name="product_id"]')

    row = ProductRow(
        url=normalize_url(url, keep_query=True),
        final_url=final,
        name=name,
        sku=sku or props.get("Артикул", ""),
        product_id=norm_text(product_id_node.get("value", "") if product_id_node else ""),
        brand=brand or props.get("Бренд", ""),
        price=price,
        old_price=parse_int((soup.select_one(".product-interface__old-price, .ty-list-price") or "").get_text(" ", strip=True) if soup.select_one(".product-interface__old-price, .ty-list-price") else ""),
        price_currency=currency or "RUB",
        availability=availability,
        stock_text=norm_text(stock_node.get_text(" ", strip=True) if stock_node else ""),
        description=description,
        breadcrumbs=extract_breadcrumbs(soup),
        categories=[props.get("Категория", "")] if props.get("Категория") else [],
        properties=props,
        dimensions=select_prefixed(props, ["габарит", "высота", "ширина", "глубина", "размер", "вес", "нагрузка"]),
        colors=select_prefixed(props, ["цвет"]),
        images=extract_images(soup, product, final),
        model_links=model_links,
        download_links=downloads,
        parse_status="ok" if model_links else "no_3d_model",
    )
    return row


def write_jsonl(path: Path, rows: Iterable[Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            data = asdict(row) if hasattr(row, "__dataclass_fields__") else row
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


def write_product_csv(path: Path, rows: list[ProductRow]) -> None:
    ensure_dir(path.parent)
    fields = [
        "url",
        "final_url",
        "name",
        "sku",
        "product_id",
        "brand",
        "price",
        "price_currency",
        "availability",
        "stock_text",
        "category",
        "color",
        "dimensions",
        "model_url",
        "model_file_text",
        "images",
        "description",
        "properties_json",
        "download_links_json",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            model = row.model_links[0] if row.model_links else {}
            writer.writerow(
                {
                    "url": row.url,
                    "final_url": row.final_url,
                    "name": row.name,
                    "sku": row.sku,
                    "product_id": row.product_id,
                    "brand": row.brand,
                    "price": row.price,
                    "price_currency": row.price_currency,
                    "availability": row.availability,
                    "stock_text": row.stock_text,
                    "category": "; ".join(row.categories),
                    "color": "; ".join(f"{k}: {v}" for k, v in row.colors.items()),
                    "dimensions": "; ".join(f"{k}: {v}" for k, v in row.dimensions.items()),
                    "model_url": model.get("url", ""),
                    "model_file_text": model.get("file_text", ""),
                    "images": " | ".join(row.images),
                    "description": row.description,
                    "properties_json": json.dumps(row.properties, ensure_ascii=False),
                    "download_links_json": json.dumps(row.download_links, ensure_ascii=False),
                }
            )


def write_url_csv(path: Path, rows: list[ProductUrlRow]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ProductUrlRow.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def collect_urls(category_urls: list[str], max_pages: int, delay: float, out_dir: Path) -> list[ProductUrlRow]:
    session = make_session()
    rows: list[ProductUrlRow] = []
    seen: set[str] = set()
    for category_url in category_urls:
        first_url = category_page_url(category_url, 1)
        html_text, final_url, error = fetch_html(session, first_url)
        if error or not html_text:
            eprint(f"[WARN] listing failed {first_url}: {error}")
            continue
        category_title = extract_category_title(html_text)
        detected_pages = extract_max_page_from_listing(html_text, final_url)
        pages = min(max_pages, detected_pages) if max_pages else detected_pages
        for page_number in range(1, pages + 1):
            page_url = category_page_url(category_url, page_number)
            if page_number == 1:
                page_html = html_text
                page_final_url = final_url
            else:
                time.sleep(delay)
                page_html, page_final_url, error = fetch_html(session, page_url)
                if error or not page_html:
                    eprint(f"[WARN] listing failed {page_url}: {error}")
                    continue
            page_rows = extract_product_links_from_listing(page_html, page_final_url, category_url, category_title, page_number)
            eprint(f"[INFO] {page_url}: {len(page_rows)} product links")
            for row in page_rows:
                if row.product_url in seen:
                    continue
                seen.add(row.product_url)
                rows.append(row)
    write_url_csv(out_dir / "product_urls.csv", rows)
    write_jsonl(out_dir / "product_urls.jsonl", rows)
    return rows


def load_urls_csv(path: Path) -> list[ProductUrlRow]:
    rows: list[ProductUrlRow] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for raw in csv.DictReader(f):
            rows.append(
                ProductUrlRow(
                    category_url=raw.get("category_url", ""),
                    category_title=raw.get("category_title", ""),
                    page_number=int(raw.get("page_number") or 0),
                    page_url=raw.get("page_url", ""),
                    product_url=raw.get("product_url", ""),
                    product_slug=raw.get("product_slug", ""),
                    anchor_text=raw.get("anchor_text", ""),
                )
            )
    return rows


def parse_one_product(url_row: ProductUrlRow) -> ProductRow:
    session = make_session()
    html_text, final_url, error = fetch_html(session, url_row.product_url)
    if error or not html_text:
        return ProductRow(
            url=url_row.product_url,
            final_url=final_url,
            source_page_url=url_row.page_url,
            source_page_number=url_row.page_number,
            parse_status="error",
            error=error,
        )
    row = parse_product_html(html_text, url_row.product_url, final_url)
    row.source_page_url = url_row.page_url
    row.source_page_number = url_row.page_number
    return row


def parse_products(
    url_rows: list[ProductUrlRow],
    delay: float,
    out_dir: Path,
    keep_no_model: bool = False,
    workers: int = 1,
) -> list[ProductRow]:
    products: list[ProductRow] = []
    skipped: list[ProductRow] = []

    def handle_row(row: ProductRow) -> None:
        if row.parse_status == "error":
            skipped.append(row)
            eprint(f"[WARN] product failed {row.url}: {row.error}")
            return
        if row.model_links:
            products.append(row)
            eprint(f"[OK] 3D model: {row.name} -> {row.model_links[0]['file_text']}")
        else:
            skipped.append(row)
            eprint(f"[SKIP] no 3D model: {row.name or row.url}")

    if workers <= 1:
        for index, url_row in enumerate(url_rows, start=1):
            if index > 1:
                time.sleep(delay)
            handle_row(parse_one_product(url_row))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_url = {executor.submit(parse_one_product, row): row.product_url for row in url_rows}
            for future in as_completed(future_to_url):
                try:
                    handle_row(future.result())
                except Exception as exc:  # noqa: BLE001 - keep batch running on one bad product.
                    url = future_to_url[future]
                    skipped.append(ProductRow(url=url, parse_status="error", error=f"{type(exc).__name__}: {exc}"))
                    eprint(f"[WARN] product failed {url}: {type(exc).__name__}: {exc}")

    write_jsonl(out_dir / "products_with_3d_models.jsonl", products)
    write_product_csv(out_dir / "products_with_3d_models.csv", products)
    if keep_no_model:
        write_jsonl(out_dir / "products_without_3d_models.jsonl", skipped)
    return products


def cmd_collect_urls(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    ensure_dir(out_dir)
    rows = collect_urls(args.category_url, args.max_pages, args.delay, out_dir)
    eprint(f"[DONE] collected {len(rows)} product URLs -> {out_dir / 'product_urls.csv'}")


def cmd_parse_products(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    ensure_dir(out_dir)
    url_rows = load_urls_csv(Path(args.urls_csv))
    rows = parse_products(url_rows, args.delay, out_dir, keep_no_model=args.keep_no_model, workers=args.workers)
    eprint(f"[DONE] saved {len(rows)} products with 3D models -> {out_dir / 'products_with_3d_models.jsonl'}")


def cmd_all(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    ensure_dir(out_dir)
    url_rows = collect_urls(args.category_url, args.max_pages, args.delay, out_dir)
    rows = parse_products(url_rows, args.delay, out_dir, keep_no_model=args.keep_no_model, workers=args.workers)
    eprint(f"[DONE] saved {len(rows)} products with 3D models -> {out_dir / 'products_with_3d_models.jsonl'}")


def cmd_parse_saved_html(args: argparse.Namespace) -> None:
    html_text = Path(args.html).read_text(encoding="utf-8", errors="ignore")
    row = parse_product_html(html_text, args.url, args.url)
    print(json.dumps(asdict(row), ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", default=DEFAULT_OUT_DIR, help="Output directory.")
    common.add_argument("--delay", type=float, default=0.7, help="Delay between HTTP requests, seconds.")

    collect = subparsers.add_parser("collect-urls", parents=[common], help="Collect product URLs from category pages.")
    collect.add_argument("--category-url", action="append", default=[], help="Category URL. Can be passed multiple times.")
    collect.add_argument("--max-pages", type=int, default=0, help="Maximum pages per category. 0 = use detected pagination.")
    collect.set_defaults(func=cmd_collect_urls)

    parse = subparsers.add_parser("parse-products", parents=[common], help="Parse products from product_urls.csv.")
    parse.add_argument("--urls-csv", required=True, help="CSV created by collect-urls.")
    parse.add_argument("--keep-no-model", action="store_true", help="Also write products_without_3d_models.jsonl.")
    parse.add_argument("--workers", type=int, default=1, help="Parallel product requests. Use 1 for polite sequential scraping.")
    parse.set_defaults(func=cmd_parse_products)

    all_cmd = subparsers.add_parser("all", parents=[common], help="Collect URLs and parse products.")
    all_cmd.add_argument("--category-url", action="append", default=[], help="Category URL. Can be passed multiple times.")
    all_cmd.add_argument("--max-pages", type=int, default=0, help="Maximum pages per category. 0 = use detected pagination.")
    all_cmd.add_argument("--keep-no-model", action="store_true", help="Also write products_without_3d_models.jsonl.")
    all_cmd.add_argument("--workers", type=int, default=1, help="Parallel product requests. Use 1 for polite sequential scraping.")
    all_cmd.set_defaults(func=cmd_all)

    saved = subparsers.add_parser("parse-saved-html", help="Parse a saved product HTML file and print JSON.")
    saved.add_argument("--html", required=True, help="Saved product HTML file.")
    saved.add_argument("--url", required=True, help="Source product URL.")
    saved.set_defaults(func=cmd_parse_saved_html)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if hasattr(args, "category_url") and not args.category_url:
        args.category_url = DEFAULT_CATEGORY_URLS
    args.func(args)


if __name__ == "__main__":
    main()
