#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parser for Mosplitka ceramic tile and porcelain tile products.

The output intentionally follows the floor-material CSV shape used by
domlenta_floor_coverings_scraper.py, with extra recommendation columns:

    python3 -m src.tools.mosplitka_tile_scraper all \
      --category-url https://mosplitka.ru/catalog/plitka-dlia-kuxni/ \
      --out data/floor_materials/mosplitka \
      --max-pages 5 \
      --headed

Saved product HTML:

    python3 -m src.tools.mosplitka_tile_scraper parse-saved-html \
      --html product.html \
      --url https://mosplitka.ru/product/keramogranit-rochi-fog-60x60/ \
      --out data/floor_materials/mosplitka

Saved listing HTML:

    python3 -m src.tools.mosplitka_tile_scraper parse-listing-html \
      --html listing.html \
      --out data/floor_materials/mosplitka
"""

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
from typing import Any, Iterable
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://mosplitka.ru"
DEFAULT_CATEGORY_URLS = ["https://mosplitka.ru/catalog/plitka-dlia-kuxni/"]
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/121.0.0.0 Safari/537.36"
)

ROOM_ALIASES = {
    "для гостиной": "living_room",
    "гостиная": "living_room",
    "для коридора": "hallway",
    "коридор": "hallway",
    "для кухни": "kitchen",
    "кухня": "kitchen",
    "для ванной": "bathroom",
    "ванная": "bathroom",
    "для туалета": "bathroom",
    "туалет": "bathroom",
    "для балкона": "balcony",
    "балкон": "balcony",
    "для гаража": "garage",
    "гараж": "garage",
    "для офиса": "office",
    "офис": "office",
    "для спальни": "bedroom",
    "спальня": "bedroom",
}

STYLE_ALIASES = {
    "современный": "contemporary",
    "классический": "classic",
    "классика": "classic",
    "лофт": "loft",
    "минимализм": "minimalism",
    "скандинавский": "scandinavian",
    "прованс": "provence",
    "арт-деко": "art_deco",
}


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
    badges: list[str] = field(default_factory=list)
    variants: list[dict[str, str]] = field(default_factory=list)
    room_recommendations: list[dict[str, Any]] = field(default_factory=list)
    style_recommendations: list[dict[str, Any]] = field(default_factory=list)
    usage_recommendations: list[dict[str, Any]] = field(default_factory=list)
    recommendations_text: str = ""
    parse_status: str = "ok"
    error: str = ""


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def norm_text(value: Any) -> str:
    text = str(value or "").replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def lower(value: Any) -> str:
    return norm_text(value).lower().replace("ё", "е")


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
    if path and not path.endswith("/") and "/product/" in path:
        path += "/"
    query = parsed.query if keep_query else ""
    return urlunparse((parsed.scheme, parsed.netloc, path, "", query, ""))


def product_slug_from_url(url: str) -> str:
    return urlparse(url).path.rstrip("/").split("/")[-1]


def category_page_url(category_url: str, page_number: int) -> str:
    category_url = normalize_url(category_url, keep_query=True)
    parsed = urlparse(category_url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    if query.get("page"):
        if page_number <= 1:
            return category_url
        query["page"] = str(page_number)
        return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", urlencode(query), ""))
    if page_number <= 1:
        return category_url
    path = re.sub(r"/page/\d+/?$", "", parsed.path.rstrip("/"))
    return urlunparse((parsed.scheme, parsed.netloc, f"{path}/page/{page_number}/", "", "", ""))


def query_page_number(url: str) -> int | None:
    parsed = urlparse(absolutize(url))
    value = dict(parse_qsl(parsed.query, keep_blank_values=True)).get("page")
    if value and value.isdigit():
        return int(value)
    match = re.search(r"/page/(\d+)/?$", parsed.path)
    if match:
        return int(match.group(1))
    return None


def extract_next_listing_url(html_text: str, page_url: str, current_page_number: int) -> str:
    soup = BeautifulSoup(html_text, "html.parser")
    candidates: list[tuple[int, str, str]] = []
    for a in soup.find_all("a", href=True):
        href = normalize_url(a.get("href", ""), keep_query=True)
        parsed = urlparse(href)
        if parsed.netloc != urlparse(BASE_URL).netloc or "/catalog/" not in parsed.path:
            continue
        page_no = query_page_number(href)
        text = lower(a.get_text(" ", strip=True))
        if page_no is None and "дальше" not in text:
            continue
        priority = 0 if page_no == current_page_number + 1 else 1
        if "дальше" in text:
            priority = -1
        candidates.append((priority, page_no or current_page_number + 1, href))
    for _priority, page_no, href in sorted(candidates):
        if page_no > current_page_number:
            return href
    return ""


def get_playwright():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Playwright is required for live Mosplitka catalog pages. Install with:\n"
            "  python3 -m pip install playwright\n"
            "  python3 -m playwright install chromium"
        ) from exc
    return sync_playwright


def new_browser_context(p: Any, headed: bool) -> tuple[Any, Any]:
    browser = p.chromium.launch(headless=not headed)
    context = browser.new_context(
        user_agent=DEFAULT_USER_AGENT,
        viewport={"width": 1440, "height": 1400},
        locale="ru-RU",
    )
    return browser, context


def scroll_page(page: Any, scrolls: int, pause: float) -> None:
    last_height = 0
    stable_rounds = 0
    for _ in range(max(0, scrolls)):
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        page.wait_for_timeout(int(max(0.1, pause) * 1000))
        new_height = page.evaluate("document.body.scrollHeight")
        if new_height == last_height:
            stable_rounds += 1
        else:
            stable_rounds = 0
            last_height = new_height
        if stable_rounds >= 4:
            break


def html_text_lines(soup: BeautifulSoup) -> list[str]:
    return [norm_text(x) for x in soup.get_text("\n", strip=True).splitlines() if norm_text(x)]


def extract_category_title(html_text: str) -> str:
    soup = BeautifulSoup(html_text, "html.parser")
    node = soup.find("h1") or soup.find("title")
    return norm_text(node.get_text(" ", strip=True) if node else "")


def extract_product_links_from_html(
    html_text: str,
    page_url: str,
    category_url: str = "",
    category_title: str = "",
    page_number: int = 1,
) -> list[ProductUrlRow]:
    soup = BeautifulSoup(html_text, "html.parser")
    seen: set[str] = set()
    rows: list[ProductUrlRow] = []

    def add(raw_url: str, anchor_text: str = "") -> None:
        product_url = normalize_url(raw_url)
        if "/product/" not in urlparse(product_url).path or product_url in seen:
            return
        seen.add(product_url)
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

    for a in soup.find_all("a", href=True):
        add(a.get("href", ""), norm_text(a.get("title") or a.get_text(" ", strip=True)))

    for match in re.finditer(r'https?://mosplitka\.ru/product/[^"\'<>\s]+|/product/[^"\'<>\s]+', html_text):
        add(match.group(0))

    return rows


def dedupe_url_rows(rows: Iterable[ProductUrlRow]) -> list[ProductUrlRow]:
    seen: set[str] = set()
    out: list[ProductUrlRow] = []
    for row in rows:
        if row.product_url in seen:
            continue
        seen.add(row.product_url)
        out.append(row)
    return out


def write_product_urls(rows: list[ProductUrlRow], out_dir: Path) -> None:
    ensure_dir(out_dir)
    fieldnames = list(asdict(ProductUrlRow("", "", 0, "", "", "")).keys())
    with (out_dir / "product_urls.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    with (out_dir / "product_urls.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
    with (out_dir / "product_urls.txt").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(row.product_url + "\n")


def collect_urls_with_browser(
    category_urls: list[str],
    out_dir: Path,
    max_pages: int,
    headed: bool,
    scrolls: int,
    scroll_pause: float,
    empty_stop: int,
    save_html: bool,
) -> list[ProductUrlRow]:
    ensure_dir(out_dir)
    html_dir = out_dir / "listing_html"
    if save_html:
        ensure_dir(html_dir)
    all_rows: list[ProductUrlRow] = []
    sync_playwright = get_playwright()
    with sync_playwright() as p:
        browser, context = new_browser_context(p, headed=headed)
        page = context.new_page()
        for category_url in category_urls:
            category_title = ""
            empty_streak = 0
            page_url = category_page_url(category_url, 1)
            for page_number in range(1, max_pages + 1):
                listing_page_number = query_page_number(page_url) or page_number
                eprint(f"[INFO] {page_url}")
                try:
                    page.goto(page_url, wait_until="domcontentloaded", timeout=90_000)
                    page.wait_for_timeout(2500)
                    scroll_page(page, scrolls=scrolls, pause=scroll_pause)
                    html_text = page.content()
                except Exception as exc:
                    eprint(f"[WARN] Cannot open {page_url}: {exc}")
                    empty_streak += 1
                    if empty_streak >= empty_stop:
                        break
                    continue
                if page_number == 1:
                    category_title = extract_category_title(html_text)
                if save_html:
                    safe = clean_filename(urlparse(page_url).path.strip("/").replace("/", "_"))
                    (html_dir / f"{safe or 'category'}_p{listing_page_number:03d}.html").write_text(html_text, encoding="utf-8")
                before = len(dedupe_url_rows(all_rows))
                all_rows.extend(extract_product_links_from_html(html_text, page_url, category_url, category_title, listing_page_number))
                added = len(dedupe_url_rows(all_rows)) - before
                eprint(f"[INFO] new product urls: {added}")
                empty_streak = empty_streak + 1 if added == 0 else 0
                if empty_streak >= empty_stop:
                    break
                next_page_url = extract_next_listing_url(html_text, page_url, listing_page_number)
                if not next_page_url:
                    next_page_url = category_page_url(category_url, page_number + 1)
                if normalize_url(next_page_url, keep_query=True) == normalize_url(page_url, keep_query=True):
                    break
                page_url = next_page_url
        context.close()
        browser.close()
    rows = dedupe_url_rows(all_rows)
    write_product_urls(rows, out_dir)
    return rows


def collect_urls_from_listing_html(html_paths: list[str], out_dir: Path) -> list[ProductUrlRow]:
    rows: list[ProductUrlRow] = []
    for idx, raw_path in enumerate(html_paths, start=1):
        path = Path(raw_path)
        html_text = path.read_text(encoding="utf-8", errors="ignore")
        rows.extend(
            extract_product_links_from_html(
                html_text,
                page_url=BASE_URL,
                category_title=extract_category_title(html_text),
                page_number=idx,
            )
        )
    rows = dedupe_url_rows(rows)
    write_product_urls(rows, out_dir)
    return rows


def safe_json_loads(text: str) -> Any | None:
    try:
        return json.loads(text)
    except Exception:
        return None


def iter_json_objects(data: Any) -> Iterable[dict[str, Any]]:
    if isinstance(data, dict):
        yield data
        for value in data.values():
            if isinstance(value, (dict, list)):
                yield from iter_json_objects(value)
    elif isinstance(data, list):
        for value in data:
            if isinstance(value, (dict, list)):
                yield from iter_json_objects(value)


def is_product_jsonld(obj: dict[str, Any]) -> bool:
    kind = obj.get("@type")
    if isinstance(kind, str):
        return kind.lower() == "product"
    if isinstance(kind, list):
        return any(isinstance(x, str) and x.lower() == "product" for x in kind)
    return False


def extract_jsonld_product(soup: BeautifulSoup) -> dict[str, Any] | None:
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = script.string or script.get_text("", strip=False)
        data = safe_json_loads(raw.strip())
        if data is None:
            continue
        for obj in iter_json_objects(data):
            if is_product_jsonld(obj):
                return obj
    return None


def extract_breadcrumbs(soup: BeautifulSoup) -> list[str]:
    result: list[str] = []
    for node in soup.select('[itemtype*="BreadcrumbList"] [itemprop="name"], nav a, .breadcrumbs a, .breadcrumbs span'):
        text = norm_text(node.get("content") or node.get_text(" ", strip=True))
        if text and text not in result:
            result.append(text)
    return result


def normalize_key(key: str) -> str:
    key = norm_text(key).strip(":")
    key = re.sub(r"\s+", " ", key)
    return key


def extract_properties_from_jsonld(product: dict[str, Any]) -> dict[str, str]:
    props: dict[str, str] = {}
    raw = product.get("additionalProperty")
    if isinstance(raw, dict):
        raw = [raw]
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            name = normalize_key(str(item.get("name", "")))
            value = item.get("value", "")
            if isinstance(value, (dict, list)):
                value_text = json.dumps(value, ensure_ascii=False)
            else:
                value_text = norm_text(value)
            if name and value_text:
                props[name] = value_text
    return props


def extract_properties_from_lines(lines: list[str]) -> dict[str, str]:
    props: dict[str, str] = {}
    section = ""
    stop = {
        "отзывы",
        "другие коллекции",
        "похожие товары",
        "с этим товаром покупают",
        "гарантия и сертификаты",
    }
    known_sections = {"общие", "упаковка", "характеристики"}
    noise = {"все характеристики", "читать далее ↓", "читать далее", "скрыть"}
    try:
        i = next(idx for idx, line in enumerate(lines) if lower(line) == "характеристики")
    except StopIteration:
        i = 0
    while i < len(lines):
        line = lines[i]
        low = lower(line)
        if low in stop:
            break
        if low in noise or line == ":":
            i += 1
            continue
        if low in known_sections or line.endswith(":") and low.rstrip(":") in known_sections:
            section = line.rstrip(":")
            i += 1
            continue
        value = ""
        if i + 2 < len(lines) and lines[i + 1] == ":":
            value = lines[i + 2]
            step = 3
        elif i + 1 < len(lines) and lines[i].endswith(":"):
            value = lines[i + 1]
            line = line.rstrip(":")
            step = 2
        elif i + 1 < len(lines) and lower(lines[i + 1]) not in stop and lines[i + 1] != ":":
            # Handles compact pairs in fallback text, but avoid obvious long prose.
            if len(line) <= 80 and len(lines[i + 1]) <= 160:
                value = lines[i + 1]
                step = 2
            else:
                i += 1
                continue
        else:
            i += 1
            continue
        key = normalize_key(line)
        if key and value and key not in props:
            props[key] = norm_text(value)
            if section:
                props.setdefault(f"{section} > {key}", norm_text(value))
        i += step
    return props


def extract_description(lines: list[str]) -> str:
    try:
        start = next(i for i, line in enumerate(lines) if lower(line) == "о товаре") + 1
    except StopIteration:
        return ""
    stop_words = {"характеристики", "читать далее ↓", "читать далее"}
    parts: list[str] = []
    for line in lines[start:]:
        if lower(line) in stop_words:
            break
        if len(line) > 1:
            parts.append(line)
    return norm_text(" ".join(parts))


def extract_name(soup: BeautifulSoup, product: dict[str, Any] | None, lines: list[str]) -> str:
    if product and product.get("name"):
        return norm_text(product.get("name"))
    h1 = soup.find("h1")
    if h1:
        text = norm_text(h1.get_text(" ", strip=True))
        if text:
            return text
    for meta_name in ["og:title", "twitter:title"]:
        meta = soup.find("meta", attrs={"property": meta_name}) or soup.find("meta", attrs={"name": meta_name})
        if meta and meta.get("content"):
            return norm_text(meta.get("content"))
    title = soup.find("title")
    if title:
        text = re.sub(r"\s*\|\s*.*$", "", norm_text(title.get_text(" ", strip=True)))
        if text:
            return text
    for line in lines[:30]:
        if any(x in lower(line) for x in ["керамогранит", "плитка", "мозаика"]):
            return line
    return ""


def extract_price(lines: list[str], soup: BeautifulSoup, product: dict[str, Any] | None) -> str:
    if product:
        offers = product.get("offers")
        if isinstance(offers, dict) and offers.get("price"):
            return norm_text(offers.get("price"))
    for meta in soup.select('[itemprop="price"], meta[property="product:price:amount"]'):
        value = meta.get("content") or meta.get_text(" ", strip=True)
        if value:
            return norm_text(value)
    for line in lines:
        if "₽" in line and re.search(r"\d", line):
            return line
    return ""


def extract_availability(lines: list[str], product: dict[str, Any] | None) -> str:
    if product:
        offers = product.get("offers")
        if isinstance(offers, dict) and offers.get("availability"):
            return norm_text(offers.get("availability"))
    text = "\n".join(lines)
    if "Нет в наличии" in text:
        return "Нет в наличии"
    if "В наличии" in text:
        return "В наличии"
    return ""


def extract_badges(lines: list[str], soup: BeautifulSoup) -> list[str]:
    known = ["новинка", "только в мосплитка", "популярный", "цена огонь", "скидк"]
    badges: list[str] = []
    for node in soup.select("span, div"):
        text = norm_text(node.get_text(" ", strip=True))
        if 0 < len(text) <= 40 and any(k in lower(text) for k in known) and text not in badges:
            badges.append(text)
    for line in lines[:80]:
        if len(line) <= 40 and any(k in lower(line) for k in known) and line not in badges:
            badges.append(line)
    return badges[:20]


def normalize_image_url(url: str) -> str:
    url = absolutize(url)
    parsed = urlparse(url)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", parsed.query, ""))


def extract_images(soup: BeautifulSoup, html_text: str, product: dict[str, Any] | None) -> list[str]:
    images: list[str] = []
    seen: set[str] = set()

    def add(url: str) -> None:
        if not url:
            return
        full = normalize_image_url(url.strip("'\""))
        parsed = urlparse(full)
        if "cdn.mosplitka.ru" not in parsed.netloc:
            return
        if Path(parsed.path).suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            return
        if full not in seen:
            seen.add(full)
            images.append(full)

    if product:
        raw = product.get("image")
        if isinstance(raw, str):
            add(raw)
        elif isinstance(raw, list):
            for item in raw:
                if isinstance(item, str):
                    add(item)

    for img in soup.find_all("img"):
        for attr in ["src", "data-src", "data-original"]:
            add(img.get(attr, ""))
        for part in str(img.get("srcset") or "").split(","):
            add(part.strip().split(" ")[0])

    for match in re.finditer(r'https?://cdn\.mosplitka\.ru/[^"\'<>\s)]+', html_text):
        add(match.group(0))
    for match in re.finditer(r"background-image:\s*url\(([^)]+)\)", html_text):
        add(match.group(1))

    return images


def extract_variants(soup: BeautifulSoup) -> list[dict[str, str]]:
    variants: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for a in soup.find_all("a", href=True):
        href = normalize_url(a.get("href", ""))
        if "/product/" not in urlparse(href).path:
            continue
        text = norm_text(a.get_text(" ", strip=True))
        title = ""
        swatch = a.select_one("[title]")
        if swatch:
            title = norm_text(swatch.get("title"))
        image = ""
        node_with_style = a.select_one("[style*='background-image']")
        if node_with_style:
            match = re.search(r"url\(([^)]+)\)", node_with_style.get("style") or "")
            if match:
                image = normalize_image_url(match.group(1).strip("'\""))
        kind = "color" if title else ("format" if re.search(r"\d+\s*[хx]\s*\d+", text) else "link")
        value = title or text
        key = (href, kind, value)
        if value and key not in seen:
            seen.add(key)
            variants.append({"kind": kind, "value": value, "url": href, "image_url": image})
    return variants


def split_csvish(value: str) -> list[str]:
    parts = re.split(r"[,;/]+", value or "")
    return [norm_text(x) for x in parts if norm_text(x)]


def normalize_room(value: str) -> str:
    low = lower(value)
    return ROOM_ALIASES.get(low, ROOM_ALIASES.get(low.replace("для ", ""), low.replace(" ", "_")))


def build_recommendations(props: dict[str, str], description: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], str]:
    room_values = split_csvish(props.get("Помещение", ""))
    purpose_values = split_csvish(props.get("Назначение", ""))
    style_values = split_csvish(props.get("Стиль", ""))
    feature_values = split_csvish(props.get("Особенности", ""))
    surface = props.get("Поверхность", "")
    material_type = props.get("Тип", "") or props.get("Тип материала", "")
    drawing = props.get("Рисунок", "")
    thickness = props.get("Толщина, мм", "") or props.get("Толщина", "")

    room_recs: list[dict[str, Any]] = []
    for room in room_values:
        reasons: list[str] = []
        room_key = normalize_room(room)
        if room_key in {"bathroom", "kitchen", "balcony", "garage", "hallway"} and any("мороз" in lower(x) for x in feature_values):
            reasons.append("морозостойкость полезна для влажных, входных и неотапливаемых зон")
        if room_key in {"hallway", "kitchen", "office", "garage"} and any("ступ" in lower(x) for x in feature_values):
            reasons.append("допуск для ступеней и входных зон подходит для повышенного трафика")
        if lower(surface).startswith("мат"):
            reasons.append("матовая поверхность меньше бликует и практичнее на полу")
        if not reasons:
            reasons.append("помещение указано производителем или продавцом в характеристиках")
        room_recs.append({"room": room_key, "label": room, "confidence": "explicit", "reasons": reasons})

    style_recs: list[dict[str, Any]] = []
    for style in style_values:
        style_key = STYLE_ALIASES.get(lower(style), lower(style).replace(" ", "_"))
        reasons = []
        if drawing:
            reasons.append(f"рисунок: {drawing}")
        colors = props.get("Цвет точно", "") or props.get("Цвет", "")
        if colors:
            reasons.append(f"цвет: {colors}")
        if surface:
            reasons.append(f"поверхность: {surface}")
        style_recs.append({"style": style_key, "label": style, "confidence": "explicit", "reasons": reasons})

    usage_recs: list[dict[str, Any]] = []
    for purpose in purpose_values:
        usage_recs.append({"usage": lower(purpose).replace("для ", "").replace(" ", "_"), "label": purpose, "confidence": "explicit"})
    for feature in feature_values:
        usage_recs.append({"usage": lower(feature).replace(" ", "_"), "label": feature, "confidence": "explicit"})
    if material_type:
        usage_recs.append({"usage": lower(material_type).replace(" ", "_"), "label": material_type, "confidence": "explicit"})

    text_parts: list[str] = []
    if room_values:
        text_parts.append("Подходит для помещений: " + ", ".join(room_values) + ".")
    if purpose_values:
        text_parts.append("Назначение: " + ", ".join(purpose_values) + ".")
    if style_values:
        text_parts.append("Стили: " + ", ".join(style_values) + ".")
    if feature_values:
        text_parts.append("Особенности: " + ", ".join(feature_values) + ".")
    details = []
    if surface:
        details.append(f"поверхность {surface}")
    if drawing:
        details.append(f"рисунок {drawing}")
    if thickness:
        details.append(f"толщина {thickness} мм")
    if details:
        text_parts.append("Технический контекст: " + ", ".join(details) + ".")
    if description:
        desc_lower = lower(description)
        inferred = []
        if "визуально расшир" in desc_lower:
            inferred.append("крупный формат может визуально расширять небольшие помещения")
        if "сцеплен" in desc_lower or "ступен" in desc_lower:
            inferred.append("описание указывает на практичность для пола и входных зон")
        if inferred:
            text_parts.append("Из описания: " + "; ".join(inferred) + ".")
    return room_recs, style_recs, usage_recs, " ".join(text_parts)


def parse_product_html(html_text: str, url: str, final_url: str = "") -> ProductRow:
    soup = BeautifulSoup(html_text, "html.parser")
    lines = html_text_lines(soup)
    product = extract_jsonld_product(soup)
    row = ProductRow(url=url, final_url=final_url or url)
    try:
        row.name = extract_name(soup, product, lines)
        row.description = norm_text((product or {}).get("description") if product else "") or extract_description(lines)
        row.properties = extract_properties_from_jsonld(product or {})
        visible_props = extract_properties_from_lines(lines)
        for key, value in visible_props.items():
            row.properties.setdefault(key, value)
        row.sku = norm_text((product or {}).get("sku") if product else "") or row.properties.get("Артикул", "")
        row.brand = ""
        brand = (product or {}).get("brand") if product else None
        if isinstance(brand, dict):
            row.brand = norm_text(brand.get("name"))
        elif isinstance(brand, str):
            row.brand = norm_text(brand)
        row.brand = row.brand or row.properties.get("Бренд", "")
        row.price = extract_price(lines, soup, product)
        row.price_currency = "RUB"
        row.availability = extract_availability(lines, product)
        row.breadcrumbs = extract_breadcrumbs(soup)
        row.categories = [x for x in row.breadcrumbs if x not in {"Главная", row.name}]
        row.images = extract_images(soup, html_text, product)
        row.badges = extract_badges(lines, soup)
        row.variants = extract_variants(soup)
        (
            row.room_recommendations,
            row.style_recommendations,
            row.usage_recommendations,
            row.recommendations_text,
        ) = build_recommendations(row.properties, row.description)
        if not row.name:
            row.parse_status = "no_product_data"
            row.error = "Product name was not found"
    except Exception as exc:
        row.parse_status = "error"
        row.error = str(exc)
    return row


def make_requests_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
            "Referer": BASE_URL + "/",
        }
    )
    return session


def fetch_html_requests(session: requests.Session, url: str, timeout: tuple[int, int] = (20, 60)) -> tuple[str | None, str]:
    try:
        response = session.get(url, timeout=timeout)
        if response.status_code >= 400:
            return None, f"HTTP {response.status_code}"
        return response.text, ""
    except Exception as exc:
        return None, str(exc)


def parse_products_with_requests(urls: list[str], delay: float) -> tuple[list[ProductRow], list[str]]:
    session = make_requests_session()
    rows: list[ProductRow] = []
    failed: list[str] = []
    total = len(urls)
    for idx, url in enumerate(urls, start=1):
        eprint(f"[INFO] product {idx}/{total}: {url}")
        html_text, error = fetch_html_requests(session, url)
        if not html_text:
            rows.append(ProductRow(url=url, parse_status="fetch_failed", error=error))
            failed.append(url)
        else:
            row = parse_product_html(html_text, url=url, final_url=url)
            rows.append(row)
            if row.parse_status != "ok" or not row.name:
                failed.append(url)
        if delay > 0:
            time.sleep(delay)
    return rows, failed


def parse_products_with_browser(
    urls: list[str],
    headed: bool,
    scrolls: int,
    scroll_pause: float,
    save_html_dir: Path | None,
) -> list[ProductRow]:
    if save_html_dir:
        ensure_dir(save_html_dir)
    rows: list[ProductRow] = []
    sync_playwright = get_playwright()
    with sync_playwright() as p:
        browser, context = new_browser_context(p, headed=headed)
        page = context.new_page()
        total = len(urls)
        for idx, url in enumerate(urls, start=1):
            eprint(f"[INFO] browser product {idx}/{total}: {url}")
            try:
                saved_path = None
                if save_html_dir:
                    saved_path = save_html_dir / f"{clean_filename(product_slug_from_url(url)) or stable_hash(url)}.html"
                if saved_path and saved_path.exists():
                    html_text = saved_path.read_text(encoding="utf-8", errors="ignore")
                    rows.append(parse_product_html(html_text, url=url, final_url=url))
                    continue
                page.goto(url, wait_until="domcontentloaded", timeout=90_000)
                page.wait_for_timeout(2500)
                scroll_page(page, scrolls=scrolls, pause=scroll_pause)
                html_text = page.content()
                if saved_path:
                    saved_path.write_text(html_text, encoding="utf-8")
                rows.append(parse_product_html(html_text, url=url, final_url=page.url))
            except Exception as exc:
                rows.append(ProductRow(url=url, parse_status="fetch_failed", error=str(exc)))
        context.close()
        browser.close()
    return rows


def merge_rows(primary: list[ProductRow], fallback: list[ProductRow]) -> list[ProductRow]:
    fallback_by_url = {row.url: row for row in fallback}
    out: list[ProductRow] = []
    for row in primary:
        if row.parse_status == "ok" and row.name and row.properties:
            out.append(row)
        else:
            out.append(fallback_by_url.get(row.url, row))
    seen = {row.url for row in out}
    out.extend(row for row in fallback if row.url not in seen)
    return out


def read_urls(path: Path) -> list[str]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            return sorted({normalize_url(row.get("product_url") or row.get("url") or "") for row in reader if row.get("product_url") or row.get("url")})
    return sorted({normalize_url(line.strip()) for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip() and not line.startswith("#")})


def row_to_csv_dict(row: ProductRow) -> dict[str, Any]:
    data = asdict(row)
    data["breadcrumbs"] = json.dumps(row.breadcrumbs, ensure_ascii=False)
    data["categories"] = json.dumps(row.categories, ensure_ascii=False)
    data["properties_json"] = json.dumps(row.properties, ensure_ascii=False)
    data["images_json"] = json.dumps(row.images, ensure_ascii=False)
    data["local_image_paths_json"] = json.dumps(row.local_image_paths, ensure_ascii=False)
    data["badges_json"] = json.dumps(row.badges, ensure_ascii=False)
    data["variants_json"] = json.dumps(row.variants, ensure_ascii=False)
    data["room_recommendations_json"] = json.dumps(row.room_recommendations, ensure_ascii=False)
    data["style_recommendations_json"] = json.dumps(row.style_recommendations, ensure_ascii=False)
    data["usage_recommendations_json"] = json.dumps(row.usage_recommendations, ensure_ascii=False)
    for key in [
        "properties",
        "images",
        "local_image_paths",
        "badges",
        "variants",
        "room_recommendations",
        "style_recommendations",
        "usage_recommendations",
    ]:
        data.pop(key, None)
    return data


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
        "badges_json",
        "variants_json",
        "room_recommendations_json",
        "style_recommendations_json",
        "usage_recommendations_json",
        "recommendations_text",
        "parse_status",
        "error",
    ]
    with (out_dir / "products.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row_to_csv_dict(row))
    with (out_dir / "products.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row_to_csv_dict(row), ensure_ascii=False) + "\n")


def guess_extension_from_response(url: str, content_type: str | None = None) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
        return ".jpg" if suffix == ".jpeg" else suffix
    if content_type:
        ext = mimetypes.guess_extension(content_type.split(";")[0].strip().lower())
        if ext:
            return ".jpg" if ext == ".jpe" else ext
    return ".jpg"


def download_product_images(rows: list[ProductRow], out_dir: Path, retries: int, delay: float) -> list[ProductRow]:
    images_root = out_dir / "images"
    ensure_dir(images_root)
    session = requests.Session()
    session.headers.update({"User-Agent": DEFAULT_USER_AGENT, "Accept": "image/avif,image/webp,image/*,*/*;q=0.8"})
    for row in rows:
        product_dir = images_root / clean_filename(f"{row.sku or stable_hash(row.url)}_{row.name}", max_len=120)
        ensure_dir(product_dir)
        local_paths: list[str] = []
        for idx, image_url in enumerate(row.images, start=1):
            target_base = product_dir / f"{idx:02d}_{stable_hash(image_url)}"
            existing = [p for p in product_dir.glob(target_base.name + ".*") if not p.name.endswith(".part")]
            if existing:
                local_paths.append(str(existing[0].relative_to(out_dir)))
                continue
            saved: Path | None = None
            for attempt in range(1, retries + 1):
                try:
                    with session.get(image_url, headers={"Referer": row.final_url or row.url}, stream=True, timeout=(20, 90)) as response:
                        response.raise_for_status()
                        target = target_base.with_suffix(guess_extension_from_response(image_url, response.headers.get("Content-Type")))
                        tmp = target.with_suffix(target.suffix + ".part")
                        with tmp.open("wb") as f:
                            for chunk in response.iter_content(chunk_size=1024 * 128):
                                if chunk:
                                    f.write(chunk)
                        tmp.replace(target)
                        saved = target
                        break
                except Exception as exc:
                    if attempt == retries:
                        eprint(f"[WARN] image failed: {image_url} | {exc}")
                    time.sleep(0.8 * attempt)
            if saved:
                local_paths.append(str(saved.relative_to(out_dir)))
            if delay > 0:
                time.sleep(delay)
        row.local_image_paths = local_paths
    return rows


def parse_saved_html(html_paths: list[str], urls: list[str], out_dir: Path, download_images: bool, retries: int, delay: float) -> list[ProductRow]:
    rows: list[ProductRow] = []
    for idx, raw_path in enumerate(html_paths):
        path = Path(raw_path)
        url = urls[idx] if idx < len(urls) and urls[idx] else f"{BASE_URL}/product/{path.stem}/"
        row = parse_product_html(path.read_text(encoding="utf-8", errors="ignore"), url=url, final_url=url)
        rows.append(row)
    if download_images:
        rows = download_product_images(rows, out_dir, retries=retries, delay=delay)
    write_products(rows, out_dir)
    return rows


def parse_products_command(
    urls: list[str],
    out_dir: Path,
    headed: bool,
    use_browser_fallback: bool,
    save_html: bool,
    download_images: bool,
    retries: int,
    delay: float,
    scrolls: int,
    scroll_pause: float,
) -> list[ProductRow]:
    rows, failed = parse_products_with_requests(urls, delay=delay)
    if failed and use_browser_fallback:
        browser_rows = parse_products_with_browser(
            failed,
            headed=headed,
            scrolls=scrolls,
            scroll_pause=scroll_pause,
            save_html_dir=(out_dir / "product_html") if save_html else None,
        )
        rows = merge_rows(rows, browser_rows)
    if download_images:
        rows = download_product_images(rows, out_dir, retries=retries, delay=delay)
    write_products(rows, out_dir)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parse Mosplitka tile catalog/product pages.")
    sub = parser.add_subparsers(dest="command", required=True)

    collect = sub.add_parser("collect-urls")
    collect.add_argument("--category-url", action="append", dest="category_urls", default=[])
    collect.add_argument("--out", required=True)
    collect.add_argument("--max-pages", type=int, default=3)
    collect.add_argument("--headed", action="store_true")
    collect.add_argument("--scrolls", type=int, default=8)
    collect.add_argument("--scroll-pause", type=float, default=0.7)
    collect.add_argument("--empty-stop", type=int, default=2)
    collect.add_argument("--save-html", action="store_true")
    collect.add_argument("--listing-html", action="append", default=[])

    parse = sub.add_parser("parse-products")
    parse.add_argument("--urls-csv")
    parse.add_argument("--url", action="append", dest="urls", default=[])
    parse.add_argument("--out", required=True)
    parse.add_argument("--headed", action="store_true")
    parse.add_argument("--no-browser-fallback", action="store_true")
    parse.add_argument("--save-html", action="store_true")
    parse.add_argument("--download-images", action="store_true")
    parse.add_argument("--retries", type=int, default=3)
    parse.add_argument("--delay", type=float, default=0.2)
    parse.add_argument("--scrolls", type=int, default=6)
    parse.add_argument("--scroll-pause", type=float, default=0.6)

    saved = sub.add_parser("parse-saved-html")
    saved.add_argument("--html", action="append", required=True, dest="html_paths")
    saved.add_argument("--url", action="append", dest="urls", default=[])
    saved.add_argument("--out", required=True)
    saved.add_argument("--download-images", action="store_true")
    saved.add_argument("--retries", type=int, default=3)
    saved.add_argument("--delay", type=float, default=0.2)

    listing = sub.add_parser("parse-listing-html")
    listing.add_argument("--html", action="append", required=True, dest="html_paths")
    listing.add_argument("--out", required=True)

    all_cmd = sub.add_parser("all")
    all_cmd.add_argument("--category-url", action="append", dest="category_urls", default=[])
    all_cmd.add_argument("--out", required=True)
    all_cmd.add_argument("--max-pages", type=int, default=3)
    all_cmd.add_argument("--headed", action="store_true")
    all_cmd.add_argument("--scrolls", type=int, default=8)
    all_cmd.add_argument("--scroll-pause", type=float, default=0.7)
    all_cmd.add_argument("--empty-stop", type=int, default=2)
    all_cmd.add_argument("--save-html", action="store_true")
    all_cmd.add_argument("--download-images", action="store_true")
    all_cmd.add_argument("--retries", type=int, default=3)
    all_cmd.add_argument("--delay", type=float, default=0.2)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    out_dir = Path(args.out)

    if args.command == "collect-urls":
        if args.listing_html:
            rows = collect_urls_from_listing_html(args.listing_html, out_dir)
        else:
            rows = collect_urls_with_browser(
                args.category_urls or DEFAULT_CATEGORY_URLS,
                out_dir,
                max_pages=args.max_pages,
                headed=args.headed,
                scrolls=args.scrolls,
                scroll_pause=args.scroll_pause,
                empty_stop=args.empty_stop,
                save_html=args.save_html,
            )
        print(f"Product URLs: {len(rows)}")
        print(f"Saved: {out_dir / 'product_urls.csv'}")
        return 0

    if args.command == "parse-listing-html":
        rows = collect_urls_from_listing_html(args.html_paths, out_dir)
        print(f"Product URLs: {len(rows)}")
        print(f"Saved: {out_dir / 'product_urls.csv'}")
        return 0

    if args.command == "parse-saved-html":
        rows = parse_saved_html(args.html_paths, args.urls, out_dir, args.download_images, args.retries, args.delay)
        print(f"Products: {len(rows)}")
        print(f"Saved: {out_dir / 'products.csv'}")
        return 0

    if args.command == "parse-products":
        urls = list(args.urls)
        if args.urls_csv:
            urls.extend(read_urls(Path(args.urls_csv)))
        urls = sorted({normalize_url(url) for url in urls if url})
        rows = parse_products_command(
            urls,
            out_dir,
            headed=args.headed,
            use_browser_fallback=not args.no_browser_fallback,
            save_html=args.save_html,
            download_images=args.download_images,
            retries=args.retries,
            delay=args.delay,
            scrolls=args.scrolls,
            scroll_pause=args.scroll_pause,
        )
        print(f"Products: {len(rows)}")
        print(f"Saved: {out_dir / 'products.csv'}")
        return 0

    if args.command == "all":
        url_rows = collect_urls_with_browser(
            args.category_urls or DEFAULT_CATEGORY_URLS,
            out_dir,
            max_pages=args.max_pages,
            headed=args.headed,
            scrolls=args.scrolls,
            scroll_pause=args.scroll_pause,
            empty_stop=args.empty_stop,
            save_html=args.save_html,
        )
        rows = parse_products_command(
            [row.product_url for row in url_rows],
            out_dir,
            headed=args.headed,
            use_browser_fallback=True,
            save_html=args.save_html,
            download_images=args.download_images,
            retries=args.retries,
            delay=args.delay,
            scrolls=args.scrolls,
            scroll_pause=args.scroll_pause,
        )
        print(f"Product URLs: {len(url_rows)}")
        print(f"Products: {len(rows)}")
        print(f"Saved: {out_dir / 'products.csv'}")
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
