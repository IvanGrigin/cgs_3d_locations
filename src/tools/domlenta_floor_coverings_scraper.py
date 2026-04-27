#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
domlenta_floor_coverings_scraper.py

Задача:
    1) Собрать URL всех товаров из категории Domlenta "Напольные покрытия"
       и её страниц пагинации.
    2) Спарсить каждую карточку товара:
       - название;
       - артикул / sku;
       - бренд;
       - цену;
       - наличие;
       - описание;
       - характеристики;
       - хлебные крошки / категории;
       - все изображения из галереи.
    3) Сохранить CSV/JSONL.
    4) При необходимости скачать изображения.

Базовый запуск:
    python3 domlenta_floor_coverings_scraper.py all \
      -o domlenta_floor_coverings \
      --max-pages 30 \
      --download-images

Если сайт показывает защиту/cookies, запусти с видимым браузером:
    python3 domlenta_floor_coverings_scraper.py all \
      -o domlenta_floor_coverings \
      --max-pages 30 \
      --download-images \
      --headed

Только собрать URL товаров:
    python3 domlenta_floor_coverings_scraper.py collect-urls \
      -o domlenta_floor_coverings \
      --max-pages 30 \
      --headed

Только распарсить уже собранные URL:
    python3 domlenta_floor_coverings_scraper.py parse-products \
      -o domlenta_floor_coverings \
      --urls-csv domlenta_floor_coverings/product_urls.csv \
      --download-images \
      --headed

Парсинг сохранённой HTML-страницы категории:
    python3 domlenta_floor_coverings_scraper.py collect-urls \
      -o domlenta_floor_coverings \
      --listing-html page1.html page2.html

Зависимости:
    pip install beautifulsoup4 requests tqdm playwright
    python3 -m playwright install chromium

Важно:
    Скрипт не обходит авторизацию, платный доступ, капчу, антибот-защиту и закрытые API.
    Он работает с публичными HTML-страницами и JSON-LD, которые уже отдаются сайтом.
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
from typing import Any, Iterable, Optional
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


BASE_URL = "https://domlenta.ru"
OBI_BASE_URL = "https://obi.ru"
DEFAULT_CATEGORY_URLS = [
    "https://domlenta.ru/catalog/napolnye-pokrytiya-220003203/",
]
DEFAULT_OBI_CATEGORY_URLS = [
    # Core floor-covering categories, without plinths/accessories/glues.
    "https://obi.ru/napolnye-pokrytija/otdelochnye-materialy",
    "https://obi.ru/napolnye-pokrytija/otdelochnye-materialy/laminat",
    "https://obi.ru/napolnye-pokrytija/otdelochnye-materialy/linoleum",
    "https://obi.ru/napolnye-pokrytija/otdelochnye-materialy/pvh-i-kvarcvinilovaja-plitka",
    "https://obi.ru/napolnye-pokrytija/otdelochnye-materialy/kovrolin",
    "https://obi.ru/napolnye-pokrytija/otdelochnye-materialy/kovrovye-dorozhki",
    "https://obi.ru/napolnye-pokrytija/otdelochnye-materialy/parketnaja-doska",
    "https://obi.ru/napolnye-pokrytija/otdelochnye-materialy/doska-pola",
    "https://obi.ru/napolnye-pokrytija/otdelochnye-materialy/iskusstvennyj-gazon",
]
DEFAULT_CATEGORY_URLS_BY_SITE = {
    "domlenta": DEFAULT_CATEGORY_URLS,
    "obi": DEFAULT_OBI_CATEGORY_URLS,
}

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
    brand: str = ""
    price: str = ""
    price_currency: str = ""
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


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def norm_text(value: str | None) -> str:
    if not value:
        return ""
    value = re.sub(r"\s+", " ", value.replace("\xa0", " "))
    return value.strip()


def stable_hash(value: str, n: int = 12) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()[:n]


def clean_filename(value: str, max_len: int = 120) -> str:
    value = norm_text(value).lower()
    value = re.sub(r"https?://", "", value)
    value = re.sub(r"[^\wа-яА-ЯёЁ\-\. ]+", "_", value, flags=re.UNICODE)
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"_+", "_", value)
    value = value.strip("._- ")
    if not value:
        value = "item"
    return value[:max_len].strip("._- ")


def absolutize(url: str, base_url: str = BASE_URL) -> str:
    if not url:
        return ""
    return urljoin(base_url, url)


def site_base_url(site: str) -> str:
    return OBI_BASE_URL if site == "obi" else BASE_URL


def infer_site_from_url(url: str) -> str:
    host = urlparse(url).netloc.lower()
    if "obi.ru" in host:
        return "obi"
    return "domlenta"


def normalize_url(url: str, keep_query: bool = False) -> str:
    parsed = urlparse(url)
    path = parsed.path
    if path and not path.endswith("/") and "/product/" in path:
        path += "/"
    query = parsed.query if keep_query else ""
    return urlunparse((parsed.scheme, parsed.netloc, path, "", query, ""))


def product_slug_from_url(url: str) -> str:
    path = urlparse(url).path.rstrip("/")
    return path.split("/")[-1] if path else ""


def category_page_url(category_url: str, page_number: int, site: str = "domlenta") -> str:
    """
    У Domlenta в индексе встречаются варианты:
        /catalog/napolnye-pokrytiya-220003203/
        /catalog/napolnye-pokrytiya-220003203/page/2/
    Иногда поисковики показывают ?page=13, но path-pagination обычно читается сайтом нормально.
    """
    category_url = normalize_url(category_url, keep_query=False)
    if page_number <= 1:
        return category_url

    parsed = urlparse(category_url)
    if site == "obi":
        return urlunparse((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", f"page={page_number}", ""))

    path = parsed.path.rstrip("/")
    path = re.sub(r"/page/\d+$", "", path)
    path = f"{path}/page/{page_number}/"
    return urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))


def extract_category_title(html_text: str) -> str:
    soup = BeautifulSoup(html_text, "html.parser")

    for selector in ["h1", '[itemprop="name"]', "title"]:
        node = soup.select_one(selector)
        if node:
            text = norm_text(node.get_text(" ", strip=True))
            if text:
                return text

    return ""


def extract_product_links_from_html(
    html_text: str,
    page_url: str,
    category_url: str = "",
    category_title: str = "",
    page_number: int = 1,
    site: str = "domlenta",
) -> list[ProductUrlRow]:
    soup = BeautifulSoup(html_text, "html.parser")
    rows: list[ProductUrlRow] = []
    seen: set[str] = set()
    base_url = site_base_url(site)

    # 1. Обычные ссылки из SSR HTML.
    for a in soup.find_all("a", href=True):
        href = absolutize(a.get("href", ""), page_url or base_url)
        path = urlparse(href).path

        if site == "obi":
            is_product = path.startswith("/products/") or "/product/" in path
        else:
            is_product = "/product/" in path

        if not is_product:
            continue

        product_url = normalize_url(href)
        if product_url in seen:
            continue

        seen.add(product_url)
        rows.append(
            ProductUrlRow(
                category_url=category_url,
                category_title=category_title,
                page_number=page_number,
                page_url=page_url,
                product_url=product_url,
                product_slug=product_slug_from_url(product_url),
                anchor_text=norm_text(a.get_text(" ", strip=True)),
            )
        )

    # 2. Запасной regex по всему HTML: иногда приложение хранит ссылки в данных.
    if site == "obi":
        product_re = (
            r'https?://obi\.ru/products/[^"\'<>\s]+|/products/[^"\'<>\s]+|'
            r'https?://domlenta\.ru/product/[^"\'<>\s]+|/product/[^"\'<>\s]+'
        )
    else:
        product_re = r'https?://domlenta\.ru/product/[^"\'<>\s]+|/product/[^"\'<>\s]+'

    for match in re.finditer(product_re, html_text):
        raw = match.group(0)
        product_url = normalize_url(absolutize(raw, page_url or base_url))

        if product_url in seen:
            continue

        seen.add(product_url)
        rows.append(
            ProductUrlRow(
                category_url=category_url,
                category_title=category_title,
                page_number=page_number,
                page_url=page_url,
                product_url=product_url,
                product_slug=product_slug_from_url(product_url),
                anchor_text="",
            )
        )

    return rows


def dedupe_url_rows(rows: Iterable[ProductUrlRow]) -> list[ProductUrlRow]:
    seen: set[str] = set()
    result: list[ProductUrlRow] = []

    for row in rows:
        key = row.product_url
        if key in seen:
            continue
        seen.add(key)
        result.append(row)

    return result


def write_product_urls(rows: list[ProductUrlRow], out_dir: Path) -> None:
    ensure_dir(out_dir)

    csv_path = out_dir / "product_urls.csv"
    jsonl_path = out_dir / "product_urls.jsonl"
    txt_path = out_dir / "product_urls.txt"

    fieldnames = [
        "category_url",
        "category_title",
        "page_number",
        "page_url",
        "product_url",
        "product_slug",
        "anchor_text",
    ]

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")

    with txt_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(row.product_url + "\n")


def get_playwright():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Не установлен Playwright.\n"
            "Установи:\n"
            "  python3 -m pip install playwright\n"
            "  python3 -m playwright install chromium"
        ) from exc

    return sync_playwright


def scroll_page(page, scrolls: int, pause: float) -> None:
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


def validate_category_urls(category_urls: list[str], site: str) -> None:
    expected_host = "obi.ru" if site == "obi" else "domlenta.ru"
    other_host = "domlenta.ru" if site == "obi" else "obi.ru"

    bad_urls = []
    for url in category_urls:
        host = urlparse(url).netloc.lower()
        if other_host in host or (host and expected_host not in host):
            bad_urls.append(url)

    if bad_urls:
        joined = "\n  ".join(bad_urls)
        raise ValueError(
            f"URL категории не соответствует --site {site}. Ожидается домен {expected_host}.\n"
            f"Проверь эти URL:\n  {joined}"
        )


def new_browser_context(
    p: Any,
    headed: bool,
    user_data_dir: str = "",
    cdp_url: str = "",
) -> tuple[Any, Any, bool, bool]:
    if cdp_url:
        browser = p.chromium.connect_over_cdp(cdp_url)
        context = browser.contexts[0] if browser.contexts else browser.new_context()
        return browser, context, False, False

    context_kwargs = {
        "user_agent": DEFAULT_USER_AGENT,
        "viewport": {"width": 1440, "height": 1200},
        "locale": "ru-RU",
    }

    if user_data_dir:
        context = p.chromium.launch_persistent_context(
            user_data_dir,
            headless=not headed,
            **context_kwargs,
        )
        return None, context, False, True

    browser = p.chromium.launch(headless=not headed)
    context = browser.new_context(**context_kwargs)
    return browser, context, True, True


def collect_urls_with_browser(
    category_urls: list[str],
    out_dir: Path,
    max_pages: int,
    site: str,
    headed: bool,
    user_data_dir: str,
    cdp_url: str,
    scrolls: int,
    scroll_pause: float,
    empty_stop: int,
    save_html: bool,
) -> list[ProductUrlRow]:
    ensure_dir(out_dir)
    source_dir = out_dir / "listing_html"
    if save_html:
        ensure_dir(source_dir)

    all_rows: list[ProductUrlRow] = []

    sync_playwright = get_playwright()

    with sync_playwright() as p:
        browser, context, close_browser, close_context = new_browser_context(
            p,
            headed=headed,
            user_data_dir=user_data_dir,
            cdp_url=cdp_url,
        )
        page = context.new_page()

        for category_url in category_urls:
            category_url = normalize_url(category_url, keep_query=False)
            empty_streak = 0
            category_title = ""

            for page_number in range(1, max_pages + 1):
                page_url = category_page_url(category_url, page_number, site=site)
                eprint(f"[INFO] Категория: {category_url} | страница {page_number}: {page_url}")

                try:
                    page.goto(page_url, wait_until="domcontentloaded", timeout=90_000)
                    page.wait_for_timeout(2000)
                    scroll_page(page, scrolls=scrolls, pause=scroll_pause)
                    html_text = page.content()
                except Exception as exc:
                    eprint(f"[WARN] Не удалось открыть {page_url}: {exc}")
                    empty_streak += 1
                    if empty_streak >= empty_stop:
                        break
                    continue

                if page_number == 1:
                    category_title = extract_category_title(html_text)

                if save_html:
                    safe = clean_filename(urlparse(page_url).path.strip("/").replace("/", "_"))
                    (source_dir / f"{safe or 'category'}_p{page_number:03d}.html").write_text(
                        html_text, encoding="utf-8"
                    )

                rows = extract_product_links_from_html(
                    html_text=html_text,
                    page_url=page_url,
                    category_url=category_url,
                    category_title=category_title,
                    page_number=page_number,
                    site=site,
                )

                new_count_before = len(dedupe_url_rows(all_rows))
                all_rows.extend(rows)
                new_count_after = len(dedupe_url_rows(all_rows))
                added = new_count_after - new_count_before

                eprint(f"[INFO] Найдено ссылок на странице: {len(rows)}, новых: {added}")

                if added == 0:
                    empty_streak += 1
                else:
                    empty_streak = 0

                if empty_streak >= empty_stop:
                    eprint(f"[INFO] Останавливаю категорию: {empty_streak} пустых страниц подряд.")
                    break

        page.close()
        if close_context:
            context.close()
        if browser and close_browser:
            browser.close()

    result = dedupe_url_rows(all_rows)
    write_product_urls(result, out_dir)
    return result


def collect_urls_from_listing_html(
    html_paths: list[str],
    out_dir: Path,
    site: str,
) -> list[ProductUrlRow]:
    all_rows: list[ProductUrlRow] = []

    for idx, raw_path in enumerate(html_paths, start=1):
        path = Path(raw_path)
        if not path.exists():
            eprint(f"[WARN] HTML не найден: {path}")
            continue

        html_text = path.read_text(encoding="utf-8", errors="ignore")
        category_title = extract_category_title(html_text)

        rows = extract_product_links_from_html(
            html_text=html_text,
            page_url=site_base_url(site),
            category_url="",
            category_title=category_title,
            page_number=idx,
            site=site,
        )
        all_rows.extend(rows)

    result = dedupe_url_rows(all_rows)
    write_product_urls(result, out_dir)
    return result


def read_urls_csv_or_txt(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)

    urls: list[str] = []

    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                url = row.get("product_url") or row.get("url") or ""
                url = url.strip()
                if url:
                    urls.append(normalize_url(url))
    else:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(normalize_url(line))

    return sorted(set(urls))


def safe_json_loads(text: str) -> Optional[Any]:
    text = text.strip()
    if not text:
        return None

    # Иногда в script попадает мусор вокруг JSON. Здесь не лечим JS, только аккуратно режем пробелы.
    try:
        return json.loads(text)
    except Exception:
        return None


def iter_json_ld_objects(data: Any) -> Iterable[dict[str, Any]]:
    if isinstance(data, dict):
        yield data
        graph = data.get("@graph")
        if isinstance(graph, list):
            for item in graph:
                if isinstance(item, dict):
                    yield from iter_json_ld_objects(item)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield from iter_json_ld_objects(item)


def is_product_jsonld(obj: dict[str, Any]) -> bool:
    t = obj.get("@type")
    if isinstance(t, str):
        return t.lower() == "product"
    if isinstance(t, list):
        return any(isinstance(x, str) and x.lower() == "product" for x in t)
    return False


def extract_jsonld_product(soup: BeautifulSoup) -> Optional[dict[str, Any]]:
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = script.string or script.get_text("", strip=False)
        data = safe_json_loads(raw)
        if data is None:
            continue

        for obj in iter_json_ld_objects(data):
            if is_product_jsonld(obj):
                return obj

    return None


def extract_breadcrumbs(soup: BeautifulSoup) -> list[str]:
    result: list[str] = []

    # 1. Микроразметка breadcrumbs.
    for meta in soup.select('[itemtype*="BreadcrumbList"] [itemprop="name"][content]'):
        text = norm_text(meta.get("content"))
        if text and text not in result:
            result.append(text)

    # 2. Визуальные ссылки breadcrumbs.
    if not result:
        for node in soup.select(".breadcrumbs-link, .breadcrumbs a, lui-breadcrumbs a, lui-breadcrumbs span"):
            text = norm_text(node.get_text(" ", strip=True))
            if text and text not in result:
                result.append(text)

    return result


def extract_categories_from_html(soup: BeautifulSoup) -> list[str]:
    result: list[str] = []

    for a in soup.select('a[href*="/catalog/"], a[href*="/napolnye-pokrytija/"]'):
        text = norm_text(a.get_text(" ", strip=True))
        href = a.get("href", "")
        if not text:
            continue
        if "/catalog/" not in href and "/napolnye-pokrytija/" not in href:
            continue
        if text not in result:
            result.append(text)

    return result


def extract_properties_from_jsonld(product: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}

    props = product.get("additionalProperty")
    if isinstance(props, dict):
        props = [props]

    if isinstance(props, list):
        for item in props:
            if not isinstance(item, dict):
                continue
            name = norm_text(str(item.get("name", "")))
            value = item.get("value", "")
            if isinstance(value, (list, dict)):
                value_text = json.dumps(value, ensure_ascii=False)
            else:
                value_text = norm_text(str(value))
            if name:
                result[name] = value_text

    return result


def extract_properties_from_visible_html(soup: BeautifulSoup) -> dict[str, str]:
    """
    Запасной парсер видимого блока характеристик.
    На Domlenta строки обычно выглядят так:
        .line-name
        .line-text
    """
    result: dict[str, str] = {}

    for line in soup.select(".line"):
        name_node = line.select_one(".line-name")
        value_node = line.select_one(".line-text, .text-value")
        if not name_node or not value_node:
            continue

        name = norm_text(name_node.get_text(" ", strip=True))
        value = norm_text(value_node.get_text(" ", strip=True))
        if name and value:
            result[name] = value

    return result


def extract_obi_properties_from_text(soup: BeautifulSoup) -> dict[str, str]:
    """
    OBI SSR text is regular enough to recover the "Характеристики" table from lines:
        ### Основные
        Наименование товара
            ...
        Бренд
            ...
    The DOM class names are less stable than this text structure.
    """
    result: dict[str, str] = {}
    raw_lines = soup.get_text("\n", strip=True).splitlines()
    lines = [norm_text(line) for line in raw_lines if norm_text(line)]

    try:
        start = next(i for i, line in enumerate(lines) if line.lower() == "характеристики")
    except StopIteration:
        return result

    stop_words = {"отзывы", "похожие товары", "с этим товаром покупают", "наличие в магазинах"}
    section_names = {
        "служебные",
        "основные",
        "упаковка",
        "гарантия",
        "технические характеристики",
        "дополнительные",
    }
    noise = {"скрыть", "все характеристики"}

    section = ""
    i = start + 1
    while i < len(lines):
        line = lines[i]
        low = line.lower()
        if low in stop_words:
            break
        if low in noise:
            i += 1
            continue
        if low in section_names:
            section = line
            i += 1
            continue
        if i + 1 >= len(lines):
            break

        value = lines[i + 1]
        value_low = value.lower()
        if value_low in stop_words:
            break
        if value_low in section_names or value_low in noise:
            i += 1
            continue

        key = f"{section} > {line}" if section else line
        result.setdefault(key, value)
        result.setdefault(line, value)
        i += 2

    return result


def extract_obi_price_and_availability(soup: BeautifulSoup) -> tuple[str, str]:
    text = soup.get_text("\n", strip=True)
    lines = [norm_text(line) for line in text.splitlines() if norm_text(line)]

    price = ""
    for line in lines:
        if "₽" in line and re.search(r"\d", line):
            price = line
            break

    availability = ""
    for line in lines:
        if line in {"В наличии", "Нет в наличии", "Только в гипермаркете"}:
            availability = line
            break

    return price, availability


def normalize_image_url(url: str, raw_image_url: bool, image_resample: str, base_url: str = BASE_URL) -> str:
    url = absolutize(url, base_url)
    parsed = urlparse(url)

    if "cdn.api.domlenta.ru" not in parsed.netloc:
        return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", parsed.query, ""))

    if raw_image_url:
        path = re.sub(r"^/resample/webp/\d+x\d+/", "/", parsed.path)
        return urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))

    if image_resample:
        path = re.sub(r"/resample/webp/\d+x\d+/", f"/resample/webp/{image_resample}/", parsed.path)
        return urlunparse((parsed.scheme, parsed.netloc, path, "", parsed.query, ""))

    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", parsed.query, ""))


def extract_images(
    soup: BeautifulSoup,
    product: Optional[dict[str, Any]],
    raw_image_url: bool,
    image_resample: str,
    site: str = "domlenta",
) -> list[str]:
    images: list[str] = []
    seen: set[str] = set()
    base_url = site_base_url(site)

    def add(url: str) -> None:
        if not url:
            return
        full = normalize_image_url(
            url,
            raw_image_url=raw_image_url,
            image_resample=image_resample,
            base_url=base_url,
        )

        host = urlparse(full).netloc
        path = urlparse(full).path

        # Оставляем именно товарные изображения, а не логотипы/иконки.
        if site == "obi":
            if "media.obi.ru" not in host and "cdn.api.domlenta.ru" not in host:
                return
            if "cdn.api.domlenta.ru" in host and "/photo/" not in path and "/catalog-image/" not in path:
                return
            if "media.obi.ru" in host and Path(path).suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                return
        else:
            if "cdn.api.domlenta.ru" not in host:
                return
            if "/photo/" not in path and "/catalog-image/" not in path:
                return

        if full not in seen:
            seen.add(full)
            images.append(full)

    if product:
        image_value = product.get("image")
        if isinstance(image_value, str):
            add(image_value)
        elif isinstance(image_value, list):
            for x in image_value:
                if isinstance(x, str):
                    add(x)

    for img in soup.find_all("img"):
        for attr in ["src", "data-src", "ng-src"]:
            value = img.get(attr)
            if value:
                add(value)

        srcset = img.get("srcset")
        if srcset:
            for part in srcset.split(","):
                url = part.strip().split(" ")[0]
                add(url)

    # Дополнительно regex по HTML для URL, которые могли лежать в JSON внутри Angular.
    html_text = str(soup)
    if site == "obi":
        image_re = r'https?://media\.obi\.ru/[^"\'<>\s)]+|https?://cdn\.api\.domlenta\.ru/[^"\'<>\s)]+'
    else:
        image_re = r'https?://cdn\.api\.domlenta\.ru/[^"\'<>\s)]+'

    for match in re.finditer(image_re, html_text):
        add(match.group(0))

    return images


def parse_product_html(
    html_text: str,
    url: str,
    final_url: str = "",
    raw_image_url: bool = False,
    image_resample: str = "900x900",
) -> ProductRow:
    soup = BeautifulSoup(html_text, "html.parser")
    product = extract_jsonld_product(soup)
    site = infer_site_from_url(final_url or url)

    row = ProductRow(url=url, final_url=final_url or url)

    try:
        if product:
            row.name = norm_text(str(product.get("name", "")))
            row.sku = norm_text(str(product.get("sku", "")))
            row.description = norm_text(str(product.get("description", "")))

            brand = product.get("brand")
            if isinstance(brand, dict):
                row.brand = norm_text(str(brand.get("name", "")))
            elif isinstance(brand, str):
                row.brand = norm_text(brand)

            offers = product.get("offers")
            if isinstance(offers, dict):
                row.price = norm_text(str(offers.get("price", "")))
                row.price_currency = norm_text(str(offers.get("priceCurrency", "")))
                row.availability = norm_text(str(offers.get("availability", "")))

            row.properties = extract_properties_from_jsonld(product)

        if not row.name:
            h1 = soup.find("h1")
            if h1:
                row.name = norm_text(h1.get_text(" ", strip=True))

        if not row.sku:
            text = soup.get_text(" ", strip=True)
            match = re.search(r"(?:Арт\.|Артикул:?)\s*(\d+)", text)
            if match:
                row.sku = match.group(1)

        visible_props = extract_properties_from_visible_html(soup)
        for key, value in visible_props.items():
            row.properties.setdefault(key, value)

        if site == "obi":
            obi_props = extract_obi_properties_from_text(soup)
            for key, value in obi_props.items():
                row.properties.setdefault(key, value)

            if not row.brand:
                row.brand = row.properties.get("Бренд", "")
            if not row.price or not row.availability:
                price, availability = extract_obi_price_and_availability(soup)
                row.price = row.price or price
                row.availability = row.availability or availability
            if row.price and not row.price_currency:
                row.price_currency = "RUB"

        row.breadcrumbs = extract_breadcrumbs(soup)
        row.categories = extract_categories_from_html(soup)
        row.images = extract_images(
            soup=soup,
            product=product,
            raw_image_url=raw_image_url,
            image_resample=image_resample,
            site=site,
        )

        if not product and not row.name:
            row.parse_status = "no_product_data"
            row.error = "Не найден JSON-LD Product и не найден h1."

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
            "Cache-Control": "no-cache",
        }
    )
    return session


def fetch_html_requests(session: requests.Session, url: str, timeout: tuple[int, int] = (20, 60)) -> tuple[Optional[str], str]:
    try:
        response = session.get(url, timeout=timeout)
        if response.status_code >= 400:
            return None, f"HTTP {response.status_code}"
        return response.text, ""
    except Exception as exc:
        return None, str(exc)


def parse_products_with_requests(
    urls: list[str],
    raw_image_url: bool,
    image_resample: str,
    delay: float,
) -> tuple[list[ProductRow], list[str]]:
    session = make_requests_session()
    rows: list[ProductRow] = []
    failed_urls: list[str] = []

    for url in tqdm(urls, desc="Parsing product pages with requests"):
        html_text, error = fetch_html_requests(session, url)
        if not html_text:
            failed_urls.append(url)
            rows.append(ProductRow(url=url, parse_status="fetch_failed", error=error))
            continue

        row = parse_product_html(
            html_text=html_text,
            url=url,
            final_url=url,
            raw_image_url=raw_image_url,
            image_resample=image_resample,
        )

        if row.parse_status != "ok":
            failed_urls.append(url)

        rows.append(row)

        if delay > 0:
            time.sleep(delay)

    return rows, failed_urls


def parse_products_with_browser(
    urls: list[str],
    raw_image_url: bool,
    image_resample: str,
    headed: bool,
    user_data_dir: str,
    cdp_url: str,
    scrolls: int,
    scroll_pause: float,
    page_timeout_sec: float,
    save_html_dir: Optional[Path],
) -> list[ProductRow]:
    rows: list[ProductRow] = []

    if save_html_dir:
        ensure_dir(save_html_dir)

    sync_playwright = get_playwright()

    with sync_playwright() as p:
        browser, context, close_browser, close_context = new_browser_context(
            p,
            headed=headed,
            user_data_dir=user_data_dir,
            cdp_url=cdp_url,
        )
        page = context.new_page()

        for url in tqdm(urls, desc="Parsing product pages with browser"):
            try:
                saved_html_path: Optional[Path] = None
                if save_html_dir:
                    slug = clean_filename(product_slug_from_url(url))
                    saved_html_path = save_html_dir / f"{slug or stable_hash(url)}.html"

                if saved_html_path and saved_html_path.exists():
                    html_text = saved_html_path.read_text(encoding="utf-8", errors="ignore")
                    row = parse_product_html(
                        html_text=html_text,
                        url=url,
                        final_url=url,
                        raw_image_url=raw_image_url,
                        image_resample=image_resample,
                    )
                    rows.append(row)
                    continue

                page.goto(url, wait_until="domcontentloaded", timeout=int(max(1.0, page_timeout_sec) * 1000))
                page.wait_for_timeout(1500)
                scroll_page(page, scrolls=scrolls, pause=scroll_pause)

                html_text = page.content()
                final_url = page.url

                if saved_html_path:
                    saved_html_path.write_text(html_text, encoding="utf-8")

                row = parse_product_html(
                    html_text=html_text,
                    url=url,
                    final_url=final_url,
                    raw_image_url=raw_image_url,
                    image_resample=image_resample,
                )
                rows.append(row)

            except Exception as exc:
                rows.append(ProductRow(url=url, parse_status="fetch_failed", error=str(exc)))

        page.close()
        if close_context:
            context.close()
        if browser and close_browser:
            browser.close()

    return rows


def merge_product_rows(primary: list[ProductRow], fallback: list[ProductRow]) -> list[ProductRow]:
    fallback_by_url = {row.url: row for row in fallback}
    result: list[ProductRow] = []

    for row in primary:
        if row.parse_status == "ok" and row.name:
            result.append(row)
        else:
            result.append(fallback_by_url.get(row.url, row))

    primary_urls = {row.url for row in primary}
    for row in fallback:
        if row.url not in primary_urls:
            result.append(row)

    return result


def guess_extension_from_response(url: str, content_type: str | None = None) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
        return ".jpg" if suffix == ".jpeg" else suffix

    if content_type:
        content_type = content_type.split(";")[0].strip().lower()
        ext = mimetypes.guess_extension(content_type)
        if ext:
            return ".jpg" if ext == ".jpe" else ext

    return ".jpg"


def download_image(
    session: requests.Session,
    url: str,
    target_without_ext: Path,
    referer: str,
    retries: int,
) -> Optional[Path]:
    headers = {"Referer": referer or BASE_URL + "/"}

    last_error: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        try:
            with session.get(url, headers=headers, stream=True, timeout=(20, 90)) as response:
                response.raise_for_status()
                ext = guess_extension_from_response(url, response.headers.get("Content-Type"))
                target = target_without_ext.with_suffix(ext)
                tmp = target.with_suffix(target.suffix + ".part")

                with tmp.open("wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 128):
                        if chunk:
                            f.write(chunk)

                tmp.replace(target)
                return target

        except Exception as exc:
            last_error = exc
            time.sleep(0.8 * attempt)

    eprint(f"[WARN] Не удалось скачать изображение: {url} | {last_error}")
    return None


def download_product_images(
    product_rows: list[ProductRow],
    out_dir: Path,
    retries: int,
    delay: float,
) -> list[ProductRow]:
    images_root = out_dir / "images"
    ensure_dir(images_root)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        }
    )

    total = sum(len(row.images) for row in product_rows)
    progress = tqdm(total=total, desc="Downloading product images")

    for row in product_rows:
        sku_or_hash = row.sku or stable_hash(row.url)
        product_dir_name = clean_filename(f"{sku_or_hash}_{row.name}", max_len=120)
        product_dir = images_root / product_dir_name
        ensure_dir(product_dir)

        local_paths: list[str] = []

        for idx, image_url in enumerate(row.images, start=1):
            image_hash = stable_hash(image_url)
            target_without_ext = product_dir / f"{idx:02d}_{image_hash}"

            existing = [
                p for p in product_dir.glob(target_without_ext.name + ".*")
                if not p.name.endswith(".part")
            ]

            if existing:
                saved = existing[0]
            else:
                saved = download_image(
                    session=session,
                    url=image_url,
                    target_without_ext=target_without_ext,
                    referer=row.final_url or row.url,
                    retries=retries,
                )

            if saved:
                local_paths.append(str(saved.relative_to(out_dir)))

            progress.update(1)

            if delay > 0:
                time.sleep(delay)

        row.local_image_paths = local_paths

    progress.close()
    return product_rows


def parse_listing_products_from_html(
    html_paths: list[str],
    site: str,
) -> list[ProductRow]:
    rows: list[ProductRow] = []
    seen: set[str] = set()

    for raw_path in html_paths:
        path = Path(raw_path)
        if not path.exists():
            eprint(f"[WARN] HTML не найден: {path}")
            continue

        html_text = path.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(html_text, "html.parser")
        category_title = extract_category_title(html_text)

        for a in soup.select('a[automation-id="productCard"][href], a.product-card[href], a[href*="/product/"], a[href*="/products/"]'):
            href = absolutize(a.get("href", ""), site_base_url(site))
            href = normalize_url(href)
            if site == "obi":
                is_product = "/product/" in urlparse(href).path or urlparse(href).path.startswith("/products/")
            else:
                is_product = "/product/" in urlparse(href).path
            if not is_product or href in seen:
                continue

            seen.add(href)
            card = a
            row = ProductRow(url=href, final_url=href)

            name_node = card.select_one('[automation-id="product-names"], .card-name_content, [title]')
            if name_node:
                row.name = norm_text(name_node.get_text(" ", strip=True) or name_node.get("title"))
            if not row.name:
                row.name = norm_text(card.get("title") or a.get_text(" ", strip=True))

            price_node = card.select_one('[automation-id="product-price"], .product-price, .main-price')
            if price_node:
                row.price = norm_text(price_node.get_text(" ", strip=True))
                row.price_currency = "RUB" if "₽" in row.price else ""

            text = norm_text(card.get_text(" ", strip=True))
            if "Нет в наличии" in text:
                row.availability = "Нет в наличии"
            elif "В наличии" in text:
                row.availability = "В наличии"
            elif "Только в гипермаркете" in text:
                row.availability = "Только в гипермаркете"

            row.sku = product_slug_from_url(href).split("-")[-1] if product_slug_from_url(href).split("-")[-1].isdigit() else ""
            row.breadcrumbs = [category_title] if category_title else []
            row.categories = extract_categories_from_html(soup)
            row.properties = {"source": "listing_html", "listing_html": str(path)}
            row.images = extract_images(soup=BeautifulSoup(str(card), "html.parser"), product=None, raw_image_url=False, image_resample="900x900", site=site)
            rows.append(row)

    return rows


def parse_listing_products_command(
    html_paths: list[str],
    out_dir: Path,
    site: str,
    download_images: bool,
    retries: int,
    delay: float,
) -> list[ProductRow]:
    rows = parse_listing_products_from_html(html_paths=html_paths, site=site)
    if download_images:
        rows = download_product_images(product_rows=rows, out_dir=out_dir, retries=retries, delay=delay)
    write_products(rows, out_dir)
    return rows


def parse_saved_product_html_command(
    html_paths: list[str],
    out_dir: Path,
    raw_image_url: bool,
    image_resample: str,
    download_images: bool = False,
    retries: int = 3,
    delay: float = 0.2,
) -> list[ProductRow]:
    rows: list[ProductRow] = []
    product_html_root = out_dir / "product_html"
    images_root = out_dir / "images"

    local_images_by_sku: dict[str, list[str]] = {}
    if images_root.exists():
        for image_path in sorted(images_root.rglob("*")):
            if not image_path.is_file() or image_path.name.endswith(".part"):
                continue
            sku = image_path.parent.name.split("_", 1)[0]
            if sku:
                local_images_by_sku.setdefault(sku, []).append(str(image_path.relative_to(out_dir)))

    for raw_path in html_paths:
        path = Path(raw_path)
        if not path.exists():
            eprint(f"[WARN] HTML не найден: {path}")
            continue

        html_text = path.read_text(encoding="utf-8", errors="ignore")
        slug = path.stem
        url = f"https://domlenta.ru/product/{slug}/"
        row = parse_product_html(
            html_text=html_text,
            url=url,
            final_url=url,
            raw_image_url=raw_image_url,
            image_resample=image_resample,
        )
        if row.sku in local_images_by_sku:
            row.local_image_paths = local_images_by_sku[row.sku]
        rows.append(row)

    if download_images:
        rows = download_product_images(
            product_rows=rows,
            out_dir=out_dir,
            retries=retries,
            delay=delay,
        )

    write_products(rows, out_dir)
    return rows


def write_products(product_rows: list[ProductRow], out_dir: Path) -> None:
    ensure_dir(out_dir)

    products_csv = out_dir / "products.csv"
    products_jsonl = out_dir / "products.jsonl"
    images_csv = out_dir / "product_images.csv"
    properties_csv = out_dir / "product_properties.csv"

    product_fields = [
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

    with products_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=product_fields)
        writer.writeheader()

        for row in product_rows:
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
                    "breadcrumbs": " > ".join(row.breadcrumbs),
                    "categories": "; ".join(row.categories),
                    "properties_json": json.dumps(row.properties, ensure_ascii=False),
                    "images_json": json.dumps(row.images, ensure_ascii=False),
                    "local_image_paths_json": json.dumps(row.local_image_paths, ensure_ascii=False),
                    "parse_status": row.parse_status,
                    "error": row.error,
                }
            )

    with products_jsonl.open("w", encoding="utf-8") as f:
        for row in product_rows:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")

    with images_csv.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = ["product_url", "sku", "name", "image_index", "image_url", "local_image_path"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in product_rows:
            max_len = max(len(row.images), len(row.local_image_paths))
            for idx in range(max_len):
                writer.writerow(
                    {
                        "product_url": row.url,
                        "sku": row.sku,
                        "name": row.name,
                        "image_index": idx + 1,
                        "image_url": row.images[idx] if idx < len(row.images) else "",
                        "local_image_path": row.local_image_paths[idx] if idx < len(row.local_image_paths) else "",
                    }
                )

    with properties_csv.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = ["product_url", "sku", "name", "property_name", "property_value"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in product_rows:
            for key, value in row.properties.items():
                writer.writerow(
                    {
                        "product_url": row.url,
                        "sku": row.sku,
                        "name": row.name,
                        "property_name": key,
                        "property_value": value,
                    }
                )


def parse_products_command(
    urls: list[str],
    out_dir: Path,
    fetch_mode: str,
    headed: bool,
    user_data_dir: str,
    cdp_url: str,
    scrolls: int,
    scroll_pause: float,
    page_timeout_sec: float,
    save_product_html: bool,
    raw_image_url: bool,
    image_resample: str,
    download_images: bool,
    retries: int,
    delay: float,
) -> list[ProductRow]:
    ensure_dir(out_dir)

    save_html_dir = out_dir / "product_html" if save_product_html else None

    if fetch_mode == "requests":
        rows, _ = parse_products_with_requests(
            urls=urls,
            raw_image_url=raw_image_url,
            image_resample=image_resample,
            delay=delay,
        )

    elif fetch_mode == "playwright":
        rows = parse_products_with_browser(
            urls=urls,
            raw_image_url=raw_image_url,
            image_resample=image_resample,
            headed=headed,
            user_data_dir=user_data_dir,
            cdp_url=cdp_url,
            scrolls=scrolls,
            scroll_pause=scroll_pause,
            page_timeout_sec=page_timeout_sec,
            save_html_dir=save_html_dir,
        )

    else:
        request_rows, failed_urls = parse_products_with_requests(
            urls=urls,
            raw_image_url=raw_image_url,
            image_resample=image_resample,
            delay=delay,
        )

        failed_urls = sorted(set(failed_urls))
        if failed_urls:
            eprint(f"[INFO] Requests не разобрал {len(failed_urls)} страниц. Пробую Playwright fallback.")
            fallback_rows = parse_products_with_browser(
                urls=failed_urls,
                raw_image_url=raw_image_url,
                image_resample=image_resample,
                headed=headed,
                user_data_dir=user_data_dir,
                cdp_url=cdp_url,
                scrolls=scrolls,
                scroll_pause=scroll_pause,
                page_timeout_sec=page_timeout_sec,
                save_html_dir=save_html_dir,
            )
            rows = merge_product_rows(request_rows, fallback_rows)
        else:
            rows = request_rows

    write_products(rows, out_dir)

    if download_images:
        rows = download_product_images(
            product_rows=rows,
            out_dir=out_dir,
            retries=retries,
            delay=delay,
        )

    write_products(rows, out_dir)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Сбор URL товаров Domlenta по напольным покрытиям и парсинг карточек товаров."
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_output(p: argparse.ArgumentParser) -> None:
        p.add_argument("-o", "--out", default="domlenta_floor_coverings", help="Папка результата.")
        p.add_argument(
            "--site",
            choices=["domlenta", "obi"],
            default="domlenta",
            help="Источник/формат сайта. Для OBI используй --site obi.",
        )

    def add_browser_opts(p: argparse.ArgumentParser) -> None:
        p.add_argument("--headed", action="store_true", help="Запустить браузер Playwright с окном.")
        p.add_argument(
            "--user-data-dir",
            default="",
            help="Постоянный профиль Playwright для cookies/защиты сайта, например data/sourse/playwright_profiles/obi.",
        )
        p.add_argument(
            "--cdp-url",
            default="",
            help="Подключиться к уже запущенному Chrome через DevTools, например http://127.0.0.1:9222.",
        )
        p.add_argument("--scrolls", type=int, default=8, help="Количество прокруток страницы.")
        p.add_argument("--scroll-pause", type=float, default=0.8, help="Пауза после прокрутки.")
        p.add_argument("--page-timeout-sec", type=float, default=90.0, help="Таймаут открытия страницы в браузере.")

    collect = subparsers.add_parser("collect-urls", help="Собрать URL товаров из категорий.")
    add_common_output(collect)
    add_browser_opts(collect)
    collect.add_argument(
        "--category-url",
        action="append",
        default=[],
        help="URL категории. Можно указывать несколько раз. По умолчанию зависит от --site.",
    )
    collect.add_argument("--max-pages", type=int, default=30, help="Максимум страниц пагинации на категорию.")
    collect.add_argument("--empty-stop", type=int, default=2, help="Останов после N страниц без новых товаров.")
    collect.add_argument("--save-html", action="store_true", help="Сохранять HTML страниц категорий.")
    collect.add_argument(
        "--listing-html",
        nargs="+",
        help="Вместо браузера распарсить локальные HTML-файлы выдачи.",
    )

    parse = subparsers.add_parser("parse-products", help="Распарсить карточки товаров по списку URL.")
    add_common_output(parse)
    add_browser_opts(parse)
    parse.add_argument("--urls-csv", required=True, help="CSV/TXT со списком product_url.")
    parse.add_argument(
        "--fetch-mode",
        choices=["auto", "requests", "playwright"],
        default="auto",
        help="Как получать страницы товаров.",
    )
    parse.add_argument("--save-product-html", action="store_true", help="Сохранять HTML карточек товаров.")
    parse.add_argument("--download-images", action="store_true", help="Скачать изображения товаров.")
    parse.add_argument("--raw-image-url", action="store_true", help="Пробовать URL без /resample/webp/900x900/.")
    parse.add_argument(
        "--image-resample",
        default="900x900",
        help="Размер resample в URL картинок, например 900x900 или 1500x1500. Игнорируется при --raw-image-url.",
    )
    parse.add_argument("--retries", type=int, default=3, help="Повторы скачивания изображений.")
    parse.add_argument("--delay", type=float, default=0.2, help="Пауза между запросами.")

    listing_products = subparsers.add_parser(
        "parse-listing-products",
        help="Собрать доступные данные товаров прямо из сохраненных HTML страниц каталога.",
    )
    add_common_output(listing_products)
    listing_products.add_argument("--listing-html", nargs="+", required=True, help="Локальные HTML-файлы выдачи.")
    listing_products.add_argument("--download-images", action="store_true", help="Скачать изображения товаров из листинга.")
    listing_products.add_argument("--retries", type=int, default=3, help="Повторы скачивания изображений.")
    listing_products.add_argument("--delay", type=float, default=0.2, help="Пауза между запросами.")

    saved_products = subparsers.add_parser(
        "parse-saved-products",
        help="Распарсить локально сохраненные HTML карточек товаров и записать products.csv.",
    )
    add_common_output(saved_products)
    saved_products.add_argument("--product-html", nargs="+", required=True, help="Локальные HTML-файлы карточек.")
    saved_products.add_argument("--raw-image-url", action="store_true", help="Пробовать URL без /resample/webp/900x900/.")
    saved_products.add_argument(
        "--image-resample",
        default="1500x1500",
        help="Размер resample в URL картинок, например 900x900 или 1500x1500. Игнорируется при --raw-image-url.",
    )
    saved_products.add_argument("--download-images", action="store_true", help="Скачать изображения товаров из локальных HTML карточек.")
    saved_products.add_argument("--retries", type=int, default=3, help="Повторы скачивания изображений.")
    saved_products.add_argument("--delay", type=float, default=0.2, help="Пауза между запросами.")

    all_cmd = subparsers.add_parser("all", help="Собрать URL товаров и сразу распарсить карточки.")
    add_common_output(all_cmd)
    add_browser_opts(all_cmd)
    all_cmd.add_argument(
        "--category-url",
        action="append",
        default=[],
        help="URL категории. Можно указывать несколько раз. По умолчанию зависит от --site.",
    )
    all_cmd.add_argument("--max-pages", type=int, default=30, help="Максимум страниц пагинации на категорию.")
    all_cmd.add_argument("--empty-stop", type=int, default=2, help="Останов после N страниц без новых товаров.")
    all_cmd.add_argument("--save-html", action="store_true", help="Сохранять HTML страниц категорий.")
    all_cmd.add_argument(
        "--fetch-mode",
        choices=["auto", "requests", "playwright"],
        default="auto",
        help="Как получать страницы товаров.",
    )
    all_cmd.add_argument("--save-product-html", action="store_true", help="Сохранять HTML карточек товаров.")
    all_cmd.add_argument("--download-images", action="store_true", help="Скачать изображения товаров.")
    all_cmd.add_argument("--raw-image-url", action="store_true", help="Пробовать URL без /resample/webp/900x900/.")
    all_cmd.add_argument(
        "--image-resample",
        default="900x900",
        help="Размер resample в URL картинок, например 900x900 или 1500x1500. Игнорируется при --raw-image-url.",
    )
    all_cmd.add_argument("--retries", type=int, default=3, help="Повторы скачивания изображений.")
    all_cmd.add_argument("--delay", type=float, default=0.2, help="Пауза между запросами.")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    out_dir = Path(args.out).resolve()
    ensure_dir(out_dir)

    if args.command == "collect-urls":
        if args.listing_html:
            rows = collect_urls_from_listing_html(args.listing_html, out_dir=out_dir, site=args.site)
        else:
            category_urls = args.category_url or DEFAULT_CATEGORY_URLS_BY_SITE[args.site]
            validate_category_urls(category_urls, args.site)
            rows = collect_urls_with_browser(
                category_urls=category_urls,
                out_dir=out_dir,
                max_pages=max(1, args.max_pages),
                site=args.site,
                headed=args.headed,
                user_data_dir=args.user_data_dir,
                cdp_url=args.cdp_url,
                scrolls=max(0, args.scrolls),
                scroll_pause=max(0.1, args.scroll_pause),
                empty_stop=max(1, args.empty_stop),
                save_html=args.save_html,
            )

        print(f"Готово. Собрано уникальных URL товаров: {len(rows)}")
        print(f"CSV: {out_dir / 'product_urls.csv'}")
        print(f"TXT: {out_dir / 'product_urls.txt'}")
        return 0

    if args.command == "parse-products":
        urls = read_urls_csv_or_txt(Path(args.urls_csv))
        rows = parse_products_command(
            urls=urls,
            out_dir=out_dir,
            fetch_mode=args.fetch_mode,
            headed=args.headed,
            user_data_dir=args.user_data_dir,
            cdp_url=args.cdp_url,
            scrolls=max(0, args.scrolls),
            scroll_pause=max(0.1, args.scroll_pause),
            page_timeout_sec=max(1.0, args.page_timeout_sec),
            save_product_html=args.save_product_html,
            raw_image_url=args.raw_image_url,
            image_resample=args.image_resample,
            download_images=args.download_images,
            retries=max(1, args.retries),
            delay=max(0.0, args.delay),
        )

        ok = sum(1 for r in rows if r.parse_status == "ok")
        print(f"Готово. Карточек: {len(rows)}, успешно: {ok}, с ошибками: {len(rows) - ok}")
        print(f"Товары CSV: {out_dir / 'products.csv'}")
        print(f"Картинки CSV: {out_dir / 'product_images.csv'}")
        print(f"Характеристики CSV: {out_dir / 'product_properties.csv'}")
        return 0

    if args.command == "parse-listing-products":
        rows = parse_listing_products_command(
            html_paths=args.listing_html,
            out_dir=out_dir,
            site=args.site,
            download_images=args.download_images,
            retries=max(1, args.retries),
            delay=max(0.0, args.delay),
        )
        print(f"Готово. Товаров из HTML листинга: {len(rows)}")
        print(f"Товары CSV: {out_dir / 'products.csv'}")
        print(f"Картинки CSV: {out_dir / 'product_images.csv'}")
        return 0

    if args.command == "parse-saved-products":
        rows = parse_saved_product_html_command(
            html_paths=args.product_html,
            out_dir=out_dir,
            raw_image_url=args.raw_image_url,
            image_resample=args.image_resample,
            download_images=args.download_images,
            retries=max(1, args.retries),
            delay=max(0.0, args.delay),
        )
        ok = sum(1 for r in rows if r.parse_status == "ok")
        print(f"Готово. Локальных карточек: {len(rows)}, успешно: {ok}, с ошибками: {len(rows) - ok}")
        print(f"Товары CSV: {out_dir / 'products.csv'}")
        print(f"Картинки CSV: {out_dir / 'product_images.csv'}")
        print(f"Характеристики CSV: {out_dir / 'product_properties.csv'}")
        return 0

    if args.command == "all":
        category_urls = args.category_url or DEFAULT_CATEGORY_URLS_BY_SITE[args.site]
        validate_category_urls(category_urls, args.site)

        url_rows = collect_urls_with_browser(
            category_urls=category_urls,
            out_dir=out_dir,
            max_pages=max(1, args.max_pages),
            site=args.site,
            headed=args.headed,
            user_data_dir=args.user_data_dir,
            cdp_url=args.cdp_url,
            scrolls=max(0, args.scrolls),
            scroll_pause=max(0.1, args.scroll_pause),
            empty_stop=max(1, args.empty_stop),
            save_html=args.save_html,
        )

        urls = [row.product_url for row in url_rows]
        if not urls:
            print("Готово. URL товаров: 0")
            print("Карточки не парсились, потому что страницы категорий не отдали ни одной ссылки товара.")
            print(f"Проверь сохраненные HTML: {out_dir / 'listing_html'}")
            return 1

        product_rows = parse_products_command(
            urls=urls,
            out_dir=out_dir,
            fetch_mode=args.fetch_mode,
            headed=args.headed,
            user_data_dir=args.user_data_dir,
            cdp_url=args.cdp_url,
            scrolls=max(0, args.scrolls),
            scroll_pause=max(0.1, args.scroll_pause),
            page_timeout_sec=max(1.0, args.page_timeout_sec),
            save_product_html=args.save_product_html,
            raw_image_url=args.raw_image_url,
            image_resample=args.image_resample,
            download_images=args.download_images,
            retries=max(1, args.retries),
            delay=max(0.0, args.delay),
        )

        ok = sum(1 for r in product_rows if r.parse_status == "ok")
        print(f"Готово. URL товаров: {len(url_rows)}")
        print(f"Карточек: {len(product_rows)}, успешно: {ok}, с ошибками: {len(product_rows) - ok}")
        print(f"URL CSV: {out_dir / 'product_urls.csv'}")
        print(f"Товары CSV: {out_dir / 'products.csv'}")
        print(f"Картинки CSV: {out_dir / 'product_images.csv'}")
        print(f"Характеристики CSV: {out_dir / 'product_properties.csv'}")
        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
