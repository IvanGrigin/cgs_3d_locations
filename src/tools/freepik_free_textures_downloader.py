#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
freepik_free_textures_downloader.py

Назначение:
    1) Парсит сохранённый HTML страницы выдачи Freepik.
    2) Находит карточки материалов/текстур.
    3) Оставляет только бесплатные карточки.
    4) Скачивает изображения-превью из img.freepik.com.
    5) Сохраняет metadata.csv, metadata.jsonl и source_pages/*.html.

Режимы:
    A. Основной надёжный режим:
        python3 freepik_free_textures_downloader.py --html freepik_page.html -o freepik_textures

    B. Несколько HTML-файлов:
        python3 freepik_free_textures_downloader.py --html html_pages/*.html -o freepik_textures

    C. Автосбор страниц через браузер Playwright:
        python3 freepik_free_textures_downloader.py \
          --url "https://ru.freepik.com/free-photos-vectors/текстуры-обоев-интерьера" \
          --pages 5 \
          --scrolls 25 \
          -o freepik_textures

Зависимости:
    pip install beautifulsoup4 requests tqdm
    Для режима --url:
        pip install playwright
        python3 -m playwright install chromium

Важно:
    Скрипт не обходит платный доступ, логин, капчу и ограничения Freepik.
    Он скачивает только те изображения, URL которых уже присутствуют в HTML выдачи.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import mimetypes
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/121.0.0.0 Safari/537.36"
)


@dataclass
class FreepikItem:
    source_html: str
    position: Optional[int]
    page_url: str
    image_url: str
    alt: str
    title: str
    tags: list[str]
    author_id: str
    width: Optional[int]
    height: Optional[int]
    is_premium: bool
    is_ai_generated: bool
    local_image_path: str = ""


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    value = html.unescape(value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def clean_filename(value: str, max_len: int = 90) -> str:
    value = normalize_text(value).lower()
    value = re.sub(r"https?://", "", value)
    value = re.sub(r"[^\wа-яА-ЯёЁ\-\. ]+", "_", value, flags=re.UNICODE)
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"_+", "_", value)
    value = value.strip("._- ")
    if not value:
        value = "item"
    return value[:max_len].strip("._- ")


def stable_hash(value: str, n: int = 12) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()[:n]


def strip_query_for_extension(url: str) -> str:
    parsed = urlparse(url)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))


def guess_extension(url: str, content_type: str | None = None) -> str:
    clean_url = strip_query_for_extension(url)
    suffix = Path(urlparse(clean_url).path).suffix.lower()

    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
        return ".jpg" if suffix == ".jpeg" else suffix

    if content_type:
        content_type = content_type.split(";")[0].strip().lower()
        ext = mimetypes.guess_extension(content_type)
        if ext:
            return ".jpg" if ext == ".jpe" else ext

    return ".jpg"


def parse_int(value: str | None) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def detect_position_from_href(href: str) -> Optional[int]:
    match = re.search(r"[?&#]position=(\d+)", href)
    if match:
        return int(match.group(1))
    return None


def absolutize_url(url: str, base_url: str = "https://ru.freepik.com") -> str:
    if not url:
        return ""
    url = html.unescape(url)
    return urljoin(base_url, url)


def canonical_image_url(url: str, image_width: Optional[int] = None, keep_query: bool = True) -> str:
    """
    Freepik preview URLs часто имеют параметры:
        ?semt=ais_hybrid&w=740&q=80

    keep_query=True сохраняет видимый preview.
    keep_query=False убирает query; иногда это даёт исходный preview-файл,
    но не гарантирует оригинальный ресурс.
    """
    url = html.unescape(url)
    if keep_query:
        return url

    parsed = urlparse(url)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))


def figure_is_premium(figure) -> bool:
    text = normalize_text(figure.get_text(" ", strip=True)).lower()
    hrefs = " ".join(a.get("href", "") for a in figure.find_all("a"))
    imgs = " ".join(img.get("src", "") for img in figure.find_all("img"))

    premium_markers = [
        "premium",
        "/premium-",
        "premium-photo",
        "premium-ai-image",
        "search-results-crown",
        "text-premium-gold",
    ]

    probe = f"{text} {hrefs} {imgs}".lower()
    return any(marker in probe for marker in premium_markers)


def figure_is_ai_generated(figure) -> bool:
    text = normalize_text(figure.get_text(" ", strip=True)).lower()
    hrefs = " ".join(a.get("href", "") for a in figure.find_all("a")).lower()
    imgs = " ".join(img.get("src", "") for img in figure.find_all("img")).lower()

    markers = [
        "сгенерировано с помощью ии",
        "free-ai-image",
        "premium-ai-image",
        "/free-ai-image/",
        "/premium-ai-image/",
    ]
    probe = f"{text} {hrefs} {imgs}"
    return any(marker in probe for marker in markers)


def extract_tags(figure) -> list[str]:
    tags: list[str] = []
    for tag_link in figure.select('[data-cy="related-tags"] a, li[data-cy="related-tag"] a'):
        text = normalize_text(tag_link.get_text(" ", strip=True))
        if text and text not in tags:
            tags.append(text)
    return tags


def extract_title_from_url_or_alt(page_url: str, alt: str) -> str:
    alt = normalize_text(alt)
    if alt:
        alt = re.sub(r"^(Бесплатный|Фотография|Вектор)\s+", "", alt, flags=re.IGNORECASE).strip()
        return alt

    parsed = urlparse(page_url)
    slug = Path(parsed.path).stem
    slug = re.sub(r"_\d+$", "", slug)
    slug = slug.replace("-", " ").replace("_", " ")
    return normalize_text(slug)


def parse_freepik_html(
    html_text: str,
    source_html_name: str,
    base_url: str = "https://ru.freepik.com",
    include_ai: bool = True,
    free_only: bool = True,
    keep_image_query: bool = True,
) -> list[FreepikItem]:
    soup = BeautifulSoup(html_text, "html.parser")

    items: list[FreepikItem] = []

    for figure in soup.select('figure[data-cy="resource-thumbnail"]'):
        a = figure.find("a", href=True)
        img = figure.find("img", src=True)

        if not a or not img:
            continue

        page_url = absolutize_url(a.get("href", ""), base_url)
        image_url_raw = absolutize_url(img.get("src", ""), base_url)

        if "img.freepik.com" not in urlparse(image_url_raw).netloc:
            continue

        is_premium = figure_is_premium(figure)
        is_ai_generated = figure_is_ai_generated(figure)

        if free_only and is_premium:
            continue

        if not include_ai and is_ai_generated:
            continue

        alt = normalize_text(img.get("alt"))
        title = extract_title_from_url_or_alt(page_url, alt)

        item = FreepikItem(
            source_html=source_html_name,
            position=detect_position_from_href(page_url),
            page_url=page_url,
            image_url=canonical_image_url(
                image_url_raw,
                image_width=parse_int(img.get("width")),
                keep_query=keep_image_query,
            ),
            alt=alt,
            title=title,
            tags=extract_tags(figure),
            author_id=normalize_text(figure.get("data-author")),
            width=parse_int(img.get("width")),
            height=parse_int(img.get("height")),
            is_premium=is_premium,
            is_ai_generated=is_ai_generated,
        )

        items.append(item)

    return items


def deduplicate_items(items: Iterable[FreepikItem]) -> list[FreepikItem]:
    seen: set[str] = set()
    result: list[FreepikItem] = []

    for item in items:
        key = item.page_url or item.image_url
        if key in seen:
            continue
        seen.add(key)
        result.append(item)

    return result


def write_metadata(items: list[FreepikItem], out_dir: Path) -> None:
    csv_path = out_dir / "metadata.csv"
    jsonl_path = out_dir / "metadata.jsonl"

    fieldnames = [
        "source_html",
        "position",
        "title",
        "alt",
        "tags",
        "page_url",
        "image_url",
        "local_image_path",
        "author_id",
        "width",
        "height",
        "is_premium",
        "is_ai_generated",
    ]

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for item in items:
            row = asdict(item)
            row["tags"] = "; ".join(item.tags)
            writer.writerow(row)

    with jsonl_path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")


def download_file(
    session: requests.Session,
    url: str,
    target_path_without_ext: Path,
    timeout: tuple[int, int] = (15, 60),
    retries: int = 3,
    sleep_base: float = 1.0,
) -> Optional[Path]:
    last_error: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        try:
            with session.get(url, stream=True, timeout=timeout) as response:
                response.raise_for_status()

                content_type = response.headers.get("Content-Type", "")
                ext = guess_extension(url, content_type)
                target_path = target_path_without_ext.with_suffix(ext)

                tmp_path = target_path.with_suffix(target_path.suffix + ".part")

                with tmp_path.open("wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 128):
                        if chunk:
                            f.write(chunk)

                tmp_path.replace(target_path)
                return target_path

        except Exception as exc:
            last_error = exc
            time.sleep(sleep_base * attempt)

    print(f"[WARN] Не удалось скачать: {url} | ошибка: {last_error}", file=sys.stderr)
    return None


def download_images(
    items: list[FreepikItem],
    out_dir: Path,
    delay: float = 0.25,
    retries: int = 3,
) -> list[FreepikItem]:
    images_dir = out_dir / "images"
    ensure_dir(images_dir)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
            "Referer": "https://ru.freepik.com/",
        }
    )

    for idx, item in enumerate(tqdm(items, desc="Downloading images"), start=1):
        title_part = clean_filename(item.title)
        hash_part = stable_hash(item.page_url or item.image_url)
        pos_part = f"{item.position:04d}" if item.position is not None else f"{idx:04d}"

        target_without_ext = images_dir / f"{pos_part}_{title_part}_{hash_part}"

        existing = list(images_dir.glob(target_without_ext.name + ".*"))
        existing = [p for p in existing if not p.name.endswith(".part")]
        if existing:
            item.local_image_path = str(existing[0].relative_to(out_dir))
            continue

        saved_path = download_file(
            session=session,
            url=item.image_url,
            target_path_without_ext=target_without_ext,
            retries=retries,
        )

        if saved_path:
            item.local_image_path = str(saved_path.relative_to(out_dir))

        if delay > 0:
            time.sleep(delay)

    return items


def read_html_inputs(paths: list[str]) -> list[tuple[str, str]]:
    expanded: list[Path] = []

    for raw in paths:
        matches = list(Path().glob(raw)) if any(ch in raw for ch in "*?[]") else [Path(raw)]
        expanded.extend(matches)

    html_files: list[Path] = []
    for path in expanded:
        if path.is_dir():
            html_files.extend(sorted(path.glob("*.html")))
            html_files.extend(sorted(path.glob("*.htm")))
        elif path.is_file():
            html_files.append(path)

    result: list[tuple[str, str]] = []
    for path in html_files:
        result.append((path.name, path.read_text(encoding="utf-8", errors="ignore")))

    return result


def find_next_page_url(html_text: str, current_url: str, base_url: str = "https://ru.freepik.com") -> Optional[str]:
    soup = BeautifulSoup(html_text, "html.parser")

    candidates: list[str] = []

    for a in soup.find_all("a", href=True):
        href = absolutize_url(a.get("href", ""), base_url)

        if not href:
            continue

        if href == current_url:
            continue

        if "freepik.com" not in urlparse(href).netloc:
            continue

        if re.search(r"/\d+(?:#|\?|$)", urlparse(href).path + ("#" + urlparse(href).fragment if urlparse(href).fragment else "")):
            candidates.append(href)

    if not candidates:
        return None

    def page_number(url: str) -> int:
        parsed = urlparse(url)
        match = re.search(r"/(\d+)$", parsed.path)
        if match:
            return int(match.group(1))
        return 10**9

    candidates = sorted(set(candidates), key=page_number)
    return candidates[0]


def collect_html_with_playwright(
    url: str,
    out_dir: Path,
    pages: int,
    scrolls: int,
    scroll_pause: float,
    headless: bool,
) -> list[tuple[str, str]]:
    """
    Открывает страницу в реальном браузере, прокручивает выдачу, сохраняет HTML.
    Не обходит логин, платные ограничения, капчу и антибот-защиту.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Для режима --url нужно установить Playwright:\n"
            "  pip install playwright\n"
            "  python3 -m playwright install chromium"
        ) from exc

    source_dir = out_dir / "source_pages"
    ensure_dir(source_dir)

    collected: list[tuple[str, str]] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(
            user_agent=DEFAULT_USER_AGENT,
            viewport={"width": 1440, "height": 1200},
            locale="ru-RU",
        )
        page = context.new_page()

        current_url: Optional[str] = url

        for page_idx in range(1, pages + 1):
            if not current_url:
                break

            print(f"[INFO] Открываю страницу {page_idx}: {current_url}", file=sys.stderr)
            page.goto(current_url, wait_until="domcontentloaded", timeout=90_000)
            page.wait_for_timeout(2500)

            last_height = 0
            stable_rounds = 0

            for _ in range(scrolls):
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                page.wait_for_timeout(int(scroll_pause * 1000))

                new_height = page.evaluate("document.body.scrollHeight")
                if new_height == last_height:
                    stable_rounds += 1
                else:
                    stable_rounds = 0
                    last_height = new_height

                if stable_rounds >= 4:
                    break

            html_text = page.content()
            source_name = f"freepik_page_{page_idx:03d}.html"
            source_path = source_dir / source_name
            source_path.write_text(html_text, encoding="utf-8")
            collected.append((source_name, html_text))

            next_url = find_next_page_url(html_text, current_url)
            if not next_url or next_url == current_url:
                break

            current_url = next_url

        context.close()
        browser.close()

    return collected


def save_source_html_copies(html_inputs: list[tuple[str, str]], out_dir: Path) -> None:
    source_dir = out_dir / "source_pages"
    ensure_dir(source_dir)

    for name, text in html_inputs:
        safe_name = clean_filename(Path(name).stem, max_len=120) + ".html"
        path = source_dir / safe_name
        if not path.exists():
            path.write_text(text, encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Скачать бесплатные изображения-превью материалов/текстур из HTML выдачи Freepik."
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--html",
        nargs="+",
        help="HTML-файл, папка с HTML или glob-маска. Пример: --html freepik_page.html или --html html_pages/*.html",
    )
    source.add_argument(
        "--url",
        help="URL страницы выдачи Freepik. Для этого режима нужен Playwright.",
    )

    parser.add_argument(
        "-o",
        "--out",
        default="freepik_textures_download",
        help="Папка для результата.",
    )
    parser.add_argument(
        "--include-ai",
        action="store_true",
        help="Не исключать AI-generated карточки. По умолчанию AI-generated исключаются.",
    )
    parser.add_argument(
        "--include-premium",
        action="store_true",
        help="Не исключать Premium. Обычно не нужно; по умолчанию Premium исключаются.",
    )
    parser.add_argument(
        "--strip-image-query",
        action="store_true",
        help="Убирать query-параметры из URL изображений. По умолчанию сохраняется видимый preview URL.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Только собрать metadata.csv/jsonl, без скачивания изображений.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.25,
        help="Пауза между скачиваниями изображений в секундах.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Число повторов скачивания одного изображения.",
    )

    parser.add_argument(
        "--pages",
        type=int,
        default=1,
        help="Сколько страниц собрать в режиме --url.",
    )
    parser.add_argument(
        "--scrolls",
        type=int,
        default=25,
        help="Сколько прокруток вниз делать на каждой странице в режиме --url.",
    )
    parser.add_argument(
        "--scroll-pause",
        type=float,
        default=1.0,
        help="Пауза после каждой прокрутки в режиме --url.",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Показывать окно браузера Playwright. Полезно, если сайт просит принять cookies.",
    )

    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    out_dir = Path(args.out).resolve()
    ensure_dir(out_dir)

    if args.html:
        html_inputs = read_html_inputs(args.html)
        if not html_inputs:
            print("[ERROR] HTML-файлы не найдены.", file=sys.stderr)
            return 2
        save_source_html_copies(html_inputs, out_dir)
    else:
        html_inputs = collect_html_with_playwright(
            url=args.url,
            out_dir=out_dir,
            pages=max(1, args.pages),
            scrolls=max(1, args.scrolls),
            scroll_pause=max(0.1, args.scroll_pause),
            headless=not args.headed,
        )

    all_items: list[FreepikItem] = []

    for source_name, html_text in html_inputs:
        page_items = parse_freepik_html(
            html_text=html_text,
            source_html_name=source_name,
            include_ai=args.include_ai,
            free_only=not args.include_premium,
            keep_image_query=not args.strip_image_query,
        )
        all_items.extend(page_items)

    items = deduplicate_items(all_items)

    if not args.no_download:
        items = download_images(
            items=items,
            out_dir=out_dir,
            delay=max(0.0, args.delay),
            retries=max(1, args.retries),
        )

    write_metadata(items, out_dir)

    print()
    print("Готово.")
    print(f"Папка результата: {out_dir}")
    print(f"Найдено уникальных бесплатных карточек: {len(items)}")
    print(f"CSV: {out_dir / 'metadata.csv'}")
    print(f"JSONL: {out_dir / 'metadata.jsonl'}")
    print(f"Изображения: {out_dir / 'images'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
