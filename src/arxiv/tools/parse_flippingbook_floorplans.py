#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
parse_flippingbook_floorplans_debug_v3.py

Отдельный диагностический парсер страниц FlippingBook-журнала и планировок.

Главное отличие от старого parse_flippingbook_floorplans.py:
1. По умолчанию НЕ делает долгий probing тысяч URL.
2. Каждое действие логируется с timestamp.
3. Все HTTP-запросы имеют короткий timeout.
4. Browser discovery не ждёт бесконечный networkidle, а работает через domcontentloaded + ограниченные ожидания.
5. Есть heartbeat-логи, чтобы было видно, где именно выполняется скрипт.
6. Можно работать только с уже скачанными страницами через --input-dir.
7. Все найденные URL, кандидаты, ошибки и стадии пишутся в debug_log.jsonl.
8. Планировки ищутся процедурно: белый фон + тёмные линии + контуры + связная структура + фильтр текста/фото.

Рекомендуемый запуск для твоего случая:
x
    python src/tools/parse_flippingbook_floorplans_debug.py \
      --url "https://houses.ru/magazines/2025/100kk-2025/18/" \
      --pages 1-242 \
      --out data/housesru/houses_100kk_2025_plans_debug \
      --use-browser-discovery \
      --browser-flips 10 \
      --no-probe \
      --preset balanced \
      --debug

Если браузер не находит все 242 страницы, сначала вытащи реальные URL страниц через:

    python src/tools/parse_flippingbook_floorplans_debug.py \
      --url "https://houses.ru/magazines/2025/100kk-2025/18/" \
      --pages 1-242 \
      --out data/housesru/houses_100kk_2025_discovery \
      --use-browser-discovery \
      --browser-flips 80 \
      --no-probe \
      --download-only \
      --debug

Если страницы уже скачаны:

    python src/tools/parse_flippingbook_floorplans_debug.py \
      --input-dir data/housesru/houses_100kk_2025_plans_debug/pages \
      --out data/housesru/houses_100kk_2025_floorplans_from_pages \
      --preset balanced \
      --debug
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import urljoin, urlparse

import cv2
import numpy as np
import requests
from bs4 import BeautifulSoup


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
IMAGE_EXT_RE = re.compile(r"\.(?:jpg|jpeg|png|webp|bmp|tif|tiff)(?:\?|$)", re.IGNORECASE)


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------


class RunLogger:
    def __init__(self, out_dir: Path, verbose: bool = True) -> None:
        self.out_dir = out_dir
        self.verbose = verbose
        self.started_at = time.time()
        self.log_path = out_dir / "debug_log.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def now_rel(self) -> float:
        return time.time() - self.started_at

    def event(self, stage: str, message: str, **data: Any) -> None:
        row = {
            "t_rel_sec": round(self.now_rel(), 3),
            "stage": stage,
            "message": message,
            **data,
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if self.verbose:
            extra = ""
            if data:
                compact = " ".join(f"{k}={v}" for k, v in data.items() if isinstance(v, (str, int, float, bool)))
                extra = f" | {compact}" if compact else ""
            print(f"[{self.now_rel():8.2f}s] [{stage}] {message}{extra}", flush=True)

    def exception(self, stage: str, message: str, exc: BaseException) -> None:
        self.event(
            stage,
            message,
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )


# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PageCandidate:
    page: int
    url: str
    source: str


@dataclasses.dataclass
class DetectionConfig:
    min_area_ratio: float = 0.0006
    max_area_ratio: float = 0.25
    min_width_px: int = 60
    min_height_px: int = 45
    min_aspect: float = 0.20
    max_aspect: float = 5.00
    nms_iou: float = 0.35
    margin: int = 8

    white_gray_thr: int = 198
    white_sat_thr: int = 110
    dark_gray_thr: int = 170
    dark_sat_thr: int = 180

    min_white_ratio: float = 0.55
    min_dark_ratio: float = 0.008
    max_dark_ratio: float = 0.45
    min_edge_ratio: float = 0.018

    min_row_coverage: float = 0.30
    min_col_coverage: float = 0.30
    min_largest_cc_ratio: float = 0.020
    min_orthogonal_line_count: int = 4
    min_orthogonal_ratio: float = 0.35

    max_text_like_score: float = 0.86
    max_photo_like_score: float = 0.82
    min_final_score: float = 1.05

    component_close_base: int = 9
    component_close_large: int = 21
    grid_cell: int = 16


@dataclasses.dataclass(frozen=True)
class CandidateRect:
    x: int
    y: int
    w: int
    h: int
    source_method: str

    @property
    def area(self) -> int:
        return max(0, self.w) * max(0, self.h)

    def expanded(self, image_w: int, image_h: int, margin: int) -> "CandidateRect":
        x0 = max(0, self.x - margin)
        y0 = max(0, self.y - margin)
        x1 = min(image_w, self.x + self.w + margin)
        y1 = min(image_h, self.y + self.h + margin)
        return CandidateRect(x0, y0, x1 - x0, y1 - y0, self.source_method)


@dataclasses.dataclass(frozen=True)
class CropFeatures:
    white_ratio: float
    dark_ratio: float
    edge_ratio: float
    row_coverage: float
    col_coverage: float

    # Connected components.
    raw_largest_cc_ratio: float
    largest_cc_ratio: float
    component_count: int
    small_component_ratio: float
    component_density: float

    # Hough-line structure.
    line_count: int
    orthogonal_line_count: int
    horizontal_line_count: int
    vertical_line_count: int
    diagonal_line_count: int
    orthogonal_ratio: float
    hv_balance: float
    max_line_length_ratio: float

    # Negative evidence.
    colorfulness: float
    gradient_smoothness: float
    text_like_score: float
    photo_like_score: float

    # Final score.
    final_score: float


@dataclasses.dataclass(frozen=True)
class CropResult:
    page: int
    crop_index: int
    page_image: str
    crop_image: str
    x: int
    y: int
    w: int
    h: int
    source_url: Optional[str]
    source_method: str
    features: dict[str, float | int]


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def parse_pages_spec(spec: str) -> list[int]:
    pages: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a_str, b_str = part.split("-", 1)
            a, b = int(a_str), int(b_str)
            if a > b:
                a, b = b, a
            pages.update(range(a, b + 1))
        else:
            pages.add(int(part))
    return sorted(pages)


def normalize_magazine_root(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path
    match = re.search(r"^(.*/magazines/[^/]+/[^/]+/)", path)
    if match:
        base_path = match.group(1)
    else:
        base_path = path if path.endswith("/") else path.rsplit("/", 1)[0] + "/"
    return f"{parsed.scheme}://{parsed.netloc}{base_path}"


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
            "Connection": "close",
        }
    )
    return session


def http_get(
    session: requests.Session,
    url: str,
    *,
    connect_timeout: float,
    read_timeout: float,
    logger: RunLogger,
    stage: str,
) -> Optional[requests.Response]:
    logger.event(stage, "HTTP GET start", url=url, connect_timeout=connect_timeout, read_timeout=read_timeout)
    started = time.time()
    try:
        resp = session.get(url, timeout=(connect_timeout, read_timeout), allow_redirects=True)
        elapsed = time.time() - started
        logger.event(
            stage,
            "HTTP GET done",
            url=url,
            status_code=resp.status_code,
            elapsed_sec=round(elapsed, 3),
            content_type=resp.headers.get("content-type", ""),
            bytes=len(resp.content or b""),
        )
        if resp.status_code == 200:
            return resp
        return None
    except requests.Timeout as exc:
        logger.event(stage, "HTTP GET timeout", url=url, error=str(exc), elapsed_sec=round(time.time() - started, 3))
        return None
    except requests.RequestException as exc:
        logger.event(stage, "HTTP GET error", url=url, error_type=type(exc).__name__, error=str(exc))
        return None


def looks_like_image_response(resp: requests.Response) -> bool:
    ctype = resp.headers.get("content-type", "").lower()
    if "image/" in ctype:
        return True
    content = resp.content[:16]
    return (
        content.startswith(b"\xff\xd8\xff")
        or content.startswith(b"\x89PNG\r\n\x1a\n")
        or (content.startswith(b"RIFF") and b"WEBP" in content[:16])
    )


def infer_page_from_url(url: str) -> Optional[int]:
    """
    Извлекает номер страницы из URL.

    Важно: у houses.ru встречаются URL вида:
        page0021_3.webp
    Старый вариант брал последнее число и получал 3 вместо 21.
    Поэтому сначала ищем pageXXXX, потом vectorlayers/XXXX, и только потом fallback.
    """
    path = urlparse(url).path

    m = re.search(r"/page(?:s)?/page(\d{1,5})(?:_|\.|$)", path, flags=re.IGNORECASE)
    if m:
        value = int(m.group(1))
        return value if value > 0 else None

    m = re.search(r"/page-html5-substrates/page(\d{1,5})(?:_|\.|$)", path, flags=re.IGNORECASE)
    if m:
        value = int(m.group(1))
        return value if value > 0 else None

    m = re.search(r"/page-vectorlayers/(\d{1,5})\.svg", path, flags=re.IGNORECASE)
    if m:
        value = int(m.group(1))
        return value if value > 0 else None

    name = Path(path).stem
    nums = re.findall(r"\d+", name)
    if not nums:
        return None

    # fallback: берём первое значимое число, а не последнее, чтобы page0021_3 -> 21.
    value = int(nums[0])
    return value if value > 0 else None


def choose_best_page_candidates(candidates: list[PageCandidate], logger: RunLogger) -> list[PageCandidate]:
    priority_words = ["large", "page", "pages", "mobile"]

    def priority(c: PageCandidate) -> tuple[int, int, str]:
        lower = c.url.lower()
        word_score = 999
        for i, word in enumerate(priority_words):
            if f"/{word}/" in lower or f"/{word}" in lower:
                word_score = i
                break
        # длинный URL не всегда лучше, но часто high-res лежит глубже
        return (word_score, -len(c.url), c.url)

    by_page: dict[int, list[PageCandidate]] = {}
    for c in candidates:
        by_page.setdefault(c.page, []).append(c)

    chosen: list[PageCandidate] = []
    for page, items in sorted(by_page.items()):
        best = sorted(items, key=priority)[0]
        chosen.append(best)
        logger.event("candidate", "chosen page candidate", page=page, source=best.source, url=best.url, alternatives=len(items))

    return chosen


def candidates_from_discovered_urls(urls: list[str], pages: list[int], logger: RunLogger) -> list[PageCandidate]:
    wanted = set(pages)
    result: list[PageCandidate] = []
    seen: set[tuple[int, str]] = set()

    for url in urls:
        if not IMAGE_EXT_RE.search(url):
            continue
        page = infer_page_from_url(url)
        if page is None:
            logger.event("candidate", "cannot infer page from url", url=url)
            continue
        if page not in wanted:
            continue
        key = (page, url)
        if key in seen:
            continue
        seen.add(key)
        result.append(PageCandidate(page=page, url=url, source="discovered"))

    logger.event("candidate", "page candidates from discovered urls", count=len(result))
    return sorted(result, key=lambda x: (x.page, x.url))



def discover_housesru_direct_templates(
    session: requests.Session,
    root: str,
    pages: list[int],
    *,
    logger: RunLogger,
    connect_timeout: float,
    read_timeout: float,
    max_pages: Optional[int] = None,
) -> list[PageCandidate]:
    """
    Быстрое прямое скачивание по шаблонам, найденным в Network:

        files/assets/common/page-html5-substrates/page0023_3.jpg
        files/assets/common/page-html5-substrates/page0021_3.webp
        files/assets/flash/pages/page0023_w.webp

    Для каждой страницы выбирается самый тяжёлый/качественный доступный URL.
    Это НЕ probing тысяч вариантов: всего несколько известных шаблонов на страницу.
    """
    logger.event("discover:direct", "start", pages=len(pages), root=root)
    result: list[PageCandidate] = []
    limited_pages = pages[:max_pages] if max_pages else pages

    templates = [
        ("html5_jpg", "files/assets/common/page-html5-substrates/page{page:04d}_3.jpg"),
        ("html5_webp", "files/assets/common/page-html5-substrates/page{page:04d}_3.webp"),
        ("flash_webp", "files/assets/flash/pages/page{page:04d}_w.webp"),
    ]

    for idx, page in enumerate(limited_pages, start=1):
        best_url = None
        best_source = None
        best_score = -1

        for source, rel in templates:
            url = urljoin(root, rel.format(page=page))
            resp = http_get(
                session,
                url,
                connect_timeout=connect_timeout,
                read_timeout=read_timeout,
                logger=logger,
                stage="discover:direct:http",
            )
            if resp is None or not looks_like_image_response(resp):
                continue

            size = len(resp.content or b"")
            logger.event("discover:direct", "template ok", page=page, source=source, bytes=size, url=url)

            # html5-substrate обычно крупнее и лучше flash thumbnail.
            source_bonus = 10_000_000 if source.startswith("html5") else 0
            score = source_bonus + size
            if score > best_score:
                best_score = score
                best_url = url
                best_source = "direct_" + source

        if best_url:
            result.append(PageCandidate(page=page, url=best_url, source=best_source or "direct"))
            logger.event("discover:direct", "page chosen", page=page, url=best_url, source=best_source)
        else:
            logger.event("discover:direct", "page not found", page=page)

        if idx % 10 == 0 or idx == len(limited_pages):
            logger.event("discover:direct", "progress", index=idx, total=len(limited_pages), found=len(result))

    logger.event("discover:direct", "done", found=len(result))
    return result


# -----------------------------------------------------------------------------
# Discovery
# -----------------------------------------------------------------------------


def discover_urls_from_html(
    session: requests.Session,
    url: str,
    root: str,
    *,
    logger: RunLogger,
    connect_timeout: float,
    read_timeout: float,
) -> list[str]:
    logger.event("discover:html", "start", url=url)
    resp = http_get(
        session,
        url,
        connect_timeout=connect_timeout,
        read_timeout=read_timeout,
        logger=logger,
        stage="discover:html",
    )
    if resp is None:
        logger.event("discover:html", "no response")
        return []

    text = resp.text
    soup = BeautifulSoup(text, "html.parser")
    urls: set[str] = set()

    for tag in soup.find_all(["img", "source", "script", "link"]):
        for attr in ("src", "data-src", "href"):
            value = tag.get(attr)
            if value and IMAGE_EXT_RE.search(value):
                urls.add(urljoin(url, value))

    for match in re.finditer(r"[\"']([^\"']+\.(?:jpg|jpeg|png|webp)(?:\?[^\"']*)?)[\"']", text, re.IGNORECASE):
        value = match.group(1)
        urls.add(urljoin(root, value))

    result = sorted(urls)
    logger.event("discover:html", "done", image_like_urls=len(result))
    for u in result[:30]:
        logger.event("discover:html:url", "url", url=u)
    if len(result) > 30:
        logger.event("discover:html:url", "truncated url log", remaining=len(result) - 30)

    return result


def discover_urls_with_browser(
    url: str,
    *,
    logger: RunLogger,
    wait_ms: int,
    flips: int,
    flip_wait_ms: int,
    goto_timeout_ms: int,
    browser_timeout_sec: int,
    headful: bool,
    screenshot: bool,
    out_dir: Path,
) -> list[str]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError("Нужен playwright: python -m pip install playwright && python -m playwright install chromium") from exc

    started = time.time()
    urls: set[str] = set()
    page_errors: list[str] = []

    logger.event(
        "discover:browser",
        "start",
        url=url,
        wait_ms=wait_ms,
        flips=flips,
        flip_wait_ms=flip_wait_ms,
        goto_timeout_ms=goto_timeout_ms,
        browser_timeout_sec=browser_timeout_sec,
        headful=headful,
    )

    with sync_playwright() as p:
        logger.event("discover:browser", "launch chromium")
        browser = p.chromium.launch(headless=not headful)
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            ),
        )
        page = context.new_page()
        page.set_default_timeout(10000)

        def on_response(response) -> None:
            try:
                rurl = response.url
                ctype = response.headers.get("content-type", "").lower()
                if "image/" in ctype or IMAGE_EXT_RE.search(rurl):
                    urls.add(rurl)
                    logger.event("discover:browser:response", "image response", url=rurl, content_type=ctype)
            except Exception as exc:
                page_errors.append(f"response handler error: {type(exc).__name__}: {exc}")

        def on_request_failed(request) -> None:
            try:
                failure = request.failure
                logger.event("discover:browser:request_failed", "request failed", url=request.url, failure=str(failure))
            except Exception:
                pass

        page.on("response", on_response)
        page.on("requestfailed", on_request_failed)

        logger.event("discover:browser", "goto domcontentloaded")
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=goto_timeout_ms)
            logger.event("discover:browser", "goto done")
        except Exception as exc:
            logger.event("discover:browser", "goto warning", error_type=type(exc).__name__, error=str(exc))

        logger.event("discover:browser", "initial wait", wait_ms=wait_ms)
        page.wait_for_timeout(wait_ms)

        if screenshot:
            shot_path = out_dir / "browser_initial.png"
            try:
                page.screenshot(path=str(shot_path), full_page=True)
                logger.event("discover:browser", "saved initial screenshot", path=str(shot_path))
            except Exception as exc:
                logger.event("discover:browser", "screenshot failed", error=str(exc))

        # Дополнительный сбор DOM-картинок.
        try:
            dom_urls = page.evaluate(
                r"""
                () => {
                    const out = [];
                    for (const img of document.images) {
                        if (img.currentSrc) out.push(img.currentSrc);
                        if (img.src) out.push(img.src);
                        if (img.dataset && img.dataset.src) out.push(img.dataset.src);
                    }
                    for (const el of document.querySelectorAll('[style]')) {
                        const s = el.getAttribute('style') || '';
                        const m = [...s.matchAll(/url\(["']?([^"')]+)["']?\)/g)];
                        for (const x of m) out.push(x[1]);
                    }
                    return out.filter(u => /\.(jpg|jpeg|png|webp)(\?|$)/i.test(u));
                }
                """
            )
            for u in dom_urls:
                urls.add(urljoin(url, u))
            logger.event("discover:browser", "dom image urls collected", count=len(dom_urls), total=len(urls))
        except Exception as exc:
            logger.event("discover:browser", "dom image collection failed", error_type=type(exc).__name__, error=str(exc))

        for i in range(max(0, flips)):
            if time.time() - started > browser_timeout_sec:
                logger.event("discover:browser", "browser timeout reached, stop flips", elapsed_sec=round(time.time() - started, 3))
                break

            try:
                page.keyboard.press("ArrowRight")
            except Exception as exc:
                logger.event("discover:browser", "ArrowRight failed", index=i + 1, error=str(exc))

            page.wait_for_timeout(flip_wait_ms)

            # На некоторых viewers стрелка не работает без клика по canvas.
            if (i + 1) % 5 == 0:
                try:
                    page.mouse.click(1600, 540)
                    page.wait_for_timeout(150)
                except Exception:
                    pass

            if (i + 1) % 1 == 0:
                logger.event("discover:browser", "flip progress", flip=i + 1, flips=flips, urls=len(urls))

        logger.event("discover:browser", "collect performance entries")
        try:
            perf_urls = page.evaluate(
                r"""
                () => performance.getEntriesByType('resource')
                    .map(e => e.name)
                    .filter(u => /\.(jpg|jpeg|png|webp)(\?|$)/i.test(u))
                """
            )
            for u in perf_urls:
                urls.add(u)
            logger.event("discover:browser", "performance urls collected", count=len(perf_urls), total=len(urls))
        except Exception as exc:
            logger.event("discover:browser", "performance collection failed", error_type=type(exc).__name__, error=str(exc))

        if screenshot:
            shot_path = out_dir / "browser_final.png"
            try:
                page.screenshot(path=str(shot_path), full_page=True)
                logger.event("discover:browser", "saved final screenshot", path=str(shot_path))
            except Exception as exc:
                logger.event("discover:browser", "final screenshot failed", error=str(exc))

        context.close()
        browser.close()

    result = sorted(urls)
    logger.event("discover:browser", "done", image_like_urls=len(result), page_errors=len(page_errors))
    for err in page_errors[:20]:
        logger.event("discover:browser:error", "handler error", error=err)

    urls_path = out_dir / "discovered_image_urls.txt"
    urls_path.write_text("\n".join(result), encoding="utf-8")
    logger.event("discover:browser", "saved discovered urls", path=str(urls_path))

    return result


def candidate_url_patterns(root: str, page: int) -> Iterable[str]:
    nums = [str(page), f"{page:02d}", f"{page:03d}", f"{page:04d}"]
    dirs = [
        "files/mobile/",
        "files/large/",
        "files/page/",
        "files/pages/",
        "files/assets/pages/",
        "mobile/",
        "large/",
        "page/",
        "pages/",
    ]
    prefixes = ["", "page", "Page", "p"]
    exts = ["jpg", "jpeg", "png", "webp"]
    seen: set[str] = set()
    for d in dirs:
        for n in nums:
            for prefix in prefixes:
                stem = f"{prefix}{n}" if prefix else n
                for ext in exts:
                    url = urljoin(root, f"{d}{stem}.{ext}")
                    if url not in seen:
                        seen.add(url)
                        yield url


def discover_page_images_by_probing(
    session: requests.Session,
    root: str,
    pages: list[int],
    *,
    logger: RunLogger,
    connect_timeout: float,
    read_timeout: float,
    max_patterns_per_page: int,
    delay: float,
) -> list[PageCandidate]:
    logger.event(
        "probe",
        "start",
        pages=len(pages),
        connect_timeout=connect_timeout,
        read_timeout=read_timeout,
        max_patterns_per_page=max_patterns_per_page,
    )
    found: list[PageCandidate] = []

    for idx, page in enumerate(pages, start=1):
        page_found = False
        tried = 0

        for url in candidate_url_patterns(root, page):
            tried += 1
            if tried > max_patterns_per_page:
                logger.event("probe", "max patterns reached for page", page=page, tried=tried - 1)
                break

            logger.event("probe", "try url", page=page, try_index=tried, url=url)
            resp = http_get(
                session,
                url,
                connect_timeout=connect_timeout,
                read_timeout=read_timeout,
                logger=logger,
                stage="probe:http",
            )
            if resp is not None and looks_like_image_response(resp):
                found.append(PageCandidate(page=page, url=url, source="probe"))
                logger.event("probe", "found page image", page=page, tries=tried, url=url)
                page_found = True
                break

            if delay > 0:
                time.sleep(delay)

        if not page_found:
            logger.event("probe", "page not found", page=page, tried=tried)

        logger.event("probe", "progress", index=idx, total=len(pages), found=len(found))

    logger.event("probe", "done", found=len(found))
    return found


# -----------------------------------------------------------------------------
# Download / local pages
# -----------------------------------------------------------------------------


def download_pages(
    session: requests.Session,
    candidates: list[PageCandidate],
    out_dir: Path,
    *,
    logger: RunLogger,
    connect_timeout: float,
    read_timeout: float,
) -> list[tuple[int, Path, Optional[str]]]:
    pages_dir = out_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[tuple[int, Path, Optional[str]]] = []

    logger.event("download", "start", candidates=len(candidates))

    for idx, c in enumerate(candidates, start=1):
        logger.event("download", "page start", index=idx, total=len(candidates), page=c.page, url=c.url)
        resp = http_get(
            session,
            c.url,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            logger=logger,
            stage="download:http",
        )
        if resp is None:
            logger.event("download", "skip no response", page=c.page)
            continue
        if not looks_like_image_response(resp):
            logger.event("download", "skip non-image response", page=c.page, content_type=resp.headers.get("content-type", ""))
            continue

        ext = Path(urlparse(c.url).path).suffix.lower() or ".jpg"
        if ext not in IMAGE_EXTENSIONS:
            ext = ".jpg"
        page_path = pages_dir / f"page_{c.page:04d}{ext}"
        page_path.write_bytes(resp.content)
        downloaded.append((c.page, page_path, c.url))
        logger.event("download", "page saved", page=c.page, path=str(page_path), bytes=page_path.stat().st_size)

    logger.event("download", "done", downloaded=len(downloaded))
    return downloaded


def collect_local_pages(input_dir: Path, logger: RunLogger) -> list[tuple[int, Path, Optional[str]]]:
    paths = sorted(p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    result: list[tuple[int, Path, Optional[str]]] = []

    for idx, path in enumerate(paths, start=1):
        nums = re.findall(r"\d+", path.stem)
        page = int(nums[-1]) if nums else idx
        result.append((page, path, None))

    logger.event("input", "local pages collected", input_dir=str(input_dir), count=len(result))
    return sorted(result, key=lambda x: (x[0], str(x[1])))


# -----------------------------------------------------------------------------
# Image detection
# -----------------------------------------------------------------------------


def imread_bgr(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Не удалось прочитать изображение: {path}")
    return img


def write_image(path: Path, image_bgr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext not in IMAGE_EXTENSIONS:
        ext = ".png"
    ok, encoded = cv2.imencode(ext, image_bgr)
    if not ok:
        raise RuntimeError(f"Не удалось записать изображение: {path}")
    encoded.tofile(str(path))


def rect_iou(a: CandidateRect, b: CandidateRect) -> float:
    ax1, ay1, ax2, ay2 = a.x, a.y, a.x + a.w, a.y + a.h
    bx1, by1, bx2, by2 = b.x, b.y, b.x + b.w, b.y + b.h
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def filter_rect_geometry(rects: list[CandidateRect], image_w: int, image_h: int, cfg: DetectionConfig) -> list[CandidateRect]:
    page_area = image_w * image_h
    result: list[CandidateRect] = []
    seen: set[tuple[int, int, int, int]] = set()

    for r in rects:
        if r.w <= 0 or r.h <= 0:
            continue
        if r.w < cfg.min_width_px or r.h < cfg.min_height_px:
            continue
        area_ratio = r.area / max(1, page_area)
        if area_ratio < cfg.min_area_ratio or area_ratio > cfg.max_area_ratio:
            continue
        aspect = r.w / max(1, r.h)
        if aspect < cfg.min_aspect or aspect > cfg.max_aspect:
            continue
        key = (r.x // 3, r.y // 3, r.w // 3, r.h // 3)
        if key in seen:
            continue
        seen.add(key)
        result.append(r)

    return result


def merge_overlapping_rects(rects: list[CandidateRect], image_w: int, image_h: int, iou_thr: float = 0.65) -> list[CandidateRect]:
    result: list[CandidateRect] = []
    for rect in sorted(rects, key=lambda r: r.area, reverse=True):
        merged = False
        for i, existing in enumerate(result):
            if rect_iou(rect, existing) >= iou_thr:
                x0 = min(rect.x, existing.x)
                y0 = min(rect.y, existing.y)
                x1 = max(rect.x + rect.w, existing.x + existing.w)
                y1 = max(rect.y + rect.h, existing.y + existing.h)
                result[i] = CandidateRect(
                    x=max(0, x0),
                    y=max(0, y0),
                    w=min(image_w, x1) - max(0, x0),
                    h=min(image_h, y1) - max(0, y0),
                    source_method=existing.source_method + "+merge",
                )
                merged = True
                break
        if not merged:
            result.append(rect)
    return result


def build_basic_masks(image_bgr: np.ndarray, cfg: DetectionConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    white_mask = ((gray >= cfg.white_gray_thr) & (sat <= cfg.white_sat_thr)).astype(np.uint8)
    dark_mask = ((gray <= cfg.dark_gray_thr) & (sat <= cfg.dark_sat_thr)).astype(np.uint8)
    edges = cv2.Canny(gray, 60, 160)
    dark_or_edges = dark_mask | ((edges > 0).astype(np.uint8) & (sat <= cfg.dark_sat_thr).astype(np.uint8))
    return gray, hsv, sat, white_mask, dark_or_edges


def candidates_from_connected_dark_components(dark_mask: np.ndarray, cfg: DetectionConfig, image_w: int, image_h: int) -> list[CandidateRect]:
    rects: list[CandidateRect] = []

    variants = [
        (cfg.component_close_base, 2, "dark_cc_base"),
        (cfg.component_close_large, 1, "dark_cc_large"),
        (max(5, cfg.component_close_base // 2), 3, "dark_cc_small_iter"),
    ]

    for kernel_size, iterations, method_name in variants:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(dark_mask * 255, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            rects.append(CandidateRect(x, y, w, h, method_name).expanded(image_w, image_h, cfg.margin))

    return rects


def candidates_from_white_regions(white_mask: np.ndarray, dark_mask: np.ndarray, cfg: DetectionConfig, image_w: int, image_h: int) -> list[CandidateRect]:
    rects: list[CandidateRect] = []
    mask = cv2.morphologyEx(white_mask * 255, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)), iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < cfg.min_width_px or h < cfg.min_height_px:
            continue
        crop_dark = dark_mask[y : y + h, x : x + w]
        dark_ratio = np.count_nonzero(crop_dark) / max(1, w * h)
        if dark_ratio >= cfg.min_dark_ratio:
            rects.append(CandidateRect(x, y, w, h, "white_region"))

    return rects


def candidates_from_grid(white_mask: np.ndarray, dark_mask: np.ndarray, cfg: DetectionConfig, image_w: int, image_h: int) -> list[CandidateRect]:
    cell = cfg.grid_cell
    gh = image_h // cell
    gw = image_w // cell
    if gh <= 0 or gw <= 0:
        return []

    good = np.zeros((gh, gw), dtype=np.uint8)
    for gy in range(gh):
        y0 = gy * cell
        y1 = y0 + cell
        for gx in range(gw):
            x0 = gx * cell
            x1 = x0 + cell
            area = cell * cell
            wr = np.count_nonzero(white_mask[y0:y1, x0:x1]) / area
            dr = np.count_nonzero(dark_mask[y0:y1, x0:x1]) / area
            if wr >= max(0.45, cfg.min_white_ratio - 0.15) and dr >= max(0.002, cfg.min_dark_ratio * 0.4):
                good[gy, gx] = 255

    good = cv2.morphologyEx(good, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    contours, _ = cv2.findContours(good, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects: list[CandidateRect] = []
    for cnt in contours:
        gx, gy, gw2, gh2 = cv2.boundingRect(cnt)
        if gw2 * gh2 < 8:
            continue
        x = gx * cell
        y = gy * cell
        w = min(image_w - x, gw2 * cell)
        h = min(image_h - y, gh2 * cell)
        rects.append(CandidateRect(x, y, w, h, "grid_white_dark").expanded(image_w, image_h, cfg.margin))

    return rects


def estimate_colorfulness(crop_bgr: np.ndarray) -> float:
    b, g, r = cv2.split(crop_bgr.astype(np.float32))
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)
    std_root = math.sqrt(float(np.var(rg) + np.var(yb)))
    mean_root = math.sqrt(float(np.mean(rg) ** 2 + np.mean(yb) ** 2))
    raw = std_root + 0.3 * mean_root
    return float(min(1.0, raw / 120.0))


def estimate_gradient_smoothness(gray: np.ndarray) -> float:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    if mag.size == 0:
        return 0.0
    low = float(np.mean((mag > 3) & (mag < 35)))
    return min(1.0, low * 2.0)


def hough_line_features(crop_gray: np.ndarray) -> tuple[int, int, int, int, int, float, float, float]:
    """
    Возвращает:
        total, orthogonal, horizontal, vertical, diagonal,
        orthogonal_ratio, hv_balance, max_line_length_ratio.

    Главный смысл: текстовые блоки дают много коротких горизонтальных линий,
    а планировка даёт и горизонтальные, и вертикальные длинные сегменты.
    """
    if crop_gray.shape[0] < 20 or crop_gray.shape[1] < 20:
        return 0, 0, 0, 0, 0, 0.0, 0.0, 0.0

    edges = cv2.Canny(crop_gray, 60, 160)
    min_len = max(16, min(crop_gray.shape[:2]) // 7)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=22,
        minLineLength=min_len,
        maxLineGap=5,
    )
    if lines is None:
        return 0, 0, 0, 0, 0, 0.0, 0.0, 0.0

    total = 0
    orth = 0
    horiz = 0
    vert = 0
    diag = 0
    max_len = 0.0
    diag_norm = max(1.0, math.hypot(crop_gray.shape[1], crop_gray.shape[0]))

    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = map(float, line)
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length < min_len:
            continue

        total += 1
        max_len = max(max_len, length)
        angle = abs(math.degrees(math.atan2(dy, dx))) % 180.0

        dist_to_0 = min(abs(angle - 0), abs(angle - 180))
        dist_to_90 = abs(angle - 90)

        if dist_to_0 <= 12.0:
            horiz += 1
            orth += 1
        elif dist_to_90 <= 12.0:
            vert += 1
            orth += 1
        else:
            diag += 1

    orth_ratio = orth / total if total > 0 else 0.0
    hv_balance = min(horiz, vert) / max(1, max(horiz, vert))
    max_line_length_ratio = max_len / diag_norm
    return total, orth, horiz, vert, diag, float(orth_ratio), float(hv_balance), float(max_line_length_ratio)


def largest_connected_component_ratio(binary_mask: np.ndarray) -> float:
    if binary_mask.size == 0:
        return 0.0
    num_labels, _labels, stats, _ = cv2.connectedComponentsWithStats((binary_mask > 0).astype(np.uint8) * 255, connectivity=8)
    if num_labels <= 1:
        return 0.0
    largest = int(np.max(stats[1:, cv2.CC_STAT_AREA]))
    return largest / float(binary_mask.shape[0] * binary_mask.shape[1])


def connected_component_shape_stats(binary_mask: np.ndarray) -> tuple[float, int, float, float]:
    """
    Статистика по сырым connected components:
        raw_largest_cc_ratio,
        component_count,
        small_component_ratio,
        component_density.

    Для текста обычно:
        - много мелких компонент;
        - высокая доля small components;
        - низкая площадь крупнейшей сырой компоненты.

    Для планировки обычно:
        - есть компоненты стен/контуров;
        - после close структура становится связной;
        - одновременно есть длинные H/V линии.
    """
    if binary_mask.size == 0:
        return 0.0, 0, 0.0, 0.0

    mask = (binary_mask > 0).astype(np.uint8) * 255
    num_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return 0.0, 0, 0.0, 0.0

    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
    img_area = float(binary_mask.shape[0] * binary_mask.shape[1])
    component_count = int(len(areas))
    raw_largest = float(np.max(areas) / max(1.0, img_area))

    # Мелкие компоненты: буквы, точки, элементы логотипов. Порог относительный,
    # чтобы работать и на больших страницах, и на миниатюрах.
    small_thr = max(3.0, img_area * 0.0008)
    small_component_ratio = float(np.mean(areas <= small_thr)) if component_count else 0.0

    # Нормированная плотность компонент. 1.0 означает "очень много компонент".
    component_density = float(min(1.0, component_count / max(1.0, img_area / 900.0)))
    return raw_largest, component_count, small_component_ratio, component_density


def estimate_text_like_score(
    crop_gray: np.ndarray,
    crop_dark: np.ndarray,
    row_coverage: float,
    col_coverage: float,
    largest_cc: float,
    raw_largest_cc: float,
    component_count: int,
    small_component_ratio: float,
    component_density: float,
    horizontal_lines: int,
    vertical_lines: int,
    hv_balance: float,
    max_line_length_ratio: float,
) -> float:
    h, w = crop_gray.shape[:2]
    if h <= 0 or w <= 0:
        return 1.0

    row_dark_counts = np.sum(crop_dark > 0, axis=1)
    active_rows = row_dark_counts > max(1, int(0.01 * w))
    transitions = int(np.count_nonzero(active_rows[1:] != active_rows[:-1])) if h > 1 else 0
    line_band_score = min(1.0, transitions / max(1.0, h / 6.0))

    # Текст: много мелких компонент и регулярные горизонтальные строки.
    many_small_components = 0.55 * small_component_ratio + 0.45 * component_density

    # Текст обычно горизонтально доминирует. Планировка должна иметь заметные vertical lines.
    horizontal_dominance = horizontal_lines / max(1, horizontal_lines + vertical_lines)
    no_vertical_structure = 1.0 - min(1.0, vertical_lines / 3.0)

    weak_raw_structure = max(0.0, 1.0 - raw_largest_cc / 0.035)
    weak_closed_structure = max(0.0, 1.0 - largest_cc / 0.12)

    narrow_column = 1.0 if (w / max(1, h) < 0.75 or h / max(1, w) < 0.75) else 0.25
    short_lines = max(0.0, 1.0 - max_line_length_ratio / 0.22)

    score = (
        0.20 * line_band_score
        + 0.26 * many_small_components
        + 0.18 * weak_raw_structure
        + 0.12 * weak_closed_structure
        + 0.10 * no_vertical_structure
        + 0.06 * horizontal_dominance
        + 0.04 * narrow_column
        + 0.04 * short_lines
    )

    # Если есть сбалансированные H/V линии и длинные линии, это больше похоже на план.
    if vertical_lines >= 2 and horizontal_lines >= 2 and hv_balance >= 0.35 and max_line_length_ratio >= 0.18:
        score *= 0.50

    if row_coverage > 0.75 and col_coverage > 0.65 and largest_cc > 0.10 and hv_balance >= 0.35:
        score *= 0.60

    return float(min(1.0, max(0.0, score)))


def compute_crop_features(
    image_bgr: np.ndarray,
    rect: CandidateRect,
    masks: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> CropFeatures:
    gray, _hsv, _sat, white_mask, dark_mask = masks
    x, y, w, h = rect.x, rect.y, rect.w, rect.h
    area = max(1, w * h)

    crop_bgr = image_bgr[y : y + h, x : x + w]
    crop_gray = gray[y : y + h, x : x + w]
    crop_white = white_mask[y : y + h, x : x + w]
    crop_dark = dark_mask[y : y + h, x : x + w]

    white_ratio = float(np.count_nonzero(crop_white) / area)
    dark_ratio = float(np.count_nonzero(crop_dark) / area)

    edges = cv2.Canny(crop_gray, 60, 160)
    edge_ratio = float(np.count_nonzero(edges) / area)

    row_coverage = float(np.mean(np.sum(crop_dark, axis=1) > 0)) if h > 0 else 0.0
    col_coverage = float(np.mean(np.sum(crop_dark, axis=0) > 0)) if w > 0 else 0.0

    raw_largest_cc, component_count, small_component_ratio, component_density = connected_component_shape_stats(crop_dark)

    cc_mask = cv2.morphologyEx(
        crop_dark.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=1,
    )
    largest_cc = float(largest_connected_component_ratio(cc_mask))

    (
        line_count,
        orth_count,
        horizontal_count,
        vertical_count,
        diagonal_count,
        orth_ratio,
        hv_balance,
        max_line_length_ratio,
    ) = hough_line_features(crop_gray)

    colorfulness = estimate_colorfulness(crop_bgr)
    gradient_smoothness = estimate_gradient_smoothness(crop_gray)

    photo_like_score = float(min(1.0, 0.65 * colorfulness + 0.35 * gradient_smoothness))
    text_like_score = estimate_text_like_score(
        crop_gray,
        crop_dark,
        row_coverage,
        col_coverage,
        largest_cc,
        raw_largest_cc,
        component_count,
        small_component_ratio,
        component_density,
        horizontal_count,
        vertical_count,
        hv_balance,
        max_line_length_ratio,
    )

    plan_line_score = (
        0.45 * min(horizontal_count / 4.0, 1.0)
        + 0.45 * min(vertical_count / 4.0, 1.0)
        + 0.35 * hv_balance
        + 0.35 * min(max_line_length_ratio / 0.30, 1.0)
    )

    final_score = (
        0.70 * white_ratio
        + 1.70 * min(dark_ratio / 0.18, 1.0)
        + 1.40 * min(edge_ratio / 0.16, 1.0)
        + 1.20 * min(largest_cc / 0.18, 1.0)
        + 0.70 * min((row_coverage + col_coverage) / 1.6, 1.0)
        + 1.60 * plan_line_score
        + 0.20 * orth_ratio
        - 1.85 * text_like_score
        - 1.15 * photo_like_score
    )

    return CropFeatures(
        white_ratio=white_ratio,
        dark_ratio=dark_ratio,
        edge_ratio=edge_ratio,
        row_coverage=row_coverage,
        col_coverage=col_coverage,
        raw_largest_cc_ratio=float(raw_largest_cc),
        largest_cc_ratio=largest_cc,
        component_count=int(component_count),
        small_component_ratio=float(small_component_ratio),
        component_density=float(component_density),
        line_count=int(line_count),
        orthogonal_line_count=int(orth_count),
        horizontal_line_count=int(horizontal_count),
        vertical_line_count=int(vertical_count),
        diagonal_line_count=int(diagonal_count),
        orthogonal_ratio=float(orth_ratio),
        hv_balance=float(hv_balance),
        max_line_length_ratio=float(max_line_length_ratio),
        colorfulness=float(colorfulness),
        gradient_smoothness=float(gradient_smoothness),
        text_like_score=float(text_like_score),
        photo_like_score=float(photo_like_score),
        final_score=float(final_score),
    )


def is_floorplan_candidate(features: CropFeatures, cfg: DetectionConfig) -> bool:
    if features.white_ratio < cfg.min_white_ratio:
        return False
    if not (cfg.min_dark_ratio <= features.dark_ratio <= cfg.max_dark_ratio):
        return False
    if features.edge_ratio < cfg.min_edge_ratio:
        return False
    if features.row_coverage < cfg.min_row_coverage:
        return False
    if features.col_coverage < cfg.min_col_coverage:
        return False
    if features.largest_cc_ratio < cfg.min_largest_cc_ratio:
        return False

    # Новый ключевой фильтр против текстовых блоков и логотипов:
    # у планировки должны быть и горизонтальные, и вертикальные линии.
    has_balanced_plan_lines = (
        features.horizontal_line_count >= 2
        and features.vertical_line_count >= 2
        and features.hv_balance >= 0.22
        and features.max_line_length_ratio >= 0.12
    )
    has_strong_connected_plan = (
        features.largest_cc_ratio >= 0.16
        and features.horizontal_line_count >= 2
        and features.vertical_line_count >= 1
        and features.max_line_length_ratio >= 0.18
    )
    if not (has_balanced_plan_lines or has_strong_connected_plan):
        return False

    if features.orthogonal_line_count < cfg.min_orthogonal_line_count and features.orthogonal_ratio < cfg.min_orthogonal_ratio:
        return False

    # Текстовые блоки: много мелких компонент, слабая вертикальная структура.
    if (
        features.small_component_ratio > 0.70
        and features.component_density > 0.45
        and features.vertical_line_count < 2
    ):
        return False

    if features.text_like_score > cfg.max_text_like_score:
        return False
    if features.photo_like_score > cfg.max_photo_like_score:
        return False
    if features.final_score < cfg.min_final_score:
        return False
    return True


def generate_candidate_rects(
    image_bgr: np.ndarray,
    cfg: DetectionConfig,
    logger: RunLogger,
    page_num: int,
) -> tuple[list[CandidateRect], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    image_h, image_w = image_bgr.shape[:2]
    masks = build_basic_masks(image_bgr, cfg)
    _gray, _hsv, _sat, white_mask, dark_mask = masks

    rects: list[CandidateRect] = []
    a = candidates_from_connected_dark_components(dark_mask, cfg, image_w, image_h)
    b = candidates_from_white_regions(white_mask, dark_mask, cfg, image_w, image_h)
    c = candidates_from_grid(white_mask, dark_mask, cfg, image_w, image_h)

    rects.extend(a)
    rects.extend(b)
    rects.extend(c)

    logger.event(
        "detect",
        "raw candidate rects",
        page=page_num,
        dark_cc=len(a),
        white_regions=len(b),
        grid=len(c),
        total=len(rects),
    )

    rects = filter_rect_geometry(rects, image_w, image_h, cfg)
    logger.event("detect", "after geometry filter", page=page_num, count=len(rects))
    rects = merge_overlapping_rects(rects, image_w, image_h, iou_thr=0.65)
    rects = filter_rect_geometry(rects, image_w, image_h, cfg)
    logger.event("detect", "after merge", page=page_num, count=len(rects))

    return rects, masks


def nms_detections(items: list[tuple[CandidateRect, CropFeatures]], iou_thr: float) -> list[tuple[CandidateRect, CropFeatures]]:
    sorted_items = sorted(items, key=lambda item: item[1].final_score, reverse=True)
    kept: list[tuple[CandidateRect, CropFeatures]] = []
    for rect, feat in sorted_items:
        if all(rect_iou(rect, kept_rect) < iou_thr for kept_rect, _ in kept):
            kept.append((rect, feat))
    return kept


def detect_floorplans(
    image_bgr: np.ndarray,
    cfg: DetectionConfig,
    logger: RunLogger,
    page_num: int,
    save_rejects: bool,
    rejects_dir: Path,
) -> list[tuple[CandidateRect, CropFeatures]]:
    rects, masks = generate_candidate_rects(image_bgr, cfg, logger, page_num)
    accepted: list[tuple[CandidateRect, CropFeatures]] = []

    for idx, rect in enumerate(rects, start=1):
        features = compute_crop_features(image_bgr, rect, masks)
        ok = is_floorplan_candidate(features, cfg)

        logger.event(
            "detect:candidate",
            "candidate evaluated",
            page=page_num,
            index=idx,
            ok=ok,
            x=rect.x,
            y=rect.y,
            w=rect.w,
            h=rect.h,
            source=rect.source_method,
            score=round(features.final_score, 4),
            white=round(features.white_ratio, 4),
            dark=round(features.dark_ratio, 4),
            edge=round(features.edge_ratio, 4),
            cc=round(features.largest_cc_ratio, 4),
            lines=features.orthogonal_line_count,
            h_lines=features.horizontal_line_count,
            v_lines=features.vertical_line_count,
            hv=round(features.hv_balance, 4),
            max_line=round(features.max_line_length_ratio, 4),
            raw_cc=round(features.raw_largest_cc_ratio, 4),
            small_cc=round(features.small_component_ratio, 4),
            text=round(features.text_like_score, 4),
            photo=round(features.photo_like_score, 4),
        )

        if ok:
            accepted.append((rect, features))
        elif save_rejects:
            crop = image_bgr[rect.y : rect.y + rect.h, rect.x : rect.x + rect.w]
            reject_path = rejects_dir / f"page_{page_num:04d}_reject_{idx:03d}_score_{features.final_score:.2f}.png"
            write_image(reject_path, crop)

    accepted = nms_detections(accepted, cfg.nms_iou)
    logger.event("detect", "accepted after nms", page=page_num, count=len(accepted))
    return accepted


def crop_features_to_dict(features: CropFeatures) -> dict[str, float | int]:
    return {
        "white_ratio": features.white_ratio,
        "dark_ratio": features.dark_ratio,
        "edge_ratio": features.edge_ratio,
        "row_coverage": features.row_coverage,
        "col_coverage": features.col_coverage,
        "raw_largest_cc_ratio": features.raw_largest_cc_ratio,
        "largest_cc_ratio": features.largest_cc_ratio,
        "component_count": features.component_count,
        "small_component_ratio": features.small_component_ratio,
        "component_density": features.component_density,
        "line_count": features.line_count,
        "orthogonal_line_count": features.orthogonal_line_count,
        "horizontal_line_count": features.horizontal_line_count,
        "vertical_line_count": features.vertical_line_count,
        "diagonal_line_count": features.diagonal_line_count,
        "orthogonal_ratio": features.orthogonal_ratio,
        "hv_balance": features.hv_balance,
        "max_line_length_ratio": features.max_line_length_ratio,
        "colorfulness": features.colorfulness,
        "gradient_smoothness": features.gradient_smoothness,
        "text_like_score": features.text_like_score,
        "photo_like_score": features.photo_like_score,
        "final_score": features.final_score,
    }


def draw_debug(image_bgr: np.ndarray, detections: list[tuple[CandidateRect, CropFeatures]], out_path: Path) -> None:
    debug = image_bgr.copy()
    for i, (rect, feat) in enumerate(detections, start=1):
        x, y, w, h = rect.x, rect.y, rect.w, rect.h
        cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 0, 255), 3)
        label = f"#{i} score={feat.final_score:.2f} W={feat.white_ratio:.2f} D={feat.dark_ratio:.2f} CC={feat.largest_cc_ratio:.2f}"
        cv2.putText(debug, label, (x, max(25, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    write_image(out_path, debug)


def process_pages(
    pages: list[tuple[int, Path, Optional[str]]],
    out_dir: Path,
    cfg: DetectionConfig,
    *,
    logger: RunLogger,
    debug: bool,
    save_rejects: bool,
) -> list[CropResult]:
    crops_dir = out_dir / "floorplans"
    debug_dir = out_dir / "debug"
    rejects_dir = out_dir / "rejects"
    results: list[CropResult] = []

    logger.event("process", "start", pages=len(pages))

    for idx, (page_num, page_path, source_url) in enumerate(pages, start=1):
        logger.event("process", "page start", index=idx, total=len(pages), page=page_num, path=str(page_path))
        try:
            image = imread_bgr(page_path)
        except Exception as exc:
            logger.exception("process", "cannot read page", exc)
            continue

        logger.event("process", "page image loaded", page=page_num, width=image.shape[1], height=image.shape[0])

        try:
            detections = detect_floorplans(
                image,
                cfg,
                logger,
                page_num,
                save_rejects=save_rejects,
                rejects_dir=rejects_dir,
            )
        except Exception as exc:
            logger.exception("process", "detection failed", exc)
            continue

        if debug:
            debug_path = debug_dir / f"page_{page_num:04d}_debug.jpg"
            draw_debug(image, detections, debug_path)
            logger.event("process", "debug image saved", page=page_num, path=str(debug_path))

        for crop_idx, (rect, features) in enumerate(detections, start=1):
            crop = image[rect.y : rect.y + rect.h, rect.x : rect.x + rect.w]
            crop_path = crops_dir / f"page_{page_num:04d}_plan_{crop_idx:02d}.png"
            write_image(crop_path, crop)

            result = CropResult(
                page=page_num,
                crop_index=crop_idx,
                page_image=str(page_path),
                crop_image=str(crop_path),
                x=rect.x,
                y=rect.y,
                w=rect.w,
                h=rect.h,
                source_url=source_url,
                source_method=rect.source_method,
                features=crop_features_to_dict(features),
            )
            results.append(result)
            logger.event(
                "process",
                "crop saved",
                page=page_num,
                crop_index=crop_idx,
                path=str(crop_path),
                score=round(features.final_score, 4),
            )

    logger.event("process", "done", crops=len(results))
    return results


# -----------------------------------------------------------------------------
# Config / manifest
# -----------------------------------------------------------------------------


def config_from_preset(preset: str) -> DetectionConfig:
    cfg = DetectionConfig()

    if preset == "balanced":
        return cfg

    if preset == "recall":
        cfg.min_area_ratio = 0.00030
        cfg.max_area_ratio = 0.35
        cfg.min_white_ratio = 0.45
        cfg.min_dark_ratio = 0.004
        cfg.min_edge_ratio = 0.010
        cfg.min_row_coverage = 0.20
        cfg.min_col_coverage = 0.20
        cfg.min_largest_cc_ratio = 0.010
        cfg.min_orthogonal_line_count = 2
        cfg.min_orthogonal_ratio = 0.25
        cfg.max_text_like_score = 0.92
        cfg.max_photo_like_score = 0.90
        cfg.min_final_score = 0.65
        return cfg

    if preset == "precision":
        cfg.min_area_ratio = 0.0010
        cfg.max_area_ratio = 0.18
        cfg.min_white_ratio = 0.62
        cfg.min_dark_ratio = 0.018
        cfg.min_edge_ratio = 0.032
        cfg.min_row_coverage = 0.45
        cfg.min_col_coverage = 0.45
        cfg.min_largest_cc_ratio = 0.045
        cfg.min_orthogonal_line_count = 10
        cfg.min_orthogonal_ratio = 0.50
        cfg.max_text_like_score = 0.68
        cfg.max_photo_like_score = 0.62
        cfg.min_final_score = 1.85
        return cfg

    raise ValueError(f"Unknown preset: {preset}")


def save_manifest(
    out_dir: Path,
    cfg: DetectionConfig,
    pages: list[tuple[int, Path, Optional[str]]],
    crops: list[CropResult],
    page_candidates: list[PageCandidate],
) -> None:
    manifest = {
        "schema": "flippingbook_floorplan_debug_parser/v1",
        "config": dataclasses.asdict(cfg),
        "pages": [
            {"page": page_num, "path": str(page_path), "source_url": source_url}
            for page_num, page_path, source_url in pages
        ],
        "page_candidates": [dataclasses.asdict(c) for c in page_candidates],
        "floorplan_crops": [dataclasses.asdict(crop) for crop in crops],
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Debug FlippingBook floorplan parser with detailed logs.")
    parser.add_argument("--url", help="URL журнала")
    parser.add_argument("--input-dir", help="Папка с уже скачанными страницами")
    parser.add_argument("--out", required=True, help="Выходная папка")
    parser.add_argument("--pages", default="1-242")
    parser.add_argument("--use-browser-discovery", action="store_true")
    parser.add_argument("--use-direct-template-discovery", action="store_true", help="Быстро проверить известные шаблоны houses.ru page-html5-substrates/flash pages")
    parser.add_argument("--browser-flips", type=int, default=10)
    parser.add_argument("--browser-wait-ms", type=int, default=2500)
    parser.add_argument("--browser-flip-wait-ms", type=int, default=350)
    parser.add_argument("--browser-goto-timeout-ms", type=int, default=30000)
    parser.add_argument("--browser-timeout-sec", type=int, default=120)
    parser.add_argument("--browser-headful", action="store_true", help="Открыть видимый Chromium")
    parser.add_argument("--browser-screenshot", action="store_true", help="Сохранить browser_initial.png и browser_final.png")
    parser.add_argument("--no-probe", action="store_true")
    parser.add_argument("--probe-max-patterns-per-page", type=int, default=12)
    parser.add_argument("--connect-timeout", type=float, default=3.0)
    parser.add_argument("--read-timeout", type=float, default=4.0)
    parser.add_argument("--download-connect-timeout", type=float, default=5.0)
    parser.add_argument("--download-read-timeout", type=float, default=20.0)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--save-rejects", action="store_true")
    parser.add_argument("--preset", choices=["balanced", "recall", "precision"], default="balanced")
    parser.add_argument("--quiet", action="store_true")

    # Тонкая настройка детектора.
    parser.add_argument("--min-area-ratio", type=float)
    parser.add_argument("--max-area-ratio", type=float)
    parser.add_argument("--min-white-ratio", type=float)
    parser.add_argument("--min-dark-ratio", type=float)
    parser.add_argument("--min-edge-ratio", type=float)
    parser.add_argument("--min-final-score", type=float)
    parser.add_argument("--margin", type=int)

    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = RunLogger(out_dir, verbose=not args.quiet)
    logger.event("main", "start", argv=" ".join(sys.argv), cwd=os.getcwd())

    if not args.url and not args.input_dir:
        logger.event("main", "error: need --url or --input-dir")
        return 2

    cfg = config_from_preset(args.preset)
    if args.min_area_ratio is not None:
        cfg.min_area_ratio = args.min_area_ratio
    if args.max_area_ratio is not None:
        cfg.max_area_ratio = args.max_area_ratio
    if args.min_white_ratio is not None:
        cfg.min_white_ratio = args.min_white_ratio
    if args.min_dark_ratio is not None:
        cfg.min_dark_ratio = args.min_dark_ratio
    if args.min_edge_ratio is not None:
        cfg.min_edge_ratio = args.min_edge_ratio
    if args.min_final_score is not None:
        cfg.min_final_score = args.min_final_score
    if args.margin is not None:
        cfg.margin = args.margin

    logger.event("main", "config ready", preset=args.preset, config=json.dumps(dataclasses.asdict(cfg), ensure_ascii=False))

    page_candidates: list[PageCandidate] = []

    if args.input_dir:
        pages = collect_local_pages(Path(args.input_dir).expanduser().resolve(), logger)
    else:
        assert args.url is not None
        session = make_session()
        page_numbers = parse_pages_spec(args.pages)
        root = normalize_magazine_root(args.url)

        logger.event("main", "root and pages parsed", root=root, first_page=page_numbers[0], last_page=page_numbers[-1], pages=len(page_numbers))

        discovered_urls: list[str] = []
        discovered_urls.extend(
            discover_urls_from_html(
                session,
                args.url,
                root,
                logger=logger,
                connect_timeout=args.connect_timeout,
                read_timeout=args.read_timeout,
            )
        )
        logger.event("main", "after html discovery", urls=len(set(discovered_urls)))

        direct_candidates: list[PageCandidate] = []
        if args.use_direct_template_discovery:
            direct_candidates = discover_housesru_direct_templates(
                session,
                root,
                page_numbers,
                logger=logger,
                connect_timeout=args.connect_timeout,
                read_timeout=args.read_timeout,
            )
            logger.event("main", "after direct template discovery", candidates=len(direct_candidates))

        if args.use_browser_discovery:
            try:
                browser_urls = discover_urls_with_browser(
                    args.url,
                    logger=logger,
                    wait_ms=args.browser_wait_ms,
                    flips=args.browser_flips,
                    flip_wait_ms=args.browser_flip_wait_ms,
                    goto_timeout_ms=args.browser_goto_timeout_ms,
                    browser_timeout_sec=args.browser_timeout_sec,
                    headful=args.browser_headful,
                    screenshot=args.browser_screenshot,
                    out_dir=out_dir,
                )
                discovered_urls.extend(browser_urls)
            except Exception as exc:
                logger.exception("main", "browser discovery failed", exc)
            logger.event("main", "after browser discovery", urls=len(set(discovered_urls)))

        discovered_candidates = candidates_from_discovered_urls(sorted(set(discovered_urls)), page_numbers, logger)
        discovered_candidates.extend(direct_candidates)

        probed_candidates: list[PageCandidate] = []
        if not args.no_probe:
            logger.event("main", "probe enabled")
            probed_candidates = discover_page_images_by_probing(
                session,
                root,
                page_numbers,
                logger=logger,
                connect_timeout=args.connect_timeout,
                read_timeout=args.read_timeout,
                max_patterns_per_page=args.probe_max_patterns_per_page,
                delay=0.0,
            )
        else:
            logger.event("main", "probe skipped because --no-probe is set")

        page_candidates = choose_best_page_candidates(discovered_candidates + probed_candidates, logger)
        logger.event("main", "chosen page candidates", count=len(page_candidates))

        if not page_candidates:
            logger.event("main", "no page candidates found")
            save_manifest(out_dir, cfg, [], [], [])
            return 2

        pages = download_pages(
            session,
            page_candidates,
            out_dir,
            logger=logger,
            connect_timeout=args.download_connect_timeout,
            read_timeout=args.download_read_timeout,
        )

    if not pages:
        logger.event("main", "no pages to process")
        save_manifest(out_dir, cfg, [], [], page_candidates)
        return 2

    crops: list[CropResult] = []
    if not args.download_only:
        crops = process_pages(
            pages,
            out_dir,
            cfg,
            logger=logger,
            debug=args.debug,
            save_rejects=args.save_rejects,
        )
    else:
        logger.event("main", "download-only mode, skip detection")

    save_manifest(out_dir, cfg, pages, crops, page_candidates)
    logger.event("main", "done", manifest=str(out_dir / "manifest.json"), crops=len(crops), pages=len(pages))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
