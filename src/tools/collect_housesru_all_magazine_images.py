#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collect_housesru_all_magazine_images.py

Задача
=====
Сначала собрать все изображения страниц журналов HOUSES.RU за все доступные годы,
без фильтрации планировок.

Скрипт:
1. Находит root-URL журналов вида:
      https://houses.ru/magazines/2025/100kk-2025/
2. Для каждого журнала определяет доступные страницы по прямым шаблонам:
      files/assets/common/page-html5-substrates/page0001_3.jpg
      files/assets/common/page-html5-substrates/page0001_3.webp
      files/assets/flash/pages/page0001_w.webp
3. Скачивает страницы в:
      data/housesru/all_magazines/<year>/<slug>/pages/
4. Пишет manifest:
      data/housesru/all_magazines/manifest_all_magazines.json

Главное:
- Скрипт НЕ ищет планировки.
- Скрипт НЕ делает медленный brute-force по десяткам шаблонов.
- Скрипт скачивает только страницы журналов.
- После этого можно отдельно запускать твой текущий parse_flippingbook_floorplans_debug_v4.py
  по каждой папке pages через --input-dir.

Пример запуска
==============

    python src/tools/collect_housesru_all_magazine_images.py \
      --out data/housesru/all_magazines \
      --years 2018-2025 \
      --max-pages 400 \
      --miss-limit 12

Если нужно только один год:

    python src/tools/collect_housesru_all_magazine_images.py \
      --out data/housesru/all_magazines \
      --years 2025 \
      --max-pages 400 \
      --miss-limit 12

Если roots уже известны, можно передать файл:

    python src/tools/collect_housesru_all_magazine_images.py \
      --roots-file data/housesru/magazine_roots.txt \
      --out data/housesru/all_magazines

Формат roots-file:
    https://houses.ru/magazines/2025/100kk-2025/
    https://houses.ru/magazines/2024/100kk-2024/

Зависимости:
    python -m pip install requests beautifulsoup4
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import time
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


MAGAZINE_ROOT_RE = re.compile(r"https?://houses\.ru/magazines/(\d{4})/([^/?#]+)/?", re.IGNORECASE)
REL_MAGAZINE_RE = re.compile(r"/magazines/(\d{4})/([^/?#]+)/?", re.IGNORECASE)


@dataclasses.dataclass(frozen=True)
class MagazineRoot:
    year: int
    slug: str
    url: str


@dataclasses.dataclass(frozen=True)
class PageDownload:
    page: int
    url: str
    path: str
    bytes: int
    source: str


@dataclasses.dataclass(frozen=True)
class MagazineReport:
    year: int
    slug: str
    root_url: str
    out_dir: str
    status: str
    page_count: int
    pages: list[dict[str, Any]]
    errors: list[str]


class Logger:
    def __init__(self, out_dir: Path) -> None:
        self.t0 = time.time()
        self.path = out_dir / "collect_log.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, stage: str, message: str, **data: Any) -> None:
        row = {
            "t": round(time.time() - self.t0, 3),
            "stage": stage,
            "message": message,
            **data,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        extra = ""
        if data:
            compact = " ".join(f"{k}={v}" for k, v in data.items() if isinstance(v, (str, int, float, bool)))
            extra = f" | {compact}" if compact else ""
        print(f"[{row['t']:8.2f}s] [{stage}] {message}{extra}", flush=True)


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
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
    return s


def http_get(
    session: requests.Session,
    url: str,
    *,
    connect_timeout: float,
    read_timeout: float,
    logger: Logger,
    stage: str,
) -> Optional[requests.Response]:
    try:
        logger.log(stage, "GET", url=url)
        r = session.get(url, timeout=(connect_timeout, read_timeout), allow_redirects=True)
        logger.log(
            stage,
            "GET done",
            url=url,
            status=r.status_code,
            content_type=r.headers.get("content-type", ""),
            bytes=len(r.content or b""),
        )
        return r if r.status_code == 200 else None
    except requests.RequestException as exc:
        logger.log(stage, "GET failed", url=url, error_type=type(exc).__name__, error=str(exc))
        return None


def looks_like_image(resp: requests.Response) -> bool:
    ctype = resp.headers.get("content-type", "").lower()
    if "image/" in ctype:
        return True
    b = resp.content[:16]
    return (
        b.startswith(b"\xff\xd8\xff")
        or b.startswith(b"\x89PNG\r\n\x1a\n")
        or (b.startswith(b"RIFF") and b"WEBP" in b[:16])
    )


def parse_years_spec(spec: str) -> list[int]:
    years: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            y1, y2 = int(a), int(b)
            if y1 > y2:
                y1, y2 = y2, y1
            years.update(range(y1, y2 + 1))
        else:
            years.add(int(part))
    return sorted(years)


def normalize_root(url: str) -> Optional[MagazineRoot]:
    url = url.strip()
    if not url:
        return None

    m = MAGAZINE_ROOT_RE.search(url)
    if not m:
        return None

    year = int(m.group(1))
    slug = m.group(2).strip("/")
    root = f"https://houses.ru/magazines/{year}/{slug}/"
    return MagazineRoot(year=year, slug=slug, url=root)


def extract_roots_from_html(base_url: str, html: str) -> list[MagazineRoot]:
    roots: dict[tuple[int, str], MagazineRoot] = {}

    for m in REL_MAGAZINE_RE.finditer(html):
        year = int(m.group(1))
        slug = m.group(2).strip("/")
        url = f"https://houses.ru/magazines/{year}/{slug}/"
        roots[(year, slug)] = MagazineRoot(year=year, slug=slug, url=url)

    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a"):
        href = a.get("href")
        if not href:
            continue
        abs_url = urljoin(base_url, href)
        root = normalize_root(abs_url)
        if root:
            roots[(root.year, root.slug)] = root

    return sorted(roots.values(), key=lambda r: (r.year, r.slug))


def discover_roots_by_site_pages(
    session: requests.Session,
    years: list[int],
    *,
    logger: Logger,
    connect_timeout: float,
    read_timeout: float,
) -> list[MagazineRoot]:
    """
    Пытается найти журналы через несколько очевидных страниц сайта.
    Это не гарантирует полный охват, но хорошо собирает публичные ссылки.
    """
    urls_to_check = [
        "https://houses.ru/",
        "https://houses.ru/magazines/",
    ]

    for year in years:
        urls_to_check.extend(
            [
                f"https://houses.ru/magazines/{year}/",
                f"https://houses.ru/magazines/{year}",
                f"https://houses.ru/magazines/?year={year}",
            ]
        )

    found: dict[tuple[int, str], MagazineRoot] = {}

    for url in urls_to_check:
        resp = http_get(
            session,
            url,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            logger=logger,
            stage="discover:site",
        )
        if resp is None:
            continue

        roots = extract_roots_from_html(url, resp.text)
        logger.log("discover:site", "roots from page", url=url, count=len(roots))
        for root in roots:
            if root.year in years:
                found[(root.year, root.slug)] = root

    return sorted(found.values(), key=lambda r: (r.year, r.slug))


def discover_roots_by_search_guesses(years: list[int]) -> list[MagazineRoot]:
    """
    Резервный список наиболее вероятных slug-ов.
    Добавляй сюда руками новые серии, когда найдёшь их на сайте.
    """
    roots: dict[tuple[int, str], MagazineRoot] = {}

    slug_templates = [
        "100kk-{year}",
        "100-design-{year}",
        "100-design-projects-{year}",
        "krasivye-kvartiry-{year}",
        "kd-{year}",
    ]

    for year in years:
        for tmpl in slug_templates:
            slug = tmpl.format(year=year)
            root = MagazineRoot(year=year, slug=slug, url=f"https://houses.ru/magazines/{year}/{slug}/")
            roots[(year, slug)] = root

    return sorted(roots.values(), key=lambda r: (r.year, r.slug))


def load_roots_file(path: Path) -> list[MagazineRoot]:
    roots: dict[tuple[int, str], MagazineRoot] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        root = normalize_root(line)
        if root:
            roots[(root.year, root.slug)] = root
    return sorted(roots.values(), key=lambda r: (r.year, r.slug))


def page_templates(root_url: str, page: int) -> list[tuple[str, str]]:
    """
    Шаблоны, найденные в Network у houses.ru.
    Порядок важен: сначала качественные html5-substrates, потом миниатюры flash.
    """
    rels = [
        ("html5_jpg", f"files/assets/common/page-html5-substrates/page{page:04d}_3.jpg"),
        ("html5_webp", f"files/assets/common/page-html5-substrates/page{page:04d}_3.webp"),
        ("flash_webp", f"files/assets/flash/pages/page{page:04d}_w.webp"),
    ]
    return [(source, urljoin(root_url, rel)) for source, rel in rels]


def choose_best_page_url(
    session: requests.Session,
    root: MagazineRoot,
    page: int,
    *,
    logger: Logger,
    connect_timeout: float,
    read_timeout: float,
) -> Optional[tuple[str, bytes, str]]:
    best: Optional[tuple[str, bytes, str, int]] = None

    for source, url in page_templates(root.url, page):
        resp = http_get(
            session,
            url,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            logger=logger,
            stage="page:check",
        )
        if resp is None or not looks_like_image(resp):
            continue

        content = resp.content
        source_bonus = 10_000_000 if source.startswith("html5") else 0
        score = source_bonus + len(content)

        if best is None or score > best[3]:
            best = (url, content, source, score)

    if best is None:
        return None

    return best[0], best[1], best[2]


def safe_slug(slug: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", slug).strip("_") or "magazine"


def download_magazine_pages(
    session: requests.Session,
    root: MagazineRoot,
    out_root: Path,
    *,
    logger: Logger,
    max_pages: int,
    miss_limit: int,
    connect_timeout: float,
    read_timeout: float,
    overwrite: bool,
) -> MagazineReport:
    mag_dir = out_root / str(root.year) / safe_slug(root.slug)
    pages_dir = mag_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    logger.log("magazine", "start", year=root.year, slug=root.slug, url=root.url)

    pages: list[PageDownload] = []
    errors: list[str] = []
    consecutive_misses = 0

    for page in range(1, max_pages + 1):
        existing = sorted(pages_dir.glob(f"page_{page:04d}.*"))
        if existing and not overwrite:
            path = existing[0]
            pages.append(
                PageDownload(
                    page=page,
                    url="existing",
                    path=str(path),
                    bytes=path.stat().st_size,
                    source="existing",
                )
            )
            consecutive_misses = 0
            logger.log("magazine", "page exists", page=page, path=str(path))
            continue

        found = choose_best_page_url(
            session,
            root,
            page,
            logger=logger,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
        )

        if found is None:
            consecutive_misses += 1
            logger.log("magazine", "page miss", page=page, consecutive_misses=consecutive_misses)
            if consecutive_misses >= miss_limit:
                logger.log("magazine", "stop by miss limit", page=page, miss_limit=miss_limit)
                break
            continue

        url, content, source = found
        suffix = Path(urlparse(url).path).suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
            suffix = ".jpg"

        path = pages_dir / f"page_{page:04d}{suffix}"
        path.write_bytes(content)
        pages.append(
            PageDownload(
                page=page,
                url=url,
                path=str(path),
                bytes=len(content),
                source=source,
            )
        )
        consecutive_misses = 0
        logger.log("magazine", "page saved", page=page, source=source, bytes=len(content), path=str(path))

    status = "ok" if pages else "no_pages"
    report = MagazineReport(
        year=root.year,
        slug=root.slug,
        root_url=root.url,
        out_dir=str(mag_dir),
        status=status,
        page_count=len(pages),
        pages=[dataclasses.asdict(p) for p in pages],
        errors=errors,
    )

    (mag_dir / "magazine_manifest.json").write_text(
        json.dumps(dataclasses.asdict(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.log("magazine", "done", year=root.year, slug=root.slug, pages=len(pages), status=status)
    return report


def save_roots(path: Path, roots: list[MagazineRoot]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(root.url for root in roots) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Collect all HOUSES.RU magazine page images by year.")
    p.add_argument("--out", required=True)
    p.add_argument("--years", default="2018-2025")
    p.add_argument("--roots-file")
    p.add_argument("--include-guess-roots", action="store_true")
    p.add_argument("--max-pages", type=int, default=400)
    p.add_argument("--miss-limit", type=int, default=12)
    p.add_argument("--connect-timeout", type=float, default=3.0)
    p.add_argument("--read-timeout", type=float, default=8.0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--only-root-regex", help="Обработать только root URL, подходящие под regex.")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = Logger(out_dir)
    logger.log("main", "start", out=str(out_dir), years=args.years)

    years = parse_years_spec(args.years)
    session = make_session()

    roots: dict[tuple[int, str], MagazineRoot] = {}

    if args.roots_file:
        for root in load_roots_file(Path(args.roots_file).expanduser()):
            roots[(root.year, root.slug)] = root
        logger.log("main", "roots loaded from file", count=len(roots))

    discovered = discover_roots_by_site_pages(
        session,
        years,
        logger=logger,
        connect_timeout=args.connect_timeout,
        read_timeout=args.read_timeout,
    )
    for root in discovered:
        roots[(root.year, root.slug)] = root
    logger.log("main", "roots after site discovery", count=len(roots))

    if args.include_guess_roots:
        for root in discover_roots_by_search_guesses(years):
            roots[(root.year, root.slug)] = root
        logger.log("main", "roots after guesses", count=len(roots))

    roots_list = sorted(roots.values(), key=lambda r: (r.year, r.slug))

    if args.only_root_regex:
        pat = re.compile(args.only_root_regex)
        roots_list = [r for r in roots_list if pat.search(r.url)]

    roots_path = out_dir / "magazine_roots.discovered.txt"
    save_roots(roots_path, roots_list)
    logger.log("main", "roots saved", path=str(roots_path), count=len(roots_list))

    reports: list[MagazineReport] = []
    for idx, root in enumerate(roots_list, start=1):
        logger.log("main", "process root", index=idx, total=len(roots_list), url=root.url)
        report = download_magazine_pages(
            session,
            root,
            out_dir,
            logger=logger,
            max_pages=args.max_pages,
            miss_limit=args.miss_limit,
            connect_timeout=args.connect_timeout,
            read_timeout=args.read_timeout,
            overwrite=args.overwrite,
        )
        reports.append(report)

    manifest = {
        "schema": "housesru_all_magazine_images/v1",
        "years": years,
        "out_dir": str(out_dir),
        "root_count": len(roots_list),
        "magazine_count": len(reports),
        "total_pages": sum(r.page_count for r in reports),
        "magazines": [dataclasses.asdict(r) for r in reports],
    }

    manifest_path = out_dir / "manifest_all_magazines.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.log("main", "done", manifest=str(manifest_path), total_pages=manifest["total_pages"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
