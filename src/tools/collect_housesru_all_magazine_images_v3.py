#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collect_housesru_all_magazine_images_v2.py

Сбор изображений страниц журналов houses.ru.

Поддерживает оба типа magazine root:
  1) новые:
     https://houses.ru/magazines/2025/100kk-2025/
  2) старые:
     https://houses.ru/magazines/100kk-2015/
     https://houses.ru/magazines/100kk-2015/22/  -> нормализуется в /magazines/100kk-2015/

Примеры:

1. Старые 100kk за 2011-2025:
   python src/tools/collect_housesru_all_magazine_images_v2.py \
     --out data/housesru/all_magazines_v2 \
     --years 2011-2025 \
     --include-old-100kk-roots \
     --max-pages 400 \
     --miss-limit 3

2. Старые + новые 100kk:
   python src/tools/collect_housesru_all_magazine_images_v2.py \
     --out data/housesru/all_magazines_v2 \
     --years 2011-2025 \
     --include-old-100kk-roots \
     --include-new-100kk-roots \
     --max-pages 400 \
     --miss-limit 3

3. По ручному списку:
   cat > data/housesru/magazine_roots_extended.txt <<'EOF'
   https://houses.ru/magazines/100kk-2011/22/
   https://houses.ru/magazines/100kk-2015/22/
   https://houses.ru/magazines/2024/100kk-2024/
   https://houses.ru/magazines/2025/100kk-2025/
   EOF

   python src/tools/collect_housesru_all_magazine_images_v2.py \
     --roots-file data/housesru/magazine_roots_extended.txt \
     --out data/housesru/all_magazines_v2 \
     --max-pages 400 \
     --miss-limit 3
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


NEW_ROOT_RE = re.compile(r"https?://houses\.ru/magazines/(\d{4})/([^/?#]+)/?", re.IGNORECASE)
OLD_ROOT_RE = re.compile(r"https?://houses\.ru/magazines/([^/?#]+)/?(?:\d+)?/?(?:[?#].*)?$", re.IGNORECASE)
REL_NEW_ROOT_RE = re.compile(r"/magazines/(\d{4})/([^/?#]+)/?", re.IGNORECASE)
REL_OLD_ROOT_RE = re.compile(r"/magazines/([^/?#]+)/?(?:\d+)?/?(?:[?#].*)?$", re.IGNORECASE)
YEAR_IN_SLUG_RE = re.compile(r"(19|20)\d{2}")


@dataclasses.dataclass(frozen=True)
class MagazineRoot:
    year: int
    slug: str
    url: str
    scheme: str


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
    scheme: str
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
        row = {"t": round(time.time() - self.t0, 3), "stage": stage, "message": message, **data}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
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


def http_get(session: requests.Session, url: str, *, connect_timeout: float, read_timeout: float, logger: Logger, stage: str, log_non_200: bool = False) -> Optional[requests.Response]:
    try:
        r = session.get(url, timeout=(connect_timeout, read_timeout), allow_redirects=True)
        if r.status_code == 200:
            logger.log(stage, "GET 200", url=url, content_type=r.headers.get("content-type", ""), bytes=len(r.content or b""))
            return r
        if log_non_200 or r.status_code not in {403, 404}:
            logger.log(stage, "GET non-200", url=url, status=r.status_code, content_type=r.headers.get("content-type", ""), bytes=len(r.content or b""))
        return None
    except requests.RequestException as exc:
        logger.log(stage, "GET failed", url=url, error_type=type(exc).__name__, error=str(exc))
        return None


def looks_like_image(resp: requests.Response) -> bool:
    ctype = resp.headers.get("content-type", "").lower()
    if "image/" in ctype:
        return True
    b = resp.content[:16]
    return b.startswith(b"\xff\xd8\xff") or b.startswith(b"\x89PNG\r\n\x1a\n") or (b.startswith(b"RIFF") and b"WEBP" in b[:16])


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


def infer_year_from_slug(slug: str) -> int:
    m = YEAR_IN_SLUG_RE.search(slug)
    return int(m.group(0)) if m else 0


def normalize_root(raw_url: str) -> Optional[MagazineRoot]:
    """
    Нормализует magazine URL.

    Примеры:
      /magazines/2025/100kk-2025/18/ -> /magazines/2025/100kk-2025/
      /magazines/100kk-2015/22/      -> /magazines/100kk-2015/

    Важная деталь: старый slug может содержать цифры, поэтому нельзя regex-ом
    лениво захватывать `100kk-` и считать `2015` номером страницы.
    """
    raw_url = raw_url.strip()
    if not raw_url or raw_url.startswith("#"):
        return None

    raw_url = urljoin("https://houses.ru", raw_url)
    parsed = urlparse(raw_url)
    parts = [p for p in parsed.path.split("/") if p]

    if not parts or parts[0] != "magazines":
        return None

    # Новый формат: /magazines/2025/100kk-2025/18/
    if len(parts) >= 3 and re.fullmatch(r"\d{4}", parts[1]):
        year = int(parts[1])
        slug = parts[2].strip("/")
        if not slug or slug.isdigit():
            return None
        return MagazineRoot(
            year=year,
            slug=slug,
            url=f"https://houses.ru/magazines/{year}/{slug}/",
            scheme="new_year_slug",
        )

    # Старый формат: /magazines/100kk-2015/22/
    if len(parts) >= 2:
        slug = parts[1].strip("/")
        if not slug or slug.isdigit() or slug == "magazines":
            return None
        year = infer_year_from_slug(slug)
        return MagazineRoot(
            year=year,
            slug=slug,
            url=f"https://houses.ru/magazines/{slug}/",
            scheme="old_slug",
        )

    return None


def extract_roots_from_html(base_url: str, html: str) -> list[MagazineRoot]:
    roots: dict[str, MagazineRoot] = {}
    soup = BeautifulSoup(html, "html.parser")
    candidates: list[str] = []

    for a in soup.find_all("a"):
        href = a.get("href")
        if href:
            candidates.append(urljoin(base_url, href))

    for m in REL_NEW_ROOT_RE.finditer(html):
        candidates.append(urljoin(base_url, f"/magazines/{m.group(1)}/{m.group(2)}/"))

    for m in REL_OLD_ROOT_RE.finditer(html):
        slug = m.group(1)
        if YEAR_IN_SLUG_RE.search(slug):
            candidates.append(urljoin(base_url, f"/magazines/{slug}/"))

    for url in candidates:
        root = normalize_root(url)
        if root:
            roots[root.url] = root

    return sorted(roots.values(), key=lambda r: (r.year, r.slug, r.url))


def discover_roots_by_site_pages(session: requests.Session, years: list[int], *, logger: Logger, connect_timeout: float, read_timeout: float) -> list[MagazineRoot]:
    urls = ["https://houses.ru/", "https://houses.ru/magazines/"]
    for year in years:
        urls.extend(
            [
                f"https://houses.ru/magazines/?year={year}",
                f"https://houses.ru/search/?q=100kk-{year}",
                f"https://houses.ru/search/?q=100kk%20{year}",
            ]
        )

    found: dict[str, MagazineRoot] = {}
    for url in urls:
        resp = http_get(session, url, connect_timeout=connect_timeout, read_timeout=read_timeout, logger=logger, stage="discover:site")
        if resp is None:
            continue
        roots = extract_roots_from_html(url, resp.text)
        logger.log("discover:site", "roots from page", url=url, count=len(roots))
        for root in roots:
            if root.year == 0 or root.year in years:
                found[root.url] = root

    return sorted(found.values(), key=lambda r: (r.year, r.slug, r.url))


def guess_roots(years: list[int], *, old_100kk: bool, new_100kk: bool, extra: bool) -> list[MagazineRoot]:
    roots: dict[str, MagazineRoot] = {}

    for year in years:
        if old_100kk:
            root = normalize_root(f"https://houses.ru/magazines/100kk-{year}/")
            if root:
                roots[root.url] = root

        if new_100kk:
            root = normalize_root(f"https://houses.ru/magazines/{year}/100kk-{year}/")
            if root:
                roots[root.url] = root

        if extra:
            for slug in [f"krasivye-kvartiry-{year}", f"bs-{year}", f"ko-{year}"]:
                for url in [f"https://houses.ru/magazines/{slug}/", f"https://houses.ru/magazines/{year}/{slug}/"]:
                    root = normalize_root(url)
                    if root:
                        roots[root.url] = root

    return sorted(roots.values(), key=lambda r: (r.year, r.slug, r.url))


def load_roots_file(path: Path) -> list[MagazineRoot]:
    roots: dict[str, MagazineRoot] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        root = normalize_root(line)
        if root:
            roots[root.url] = root
    return sorted(roots.values(), key=lambda r: (r.year, r.slug, r.url))


def page_templates(root_url: str, page: int) -> list[tuple[str, str]]:
    return [
        ("html5_jpg", urljoin(root_url, f"files/assets/common/page-html5-substrates/page{page:04d}_3.jpg")),
        ("html5_webp", urljoin(root_url, f"files/assets/common/page-html5-substrates/page{page:04d}_3.webp")),
        ("flash_webp", urljoin(root_url, f"files/assets/flash/pages/page{page:04d}_w.webp")),
    ]


def choose_best_page_url(session: requests.Session, root: MagazineRoot, page: int, *, logger: Logger, connect_timeout: float, read_timeout: float, fast_jpg: bool) -> Optional[tuple[str, bytes, str]]:
    best: Optional[tuple[str, bytes, str, int]] = None

    for source, url in page_templates(root.url, page):
        resp = http_get(session, url, connect_timeout=connect_timeout, read_timeout=read_timeout, logger=logger, stage="page:check", log_non_200=False)
        if resp is None or not looks_like_image(resp):
            continue

        content = resp.content
        if fast_jpg and source == "html5_jpg":
            return url, content, source

        source_bonus = 10_000_000 if source.startswith("html5") else 0
        score = source_bonus + len(content)
        if best is None or score > best[3]:
            best = (url, content, source, score)

    if best is None:
        return None
    return best[0], best[1], best[2]


def safe_slug(slug: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", slug).strip("_") or "magazine"


def download_magazine_pages(session: requests.Session, root: MagazineRoot, out_root: Path, *, logger: Logger, max_pages: int, miss_limit: int, connect_timeout: float, read_timeout: float, overwrite: bool, fast_jpg: bool) -> MagazineReport:
    year_dir = str(root.year) if root.year else "unknown_year"
    mag_dir = out_root / year_dir / safe_slug(root.slug)
    pages_dir = mag_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    logger.log("magazine", "start", year=root.year, slug=root.slug, scheme=root.scheme, url=root.url)

    pages: list[PageDownload] = []
    errors: list[str] = []
    consecutive_misses = 0

    for page in range(1, max_pages + 1):
        existing = sorted(pages_dir.glob(f"page_{page:04d}.*"))
        if existing and not overwrite:
            path = existing[0]
            pages.append(PageDownload(page=page, url="existing", path=str(path), bytes=path.stat().st_size, source="existing"))
            consecutive_misses = 0
            if page <= 3 or page % 25 == 0:
                logger.log("magazine", "page exists", page=page, pages=len(pages))
            continue

        found = choose_best_page_url(session, root, page, logger=logger, connect_timeout=connect_timeout, read_timeout=read_timeout, fast_jpg=fast_jpg)

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

        pages.append(PageDownload(page=page, url=url, path=str(path), bytes=len(content), source=source))
        consecutive_misses = 0

        if page <= 3 or page % 25 == 0:
            logger.log("magazine", "page saved", page=page, source=source, bytes=len(content), path=str(path), pages=len(pages))

    status = "ok" if pages else "no_pages"
    report = MagazineReport(
        year=root.year,
        slug=root.slug,
        root_url=root.url,
        scheme=root.scheme,
        out_dir=str(mag_dir),
        status=status,
        page_count=len(pages),
        pages=[dataclasses.asdict(p) for p in pages],
        errors=errors,
    )

    (mag_dir / "magazine_manifest.json").write_text(json.dumps(dataclasses.asdict(report), ensure_ascii=False, indent=2), encoding="utf-8")
    logger.log("magazine", "done", year=root.year, slug=root.slug, pages=len(pages), status=status)
    return report


def save_roots(path: Path, roots: list[MagazineRoot]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(r.url for r in roots) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Collect houses.ru magazine page images, including old /magazines/100kk-2015/ roots.")
    p.add_argument("--out", required=True)
    p.add_argument("--years", default="2011-2025")
    p.add_argument("--roots-file")
    p.add_argument("--discover-site", action="store_true")
    p.add_argument("--include-old-100kk-roots", action="store_true")
    p.add_argument("--include-new-100kk-roots", action="store_true")
    p.add_argument("--include-extra-guesses", action="store_true")
    p.add_argument("--max-pages", type=int, default=400)
    p.add_argument("--miss-limit", type=int, default=3)
    p.add_argument("--connect-timeout", type=float, default=2.5)
    p.add_argument("--read-timeout", type=float, default=5.0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--fast-jpg", action="store_true", default=True)
    p.add_argument("--only-root-regex")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(out_dir)

    years = parse_years_spec(args.years)
    session = make_session()
    roots: dict[str, MagazineRoot] = {}

    logger.log("main", "start", out=str(out_dir), years=args.years)

    if args.roots_file:
        for root in load_roots_file(Path(args.roots_file).expanduser()):
            roots[root.url] = root
        logger.log("main", "roots loaded from file", count=len(roots))

    if args.discover_site:
        for root in discover_roots_by_site_pages(session, years, logger=logger, connect_timeout=args.connect_timeout, read_timeout=args.read_timeout):
            roots[root.url] = root
        logger.log("main", "roots after site discovery", count=len(roots))

    for root in guess_roots(years, old_100kk=args.include_old_100kk_roots, new_100kk=args.include_new_100kk_roots, extra=args.include_extra_guesses):
        roots[root.url] = root

    roots_list = sorted(roots.values(), key=lambda r: (r.year, r.slug, r.url))

    if args.only_root_regex:
        pat = re.compile(args.only_root_regex)
        roots_list = [r for r in roots_list if pat.search(r.url)]

    roots_path = out_dir / "magazine_roots.discovered.txt"
    save_roots(roots_path, roots_list)
    logger.log("main", "roots ready", path=str(roots_path), count=len(roots_list))

    reports: list[MagazineReport] = []
    for idx, root in enumerate(roots_list, start=1):
        logger.log("main", "process root", index=idx, total=len(roots_list), url=root.url)
        reports.append(
            download_magazine_pages(
                session,
                root,
                out_dir,
                logger=logger,
                max_pages=args.max_pages,
                miss_limit=args.miss_limit,
                connect_timeout=args.connect_timeout,
                read_timeout=args.read_timeout,
                overwrite=args.overwrite,
                fast_jpg=args.fast_jpg,
            )
        )

    manifest = {
        "schema": "housesru_all_magazine_images/v2",
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
