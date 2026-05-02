#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Iterator
from urllib.parse import urljoin, urlparse, urlunparse

import numpy as np
import requests
import urllib3
from bs4 import BeautifulSoup
from PIL import Image
from requests.exceptions import SSLError


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
IMAGE_EXT_RE = r"(?:jpg|jpeg|png|webp|bmp)"
DEFAULT_CONNECT_TIMEOUT = 15
DEFAULT_READ_TIMEOUT = 45
DEFAULT_SLEEP = 0.12
DEFAULT_PAGE_RETRIES = 2
DEFAULT_IMAGE_RETRIES = 2
DEFAULT_BACKOFF = 1.5
DEFAULT_OUTPUT = Path("data/sourse/floorplans")
DEFAULT_SOURCE_URLS = (
    "https://bigfoto.name/photo/9517-dvuhkomnatnaja-kvartira-planirovka-s-razmerami-55-foto.html",
    "https://bigfoto.name/9896-planirovka-kvartiry-s-razmerami-sten-73-foto.html",
    "https://bigfoto.name/15367-standartnye-planirovki-kvartir-69-foto.html",
    "https://bigfoto.name/15819-standartnaja-planirovka-71-foto.html",
    "https://bigfoto.name/photo/3212-kvartira-studija-chertezh-59-foto.html",
    "https://bigfoto.name/photo/632-obmerochnyj-chertezh-kvartiry-84-foto.html",
    "https://bigfoto.name/photo/18265-planirovka-kvartir-v-hruschevke-84-foto.html",
    "https://idei.club/49503-planirovka-doma-s-rasstanovkoj-mebeli-85-foto.html",
    "https://idei.club/49504-planirovki-domov-s-rasstanovkoj-mebeli-94-foto.html",
    "https://idei.club/48565-plan-rasstanovki-mebeli-146-foto.html",
    "https://idei.club/99184-planirovka-komnaty-chertezh-68-foto.html",
    "https://idei.club/95786-plan-komnaty-chertezh-69-foto.html",
)
DISCOVERY_SITEMAPS = {
    "bigfoto.name": ("https://bigfoto.name/sitemap.xml",),
    "idei.club": ("https://idei.club/sitemap.xml",),
    "amiel.club": ("https://amiel.club/sitemap.xml",),
}
DISCOVERY_TECH_TOKENS = (
    "plan",
    "planirov",
    "chertezh",
    "obmer",
    "rasstanovk",
    "skhem",
    "shem",
    "razmer",
    "pereplan",
    "sverhu",
)
DISCOVERY_SPACE_TOKENS = (
    "kvartir",
    "komnat",
    "dom",
    "kuhn",
    "gostin",
    "spal",
    "prihozh",
    "koridor",
    "zal",
    "studi",
    "sanuz",
    "vann",
    "detsk",
    "etazh",
    "mansard",
    "kottedzh",
    "hruschevk",
    "dvushk",
    "odnokomnat",
    "trehkomnat",
    "pomesh",
)
DISCOVERY_EXCLUDE_TOKENS = (
    "novomu-godu",
    "ploshchadk",
    "ploschadk",
    "restoran",
    "kafe",
    "ofis",
    "magazin",
    "garazh",
    "territor",
    "sadik",
    "skhaf",
    "shkaf",
    "proektor",
    "interer",
    "dizajn",
)

URL_REGEX = re.compile(
    rf"""
    (?P<url>
        (?:https?:)?//[^\s'\"<>]+?\.(?:{IMAGE_EXT_RE})(?:\?[^\s'\"<>]*)?
        |
        /[^\s'\"<>]+?\.(?:{IMAGE_EXT_RE})(?:\?[^\s'\"<>]*)?
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

BACKGROUND_IMAGE_REGEX = re.compile(
    rf"background-image\s*:\s*url\((['\"]?)(?P<url>.+?\.(?:{IMAGE_EXT_RE})(?:\?[^'\")]+)?)\1\)",
    re.IGNORECASE,
)

TBEGIN_REGEX = re.compile(
    rf"<!--\s*TBegin:(?P<url>https?://[^|<>\s]+?\.(?:{IMAGE_EXT_RE})(?:\?[^\s|<>]*)?)\|\|",
    re.IGNORECASE,
)
XML_LOC_REGEX = re.compile(r"<loc>\s*(?P<url>.*?)\s*</loc>", re.IGNORECASE)

LIKELY_NOISE_PATTERNS = (
    "captcha",
    "favicon",
    "logo",
    "sprite",
    "/engine/",
    "/templates/",
    "/avatars/",
    "/icons/",
)

ATTRIBUTE_CANDIDATES = (
    "href",
    "src",
    "data-src",
    "data-lazy-src",
    "data-original",
    "data-image",
    "data-full",
    "data-url",
    "content",
)

SRCSET_CANDIDATES = (
    "srcset",
    "data-srcset",
)

FLOORPLAN_SAT_MAX_MONO = 0.08
FLOORPLAN_SAT_MAX_COLOR = 0.14
FLOORPLAN_SAT_MAX_ORTHO = 0.26
FLOORPLAN_EDGE_MIN_MONO = 0.10
FLOORPLAN_EDGE_MIN_COLOR = 0.14
FLOORPLAN_WHITE_MIN_MONO = 0.20
FLOORPLAN_WHITE_MIN_COLOR = 0.22
FLOORPLAN_BLACK_MIN_MONO = 0.25
FLOORPLAN_MIDSTD_MIN_MONO = 0.55
FLOORPLAN_ORTHO_MIN = 0.78
FLOORPLAN_DIAG_MAX = 0.10
FLOORPLAN_STRONG_EDGE_MIN = 0.18


def log(message: str) -> None:
    print(message, flush=True)


def infer_ext_from_url(url: str) -> str:
    path = urlparse(url).path.lower()
    for ext in IMAGE_EXTENSIONS:
        if path.endswith(ext):
            return ext
    return ".jpg"


def normalize_bigfoto_url(url: str) -> str:
    url = url.replace("\\/", "/")
    return re.sub(
        r"/thumbs/(?=[^/]+\.(?:jpg|jpeg|png|webp|bmp)(?:\?|$))",
        "/",
        url,
        flags=re.IGNORECASE,
    )


def canonicalize_url(page_url: str, raw_url: str, prefer_full: bool = True) -> str | None:
    if not raw_url:
        return None

    raw_url = raw_url.strip().strip("\"'")
    if not raw_url or raw_url.startswith("data:"):
        return None

    abs_url = urljoin(page_url, raw_url.replace("\\/", "/"))
    parsed = urlparse(abs_url)
    if parsed.scheme not in {"http", "https"}:
        return None

    cleaned = urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, parsed.query, ""))
    host = urlparse(page_url).netloc.lower()
    if prefer_full and "bigfoto.name" in host:
        cleaned = normalize_bigfoto_url(cleaned)
    return cleaned


def same_host(url_a: str, url_b: str) -> bool:
    return urlparse(url_a).netloc.lower() == urlparse(url_b).netloc.lower()


def looks_like_gallery_image(page_url: str, image_url: str) -> bool:
    parsed = urlparse(image_url)
    path = parsed.path.lower()
    if not path.endswith(IMAGE_EXTENSIONS):
        return False
    if "/thumbs/" in path:
        return False
    if any(pattern in path for pattern in LIKELY_NOISE_PATTERNS):
        return False
    if not same_host(page_url, image_url):
        return False

    host = urlparse(page_url).netloc.lower()
    if "bigfoto.name" in host:
        return (
            "/uploads/posts/" in path
            or "/photo/uploads/posts/" in path
            or path.startswith("/uploads/")
            or path.startswith("/photo/")
        )
    if "idei.club" in host:
        return "/uploads/posts/" in path or path.startswith("/uploads/")
    if "amiel.club" in host:
        return "/uploads/posts/" in path or path.startswith("/uploads/")
    return True


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "ru,en;q=0.9",
            "Connection": "keep-alive",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        }
    )
    return session


def request_with_ssl_fallback(
    session: requests.Session,
    method: str,
    url: str,
    *,
    timeout: tuple[int, int],
    stream: bool = False,
    headers: dict[str, str] | None = None,
) -> requests.Response:
    try:
        return session.request(method, url, timeout=timeout, stream=stream, headers=headers)
    except SSLError:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        return session.request(method, url, timeout=timeout, stream=stream, headers=headers, verify=False)


def fetch_html(
    session: requests.Session,
    url: str,
    connect_timeout: int,
    read_timeout: int,
    page_retries: int,
    backoff: float,
) -> str:
    last_exc: Exception | None = None
    for attempt in range(1, page_retries + 1):
        try:
            response = request_with_ssl_fallback(
                session,
                "GET",
                url,
                timeout=(connect_timeout, read_timeout),
            )
            response.raise_for_status()
            response.encoding = response.encoding or "utf-8"
            return response.text
        except Exception as exc:
            last_exc = exc
            if attempt < page_retries:
                time.sleep(backoff * attempt)
    assert last_exc is not None
    raise last_exc


def fetch_text(
    session: requests.Session,
    url: str,
    connect_timeout: int,
    read_timeout: int,
    retries: int,
    backoff: float,
) -> str:
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = request_with_ssl_fallback(
                session,
                "GET",
                url,
                timeout=(connect_timeout, read_timeout),
            )
            response.raise_for_status()
            response.encoding = response.encoding or "utf-8"
            return response.text
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(backoff * attempt)
    assert last_exc is not None
    raise last_exc


def extract_title(soup: BeautifulSoup) -> str:
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(" ", strip=True)
    if soup.title and soup.title.get_text(strip=True):
        return soup.title.get_text(" ", strip=True)
    return "gallery"


def parse_srcset(srcset: str) -> Iterator[str]:
    for part in srcset.split(","):
        part = part.strip()
        if not part:
            continue
        yield part.split()[0]


def add_url(found: list[str], seen: set[str], page_url: str, raw_url: str | None, prefer_full: bool = True) -> None:
    normalized = canonicalize_url(page_url, raw_url or "", prefer_full=prefer_full)
    if not normalized:
        return
    if not looks_like_gallery_image(page_url, normalized):
        return
    if normalized in seen:
        return
    seen.add(normalized)
    found.append(normalized)


def extract_bigfoto_urls(page_url: str, html: str, soup: BeautifulSoup) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()

    for match in TBEGIN_REGEX.finditer(html):
        add_url(found, seen, page_url, match.group("url"), prefer_full=True)

    for tag in soup.select("a.highslide[href]"):
        add_url(found, seen, page_url, tag.get("href"), prefer_full=True)

    for tag in soup.find_all("a", href=True):
        href = tag.get("href")
        if isinstance(href, str) and ("/uploads/posts/" in href or "/photo/uploads/posts/" in href):
            add_url(found, seen, page_url, href, prefer_full=True)

    if not found:
        for match in URL_REGEX.finditer(html):
            add_url(found, seen, page_url, match.group("url"), prefer_full=True)
    return found


def extract_idei_urls(page_url: str, html: str, soup: BeautifulSoup) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()

    article = soup.select_one(".ftext.full-text")
    article_html = str(article) if article else html

    for match in TBEGIN_REGEX.finditer(article_html):
        add_url(found, seen, page_url, match.group("url"), prefer_full=True)

    if article:
        for tag in article.select("a.highslide[href]"):
            add_url(found, seen, page_url, tag.get("href"), prefer_full=True)

    return found


def extract_article_gallery_urls(page_url: str, html: str, soup: BeautifulSoup, selectors: tuple[str, ...]) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()

    article = None
    for selector in selectors:
        article = soup.select_one(selector)
        if article:
            break

    article_html = str(article) if article else html

    for match in URL_REGEX.finditer(article_html):
        add_url(found, seen, page_url, match.group("url"), prefer_full=True)

    if article:
        for tag in article.select("a.highslide[href], a.cursor[href], a[download][href]"):
            add_url(found, seen, page_url, tag.get("href"), prefer_full=True)
        for tag in article.select("img[data-src], img[src]"):
            add_url(found, seen, page_url, tag.get("data-src") or tag.get("src"), prefer_full=True)

    return found


def extract_from_dom_generic(page_url: str, soup: BeautifulSoup) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()

    for tag in soup.find_all(True):
        for attr in ATTRIBUTE_CANDIDATES:
            value = tag.get(attr)
            if isinstance(value, str):
                add_url(found, seen, page_url, value, prefer_full=True)

        for attr in SRCSET_CANDIDATES:
            value = tag.get(attr)
            if isinstance(value, str):
                for item in parse_srcset(value):
                    add_url(found, seen, page_url, item, prefer_full=True)

        style = tag.get("style")
        if isinstance(style, str):
            for match in BACKGROUND_IMAGE_REGEX.finditer(style):
                add_url(found, seen, page_url, match.group("url"), prefer_full=True)
    return found


def extract_from_raw_html_generic(page_url: str, html: str) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for match in URL_REGEX.finditer(html):
        add_url(found, seen, page_url, match.group("url"), prefer_full=True)
    return found


def extract_image_urls(page_url: str, html: str) -> tuple[str, list[str]]:
    soup = BeautifulSoup(html, "html.parser")
    title = extract_title(soup)
    host = urlparse(page_url).netloc.lower()

    if "bigfoto.name" in host:
        return title, extract_bigfoto_urls(page_url, html, soup)
    if "idei.club" in host:
        return title, extract_idei_urls(page_url, html, soup)
    if "amiel.club" in host:
        return title, extract_article_gallery_urls(page_url, html, soup, selectors=(".fulstr",))

    dom_urls = extract_from_dom_generic(page_url, soup)
    raw_urls = extract_from_raw_html_generic(page_url, html)

    merged: list[str] = []
    seen: set[str] = set()
    for image_url in dom_urls + raw_urls:
        if image_url not in seen:
            seen.add(image_url)
            merged.append(image_url)
    return title, merged


def load_state(state_path: Path) -> dict:
    if state_path.exists():
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {
        "next_index": 1,
        "completed": {},
        "failed": {},
        "sha1_to_file": {},
    }


def save_state(state_path: Path, state: dict) -> None:
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha1_of_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_image_rgb(path: Path, size: int = 256) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        image = Image.open(path).convert("RGB").resize((size, size))
    return np.asarray(image, dtype=np.float32) / 255.0


def analyze_floorplan_image(path: Path) -> dict[str, float]:
    rgb = load_image_rgb(path)
    gray = rgb.mean(axis=2)
    saturation = rgb.max(axis=2) - rgb.min(axis=2)

    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    edge_density = float((((gx > 0.08).mean()) + ((gy > 0.08).mean())) / 2.0)

    white_fraction = float((gray > 0.92).mean())
    black_fraction = float((gray < 0.10).mean())

    blocks = gray.reshape(16, 16, 16, 16).transpose(0, 2, 1, 3).reshape(256, 16, 16)
    low_variance_fraction = float((blocks.std(axis=(1, 2)) < 0.06).mean())

    dx = np.zeros_like(gray)
    dy = np.zeros_like(gray)
    dx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    dy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    grad_mag = np.hypot(dx, dy)
    strong_mask = grad_mag > 0.08

    ortho_fraction = 0.0
    diagonal_fraction = 0.0
    strong_fraction = float(strong_mask.mean())
    if strong_mask.any():
        angles = np.abs(np.degrees(np.arctan2(dy[strong_mask], dx[strong_mask])))
        angles = np.mod(angles, 90.0)
        ortho_fraction = float((np.minimum(angles, 90.0 - angles) <= 12.0).mean())
        diagonal_fraction = float((np.abs(angles - 45.0) <= 12.0).mean())

    return {
        "sat_mean": float(saturation.mean()),
        "white_fraction": white_fraction,
        "black_fraction": black_fraction,
        "edge_density": edge_density,
        "low_variance_fraction": low_variance_fraction,
        "ortho_fraction": ortho_fraction,
        "diagonal_fraction": diagonal_fraction,
        "strong_edge_fraction": strong_fraction,
    }


def is_likely_floorplan(path: Path) -> tuple[bool, dict[str, float]]:
    metrics = analyze_floorplan_image(path)
    keep = (
        (
            metrics["sat_mean"] <= FLOORPLAN_SAT_MAX_MONO
            and (
                metrics["edge_density"] >= FLOORPLAN_EDGE_MIN_MONO
                or metrics["white_fraction"] >= FLOORPLAN_WHITE_MIN_MONO
                or metrics["black_fraction"] >= FLOORPLAN_BLACK_MIN_MONO
                or metrics["low_variance_fraction"] >= FLOORPLAN_MIDSTD_MIN_MONO
            )
        )
        or (
            metrics["sat_mean"] <= FLOORPLAN_SAT_MAX_COLOR
            and (
                metrics["white_fraction"] >= FLOORPLAN_WHITE_MIN_COLOR
                or metrics["edge_density"] >= FLOORPLAN_EDGE_MIN_COLOR
            )
        )
        or (
            metrics["sat_mean"] <= FLOORPLAN_SAT_MAX_ORTHO
            and metrics["ortho_fraction"] >= FLOORPLAN_ORTHO_MIN
            and metrics["diagonal_fraction"] <= FLOORPLAN_DIAG_MAX
            and metrics["strong_edge_fraction"] >= FLOORPLAN_STRONG_EDGE_MIN
        )
    )
    return keep, metrics


def format_floorplan_metrics(metrics: dict[str, float]) -> str:
    return ", ".join(f"{key}={value:.3f}" for key, value in sorted(metrics.items()))


def read_urls_from_file(path: Path) -> list[str]:
    urls: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        urls.append(stripped)
    return urls


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def dedupe_preserve_order(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def matches_host_filter(page_url: str, include_host: str | None, exclude_host: str | None) -> bool:
    host = urlparse(page_url).netloc.lower()
    if include_host and include_host.lower() not in host:
        return False
    if exclude_host and exclude_host.lower() in host:
        return False
    return True


def current_unique_file_count(state: dict) -> int:
    return len(
        {
            record.get("saved_as")
            for record in state.get("completed", {}).values()
            if isinstance(record, dict) and record.get("saved_as")
        }
    )


def parse_xml_locs(xml_text: str) -> list[str]:
    return [match.group("url").strip() for match in XML_LOC_REGEX.finditer(xml_text)]


def discover_urls_from_sitemap(
    session: requests.Session,
    sitemap_url: str,
    args: argparse.Namespace,
    seen_sitemaps: set[str] | None = None,
) -> list[str]:
    if seen_sitemaps is None:
        seen_sitemaps = set()
    if sitemap_url in seen_sitemaps:
        return []
    seen_sitemaps.add(sitemap_url)

    xml_text = fetch_text(
        session,
        sitemap_url,
        connect_timeout=max(args.connect_timeout, 30),
        read_timeout=max(args.read_timeout, 120),
        retries=max(args.page_retries, 3),
        backoff=args.backoff,
    )
    locs = parse_xml_locs(xml_text)
    if "<sitemapindex" in xml_text.lower():
        discovered: list[str] = []
        for child_url in locs:
            discovered.extend(discover_urls_from_sitemap(session, child_url, args, seen_sitemaps))
        return discovered
    return locs


def url_matches_discovery_keywords(page_url: str) -> bool:
    path = urlparse(page_url).path.lower()
    if not path.endswith(".html"):
        return False
    if any(token in path for token in DISCOVERY_EXCLUDE_TOKENS):
        return False
    if not any(token in path for token in DISCOVERY_TECH_TOKENS):
        return False
    if not any(token in path for token in DISCOVERY_SPACE_TOKENS):
        return False
    return True


def round_robin_merge(groups: dict[str, list[str]]) -> list[str]:
    merged: list[str] = []
    cursors = {host: 0 for host in groups}
    while True:
        progressed = False
        for host, urls in groups.items():
            index = cursors[host]
            if index >= len(urls):
                continue
            merged.append(urls[index])
            cursors[host] = index + 1
            progressed = True
        if not progressed:
            return merged


def allocate_output_path(root_output: Path, state: dict, source_url: str) -> Path:
    ext = infer_ext_from_url(source_url)
    next_index = int(state.get("next_index", 1))
    out_path = root_output / f"{next_index:06d}{ext}"
    state["next_index"] = next_index + 1
    return out_path


def download_one(
    session: requests.Session,
    source_url: str,
    temp_path: Path,
    page_url: str,
    connect_timeout: int,
    read_timeout: int,
    image_retries: int,
    backoff: float,
) -> tuple[bool, str, int]:
    last_error = ""
    for attempt in range(1, image_retries + 1):
        try:
            with request_with_ssl_fallback(
                session,
                "GET",
                source_url,
                timeout=(connect_timeout, read_timeout),
                stream=True,
                headers={
                    "Referer": page_url,
                    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
                },
            ) as response:
                if response.status_code != 200:
                    last_error = f"HTTP {response.status_code}"
                    if attempt < image_retries:
                        time.sleep(backoff * attempt)
                    continue

                content_type = (response.headers.get("Content-Type") or "").lower()
                if not content_type.startswith("image/"):
                    last_error = f"unexpected content-type: {content_type}"
                    break

                size_bytes = 0
                with temp_path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 128):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        size_bytes += len(chunk)
                return True, response.url, size_bytes
        except Exception as exc:
            last_error = str(exc)
            if attempt < image_retries:
                time.sleep(backoff * attempt)
    return False, last_error, 0


def process_page(
    session: requests.Session,
    page_index: int,
    page_url: str,
    args: argparse.Namespace,
    state: dict,
    file_rows: list[dict],
    page_rows: list[dict],
    failure_rows: list[dict],
) -> None:
    log(f"[{page_index}] page {page_url}")

    html = fetch_html(
        session,
        page_url,
        connect_timeout=args.connect_timeout,
        read_timeout=args.read_timeout,
        page_retries=args.page_retries,
        backoff=args.backoff,
    )

    title, image_urls = extract_image_urls(page_url, html)
    image_urls = dedupe_preserve_order(image_urls)

    if args.save_debug_html:
        debug_path = args.output / f"page_{page_index:02d}.html"
        debug_path.write_text(html, encoding="utf-8")

    log(f"[{page_index}] found {len(image_urls)} candidate images")

    completed = state.setdefault("completed", {})
    failed = state.setdefault("failed", {})
    sha1_to_file = state.setdefault("sha1_to_file", {})
    downloaded_now = 0
    reused_existing = 0
    page_failed = 0

    for image_idx, source_url in enumerate(image_urls, start=1):
        existing = completed.get(source_url)
        if isinstance(existing, dict):
            saved_as = existing.get("saved_as")
            if saved_as and (args.output / saved_as).exists():
                reused_existing += 1
                log(f"[{page_index}:{image_idx:03d}/{len(image_urls):03d}] skip {saved_as}")
                continue

        out_path = allocate_output_path(args.output, state, source_url)
        temp_path = out_path.with_suffix(out_path.suffix + ".part")
        ok, info, size_bytes = download_one(
            session,
            source_url,
            temp_path,
            page_url,
            connect_timeout=args.connect_timeout,
            read_timeout=args.read_timeout,
            image_retries=args.image_retries,
            backoff=args.backoff,
        )

        if ok:
            metrics: dict[str, float] = {}
            if not args.no_floorplan_filter:
                try:
                    is_floorplan, metrics = is_likely_floorplan(temp_path)
                except Exception as exc:
                    is_floorplan = False
                    metrics = {"classifier_error": 1.0}
                    info = f"floorplan classifier failed: {exc}"

                if not is_floorplan:
                    quarantine_name = f"{out_path.stem}_rejected{out_path.suffix}"
                    quarantine_path = args.quarantine_dir / quarantine_name
                    args.quarantine_dir.mkdir(parents=True, exist_ok=True)
                    temp_path.replace(quarantine_path)
                    failed[source_url] = {
                        "page_index": page_index,
                        "page_url": page_url,
                        "page_title": title,
                        "error": f"rejected_by_floorplan_filter: {format_floorplan_metrics(metrics)}",
                    }
                    page_failed += 1
                    log(
                        f"[{page_index}:{image_idx:03d}/{len(image_urls):03d}] reject {source_url} :: "
                        f"{format_floorplan_metrics(metrics)}"
                    )
                    save_state(args.output / "state.json", state)
                    time.sleep(max(args.sleep, 0.0))
                    continue

            file_sha1 = sha1_of_file(temp_path)
            duplicate_name = sha1_to_file.get(file_sha1)
            if duplicate_name and (args.output / duplicate_name).exists():
                temp_path.unlink(missing_ok=True)
                saved_as = duplicate_name
                deduped_by_sha1 = True
            else:
                temp_path.replace(out_path)
                saved_as = out_path.name
                sha1_to_file[file_sha1] = saved_as
                downloaded_now += 1
                deduped_by_sha1 = False

            completed[source_url] = {
                "saved_as": saved_as,
                "page_index": page_index,
                "page_url": page_url,
                "page_title": title,
                "attempted_url": info,
                "sha1": file_sha1,
                "size_bytes": size_bytes,
                "deduped_by_sha1": deduped_by_sha1,
                "floorplan_metrics": metrics if not args.no_floorplan_filter else {},
            }
            failed.pop(source_url, None)
            log(f"[{page_index}:{image_idx:03d}/{len(image_urls):03d}] ok   {saved_as}")
        else:
            temp_path.unlink(missing_ok=True)
            failed[source_url] = {
                "page_index": page_index,
                "page_url": page_url,
                "page_title": title,
                "error": info,
            }
            page_failed += 1
            log(f"[{page_index}:{image_idx:03d}/{len(image_urls):03d}] fail {source_url} :: {info}")

        save_state(args.output / "state.json", state)
        time.sleep(max(args.sleep, 0.0))

    page_rows.append(
        {
            "page_index": page_index,
            "page_url": page_url,
            "title": title,
            "candidate_images": len(image_urls),
            "downloaded_new_files": downloaded_now,
            "reused_existing": reused_existing,
            "failed": page_failed,
            "status": "ok" if page_failed == 0 else "partial",
        }
    )

    for source_url in image_urls:
        record = completed.get(source_url)
        if not isinstance(record, dict):
            continue
        file_rows.append(
            {
                "saved_as": record.get("saved_as", ""),
                "page_index": page_index,
                "page_url": page_url,
                "page_title": title,
                "source_url": source_url,
                "attempted_url": record.get("attempted_url", ""),
                "sha1": record.get("sha1", ""),
                "size_bytes": record.get("size_bytes", 0),
                "deduped_by_sha1": record.get("deduped_by_sha1", False),
                "floorplan_metrics": json.dumps(record.get("floorplan_metrics", {}), ensure_ascii=False, sort_keys=True),
            }
        )

    for source_url, error_info in failed.items():
        if not isinstance(error_info, dict):
            continue
        if error_info.get("page_url") != page_url:
            continue
        failure_rows.append(
            {
                "page_index": page_index,
                "page_url": page_url,
                "page_title": error_info.get("page_title", title),
                "source_url": source_url,
                "error": error_info.get("error", ""),
            }
        )


def infer_next_index_from_files(output_dir: Path, current_next_index: int) -> int:
    max_index = 0
    for path in output_dir.iterdir():
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        try:
            max_index = max(max_index, int(path.stem))
        except ValueError:
            continue
    return max(current_next_index, max_index + 1)


def recheck_existing_dataset(args: argparse.Namespace, state: dict) -> tuple[int, int]:
    completed = state.setdefault("completed", {})
    failed = state.setdefault("failed", {})
    quarantine_dir = args.quarantine_dir
    quarantine_dir.mkdir(parents=True, exist_ok=True)

    rejected_files: set[str] = set()
    checked_files = 0
    rejected_count = 0

    for image_path in sorted(args.output.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        checked_files += 1
        keep, metrics = is_likely_floorplan(image_path)
        if keep:
            continue
        rejected_count += 1
        rejected_files.add(image_path.name)
        quarantine_path = quarantine_dir / image_path.name
        if quarantine_path.exists():
            quarantine_path.unlink()
        image_path.replace(quarantine_path)
        log(f"[recheck] reject {image_path.name} :: {format_floorplan_metrics(metrics)}")

    if rejected_files:
        remaining_completed: dict[str, dict] = {}
        for source_url, record in completed.items():
            if not isinstance(record, dict):
                continue
            saved_as = record.get("saved_as")
            if saved_as in rejected_files:
                failed[source_url] = {
                    "page_index": record.get("page_index", ""),
                    "page_url": record.get("page_url", ""),
                    "page_title": record.get("page_title", ""),
                    "error": "rejected_by_floorplan_filter_during_recheck",
                }
                continue
            if saved_as and (args.output / saved_as).exists():
                remaining_completed[source_url] = record
        state["completed"] = remaining_completed
        state["sha1_to_file"] = {
            record.get("sha1"): record.get("saved_as")
            for record in remaining_completed.values()
            if isinstance(record, dict)
            and record.get("sha1")
            and record.get("saved_as")
            and (args.output / str(record.get("saved_as"))).exists()
        }
        state["next_index"] = infer_next_index_from_files(args.output, int(state.get("next_index", 1)))

    return checked_files, rejected_count


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download designer floorplan images from gallery pages and store all unique files "
            "in a single dataset directory."
        )
    )
    parser.add_argument("urls", nargs="*", help="Gallery page URLs")
    parser.add_argument("--urls-file", help="Text file with page URLs, one per line")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Dataset output directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP, help="Pause between image downloads, seconds")
    parser.add_argument("--connect-timeout", type=int, default=DEFAULT_CONNECT_TIMEOUT)
    parser.add_argument("--read-timeout", type=int, default=DEFAULT_READ_TIMEOUT)
    parser.add_argument("--page-retries", type=int, default=DEFAULT_PAGE_RETRIES)
    parser.add_argument("--image-retries", type=int, default=DEFAULT_IMAGE_RETRIES)
    parser.add_argument("--backoff", type=float, default=DEFAULT_BACKOFF)
    parser.add_argument("--save-debug-html", action="store_true")
    parser.add_argument("--include-host", default=None, help="Only process pages whose host contains this substring")
    parser.add_argument("--exclude-host", default=None, help="Skip pages whose host contains this substring")
    parser.add_argument(
        "--discover-from-sitemaps",
        action="store_true",
        help="Auto-discover gallery page URLs from supported site sitemaps using floorplan-related slug filters",
    )
    parser.add_argument(
        "--discover-limit-per-host",
        type=int,
        default=0,
        help="Maximum number of auto-discovered pages to keep per host, 0 means no explicit cap",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=0,
        help="Maximum number of gallery pages to process after discovery and filtering, 0 means no cap",
    )
    parser.add_argument(
        "--target-unique-files",
        type=int,
        default=0,
        help="Stop once at least this many unique local image files exist in the dataset directory",
    )
    parser.add_argument(
        "--discovered-urls-out",
        type=Path,
        default=None,
        help="Optional text file where the final discovered/selected page URLs will be written",
    )
    parser.add_argument(
        "--no-default-urls",
        action="store_true",
        help="Do not use the built-in page list; require --urls-file and/or positional URLs",
    )
    parser.add_argument(
        "--no-floorplan-filter",
        action="store_true",
        help="Disable image-level floorplan filtering and keep every downloaded image",
    )
    parser.add_argument(
        "--recheck-existing",
        action="store_true",
        help="Re-scan already downloaded local files and move non-floorplans into quarantine",
    )
    parser.add_argument(
        "--quarantine-dir",
        type=Path,
        default=None,
        help="Directory for rejected non-floorplan images, default is <output>/quarantine",
    )
    return parser


def collect_page_urls(session: requests.Session, args: argparse.Namespace) -> list[str]:
    page_urls: list[str] = []
    if not args.no_default_urls:
        page_urls.extend(DEFAULT_SOURCE_URLS)
    if args.discover_from_sitemaps:
        discovered_by_host: dict[str, list[str]] = {}
        for host, sitemap_urls in DISCOVERY_SITEMAPS.items():
            if not matches_host_filter(f"https://{host}/", args.include_host, args.exclude_host):
                continue
            host_urls: list[str] = []
            try:
                for sitemap_url in sitemap_urls:
                    host_urls.extend(discover_urls_from_sitemap(session, sitemap_url, args))
                host_urls = [
                    url
                    for url in dedupe_preserve_order(host_urls)
                    if matches_host_filter(url, args.include_host, args.exclude_host) and url_matches_discovery_keywords(url)
                ]
                if args.discover_limit_per_host > 0:
                    host_urls = host_urls[: args.discover_limit_per_host]
                discovered_by_host[host] = host_urls
                log(f"[discover] {host}: {len(host_urls)} candidate pages")
            except Exception as exc:
                log(f"[discover] {host}: skipped due to error: {exc}")
        page_urls.extend(round_robin_merge(discovered_by_host))
    if args.urls_file:
        page_urls.extend(read_urls_from_file(args.urls_file))
    if args.urls:
        page_urls.extend(args.urls)

    page_urls = dedupe_preserve_order(page_urls)
    filtered_urls = [
        page_url
        for page_url in page_urls
        if matches_host_filter(page_url, args.include_host, args.exclude_host)
    ]
    if args.max_pages > 0:
        filtered_urls = filtered_urls[: args.max_pages]
    return filtered_urls


def write_outputs(
    output_dir: Path,
    input_urls: list[str],
    file_rows: list[dict],
    page_rows: list[dict],
    failure_rows: list[dict],
    state: dict,
) -> None:
    (output_dir / "source_pages.txt").write_text("\n".join(input_urls) + "\n", encoding="utf-8")

    completed = state.get("completed", {})
    manifest_rows: list[dict] = []
    for source_url, record in completed.items():
        if not isinstance(record, dict):
            continue
        manifest_rows.append(
            {
                "saved_as": record.get("saved_as", ""),
                "page_index": record.get("page_index", ""),
                "page_url": record.get("page_url", ""),
                "page_title": record.get("page_title", ""),
                "source_url": source_url,
                "attempted_url": record.get("attempted_url", ""),
                "sha1": record.get("sha1", ""),
                "size_bytes": record.get("size_bytes", 0),
                "deduped_by_sha1": record.get("deduped_by_sha1", False),
            }
        )
    manifest_rows.sort(key=lambda row: (str(row["saved_as"]), str(row["source_url"])))

    failed = state.get("failed", {})
    failed_rows_from_state: list[dict] = []
    for source_url, record in failed.items():
        if not isinstance(record, dict):
            continue
        failed_rows_from_state.append(
            {
                "page_index": record.get("page_index", ""),
                "page_url": record.get("page_url", ""),
                "page_title": record.get("page_title", ""),
                "source_url": source_url,
                "error": record.get("error", ""),
            }
        )
    failed_rows_from_state.sort(key=lambda row: (str(row["page_url"]), str(row["source_url"])))

    write_csv(
        output_dir / "manifest.csv",
        manifest_rows,
        fieldnames=[
            "saved_as",
            "page_index",
            "page_url",
            "page_title",
            "source_url",
            "attempted_url",
            "sha1",
            "size_bytes",
            "deduped_by_sha1",
            "floorplan_metrics",
        ],
    )

    write_csv(
        output_dir / "pages.csv",
        page_rows,
        fieldnames=[
            "page_index",
            "page_url",
            "title",
            "candidate_images",
            "downloaded_new_files",
            "reused_existing",
            "failed",
            "status",
        ],
    )

    write_csv(
        output_dir / "failed_urls.csv",
        failed_rows_from_state,
        fieldnames=["page_index", "page_url", "page_title", "source_url", "error"],
    )
    unique_files = {
        record.get("saved_as")
        for record in completed.values()
        if isinstance(record, dict) and record.get("saved_as")
    }

    summary = (
        f"Pages processed: {len(page_rows)}\n"
        f"Input pages: {len(input_urls)}\n"
        f"Unique files stored: {len(unique_files)}\n"
        f"Successful source URLs: {len(completed)}\n"
        f"Failed source URLs: {len(failed)}\n\n"
        "Files:\n"
        "- 000001.jpg ... downloaded images stored flat in this directory\n"
        "- manifest.csv   one row per extracted source URL mapped to a local file\n"
        "- pages.csv      per-page extraction summary\n"
        "- failed_urls.csv failed image downloads\n"
        "- source_pages.txt input gallery pages\n"
        "- state.json     resumable downloader state\n"
    )
    (output_dir / "README.txt").write_text(summary, encoding="utf-8")


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.quarantine_dir is None:
        args.quarantine_dir = args.output / "quarantine"

    session = make_session()
    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / "state.json"
    state = load_state(state_path)

    file_rows: list[dict] = []
    page_rows = read_csv_rows(args.output / "pages.csv")
    failure_rows: list[dict] = []
    had_errors = False
    page_urls: list[str] = []

    if args.recheck_existing:
        checked_files, rejected_count = recheck_existing_dataset(args, state)
        log(f"[recheck] checked={checked_files} rejected={rejected_count}")

    try:
        page_urls = collect_page_urls(session, args)
    except Exception as exc:
        if not args.recheck_existing:
            print(f"Failed to collect gallery URLs: {exc}", file=sys.stderr)
            return 2
        had_errors = True
        log(f"[collect] error :: {exc}")

    if not page_urls and not args.recheck_existing:
        print("No gallery URLs to process.", file=sys.stderr)
        return 2

    if page_urls and args.discovered_urls_out:
        args.discovered_urls_out.parent.mkdir(parents=True, exist_ok=True)
        args.discovered_urls_out.write_text("\n".join(page_urls) + "\n", encoding="utf-8")

    for page_index, page_url in enumerate(page_urls, start=1):
        if args.target_unique_files > 0 and current_unique_file_count(state) >= args.target_unique_files:
            log(f"Target reached: {current_unique_file_count(state)} unique files")
            break
        try:
            process_page(session, page_index, page_url, args, state, file_rows, page_rows, failure_rows)
        except Exception as exc:
            had_errors = True
            log(f"[{page_index}] error {page_url} :: {exc}")
            page_rows.append(
                {
                    "page_index": page_index,
                    "page_url": page_url,
                    "title": "",
                    "candidate_images": 0,
                    "downloaded_new_files": 0,
                    "reused_existing": 0,
                    "failed": 0,
                    "status": f"page_error: {exc}",
                }
            )
            save_state(state_path, state)

    if any(row["status"] != "ok" for row in page_rows):
        had_errors = True
    if state.get("failed"):
        had_errors = True

    output_page_urls = page_urls
    if not output_page_urls and (args.output / "source_pages.txt").exists():
        output_page_urls = read_urls_from_file(args.output / "source_pages.txt")

    write_outputs(args.output, output_page_urls, file_rows, page_rows, failure_rows, state)
    save_state(state_path, state)

    unique_files = {
        record.get("saved_as")
        for record in state.get("completed", {}).values()
        if isinstance(record, dict) and record.get("saved_as")
    }
    log(
        f"Done. Pages={len(page_rows)} unique_files={len(unique_files)} "
        f"completed_urls={len(state.get('completed', {}))} failed_urls={len(state.get('failed', {}))}"
    )
    log(f"Output: {args.output.resolve()}")
    return 1 if had_errors else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        raise SystemExit(130)
