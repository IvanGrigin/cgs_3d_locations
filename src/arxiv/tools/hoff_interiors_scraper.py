#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Scrape Hoff interiors/project cards into a room-reference image dataset.

The Hoff interiors page is Vue-driven and may be protected by Qrator for plain
HTTP clients. The scraper therefore supports both direct page crawling and
saved HTML input copied from a browser/devtools session.

Examples:
    # Direct crawl. May fail early if Hoff returns the Qrator challenge page.
    python3 -m src.tools.hoff_interiors_scraper --out data/input/hoff_interiors

    # Parse saved listing HTML and download one image per room type/project.
    python3 -m src.tools.hoff_interiors_scraper \
      --html-file /tmp/hoff_interiors.html \
      --out data/input/hoff_interiors \
      --per-room 10

    # Use manually collected project URLs.
    python3 -m src.tools.hoff_interiors_scraper \
      --project-url-file data/input/hoff_project_urls.txt \
      --out data/input/hoff_interiors
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
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://hoff.ru"
DEFAULT_INTERIORS_URL = "https://hoff.ru/interiors/"
DEFAULT_OUT_DIR = "data/input/hoff_interiors"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/121.0.0.0 Safari/537.36"
)
DEFAULT_ROOM_TYPES = ["bedroom", "kitchen", "living_room", "toilet", "bathroom"]

ROOM_KEYWORDS: dict[str, tuple[str, ...]] = {
    "bedroom": ("спальн", "bedroom"),
    "kitchen": ("кухн", "кухня", "kitchen"),
    "living_room": ("гостин", "кухня-гостиная", "кухня гостиная", "living", "lounge"),
    "toilet": ("туалет", "санузел", "wc", "toilet"),
    "bathroom": ("ванн", "душев", "bathroom", "shower"),
}
ROOM_ALIASES = {
    "living": "living_room",
    "livingroom": "living_room",
    "гостиная": "living_room",
    "гостинная": "living_room",
    "спальня": "bedroom",
    "кухня": "kitchen",
    "туалет": "toilet",
    "ванная": "bathroom",
}


@dataclass
class HoffProject:
    project_url: str = ""
    title: str = ""
    tags: list[str] = field(default_factory=list)
    params: list[str] = field(default_factory=list)
    listing_image_urls: list[str] = field(default_factory=list)
    detail_image_urls: list[str] = field(default_factory=list)
    source_page_url: str = ""
    raw_status: str = "ok"
    error: str = ""


@dataclass
class HoffImageRow:
    id: str
    room_type: str
    project_url: str
    project_title: str
    image_url: str
    source: str
    tags: list[str] = field(default_factory=list)
    params: list[str] = field(default_factory=list)
    description: str = ""
    local_path: str = ""
    download_status: str = "pending"
    error: str = ""


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr, flush=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def norm_text(value: Any) -> str:
    text = html.unescape(str(value or "")).replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def lower(value: Any) -> str:
    return norm_text(value).lower().replace("ё", "е")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def stable_hash(value: str, n: int = 12) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()[:n]


def slugify(value: str, max_len: int = 80) -> str:
    value = lower(value)
    value = re.sub(r"[^a-zа-я0-9]+", "_", value, flags=re.IGNORECASE).strip("_")
    return (value[:max_len].strip("_") or "item")


def unique_list(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        value = norm_text(value)
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def normalize_url(url: str, *, keep_query: bool = False) -> str:
    url = norm_text(url)
    if not url or url.startswith(("data:", "javascript:", "mailto:", "tel:")):
        return ""
    parsed = urlparse(urljoin(BASE_URL, url))
    query = parsed.query if keep_query else ""
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path or "/", "", query, ""))


def infer_extension(url: str, content_type: str = "") -> str:
    path_ext = Path(urlparse(url).path).suffix.lower()
    if path_ext in {".jpg", ".jpeg", ".png", ".webp", ".avif"}:
        return path_ext
    mime = content_type.split(";", 1)[0].strip().lower()
    return mimetypes.guess_extension(mime) or ".jpg"


def session() -> requests.Session:
    sess = requests.Session()
    sess.headers.update(
        {
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.6",
        }
    )
    return sess


def fetch_text(sess: requests.Session, url: str, *, retries: int = 3, sleep: float = 0.5) -> str:
    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            resp = sess.get(url, timeout=(20, 70))
            text = resp.text or ""
            if resp.status_code == 200 and "__qrator/qauth.js" in text:
                raise RuntimeError("Hoff returned Qrator challenge HTML; use --html-file or --project-url-file from a browser session")
            resp.raise_for_status()
            return text
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(sleep * attempt)
    raise RuntimeError(f"fetch failed for {url}: {last_error}")


def srcset_urls(srcset: str) -> list[str]:
    urls: list[str] = []
    for chunk in (srcset or "").split(","):
        first = chunk.strip().split(" ", 1)[0]
        if first:
            urls.append(first)
    return urls


def image_urls_from_node(node: Any) -> list[str]:
    urls: list[str] = []
    for attr in ("src", "data-src", "data-original", "data-lazy", "href"):
        value = node.get(attr) if hasattr(node, "get") else None
        if value:
            urls.append(str(value))
    for attr in ("srcset", "data-srcset"):
        value = node.get(attr) if hasattr(node, "get") else None
        if value:
            urls.extend(srcset_urls(str(value)))
    return [url for url in (normalize_url(x, keep_query=True) for x in urls) if is_hoff_image_url(url)]


def is_hoff_image_url(url: str) -> bool:
    parsed = urlparse(url)
    if not parsed.netloc.endswith("hoff.ru"):
        return False
    path = parsed.path.lower()
    return "/upload/" in path and any(ext in path for ext in (".jpg", ".jpeg", ".png", ".webp", ".avif"))


def parse_cards_from_html(html_text: str, source_page_url: str) -> list[HoffProject]:
    soup = BeautifulSoup(html_text, "html.parser")
    cards: list[HoffProject] = []
    for card in soup.select(".card.page__project, .page__project"):
        links = [a for a in card.find_all("a", href=True) if "/interiors/projects/" in str(a.get("href"))]
        if not links:
            continue
        project_url = normalize_url(str(links[0].get("href")))
        title_node = card.select_one(".card__name")
        title = norm_text(title_node.get_text(" ", strip=True) if title_node else "")
        if not title:
            title = norm_text(links[-1].get_text(" ", strip=True))
        tags = unique_list(x.get_text(" ", strip=True) for x in card.select(".card__tag, .hoff-label"))
        params = unique_list(x.get_text(" ", strip=True) for x in card.select(".params__text"))
        images: list[str] = []
        for node in card.select("img, source, a"):
            images.extend(image_urls_from_node(node))
        cards.append(
            HoffProject(
                project_url=project_url,
                title=title,
                tags=tags,
                params=params,
                listing_image_urls=unique_list(images),
                source_page_url=source_page_url,
            )
        )
    if cards:
        return dedupe_projects(cards)
    return fallback_projects_from_html(html_text, source_page_url)


def fallback_projects_from_html(html_text: str, source_page_url: str) -> list[HoffProject]:
    urls = unique_list(normalize_url(x) for x in re.findall(r'["\']([^"\']*/interiors/projects/[^"\']+)["\']', html_text))
    images = unique_list(
        normalize_url(x, keep_query=True)
        for x in re.findall(r'["\']([^"\']*/upload/[^"\']+\.(?:jpe?g|png|webp|avif)(?:/[^"\']*)?)["\']', html_text, flags=re.IGNORECASE)
    )
    projects: list[HoffProject] = []
    for index, url in enumerate(urls):
        title = slugify(Path(urlparse(url).path.rstrip("/")).name).replace("_", " ")
        projects.append(
            HoffProject(
                project_url=url,
                title=title,
                listing_image_urls=images if len(urls) == 1 else [],
                source_page_url=source_page_url,
            )
        )
    return dedupe_projects(projects)


def dedupe_projects(projects: Iterable[HoffProject]) -> list[HoffProject]:
    by_url: dict[str, HoffProject] = {}
    for project in projects:
        if not project.project_url:
            continue
        existing = by_url.get(project.project_url)
        if not existing:
            by_url[project.project_url] = project
            continue
        existing.title = existing.title or project.title
        existing.tags = unique_list([*existing.tags, *project.tags])
        existing.params = unique_list([*existing.params, *project.params])
        existing.listing_image_urls = unique_list([*existing.listing_image_urls, *project.listing_image_urls])
    return list(by_url.values())


def parse_detail_html(project: HoffProject, html_text: str) -> HoffProject:
    soup = BeautifulSoup(html_text, "html.parser")
    title_node = soup.find("h1") or soup.select_one(".project__title, .page-title")
    if title_node:
        project.title = project.title or norm_text(title_node.get_text(" ", strip=True))
    page_text = soup.get_text(" ", strip=True)
    tags = list(project.tags)
    for room_type, keywords in ROOM_KEYWORDS.items():
        if any(keyword in lower(page_text) for keyword in keywords):
            tags.append(room_type)
    for node in soup.select(".hoff-label, .tag, [class*=tag], [class*=label]"):
        text = norm_text(node.get_text(" ", strip=True))
        if 1 <= len(text) <= 80:
            tags.append(text)
    project.tags = unique_list(tags)

    images: list[str] = []
    for node in soup.select("img, source, a"):
        images.extend(image_urls_from_node(node))
    project.detail_image_urls = unique_list(images)
    return project


def collect_projects(
    sess: requests.Session,
    *,
    listing_url: str,
    html_files: list[Path],
    project_urls: list[str],
    fetch_details: bool,
    sleep: float,
    retries: int,
) -> list[HoffProject]:
    projects: list[HoffProject] = []
    if html_files:
        for path in html_files:
            eprint(f"[html] {path}")
            if not path.exists():
                raise FileNotFoundError(
                    f"HTML file does not exist: {path}. Pass a real saved Hoff page path, "
                    "for example data/input/hoff_interiors_full.html, not a placeholder."
                )
            projects.extend(parse_cards_from_html(path.read_text(encoding="utf-8", errors="ignore"), str(path)))
    elif not project_urls:
        eprint(f"[list] {listing_url}")
        html_text = fetch_text(sess, listing_url, retries=retries, sleep=sleep)
        projects.extend(parse_cards_from_html(html_text, listing_url))

    for url in project_urls:
        projects.append(HoffProject(project_url=normalize_url(url), source_page_url="project_url_file"))
    projects = dedupe_projects(projects)
    eprint(f"[projects] collected={len(projects)}")

    if not fetch_details:
        return projects

    for index, project in enumerate(projects, start=1):
        try:
            eprint(f"[detail] {index}/{len(projects)} {project.project_url}")
            html_text = fetch_text(sess, project.project_url, retries=retries, sleep=sleep)
            parse_detail_html(project, html_text)
        except Exception as exc:
            project.raw_status = "error"
            project.error = f"{type(exc).__name__}: {exc}"
            eprint(f"[detail:error] {project.project_url}: {project.error}")
        time.sleep(sleep)
    return projects


def classify_project(project: HoffProject) -> list[str]:
    haystack = lower(" ".join([project.title, *project.tags, *project.params, project.project_url]))
    room_types = [room for room, keywords in ROOM_KEYWORDS.items() if any(keyword in haystack for keyword in keywords)]
    for alias, room_type in ROOM_ALIASES.items():
        if lower(alias) in haystack and room_type not in room_types:
            room_types.append(room_type)
    return [room for room in DEFAULT_ROOM_TYPES if room in room_types]


def choose_rows(projects: list[HoffProject], room_types: list[str], per_room: int, images_per_project: int) -> list[HoffImageRow]:
    rows: list[HoffImageRow] = []
    counts = {room: 0 for room in room_types}
    used_project_by_room: dict[str, set[str]] = {room: set() for room in room_types}
    for project in projects:
        matched_rooms = [room for room in classify_project(project) if room in counts]
        if not matched_rooms:
            continue
        images = unique_list([*project.detail_image_urls, *project.listing_image_urls])
        if not images:
            continue
        for room in matched_rooms:
            if counts[room] >= per_room or project.project_url in used_project_by_room[room]:
                continue
            for image_index, image_url in enumerate(images[: max(1, images_per_project)], start=1):
                if counts[room] >= per_room:
                    break
                row_id = f"hoff_{room}_{counts[room] + 1:02d}_{stable_hash(project.project_url + image_url)}"
                rows.append(
                    HoffImageRow(
                        id=row_id,
                        room_type=room,
                        project_url=project.project_url,
                        project_title=project.title,
                        image_url=image_url,
                        source="detail" if image_url in project.detail_image_urls else "listing",
                        tags=project.tags,
                        params=project.params,
                    )
                )
                counts[room] += 1
            used_project_by_room[room].add(project.project_url)
    eprint("[select] " + " ".join(f"{room}={counts[room]}/{per_room}" for room in room_types))
    return rows


def download_image(sess: requests.Session, row: HoffImageRow, out_dir: Path, retries: int, sleep: float) -> HoffImageRow:
    room_dir = out_dir / "images" / row.room_type
    ensure_dir(room_dir)
    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            with sess.get(
                row.image_url,
                headers={"Referer": row.project_url, "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8"},
                stream=True,
                timeout=(20, 90),
            ) as resp:
                if resp.status_code != 200:
                    raise RuntimeError(f"HTTP {resp.status_code}")
                content_type = resp.headers.get("Content-Type", "")
                if not content_type.lower().startswith("image/"):
                    raise RuntimeError(f"unexpected content-type: {content_type}")
                ext = infer_extension(row.image_url, content_type)
                target = room_dir / f"{row.id}{ext}"
                tmp = target.with_suffix(target.suffix + ".part")
                with tmp.open("wb") as handle:
                    for chunk in resp.iter_content(chunk_size=1024 * 128):
                        if chunk:
                            handle.write(chunk)
                tmp.replace(target)
                row.local_path = str(target.relative_to(out_dir))
                row.download_status = "ok"
                return row
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(sleep * attempt)
    row.download_status = "error"
    row.error = last_error
    return row


def download_image_with_curl(row: HoffImageRow, out_dir: Path, retries: int, sleep: float) -> HoffImageRow:
    room_dir = out_dir / "images" / row.room_type
    ensure_dir(room_dir)
    ext = infer_extension(row.image_url)
    target = room_dir / f"{row.id}{ext}"
    tmp = target.with_suffix(target.suffix + ".part")
    cmd = [
        "curl",
        "-L",
        "--fail",
        "--retry",
        str(max(0, retries - 1)),
        "--retry-delay",
        str(max(1, int(sleep))),
        "-H",
        f"Referer: {row.project_url or DEFAULT_INTERIORS_URL}",
        "-H",
        "Accept: image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "-o",
        str(tmp),
        row.image_url,
    ]
    try:
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        if proc.returncode != 0:
            raise RuntimeError((proc.stderr or "").strip() or f"curl exited {proc.returncode}")
        tmp.replace(target)
        row.local_path = str(target.relative_to(out_dir))
        row.download_status = "ok"
    except Exception as exc:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        row.download_status = "error"
        row.error = f"{type(exc).__name__}: {exc}"
    return row


def download_selected_image(sess: requests.Session, row: HoffImageRow, out_dir: Path, retries: int, sleep: float, method: str) -> HoffImageRow:
    if method == "curl":
        return download_image_with_curl(row, out_dir, retries=retries, sleep=sleep)
    row = download_image(sess, row, out_dir, retries=retries, sleep=sleep)
    if method == "auto" and row.download_status != "ok":
        eprint(f"[image:fallback-curl] {row.image_url} after {row.error}")
        return download_image_with_curl(row, out_dir, retries=retries, sleep=sleep)
    return row


def write_json(path: Path, data: Any) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["id", "room_type", "project_title", "project_url", "image_url", "local_path", "download_status", "tags", "params", "error"]
    fields = ["id", "room_type", "project_title", "project_url", "image_url", "local_path", "download_status", "tags", "params", "description", "error"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            item = dict(row)
            item["tags"] = "; ".join(item.get("tags") or [])
            item["params"] = "; ".join(item.get("params") or [])
            writer.writerow(item)


def normalize_room_types(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        key = ROOM_ALIASES.get(lower(value), lower(value).replace("-", "_"))
        if key not in ROOM_KEYWORDS:
            raise ValueError(f"unknown room type {value!r}; expected one of {', '.join(DEFAULT_ROOM_TYPES)}")
        if key not in out:
            out.append(key)
    return out or list(DEFAULT_ROOM_TYPES)


def read_url_file(path: str) -> list[str]:
    if not path:
        return []
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#")]


def read_image_url_file(path: str) -> list[HoffImageRow]:
    if not path:
        return []
    source = Path(path)
    rows: list[HoffImageRow] = []
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        if "," in sample.splitlines()[0]:
            reader = csv.DictReader(handle)
            for index, item in enumerate(reader, start=1):
                room_type = ROOM_ALIASES.get(lower(item.get("room_type")), lower(item.get("room_type")).replace("-", "_"))
                image_url = normalize_url(item.get("image_url") or item.get("url") or "", keep_query=True)
                if room_type not in ROOM_KEYWORDS or not image_url:
                    continue
                row_id = norm_text(item.get("id")) or f"hoff_{room_type}_{index:03d}_{stable_hash(image_url)}"
                rows.append(
                    HoffImageRow(
                        id=row_id,
                        room_type=room_type,
                        project_url=normalize_url(item.get("project_url") or DEFAULT_INTERIORS_URL, keep_query=True),
                        project_title=norm_text(item.get("project_title") or item.get("title")),
                        image_url=image_url,
                        source=norm_text(item.get("source") or "manual_url_file"),
                        tags=unique_list((item.get("tags") or "").split(";")),
                        description=norm_text(item.get("description")),
                    )
                )
            return rows
        for index, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [part.strip() for part in re.split(r"[\t,]", line, maxsplit=2)]
            if len(parts) < 2:
                continue
            room_type = ROOM_ALIASES.get(lower(parts[0]), lower(parts[0]).replace("-", "_"))
            image_url = normalize_url(parts[1], keep_query=True)
            if room_type not in ROOM_KEYWORDS or not image_url:
                continue
            rows.append(
                HoffImageRow(
                    id=f"hoff_{room_type}_{index:03d}_{stable_hash(image_url)}",
                    room_type=room_type,
                    project_url=DEFAULT_INTERIORS_URL,
                    project_title=parts[2] if len(parts) > 2 else "",
                    image_url=image_url,
                    source="manual_url_file",
                )
            )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scrape Hoff interiors room-reference images.")
    parser.add_argument("--listing-url", default=DEFAULT_INTERIORS_URL)
    parser.add_argument("--html-file", action="append", default=[], help="Saved Hoff listing/detail HTML. Can be passed multiple times.")
    parser.add_argument("--project-url", action="append", default=[], help="Hoff project URL. Can be passed multiple times.")
    parser.add_argument("--project-url-file", default="", help="Text file with one Hoff project URL per line.")
    parser.add_argument("--image-url-file", default="", help="CSV with room_type,image_url[,project_url,project_title] for direct image downloads.")
    parser.add_argument("--out", default=DEFAULT_OUT_DIR)
    parser.add_argument("--room-type", action="append", default=[], help="Target room type. Defaults to bedroom/kitchen/living_room/toilet/bathroom.")
    parser.add_argument("--per-room", type=int, default=10, help="Number of distinct project images per room type.")
    parser.add_argument("--images-per-project", type=int, default=1, help="Max selected images from one project per room type.")
    parser.add_argument("--no-detail", action="store_true", help="Only use listing/saved HTML images; do not fetch project detail pages.")
    parser.add_argument("--no-download", action="store_true", help="Only write URL manifests.")
    parser.add_argument("--download-method", choices=["auto", "requests", "curl"], default="auto", help="Image download backend. auto uses requests then curl fallback.")
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--retries", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out)
    ensure_dir(out_dir)
    parsed_at = now_iso()

    room_types = normalize_room_types(args.room_type)
    html_files = [Path(x) for x in args.html_file]
    project_urls = [*args.project_url, *read_url_file(args.project_url_file)]

    sess = session()
    projects: list[HoffProject] = []
    try:
        if args.image_url_file:
            rows = [row for row in read_image_url_file(args.image_url_file) if row.room_type in room_types]
            per_room_seen: dict[str, int] = {room: 0 for room in room_types}
            limited_rows: list[HoffImageRow] = []
            for row in rows:
                if per_room_seen[row.room_type] >= max(1, args.per_room):
                    continue
                per_room_seen[row.room_type] += 1
                limited_rows.append(row)
            rows = limited_rows
            eprint("[image-url-file] " + " ".join(f"{room}={per_room_seen[room]}/{args.per_room}" for room in room_types))
        else:
            projects = collect_projects(
                sess,
                listing_url=normalize_url(args.listing_url, keep_query=True),
                html_files=html_files,
                project_urls=project_urls,
                fetch_details=not args.no_detail,
                sleep=args.sleep,
                retries=args.retries,
            )
            rows = choose_rows(projects, room_types, max(1, args.per_room), max(1, args.images_per_project))
    except FileNotFoundError as exc:
        eprint(f"[error] {exc}")
        return 2
    if not args.no_download:
        for index, row in enumerate(rows, start=1):
            eprint(f"[image] {index}/{len(rows)} {row.room_type} {row.image_url}")
            download_selected_image(sess, row, out_dir, retries=args.retries, sleep=args.sleep, method=args.download_method)
            time.sleep(args.sleep)

    project_rows = [asdict(project) for project in projects]
    image_rows = [asdict(row) for row in rows]
    export = {
        "schema": "hoff_interiors_scrape/v1",
        "meta": {
            "source_site": "hoff.ru/interiors",
            "parsed_at": parsed_at,
            "room_types": room_types,
            "per_room": args.per_room,
            "project_count": len(projects),
            "selected_image_count": len(rows),
            "downloaded_image_count": sum(1 for row in rows if row.download_status == "ok"),
        },
        "projects": project_rows,
        "images": image_rows,
    }
    write_json(out_dir / "hoff_interiors_scrape.json", export)
    write_jsonl(out_dir / "hoff_interiors_images.jsonl", image_rows)
    write_csv(out_dir / "hoff_interiors_images.csv", image_rows)
    eprint(f"[out] {out_dir / 'hoff_interiors_scrape.json'}")
    eprint(f"[out] {out_dir / 'hoff_interiors_images.jsonl'}")
    eprint(f"[done] selected={len(rows)} downloaded={export['meta']['downloaded_image_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
