#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Scrape interior images from a Divan.ru article.

The scraper keeps only article interior images (`wikidivania-article__image`)
and skips product-card gallery images.
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
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Tag


DEFAULT_URL = "https://www.divan.ru/idei-i-trendy/budzetno-i-stilno-dizajnerskie-resenia-dla-interera-kotorye-smelo-mozno-primenat-v-svoej-kvartire"
DEFAULT_OUT_DIR = "data/input/divan_ru_indoor"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/121.0.0.0 Safari/537.36"
)

ROOM_KEYWORDS: dict[str, tuple[str, ...]] = {
    "bedroom": ("спальн", "изголов", "кровать", "bedroom"),
    "kitchen": ("кухн", "обеденн", "столов", "фартук", "kitchen"),
    "living_room": ("гостин", "диван", "журнальн", "living", "sofa"),
    "bathroom": ("ванн", "туалет", "сануз", "кафель", "плитк", "bathroom", "toilet"),
    "home_office": ("кабинет", "рабоч", "office"),
}


@dataclass
class InteriorImage:
    id: str
    source_url: str
    article_url: str
    article_title: str
    image_url: str
    local_path: str = ""
    room_type: str = "interior"
    section_title: str = ""
    caption: str = ""
    photo_source: str = ""
    description: str = ""
    image_alt: str = ""
    width: int | None = None
    height: int | None = None
    status: str = "pending"
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


def stable_hash(value: str, n: int = 10) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()[:n]


def normalize_url(url: str, base_url: str = DEFAULT_URL) -> str:
    url = norm_text(url)
    if not url or url.startswith(("data:", "javascript:", "mailto:", "tel:")):
        return ""
    return urljoin(base_url, url)


def infer_ext(url: str, content_type: str = "") -> str:
    ext = Path(urlparse(url).path).suffix.lower()
    if ext in {".jpg", ".jpeg", ".png", ".webp", ".avif"}:
        return ext
    mime = content_type.split(";", 1)[0].strip().lower()
    return mimetypes.guess_extension(mime) or ".jpg"


def make_session() -> requests.Session:
    sess = requests.Session()
    sess.headers.update(
        {
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.7",
        }
    )
    return sess


def fetch_html(sess: requests.Session, url: str, retries: int, sleep: float) -> str:
    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            resp = sess.get(url, timeout=(20, 70))
            if resp.status_code == 404:
                raise RuntimeError("HTTP 404 from Divan; use --html-file with saved/article HTML")
            resp.raise_for_status()
            return resp.text
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(sleep * attempt)
    raise RuntimeError(f"fetch failed for {url}: {last_error}")


def article_title(soup: BeautifulSoup) -> str:
    h1 = soup.find("h1")
    if h1:
        return norm_text(h1.get_text(" ", strip=True))
    title = soup.find("title")
    return norm_text(title.get_text(" ", strip=True) if title else "")


def is_article_image(img: Tag) -> bool:
    classes = img.get("class") or []
    if any("wikidivania-article__image" == str(cls) for cls in classes):
        return True
    src = normalize_url(str(img.get("src") or ""))
    if "cdn.divan.ru/img/" not in src:
        return False
    class_text = " ".join(str(x) for x in classes)
    if "ProductImage" in class_text or "Img_image" in class_text:
        return False
    encoded = src.rsplit("/", 1)[-1]
    return "wiki" in src or encoded.endswith(".jpg")


def nearest_caption(img: Tag) -> str:
    parent = img.find_parent("p") or img
    node = parent.find_next_sibling()
    if not isinstance(node, Tag):
        container = parent.find_parent("div")
        node = container.find_next_sibling() if container else None
    for _ in range(3):
        if not isinstance(node, Tag):
            break
        text = norm_text(node.get_text(" ", strip=True))
        if text.lower().startswith("фото:"):
            return text
        if text and not node.find("img"):
            break
        node = node.find_next_sibling()
    return ""


def infer_photo_source(caption: str) -> str:
    text = norm_text(caption)
    if text.lower().startswith("фото:"):
        return norm_text(text.split(":", 1)[1])
    return ""


def infer_room_type(*texts: str) -> str:
    haystack = lower(" ".join(texts))
    hits: list[str] = []
    for room_type, keywords in ROOM_KEYWORDS.items():
        if any(keyword in haystack for keyword in keywords):
            hits.append(room_type)
    if "bathroom" in hits:
        return "bathroom"
    if "kitchen" in hits:
        return "kitchen"
    if "bedroom" in hits:
        return "bedroom"
    if "living_room" in hits:
        return "living_room"
    return hits[0] if hits else "interior"


def parse_article(html_text: str, article_url: str) -> list[InteriorImage]:
    soup = BeautifulSoup(html_text, "html.parser")
    title = article_title(soup)
    body = soup.select_one('[itemprop="articleBody"]') or soup
    rows: list[InteriorImage] = []
    current_section = ""
    last_paragraph = ""
    seen_urls: set[str] = set()

    for node in body.descendants:
        if not isinstance(node, Tag):
            continue
        if node.name in {"h2", "h3"}:
            current_section = norm_text(node.get_text(" ", strip=True))
            continue
        if node.name == "p" and not node.find("img"):
            text = norm_text(node.get_text(" ", strip=True))
            if text and not text.lower().startswith("фото:"):
                last_paragraph = text
            continue
        if node.name != "img" or not is_article_image(node):
            continue
        image_url = normalize_url(str(node.get("src") or ""), article_url)
        if not image_url or image_url in seen_urls:
            continue
        seen_urls.add(image_url)
        caption = nearest_caption(node)
        section = current_section
        description = last_paragraph
        room_type = infer_room_type(section, description, caption, norm_text(node.get("alt")))
        row_id = f"divan_ru_indoor_{len(rows) + 1:03d}_{stable_hash(image_url)}"
        rows.append(
            InteriorImage(
                id=row_id,
                source_url=article_url,
                article_url=article_url,
                article_title=title,
                image_url=image_url,
                room_type=room_type,
                section_title=section,
                caption=caption,
                photo_source=infer_photo_source(caption),
                description=description,
                image_alt=norm_text(node.get("alt")),
            )
        )
    return rows


def download_with_requests(sess: requests.Session, row: InteriorImage, target: Path, retries: int, sleep: float) -> tuple[bool, str]:
    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            with sess.get(row.image_url, headers={"Referer": row.article_url, "Accept": "image/avif,image/webp,image/*,*/*;q=0.8"}, stream=True, timeout=(20, 90)) as resp:
                resp.raise_for_status()
                content_type = resp.headers.get("Content-Type", "")
                if not content_type.lower().startswith("image/"):
                    raise RuntimeError(f"unexpected content-type: {content_type}")
                tmp = target.with_suffix(target.suffix + ".part")
                with tmp.open("wb") as handle:
                    for chunk in resp.iter_content(chunk_size=1024 * 128):
                        if chunk:
                            handle.write(chunk)
                tmp.replace(target)
                return True, ""
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(sleep * attempt)
    return False, last_error


def download_with_curl(row: InteriorImage, target: Path, retries: int, sleep: float) -> tuple[bool, str]:
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
        f"Referer: {row.article_url}",
        "-H",
        "Accept: image/avif,image/webp,image/*,*/*;q=0.8",
        "-o",
        str(tmp),
        row.image_url,
    ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        return False, (proc.stderr or "").strip() or f"curl exited {proc.returncode}"
    tmp.replace(target)
    return True, ""


def download_rows(sess: requests.Session, rows: list[InteriorImage], out_dir: Path, retries: int, sleep: float, method: str) -> None:
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for idx, row in enumerate(rows, start=1):
        ext = infer_ext(row.image_url)
        target = images_dir / f"{idx:03d}_{row.room_type}_{stable_hash(row.image_url)}{ext}"
        if target.exists():
            row.local_path = str(target.relative_to(out_dir))
            row.status = "exists"
            try:
                from PIL import Image

                with Image.open(target) as img:
                    row.width, row.height = img.size
            except Exception:
                pass
            continue
        eprint(f"[image] {idx}/{len(rows)} {row.image_url}")
        ok, error = download_with_requests(sess, row, target, retries, sleep) if method in {"auto", "requests"} else (False, "")
        if not ok and method in {"auto", "curl"}:
            ok, error = download_with_curl(row, target, retries, sleep)
        if ok:
            row.local_path = str(target.relative_to(out_dir))
            row.status = "ok"
            try:
                from PIL import Image

                with Image.open(target) as img:
                    row.width, row.height = img.size
            except Exception:
                pass
        else:
            row.status = "error"
            row.error = error
        time.sleep(sleep)


def write_json(path: Path, data: Any) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "id",
        "room_type",
        "local_path",
        "image_url",
        "article_title",
        "section_title",
        "caption",
        "photo_source",
        "description",
        "width",
        "height",
        "status",
        "error",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scrape interior images from a Divan.ru article.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--html-file", default="")
    parser.add_argument("--out", default=DEFAULT_OUT_DIR)
    parser.add_argument("--download-method", choices=["auto", "requests", "curl"], default="auto")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=0.2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    article_url = args.url
    sess = make_session()
    if args.html_file:
        html_text = Path(args.html_file).read_text(encoding="utf-8", errors="ignore")
    else:
        html_text = fetch_html(sess, article_url, retries=args.retries, sleep=args.sleep)
    (out_dir / "source.html").write_text(html_text, encoding="utf-8")
    rows = parse_article(html_text, article_url)
    eprint(f"[parse] article_images={len(rows)}")
    if not args.no_download:
        download_rows(sess, rows, out_dir, retries=args.retries, sleep=args.sleep, method=args.download_method)
    raw_rows = [asdict(row) for row in rows]
    export = {
        "schema": "divan_ru_article_interiors/v1",
        "meta": {
            "source_url": article_url,
            "parsed_at": now_iso(),
            "image_count": len(rows),
            "downloaded_count": sum(1 for row in rows if row.status in {"ok", "exists"}),
        },
        "items": raw_rows,
    }
    write_json(out_dir / "divan_ru_indoor_scrape.json", export)
    write_jsonl(out_dir / "divan_ru_indoor_images.jsonl", raw_rows)
    write_csv(out_dir / "divan_ru_indoor_images.csv", raw_rows)
    eprint(f"[out] {out_dir / 'divan_ru_indoor_scrape.json'}")
    eprint(f"[done] images={len(rows)} downloaded={export['meta']['downloaded_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
