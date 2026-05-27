#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Download Mosplitka product images from already saved product HTML.

This is intentionally separate from mosplitka_tile_scraper.py so images can be
downloaded while the main scraper is still collecting product pages.
"""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from src.tools.mosplitka_tile_scraper import (
    DEFAULT_USER_AGENT,
    clean_filename,
    extract_jsonld_product,
    guess_extension_from_response,
    normalize_image_url,
    product_slug_from_url,
    read_urls,
    stable_hash,
)


def load_product_urls(root: Path) -> dict[str, str]:
    path = root / "product_urls.txt"
    if not path.exists():
        return {}
    urls = read_urls(path)
    return {product_slug_from_url(url): url for url in urls}


def jsonld_images_from_html(path: Path) -> tuple[dict[str, Any], list[str]]:
    html_text = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html_text, "html.parser")
    product = extract_jsonld_product(soup) or {}
    raw = product.get("image")
    images: list[str] = []
    if isinstance(raw, str):
        images.append(raw)
    elif isinstance(raw, list):
        images.extend(x for x in raw if isinstance(x, str))
    seen: set[str] = set()
    out: list[str] = []
    for image in images:
        image_url = normalize_image_url(image)
        parsed = urlparse(image_url)
        if "cdn.mosplitka.ru" not in parsed.netloc:
            continue
        if Path(parsed.path).suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        if image_url not in seen:
            seen.add(image_url)
            out.append(image_url)
    return product, out


def target_for_image(out_root: Path, html_path: Path, product: dict[str, Any], image_url: str, idx: int) -> Path:
    sku = str(product.get("sku") or "").strip()
    name = str(product.get("name") or html_path.stem).strip()
    product_dir = out_root / clean_filename(f"{sku or stable_hash(html_path.stem)}_{name}", max_len=120)
    return product_dir / f"{idx:02d}_{stable_hash(image_url)}"


def download_one(
    image_url: str,
    target_base: Path,
    referer: str,
    retries: int,
    timeout: tuple[int, int],
    lock: threading.Lock,
) -> tuple[str, str]:
    target_base.parent.mkdir(parents=True, exist_ok=True)
    existing = [p for p in target_base.parent.glob(target_base.name + ".*") if not p.name.endswith(".part")]
    if existing:
        return str(existing[0]), "exists"

    session = requests.Session()
    session.headers.update({"User-Agent": DEFAULT_USER_AGENT, "Accept": "image/avif,image/webp,image/*,*/*;q=0.8"})
    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            with session.get(image_url, headers={"Referer": referer}, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                ext = guess_extension_from_response(image_url, response.headers.get("Content-Type"))
                if ext == ".bin":
                    ext = mimetypes.guess_extension(response.headers.get("Content-Type", "").split(";")[0]) or ".jpg"
                target = target_base.with_suffix(ext)
                tmp = target.with_suffix(target.suffix + ".part")
                with tmp.open("wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 128):
                        if chunk:
                            f.write(chunk)
                tmp.replace(target)
                with lock:
                    print(f"[INFO] saved {target}", flush=True)
                return str(target), "ok"
        except Exception as exc:
            last_error = str(exc)
            if attempt == retries:
                return "", last_error
    return "", last_error or "failed"


def build_jobs(root: Path, out_root: Path, limit: int | None = None) -> list[dict[str, Any]]:
    slug_to_url = load_product_urls(root)
    html_paths = sorted((root / "product_html").glob("*.html"))
    if limit:
        html_paths = html_paths[:limit]
    jobs: list[dict[str, Any]] = []
    for html_path in html_paths:
        product, images = jsonld_images_from_html(html_path)
        product_url = slug_to_url.get(html_path.stem, f"https://mosplitka.ru/product/{html_path.stem}/")
        for idx, image_url in enumerate(images, start=1):
            jobs.append(
                {
                    "html_file": str(html_path.relative_to(root)),
                    "product_url": product_url,
                    "product_name": str(product.get("name") or ""),
                    "sku": str(product.get("sku") or ""),
                    "image_index": idx,
                    "image_url": image_url,
                    "target_base": target_for_image(out_root, html_path, product, image_url, idx),
                }
            )
    return jobs


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "html_file",
        "product_url",
        "product_name",
        "sku",
        "image_index",
        "image_url",
        "local_path",
        "status",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with path.with_suffix(".jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Mosplitka JSON-LD product images from saved HTML.")
    parser.add_argument("--root", default="data/floor_materials/mosplitka")
    parser.add_argument("--images-dir", default="")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    root = Path(args.root)
    out_root = Path(args.images_dir) if args.images_dir else root / "images_jsonld"
    manifest = Path(args.manifest) if args.manifest else root / "image_download_manifest.csv"
    jobs = build_jobs(root, out_root, limit=args.limit)
    print(f"[INFO] jobs: {len(jobs)} images from saved HTML", flush=True)

    rows: list[dict[str, Any]] = []
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                download_one,
                job["image_url"],
                job["target_base"],
                job["product_url"],
                args.retries,
                (20, 90),
                lock,
            ): job
            for job in jobs
        }
        for future in as_completed(futures):
            job = futures[future]
            local_path, status = future.result()
            row = dict(job)
            row.pop("target_base", None)
            row["local_path"] = str(Path(local_path).relative_to(root)) if local_path and Path(local_path).is_relative_to(root) else local_path
            row["status"] = status
            rows.append(row)

    rows.sort(key=lambda x: (x["html_file"], int(x["image_index"])))
    write_manifest(manifest, rows)
    ok = sum(1 for row in rows if row["status"] in {"ok", "exists"})
    print(f"[INFO] downloaded/exists: {ok}/{len(rows)}", flush=True)
    print(f"[INFO] manifest: {manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
