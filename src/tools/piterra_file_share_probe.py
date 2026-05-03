#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Collect metadata for Piterra Nextcloud shared asset links without downloading archives."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup


USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
)


@dataclass
class ShareLink:
    url: str
    source_product_url: str = ""
    source_product_name: str = ""
    link_title: str = ""
    share_title: str = ""
    filename: str = ""
    mimetype: str = ""
    filesize_bytes: int | None = None
    filesize_human: str = ""
    download_url: str = ""
    is_folder: bool = False
    status: str = "ok"
    error: str = ""


def norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("\xa0", " ")).strip()


def parse_int(value: str) -> int | None:
    digits = re.sub(r"\D+", "", value or "")
    return int(digits) if digits else None


def human_size(num: int | None) -> str:
    if num is None:
        return ""
    size = float(num)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return str(num)


def hidden_value(soup: BeautifulSoup, name: str) -> str:
    node = soup.select_one(f'input[name="{name}"], input#{name}')
    return norm(node.get("value")) if node else ""


def probe_share(session: requests.Session, link: ShareLink, timeout: float) -> ShareLink:
    try:
        response = session.get(link.url, timeout=timeout)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        link.share_title = norm((soup.select_one("meta[property='og:title']") or {}).get("content", ""))
        link.filename = hidden_value(soup, "filename")
        link.mimetype = hidden_value(soup, "mimetype")
        link.download_url = hidden_value(soup, "downloadURL") or link.url.rstrip("/") + "/download"
        link.filesize_bytes = parse_int(hidden_value(soup, "filesize"))
        link.filesize_human = human_size(link.filesize_bytes)
        link.is_folder = link.mimetype in {"httpd/unix-directory", "inode/directory"}
        if not link.share_title:
            title = soup.select_one("title")
            link.share_title = norm(title.get_text(" ", strip=True) if title else "")
    except Exception as exc:
        link.status = "error"
        link.error = norm(exc)
    return link


def iter_links(products_path: Path) -> list[ShareLink]:
    seen: dict[str, ShareLink] = {}
    with products_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            product = json.loads(line)
            for asset in product.get("asset_links") or []:
                url = norm(asset.get("url"))
                if "file.piterra.ru/s/" not in url:
                    continue
                item = seen.setdefault(url, ShareLink(url=url))
                if not item.source_product_url:
                    item.source_product_url = product.get("url", "")
                    item.source_product_name = product.get("name", "")
                    item.link_title = asset.get("title", "")
    return list(seen.values())


def write_outputs(out_dir: Path, links: list[ShareLink]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "piterra_file_share_links.jsonl").open("w", encoding="utf-8") as f:
        for link in links:
            f.write(json.dumps(asdict(link), ensure_ascii=False) + "\n")
    fields = list(asdict(links[0]).keys()) if links else list(ShareLink.__dataclass_fields__)
    with (out_dir / "piterra_file_share_links.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for link in links:
            writer.writerow(asdict(link))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--products", default="data/floor_materials/piterra/products.jsonl")
    parser.add_argument("--out", default="data/floor_materials/piterra")
    parser.add_argument("--url", action="append", default=[])
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--limit", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    links = [ShareLink(url=url) for url in args.url if url]
    if not links:
        links = iter_links(Path(args.products)) if Path(args.products).exists() else []
    if args.limit:
        links = links[: args.limit]
    links = [probe_share(session, link, args.timeout) for link in links]
    write_outputs(Path(args.out), links)
    summary = {
        "links_total": len(links),
        "folders": sum(link.is_folder for link in links),
        "errors": sum(link.status != "ok" for link in links),
        "total_size_bytes": sum(link.filesize_bytes or 0 for link in links),
    }
    summary["total_size_human"] = human_size(summary["total_size_bytes"])
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
