# -*- coding: utf-8 -*-
"""
This script parses Dimensiva pages into the legacy ParsedAsset schema.
It supports experimental ingestion outside the main supplier adapters.
The code focuses on portable extraction helpers and format inference.
It is intentionally standalone and loosely coupled to newer workflows.
Keep it compatible with existing JSON exports.
"""
from __future__ import annotations

import argparse
import re

from src.suppliers.common import (
    ParsedAsset,
    compute_blender_ready_score,
    extract_json_ld,
    fetch_html,
    first_jsonld_of_type,
    normalize_category,
    parse_dimensions_from_text,
    polite_sleep,
    save_assets_json,
    soup_from_html,
    text_or_none,
    uniq_keep_order,
)


def infer_formats_from_text(text: str) -> list[str]:
    formats: list[str] = []
    checks = {
        "max": ["3ds max", "vray file", "corona file"],
        "obj": [" obj "],
        "fbx": [" fbx "],
        "blend": [" blend "],
        "glb": [" glb "],
        "gltf": [" gltf "],
        "3ds": [" 3ds "],
    }

    low = f" {text.lower()} "
    for fmt, keys in checks.items():
        if any(key in low for key in keys):
            formats.append(fmt)
    return uniq_keep_order(formats)


def parse_dimensiva_product(url: str) -> ParsedAsset:
    html = fetch_html(url)
    soup = soup_from_html(html)
    jsonlds = extract_json_ld(soup)
    product_ld = first_jsonld_of_type(jsonlds, "Product") or {}

    title = None
    h1 = soup.select_one("h1")
    if h1:
        title = text_or_none(h1)
    if not title:
        title = product_ld.get("name")

    page_text = soup.get_text("\n", strip=True)

    brand = None
    designer = None
    collection = None

    m = re.search(r"designed by\s+(.+?)\s+for\s+(.+?)(?:\.|and launched|$)", page_text, re.I)
    if m:
        designer = m.group(1).strip()
        brand = m.group(2).strip()

    description = None
    paragraphs = [p.get_text(" ", strip=True) for p in soup.select("p")]
    long_paragraphs = [p for p in paragraphs if len(p) > 120]
    if long_paragraphs:
        description = long_paragraphs[0]
    if not description:
        description = product_ld.get("description")

    width_m, depth_m, height_m = parse_dimensions_from_text(page_text)

    download_formats = infer_formats_from_text(page_text)
    if "materials in place ready to render" in page_text.lower() and "max" not in download_formats:
        download_formats.append("max")
    download_formats = uniq_keep_order(download_formats)

    preview_images: list[str] = []
    for img in soup.select("img"):
        src = img.get("src")
        if src and src.startswith("http"):
            preview_images.append(src)
    preview_images = uniq_keep_order(preview_images)[:10]

    price_type = "subscription" if "PRO" in page_text or "pricing" in page_text.lower() else "unknown"

    return ParsedAsset(
        supplier="dimensiva",
        source_url=url,
        title=title,
        brand=brand,
        collection=collection,
        designer=designer,
        category_raw=title,
        category_norm=normalize_category(title),
        description=description,
        width_m=width_m,
        depth_m=depth_m,
        height_m=height_m,
        price_type=price_type,
        download_formats=download_formats,
        blender_ready_score=compute_blender_ready_score(download_formats),
        preview_images=preview_images,
        raw_meta={
            "jsonld_product": product_ld,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urls", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sleep", type=float, default=1.5)
    args = parser.parse_args()

    assets: list[ParsedAsset] = []
    for url in args.urls:
        try:
            assets.append(parse_dimensiva_product(url))
        finally:
            polite_sleep(args.sleep)

    save_assets_json(args.out, assets)


if __name__ == "__main__":
    main()
