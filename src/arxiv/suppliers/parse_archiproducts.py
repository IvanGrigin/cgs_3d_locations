# -*- coding: utf-8 -*-
"""
This script parses Archiproducts pages into the legacy ParsedAsset schema.
It is a standalone helper for one-off catalog extraction experiments.
The logic predates the current adapter-based supplier ingestion flow.
It remains useful as a reference implementation for generic parsing patterns.
Keep it isolated from main supplier database contracts.
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
    parse_first_price,
    polite_sleep,
    save_assets_json,
    soup_from_html,
    text_or_none,
    uniq_keep_order,
)


FORMAT_KEYS = [
    "Wavefront",
    "Revit",
    "Sketchup",
    "3D Studio MAX",
    "Cinema 4D",
    "AutoCAD",
    "DXF",
    "ArchiCAD",
    "IFC",
    "Collada",
    "Step",
    "Materials & Textures",
    "Other",
]


def parse_archiproducts_product(url: str) -> ParsedAsset:
    html = fetch_html(url)
    soup = soup_from_html(html)
    jsonlds = extract_json_ld(soup)

    product_ld = (
        first_jsonld_of_type(jsonlds, "Product")
        or first_jsonld_of_type(jsonlds, "IndividualProduct")
        or {}
    )

    title = None
    h1 = soup.select_one("h1")
    if h1:
        title = text_or_none(h1)
    if not title:
        title = product_ld.get("name")

    brand = None
    brand_block = soup.find(string=re.compile(r"Brand", re.I))
    if brand_block and brand_block.parent:
        brand = brand_block.parent.get_text(" ", strip=True)
    if not brand:
        brand_val = product_ld.get("brand")
        if isinstance(brand_val, dict):
            brand = brand_val.get("name")
        elif isinstance(brand_val, str):
            brand = brand_val

    page_text = soup.get_text("\n", strip=True)

    collection = None
    designer = None
    category_raw = None

    m = re.search(r"Collection\s+([^\n]+)", page_text, re.I)
    if m:
        collection = m.group(1).strip()

    m = re.search(r"Designer\s+([^\n]+)", page_text, re.I)
    if m:
        designer = m.group(1).strip()

    m = re.search(r"Type\s+([^\n]+)", page_text, re.I)
    if m:
        category_raw = m.group(1).strip()

    description = None
    overview_header = soup.find(string=re.compile(r"Overview", re.I))
    if overview_header:
        parent = overview_header.find_parent()
        if parent:
            description = parent.get_text(" ", strip=True)
    if not description:
        description = product_ld.get("description")

    width_m, depth_m, height_m = parse_dimensions_from_text(page_text)

    price_value, price_currency = parse_first_price(page_text)
    if price_value is not None:
        price_type = "explicit"
    elif "Request prices/quote" in page_text or "Request Catalogues/Prices" in page_text:
        price_type = "request"
    else:
        price_type = "unknown"

    found_formats: list[str] = []
    lower_text = page_text.lower()
    for key in FORMAT_KEYS:
        if key.lower() in lower_text:
            found_formats.append(key)

    download_formats = uniq_keep_order(found_formats)

    preview_images: list[str] = []
    for img in soup.select("img"):
        src = img.get("src")
        if src and src.startswith("http"):
            preview_images.append(src)
    preview_images = uniq_keep_order(preview_images)[:10]

    return ParsedAsset(
        supplier="archiproducts",
        source_url=url,
        title=title,
        brand=brand,
        collection=collection,
        designer=designer,
        category_raw=category_raw,
        category_norm=normalize_category(category_raw or title),
        description=description,
        width_m=width_m,
        depth_m=depth_m,
        height_m=height_m,
        price_value=price_value,
        price_currency=price_currency,
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
            assets.append(parse_archiproducts_product(url))
        finally:
            polite_sleep(args.sleep)

    save_assets_json(args.out, assets)


if __name__ == "__main__":
    main()
