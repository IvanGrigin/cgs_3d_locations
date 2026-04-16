# -*- coding: utf-8 -*-
"""
This script parses a single 3DSky product page into a legacy ParsedAsset shape.
It belongs to the older standalone parsing path kept for reference and reuse.
The script is useful for quick experiments outside the main adapter pipeline.
Its output schema differs from the normalized ProductRecord flow.
Keep it stable but do not extend it with new ingestion contracts.
"""
from __future__ import annotations

import argparse
import re

from src.suppliers.common import (
    ParsedAsset,
    compute_blender_ready_score,
    fetch_html,
    normalize_category,
    parse_dimensions_from_text,
    parse_first_price,
    polite_sleep,
    save_assets_json,
    soup_from_html,
    uniq_keep_order,
)


def extract_field(text: str, field_name: str) -> str | None:
    pattern = re.compile(rf"{re.escape(field_name)}\s*:\s*(.+)", re.I)
    match = pattern.search(text)
    if not match:
        return None
    line = match.group(1).strip()
    line = line.split("\n")[0].strip()
    return line or None


def parse_3dsky_product(url: str) -> ParsedAsset:
    html = fetch_html(url)
    soup = soup_from_html(html)
    page_text = soup.get_text("\n", strip=True)

    h1 = soup.select_one("h1")
    title = h1.get_text(" ", strip=True) if h1 else None

    materials_line = extract_field(page_text, "Materials")
    style_line = extract_field(page_text, "Style")
    formfactor_line = extract_field(page_text, "Formfactor")

    width_m, depth_m, height_m = parse_dimensions_from_text(page_text)

    if width_m is None:
        match = re.search(
            r"Width:\s*(\d+(?:[.,]\d+)?)\s*cm.*?"
            r"Depth:\s*(\d+(?:[.,]\d+)?)\s*cm.*?"
            r"Height:\s*(\d+(?:[.,]\d+)?)\s*cm",
            page_text,
            re.I | re.DOTALL,
        )
        if match:
            width_m = float(match.group(1).replace(",", ".")) / 100.0
            depth_m = float(match.group(2).replace(",", ".")) / 100.0
            height_m = float(match.group(3).replace(",", ".")) / 100.0

    price_value, price_currency = parse_first_price(page_text)
    price_type = "explicit" if price_value is not None else "unknown"

    download_formats = ["max"]
    if "official 3d model" in page_text.lower() or " om " in f" {page_text.lower()} ":
        download_formats.append("official")
    download_formats = uniq_keep_order(download_formats)

    preview_images: list[str] = []
    for img in soup.select("img"):
        src = img.get("src")
        if src and src.startswith("http"):
            preview_images.append(src)
    preview_images = uniq_keep_order(preview_images)[:10]

    style = [x.strip() for x in (style_line or "").split(",") if x.strip()]
    materials = [x.strip() for x in (materials_line or "").split(",") if x.strip()]

    return ParsedAsset(
        supplier="3dsky",
        source_url=url,
        title=title,
        category_raw=formfactor_line or title,
        category_norm=normalize_category(formfactor_line or title),
        description=page_text[:3000],
        style=style or None,
        materials=materials or None,
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
            "style_raw": style_line,
            "materials_raw": materials_line,
            "formfactor_raw": formfactor_line,
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
            assets.append(parse_3dsky_product(url))
        finally:
            polite_sleep(args.sleep)

    save_assets_json(args.out, assets)


if __name__ == "__main__":
    main()
