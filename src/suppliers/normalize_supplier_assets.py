# -*- coding: utf-8 -*-
"""
This script normalizes legacy supplier asset JSON exports into a stable shape.
It assigns deterministic asset identifiers and preserves important metadata.
The output is intended for downstream matching and inspection workflows.
This is a lightweight transformation tool rather than a crawler.
Keep normalization rules deterministic and schema-safe.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def save_json(path: str | Path, data: Any) -> None:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def make_asset_id(item: dict[str, Any], index: int) -> str:
    supplier = str(item.get("supplier") or "unknown").strip().lower()
    title = str(item.get("title") or f"item_{index}").strip().lower()
    slug = "".join(ch if ch.isalnum() else "_" for ch in title).strip("_")
    slug = "_".join(filter(None, slug.split("_")))
    return f"{supplier}_{index:06d}_{slug[:80]}"


def normalize_one(item: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "asset_id": make_asset_id(item, index),
        "supplier": item.get("supplier"),
        "source_url": item.get("source_url"),
        "title": item.get("title"),
        "brand": item.get("brand"),
        "collection": item.get("collection"),
        "designer": item.get("designer"),
        "category_raw": item.get("category_raw"),
        "category_norm": item.get("category_norm"),
        "description": item.get("description"),
        "style": item.get("style") or [],
        "materials": item.get("materials") or [],
        "colors": item.get("colors") or [],
        "room_tags": item.get("room_tags") or [],
        "dimensions_m": {
            "width": item.get("width_m"),
            "depth": item.get("depth_m"),
            "height": item.get("height_m"),
        },
        "price_value": item.get("price_value"),
        "price_currency": item.get("price_currency"),
        "price_type": item.get("price_type"),
        "download_formats": item.get("download_formats") or [],
        "blender_ready_score": item.get("blender_ready_score"),
        "preview_images": item.get("preview_images") or [],
        "download_url": item.get("download_url"),
        "raw_meta": item.get("raw_meta") or {},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    merged: list[dict[str, Any]] = []
    for path in args.inputs:
        data = load_json(path)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    merged.append(item)

    normalized = [normalize_one(item, i) for i, item in enumerate(merged)]
    save_json(args.out, normalized)
    print(f"saved {len(normalized)} assets -> {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
