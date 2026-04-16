#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script exports supplier product rows from SQLite into a compact JSON catalog.
It joins optional asset state, applies simple filters, and keeps parsed payloads.
The result is used by matching, analysis, and offline inspection tools.
It should stay schema-stable across catalog rebuilds.
Keep export structure predictable and easy to diff.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any
from src.suppliers.utils import json_loads_or, sqlite_table_exists


def _load_rows(db_path: Path, sites: set[str] | None) -> list[dict[str, Any]]:
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        has_asset = sqlite_table_exists(con, "supplier_asset")

        sql = """
        SELECT
            sp.unique_key,
            sp.source_site,
            sp.source_url,
            sp.parsed_at,
            sp.external_id,
            sp.category_raw,
            sp.category_norm,
            sp.title,
            sp.brand,
            sp.collection,
            sp.product_url,
            sp.model_link_type,
            sp.model_page_url,
            sp.model_download_url,
            sp.model_download_landing_url,
            sp.model_vendor_url,
            sp.model_extraction_method,
            sp.model_download_filename,
            sp.model_format,
            sp.price_value,
            sp.price_currency,
            sp.old_price_value,
            sp.style,
            sp.color,
            sp.description,
            sp.width_cm,
            sp.depth_cm,
            sp.height_cm,
            sp.weight_kg,
            sp.room,
            sp.materials,
            sp.availability,
            sp.country_brand,
            sp.production_country,
            sp.tags_json,
            sp.images_json,
            sp.related_json,
            sp.extra_json,
            {asset_status},
            {asset_format},
            {asset_local_path},
            {preview_local_path},
            {asset_source_url}
        FROM supplier_product sp
        {asset_join}
        """

        if has_asset:
            sql = sql.format(
                asset_status="sa.asset_status AS asset_status",
                asset_format="sa.asset_format AS asset_format",
                asset_local_path="sa.asset_local_path AS asset_local_path",
                preview_local_path="sa.preview_local_path AS preview_local_path",
                asset_source_url="sa.asset_source_url AS asset_source_url",
                asset_join="LEFT JOIN supplier_asset sa ON sa.unique_key = sp.unique_key",
            )
        else:
            sql = sql.format(
                asset_status="NULL AS asset_status",
                asset_format="NULL AS asset_format",
                asset_local_path="NULL AS asset_local_path",
                preview_local_path="NULL AS preview_local_path",
                asset_source_url="NULL AS asset_source_url",
                asset_join="",
            )

        params: list[Any] = []
        if sites:
            placeholders = ", ".join("?" for _ in sites)
            sql += f" WHERE sp.source_site IN ({placeholders})"
            params.extend(sorted(sites))

        sql += " ORDER BY sp.source_site, sp.title, sp.unique_key"
        rows = con.execute(sql, params).fetchall()
        return [dict(row) for row in rows]


def _has_text(value: Any) -> bool:
    return bool(str(value or "").strip())


def _has_full_dimensions(row: dict[str, Any]) -> bool:
    return (
        row.get("width_cm") is not None
        and row.get("depth_cm") is not None
        and row.get("height_cm") is not None
    )


def _has_category(row: dict[str, Any]) -> bool:
    return _has_text(row.get("category_raw")) or _has_text(row.get("category_norm"))


def _rich_card_flags(row: dict[str, Any]) -> dict[str, bool]:
    return {
        "has_title": _has_text(row.get("title")),
        "has_price": row.get("price_value") is not None,
        "has_full_dimensions": _has_full_dimensions(row),
        "has_description": _has_text(row.get("description")),
        "has_category": _has_category(row),
        "has_brand": _has_text(row.get("brand")),
        "has_model_link": _has_text(row.get("model_download_url")) or _has_text(row.get("model_page_url")),
    }


def _is_rich_card(row: dict[str, Any]) -> bool:
    flags = _rich_card_flags(row)
    return all(
        flags[key]
        for key in (
            "has_title",
            "has_price",
            "has_full_dimensions",
            "has_description",
            "has_category",
            "has_brand",
        )
    )


def build_catalog_export(
    db_paths: list[Path],
    sites: set[str] | None = None,
    only_with_model_url: bool = False,
    only_with_asset: bool = False,
    only_rich: bool = False,
) -> dict[str, Any]:
    dedup: dict[str, dict[str, Any]] = {}

    for db_path in db_paths:
        for row in _load_rows(db_path, sites):
            if only_with_model_url and not row.get("model_download_url"):
                continue
            if only_with_asset and not row.get("asset_local_path"):
                continue
            if only_rich and not _is_rich_card(row):
                continue

            completeness = _rich_card_flags(row)
            item = {
                "unique_key": row.get("unique_key"),
                "source_site": row.get("source_site"),
                "source_db": str(db_path.resolve()),
                "source_url": row.get("source_url"),
                "parsed_at": row.get("parsed_at"),
                "external_id": row.get("external_id"),
                "title": row.get("title"),
                "brand": row.get("brand"),
                "collection": row.get("collection"),
                "category_raw": row.get("category_raw"),
                "category_norm": row.get("category_norm"),
                "product_url": row.get("product_url"),
                "model_link_type": row.get("model_link_type"),
                "model_page_url": row.get("model_page_url"),
                "model_download_url": row.get("model_download_url"),
                "model_download_landing_url": row.get("model_download_landing_url"),
                "model_vendor_url": row.get("model_vendor_url"),
                "model_extraction_method": row.get("model_extraction_method"),
                "model_download_filename": row.get("model_download_filename"),
                "model_format": row.get("model_format"),
                "asset_status": row.get("asset_status"),
                "asset_format": row.get("asset_format"),
                "asset_local_path": row.get("asset_local_path"),
                "preview_local_path": row.get("preview_local_path"),
                "asset_source_url": row.get("asset_source_url"),
                "price_value": row.get("price_value"),
                "price_currency": row.get("price_currency"),
                "old_price_value": row.get("old_price_value"),
                "style": row.get("style"),
                "color": row.get("color"),
                "description": row.get("description"),
                "dimensions_cm": {
                    "width": row.get("width_cm"),
                    "depth": row.get("depth_cm"),
                    "height": row.get("height_cm"),
                    "weight_kg": row.get("weight_kg"),
                },
                "room": row.get("room"),
                "materials": row.get("materials"),
                "availability": row.get("availability"),
                "country_brand": row.get("country_brand"),
                "production_country": row.get("production_country"),
                "tags": json_loads_or(row.get("tags_json"), []),
                "images": json_loads_or(row.get("images_json"), []),
                "related": json_loads_or(row.get("related_json"), []),
                "extra": json_loads_or(row.get("extra_json"), {}),
                "completeness": {
                    **completeness,
                    "rich_card": _is_rich_card(row),
                },
            }
            dedup[item["unique_key"]] = item

    items = sorted(
        dedup.values(),
        key=lambda x: (
            str(x.get("source_site") or ""),
            str(x.get("title") or ""),
            str(x.get("unique_key") or ""),
        ),
    )
    return {
        "schema": "supplier_catalog_export/v1",
        "meta": {
            "db_paths": [str(p.resolve()) for p in db_paths],
            "site_filter": sorted(sites) if sites else None,
            "only_with_model_url": only_with_model_url,
            "only_with_asset": only_with_asset,
            "only_rich": only_rich,
            "item_count": len(items),
        },
        "items": items,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Export supplier_product catalog into JSON.")
    ap.add_argument("--db", action="append", required=True, help="SQLite DB path; may be repeated")
    ap.add_argument("--site", action="append", default=[], help="Optional site filter; may be repeated")
    ap.add_argument("--only-with-model-url", action="store_true")
    ap.add_argument("--only-with-asset", action="store_true")
    ap.add_argument("--only-rich", action="store_true", help="Keep only cards with title, price, full dimensions, description, category and brand")
    ap.add_argument("--out", required=True, help="Output JSON path")
    args = ap.parse_args()

    db_paths = [Path(x).expanduser().resolve() for x in args.db]
    sites = {str(x).strip() for x in args.site if str(x).strip()} or None
    out_path = Path(args.out).expanduser().resolve()

    data = build_catalog_export(
        db_paths=db_paths,
        sites=sites,
        only_with_model_url=bool(args.only_with_model_url),
        only_with_asset=bool(args.only_with_asset),
        only_rich=bool(args.only_rich),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"items = {data['meta']['item_count']}")
    print(f"saved = {out_path}")


if __name__ == "__main__":
    main()
