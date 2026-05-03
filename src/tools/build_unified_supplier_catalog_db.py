#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _load_items(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        return [x for x in data["items"] if isinstance(x, dict)]
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    raise ValueError(f"Unsupported catalog shape in {path}")


def _text(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def _json_dumps(value: Any) -> str | None:
    if value in (None, "", [], {}):
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _create_schema(con: sqlite3.Connection) -> None:
    con.executescript(
        """
        DROP TABLE IF EXISTS supplier_item;

        CREATE TABLE supplier_item (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            unique_key TEXT,
            source_site TEXT,
            source_url TEXT,
            parsed_at TEXT,
            external_id TEXT,
            title TEXT,
            brand TEXT,
            collection TEXT,
            category_raw TEXT,
            category_norm TEXT,
            room TEXT,
            style TEXT,
            color TEXT,
            availability TEXT,
            model_format TEXT,
            model_link_type TEXT,
            model_download_url TEXT,
            model_download_landing_url TEXT,
            model_page_url TEXT,
            product_url TEXT,
            asset_status TEXT,
            asset_format TEXT,
            asset_local_path TEXT,
            preview_local_path TEXT,
            price_currency TEXT,
            price_value REAL,
            old_price_value REAL,
            dimensions_json TEXT,
            materials_json TEXT,
            tags_json TEXT,
            related_json TEXT,
            extra_json TEXT,
            raw_json TEXT
        );

        CREATE INDEX idx_supplier_item_source_site ON supplier_item(source_site);
        CREATE INDEX idx_supplier_item_category_norm ON supplier_item(category_norm);
        CREATE INDEX idx_supplier_item_color ON supplier_item(color);
        CREATE INDEX idx_supplier_item_model_format ON supplier_item(model_format);
        CREATE INDEX idx_supplier_item_asset_status ON supplier_item(asset_status);
        CREATE INDEX idx_supplier_item_external_id ON supplier_item(external_id);
        """
    )


def _insert_items(con: sqlite3.Connection, items: list[dict[str, Any]]) -> None:
    rows = []
    for item in items:
        rows.append(
            (
                _text(item.get("unique_key")),
                _text(item.get("source_site")),
                _text(item.get("source_url")),
                _text(item.get("parsed_at")),
                _text(item.get("external_id")),
                _text(item.get("title")),
                _text(item.get("brand")),
                _text(item.get("collection")),
                _text(item.get("category_raw")),
                _text(item.get("category_norm")),
                _text(item.get("room")),
                _text(item.get("style")),
                _text(item.get("color")),
                _text(item.get("availability")),
                _text(item.get("model_format")),
                _text(item.get("model_link_type")),
                _text(item.get("model_download_url")),
                _text(item.get("model_download_landing_url")),
                _text(item.get("model_page_url")),
                _text(item.get("product_url")),
                _text(item.get("asset_status")),
                _text(item.get("asset_format")),
                _text(item.get("asset_local_path")),
                _text(item.get("preview_local_path")),
                _text(item.get("price_currency")),
                item.get("price_value"),
                item.get("old_price_value"),
                _json_dumps(item.get("dimensions_cm")),
                _json_dumps(item.get("materials")),
                _json_dumps(item.get("tags")),
                _json_dumps(item.get("related")),
                _json_dumps(item.get("extra")),
                _json_dumps(item),
            )
        )
    con.executemany(
        """
        INSERT INTO supplier_item (
            unique_key, source_site, source_url, parsed_at, external_id, title, brand,
            collection, category_raw, category_norm, room, style, color, availability,
            model_format, model_link_type, model_download_url, model_download_landing_url,
            model_page_url, product_url, asset_status, asset_format, asset_local_path,
            preview_local_path, price_currency, price_value, old_price_value,
            dimensions_json, materials_json, tags_json, related_json, extra_json, raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    con.commit()


def _top_counter(counter: Counter[str], limit: int = 100) -> list[dict[str, Any]]:
    return [{"name": name, "count": count} for name, count in counter.most_common(limit)]


def _build_stats(items: list[dict[str, Any]]) -> dict[str, Any]:
    source_site = Counter()
    category_norm = Counter()
    category_raw = Counter()
    color = Counter()
    model_format = Counter()
    asset_status = Counter()
    site_category = defaultdict(Counter)

    for item in items:
        site = _text(item.get("source_site")) or "UNKNOWN"
        cat_norm = _text(item.get("category_norm")) or "UNKNOWN"
        cat_raw = _text(item.get("category_raw")) or "UNKNOWN"
        clr = _text(item.get("color")) or "UNKNOWN"
        fmt = _text(item.get("model_format")) or "UNKNOWN"
        asset = _text(item.get("asset_status")) or "UNKNOWN"

        source_site[site] += 1
        category_norm[cat_norm] += 1
        category_raw[cat_raw] += 1
        color[clr] += 1
        model_format[fmt] += 1
        asset_status[asset] += 1
        site_category[site][cat_norm] += 1

    rows_with_model_hint = 0
    rows_with_download_url = 0
    rows_with_local_asset = 0
    for item in items:
        model_format_value = (_text(item.get("model_format")) or "").upper()
        if model_format_value and model_format_value != "UNKNOWN":
            rows_with_model_hint += 1
        if _text(item.get("model_download_url")):
            rows_with_download_url += 1
        if _text(item.get("asset_local_path")):
            rows_with_local_asset += 1

    return {
        "item_count": len(items),
        "source_site_counts": _top_counter(source_site, limit=100),
        "category_norm_counts": _top_counter(category_norm, limit=200),
        "category_raw_counts": _top_counter(category_raw, limit=200),
        "color_counts": _top_counter(color, limit=100),
        "model_format_counts": _top_counter(model_format, limit=50),
        "asset_status_counts": _top_counter(asset_status, limit=50),
        "rows_with_model_hint": rows_with_model_hint,
        "rows_with_download_url": rows_with_download_url,
        "rows_with_local_asset": rows_with_local_asset,
        "site_category_top": {
            site: _top_counter(counter, limit=30) for site, counter in sorted(site_category.items())
        },
    }


def _write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build unified supplier SQLite DB and category statistics from a supplier catalog JSON")
    ap.add_argument("--catalog-json", required=True)
    ap.add_argument("--out-db", required=True)
    ap.add_argument("--out-stats-json", required=True)
    ap.add_argument("--out-category-csv", default=None)
    ap.add_argument("--out-site-category-csv", default=None)
    args = ap.parse_args()

    catalog_json = Path(args.catalog_json).expanduser().resolve()
    out_db = Path(args.out_db).expanduser().resolve()
    out_stats_json = Path(args.out_stats_json).expanduser().resolve()
    out_category_csv = Path(args.out_category_csv).expanduser().resolve() if args.out_category_csv else None
    out_site_category_csv = Path(args.out_site_category_csv).expanduser().resolve() if args.out_site_category_csv else None

    items = _load_items(catalog_json)

    out_db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(out_db))
    try:
        _create_schema(con)
        _insert_items(con, items)
    finally:
        con.close()

    stats = _build_stats(items)
    out_stats_json.parent.mkdir(parents=True, exist_ok=True)
    out_stats_json.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    if out_category_csv:
        rows = [[row["name"], row["count"]] for row in stats["category_norm_counts"]]
        _write_csv(out_category_csv, ["category_norm", "count"], rows)

    if out_site_category_csv:
        rows = []
        for site, counters in stats["site_category_top"].items():
            for row in counters:
                rows.append([site, row["name"], row["count"]])
        _write_csv(out_site_category_csv, ["source_site", "category_norm", "count"], rows)

    print(f"catalog_json = {catalog_json}")
    print(f"out_db = {out_db}")
    print(f"out_stats_json = {out_stats_json}")
    if out_category_csv:
        print(f"out_category_csv = {out_category_csv}")
    if out_site_category_csv:
        print(f"out_site_category_csv = {out_site_category_csv}")
    print(f"item_count = {stats['item_count']}")
    print(f"rows_with_model_hint = {stats['rows_with_model_hint']}")
    print(f"rows_with_download_url = {stats['rows_with_download_url']}")
    print(f"rows_with_local_asset = {stats['rows_with_local_asset']}")
    print("top_source_sites = " + ", ".join(f"{x['name']}:{x['count']}" for x in stats["source_site_counts"][:10]))
    print("top_category_norm = " + ", ".join(f"{x['name']}:{x['count']}" for x in stats["category_norm_counts"][:15]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
