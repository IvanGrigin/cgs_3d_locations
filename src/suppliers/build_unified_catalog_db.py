#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script merges supplier product databases into one unified catalog database.
It deduplicates overlapping cards, carries asset state forward, and infers groups.
The output is the main normalized catalog used by matching and analytics.
It is intentionally read-heavy and conservative about merging quality.
Keep ranking and canonical-key logic explicit.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

try:
    from src.layout_targets import _semantic_group
except ImportError:
    from layout_targets import _semantic_group

from src.suppliers.utils import sqlite_table_exists


READY_ASSET_FORMATS = {"fbx", "glb", "obj", "blend"}
REAL_ASSET_STATUSES = {
    "archive_extracted_preferred",
    "downloaded_preferred",
    "converted_with_blender",
}
LOW_QUALITY_ASSET_STATUSES = {
    "proxy_generated_with_blender",
    "needs_blender_rebuild",
}


def _normalize_url(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    return text.rstrip("/")


def _canonical_key(row: dict[str, Any]) -> str:
    source_site = str(row.get("source_site") or "").strip() or "unknown"
    product_url = _normalize_url(row.get("product_url"))
    if product_url:
        return f"url::{product_url}"
    model_url = _normalize_url(row.get("model_download_url"))
    if model_url:
        return f"model::{model_url}"
    external_id = str(row.get("external_id") or "").strip()
    if external_id:
        return f"external::{source_site}::{external_id}"
    title = str(row.get("title") or "").strip()
    if title:
        return f"title::{source_site}::{title.lower()}"
    return f"unique::{row.get('unique_key')}"


def _infer_semantic_group(row: dict[str, Any]) -> str:
    category_norm = str(row.get("category_norm") or "").strip().lower()
    category_raw = str(row.get("category_raw") or "")
    title = str(row.get("title") or "")

    direct_map = {
        "bed": "bed",
        "double_bed": "bed",
        "single_bed": "bed",
        "nightstand": "nightstand",
        "bedside_table": "nightstand",
        "bedside_cabinet": "nightstand",
        "wardrobe": "wardrobe",
        "closet": "wardrobe",
        "cabinet": "dresser",
        "sideboard": "dresser",
        "dresser": "dresser",
        "bookcase": "shelf",
        "bookshelf": "shelf",
        "shelf": "shelf",
        "desk": "desk",
        "table": "coffee_table",
        "dining_table": "coffee_table",
        "coffee_table": "coffee_table",
        "side_table": "side_table",
        "end_table": "side_table",
        "tv_stand": "tv_stand",
        "armchair": "armchair",
        "chair": "chair",
        "office_chair": "chair",
        "sofa": "sofa",
        "plant": "plant",
        "mirror": "mirror",
        "lighting": "lamp_ceiling",
        "light": "lamp_ceiling",
        "chandelier": "lamp_ceiling",
        "wall_lamp": "lamp_wall",
        "floor_lamp": "lamp_floor",
    }
    if category_norm in direct_map:
        return direct_map[category_norm]

    text = f"{category_raw} {title}".lower()
    if any(x in text for x in ("прикроват", "nightstand", "bedside")):
        return "nightstand"
    if any(x in text for x in ("wardrobe", "closet", "шкаф")):
        return "wardrobe"
    if any(x in text for x in ("комод", "dresser", "sideboard", "cabinet", "сервант")):
        return "dresser"
    if any(x in text for x in ("bookcase", "bookshelf", "стеллаж", "shelf")):
        return "shelf"

    return _semantic_group(title, category_raw, {})


def _file_exists(path_value: Any) -> bool:
    text = str(path_value or "").strip()
    if not text:
        return False
    try:
        return Path(text).expanduser().exists()
    except Exception:
        return False


def _asset_priority(row: dict[str, Any]) -> tuple[int, int, int, int]:
    status = str(row.get("asset_status") or "").strip().lower()
    fmt = str(row.get("asset_format") or "").strip().lower()
    path_exists = _file_exists(row.get("asset_local_path"))

    status_rank = 0
    if status in REAL_ASSET_STATUSES:
        status_rank = 3
    elif status and status not in LOW_QUALITY_ASSET_STATUSES:
        status_rank = 2
    elif status in LOW_QUALITY_ASSET_STATUSES:
        status_rank = 1

    format_rank = 1 if fmt in READY_ASSET_FORMATS else 0
    path_rank = 1 if path_exists else 0
    model_url_rank = 1 if row.get("model_download_url") else 0
    return status_rank, format_rank, path_rank, model_url_rank


def _row_quality(row: dict[str, Any]) -> tuple[int, int, int, int]:
    dims_count = sum(row.get(k) is not None for k in ("width_cm", "depth_cm", "height_cm"))
    text_count = sum(bool(row.get(k)) for k in ("style", "materials", "color", "description"))
    url_count = sum(bool(row.get(k)) for k in ("product_url", "model_download_url", "model_page_url"))
    price_count = 1 if row.get("price_value") is not None else 0
    return dims_count, text_count, url_count, price_count


def _table_rows(db_path: Path, sites: set[str] | None) -> list[dict[str, Any]]:
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
            sp.volume_m3,
            sp.package_width_cm,
            sp.package_depth_cm,
            sp.package_height_cm,
            sp.packed_weight_kg,
            sp.scheme_url,
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
            {asset_source_url},
            {asset_local_path},
            {preview_local_path}
        FROM supplier_product sp
        {asset_join}
        """

        if has_asset:
            sql = sql.format(
                asset_status="sa.asset_status AS asset_status",
                asset_format="sa.asset_format AS asset_format",
                asset_source_url="sa.asset_source_url AS asset_source_url",
                asset_local_path="sa.asset_local_path AS asset_local_path",
                preview_local_path="sa.preview_local_path AS preview_local_path",
                asset_join="LEFT JOIN supplier_asset sa ON sa.unique_key = sp.unique_key",
            )
        else:
            sql = sql.format(
                asset_status="NULL AS asset_status",
                asset_format="NULL AS asset_format",
                asset_source_url="NULL AS asset_source_url",
                asset_local_path="NULL AS asset_local_path",
                preview_local_path="NULL AS preview_local_path",
                asset_join="",
            )

        items: list[dict[str, Any]] = []
        for row in con.execute(sql):
            item = dict(row)
            if sites and str(item.get("source_site") or "") not in sites:
                continue
            if not item.get("title"):
                continue
            item["source_db_path"] = str(db_path.resolve())
            item["semantic_group"] = _infer_semantic_group(item)
            items.append(item)
        return items


def _merge_value(primary: Any, secondary: Any) -> Any:
    if primary not in (None, "", [], {}):
        return primary
    return secondary


def _merged_record(rows: list[dict[str, Any]]) -> dict[str, Any]:
    best_meta = max(rows, key=lambda x: (_row_quality(x), _asset_priority(x)))
    best_asset = max(rows, key=lambda x: (_asset_priority(x), _row_quality(x)))

    merged = dict(best_meta)
    for key, value in best_asset.items():
        if key.startswith("asset_") or key in {"preview_local_path", "source_db_path"}:
            merged[key] = value

    for row in rows:
        for key, value in row.items():
            merged[key] = _merge_value(merged.get(key), value)

    merged["merged_unique_keys_json"] = json.dumps(
        sorted({str(r.get("unique_key")) for r in rows if r.get("unique_key")}),
        ensure_ascii=False,
    )
    merged["merged_source_dbs_json"] = json.dumps(
        sorted({str(r.get("source_db_path")) for r in rows if r.get("source_db_path")}),
        ensure_ascii=False,
    )
    merged["mesh_local_path"] = best_asset.get("asset_local_path")
    merged["mesh_format"] = best_asset.get("asset_format")
    merged["mesh_status"] = best_asset.get("asset_status")
    merged["mesh_source_url"] = best_asset.get("asset_source_url") or best_asset.get("model_download_url")
    merged["mesh_ready"] = int(
        bool(best_asset.get("asset_local_path"))
        and str(best_asset.get("asset_format") or "").lower() in READY_ASSET_FORMATS
        and str(best_asset.get("asset_status") or "").lower() in REAL_ASSET_STATUSES
        and _file_exists(best_asset.get("asset_local_path"))
    )
    merged["mesh_available"] = int(
        merged["mesh_ready"]
        or bool(merged.get("model_download_url"))
        or bool(merged.get("mesh_local_path"))
    )
    return merged


def init_out_db(out_db: Path) -> None:
    with sqlite3.connect(out_db) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS supplier_mesh_catalog (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                canonical_key TEXT NOT NULL UNIQUE,
                unique_key TEXT,
                source_site TEXT,
                source_db_path TEXT,
                source_url TEXT,
                parsed_at TEXT,
                external_id TEXT,
                title TEXT,
                brand TEXT,
                collection TEXT,
                category_raw TEXT,
                category_norm TEXT,
                semantic_group TEXT,
                product_url TEXT,
                model_link_type TEXT,
                model_page_url TEXT,
                model_download_url TEXT,
                model_download_landing_url TEXT,
                model_vendor_url TEXT,
                model_extraction_method TEXT,
                model_download_filename TEXT,
                model_format TEXT,
                price_value REAL,
                price_currency TEXT,
                old_price_value REAL,
                style TEXT,
                color TEXT,
                description TEXT,
                width_cm REAL,
                depth_cm REAL,
                height_cm REAL,
                weight_kg REAL,
                volume_m3 REAL,
                package_width_cm REAL,
                package_depth_cm REAL,
                package_height_cm REAL,
                packed_weight_kg REAL,
                scheme_url TEXT,
                room TEXT,
                materials TEXT,
                availability TEXT,
                country_brand TEXT,
                production_country TEXT,
                tags_json TEXT NOT NULL DEFAULT '[]',
                images_json TEXT NOT NULL DEFAULT '[]',
                related_json TEXT NOT NULL DEFAULT '[]',
                extra_json TEXT NOT NULL DEFAULT '{}',
                asset_status TEXT,
                asset_format TEXT,
                asset_source_url TEXT,
                asset_local_path TEXT,
                preview_local_path TEXT,
                mesh_local_path TEXT,
                mesh_format TEXT,
                mesh_status TEXT,
                mesh_source_url TEXT,
                mesh_ready INTEGER NOT NULL DEFAULT 0,
                mesh_available INTEGER NOT NULL DEFAULT 0,
                merged_unique_keys_json TEXT NOT NULL DEFAULT '[]',
                merged_source_dbs_json TEXT NOT NULL DEFAULT '[]'
            );
            """
        )
        existing = {str(row[1]) for row in con.execute("PRAGMA table_info(supplier_mesh_catalog)").fetchall()}
        for name, sql_type in {
            "volume_m3": "REAL",
            "package_width_cm": "REAL",
            "package_depth_cm": "REAL",
            "package_height_cm": "REAL",
            "packed_weight_kg": "REAL",
            "scheme_url": "TEXT",
        }.items():
            if name not in existing:
                con.execute(f"ALTER TABLE supplier_mesh_catalog ADD COLUMN {name} {sql_type};")
        con.execute("CREATE INDEX IF NOT EXISTS idx_supplier_mesh_catalog_site ON supplier_mesh_catalog(source_site);")
        con.execute("CREATE INDEX IF NOT EXISTS idx_supplier_mesh_catalog_group ON supplier_mesh_catalog(semantic_group);")
        con.execute("CREATE INDEX IF NOT EXISTS idx_supplier_mesh_catalog_mesh_ready ON supplier_mesh_catalog(mesh_ready);")
        con.execute("CREATE INDEX IF NOT EXISTS idx_supplier_mesh_catalog_product_url ON supplier_mesh_catalog(product_url);")


def save_records(out_db: Path, records: list[dict[str, Any]]) -> None:
    with sqlite3.connect(out_db) as con:
        con.execute("DELETE FROM supplier_mesh_catalog;")
        for row in records:
            canonical_key = _canonical_key(row)
            con.execute(
                """
                INSERT INTO supplier_mesh_catalog (
                    canonical_key, unique_key, source_site, source_db_path, source_url, parsed_at, external_id,
                    title, brand, collection, category_raw, category_norm, semantic_group, product_url,
                    model_link_type, model_page_url, model_download_url, model_download_landing_url, model_vendor_url,
                    model_extraction_method, model_download_filename, model_format, price_value, price_currency,
                    old_price_value, style, color, description, width_cm, depth_cm, height_cm, weight_kg,
                    volume_m3, package_width_cm, package_depth_cm, package_height_cm, packed_weight_kg, scheme_url, room,
                    materials, availability, country_brand, production_country, tags_json, images_json, related_json,
                    extra_json, asset_status, asset_format, asset_source_url, asset_local_path, preview_local_path,
                    mesh_local_path, mesh_format, mesh_status, mesh_source_url, mesh_ready, mesh_available,
                    merged_unique_keys_json, merged_source_dbs_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    canonical_key,
                    row.get("unique_key"),
                    row.get("source_site"),
                    row.get("source_db_path"),
                    row.get("source_url"),
                    row.get("parsed_at"),
                    row.get("external_id"),
                    row.get("title"),
                    row.get("brand"),
                    row.get("collection"),
                    row.get("category_raw"),
                    row.get("category_norm"),
                    row.get("semantic_group"),
                    row.get("product_url"),
                    row.get("model_link_type"),
                    row.get("model_page_url"),
                    row.get("model_download_url"),
                    row.get("model_download_landing_url"),
                    row.get("model_vendor_url"),
                    row.get("model_extraction_method"),
                    row.get("model_download_filename"),
                    row.get("model_format"),
                    row.get("price_value"),
                    row.get("price_currency"),
                    row.get("old_price_value"),
                    row.get("style"),
                    row.get("color"),
                    row.get("description"),
                    row.get("width_cm"),
                    row.get("depth_cm"),
                    row.get("height_cm"),
                    row.get("weight_kg"),
                    row.get("volume_m3"),
                    row.get("package_width_cm"),
                    row.get("package_depth_cm"),
                    row.get("package_height_cm"),
                    row.get("packed_weight_kg"),
                    row.get("scheme_url"),
                    row.get("room"),
                    row.get("materials"),
                    row.get("availability"),
                    row.get("country_brand"),
                    row.get("production_country"),
                    row.get("tags_json") or "[]",
                    row.get("images_json") or "[]",
                    row.get("related_json") or "[]",
                    row.get("extra_json") or "{}",
                    row.get("asset_status"),
                    row.get("asset_format"),
                    row.get("asset_source_url"),
                    row.get("asset_local_path"),
                    row.get("preview_local_path"),
                    row.get("mesh_local_path"),
                    row.get("mesh_format"),
                    row.get("mesh_status"),
                    row.get("mesh_source_url"),
                    int(row.get("mesh_ready") or 0),
                    int(row.get("mesh_available") or 0),
                    row.get("merged_unique_keys_json") or "[]",
                    row.get("merged_source_dbs_json") or "[]",
                ),
            )


def build_catalog(db_paths: list[Path], sites: set[str] | None) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for db_path in db_paths:
        for row in _table_rows(db_path, sites):
            grouped.setdefault(_canonical_key(row), []).append(row)

    merged = [_merged_record(rows) for rows in grouped.values()]
    merged.sort(
        key=lambda x: (
            str(x.get("source_site") or ""),
            str(x.get("semantic_group") or ""),
            str(x.get("title") or ""),
        )
    )
    return merged


def main() -> None:
    ap = argparse.ArgumentParser(description="Build unified supplier mesh catalog DB.")
    ap.add_argument("--db", action="append", help="Input SQLite DB; may be repeated")
    ap.add_argument("--db-dir", help="Optional directory with *.db to include")
    ap.add_argument("--site", action="append", default=[], help="Optional site filter; may be repeated")
    ap.add_argument("--out-db", required=True, help="Output SQLite DB")
    ap.add_argument("--out-json", help="Optional output JSON snapshot")
    args = ap.parse_args()

    db_paths: list[Path] = []
    if args.db_dir:
        db_paths.extend(sorted(Path(args.db_dir).expanduser().resolve().glob("*.db")))
    if args.db:
        db_paths.extend(Path(x).expanduser().resolve() for x in args.db)

    unique_db_paths: list[Path] = []
    seen = set()
    for path in db_paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique_db_paths.append(path)

    if not unique_db_paths:
        raise SystemExit("No input DBs provided")

    sites = {str(x).strip() for x in args.site if str(x).strip()} or None
    records = build_catalog(unique_db_paths, sites)

    out_db = Path(args.out_db).expanduser().resolve()
    out_db.parent.mkdir(parents=True, exist_ok=True)
    init_out_db(out_db)
    save_records(out_db, records)

    if args.out_json:
        out_json = Path(args.out_json).expanduser().resolve()
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(
            json.dumps(
                {
                    "schema": "supplier_mesh_catalog_snapshot/v1",
                    "meta": {
                        "db_paths": [str(p) for p in unique_db_paths],
                        "site_filter": sorted(sites) if sites else None,
                        "item_count": len(records),
                        "mesh_ready_count": sum(int(r.get("mesh_ready") or 0) for r in records),
                        "mesh_available_count": sum(int(r.get("mesh_available") or 0) for r in records),
                    },
                    "items": records,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    print(f"items = {len(records)}")
    print(f"mesh_ready = {sum(int(r.get('mesh_ready') or 0) for r in records)}")
    print(f"mesh_available = {sum(int(r.get('mesh_available') or 0) for r in records)}")
    print(f"saved_db = {out_db}")
    if args.out_json:
        print(f"saved_json = {Path(args.out_json).expanduser().resolve()}")


if __name__ == "__main__":
    main()
