# -*- coding: utf-8 -*-
"""
This module defines the core supplier product database schema and writes.
It owns table creation, product upserts, and fetch-log persistence.
Most supplier collection scripts depend on this schema contract.
The code should remain simple, explicit, and migration-friendly.
Keep SQL shape changes synchronized with downstream exporters.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

from src.suppliers.models import ProductRecord


PRODUCT_COLUMNS = [
    "unique_key",
    "source_site",
    "source_url",
    "parsed_at",
    "external_id",
    "category_raw",
    "category_norm",
    "title",
    "brand",
    "collection",
    "product_url",
    "model_link_type",
    "model_page_url",
    "model_download_url",
    "model_download_landing_url",
    "model_vendor_url",
    "model_extraction_method",
    "model_download_filename",
    "model_format",
    "price_value",
    "price_currency",
    "old_price_value",
    "style",
    "color",
    "description",
    "width_cm",
    "depth_cm",
    "height_cm",
    "weight_kg",
    "room",
    "materials",
    "availability",
    "country_brand",
    "production_country",
    "tags_json",
    "images_json",
    "related_json",
    "extra_json",
    "raw_html",
]


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as con:
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")

        con.execute(
            """
            CREATE TABLE IF NOT EXISTS supplier_product (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                unique_key TEXT NOT NULL UNIQUE,
                source_site TEXT NOT NULL,
                source_url TEXT NOT NULL,
                parsed_at TEXT NOT NULL,

                external_id TEXT,

                category_raw TEXT,
                category_norm TEXT,

                title TEXT,
                brand TEXT,
                collection TEXT,

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

                room TEXT,
                materials TEXT,
                availability TEXT,
                country_brand TEXT,
                production_country TEXT,

                tags_json TEXT NOT NULL DEFAULT '[]',
                images_json TEXT NOT NULL DEFAULT '[]',
                related_json TEXT NOT NULL DEFAULT '[]',
                extra_json TEXT NOT NULL DEFAULT '{}',

                raw_html TEXT NOT NULL DEFAULT ''
            );
            """
        )

        con.execute(
            """
            CREATE TABLE IF NOT EXISTS supplier_fetch_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_site TEXT NOT NULL,
                source_url TEXT NOT NULL,
                fetched_at TEXT NOT NULL,
                ok INTEGER NOT NULL,
                error TEXT
            );
            """
        )

        con.execute("CREATE INDEX IF NOT EXISTS idx_supplier_product_site ON supplier_product(source_site);")
        con.execute("CREATE INDEX IF NOT EXISTS idx_supplier_product_title ON supplier_product(title);")
        con.execute("CREATE INDEX IF NOT EXISTS idx_supplier_product_brand ON supplier_product(brand);")
        con.execute("CREATE INDEX IF NOT EXISTS idx_supplier_product_collection ON supplier_product(collection);")
        con.execute("CREATE INDEX IF NOT EXISTS idx_supplier_product_category_norm ON supplier_product(category_norm);")
        con.execute("CREATE INDEX IF NOT EXISTS idx_supplier_product_model_download_url ON supplier_product(model_download_url);")
        con.execute("CREATE INDEX IF NOT EXISTS idx_supplier_fetch_log_site_url ON supplier_fetch_log(source_site, source_url);")


def upsert_product(db_path: Path, product: ProductRecord) -> None:
    placeholders = ", ".join(["?"] * len(PRODUCT_COLUMNS))
    columns_sql = ", ".join(PRODUCT_COLUMNS)
    update_sql = ",\n                ".join(
        f"{column}=excluded.{column}" for column in PRODUCT_COLUMNS if column != "unique_key"
    )
    values = tuple(getattr(product, column) for column in PRODUCT_COLUMNS)

    with sqlite3.connect(db_path) as con:
        con.execute(
            f"""
            INSERT INTO supplier_product ({columns_sql})
            VALUES ({placeholders})
            ON CONFLICT(unique_key) DO UPDATE SET
                {update_sql}
            ;
            """,
            values,
        )


def upsert_products(db_path: Path, products: Iterable[ProductRecord]) -> None:
    for product in products:
        upsert_product(db_path, product)


def insert_fetch_log(
    db_path: Path,
    source_site: str,
    source_url: str,
    fetched_at: str,
    ok: bool,
    error: str | None,
) -> None:
    with sqlite3.connect(db_path) as con:
        con.execute(
            """
            INSERT INTO supplier_fetch_log (
                source_site,
                source_url,
                fetched_at,
                ok,
                error
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                source_site,
                source_url,
                fetched_at,
                1 if ok else 0,
                error,
            ),
        )
