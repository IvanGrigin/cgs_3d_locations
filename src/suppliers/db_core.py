# -*- coding: utf-8 -*-
"""
This module extends the base supplier database with download and asset tables.
It provides a single initialization path for acquisition-oriented workflows.
The schema here tracks downloaded files and resolved local mesh assets.
It is intentionally thin and delegates product writes to the base DB layer.
Keep asset-table changes backward compatible for existing databases.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from src.suppliers.db import init_db as init_supplier_db
from src.suppliers.db import upsert_product
from src.suppliers.site_models import SupplierAssetRecord


def init_db(db_path: Path) -> None:
    init_supplier_db(db_path)

    with sqlite3.connect(db_path) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS supplier_download (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                unique_key TEXT NOT NULL,
                downloaded_at TEXT NOT NULL,
                final_url TEXT,
                local_path TEXT,
                filename TEXT,
                content_type TEXT,
                ok INTEGER NOT NULL,
                size_bytes INTEGER NOT NULL,
                error TEXT
            );
            """
        )
        con.execute("CREATE INDEX IF NOT EXISTS idx_supplier_download_key ON supplier_download(unique_key);")
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS supplier_asset (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                unique_key TEXT NOT NULL UNIQUE,
                updated_at TEXT NOT NULL,
                source_site TEXT NOT NULL,
                product_url TEXT,
                title TEXT,
                asset_status TEXT NOT NULL,
                asset_format TEXT,
                asset_source_url TEXT,
                asset_local_path TEXT,
                preview_local_path TEXT,
                blender_job_path TEXT,
                notes_json TEXT NOT NULL DEFAULT '[]',
                extra_json TEXT NOT NULL DEFAULT '{}'
            );
            """
        )
        con.execute("CREATE INDEX IF NOT EXISTS idx_supplier_asset_site ON supplier_asset(source_site);")
        con.execute("CREATE INDEX IF NOT EXISTS idx_supplier_asset_status ON supplier_asset(asset_status);")


def insert_download(
    db_path: Path,
    unique_key: str,
    downloaded_at: str,
    final_url: str | None,
    local_path: str | None,
    filename: str | None,
    content_type: str | None,
    ok: bool,
    size_bytes: int,
    error: str | None,
) -> None:
    with sqlite3.connect(db_path) as con:
        con.execute(
            """
            INSERT INTO supplier_download (
                unique_key,
                downloaded_at,
                final_url,
                local_path,
                filename,
                content_type,
                ok,
                size_bytes,
                error
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                unique_key,
                downloaded_at,
                final_url,
                local_path,
                filename,
                content_type,
                1 if ok else 0,
                size_bytes,
                error,
            ),
        )


def upsert_asset(db_path: Path, asset: SupplierAssetRecord) -> None:
    with sqlite3.connect(db_path) as con:
        con.execute(
            """
            INSERT INTO supplier_asset (
                unique_key,
                updated_at,
                source_site,
                product_url,
                title,
                asset_status,
                asset_format,
                asset_source_url,
                asset_local_path,
                preview_local_path,
                blender_job_path,
                notes_json,
                extra_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(unique_key) DO UPDATE SET
                updated_at=excluded.updated_at,
                source_site=excluded.source_site,
                product_url=excluded.product_url,
                title=excluded.title,
                asset_status=excluded.asset_status,
                asset_format=excluded.asset_format,
                asset_source_url=excluded.asset_source_url,
                asset_local_path=excluded.asset_local_path,
                preview_local_path=excluded.preview_local_path,
                blender_job_path=excluded.blender_job_path,
                notes_json=excluded.notes_json,
                extra_json=excluded.extra_json
            ;
            """,
            (
                asset.unique_key,
                asset.updated_at,
                asset.source_site,
                asset.product_url,
                asset.title,
                asset.asset_status,
                asset.asset_format,
                asset.asset_source_url,
                asset.asset_local_path,
                asset.preview_local_path,
                asset.blender_job_path,
                asset.notes_json,
                asset.extra_json,
            ),
        )
