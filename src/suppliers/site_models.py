# -*- coding: utf-8 -*-
"""
This module defines small dataclasses for download and local asset state.
These records carry file-resolution results across acquisition steps.
The types are intentionally separate from parsed product metadata.
They help keep download bookkeeping explicit and typed.
Keep them minimal and storage-oriented.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.suppliers.models import ProductRecord


@dataclass
class DownloadResult:
    final_url: str | None
    local_path: str | None
    filename: str | None
    content_type: str | None
    ok: bool
    size_bytes: int
    error: str | None = None


@dataclass
class SupplierAssetRecord:
    unique_key: str
    updated_at: str
    source_site: str
    product_url: str | None
    title: str | None
    asset_status: str
    asset_format: str | None
    asset_source_url: str | None
    asset_local_path: str | None
    preview_local_path: str | None
    blender_job_path: str | None
    notes_json: str
    extra_json: str
