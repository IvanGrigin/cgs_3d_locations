# -*- coding: utf-8 -*-
"""
This module defines normalized supplier record dataclasses.
These records are the core interchange format across parsers and exporters.
The schema is intentionally flat to simplify persistence and diffing.
Most supplier modules construct or transform these dataclasses directly.
Keep field additions deliberate and broadly useful.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional


@dataclass
class ProductRecord:
    unique_key: str
    source_site: str
    source_url: str
    parsed_at: str

    external_id: Optional[str] = None

    category_raw: Optional[str] = None
    category_norm: Optional[str] = None

    title: Optional[str] = None
    brand: Optional[str] = None
    collection: Optional[str] = None

    product_url: Optional[str] = None

    model_link_type: Optional[str] = None
    model_page_url: Optional[str] = None
    model_download_url: Optional[str] = None
    model_download_landing_url: Optional[str] = None
    model_vendor_url: Optional[str] = None
    model_extraction_method: Optional[str] = None
    model_download_filename: Optional[str] = None
    model_format: Optional[str] = None

    price_value: Optional[float] = None
    price_currency: Optional[str] = None
    old_price_value: Optional[float] = None

    style: Optional[str] = None
    color: Optional[str] = None
    description: Optional[str] = None

    width_cm: Optional[float] = None
    depth_cm: Optional[float] = None
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    volume_m3: Optional[float] = None
    package_width_cm: Optional[float] = None
    package_depth_cm: Optional[float] = None
    package_height_cm: Optional[float] = None
    packed_weight_kg: Optional[float] = None
    scheme_url: Optional[str] = None

    room: Optional[str] = None
    materials: Optional[str] = None
    availability: Optional[str] = None
    country_brand: Optional[str] = None
    production_country: Optional[str] = None

    tags_json: str = "[]"
    images_json: str = "[]"
    related_json: str = "[]"
    extra_json: str = "{}"

    raw_html: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
