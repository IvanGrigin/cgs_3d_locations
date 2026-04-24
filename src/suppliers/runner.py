# -*- coding: utf-8 -*-
"""
This module runs a single supplier parse and persists normalized records.
It bridges adapter outputs into ProductRecord objects and metadata JSON files.
The runner is the simplest operational entrypoint for page-level ingestion.
It is also reused by higher-level acquisition and debugging scripts.
Keep coercion rules centralized and backward compatible.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from src.suppliers.db import init_db, insert_fetch_log, upsert_products
from src.suppliers.models import ProductRecord
from src.suppliers.registry import find_adapter


def _metadata_slug(product: ProductRecord) -> str:
    base = re.sub(r"[^\w\-\.]+", "_", (product.title or product.unique_key), flags=re.UNICODE).strip("_")
    suffix_source = product.external_id or product.unique_key
    suffix = re.sub(r"[^\w\-\.]+", "_", str(suffix_source), flags=re.UNICODE).strip("_")
    if suffix and suffix != base:
        return f"{base}__{suffix}" if base else suffix
    return base or "product"


def save_metadata_json(product: ProductRecord, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = _metadata_slug(product)
    out_path = out_dir / f"{slug}.metadata.json"
    out_path.write_text(
        json.dumps(product.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out_path


def _to_cm(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value) * 100.0
    except Exception:
        return None


def _coerce_materials(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        clean = [str(x).strip() for x in value if str(x).strip()]
        return "; ".join(clean) if clean else None
    text = str(value).strip()
    return text or None


def _json_dumps(value: Any, default: Any) -> str:
    payload = value if value is not None else default
    return json.dumps(payload, ensure_ascii=False)


def coerce_product_record(item: ProductRecord | dict[str, Any], adapter, source_url: str, final_url: str) -> ProductRecord:
    if isinstance(item, ProductRecord):
        return item

    if not isinstance(item, dict):
        raise TypeError(f"Unsupported adapter result type: {type(item).__name__}")

    product_url = item.get("product_url")
    model_download_url = item.get("model_download_url")
    model_download_landing_url = item.get("model_download_landing_url")
    unique_key_url = product_url or model_download_url or model_download_landing_url or final_url

    extra = dict(item.get("attrs") or {})
    for key in [
        "title_short",
        "title_offer",
        "price_text",
        "volume_m3",
        "parse_stage",
        "enriched_from_product_page",
        "category",
    ]:
        if item.get(key) is not None:
            extra[key] = item.get(key)

    room_tags = item.get("room_tags") or []
    room = ", ".join(str(x).strip() for x in room_tags if str(x).strip()) or None

    return ProductRecord(
        unique_key=adapter.build_unique_key(str(unique_key_url), None),
        source_site=item.get("site") or adapter.site_name,
        source_url=item.get("source_url") or source_url,
        parsed_at=adapter.now_utc_iso(),
        external_id=item.get("external_id"),
        category_raw=item.get("category_raw"),
        category_norm=item.get("category_norm") or item.get("category"),
        title=item.get("title") or item.get("title_short"),
        brand=item.get("brand"),
        collection=item.get("collection"),
        product_url=product_url,
        model_link_type=(
            item.get("model_link_type")
            or ("direct_file" if model_download_url else "landing_page" if model_download_landing_url else None)
        ),
        model_page_url=item.get("model_page_url") or product_url or final_url,
        model_download_url=model_download_url,
        model_download_landing_url=model_download_landing_url,
        model_vendor_url=item.get("model_vendor_url") or product_url or final_url,
        model_extraction_method=item.get("model_extraction_method"),
        model_download_filename=item.get("model_download_filename") or adapter.filename_from_url(model_download_url),
        model_format=item.get("model_format") or adapter.ext_from_url(model_download_url),
        price_value=item.get("price_value"),
        price_currency=item.get("price_currency"),
        old_price_value=item.get("old_price_value"),
        style=item.get("style"),
        color=item.get("color"),
        description=item.get("description"),
        width_cm=item.get("width_cm") if item.get("width_cm") is not None else _to_cm(item.get("width_m")),
        depth_cm=item.get("depth_cm") if item.get("depth_cm") is not None else _to_cm(item.get("depth_m")),
        height_cm=item.get("height_cm") if item.get("height_cm") is not None else _to_cm(item.get("height_m")),
        weight_kg=item.get("weight_kg"),
        volume_m3=item.get("volume_m3"),
        package_width_cm=item.get("package_width_cm"),
        package_depth_cm=item.get("package_depth_cm"),
        package_height_cm=item.get("package_height_cm"),
        packed_weight_kg=item.get("packed_weight_kg"),
        scheme_url=item.get("scheme_url"),
        room=room,
        materials=_coerce_materials(item.get("materials")),
        availability=item.get("availability"),
        country_brand=item.get("country_brand"),
        production_country=item.get("production_country"),
        tags_json=_json_dumps(item.get("tags"), []),
        images_json=_json_dumps(item.get("preview_images") or item.get("images"), []),
        related_json=_json_dumps(item.get("related_items") or item.get("related"), []),
        extra_json=_json_dumps(extra, {}),
        raw_html=item.get("raw_html") or item.get("raw_html_snippet") or "",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--db", default="data/sourse/suppliers/suppliers.db")
    ap.add_argument("--out-dir", default="data/sourse/suppliers/items")
    args = ap.parse_args()

    db_path = Path(args.db).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    init_db(db_path)
    adapter = find_adapter(args.url)

    try:
        html, final_url = adapter.fetch_html(args.url)
        raw_items = adapter.parse(args.url, html, final_url)
        products = [coerce_product_record(item, adapter, args.url, final_url) for item in raw_items]

        if not products:
            if getattr(adapter, "empty_parse_is_skip", False):
                insert_fetch_log(
                    db_path=db_path,
                    source_site=adapter.site_name,
                    source_url=args.url,
                    fetched_at=adapter.now_utc_iso(),
                    ok=True,
                    error="skip: empty adapter result",
                )
                print(f"site: {adapter.site_name}")
                print("records: 0")
                print("status: skipped_empty_result")
                return
            raise ValueError(f"Адаптер {adapter.site_name} не вернул ни одной записи")

        upsert_products(db_path, products)
        insert_fetch_log(
            db_path=db_path,
            source_site=adapter.site_name,
            source_url=args.url,
            fetched_at=products[0].parsed_at,
            ok=True,
            error=None,
        )

        meta_paths = [save_metadata_json(product, out_dir) for product in products]

        print(f"site: {adapter.site_name}")
        print(f"records: {len(products)}")
        for product, meta_path in zip(products[:10], meta_paths[:10]):
            print(f"title: {product.title}")
            print(f"metadata_json: {meta_path}")
            print(f"model_download_url: {product.model_download_url}")
            print(f"model_download_landing_url: {product.model_download_landing_url}")
        if len(products) > 10:
            print(f"more_records: {len(products) - 10}")

    except Exception as e:
        insert_fetch_log(
            db_path=db_path,
            source_site=adapter.site_name,
            source_url=args.url,
            fetched_at=adapter.now_utc_iso(),
            ok=False,
            error=f"{type(e).__name__}: {e}",
        )
        raise


if __name__ == "__main__":
    main()
