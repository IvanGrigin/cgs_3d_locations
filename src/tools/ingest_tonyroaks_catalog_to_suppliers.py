#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Ingest scraped Tony Roaks products into the main supplier catalog DB."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from src.suppliers.db import init_db, upsert_products
from src.suppliers.models import ProductRecord
from src.tools.normalize_supplier_categories_taxonomy import infer_category_from_mapping


DEFAULT_INPUTS = [
    "data/sourse/suppliers/tonyroaks/catalog/products_with_3d_models.jsonl",
    "data/sourse/suppliers/tonyroaks/catalog/products_without_3d_models.jsonl",
]
DEFAULT_DB = "data/sourse/suppliers/suppliers.db"


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).replace("\xa0", " ").replace("\u2800", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def _json_dumps(value: Any) -> str:
    if value in (None, ""):
        value = [] if isinstance(value, list) else {}
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _first_option_values(row: dict[str, Any], title: str) -> list[str]:
    out: list[str] = []
    for opt in row.get("options") or []:
        if not isinstance(opt, dict):
            continue
        if (_text(opt.get("title")) or "").lower() != title.lower():
            continue
        values = opt.get("values") or []
        for value in values:
            if isinstance(value, dict):
                value = value.get("value")
            text = _text(value)
            if text and text not in out:
                out.append(text)
    return out


def _variant_dimensions_cm(row: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    variants = row.get("variants") if isinstance(row.get("variants"), list) else []
    for variant in variants:
        if not isinstance(variant, dict):
            continue
        x = _float_or_none(variant.get("pack_x"))
        y = _float_or_none(variant.get("pack_y"))
        z = _float_or_none(variant.get("pack_z"))
        if x and y and z:
            return x / 10.0, y / 10.0, z / 10.0

    dims = row.get("dimensions") if isinstance(row.get("dimensions"), dict) else {}
    variant_dims = dims.get("variant_dimensions") if isinstance(dims.get("variant_dimensions"), list) else []
    for item in variant_dims:
        if not isinstance(item, dict):
            continue
        x = _float_or_none(item.get("pack_x"))
        y = _float_or_none(item.get("pack_y"))
        z = _float_or_none(item.get("pack_z"))
        if x and y and z:
            return x / 10.0, y / 10.0, z / 10.0
    return None, None, None


def _weight_from_props(row: dict[str, Any]) -> float | None:
    props = row.get("properties") if isinstance(row.get("properties"), dict) else {}
    for key in ["Минимальный вес", "Вес"]:
        value = _text(props.get(key))
        if not value:
            continue
        match = re.search(r"\d+(?:[.,]\d+)?", value)
        if match:
            return float(match.group(0).replace(",", "."))
    return None


def _materials(row: dict[str, Any]) -> str | None:
    desc = row.get("description") or ""
    lines = [_text(line) for line in str(desc).splitlines()]
    lines = [line for line in lines if line]
    if "Материалы и покрытие:" in lines:
        start = lines.index("Материалы и покрытие:") + 1
        stop = len(lines)
        for idx in range(start, len(lines)):
            if lines[idx].endswith(":"):
                stop = idx
                break
        return "; ".join(lines[start:stop]) or None
    props = row.get("properties") if isinstance(row.get("properties"), dict) else {}
    return _text(props.get("Материалы и покрытие") or props.get("Материал"))


def _category_raw(row: dict[str, Any]) -> str | None:
    section = _text(row.get("section_title"))
    if section:
        return section
    categories = row.get("categories") if isinstance(row.get("categories"), list) else []
    for item in categories:
        text = _text(item)
        if text:
            return text
    return _text(row.get("category_path"))


def _infer_tonyroaks_category(row: dict[str, Any], category_raw: str | None, materials: str | None) -> tuple[str, str, str]:
    title = _text(row.get("name")) or ""
    category_path = _text(row.get("category_path")) or ""
    text = " ".join(x for x in [title, category_raw or "", category_path] if x).lower().replace("ё", "е")
    if "торшер" in text:
        return "floor_lamp", "tonyroaks.title.floor_lamp", title
    if "светильник" in text or "ламп" in text:
        return "floor_lamp", "tonyroaks.title.light", title
    if "зеркал" in text:
        return "mirror", "tonyroaks.title.mirror", title
    if "подсвечник" in text:
        return "decor", "tonyroaks.title.candlestick", title
    if "часы" in text or "clock" in text:
        return "wall_clock", "tonyroaks.title.clock", title
    if "скам" in text or "банкет" in text:
        return "bench", "tonyroaks.title.bench", title
    if "стул" in text:
        return "chair", "tonyroaks.title.chair", title
    if "стеллаж" in text or "теллаж" in text:
        return "shelving", "tonyroaks.title.shelving", title
    if "полк" in text:
        return "shelf", "tonyroaks.title.shelf", title
    if "комод" in text:
        return "dresser", "tonyroaks.title.dresser", title
    if "тумба под тв" in text:
        return "tv_stand", "tonyroaks.title.tv_stand", title
    if "прикроватн" in text:
        return "nightstand", "tonyroaks.title.nightstand", title
    if "офисная тумба" in text or "тумба на колесиках" in text:
        return "cabinet", "tonyroaks.title.cabinet", title
    if "консоль" in text:
        return "console_table", "tonyroaks.title.console_table", title
    if "журнальн" in text:
        return "coffee_table", "tonyroaks.title.coffee_table", title
    if "рабочий стол" in text:
        return "desk", "tonyroaks.title.desk", title
    if "стол" in text:
        return "dining_table", "tonyroaks.title.table", title
    return infer_category_from_mapping(
        {
            "source_site": "tonyroaks",
            "category_raw": category_raw,
            "category_norm": None,
            "title": title,
            "description": row.get("description") or "",
            "materials": materials,
            "tags_json": [],
            "extra_json": {},
        }
    )


def _collection(row: dict[str, Any]) -> str | None:
    title = _text(row.get("name")) or ""
    words = title.split()
    if words:
        return words[-1].upper()
    return None


def _record_from_row(row: dict[str, Any], parsed_at: str) -> ProductRecord:
    url = _text(row.get("url")) or ""
    model_links = row.get("model_links") if isinstance(row.get("model_links"), list) else []
    model_url = _text(model_links[0].get("url")) if model_links and isinstance(model_links[0], dict) else None
    colors = _first_option_values(row, "Цвет") or _first_option_values(row, "Цвет эмали")
    sizes = _first_option_values(row, "Размер")
    width_cm, depth_cm, height_cm = _variant_dimensions_cm(row)
    materials = _materials(row)
    category_raw = _category_raw(row)
    category_norm, category_rule, effective_title = _infer_tonyroaks_category(row, category_raw, materials)
    product_uid = _text(row.get("product_uid"))
    slug = urlparse(url).path.rstrip("/").split("/")[-1]
    external_id = product_uid or slug
    category_path = _text(row.get("category_path"))

    extra = {
        "source": "tonyroaks_3d_model_catalog_scraper",
        "category_path": category_path,
        "category_url": row.get("category_url"),
        "category_title": row.get("category_title"),
        "section_title": row.get("section_title"),
        "recid": row.get("recid"),
        "storepartuid": row.get("storepartuid"),
        "product_uid": product_uid,
        "external_id": row.get("external_id"),
        "model_links": model_links,
        "description_html": row.get("description_html"),
        "properties": row.get("properties") or {},
        "dimensions_raw": row.get("dimensions") or {},
        "options": row.get("options") or [],
        "variants": row.get("variants") or [],
        "raw_product": row.get("raw_product") or {},
        "category_rule": category_rule,
        "category_effective_title": effective_title,
        "color_options": colors,
        "size_options": sizes,
        "variants_count": len(row.get("variants") or []),
    }
    tags = [x for x in ["3d_model_button" if model_url else None, category_path, category_raw, category_norm] if x]

    return ProductRecord(
        unique_key=f"tonyroaks::product::{url}",
        source_site="tonyroaks",
        source_url=url,
        parsed_at=parsed_at,
        external_id=external_id,
        category_raw=category_raw,
        category_norm=category_norm,
        title=_text(row.get("name")),
        brand="Tony Roaks",
        collection=_collection(row),
        product_url=url,
        model_link_type="yandex_disk_button" if model_url else None,
        model_page_url=url if model_url else None,
        model_download_url=model_url,
        model_download_landing_url=model_url,
        model_vendor_url=url,
        model_extraction_method="tonyroaks_description_download_link" if model_url else None,
        model_download_filename=None,
        model_format=None,
        price_value=_float_or_none(row.get("price")),
        price_currency=_text(row.get("price_currency")) or "RUB",
        old_price_value=_float_or_none(row.get("old_price")),
        style=None,
        color=", ".join(colors) if colors else None,
        description=_text(row.get("description")),
        width_cm=width_cm,
        depth_cm=depth_cm,
        height_cm=height_cm,
        weight_kg=_weight_from_props(row),
        volume_m3=None,
        package_width_cm=None,
        package_depth_cm=None,
        package_height_cm=None,
        packed_weight_kg=None,
        scheme_url=None,
        room=None,
        materials=materials,
        availability=None,
        country_brand=None,
        production_country="Россия",
        tags_json=_json_dumps(tags),
        images_json=_json_dumps(row.get("images") or []),
        related_json="[]",
        extra_json=_json_dumps(extra),
        raw_html="",
    )


def _load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if not isinstance(item, dict):
                    continue
                url = _text(item.get("url"))
                if url and url in seen_urls:
                    continue
                if url:
                    seen_urls.add(url)
                rows.append(item)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", help="Tony Roaks product JSONL; can be repeated")
    parser.add_argument("--db", default=DEFAULT_DB, help="Main suppliers.db path")
    args = parser.parse_args()

    input_paths = [Path(p).expanduser().resolve() for p in (args.input or DEFAULT_INPUTS)]
    db_path = Path(args.db).expanduser().resolve()
    rows = _load_rows(input_paths)
    parsed_at = _now_utc_iso()
    records = [_record_from_row(row, parsed_at) for row in rows]

    init_db(db_path)
    upsert_products(db_path, records)
    print(f"input_rows = {len(rows)}")
    print(f"upserted_records = {len(records)}")
    print(f"with_model_download_url = {sum(1 for r in records if r.model_download_url)}")
    print(f"db = {db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
