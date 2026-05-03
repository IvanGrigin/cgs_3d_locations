#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Ingest scraped Stool Group products into the main supplier catalog DB."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from src.suppliers.db import init_db, upsert_products
from src.suppliers.models import ProductRecord
from src.tools.normalize_supplier_categories_taxonomy import infer_category_from_mapping


DEFAULT_INPUT = "data/sourse/suppliers/stoolgroup/catalog/products_with_3d_models.jsonl"
DEFAULT_DB = "data/sourse/suppliers/suppliers.db"


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).replace("\xa0", " ")
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


def _number_from_text(value: Any) -> float | None:
    text = _text(value)
    if not text:
        return None
    match = re.search(r"\d+(?:[.,]\d+)?", text.replace(" ", ""))
    return float(match.group(0).replace(",", ".")) if match else None


def _dims_from_properties(props: dict[str, Any]) -> dict[str, float | None]:
    dims: dict[str, float | None] = {
        "width_cm": None,
        "depth_cm": None,
        "height_cm": None,
        "weight_kg": None,
        "volume_m3": None,
        "package_width_cm": None,
        "package_depth_cm": None,
        "package_height_cm": None,
        "packed_weight_kg": None,
    }

    size = _text(props.get("Габариты В*Ш*Г") or props.get("Размер") or "")
    if size:
        nums = [float(x.replace(",", ".")) for x in re.findall(r"\d+(?:[.,]\d+)?", size)]
        if len(nums) >= 3:
            dims["height_cm"], dims["width_cm"], dims["depth_cm"] = nums[:3]

    label_map = {
        "Ширина изделия (габарит)": "width_cm",
        "Глубина изделия (габарит)": "depth_cm",
        "Высота изделия (габарит)": "height_cm",
        "Вес нетто изделия": "weight_kg",
        "Вес брутто изделия": "packed_weight_kg",
        "Объем": "volume_m3",
        "Ширина упаковки": "package_width_cm",
        "Глубина упаковки": "package_depth_cm",
        "Высота упаковки": "package_height_cm",
        "Вес упаковки": "packed_weight_kg",
    }
    for label, target in label_map.items():
        value = _number_from_text(props.get(label))
        if value is not None:
            dims[target] = value
    return dims


def _model_format(row: dict[str, Any]) -> str | None:
    links = row.get("model_links") or []
    text = " ".join(
        _text(part) or ""
        for link in links
        if isinstance(link, dict)
        for part in [link.get("file_text"), link.get("url"), link.get("text")]
    ).lower()
    for ext in ["zip", "rar", "7z", "fbx", "obj", "max", "glb", "gltf"]:
        if re.search(rf"\b{re.escape(ext)}\b|\.{re.escape(ext)}(?:\?|$)", text):
            return ext
    return None


def _model_filename(row: dict[str, Any], ext: str | None) -> str | None:
    links = row.get("model_links") or []
    url = links[0].get("url", "") if links and isinstance(links[0], dict) else ""
    attachment_id = parse_qs(urlparse(url).query).get("attachment_id", [""])[0]
    if attachment_id:
        return f"stoolgroup_attachment_{attachment_id}.{ext or 'archive'}"
    slug = urlparse(row.get("url") or row.get("final_url") or "").path.rstrip("/").split("/")[-1]
    return f"{slug}.{ext}" if slug and ext else None


def _category_raw(row: dict[str, Any]) -> str | None:
    props = row.get("properties") if isinstance(row.get("properties"), dict) else {}
    categories = row.get("categories") if isinstance(row.get("categories"), list) else []
    if categories:
        return _text(categories[0])
    return _text(props.get("Категория"))


def _infer_stoolgroup_category(row: dict[str, Any], category_raw: str | None, materials: str | None) -> tuple[str, str, str]:
    title = _text(row.get("name")) or ""
    text = " ".join(x for x in [title, category_raw or "", _source_category(row) or ""] if x).lower().replace("ё", "е")
    if "диван-кровать" in text:
        return "sofa_bed", "stoolgroup.title.sofa_bed", title
    if "диван" in text:
        return "sofa", "stoolgroup.title.sofa", title
    if "офисн" in text or "компьютерн" in text or "руководител" in text or "игров" in text:
        if "кресло" in text or "стул" in text:
            return "office_chair", "stoolgroup.title.office_chair", title
    if "кресло" in text:
        return "armchair", "stoolgroup.title.armchair", title
    if "барный стул" in text or "полубарный" in text or "табурет" in text:
        return "stool", "stoolgroup.title.stool", title
    if "обеденный стул" in text:
        return "dining_chair", "stoolgroup.title.dining_chair", title
    if "стул" in text:
        return "chair", "stoolgroup.title.chair", title
    if "пуф" in text or "банкет" in text or "оттоманк" in text:
        return "ottoman_pouf", "stoolgroup.title.ottoman_pouf", title
    if "консоль" in text:
        return "console_table", "stoolgroup.title.console_table", title
    if "журнальн" in text or "кофейн" in text:
        return "coffee_table", "stoolgroup.title.coffee_table", title
    if "стол" in text:
        return "dining_table", "stoolgroup.title.table", title
    if "бра" in text or "настенный" in text:
        return "wall_light", "stoolgroup.title.wall_light", title
    if "торшер" in text:
        return "floor_lamp", "stoolgroup.title.floor_lamp", title
    if "настольная лампа" in text:
        return "table_lamp", "stoolgroup.title.table_lamp", title
    if "люстра" in text:
        return "chandelier", "stoolgroup.title.chandelier", title
    if "подвес" in text or "светильник" in text:
        return "pendant_lamp", "stoolgroup.title.pendant_lamp", title
    return infer_category_from_mapping(
        {
            "source_site": "stoolgroup",
            "category_raw": category_raw,
            "category_norm": None,
            "title": title,
            "description": "",
            "materials": materials,
            "tags_json": [],
            "extra_json": {},
        }
    )


def _source_category(row: dict[str, Any]) -> str | None:
    page_url = _text(row.get("source_page_url")) or ""
    path = urlparse(page_url).path.strip("/")
    return path.split("/")[0] if path else None


def _record_from_row(row: dict[str, Any], parsed_at: str) -> ProductRecord:
    props = row.get("properties") if isinstance(row.get("properties"), dict) else {}
    dims = _dims_from_properties(props)
    url = _text(row.get("final_url") or row.get("url")) or ""
    product_id = _text(row.get("product_id"))
    slug = urlparse(url).path.rstrip("/").split("/")[-1]
    external_id = product_id or slug
    model_links = row.get("model_links") if isinstance(row.get("model_links"), list) else []
    model_url = _text(model_links[0].get("url")) if model_links and isinstance(model_links[0], dict) else None
    fmt = _model_format(row)
    category_raw = _category_raw(row)
    color = _text(row.get("colors", {}).get("Цвет обивки") if isinstance(row.get("colors"), dict) else None)
    color = color or _text(row.get("colors", {}).get("Цвет") if isinstance(row.get("colors"), dict) else None)
    materials = " ".join(
        x
        for x in [
            _text(props.get("Материал обивки")),
            _text(props.get("Материал каркаса")),
            _text(props.get("Материал ножек")),
            _text(props.get("Материал столешницы")),
        ]
        if x
    ) or None
    style = _text(props.get("Стиль"))

    category_norm, category_rule, effective_title = _infer_stoolgroup_category(row, category_raw, materials)

    extra = {
        "source": "stoolgroup_3d_model_catalog_scraper",
        "source_category_url_slug": _source_category(row),
        "source_page_url": row.get("source_page_url"),
        "source_page_number": row.get("source_page_number"),
        "download_links": row.get("download_links") or [],
        "model_links": model_links,
        "properties": props,
        "dimensions_raw": row.get("dimensions") or {},
        "colors_raw": row.get("colors") or {},
        "category_rule": category_rule,
        "category_effective_title": effective_title,
        "model_file_text": model_links[0].get("file_text") if model_links and isinstance(model_links[0], dict) else None,
        "model_format_from_button": fmt,
        "stoolgroup_product_id": product_id,
    }
    tags = [x for x in ["3d_model_button", fmt, category_raw, style, color] if x]

    return ProductRecord(
        unique_key=f"stoolgroup::product::{url}",
        source_site="stoolgroup",
        source_url=url,
        parsed_at=parsed_at,
        external_id=external_id,
        category_raw=category_raw,
        category_norm=category_norm,
        title=_text(row.get("name")),
        brand=_text(row.get("brand")) or "Stool Group",
        collection=_text(props.get("Модель")),
        product_url=url,
        model_link_type="direct_attachment_button",
        model_page_url=url,
        model_download_url=model_url,
        model_download_landing_url=None,
        model_vendor_url=url,
        model_extraction_method="stoolgroup_download_list_item",
        model_download_filename=_model_filename(row, fmt),
        model_format=fmt,
        price_value=_float_or_none(row.get("price")),
        price_currency=_text(row.get("price_currency")) or "RUB",
        old_price_value=_float_or_none(row.get("old_price")),
        style=style,
        color=color,
        description=_text(row.get("description")),
        width_cm=dims["width_cm"],
        depth_cm=dims["depth_cm"],
        height_cm=dims["height_cm"],
        weight_kg=dims["weight_kg"],
        volume_m3=dims["volume_m3"],
        package_width_cm=dims["package_width_cm"],
        package_depth_cm=dims["package_depth_cm"],
        package_height_cm=dims["package_height_cm"],
        packed_weight_kg=dims["packed_weight_kg"],
        scheme_url=None,
        room=_text(props.get("Назначение")),
        materials=materials,
        availability=_text(row.get("availability") or row.get("stock_text")),
        country_brand=None,
        production_country=_text(props.get("Страна производства")),
        tags_json=_json_dumps(tags),
        images_json=_json_dumps(row.get("images") or []),
        related_json="[]",
        extra_json=_json_dumps(extra),
        raw_html="",
    )


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if isinstance(item, dict):
                rows.append(item)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Stoolgroup products_with_3d_models.jsonl")
    parser.add_argument("--db", default=DEFAULT_DB, help="Main suppliers.db path")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    db_path = Path(args.db).expanduser().resolve()
    rows = _load_rows(input_path)
    parsed_at = _now_utc_iso()
    records = [_record_from_row(row, parsed_at) for row in rows]

    init_db(db_path)
    upsert_products(db_path, records)
    print(f"input_rows = {len(rows)}")
    print(f"upserted_records = {len(records)}")
    print(f"db = {db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
