#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from src.suppliers.utils import sqlite_table_exists
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.suppliers.utils import sqlite_table_exists


SCHEMA_VERSION = "supplier_storage_one_table/v1"

TABLE_COLUMNS: list[tuple[str, str]] = [
    ("unique_key", "TEXT NOT NULL"),
    ("source_site", "TEXT"),
    ("source_url", "TEXT"),
    ("parsed_at", "TEXT"),
    ("external_id", "TEXT"),
    ("category_raw", "TEXT"),
    ("category_norm", "TEXT"),
    ("title", "TEXT"),
    ("brand", "TEXT"),
    ("collection", "TEXT"),
    ("product_url", "TEXT"),
    ("model_link_type", "TEXT"),
    ("model_page_url", "TEXT"),
    ("model_download_url", "TEXT"),
    ("model_download_landing_url", "TEXT"),
    ("model_vendor_url", "TEXT"),
    ("model_extraction_method", "TEXT"),
    ("model_download_filename", "TEXT"),
    ("model_format", "TEXT"),
    ("price_value", "REAL"),
    ("price_currency", "TEXT"),
    ("old_price_value", "REAL"),
    ("style", "TEXT"),
    ("color", "TEXT"),
    ("description", "TEXT"),
    ("width_cm", "REAL"),
    ("depth_cm", "REAL"),
    ("height_cm", "REAL"),
    ("weight_kg", "REAL"),
    ("volume_m3", "REAL"),
    ("package_width_cm", "REAL"),
    ("package_depth_cm", "REAL"),
    ("package_height_cm", "REAL"),
    ("packed_weight_kg", "REAL"),
    ("scheme_url", "TEXT"),
    ("room", "TEXT"),
    ("materials", "TEXT"),
    ("availability", "TEXT"),
    ("country_brand", "TEXT"),
    ("production_country", "TEXT"),
    ("tags_json", "TEXT"),
    ("images_json", "TEXT"),
    ("related_json", "TEXT"),
    ("extra_json", "TEXT"),
    ("asset_status", "TEXT"),
    ("asset_format", "TEXT"),
    ("asset_local_path", "TEXT"),
    ("preview_local_path", "TEXT"),
    ("asset_source_url", "TEXT"),
    ("model_probe_has_fbx", "INTEGER"),
    ("model_probe_checked_at", "TEXT"),
    ("model_probe_error", "TEXT"),
    ("has_model_url", "INTEGER"),
    ("has_asset_local", "INTEGER"),
    ("preferred_source_kind", "TEXT"),
    ("preferred_source_file", "TEXT"),
    ("preferred_source_table", "TEXT"),
    ("preferred_source_rank", "INTEGER"),
    ("merged_from_count", "INTEGER"),
    ("merged_from_sources_json", "TEXT"),
    ("raw_json", "TEXT"),
]

BASE_FIELDS = [name for name, _ in TABLE_COLUMNS if name not in {"preferred_source_kind", "preferred_source_file", "preferred_source_table", "preferred_source_rank", "merged_from_count", "merged_from_sources_json"}]
JSON_FIELDS = {"tags_json", "images_json", "related_json", "extra_json", "merged_from_sources_json", "raw_json"}


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _json_loads_or(value: Any, default: Any) -> Any:
    if value in (None, "", b""):
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return default


def _json_dumps(value: Any) -> str | None:
    if value in (None, "", [], {}):
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _has_value(value: Any) -> bool:
    return value not in (None, "", [], {})


def _text_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _bool_to_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1 if value else 0
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return 1
    if text in {"0", "false", "no", "n"}:
        return 0
    return None


def _merge_lists(values: list[Any]) -> list[Any]:
    out: list[Any] = []
    seen: set[str] = set()
    for value in values:
        if isinstance(value, list):
            items = value
        elif _has_value(value):
            items = [value]
        else:
            continue
        for item in items:
            key = json.dumps(item, ensure_ascii=False, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
    return out


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path.resolve())


def _top_bucket(path: Path, root: Path) -> str:
    rel = path.resolve().relative_to(root.resolve())
    return rel.parts[0] if rel.parts else "."


def _dimensions_from_json(value: Any) -> dict[str, float | None]:
    data = _json_loads_or(value, {})
    if not isinstance(data, dict):
        return {}
    return {
        "width_cm": _float_or_none(data.get("width")),
        "depth_cm": _float_or_none(data.get("depth")),
        "height_cm": _float_or_none(data.get("height")),
        "weight_kg": _float_or_none(data.get("weight_kg")),
        "package_width_cm": _float_or_none(data.get("package_width")),
        "package_depth_cm": _float_or_none(data.get("package_depth")),
        "package_height_cm": _float_or_none(data.get("package_height")),
        "packed_weight_kg": _float_or_none(data.get("packed_weight_kg")),
        "volume_m3": _float_or_none(data.get("volume_m3")),
    }


def _candidate_base() -> dict[str, Any]:
    return {
        "unique_key": None,
        "source_site": None,
        "source_url": None,
        "parsed_at": None,
        "external_id": None,
        "category_raw": None,
        "category_norm": None,
        "title": None,
        "brand": None,
        "collection": None,
        "product_url": None,
        "model_link_type": None,
        "model_page_url": None,
        "model_download_url": None,
        "model_download_landing_url": None,
        "model_vendor_url": None,
        "model_extraction_method": None,
        "model_download_filename": None,
        "model_format": None,
        "price_value": None,
        "price_currency": None,
        "old_price_value": None,
        "style": None,
        "color": None,
        "description": None,
        "width_cm": None,
        "depth_cm": None,
        "height_cm": None,
        "weight_kg": None,
        "volume_m3": None,
        "package_width_cm": None,
        "package_depth_cm": None,
        "package_height_cm": None,
        "packed_weight_kg": None,
        "scheme_url": None,
        "room": None,
        "materials": None,
        "availability": None,
        "country_brand": None,
        "production_country": None,
        "tags_json": [],
        "images_json": [],
        "related_json": [],
        "extra_json": {},
        "asset_status": None,
        "asset_format": None,
        "asset_local_path": None,
        "preview_local_path": None,
        "asset_source_url": None,
        "model_probe_has_fbx": None,
        "model_probe_checked_at": None,
        "model_probe_error": None,
        "has_model_url": None,
        "has_asset_local": None,
        "raw_json": None,
    }


def _source_rank(path: Path, table: str) -> int:
    name = path.name
    if name == "suppliers.db" and table == "supplier_product":
        return 100
    if table == "supplier_product" and not name.startswith("site_assets_"):
        return 80
    if table == "supplier_product":
        return 60
    if table == "supplier_item":
        return 50
    return 10


def _candidate_score(candidate: dict[str, Any]) -> tuple[Any, ...]:
    dims_count = sum(candidate.get(key) is not None for key in ("width_cm", "depth_cm", "height_cm"))
    info_count = sum(
        1
        for key in (
            "title",
            "brand",
            "category_raw",
            "category_norm",
            "description",
            "materials",
            "product_url",
            "model_download_url",
            "asset_local_path",
            "price_value",
        )
        if _has_value(candidate.get(key))
    )
    rank = int(candidate.get("_source_rank") or 0)
    return (
        rank,
        dims_count,
        info_count,
        str(candidate.get("parsed_at") or ""),
        str(candidate.get("_source_file") or ""),
    )


def _sqlite_table_rows(con: sqlite3.Connection, table: str) -> int:
    return int(con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def _load_supplier_product_candidates(db_path: Path) -> list[dict[str, Any]]:
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        has_asset = sqlite_table_exists(con, "supplier_asset")
        present = {row[1] for row in con.execute("PRAGMA table_info(supplier_product)").fetchall()}

        def col(name: str) -> str:
            return f"sp.{name}" if name in present else f"NULL AS {name}"

        sql = """
        SELECT
            {unique_key},
            {source_site},
            {source_url},
            {parsed_at},
            {external_id},
            {category_raw},
            {category_norm},
            {title},
            {brand},
            {collection},
            {product_url},
            {model_link_type},
            {model_page_url},
            {model_download_url},
            {model_download_landing_url},
            {model_vendor_url},
            {model_extraction_method},
            {model_download_filename},
            {model_format},
            {price_value},
            {price_currency},
            {old_price_value},
            {style},
            {color},
            {description},
            {width_cm},
            {depth_cm},
            {height_cm},
            {weight_kg},
            {volume_m3},
            {package_width_cm},
            {package_depth_cm},
            {package_height_cm},
            {packed_weight_kg},
            {scheme_url},
            {room},
            {materials},
            {availability},
            {country_brand},
            {production_country},
            {tags_json},
            {images_json},
            {related_json},
            {extra_json},
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
                unique_key=col("unique_key"),
                source_site=col("source_site"),
                source_url=col("source_url"),
                parsed_at=col("parsed_at"),
                external_id=col("external_id"),
                category_raw=col("category_raw"),
                category_norm=col("category_norm"),
                title=col("title"),
                brand=col("brand"),
                collection=col("collection"),
                product_url=col("product_url"),
                model_link_type=col("model_link_type"),
                model_page_url=col("model_page_url"),
                model_download_url=col("model_download_url"),
                model_download_landing_url=col("model_download_landing_url"),
                model_vendor_url=col("model_vendor_url"),
                model_extraction_method=col("model_extraction_method"),
                model_download_filename=col("model_download_filename"),
                model_format=col("model_format"),
                price_value=col("price_value"),
                price_currency=col("price_currency"),
                old_price_value=col("old_price_value"),
                style=col("style"),
                color=col("color"),
                description=col("description"),
                width_cm=col("width_cm"),
                depth_cm=col("depth_cm"),
                height_cm=col("height_cm"),
                weight_kg=col("weight_kg"),
                volume_m3=col("volume_m3"),
                package_width_cm=col("package_width_cm"),
                package_depth_cm=col("package_depth_cm"),
                package_height_cm=col("package_height_cm"),
                packed_weight_kg=col("packed_weight_kg"),
                scheme_url=col("scheme_url"),
                room=col("room"),
                materials=col("materials"),
                availability=col("availability"),
                country_brand=col("country_brand"),
                production_country=col("production_country"),
                tags_json=col("tags_json"),
                images_json=col("images_json"),
                related_json=col("related_json"),
                extra_json=col("extra_json"),
            )
        else:
            sql = sql.format(
                asset_status="NULL AS asset_status",
                asset_format="NULL AS asset_format",
                asset_local_path="NULL AS asset_local_path",
                preview_local_path="NULL AS preview_local_path",
                asset_source_url="NULL AS asset_source_url",
                asset_join="",
                unique_key=col("unique_key"),
                source_site=col("source_site"),
                source_url=col("source_url"),
                parsed_at=col("parsed_at"),
                external_id=col("external_id"),
                category_raw=col("category_raw"),
                category_norm=col("category_norm"),
                title=col("title"),
                brand=col("brand"),
                collection=col("collection"),
                product_url=col("product_url"),
                model_link_type=col("model_link_type"),
                model_page_url=col("model_page_url"),
                model_download_url=col("model_download_url"),
                model_download_landing_url=col("model_download_landing_url"),
                model_vendor_url=col("model_vendor_url"),
                model_extraction_method=col("model_extraction_method"),
                model_download_filename=col("model_download_filename"),
                model_format=col("model_format"),
                price_value=col("price_value"),
                price_currency=col("price_currency"),
                old_price_value=col("old_price_value"),
                style=col("style"),
                color=col("color"),
                description=col("description"),
                width_cm=col("width_cm"),
                depth_cm=col("depth_cm"),
                height_cm=col("height_cm"),
                weight_kg=col("weight_kg"),
                volume_m3=col("volume_m3"),
                package_width_cm=col("package_width_cm"),
                package_depth_cm=col("package_depth_cm"),
                package_height_cm=col("package_height_cm"),
                packed_weight_kg=col("packed_weight_kg"),
                scheme_url=col("scheme_url"),
                room=col("room"),
                materials=col("materials"),
                availability=col("availability"),
                country_brand=col("country_brand"),
                production_country=col("production_country"),
                tags_json=col("tags_json"),
                images_json=col("images_json"),
                related_json=col("related_json"),
                extra_json=col("extra_json"),
            )

        candidates: list[dict[str, Any]] = []
        for row in con.execute(sql):
            raw = dict(row)
            unique_key = _text_or_none(raw.get("unique_key"))
            if not unique_key:
                continue
            extra = _json_loads_or(raw.get("extra_json"), {})
            if not isinstance(extra, dict):
                extra = {}

            candidate = _candidate_base()
            candidate.update(
                {
                    "unique_key": unique_key,
                    "source_site": _text_or_none(raw.get("source_site")),
                    "source_url": _text_or_none(raw.get("source_url")),
                    "parsed_at": _text_or_none(raw.get("parsed_at")),
                    "external_id": _text_or_none(raw.get("external_id")),
                    "category_raw": _text_or_none(raw.get("category_raw")),
                    "category_norm": _text_or_none(raw.get("category_norm")),
                    "title": _text_or_none(raw.get("title")),
                    "brand": _text_or_none(raw.get("brand")),
                    "collection": _text_or_none(raw.get("collection")),
                    "product_url": _text_or_none(raw.get("product_url")),
                    "model_link_type": _text_or_none(raw.get("model_link_type")),
                    "model_page_url": _text_or_none(raw.get("model_page_url")),
                    "model_download_url": _text_or_none(raw.get("model_download_url")),
                    "model_download_landing_url": _text_or_none(raw.get("model_download_landing_url")),
                    "model_vendor_url": _text_or_none(raw.get("model_vendor_url")),
                    "model_extraction_method": _text_or_none(raw.get("model_extraction_method")),
                    "model_download_filename": _text_or_none(raw.get("model_download_filename")),
                    "model_format": _text_or_none(raw.get("model_format")),
                    "price_value": raw.get("price_value"),
                    "price_currency": _text_or_none(raw.get("price_currency")),
                    "old_price_value": raw.get("old_price_value"),
                    "style": _text_or_none(raw.get("style")),
                    "color": _text_or_none(raw.get("color")),
                    "description": _text_or_none(raw.get("description")),
                    "width_cm": raw.get("width_cm"),
                    "depth_cm": raw.get("depth_cm"),
                    "height_cm": raw.get("height_cm"),
                    "weight_kg": raw.get("weight_kg"),
                    "volume_m3": raw.get("volume_m3"),
                    "package_width_cm": raw.get("package_width_cm"),
                    "package_depth_cm": raw.get("package_depth_cm"),
                    "package_height_cm": raw.get("package_height_cm"),
                    "packed_weight_kg": raw.get("packed_weight_kg"),
                    "scheme_url": _text_or_none(raw.get("scheme_url")),
                    "room": _text_or_none(raw.get("room")),
                    "materials": _text_or_none(raw.get("materials")),
                    "availability": _text_or_none(raw.get("availability")),
                    "country_brand": _text_or_none(raw.get("country_brand")),
                    "production_country": _text_or_none(raw.get("production_country")),
                    "tags_json": _json_loads_or(raw.get("tags_json"), []),
                    "images_json": _json_loads_or(raw.get("images_json"), []),
                    "related_json": _json_loads_or(raw.get("related_json"), []),
                    "extra_json": extra,
                    "asset_status": _text_or_none(raw.get("asset_status")),
                    "asset_format": _text_or_none(raw.get("asset_format")),
                    "asset_local_path": _text_or_none(raw.get("asset_local_path")),
                    "preview_local_path": _text_or_none(raw.get("preview_local_path")),
                    "asset_source_url": _text_or_none(raw.get("asset_source_url")),
                    "model_probe_has_fbx": _bool_to_int(extra.get("model_probe_has_fbx")),
                    "model_probe_checked_at": _text_or_none(extra.get("model_probe_checked_at")),
                    "model_probe_error": _text_or_none(extra.get("model_probe_error")),
                    "has_model_url": 1 if _text_or_none(raw.get("model_download_url")) else 0,
                    "has_asset_local": 1 if _text_or_none(raw.get("asset_local_path")) else 0,
                    "raw_json": raw,
                    "_source_kind": "db",
                    "_source_file": str(db_path.resolve()),
                    "_source_table": "supplier_product",
                    "_source_rank": _source_rank(db_path, "supplier_product"),
                }
            )
            candidates.append(candidate)
        return candidates


def _load_supplier_item_candidates(db_path: Path) -> list[dict[str, Any]]:
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("SELECT * FROM supplier_item").fetchall()

    candidates: list[dict[str, Any]] = []
    for row in rows:
        raw = dict(row)
        unique_key = _text_or_none(raw.get("unique_key"))
        if not unique_key:
            continue

        dimensions = _dimensions_from_json(raw.get("dimensions_json"))
        raw_payload = _json_loads_or(raw.get("raw_json"), {})
        if not isinstance(raw_payload, dict):
            raw_payload = {}
        extra = _json_loads_or(raw.get("extra_json"), {})
        if not isinstance(extra, dict):
            extra = {}
        payload_extra = raw_payload.get("extra")
        if isinstance(payload_extra, dict):
            merged_extra = {**payload_extra, **extra}
        else:
            merged_extra = dict(extra)

        materials_value = _json_loads_or(raw.get("materials_json"), raw.get("materials_json"))
        materials_text = None
        if isinstance(materials_value, str):
            materials_text = _text_or_none(materials_value)
        elif _has_value(materials_value):
            materials_text = _json_dumps(materials_value)

        tags_value = _json_loads_or(raw.get("tags_json"), raw_payload.get("tags", []))
        related_value = _json_loads_or(raw.get("related_json"), raw_payload.get("related", []))
        images_value = raw_payload.get("images", [])

        candidate = _candidate_base()
        candidate.update(
            {
                "unique_key": unique_key,
                "source_site": _text_or_none(raw.get("source_site") or raw_payload.get("source_site")),
                "source_url": _text_or_none(raw.get("source_url") or raw_payload.get("source_url")),
                "parsed_at": _text_or_none(raw.get("parsed_at") or raw_payload.get("parsed_at")),
                "external_id": _text_or_none(raw.get("external_id") or raw_payload.get("external_id")),
                "category_raw": _text_or_none(raw.get("category_raw") or raw_payload.get("category_raw")),
                "category_norm": _text_or_none(raw.get("category_norm") or raw_payload.get("category_norm")),
                "title": _text_or_none(raw.get("title") or raw_payload.get("title")),
                "brand": _text_or_none(raw.get("brand") or raw_payload.get("brand")),
                "collection": _text_or_none(raw.get("collection") or raw_payload.get("collection")),
                "product_url": _text_or_none(raw.get("product_url") or raw_payload.get("product_url")),
                "model_link_type": _text_or_none(raw.get("model_link_type") or raw_payload.get("model_link_type")),
                "model_page_url": _text_or_none(raw.get("model_page_url") or raw_payload.get("model_page_url")),
                "model_download_url": _text_or_none(raw.get("model_download_url") or raw_payload.get("model_download_url")),
                "model_download_landing_url": _text_or_none(raw.get("model_download_landing_url") or raw_payload.get("model_download_landing_url")),
                "model_vendor_url": _text_or_none(raw_payload.get("model_vendor_url")),
                "model_extraction_method": _text_or_none(raw_payload.get("model_extraction_method")),
                "model_download_filename": _text_or_none(raw_payload.get("model_download_filename")),
                "model_format": _text_or_none(raw.get("model_format") or raw_payload.get("model_format")),
                "price_value": raw.get("price_value") if raw.get("price_value") is not None else raw_payload.get("price_value"),
                "price_currency": _text_or_none(raw.get("price_currency") or raw_payload.get("price_currency")),
                "old_price_value": raw.get("old_price_value") if raw.get("old_price_value") is not None else raw_payload.get("old_price_value"),
                "style": _text_or_none(raw.get("style") or raw_payload.get("style")),
                "color": _text_or_none(raw.get("color") or raw_payload.get("color")),
                "description": _text_or_none(raw_payload.get("description")),
                "width_cm": dimensions.get("width_cm"),
                "depth_cm": dimensions.get("depth_cm"),
                "height_cm": dimensions.get("height_cm"),
                "weight_kg": dimensions.get("weight_kg"),
                "volume_m3": dimensions.get("volume_m3"),
                "package_width_cm": dimensions.get("package_width_cm"),
                "package_depth_cm": dimensions.get("package_depth_cm"),
                "package_height_cm": dimensions.get("package_height_cm"),
                "packed_weight_kg": dimensions.get("packed_weight_kg"),
                "scheme_url": _text_or_none(raw_payload.get("scheme_url")),
                "room": _text_or_none(raw.get("room") or raw_payload.get("room")),
                "materials": materials_text or _text_or_none(raw_payload.get("materials")),
                "availability": _text_or_none(raw.get("availability") or raw_payload.get("availability")),
                "country_brand": _text_or_none(raw_payload.get("country_brand")),
                "production_country": _text_or_none(raw_payload.get("production_country")),
                "tags_json": tags_value if isinstance(tags_value, list) else [],
                "images_json": images_value if isinstance(images_value, list) else [],
                "related_json": related_value if isinstance(related_value, list) else [],
                "extra_json": merged_extra,
                "asset_status": _text_or_none(raw.get("asset_status") or raw_payload.get("asset_status")),
                "asset_format": _text_or_none(raw.get("asset_format") or raw_payload.get("asset_format")),
                "asset_local_path": _text_or_none(raw.get("asset_local_path") or raw_payload.get("asset_local_path")),
                "preview_local_path": _text_or_none(raw.get("preview_local_path") or raw_payload.get("preview_local_path")),
                "asset_source_url": _text_or_none(raw_payload.get("asset_source_url")),
                "model_probe_has_fbx": _bool_to_int(merged_extra.get("model_probe_has_fbx")),
                "model_probe_checked_at": _text_or_none(merged_extra.get("model_probe_checked_at")),
                "model_probe_error": _text_or_none(merged_extra.get("model_probe_error")),
                "has_model_url": 1 if _text_or_none(raw.get("model_download_url") or raw_payload.get("model_download_url")) else 0,
                "has_asset_local": 1 if _text_or_none(raw.get("asset_local_path") or raw_payload.get("asset_local_path")) else 0,
                "raw_json": raw_payload or raw,
                "_source_kind": "db",
                "_source_file": str(db_path.resolve()),
                "_source_table": "supplier_item",
                "_source_rank": _source_rank(db_path, "supplier_item"),
            }
        )
        candidates.append(candidate)
    return candidates


def _collect_db_candidates(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    main_unique_keys: set[str] = set()

    main_db = root / "suppliers.db"
    if main_db.exists():
        with sqlite3.connect(main_db) as con:
            main_unique_keys = {
                str(row[0])
                for row in con.execute(
                    "SELECT unique_key FROM supplier_product WHERE unique_key IS NOT NULL AND TRIM(unique_key) != ''"
                )
            }

    for db_path in sorted(root.rglob("*.db")):
        with sqlite3.connect(db_path) as con:
            table_names = sorted(
                row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            )
            tables = [{"name": name, "row_count": _sqlite_table_rows(con, name)} for name in table_names]
            has_supplier_product = "supplier_product" in table_names
            has_supplier_item = "supplier_item" in table_names

            unique_keys: set[str] = set()
            if has_supplier_product:
                unique_keys = {
                    str(row[0])
                    for row in con.execute(
                        "SELECT unique_key FROM supplier_product WHERE unique_key IS NOT NULL AND TRIM(unique_key) != ''"
                    )
                }
            elif has_supplier_item:
                unique_keys = {
                    str(row[0])
                    for row in con.execute(
                        "SELECT unique_key FROM supplier_item WHERE unique_key IS NOT NULL AND TRIM(unique_key) != ''"
                    )
                }

        inventory.append(
            {
                "path": _relative(db_path, root),
                "size_bytes": db_path.stat().st_size,
                "tables": tables,
                "has_supplier_product": has_supplier_product,
                "has_supplier_item": has_supplier_item,
                "unique_key_count": len(unique_keys),
                "unique_keys_missing_from_suppliers_db": len(unique_keys - main_unique_keys) if db_path != main_db else 0,
            }
        )

        if has_supplier_product:
            candidates.extend(_load_supplier_product_candidates(db_path))
        if has_supplier_item:
            candidates.extend(_load_supplier_item_candidates(db_path))

    return candidates, inventory


def _classify_json_payload(path: Path, payload: Any) -> tuple[str, str | None, int | None, int]:
    if isinstance(payload, dict):
        schema = _text_or_none(payload.get("schema"))
        items = payload.get("items")
        if isinstance(items, list):
            if schema and "mesh_catalog" in schema:
                return "mesh_catalog_snapshot", schema, len(items), sum(1 for item in items if isinstance(item, dict) and item.get("unique_key"))
            return "catalog_items", schema, len(items), sum(1 for item in items if isinstance(item, dict) and item.get("unique_key"))
        if schema and "mesh_catalog" in schema:
            return "mesh_catalog_snapshot", schema, None, 0
        return "dict_report", schema, None, 0
    if isinstance(payload, list):
        return "plain_list", None, len(payload), sum(1 for item in payload if isinstance(item, dict) and item.get("unique_key"))
    return "other", None, None, 0


def _inventory_json_files(root: Path, db_unique_keys: set[str]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    inventory: list[dict[str, Any]] = []
    metadata_counts: dict[str, int] = defaultdict(int)

    for path in sorted(list(root.rglob("*.json")) + list(root.rglob("*.jsonl"))):
        if path.name.endswith(".metadata.json"):
            metadata_counts[_top_bucket(path, root)] += 1
            continue

        kind = "jsonl" if path.suffix == ".jsonl" else "json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            inventory.append(
                {
                    "path": _relative(path, root),
                    "size_bytes": path.stat().st_size,
                    "kind": kind,
                    "schema": None,
                    "item_count": None,
                    "unique_key_count": 0,
                    "product_like": False,
                    "product_like_unique_keys_missing_from_db_union": 0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        shape_kind, schema, item_count, unique_key_count = _classify_json_payload(path, payload)
        product_like = shape_kind in {"catalog_items", "plain_list"} and (schema is None or "mesh_catalog" not in (schema or ""))
        unique_keys_missing_from_db = 0
        missing_from_db = 0
        if shape_kind in {"catalog_items", "plain_list", "mesh_catalog_snapshot"}:
            if isinstance(payload, dict):
                items = payload.get("items", [])
            else:
                items = payload
            keys = {
                str(item.get("unique_key"))
                for item in items
                if isinstance(item, dict) and item.get("unique_key")
            }
            unique_keys_missing_from_db = len(keys - db_unique_keys)
            if product_like:
                missing_from_db = unique_keys_missing_from_db

        inventory.append(
            {
                "path": _relative(path, root),
                "size_bytes": path.stat().st_size,
                "kind": shape_kind,
                "schema": schema,
                "item_count": item_count,
                "unique_key_count": unique_key_count,
                "unique_keys_missing_from_db_union": unique_keys_missing_from_db,
                "product_like": product_like,
                "product_like_unique_keys_missing_from_db_union": missing_from_db,
                "error": None,
            }
        )

    return inventory, dict(sorted(metadata_counts.items()))


def _merge_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        unique_key = candidate.get("unique_key")
        if unique_key:
            grouped[str(unique_key)].append(candidate)

    merged_rows: list[dict[str, Any]] = []
    for unique_key, group in grouped.items():
        ordered = sorted(group, key=_candidate_score, reverse=True)
        winner = ordered[0]
        merged = _candidate_base()
        merged["unique_key"] = unique_key

        for field in BASE_FIELDS:
            if field in {"tags_json", "images_json", "related_json", "extra_json", "raw_json"}:
                continue
            for candidate in ordered:
                value = candidate.get(field)
                if _has_value(value):
                    merged[field] = value
                    break

        merged["tags_json"] = _merge_lists([candidate.get("tags_json") for candidate in ordered])
        merged["images_json"] = _merge_lists([candidate.get("images_json") for candidate in ordered])
        merged["related_json"] = _merge_lists([candidate.get("related_json") for candidate in ordered])

        extra: dict[str, Any] = {}
        for candidate in reversed(ordered):
            value = candidate.get("extra_json")
            if isinstance(value, dict):
                extra.update(value)
        merged["extra_json"] = extra
        merged["raw_json"] = winner.get("raw_json")

        merged["model_probe_has_fbx"] = _bool_to_int(extra.get("model_probe_has_fbx"))
        merged["model_probe_checked_at"] = _text_or_none(extra.get("model_probe_checked_at"))
        merged["model_probe_error"] = _text_or_none(extra.get("model_probe_error"))
        merged["has_model_url"] = 1 if _has_value(merged.get("model_download_url")) else 0
        merged["has_asset_local"] = 1 if _has_value(merged.get("asset_local_path")) else 0

        merged["preferred_source_kind"] = winner.get("_source_kind")
        merged["preferred_source_file"] = winner.get("_source_file")
        merged["preferred_source_table"] = winner.get("_source_table")
        merged["preferred_source_rank"] = winner.get("_source_rank")
        merged["merged_from_count"] = len(group)
        merged["merged_from_sources_json"] = [
            {
                "source_kind": candidate.get("_source_kind"),
                "source_file": candidate.get("_source_file"),
                "source_table": candidate.get("_source_table"),
                "source_rank": candidate.get("_source_rank"),
            }
            for candidate in ordered
        ]

        merged_rows.append(merged)

    merged_rows.sort(
        key=lambda row: (
            str(row.get("source_site") or ""),
            str(row.get("title") or ""),
            str(row.get("unique_key") or ""),
        )
    )
    return merged_rows


def _write_sqlite(out_db: Path, rows: list[dict[str, Any]]) -> None:
    out_db.parent.mkdir(parents=True, exist_ok=True)
    if out_db.exists():
        out_db.unlink()

    con = sqlite3.connect(out_db)
    try:
        columns_sql = ",\n            ".join(f"{name} {sql_type}" for name, sql_type in TABLE_COLUMNS)
        con.executescript(
            f"""
            CREATE TABLE supplier_catalog_one_table (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                {columns_sql}
            );

            CREATE UNIQUE INDEX idx_supplier_catalog_one_table_unique_key ON supplier_catalog_one_table(unique_key);
            CREATE INDEX idx_supplier_catalog_one_table_site ON supplier_catalog_one_table(source_site);
            CREATE INDEX idx_supplier_catalog_one_table_category_norm ON supplier_catalog_one_table(category_norm);
            CREATE INDEX idx_supplier_catalog_one_table_model_url ON supplier_catalog_one_table(model_download_url);
            CREATE INDEX idx_supplier_catalog_one_table_asset_status ON supplier_catalog_one_table(asset_status);

            CREATE VIEW supplier_catalog_accessible_items AS
            SELECT *
            FROM supplier_catalog_one_table
            WHERE COALESCE(has_model_url, 0) = 1 OR COALESCE(has_asset_local, 0) = 1;

            CREATE VIEW supplier_catalog_local_asset_items AS
            SELECT *
            FROM supplier_catalog_one_table
            WHERE COALESCE(has_asset_local, 0) = 1;
            """
        )

        insert_columns = [name for name, _ in TABLE_COLUMNS]
        placeholders = ", ".join("?" for _ in insert_columns)
        sql = f"INSERT INTO supplier_catalog_one_table ({', '.join(insert_columns)}) VALUES ({placeholders})"

        payloads: list[tuple[Any, ...]] = []
        for row in rows:
            values: list[Any] = []
            for column in insert_columns:
                value = row.get(column)
                if column in JSON_FIELDS:
                    value = _json_dumps(value)
                values.append(value)
            payloads.append(tuple(values))

        con.executemany(sql, payloads)
        con.commit()
    finally:
        con.close()


def _write_csv(out_csv: Path, rows: list[dict[str, Any]]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    header = [name for name, _ in TABLE_COLUMNS]
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            values = []
            for column in header:
                value = row.get(column)
                if column in JSON_FIELDS:
                    value = _json_dumps(value)
                values.append(value)
            writer.writerow(values)


def _write_json(out_json: Path, rows: list[dict[str, Any]], meta: dict[str, Any]) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(
            {
                "schema": SCHEMA_VERSION,
                "meta": meta,
                "items": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _build_report(
    root: Path,
    db_inventory: list[dict[str, Any]],
    json_inventory: list[dict[str, Any]],
    metadata_counts: dict[str, int],
    main_unique_keys: set[str],
    merged_rows: list[dict[str, Any]],
    out_db: Path,
    out_csv: Path | None,
    out_json: Path | None,
) -> dict[str, Any]:
    merged_by_site = Counter(str(row.get("source_site") or "<NULL>") for row in merged_rows)
    accessible_rows = [row for row in merged_rows if row.get("has_model_url") == 1 or row.get("has_asset_local") == 1]
    local_asset_rows = [row for row in merged_rows if row.get("has_asset_local") == 1]
    accessible_by_site = Counter(str(row.get("source_site") or "<NULL>") for row in accessible_rows)
    local_asset_by_site = Counter(str(row.get("source_site") or "<NULL>") for row in local_asset_rows)
    added_rows = [row for row in merged_rows if Path(str(row.get("preferred_source_file") or "")).name != "suppliers.db"]
    added_by_source = Counter(Path(str(row.get("preferred_source_file") or "")).name for row in added_rows)

    product_like_json_missing = sum(
        int(item.get("product_like_unique_keys_missing_from_db_union") or 0)
        for item in json_inventory
        if item.get("product_like")
    )
    mesh_snapshot_missing = sum(
        int(item.get("unique_keys_missing_from_db_union") or 0)
        for item in json_inventory
        if item.get("kind") == "mesh_catalog_snapshot"
    )

    return {
        "schema": SCHEMA_VERSION,
        "scanned_at": _now_utc_iso(),
        "root": str(root.resolve()),
        "file_counts": {
            "db_files": len(db_inventory),
            "json_like_non_metadata_files": len(json_inventory),
            "metadata_json_files": sum(metadata_counts.values()),
        },
        "db_inventory": db_inventory,
        "json_inventory": json_inventory,
        "metadata_json_buckets": metadata_counts,
        "canonical_recommendation": {
            "primary_product_table": "suppliers.db::supplier_product",
            "reason": "Это самый полный operational catalog. Остальные product DB в основном старые слепки, subset-выгрузки или asset-run базы.",
            "merge_needed": True,
            "why_merge": "В suppliers.db отсутствуют idealbeds_yadisk и несколько хвостов из старых product-слепков.",
        },
        "consolidated_table": {
            "table_name": "supplier_catalog_one_table",
            "accessible_view": "supplier_catalog_accessible_items",
            "local_asset_view": "supplier_catalog_local_asset_items",
            "out_db": str(out_db.resolve()),
            "out_csv": str(out_csv.resolve()) if out_csv else None,
            "out_json": str(out_json.resolve()) if out_json else None,
            "row_count": len(merged_rows),
            "accessible_row_count": len(accessible_rows),
            "local_asset_row_count": len(local_asset_rows),
            "suppliers_db_unique_keys": len(main_unique_keys),
            "rows_added_beyond_suppliers_db": len(merged_rows) - len(main_unique_keys),
            "rows_preferred_from_non_suppliers_db": len(added_rows),
            "by_site": dict(sorted(merged_by_site.items(), key=lambda kv: (-kv[1], kv[0]))),
            "accessible_by_site": dict(sorted(accessible_by_site.items(), key=lambda kv: (-kv[1], kv[0]))),
            "local_asset_by_site": dict(sorted(local_asset_by_site.items(), key=lambda kv: (-kv[1], kv[0]))),
            "added_rows_by_source_file": dict(sorted(added_by_source.items(), key=lambda kv: (-kv[1], kv[0]))),
        },
        "json_findings": {
            "product_like_unique_keys_missing_from_db_union": product_like_json_missing,
            "mesh_snapshot_unique_keys_missing_from_db_union": mesh_snapshot_missing,
            "note": "Product-like top-level JSON в этом каталоге не добавляют новых карточек поверх объединения product DB. Отдельно есть mesh snapshot JSON, его в product-table смешивать не нужно.",
        },
    }


def _report_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    counts = report["file_counts"]
    consolidated = report["consolidated_table"]

    lines.append("# Supplier Storage Summary")
    lines.append("")
    lines.append(f"- scanned_at: {report['scanned_at']}")
    lines.append(f"- db_files: {counts['db_files']}")
    lines.append(f"- json_like_non_metadata_files: {counts['json_like_non_metadata_files']}")
    lines.append(f"- metadata_json_files: {counts['metadata_json_files']}")
    lines.append("")
    lines.append("## Canonical Source")
    lines.append("")
    lines.append(f"- primary_product_table: `{report['canonical_recommendation']['primary_product_table']}`")
    lines.append(f"- merge_needed: `{report['canonical_recommendation']['merge_needed']}`")
    lines.append(f"- reason: {report['canonical_recommendation']['reason']}")
    lines.append(f"- why_merge: {report['canonical_recommendation']['why_merge']}")
    lines.append("")
    lines.append("## Consolidated Table")
    lines.append("")
    lines.append(f"- table_name: `{consolidated['table_name']}`")
    lines.append(f"- accessible_view: `{consolidated['accessible_view']}`")
    lines.append(f"- local_asset_view: `{consolidated['local_asset_view']}`")
    lines.append(f"- out_db: `{consolidated['out_db']}`")
    if consolidated.get("out_csv"):
        lines.append(f"- out_csv: `{consolidated['out_csv']}`")
    if consolidated.get("out_json"):
        lines.append(f"- out_json: `{consolidated['out_json']}`")
    lines.append(f"- row_count: {consolidated['row_count']}")
    lines.append(f"- accessible_row_count: {consolidated['accessible_row_count']}")
    lines.append(f"- local_asset_row_count: {consolidated['local_asset_row_count']}")
    lines.append(f"- suppliers_db_unique_keys: {consolidated['suppliers_db_unique_keys']}")
    lines.append(f"- rows_added_beyond_suppliers_db: {consolidated['rows_added_beyond_suppliers_db']}")
    lines.append("")
    lines.append("## By Site")
    lines.append("")
    for site, count in consolidated["by_site"].items():
        lines.append(f"- {site}: {count}")
    lines.append("")
    lines.append("## Accessible By Site")
    lines.append("")
    for site, count in consolidated["accessible_by_site"].items():
        lines.append(f"- {site}: {count}")
    lines.append("")
    lines.append("## Local Asset By Site")
    lines.append("")
    for site, count in consolidated["local_asset_by_site"].items():
        lines.append(f"- {site}: {count}")
    lines.append("")
    lines.append("## Added Beyond suppliers.db")
    lines.append("")
    for source_file, count in consolidated["added_rows_by_source_file"].items():
        lines.append(f"- {source_file}: {count}")
    lines.append("")
    lines.append("## Metadata JSON Buckets")
    lines.append("")
    for bucket, count in report["metadata_json_buckets"].items():
        lines.append(f"- {bucket}: {count}")
    lines.append("")
    lines.append("## JSON Findings")
    lines.append("")
    lines.append(
        f"- product_like_unique_keys_missing_from_db_union: {report['json_findings']['product_like_unique_keys_missing_from_db_union']}"
    )
    lines.append(
        f"- mesh_snapshot_unique_keys_missing_from_db_union: {report['json_findings']['mesh_snapshot_unique_keys_missing_from_db_union']}"
    )
    lines.append(f"- note: {report['json_findings']['note']}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Inventory supplier storage and build one consolidated supplier table.")
    ap.add_argument("--root", default="data/sourse/suppliers", help="Root directory with supplier db/json files.")
    ap.add_argument("--out-db", default=None, help="SQLite DB with the consolidated supplier_catalog_one_table.")
    ap.add_argument("--out-csv", default=None, help="Optional CSV export of the consolidated table.")
    ap.add_argument("--out-json", default=None, help="Optional JSON export of the consolidated table.")
    ap.add_argument("--out-report-json", default=None, help="Inventory report JSON.")
    ap.add_argument("--out-report-md", default=None, help="Inventory report Markdown.")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    out_db = Path(args.out_db).expanduser().resolve() if args.out_db else root / "supplier_catalog_one_table.db"
    out_csv = Path(args.out_csv).expanduser().resolve() if args.out_csv else root / "supplier_catalog_one_table.csv"
    out_json = Path(args.out_json).expanduser().resolve() if args.out_json else root / "supplier_catalog_one_table.json"
    out_report_json = (
        Path(args.out_report_json).expanduser().resolve() if args.out_report_json else root / "supplier_storage_inventory.json"
    )
    out_report_md = (
        Path(args.out_report_md).expanduser().resolve() if args.out_report_md else root / "supplier_storage_inventory.md"
    )

    db_candidates, db_inventory = _collect_db_candidates(root)
    main_unique_keys = {
        candidate["unique_key"]
        for candidate in db_candidates
        if candidate.get("_source_file") == str((root / "suppliers.db").resolve()) and candidate.get("_source_table") == "supplier_product"
    }
    db_union_unique_keys = {candidate["unique_key"] for candidate in db_candidates if candidate.get("unique_key")}
    json_inventory, metadata_counts = _inventory_json_files(root, db_union_unique_keys)
    merged_rows = _merge_candidates(db_candidates)

    _write_sqlite(out_db, merged_rows)
    _write_csv(out_csv, merged_rows)
    _write_json(
        out_json,
        merged_rows,
        {
            "generated_at": _now_utc_iso(),
            "root": str(root.resolve()),
            "row_count": len(merged_rows),
            "db_sources": [item["path"] for item in db_inventory if item["has_supplier_product"] or item["has_supplier_item"]],
        },
    )

    report = _build_report(
        root=root,
        db_inventory=db_inventory,
        json_inventory=json_inventory,
        metadata_counts=metadata_counts,
        main_unique_keys=main_unique_keys,
        merged_rows=merged_rows,
        out_db=out_db,
        out_csv=out_csv,
        out_json=out_json,
    )
    out_report_json.parent.mkdir(parents=True, exist_ok=True)
    out_report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    out_report_md.parent.mkdir(parents=True, exist_ok=True)
    out_report_md.write_text(_report_markdown(report), encoding="utf-8")

    print(f"root = {root}")
    print(f"db_candidates = {len(db_candidates)}")
    print(f"merged_rows = {len(merged_rows)}")
    print(f"out_db = {out_db}")
    print(f"out_csv = {out_csv}")
    print(f"out_json = {out_json}")
    print(f"out_report_json = {out_report_json}")
    print(f"out_report_md = {out_report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
