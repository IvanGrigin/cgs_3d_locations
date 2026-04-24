#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a consolidated supplier-model analytics bundle from suppliers.db.

Outputs:
  - summary.json
  - summary.md
  - supplier_models_with_urls.json

The report focuses on model-bearing catalog rows and also includes full DB schema
and field coverage so the catalog state is easy to inspect in one place.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.suppliers.export_catalog_json import build_catalog_export
from src.suppliers.utils import now_utc_iso


PRODUCT_FIELDS_FOR_COVERAGE = [
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
    "style",
    "color",
    "description",
    "width_cm",
    "depth_cm",
    "height_cm",
    "weight_kg",
    "volume_m3",
    "package_width_cm",
    "package_depth_cm",
    "package_height_cm",
    "packed_weight_kg",
    "scheme_url",
    "room",
    "materials",
    "availability",
    "country_brand",
    "production_country",
    "raw_html",
]


def _nonempty_text(value: Any) -> bool:
    return bool(str(value or "").strip())


def _load_table_schema(con: sqlite3.Connection, table: str) -> list[dict[str, Any]]:
    rows = con.execute(f"PRAGMA table_info({table})").fetchall()
    return [
        {
            "cid": row[0],
            "name": row[1],
            "type": row[2],
            "notnull": bool(row[3]),
            "default": row[4],
            "pk": bool(row[5]),
        }
        for row in rows
    ]


def _count_rows(con: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> int:
    return int(con.execute(sql, params).fetchone()[0])


def _category_counts(
    con: sqlite3.Connection,
    field: str,
    where_sql: str = "1=1",
    params: tuple[Any, ...] = (),
) -> list[dict[str, Any]]:
    sql = f"""
        SELECT COALESCE(NULLIF(TRIM({field}), ''), '<NULL>') AS category, COUNT(*) AS n
        FROM supplier_product
        WHERE {where_sql}
        GROUP BY category
        ORDER BY n DESC, category ASC
    """
    rows = con.execute(sql, params).fetchall()
    return [{"category": row[0], "count": int(row[1])} for row in rows]


def _site_summary(con: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = con.execute(
        """
        SELECT
            source_site,
            COUNT(*) AS total_rows,
            SUM(CASE WHEN model_download_url IS NOT NULL AND TRIM(model_download_url) != '' THEN 1 ELSE 0 END) AS with_model_url,
            COUNT(DISTINCT CASE WHEN model_download_url IS NOT NULL AND TRIM(model_download_url) != '' THEN model_download_url END) AS distinct_model_urls,
            SUM(CASE WHEN price_value IS NOT NULL THEN 1 ELSE 0 END) AS with_price,
            SUM(CASE WHEN width_cm IS NOT NULL AND depth_cm IS NOT NULL AND height_cm IS NOT NULL THEN 1 ELSE 0 END) AS with_full_dims,
            SUM(CASE WHEN category_norm IS NOT NULL AND TRIM(category_norm) != '' THEN 1 ELSE 0 END) AS with_category_norm
        FROM supplier_product
        GROUP BY source_site
        ORDER BY total_rows DESC, source_site ASC
        """
    ).fetchall()

    out: list[dict[str, Any]] = []
    for row in rows:
        total_rows = int(row[1])
        with_model = int(row[2] or 0)
        out.append(
            {
                "source_site": row[0],
                "total_rows": total_rows,
                "with_model_url": with_model,
                "distinct_model_urls": int(row[3] or 0),
                "with_price": int(row[4] or 0),
                "with_full_dims": int(row[5] or 0),
                "with_category_norm": int(row[6] or 0),
                "share_of_total_rows": round(total_rows / max(1, _count_rows(con, "SELECT COUNT(*) FROM supplier_product")), 4),
                "model_row_rate": round(with_model / max(1, total_rows), 4),
            }
        )
    return out


def _coverage(con: sqlite3.Connection, where_sql: str) -> dict[str, dict[str, Any]]:
    total = _count_rows(con, f"SELECT COUNT(*) FROM supplier_product WHERE {where_sql}")
    out: dict[str, dict[str, Any]] = {}
    for field in PRODUCT_FIELDS_FOR_COVERAGE:
        sql = f"""
            SELECT COUNT(*)
            FROM supplier_product
            WHERE {where_sql}
              AND {field} IS NOT NULL
              AND TRIM(CAST({field} AS TEXT)) != ''
        """
        non_null = _count_rows(con, sql)
        out[field] = {
            "non_null_count": non_null,
            "coverage_rate": round(non_null / max(1, total), 4),
        }
    return out


def _model_format_counts(con: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = con.execute(
        """
        SELECT COALESCE(NULLIF(TRIM(model_format), ''), '<NULL>') AS fmt, COUNT(*) AS n
        FROM supplier_product
        WHERE model_download_url IS NOT NULL AND TRIM(model_download_url) != ''
        GROUP BY fmt
        ORDER BY n DESC, fmt ASC
        """
    ).fetchall()
    return [{"model_format": row[0], "count": int(row[1])} for row in rows]


def _model_format_by_site(con: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = con.execute(
        """
        SELECT
            source_site,
            COALESCE(NULLIF(TRIM(model_format), ''), '<NULL>') AS fmt,
            COUNT(*) AS n
        FROM supplier_product
        WHERE model_download_url IS NOT NULL AND TRIM(model_download_url) != ''
        GROUP BY source_site, fmt
        ORDER BY source_site ASC, n DESC, fmt ASC
        """
    ).fetchall()
    return [
        {
            "source_site": row[0],
            "model_format": row[1],
            "count": int(row[2]),
        }
        for row in rows
    ]


def _asset_summary(con: sqlite3.Connection) -> dict[str, Any]:
    status_rows = con.execute(
        """
        SELECT COALESCE(NULLIF(TRIM(asset_status), ''), '<NULL>') AS s, COUNT(*)
        FROM supplier_asset
        GROUP BY s
        ORDER BY COUNT(*) DESC, s ASC
        """
    ).fetchall()
    format_rows = con.execute(
        """
        SELECT COALESCE(NULLIF(TRIM(asset_format), ''), '<NULL>') AS s, COUNT(*)
        FROM supplier_asset
        GROUP BY s
        ORDER BY COUNT(*) DESC, s ASC
        """
    ).fetchall()
    return {
        "total_rows": _count_rows(con, "SELECT COUNT(*) FROM supplier_asset"),
        "by_status": [{"asset_status": row[0], "count": int(row[1])} for row in status_rows],
        "by_format": [{"asset_format": row[0], "count": int(row[1])} for row in format_rows],
    }


def _category_by_site_with_model(con: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = con.execute(
        """
        SELECT
            source_site,
            COALESCE(NULLIF(TRIM(category_norm), ''), '<NULL>') AS category_norm,
            COUNT(*) AS n
        FROM supplier_product
        WHERE model_download_url IS NOT NULL AND TRIM(model_download_url) != ''
        GROUP BY source_site, category_norm
        ORDER BY source_site ASC, n DESC, category_norm ASC
        """
    ).fetchall()
    return [
        {
            "source_site": row[0],
            "category_norm": row[1],
            "count": int(row[2]),
        }
        for row in rows
    ]


def _sancos_fbx_summary(con: sqlite3.Connection) -> dict[str, Any]:
    rows = con.execute(
        """
        SELECT model_download_url, extra_json
        FROM supplier_product
        WHERE source_site = 'sancos'
          AND model_download_url IS NOT NULL
          AND TRIM(model_download_url) != ''
        """
    ).fetchall()
    distinct: dict[str, dict[str, Any]] = {}
    sku_true = 0
    sku_false = 0
    sku_error = 0
    for model_download_url, extra_json in rows:
        extra = json.loads(extra_json or "{}")
        if extra.get("model_probe_has_fbx") is True:
            sku_true += 1
        elif extra.get("model_probe_has_fbx") is False:
            sku_false += 1
        if extra.get("model_probe_error"):
            sku_error += 1
        if extra.get("model_probe_checked_at"):
            distinct[str(model_download_url)] = extra

    return {
        "distinct_checked_urls": len(distinct),
        "distinct_has_fbx_true": sum(1 for item in distinct.values() if item.get("model_probe_has_fbx") is True),
        "distinct_has_fbx_false": sum(1 for item in distinct.values() if item.get("model_probe_has_fbx") is False),
        "distinct_error_urls": sum(1 for item in distinct.values() if item.get("model_probe_error")),
        "sku_has_fbx_true": sku_true,
        "sku_has_fbx_false": sku_false,
        "sku_error": sku_error,
    }


def _build_summary(con: sqlite3.Connection, db_path: Path, out_dir: Path) -> dict[str, Any]:
    summary = {
        "generated_at": now_utc_iso(),
        "db_path": str(db_path),
        "out_dir": str(out_dir),
        "tables": {
            table: _load_table_schema(con, table)
            for table in ["supplier_product", "supplier_asset", "supplier_download", "supplier_fetch_log"]
        },
        "counts": {
            "supplier_product_total": _count_rows(con, "SELECT COUNT(*) FROM supplier_product"),
            "supplier_product_with_model_url": _count_rows(
                con,
                "SELECT COUNT(*) FROM supplier_product WHERE model_download_url IS NOT NULL AND TRIM(model_download_url) != ''",
            ),
            "supplier_product_distinct_model_urls": _count_rows(
                con,
                "SELECT COUNT(DISTINCT model_download_url) FROM supplier_product WHERE model_download_url IS NOT NULL AND TRIM(model_download_url) != ''",
            ),
            "supplier_asset_total": _count_rows(con, "SELECT COUNT(*) FROM supplier_asset"),
            "supplier_download_total": _count_rows(con, "SELECT COUNT(*) FROM supplier_download"),
            "supplier_fetch_log_total": _count_rows(con, "SELECT COUNT(*) FROM supplier_fetch_log"),
        },
        "by_site": _site_summary(con),
        "category_norm_all": _category_counts(con, "category_norm"),
        "category_norm_with_model": _category_counts(
            con,
            "category_norm",
            "model_download_url IS NOT NULL AND TRIM(model_download_url) != ''",
        ),
        "category_raw_all": _category_counts(con, "category_raw"),
        "category_raw_with_model": _category_counts(
            con,
            "category_raw",
            "model_download_url IS NOT NULL AND TRIM(model_download_url) != ''",
        ),
        "field_coverage_all": _coverage(con, "1=1"),
        "field_coverage_with_model": _coverage(
            con,
            "model_download_url IS NOT NULL AND TRIM(model_download_url) != ''",
        ),
        "model_format_with_model": _model_format_counts(con),
        "model_format_by_site_with_model": _model_format_by_site(con),
        "category_by_site_with_model": _category_by_site_with_model(con),
        "asset_summary": _asset_summary(con),
        "sancos_fbx_summary": _sancos_fbx_summary(con),
    }
    return summary


def _top_lines(rows: list[dict[str, Any]], key_name: str, value_name: str = "count", limit: int = 10) -> list[str]:
    return [f"- `{row[key_name]}`: {row[value_name]}" for row in rows[:limit]]


def _to_markdown(summary: dict[str, Any]) -> str:
    counts = summary["counts"]
    lines: list[str] = []
    lines.append("# Supplier Catalog Analytics")
    lines.append("")
    lines.append(f"- Generated at: `{summary['generated_at']}`")
    lines.append(f"- DB path: `{summary['db_path']}`")
    lines.append("")
    lines.append("## Core Counts")
    lines.append("")
    lines.append(f"- `supplier_product_total`: {counts['supplier_product_total']}")
    lines.append(f"- `supplier_product_with_model_url`: {counts['supplier_product_with_model_url']}")
    lines.append(f"- `supplier_product_distinct_model_urls`: {counts['supplier_product_distinct_model_urls']}")
    lines.append(f"- `supplier_asset_total`: {counts['supplier_asset_total']}")
    lines.append(f"- `supplier_download_total`: {counts['supplier_download_total']}")
    lines.append(f"- `supplier_fetch_log_total`: {counts['supplier_fetch_log_total']}")
    lines.append("")
    lines.append("## By Site")
    lines.append("")
    lines.append("| site | total_rows | with_model_url | distinct_model_urls | model_row_rate |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in summary["by_site"]:
        lines.append(
            f"| {row['source_site']} | {row['total_rows']} | {row['with_model_url']} | {row['distinct_model_urls']} | {row['model_row_rate']:.2%} |"
        )
    lines.append("")
    lines.append("## Top Category Norm (All)")
    lines.append("")
    lines.extend(_top_lines(summary["category_norm_all"], "category", limit=15))
    lines.append("")
    lines.append("## Top Category Norm (With Model URL)")
    lines.append("")
    lines.extend(_top_lines(summary["category_norm_with_model"], "category", limit=15))
    lines.append("")
    lines.append("## Model Formats (With Model URL)")
    lines.append("")
    lines.extend(_top_lines(summary["model_format_with_model"], "model_format", limit=20))
    lines.append("")
    lines.append("## Asset Summary")
    lines.append("")
    lines.extend(_top_lines(summary["asset_summary"]["by_status"], "asset_status", limit=10))
    lines.append("")
    sancos = summary["sancos_fbx_summary"]
    lines.append("## Sancos FBX Probe")
    lines.append("")
    lines.append(f"- `distinct_checked_urls`: {sancos['distinct_checked_urls']}")
    lines.append(f"- `distinct_has_fbx_true`: {sancos['distinct_has_fbx_true']}")
    lines.append(f"- `distinct_has_fbx_false`: {sancos['distinct_has_fbx_false']}")
    lines.append(f"- `distinct_error_urls`: {sancos['distinct_error_urls']}")
    lines.append(f"- `sku_has_fbx_true`: {sancos['sku_has_fbx_true']}")
    lines.append(f"- `sku_has_fbx_false`: {sancos['sku_has_fbx_false']}")
    lines.append(f"- `sku_error`: {sancos['sku_error']}")
    lines.append("")
    lines.append("## Product Table Fields")
    lines.append("")
    for column in summary["tables"]["supplier_product"]:
        default = column["default"] if column["default"] is not None else ""
        lines.append(
            f"- `{column['name']}`: type=`{column['type']}`, notnull={column['notnull']}, pk={column['pk']}, default=`{default}`"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build consolidated supplier catalog analytics.")
    ap.add_argument("--db", default="data/sourse/suppliers/suppliers.db")
    ap.add_argument("--out-dir", default="out/supplier_catalog_analytics_20260423")
    args = ap.parse_args()

    db_path = Path(args.db).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as con:
        summary = _build_summary(con, db_path, out_dir)

    summary_path = out_dir / "summary.json"
    summary_md_path = out_dir / "summary.md"
    models_export_path = out_dir / "supplier_models_with_urls.json"

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_md_path.write_text(_to_markdown(summary), encoding="utf-8")

    export = build_catalog_export(
        db_paths=[db_path],
        sites=None,
        only_with_model_url=True,
        only_with_asset=False,
        only_rich=False,
    )
    models_export_path.write_text(json.dumps(export, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"summary_json = {summary_path}")
    print(f"summary_md = {summary_md_path}")
    print(f"models_export = {models_export_path}")
    print(f"model_items = {export['meta']['item_count']}")


if __name__ == "__main__":
    main()
