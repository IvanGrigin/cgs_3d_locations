#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script exports grouped unified-catalog samples for quick manual review.
It reads the unified supplier database and produces curated group snapshots.
The output helps inspect canonical merges and category consistency.
It is an analysis helper, not part of the main ingestion path.
Keep the sampling rules deterministic and lightweight.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from pathlib import Path
from typing import Any
from src.suppliers.utils import json_loads_or


def _load_rows(db_path: Path) -> list[dict[str, Any]]:
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """
            SELECT
                canonical_key,
                unique_key,
                source_site,
                source_db_path,
                source_url,
                parsed_at,
                external_id,
                title,
                brand,
                collection,
                category_raw,
                category_norm,
                semantic_group,
                product_url,
                model_link_type,
                model_page_url,
                model_download_url,
                model_download_landing_url,
                model_vendor_url,
                model_extraction_method,
                model_download_filename,
                model_format,
                price_value,
                price_currency,
                old_price_value,
                style,
                color,
                description,
                width_cm,
                depth_cm,
                height_cm,
                weight_kg,
                room,
                materials,
                availability,
                country_brand,
                production_country,
                tags_json,
                images_json,
                related_json,
                extra_json,
                asset_status,
                asset_format,
                asset_source_url,
                asset_local_path,
                preview_local_path,
                mesh_local_path,
                mesh_format,
                mesh_status,
                mesh_source_url,
                mesh_ready,
                mesh_available,
                merged_unique_keys_json,
                merged_source_dbs_json
            FROM supplier_mesh_catalog
            ORDER BY semantic_group, title, canonical_key
            """
        ).fetchall()
    return [dict(row) for row in rows]


def _effective_group(row: dict[str, Any]) -> str:
    for key in ("semantic_group", "category_norm", "category_raw"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return "uncategorized"


def _has_text(value: Any) -> bool:
    return bool(str(value or "").strip())


def _has_model_link(row: dict[str, Any]) -> bool:
    return any(
        str(row.get(key) or "").strip()
        for key in ("model_download_url", "model_page_url", "product_url", "mesh_source_url")
    )


def _has_full_dimensions(row: dict[str, Any]) -> bool:
    return (
        row.get("width_cm") is not None
        and row.get("depth_cm") is not None
        and row.get("height_cm") is not None
    )


def _has_category(row: dict[str, Any]) -> bool:
    return _has_text(row.get("category_raw")) or _has_text(row.get("category_norm"))


def _is_rich_card(row: dict[str, Any]) -> bool:
    return all(
        (
            _has_text(row.get("title")),
            row.get("price_value") is not None,
            _has_full_dimensions(row),
            _has_text(row.get("description")),
            _has_category(row),
            _has_text(row.get("brand")),
        )
    )


def _dims_count(row: dict[str, Any]) -> int:
    return sum(row.get(key) is not None for key in ("width_cm", "depth_cm", "height_cm"))


def _volume_score(row: dict[str, Any]) -> float | None:
    dims = [row.get("width_cm"), row.get("depth_cm"), row.get("height_cm")]
    clean = []
    for dim in dims:
        try:
            if dim is None:
                continue
            clean.append(float(dim))
        except Exception:
            continue
    if not clean:
        return None
    if len(clean) == 3:
        return clean[0] * clean[1] * clean[2]
    score = 1.0
    for dim in clean:
        score *= dim
    return score


def _quality_key(row: dict[str, Any]) -> tuple[int, int, int, int, int]:
    return (
        1 if _has_model_link(row) else 0,
        1 if row.get("mesh_available") else 0,
        _dims_count(row),
        1 if str(row.get("images_json") or "").strip() not in {"", "[]"} else 0,
        1 if str(row.get("description") or "").strip() else 0,
    )


def _prepare_item(row: dict[str, Any]) -> dict[str, Any]:
    item = dict(row)
    item["tags"] = json_loads_or(item.pop("tags_json", None), [])
    item["images"] = json_loads_or(item.pop("images_json", None), [])
    item["related"] = json_loads_or(item.pop("related_json", None), [])
    item["extra"] = json_loads_or(item.pop("extra_json", None), {})
    item["merged_unique_keys"] = json_loads_or(item.pop("merged_unique_keys_json", None), [])
    item["merged_source_dbs"] = json_loads_or(item.pop("merged_source_dbs_json", None), [])
    item["volume_score"] = _volume_score(row)
    item["rich_card"] = _is_rich_card(row)
    return item


def _select_diverse_rows(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if len(rows) <= limit:
        return rows

    sorted_rows = sorted(
        rows,
        key=lambda row: (
            _volume_score(row) is None,
            _volume_score(row) or 0.0,
            tuple(-x for x in _quality_key(row)),
            str(row.get("title") or ""),
        ),
    )

    picked_indices: set[int] = set()
    selected: list[dict[str, Any]] = []

    for step in range(limit):
        raw_index = round(step * (len(sorted_rows) - 1) / max(1, limit - 1))
        index = max(0, min(len(sorted_rows) - 1, raw_index))
        while index in picked_indices and index + 1 < len(sorted_rows):
            index += 1
        while index in picked_indices and index - 1 >= 0:
            index -= 1
        if index in picked_indices:
            continue
        picked_indices.add(index)
        selected.append(sorted_rows[index])

    if len(selected) < limit:
        remainder = [
            row for idx, row in enumerate(sorted_rows)
            if idx not in picked_indices
        ]
        remainder.sort(
            key=lambda row: (
                tuple(-x for x in _quality_key(row)),
                -(_volume_score(row) or 0.0),
                str(row.get("title") or ""),
            ),
        )
        selected.extend(remainder[: limit - len(selected)])

    return selected[:limit]


def build_group_samples(db_path: Path, limit_per_group: int, rich_only: bool = False) -> dict[str, Any]:
    rows = _load_rows(db_path)
    groups: dict[str, list[dict[str, Any]]] = {}

    for row in rows:
        if not _has_model_link(row):
            continue
        if rich_only and not _is_rich_card(row):
            continue
        groups.setdefault(_effective_group(row), []).append(row)

    sample_groups: list[dict[str, Any]] = []
    total_items = 0

    for group_name in sorted(groups):
        group_rows = groups[group_name]
        group_rows.sort(
            key=lambda row: (
                tuple(-x for x in _quality_key(row)),
                _volume_score(row) is None,
                _volume_score(row) or 0.0,
                str(row.get("title") or ""),
            ),
            reverse=False,
        )
        selected = _select_diverse_rows(group_rows, limit_per_group)
        prepared = [_prepare_item(row) for row in selected]
        total_items += len(prepared)
        sample_groups.append(
            {
                "group": group_name,
                "source_count": len(group_rows),
                "sample_count": len(prepared),
                "items": prepared,
            }
        )

    return {
        "schema": "supplier_mesh_group_samples/v1",
        "meta": {
            "source_db": str(db_path.resolve()),
            "group_count": len(sample_groups),
            "sample_item_count": total_items,
            "limit_per_group": limit_per_group,
            "rich_only": rich_only,
        },
        "groups": sample_groups,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Export up to N diverse supplier cards per semantic group.")
    ap.add_argument("--db", required=True, help="Unified supplier mesh catalog DB")
    ap.add_argument("--limit-per-group", type=int, default=10)
    ap.add_argument("--rich-only", action="store_true", help="Keep only rich cards with title, price, dimensions, description, category and brand")
    ap.add_argument("--out", required=True, help="Output JSON path")
    args = ap.parse_args()

    db_path = Path(args.db).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    data = build_group_samples(
        db_path=db_path,
        limit_per_group=max(1, int(args.limit_per_group)),
        rich_only=bool(args.rich_only),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"groups = {data['meta']['group_count']}")
    print(f"items = {data['meta']['sample_item_count']}")
    print(f"saved = {out_path}")


if __name__ == "__main__":
    main()
