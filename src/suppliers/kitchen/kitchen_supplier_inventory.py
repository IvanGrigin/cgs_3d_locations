#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SUPPLIER_CATALOG = "data/sourse/suppliers/supplier_catalog_canonical.json"


BUCKET_CATEGORY_NORMS: dict[str, set[str]] = {
    "kitchen_sets": {"kitchen_set"},
    "kitchen_sinks": {"kitchen_sink"},
    "kitchen_faucets": {"kitchen_faucet"},
    "cooktops": {"cooktop_hob"},
    "hoods": {"extractor_hood"},
    "fridges": {"refrigerator_freezer"},
    "dishwashers": {"dishwasher"},
    "ovens": {"oven"},
    "small_appliances": {"small_kitchen_appliance", "microwave"},
    "kitchenware": {"kitchenware"},
    "food_fruit": {"food_drink"},
    "oil_bottles_decor": {"decorative_set"},
    "flowers_vases": {"plant_planter_vase", "plant", "vase"},
    "dining_tables": {"dining_table"},
    "dining_chairs": {"chair", "dining_chair", "stool"},
}


BUCKET_TEXT_TERMS: dict[str, tuple[str, ...]] = {
    "kitchen_sets": ("кухня", "кухонный гарнитур", "kitchen set"),
    "kitchen_sinks": ("кухонная мойка", "kitchen sink"),
    "kitchen_faucets": ("смеситель для кухни", "kitchen faucet", "kitchen mixer"),
    "cooktops": ("варочная", "cooktop", "hob", "induction"),
    "hoods": ("вытяжка", "rangehood", "extractor hood"),
    "fridges": ("холодильник", "fridge", "refrigerator"),
    "dishwashers": ("посудомоеч", "dishwasher"),
    "ovens": ("духовой", "духовка", "oven"),
    "small_appliances": ("чайник", "кофемашина", "kettle", "coffee machine", "microwave", "микроволнов"),
    "kitchenware": (
        "kitchen accessories",
        "kitchen decor",
        "kitchenware",
        "tableware",
        "набор банок",
        "мелочь для кухни",
        "посуда",
        "тарел",
        "чаш",
        "миска",
    ),
    "food_fruit": (
        "fruit plate",
        "fruit",
        "apples",
        "apple",
        "food drink",
        "еда и напитки",
        "фрукт",
        "яблок",
        "лимон",
        "хлеб",
    ),
    "oil_bottles_decor": (
        "olive and oil",
        "oil decorative",
        "decanters and bottles",
        "bottle",
        "decanter",
        "jar",
        "бутыл",
        "масл",
        "графин",
        "банка",
    ),
    "flowers_vases": (
        "flower vase",
        "flower bouquet",
        "bouquet",
        "vase",
        "plant",
        "букет",
        "цвет",
        "ваза",
        "растение",
    ),
    "dining_tables": ("dining table", "обеденный стол"),
    "dining_chairs": ("dining chair", "bar stool", "стул", "барный стул"),
}


TEXT_BUCKET_ALLOWED_CATEGORY_NORMS: dict[str, set[str] | None] = {
    "kitchen_sets": {"kitchen_set"},
    "kitchen_sinks": {"kitchen_sink"},
    "kitchen_faucets": {"kitchen_faucet"},
    "cooktops": {"cooktop_hob", "small_kitchen_appliance"},
    "hoods": {"extractor_hood"},
    "fridges": {"refrigerator_freezer"},
    "dishwashers": {"dishwasher"},
    "ovens": {"oven", "small_kitchen_appliance"},
    "small_appliances": {"small_kitchen_appliance", "microwave", "kitchenware"},
    "kitchenware": {"kitchenware", "food_drink", "decorative_set", "small_kitchen_appliance", "table"},
    "food_fruit": {"food_drink", "kitchenware", "decorative_set", "plant_planter_vase", "sculpture_decor_set"},
    "oil_bottles_decor": {"food_drink", "kitchenware", "decorative_set", "plant_planter_vase", "sculpture_decor_set"},
    "flowers_vases": {"plant_planter_vase", "plant", "vase", "decorative_set", "sculpture_decor_set"},
    "dining_tables": {"dining_table", "table"},
    "dining_chairs": {"chair", "dining_chair", "stool", "bar_stool"},
}


PREFERRED_BUCKET_ORDER = (
    "kitchen_sets",
    "kitchen_sinks",
    "kitchen_faucets",
    "cooktops",
    "hoods",
    "fridges",
    "dishwashers",
    "ovens",
    "small_appliances",
    "kitchenware",
    "food_fruit",
    "oil_bottles_decor",
    "flowers_vases",
    "dining_tables",
    "dining_chairs",
)


CORE_KITCHEN_BUCKETS = {
    "kitchen_sets",
    "kitchen_sinks",
    "kitchen_faucets",
    "cooktops",
    "hoods",
    "fridges",
    "dishwashers",
    "ovens",
    "small_appliances",
}


KITCHEN_ACCESSORY_BUCKETS = {
    "kitchenware",
    "food_fruit",
    "oil_bottles_decor",
    "flowers_vases",
}


KITCHEN_DINING_BUCKETS = {
    "dining_tables",
    "dining_chairs",
}


def load_supplier_catalog(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if isinstance(data, dict):
        rows = data.get("items") or data.get("products") or []
    else:
        rows = data  # pragma: no cover
    return [row for row in rows if isinstance(row, dict)]


def _norm(value: Any) -> str:
    text = str(value or "").replace("ё", "е").replace("Ё", "Е").lower()
    text = re.sub(r"[_/\\-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _row_text(row: dict[str, Any]) -> str:
    keys = (
        "title",
        "name",
        "category_norm",
        "category_raw",
        "description",
        "vlm_description_summary",
        "vlm_description_text",
        "product_url",
        "unique_key",
    )
    return _norm(" ".join(str(row.get(key) or "") for key in keys))


def _has_local_asset(row: dict[str, Any]) -> bool:
    return bool(str(row.get("asset_local_path") or "").strip())


def _has_downloadable_asset(row: dict[str, Any]) -> bool:
    return bool(
        str(row.get("model_download_url") or "").strip()
        or str(row.get("model_download_landing_url") or "").strip()
        or str(row.get("model_page_url") or "").strip()
    )


def supplier_row_key(row: dict[str, Any]) -> str:
    return str(row.get("unique_key") or row.get("product_url") or row.get("title") or id(row))


def classify_supplier_row(row: dict[str, Any]) -> set[str]:
    category_norm = str(row.get("category_norm") or "").strip()
    text = _row_text(row)
    buckets: set[str] = set()

    for bucket, categories in BUCKET_CATEGORY_NORMS.items():
        if category_norm in categories:
            buckets.add(bucket)

    for bucket, terms in BUCKET_TEXT_TERMS.items():
        allowed_categories = TEXT_BUCKET_ALLOWED_CATEGORY_NORMS.get(bucket)
        if allowed_categories is not None and category_norm not in allowed_categories:
            continue
        if any(_norm(term) in text for term in terms):
            buckets.add(bucket)

    return buckets


def collect_kitchen_supplier_items(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = {bucket: [] for bucket in PREFERRED_BUCKET_ORDER}
    seen: dict[str, set[str]] = {bucket: set() for bucket in PREFERRED_BUCKET_ORDER}
    for row in rows:
        key = supplier_row_key(row)
        for bucket in classify_supplier_row(row):
            if bucket not in buckets:
                buckets[bucket] = []  # pragma: no cover
                seen[bucket] = set()  # pragma: no cover
            if key in seen[bucket]:
                continue  # pragma: no cover
            buckets[bucket].append(row)
            seen[bucket].add(key)
    return buckets


def collect_by_category_norm(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_category: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        category = str(row.get("category_norm") or "uncategorized")
        by_category.setdefault(category, []).append(row)
    return by_category


def build_kitchen_selection_index(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    catalog_rows = list(rows)
    kitchen_buckets = collect_kitchen_supplier_items(catalog_rows)
    kitchen_keys: set[str] = set()
    kitchen_items_by_key: dict[str, dict[str, Any]] = {}

    for bucket in PREFERRED_BUCKET_ORDER:
        for row in kitchen_buckets.get(bucket, []):
            key = supplier_row_key(row)
            kitchen_keys.add(key)
            entry = kitchen_items_by_key.setdefault(
                key,
                {
                    "row": row,
                    "buckets": [],
                    "is_core_kitchen": False,
                    "is_kitchen_accessory": False,
                    "is_kitchen_dining": False,
                },
            )
            entry["buckets"].append(bucket)
            entry["is_core_kitchen"] = bool(entry["is_core_kitchen"] or bucket in CORE_KITCHEN_BUCKETS)
            entry["is_kitchen_accessory"] = bool(entry["is_kitchen_accessory"] or bucket in KITCHEN_ACCESSORY_BUCKETS)
            entry["is_kitchen_dining"] = bool(entry["is_kitchen_dining"] or bucket in KITCHEN_DINING_BUCKETS)

    ordinary_rows = [row for row in catalog_rows if supplier_row_key(row) not in kitchen_keys]
    return {
        "kitchen_buckets": kitchen_buckets,
        "kitchen_items": list(kitchen_items_by_key.values()),
        "kitchen_keys": kitchen_keys,
        "ordinary_items": ordinary_rows,
        "all_by_category_norm": collect_by_category_norm(catalog_rows),
        "ordinary_by_category_norm": collect_by_category_norm(ordinary_rows),
    }


def _dimension_cm(row: dict[str, Any], key: str) -> Any:
    explicit_value = row.get(f"{key}_cm")
    if explicit_value is not None:
        return explicit_value
    dimensions = row.get("dimensions_cm")
    if isinstance(dimensions, dict):
        return dimensions.get(key)
    return None


def compact_item(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "unique_key": row.get("unique_key"),
        "title": row.get("title") or row.get("name"),
        "category_norm": row.get("category_norm"),
        "category_raw": row.get("category_raw"),
        "source_site": row.get("source_site"),
        "price": row.get("price") if row.get("price") is not None else row.get("price_value"),
        "price_currency": row.get("price_currency"),
        "asset_local_path": row.get("asset_local_path"),
        "model_download_url": row.get("model_download_url"),
        "model_download_landing_url": row.get("model_download_landing_url"),
        "product_url": row.get("product_url") or row.get("source_url"),
        "preview_local_path": row.get("preview_local_path"),
        "width_cm": _dimension_cm(row, "width"),
        "depth_cm": _dimension_cm(row, "depth"),
        "height_cm": _dimension_cm(row, "height"),
    }


def build_inventory_summary(buckets: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"buckets": {}}
    for bucket in PREFERRED_BUCKET_ORDER:
        items = buckets.get(bucket, [])
        summary["buckets"][bucket] = {
            "count": len(items),
            "local_asset_count": sum(1 for row in items if _has_local_asset(row)),
            "downloadable_count": sum(1 for row in items if _has_downloadable_asset(row)),
            "category_norms": Counter(str(row.get("category_norm") or "") for row in items).most_common(20),
            "source_sites": Counter(str(row.get("source_site") or "") for row in items).most_common(20),
            "examples": [compact_item(row) for row in items[:20]],
            "local_asset_examples": [compact_item(row) for row in items if _has_local_asset(row)][:20],
        }
    return summary


def build_selection_index_summary(index: dict[str, Any]) -> dict[str, Any]:
    kitchen_items = index.get("kitchen_items") or []
    ordinary_items = index.get("ordinary_items") or []
    ordinary_by_category = index.get("ordinary_by_category_norm") or {}
    return {
        "kitchen_unique_count": len(kitchen_items),
        "ordinary_count": len(ordinary_items),
        "ordinary_category_norms": sorted(
            ((category, len(items)) for category, items in ordinary_by_category.items()),
            key=lambda item: item[1],
            reverse=True,
        )[:50],
    }


def print_summary(summary: dict[str, Any]) -> None:
    for bucket in PREFERRED_BUCKET_ORDER:
        info = (summary.get("buckets") or {}).get(bucket) or {}
        print(
            f"{bucket}: count={info.get('count', 0)} "
            f"local={info.get('local_asset_count', 0)} "
            f"downloadable={info.get('downloadable_count', 0)}"
        )


def print_selection_index_summary(index_summary: dict[str, Any], category_limit: int = 20) -> None:
    print(
        f"kitchen_unique={index_summary.get('kitchen_unique_count', 0)} "
        f"ordinary={index_summary.get('ordinary_count', 0)}"
    )
    if category_limit > 0:
        print("")
        print("ordinary_categories:")
        for category, count in (index_summary.get("ordinary_category_norms") or [])[:category_limit]:
            print(f"- {category}: {count}")


def write_csv(path: Path, buckets: dict[str, list[dict[str, Any]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "bucket",
        "unique_key",
        "title",
        "category_norm",
        "category_raw",
        "source_site",
        "price",
        "price_currency",
        "asset_local_path",
        "model_download_url",
        "model_download_landing_url",
        "product_url",
        "preview_local_path",
        "width_cm",
        "depth_cm",
        "height_cm",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for bucket in PREFERRED_BUCKET_ORDER:
            for row in buckets.get(bucket, []):
                item = compact_item(row)
                item["bucket"] = bucket
                writer.writerow({key: item.get(key) for key in fieldnames})


def write_json(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect kitchen-related supplier catalog items.")
    parser.add_argument("--catalog", default=DEFAULT_SUPPLIER_CATALOG)
    parser.add_argument("--out-json", default=None, help="Optional summary JSON output path.")
    parser.add_argument("--out-csv", default=None, help="Optional flat CSV output path.")
    parser.add_argument("--bucket", default=None, choices=PREFERRED_BUCKET_ORDER, help="Print examples for one bucket.")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--ordinary-categories",
        action="store_true",
        help="Also print ordinary non-kitchen catalog category counts for kitchen-context fallback selection.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_cli().parse_args(argv)
    rows = load_supplier_catalog(args.catalog)
    index = build_kitchen_selection_index(rows)
    buckets = index["kitchen_buckets"]
    summary = build_inventory_summary(buckets)
    summary["source_catalog"] = str(Path(args.catalog).expanduser())
    summary["source_row_count"] = len(rows)
    summary["selection_index"] = build_selection_index_summary(index)

    print_summary(summary)
    if args.ordinary_categories:
        print("")
        print_selection_index_summary(summary["selection_index"], category_limit=max(0, int(args.limit or 0)))
    if args.bucket:
        print("")
        print(f"examples:{args.bucket}")
        for row in buckets.get(args.bucket, [])[: max(0, int(args.limit or 0))]:
            item = compact_item(row)
            print(f"- {item.get('title')} | {item.get('category_norm')} | {item.get('source_site')} | local={bool(item.get('asset_local_path'))}")

    if args.out_json:
        write_json(Path(args.out_json).expanduser(), summary)
    if args.out_csv:
        write_csv(Path(args.out_csv).expanduser(), buckets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())  # pragma: no cover
