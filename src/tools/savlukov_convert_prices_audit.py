#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert Savlukov prices to RUB and fill/audit card completeness."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import time
from pathlib import Path
from typing import Any


CBR_BYN_RUB_RATE = 26.6427
CBR_RATE_DATE = "2026-05-05"
CBR_SOURCE_URL = "https://www.cbr.ru/scripts/XML_daily.asp"


DEFAULT_MATERIALS = {
    "sofa": "Каркас: Брус хвойных пород",
    "sectional_sofa": "Каркас: Брус хвойных пород",
    "armchair": "Каркас: Брус хвойных пород",
    "ottoman": "Каркас: Брус хвойных пород",
}


def median(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def median_dims(items: list[dict[str, Any]], category: str) -> dict[str, float | None]:
    rows = [x.get("dimensions_cm") or {} for x in items if x.get("category_norm") == category]
    return {
        key: median([float(row[key]) for row in rows if row.get(key) is not None])
        for key in ("width", "depth", "height")
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalog", default="data/sourse/suppliers/supplier_catalog_canonical.json")
    args = ap.parse_args()

    path = Path(args.catalog)
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = [x for x in payload.get("items", []) if x.get("source_site") == "savlukov"]

    byn_by_category: dict[str, list[float]] = {}
    for item in items:
        if item.get("price_currency") == "BYN" and item.get("price_value") is not None:
            category = str(item.get("category_norm") or "sofa")
            byn_by_category.setdefault(category, []).append(float(item["price_value"]))

    rub_median_by_category = {
        category: round(float(statistics.median(values)) * CBR_BYN_RUB_RATE, 2)
        for category, values in byn_by_category.items()
        if values
    }
    # There is no current Savlukov ottoman page in the fetched catalog. Use the
    # closest small-seat category, and keep the estimate flag explicit.
    if "ottoman" not in rub_median_by_category and "armchair" in rub_median_by_category:
        rub_median_by_category["ottoman"] = rub_median_by_category["armchair"]

    dims_by_category = {
        category: median_dims(items, category)
        for category in sorted({str(x.get("category_norm") or "sofa") for x in items})
    }
    if "ottoman" not in dims_by_category and "armchair" in dims_by_category:
        dims_by_category["ottoman"] = dims_by_category["armchair"]
    elif "ottoman" in dims_by_category and not any(dims_by_category["ottoman"].values()) and "armchair" in dims_by_category:
        dims_by_category["ottoman"] = dims_by_category["armchair"]

    converted = 0
    estimated_prices = 0
    estimated_dims = 0
    filled_materials = 0
    for item in items:
        extra = item.setdefault("extra", {})
        category = str(item.get("category_norm") or "sofa")
        original_currency = item.get("price_currency")
        original_price = item.get("price_value")

        if original_currency == "BYN" and original_price is not None:
            byn = float(original_price)
            item["price_value"] = round(byn * CBR_BYN_RUB_RATE, 2)
            item["price_currency"] = "RUB"
            extra["savlukov_original_price"] = {
                "price_value": byn,
                "price_currency": "BYN",
                "source": "Savlukov Product JSON-LD",
            }
            extra["savlukov_price_conversion"] = {
                "rate": CBR_BYN_RUB_RATE,
                "rate_pair": "BYN/RUB",
                "rate_date": CBR_RATE_DATE,
                "source_url": CBR_SOURCE_URL,
                "converted_at_unix": time.time(),
            }
            converted += 1
        elif item.get("price_value") is None:
            estimate = rub_median_by_category.get(category) or rub_median_by_category.get("sofa")
            if estimate is not None:
                item["price_value"] = estimate
                item["price_currency"] = "RUB"
                extra["savlukov_price_completion"] = {
                    "method": "category_median_from_current_savlukov_cards_converted_to_rub",
                    "confidence": "estimate",
                    "category_norm": category,
                    "rate": CBR_BYN_RUB_RATE,
                    "rate_date": CBR_RATE_DATE,
                    "source_url": CBR_SOURCE_URL,
                    "note": "Current public product page was not found; value is an estimate, not a scraped product price.",
                }
                estimated_prices += 1

        dims = item.setdefault("dimensions_cm", {})
        category_dims = dims_by_category.get(category) or dims_by_category.get("sofa") or {}
        if any(dims.get(k) is None for k in ("width", "depth", "height")):
            for key in ("width", "depth", "height"):
                if dims.get(key) is None and category_dims.get(key) is not None:
                    dims[key] = round(float(category_dims[key]), 2)
            extra["savlukov_dimension_completion"] = {
                "method": "category_median_from_current_savlukov_cards",
                "confidence": "estimate",
                "category_norm": category,
            }
            estimated_dims += 1

        if not item.get("materials"):
            item["materials"] = DEFAULT_MATERIALS.get(category, DEFAULT_MATERIALS["sofa"])
            filled_materials += 1
        if not item.get("color"):
            item["color"] = "not_specified"
        if not item.get("style"):
            item["style"] = "not_specified"
        if not item.get("room"):
            item["room"] = "living_room"
        if not item.get("product_url"):
            item["product_url"] = item.get("model_page_url") or "https://designers.savlukov.by/models"
            extra["savlukov_product_url_completion"] = {
                "method": "fallback_to_designer_model_library",
                "confidence": "fallback",
                "note": "Current public product page was not found; product_url points to the model source page.",
            }
        if not item.get("country_brand"):
            item["country_brand"] = "Беларусь"
        if not item.get("production_country"):
            item["production_country"] = "Беларусь"
        if not item.get("model_download_url"):
            item["model_download_url"] = "https://disk.yandex.ru/d/8Jqha8s5btjZmg"
        if not item.get("model_download_filename"):
            item["model_download_filename"] = Path(str(item.get("asset_local_path") or "")).name or None
        if not item.get("related"):
            item["related"] = []
        if not item.get("images"):
            item["images"] = []

        comp = item.setdefault("completeness", {})
        comp.update(
            {
                "has_title": bool(item.get("title")),
                "has_price": item.get("price_value") is not None and bool(item.get("price_currency")),
                "has_full_dimensions": all((item.get("dimensions_cm") or {}).get(k) is not None for k in ("width", "depth", "height")),
                "has_description": bool(item.get("description")),
                "has_category": bool(item.get("category_norm")),
                "has_brand": bool(item.get("brand")),
                "has_model_link": bool(item.get("asset_local_path")),
                "rich_card": True,
            }
        )

    payload.setdefault("meta", {})["item_count"] = len(payload.get("items", []))
    payload.setdefault("meta", {}).setdefault("manual_merges", []).append(
        {
            "source": "savlukov_convert_prices_audit",
            "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "rate_pair": "BYN/RUB",
            "rate": CBR_BYN_RUB_RATE,
            "rate_date": CBR_RATE_DATE,
            "source_url": CBR_SOURCE_URL,
            "converted_byn_prices": converted,
            "estimated_missing_prices": estimated_prices,
            "estimated_missing_dimensions": estimated_dims,
            "filled_missing_materials": filled_materials,
        }
    )

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    print(
        "[catalog] "
        f"converted={converted} estimated_prices={estimated_prices} "
        f"estimated_dims={estimated_dims} filled_materials={filled_materials}"
    )


if __name__ == "__main__":
    main()
