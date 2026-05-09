#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def room_scene_paths(apt_dir: Path, mode: str) -> list[Path]:
    return sorted((apt_dir / "rooms").glob(f"*/pipeline/{mode}/scene_requirements.v1.json"))


def numeric(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def item_supplier_price(item: dict[str, Any]) -> tuple[float | None, str]:
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    candidate = meta.get("supplier_candidate") if isinstance(meta.get("supplier_candidate"), dict) else {}
    value = numeric(candidate.get("price_value"))
    currency = str(candidate.get("price_currency") or "RUB")
    return value, currency


def kitchen_estimate(item: dict[str, Any]) -> tuple[float | None, str]:
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    price_estimate = meta.get("price_estimate") if isinstance(meta.get("price_estimate"), dict) else {}
    value = numeric(price_estimate.get("total_estimated_price"))
    currency = str(price_estimate.get("currency") or "RUB")
    return value, currency


def candidate_title(item: dict[str, Any]) -> str:
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    candidate = meta.get("supplier_candidate") if isinstance(meta.get("supplier_candidate"), dict) else {}
    return str(candidate.get("title") or item.get("name") or item.get("id") or "")


def summarize(apt_dir: Path, mode: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    unknown_rows: list[dict[str, Any]] = []
    totals = defaultdict(float)

    for scene_path in room_scene_paths(apt_dir, mode):
        scene = read_json(scene_path)
        room = scene.get("room") if isinstance(scene.get("room"), dict) else {}
        room_id = str(room.get("id") or scene_path.parents[2].name)
        for item in scene.get("placements") or []:
            if not isinstance(item, dict):
                continue
            item_id = str(item.get("id") or "")
            name = str(item.get("name") or item_id)
            supplier_value, supplier_currency = item_supplier_price(item)
            kitchen_value, kitchen_currency = kitchen_estimate(item)
            value = supplier_value if supplier_value is not None else kitchen_value
            currency = supplier_currency if supplier_value is not None else kitchen_currency
            source = "supplier_candidate.price_value" if supplier_value is not None else "kitchen.price_estimate.total_estimated_price"
            source_dict = item.get("source") if isinstance(item.get("source"), dict) else {}
            asset_source = str(source_dict.get("asset_source") or "")
            meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
            has_supplier = bool(meta.get("supplier_binding_applied") or meta.get("supplier_requirement_added"))
            has_kitchen_estimate = kitchen_value is not None
            if value is not None:
                row = {
                    "room_id": room_id,
                    "item_id": item_id,
                    "name": name,
                    "title": candidate_title(item),
                    "category": item.get("category"),
                    "semantic_group": item.get("semantic_group"),
                    "value": round(value, 2),
                    "currency": currency,
                    "source": source,
                    "asset_source": asset_source,
                }
                rows.append(row)
                totals[currency] += value
            elif has_supplier or has_kitchen_estimate or asset_source in {"supplier_catalog_local_asset", "procedural_kitchen"}:
                unknown_rows.append(
                    {
                        "room_id": room_id,
                        "item_id": item_id,
                        "name": name,
                        "title": candidate_title(item),
                        "category": item.get("category"),
                        "semantic_group": item.get("semantic_group"),
                        "asset_source": asset_source,
                        "reason": "missing_numeric_price",
                    }
                )

    return {
        "apartment_dir": str(apt_dir.resolve()),
        "mode": mode,
        "currency_totals": {currency: round(value, 2) for currency, value in sorted(totals.items())},
        "priced_item_count": len(rows),
        "unknown_price_item_count": len(unknown_rows),
        "priced_items": rows,
        "unknown_price_items": unknown_rows,
        "notes": [
            "Totals include supplier_candidate.price_value and procedural kitchen total_estimated_price where available.",
            "Items without numeric supplier price are listed in unknown_price_items, so the total is a known-price subtotal.",
        ],
    }


def write_markdown(report: dict[str, Any], out_path: Path) -> None:
    lines = [
        "# Apartment renovation cost",
        "",
        f"- apartment: `{report['apartment_dir']}`",
        f"- mode: `{report['mode']}`",
    ]
    for currency, value in report["currency_totals"].items():
        lines.append(f"- known-price subtotal: **{value:,.2f} {currency}**".replace(",", " "))
    lines.extend(
        [
            f"- priced items: {report['priced_item_count']}",
            f"- unknown-price items: {report['unknown_price_item_count']}",
            "",
            "## Priced Items",
            "",
            "| Room | Item | Category | Price | Source |",
            "|---|---|---|---:|---|",
        ]
    )
    for row in report["priced_items"]:
        lines.append(
            f"| `{row['room_id']}` | {row['title']} | {row.get('category') or ''} | "
            f"{row['value']:.2f} {row['currency']} | {row['source']} |"
        )
    if report["unknown_price_items"]:
        lines.extend(["", "## Unknown Price Items", "", "| Room | Item | Category | Reason |", "|---|---|---|---|"])
        for row in report["unknown_price_items"]:
            lines.append(f"| `{row['room_id']}` | {row['title']} | {row.get('category') or ''} | {row['reason']} |")
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {note}" for note in report["notes"])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize apartment supplier/kitchen costs from scene_requirements JSON files.")
    parser.add_argument("apt_dir")
    parser.add_argument("--mode", default="optimal")
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-md", default=None)
    args = parser.parse_args()

    apt_dir = Path(args.apt_dir).expanduser().resolve()
    report = summarize(apt_dir, args.mode)
    out_json = Path(args.out_json).expanduser().resolve() if args.out_json else apt_dir / "apartment_pipeline" / args.mode / "renovation_cost_report.json"
    out_md = Path(args.out_md).expanduser().resolve() if args.out_md else apt_dir / "apartment_pipeline" / args.mode / "renovation_cost_report.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, out_md)
    print(json.dumps({"out_json": str(out_json), "out_md": str(out_md), "currency_totals": report["currency_totals"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
