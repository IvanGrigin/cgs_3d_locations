#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
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


def polygon_area_m2(room: dict[str, Any]) -> float | None:
    poly = room.get("floor_polygon") if isinstance(room.get("floor_polygon"), list) else []
    points: list[tuple[float, float]] = []
    for point in poly:
        if not isinstance(point, dict):
            continue
        x = numeric(point.get("x"))
        y = numeric(point.get("y", point.get("z")))
        if x is not None and y is not None:
            points.append((x, y))
    if len(points) >= 3:
        area = abs(
            sum(points[i][0] * points[(i + 1) % len(points)][1] - points[(i + 1) % len(points)][0] * points[i][1] for i in range(len(points)))
        ) * 0.5
        if area > 0:
            return area
    area = numeric(room.get("area_m2"))
    if area and area > 0:
        return area
    width = numeric(room.get("width_m") or room.get("width"))
    depth = numeric(room.get("depth_m") or room.get("depth"))
    if width and depth and width > 0 and depth > 0:
        return width * depth
    return None


def polygon_perimeter_m(room: dict[str, Any]) -> float | None:
    poly = room.get("floor_polygon") if isinstance(room.get("floor_polygon"), list) else []
    points: list[tuple[float, float]] = []
    for point in poly:
        if not isinstance(point, dict):
            continue
        x = numeric(point.get("x"))
        y = numeric(point.get("y", point.get("z")))
        if x is not None and y is not None:
            points.append((x, y))
    if len(points) >= 3:
        perimeter = sum(math.hypot(points[(i + 1) % len(points)][0] - points[i][0], points[(i + 1) % len(points)][1] - points[i][1]) for i in range(len(points)))
        if perimeter > 0:
            return perimeter
    width = numeric(room.get("width_m") or room.get("width"))
    depth = numeric(room.get("depth_m") or room.get("depth"))
    if width and depth and width > 0 and depth > 0:
        return 2.0 * (width + depth)
    return None


def surface_material_cost(room: dict[str, Any], key: str) -> tuple[float | None, str, dict[str, Any]]:
    material = room.get(key) if isinstance(room.get(key), dict) else {}
    if not material:
        return None, "RUB", {}
    unit_price = numeric(material.get("price") or material.get("unit_price"))
    currency = str(material.get("price_currency") or "RUB")
    if unit_price is None:
        return None, currency, {"reason": "missing_numeric_surface_price"}

    if key == "floor_material":
        area = polygon_area_m2(room) or 0.0
        package_area = numeric(material.get("package_area_m2"))
        if package_area and package_area > 0 and area > 0:
            quantity = math.ceil(area / package_area)
            return unit_price * quantity, currency, {"area_m2": round(area, 3), "package_area_m2": package_area, "quantity": quantity}
        if area > 0:
            return unit_price * area, currency, {"area_m2": round(area, 3), "quantity": round(area, 3), "quantity_assumption": "price_per_m2"}
        return unit_price, currency, {"quantity": 1, "quantity_assumption": "single_unit"}

    perimeter = polygon_perimeter_m(room) or 0.0
    height = numeric(room.get("ceiling_height_m") or room.get("ceiling_height")) or 2.7
    wall_area = perimeter * height if perimeter > 0 else 0.0
    roll_area = numeric(material.get("roll_area_m2") or material.get("package_area_m2"))
    if roll_area and roll_area > 0 and wall_area > 0:
        quantity = math.ceil(wall_area / roll_area)
        return unit_price * quantity, currency, {"area_m2": round(wall_area, 3), "roll_area_m2": roll_area, "quantity": quantity}
    if wall_area > 0:
        return unit_price * wall_area, currency, {"area_m2": round(wall_area, 3), "quantity": round(wall_area, 3), "quantity_assumption": "price_per_m2"}
    return unit_price, currency, {"quantity": 1, "quantity_assumption": "single_unit"}


def surface_title(material: dict[str, Any], fallback: str) -> str:
    return str(material.get("name") or material.get("sku") or fallback)


def summarize(apt_dir: Path, mode: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    unknown_rows: list[dict[str, Any]] = []
    totals = defaultdict(float)

    for scene_path in room_scene_paths(apt_dir, mode):
        scene = read_json(scene_path)
        room = scene.get("room") if isinstance(scene.get("room"), dict) else {}
        room_id = str(room.get("id") or scene_path.parents[2].name)
        for material_key, category, label in (
            ("floor_material", "floor_material", "Floor material"),
            ("wall_material", "wall_material", "Wall material"),
        ):
            material = room.get(material_key) if isinstance(room.get(material_key), dict) else {}
            if not material:
                continue
            value, currency, quantity = surface_material_cost(room, material_key)
            if value is not None:
                row = {
                    "room_id": room_id,
                    "item_id": material_key,
                    "name": label,
                    "title": surface_title(material, label),
                    "category": category,
                    "semantic_group": category,
                    "value": round(value, 2),
                    "currency": currency,
                    "source": f"room.{material_key}.price",
                    "asset_source": str(material.get("source") or "surface_material_catalog"),
                    "quantity": quantity,
                }
                rows.append(row)
                totals[currency] += value
            else:
                unknown_rows.append(
                    {
                        "room_id": room_id,
                        "item_id": material_key,
                        "name": label,
                        "title": surface_title(material, label),
                        "category": category,
                        "semantic_group": category,
                        "asset_source": str(material.get("source") or "surface_material_catalog"),
                        "reason": quantity.get("reason") or "missing_numeric_surface_price",
                    }
                )
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
            "Totals include supplier_candidate.price_value, procedural kitchen total_estimated_price, and room floor/wall material estimates where available.",
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
