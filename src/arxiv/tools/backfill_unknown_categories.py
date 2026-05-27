#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any


def _norm(value: Any) -> str:
    return str(value or "").strip().lower().replace("ё", "е")


def _ensure_columns(con: sqlite3.Connection) -> None:
    existing = {
        row[1]
        for row in con.execute("PRAGMA table_info(supplier_item)")
    }
    wanted: list[tuple[str, str]] = [
        ("category_norm_v2", "TEXT"),
        ("category_confidence", "REAL"),
        ("category_rule", "TEXT"),
    ]
    for name, typ in wanted:
        if name not in existing:
            con.execute(f"ALTER TABLE supplier_item ADD COLUMN {name} {typ}")
    con.commit()


def _raw_json(row: sqlite3.Row) -> dict[str, Any]:
    text = row["raw_json"]
    if not text:
        return {}
    try:
        data = json.loads(text)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _text_sources(row: sqlite3.Row, raw: dict[str, Any]) -> dict[str, str]:
    title = str(row["title"] or "")
    category_raw = str(row["category_raw"] or "")
    product_url = str(row["product_url"] or "")
    model_page_url = str(row["model_page_url"] or "")
    source_url = str(row["source_url"] or "")
    images = raw.get("images") or []
    image_text = " ".join(str(x) for x in images if x)
    materials = raw.get("materials") or row["materials_json"] or ""
    return {
        "title": _norm(title),
        "category_raw": _norm(category_raw),
        "url": _norm(" ".join([product_url, model_page_url, source_url])),
        "images": _norm(image_text),
        "materials": _norm(materials),
        "source_site": _norm(row["source_site"]),
    }


def _contains_any(text: str, keys: list[str]) -> bool:
    return any(k in text for k in keys)


def _rule_from_category_raw(text: str) -> tuple[str | None, float, str | None]:
    rules: list[tuple[str, list[str], str, float]] = [
        ("wall_light", ["освещение > бра", "бра"], "category_raw.wall_light", 0.99),
        ("floor_lamp", ["освещение > торшеры", "торшер"], "category_raw.floor_lamp", 0.99),
        ("chandelier", ["освещение > подвесной", "освещение > потолочный", "подвесной свет"], "category_raw.chandelier", 0.99),
        ("lamp", ["освещение > настольный", "настольный свет"], "category_raw.lamp", 0.98),
        ("armchair", ["мебель > кресла"], "category_raw.armchair", 0.99),
        ("chair", ["мебель > стулья", "лавки и табуретки"], "category_raw.chair", 0.99),
        ("table", ["мебель > столы", "мебель > стол + стул"], "category_raw.table", 0.98),
        ("bed", ["мебель > кровати"], "category_raw.bed", 0.99),
        ("cabinet", ["мебель > тумбы, комоды"], "category_raw.cabinet", 0.97),
        ("sofa", ["мебель > диваны", "мебель > другая мягкая мебель"], "category_raw.sofa", 0.97),
        ("desk", ["мебель > офисная мебель"], "category_raw.desk", 0.95),
        ("bookcase", ["мебель > стеллаж"], "category_raw.bookcase", 0.98),
        ("mirror", ["декор > зеркала"], "category_raw.mirror", 0.99),
        ("wall_art", ["декор > багеты"], "category_raw.wall_art", 0.9),
        ("decor", ["декор > другие предметы интерьера", "декор > скульптуры", "кухня > еда и напитки", "кухня > мелочь для кухни"], "category_raw.decor", 0.86),
        ("plant", ["растения > комнатные"], "category_raw.plant", 0.95),
        ("bathroom_fixture", ["санузел > ванна", "санузел > унитаз и биде", "санузел > декор для санузла", "раковины"], "category_raw.bathroom_fixture", 0.88),
        ("material", ["материалы >", "текстуры >"], "category_raw.material", 0.92),
        ("vehicle", ["транспорт >"], "category_raw.vehicle", 0.9),
        ("script", ["скрипты >"], "category_raw.script", 0.9),
        ("wardrobe", ["детская > шкафы"], "category_raw.wardrobe", 0.9),
    ]
    for category, keys, rule, confidence in rules:
        if _contains_any(text, keys):
            return category, confidence, rule
    return None, 0.0, None


def _rule_from_title_and_url(text: str) -> tuple[str | None, float, str | None]:
    rules: list[tuple[str, list[str], str, float]] = [
        ("nightstand", ["прикроват", "bedside", "nightstand"], "title.nightstand", 0.98),
        ("bed", ["кровать", "double bed", "king bed", "queen bed", "camelback bed"], "title.bed", 0.98),
        ("sofa", ["диван", "sectional", "seater", "sleeper", "sofa", "chaise"], "title.sofa", 0.97),
        ("armchair", ["кресло", "armchair", "lounge chair", "easy chair"], "title.armchair", 0.98),
        ("chair", ["стул", "bar stool", "bar chair", "office chair", "chair"], "title.chair", 0.97),
        ("console_table", ["стол-консоль", "console table", "консоль"], "title.console_table", 0.98),
        ("coffee_table", ["журналь", "coffee table"], "title.coffee_table", 0.98),
        ("desk", ["письменный стол", "рабочий стол", "desk"], "title.desk", 0.98),
        ("table", ["обеденный стол", "стол", "dining table", "table"], "title.table", 0.94),
        ("bookcase", ["стеллаж", "полка", "bookcase", "bookshelf", "держател", "подвесной стеллаж"], "title.bookcase", 0.97),
        ("sideboard", ["комод", "sideboard", "dresser", "сервант"], "title.sideboard", 0.98),
        ("cabinet", ["тумба", "cabinet"], "title.cabinet", 0.95),
        ("wardrobe", ["шкаф", "wardrobe", "closet"], "title.wardrobe", 0.97),
        ("ottoman", ["банкетка", "пуфик", "footstool", "ottoman", "bench"], "title.ottoman", 0.98),
        ("mirror", ["зеркало", "mirror"], "title.mirror", 0.98),
        ("wall_light", ["бра", "sconce", "wallsconce"], "title.wall_light", 0.98),
        ("floor_lamp", ["торшер", "floor lamp"], "title.floor_lamp", 0.98),
        ("chandelier", ["подвес", "pendant lamp", "ceiling lamp", "люстра"], "title.chandelier", 0.97),
        ("lamp", ["настольн", "table lamp", "lamp"], "title.lamp", 0.93),
        ("wall_art", ["картина", "панно", "poster", "print"], "title.wall_art", 0.97),
        ("hook", ["крючк"], "title.hook", 0.97),
        ("clothes_rack", ["стойк для одежды", "напольные вешалки", "вешалк"], "title.clothes_rack", 0.96),
        ("plant", ["растение", "tree", "plant"], "title.plant", 0.92),
        ("planter", ["кашпо", "planter"], "title.planter", 0.97),
        ("decor", ["статуэт", "сувенир", "figur", "decor", "decanter", "bottles"], "title.decor", 0.85),
    ]
    for category, keys, rule, confidence in rules:
        if _contains_any(text, keys):
            return category, confidence, rule
    return None, 0.0, None


def _rule_from_loftdesigne_images(image_text: str) -> tuple[str | None, float, str | None]:
    rules: list[tuple[str, list[str], str, float]] = [
        ("bed", ["bed-depth.png"], "loft.image.bed", 0.98),
        ("chair", ["chair.png", "rect-bar-chair.png", "round-bar-chair.png", "office-chair.png"], "loft.image.chair", 0.97),
        ("armchair", ["armchair.png"], "loft.image.armchair", 0.98),
        ("coffee_table", ["rect-coffee-table.png", "round-coffee-table.png"], "loft.image.coffee_table", 0.98),
        ("table", ["rect-table_2.png", "table-round_2.png", "square-table.png", "bar-table-scheme.png"], "loft.image.table", 0.97),
        ("mirror", ["mirror-rect_2.png", "mirror-round-hang-2.png", "mirror-1.png"], "loft.image.mirror", 0.98),
        ("chandelier", ["light-ceiling-round.png"], "loft.image.chandelier", 0.97),
        ("floor_lamp", ["light-floor-round.png"], "loft.image.floor_lamp", 0.97),
        ("lamp", ["light-table-round.png", "light-table-rect.png"], "loft.image.lamp", 0.97),
        ("sofa", ["couch.png"], "loft.image.sofa", 0.98),
        ("console_table", ["console.png"], "loft.image.console_table", 0.98),
        ("wardrobe", ["shkaf-scheme.png"], "loft.image.wardrobe", 0.98),
        ("sideboard", ["komod-scheme.png"], "loft.image.sideboard", 0.98),
        ("bookcase", ["stellaz-scheme.png", "polka-scheme.png"], "loft.image.bookcase", 0.98),
        ("cabinet", ["tumba-scheme.png"], "loft.image.cabinet", 0.96),
        ("mirror", ["zerkalo-krugloe.png"], "loft.image.mirror", 0.98),
    ]
    for category, keys, rule, confidence in rules:
        if _contains_any(image_text, keys):
            return category, confidence, rule
    return None, 0.0, None


def _dims_from_raw(raw: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    dims = raw.get("dimensions_cm") or {}
    width = dims.get("width")
    depth = dims.get("depth")
    height = dims.get("height")
    return (
        float(width) if isinstance(width, (int, float)) else None,
        float(depth) if isinstance(depth, (int, float)) else None,
        float(height) if isinstance(height, (int, float)) else None,
    )


def _rule_from_loftdesigne_material_dims(
    title_text: str,
    image_text: str,
    materials_text: str,
    raw: dict[str, Any],
) -> tuple[str | None, float, str | None]:
    width, depth, height = _dims_from_raw(raw)
    known_dims = [x for x in (width, depth, height) if x is not None]
    max_dim = max(known_dims) if known_dims else None

    if "icon-no-lamps.svg" in image_text:
        if height is not None and height >= 120:
            return "floor_lamp", 0.9, "loft.material_dims.floor_lamp"
        if max_dim is not None and max_dim <= 80:
            return "lamp", 0.88, "loft.material_dims.table_lamp"
        return "lamp", 0.82, "loft.material_dims.lamp_family"

    if "tumba-scheme.png" in image_text:
        if width is not None and width <= 65 and (height is None or height <= 75):
            return "nightstand", 0.85, "loft.material_dims.nightstand"
        return "cabinet", 0.82, "loft.material_dims.cabinet"

    small_materials = [
        "керамика",
        "смола",
        "полирезин",
        "камень",
        "бетон",
        "мрамор",
        "стекло",
    ]
    if max_dim is not None and max_dim <= 25 and _contains_any(materials_text, small_materials):
        return "decor", 0.82, "loft.material_dims.small_decor"

    if max_dim is not None and max_dim <= 18 and _contains_any(materials_text, ["металл", "камень", "керамика", "смола"]):
        return "decor", 0.8, "loft.material_dims.tiny_decor"

    if _contains_any(title_text, ["сундук", "чемодан", "сертификат"]):
        return "decor", 0.8, "title.decor_storage"

    return None, 0.0, None


def _infer_category(row: sqlite3.Row) -> tuple[str | None, float, str]:
    raw = _raw_json(row)
    texts = _text_sources(row, raw)

    existing = _norm(row["category_norm"])
    if existing and existing != "unknown":
        return existing, 1.0, "existing.category_norm"

    category_raw = texts["category_raw"]
    title_and_url = " ".join([texts["title"], texts["url"], texts["category_raw"]])
    images = texts["images"]
    source_site = texts["source_site"]

    category, confidence, rule = _rule_from_category_raw(category_raw)
    if category:
        return category, confidence, rule or "category_raw"

    category, confidence, rule = _rule_from_title_and_url(title_and_url)
    if category:
        return category, confidence, rule or "title"

    if source_site == "loftdesigne":
        category, confidence, rule = _rule_from_loftdesigne_images(images)
        if category:
            return category, confidence, rule or "loft.image"
        category, confidence, rule = _rule_from_loftdesigne_material_dims(
            texts["title"], images, texts["materials"], raw
        )
        if category:
            return category, confidence, rule or "loft.material_dims"

    return None, 0.0, "unresolved"


def _write_audit_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "source_site",
        "title",
        "category_raw",
        "category_norm_old",
        "category_norm_v2",
        "category_confidence",
        "category_rule",
        "product_url",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser(description="Backfill normalized categories in unified supplier catalog DB")
    ap.add_argument("--db", required=True, help="Path to the working supplier catalog DB")
    ap.add_argument("--out-report-json", default=None)
    ap.add_argument("--out-audit-csv", default=None)
    ap.add_argument("--writeback-unknown", action="store_true", help="Copy category_norm_v2 into category_norm only for currently UNKNOWN/NULL rows")
    args = ap.parse_args()

    db_path = Path(args.db).expanduser().resolve()
    out_report_json = Path(args.out_report_json).expanduser().resolve() if args.out_report_json else None
    out_audit_csv = Path(args.out_audit_csv).expanduser().resolve() if args.out_audit_csv else None

    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    try:
        _ensure_columns(con)

        rows = list(
            con.execute(
                """
                SELECT
                    id,
                    source_site,
                    title,
                    category_raw,
                    category_norm,
                    product_url,
                    model_page_url,
                    source_url,
                    materials_json,
                    raw_json
                FROM supplier_item
                """
            )
        )

        before_unknown = 0
        after_unknown = 0
        changed = 0
        changed_unknown = 0
        by_rule: Counter[str] = Counter()
        by_category: Counter[str] = Counter()
        audit_rows: list[dict[str, Any]] = []

        for row in rows:
            old = _norm(row["category_norm"])
            if not old or old == "unknown":
                before_unknown += 1

            category, confidence, rule = _infer_category(row)
            if not category:
                category = row["category_norm"]
                confidence = 0.0 if not category else 1.0
                rule = "unresolved" if not category else "existing.category_norm"

            con.execute(
                """
                UPDATE supplier_item
                SET category_norm_v2 = ?, category_confidence = ?, category_rule = ?
                WHERE id = ?
                """,
                (category, float(confidence), rule, row["id"]),
            )

            new_eff = _norm(category)
            if not new_eff or new_eff == "unknown":
                after_unknown += 1

            if new_eff != old:
                changed += 1
                by_rule[rule] += 1
                if new_eff:
                    by_category[new_eff] += 1
                if not old or old == "unknown":
                    changed_unknown += 1
                audit_rows.append(
                    {
                        "id": row["id"],
                        "source_site": row["source_site"],
                        "title": row["title"],
                        "category_raw": row["category_raw"],
                        "category_norm_old": row["category_norm"],
                        "category_norm_v2": category,
                        "category_confidence": round(float(confidence), 3),
                        "category_rule": rule,
                        "product_url": row["product_url"],
                    }
                )

        if args.writeback_unknown:
            con.execute(
                """
                UPDATE supplier_item
                SET category_norm = category_norm_v2
                WHERE (category_norm IS NULL OR TRIM(category_norm) = '' OR LOWER(category_norm) = 'unknown')
                  AND category_norm_v2 IS NOT NULL
                  AND TRIM(category_norm_v2) != ''
                """
            )

        con.commit()

        report = {
            "db_path": str(db_path),
            "before_unknown_count": before_unknown,
            "after_unknown_count": after_unknown,
            "changed_count": changed,
            "changed_unknown_count": changed_unknown,
            "top_applied_rules": [{"rule": name, "count": count} for name, count in by_rule.most_common(50)],
            "top_new_categories": [{"category": name, "count": count} for name, count in by_category.most_common(50)],
        }

        if out_report_json:
            out_report_json.parent.mkdir(parents=True, exist_ok=True)
            out_report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        if out_audit_csv:
            _write_audit_csv(out_audit_csv, audit_rows)

        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        con.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
