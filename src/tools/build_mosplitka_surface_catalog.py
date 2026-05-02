#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build a text-only normalized surface-material catalog from Mosplitka cards.

The input catalog is intentionally raw and verbose.  This tool keeps the source
facts, removes duplicated property keys, and derives only conservative fields
from product text/properties.  VLM fields are reserved but left empty.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOM_MAP = {
    "кухн": "kitchen",
    "ванн": "bathroom",
    "душ": "bathroom",
    "гостин": "living_room",
    "спаль": "bedroom",
    "прихож": "hallway",
    "коридор": "hallway",
    "детск": "children",
    "балкон": "balcony",
    "террас": "terrace",
    "улиц": "outdoor",
    "фасад": "outdoor",
}

STYLE_MAP = {
    "современ": "contemporary",
    "класс": "classic",
    "лофт": "loft",
    "минимал": "minimalism",
    "скандинав": "scandinavian",
    "прованс": "provence",
    "ретро": "retro",
    "винтаж": "vintage",
    "барок": "baroque",
    "арт": "art_deco",
    "хай-тек": "high_tech",
    "high-tech": "high_tech",
    "кантр": "country",
    "эко": "eco",
}

PATTERN_MAP = [
    ("под дерево", "wood"),
    ("дерев", "wood"),
    ("дуб", "wood"),
    ("вяз", "wood"),
    ("орех", "wood"),
    ("мрамор", "marble"),
    ("под камень", "stone"),
    ("камень", "stone"),
    ("оникс", "onyx"),
    ("бетон", "concrete"),
    ("цемент", "concrete"),
    ("террац", "terrazzo"),
    ("кирпич", "brick"),
    ("геометр", "geometric"),
    ("орнамент", "ornament"),
    ("венз", "ornament"),
    ("дамаск", "damask"),
    ("цвет", "floral"),
    ("лист", "botanical"),
    ("растен", "botanical"),
    ("моза", "mosaic"),
    ("моноколор", "plain"),
    ("однотон", "plain"),
]

COLOR_MAP = [
    ("бел", "white"),
    ("молоч", "white"),
    ("айвори", "ivory"),
    ("слонов", "ivory"),
    ("сер", "gray"),
    ("графит", "graphite"),
    ("антрацит", "anthracite"),
    ("беж", "beige"),
    ("крем", "cream"),
    ("песоч", "sand"),
    ("корич", "brown"),
    ("каштан", "brown"),
    ("террак", "terracotta"),
    ("кирпич", "terracotta"),
    ("красн", "red"),
    ("роз", "pink"),
    ("желт", "yellow"),
    ("золот", "gold"),
    ("син", "blue"),
    ("голуб", "blue"),
    ("бирюз", "turquoise"),
    ("зелен", "green"),
    ("олив", "olive"),
    ("черн", "black"),
]


def norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("\xa0", " ")).strip()


def lower(value: Any) -> str:
    return norm(value).lower().replace("ё", "е")


def parse_float(value: Any) -> float | None:
    text = norm(value).replace(",", ".")
    match = re.search(r"\d+(?:\.\d+)?", text.replace(" ", ""))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def prop(props: dict[str, Any], *names: str) -> str:
    lowered = {lower(k): v for k, v in props.items()}
    for name in names:
        value = lowered.get(lower(name))
        if value is not None:
            return norm(value)
    return ""


def clean_properties(raw: dict[str, Any]) -> dict[str, str]:
    cleaned: dict[str, str] = {}
    feature_values: list[str] = []
    for key, value in (raw or {}).items():
        value_text = norm(value)
        if not value_text:
            continue
        key_text = norm(key)
        if ">" in key_text:
            continue
        if key_text == ",":
            feature_values.append(value_text)
            continue
        cleaned.setdefault(key_text, value_text)
    if feature_values:
        existing = cleaned.get("Особенности", "")
        parts = [p for p in [existing, *feature_values] if p]
        cleaned["Особенности"] = ", ".join(dict.fromkeys(parts))
    return cleaned


def normalize_material_type(text: str) -> str:
    t = lower(text)
    if "керамогранит" in t or "керамический гранит" in t:
        return "porcelain_tile"
    if "мозаик" in t:
        return "mosaic"
    if "плитка" in t or "кафель" in t:
        return "ceramic_tile"
    if "бордюр" in t:
        return "border_tile"
    if "декор" in t or "вставка" in t:
        return "decor_tile"
    if "ступен" in t:
        return "stair_tile"
    if "плинтус" in t:
        return "plinth_tile"
    return "unknown_surface_material"


def normalize_role(name: str, props: dict[str, Any]) -> str:
    text = lower(" ".join([name, prop(props, "Элементы плитки"), prop(props, "Тип")]))
    checks = [
        ("подступен", "riser"),
        ("ступен", "step"),
        ("плинтус", "plinth"),
        ("бордюр", "border"),
        ("карандаш", "border"),
        ("декор", "decor"),
        ("вставка", "insert"),
        ("панно", "panel"),
        ("моза", "mosaic"),
        ("базовая", "base_tile"),
    ]
    for needle, role in checks:
        if needle in text:
            return role
    if normalize_material_type(text) in {"porcelain_tile", "ceramic_tile"}:
        return "base_tile"
    return "unknown"


def normalize_pattern(text: str) -> str | None:
    t = lower(text)
    for needle, value in PATTERN_MAP:
        if needle in t:
            return value
    return None


def normalize_color(text: str) -> str | None:
    t = lower(text)
    for needle, value in COLOR_MAP:
        if needle in t:
            return value
    return None


def normalize_tone(text: str) -> str | None:
    t = lower(text)
    if any(x in t for x in ["черн", "темн", "графит", "антрацит"]):
        return "dark"
    if any(x in t for x in ["бел", "светл", "молоч", "айвори", "крем"]):
        return "light"
    if any(x in t for x in ["беж", "песоч"]):
        return "warm_light"
    if any(x in t for x in ["корич", "террак", "кирпич"]):
        return "warm_dark"
    if any(x in t for x in ["сер", "цемент", "бетон"]):
        return "neutral"
    return None


def normalize_finish(text: str) -> str | None:
    t = lower(text)
    if "мат" in t:
        return "matte"
    if "глян" in t:
        return "glossy"
    if "полир" in t or "люкс" in t:
        return "polished"
    if "лаппат" in t:
        return "lappato"
    if "рельеф" in t or "структур" in t:
        return "structured"
    return None


def normalize_edge(text: str) -> str | None:
    t = lower(text)
    if "необрез" in t:
        return "non_rectified"
    if "обрез" in t or "рект" in t or "rett" in t:
        return "rectified"
    return None


def normalize_rooms(*texts: str) -> list[str]:
    joined = lower(" ".join(texts))
    rooms = {room for needle, room in ROOM_MAP.items() if needle in joined}
    return sorted(rooms)


def normalize_styles(*texts: str) -> list[str]:
    joined = lower(" ".join(texts))
    styles = {style for needle, style in STYLE_MAP.items() if needle in joined}
    return sorted(styles)


def parse_format(value: str) -> dict[str, Any]:
    text = lower(value).replace("x", "х")
    nums = [float(x.replace(",", ".")) for x in re.findall(r"\d+(?:[,.]\d+)?", text)]
    width = nums[0] if len(nums) >= 1 else None
    height = nums[1] if len(nums) >= 2 else None
    if width and height:
        mx = max(width, height)
        if mx >= 120:
            size_class = "large_slab"
        elif mx >= 80:
            size_class = "large"
        elif mx >= 45:
            size_class = "medium"
        else:
            size_class = "small"
    else:
        size_class = None
    return {
        "tile_width_cm": width,
        "tile_height_cm": height,
        "tile_format": value or None,
        "tile_aspect_ratio": round(max(width, height) / min(width, height), 3) if width and height else None,
        "tile_size_class": size_class,
    }


def bool_from_text(text: str, positives: list[str]) -> bool:
    t = lower(text)
    return any(needle in t for needle in positives)


def normalize_availability(value: str) -> str:
    t = lower(value)
    if "instock" in t or "в наличии" in t:
        return "in_stock"
    if "preorder" in t or "под заказ" in t:
        return "preorder"
    if "outofstock" in t or "нет" in t:
        return "out_of_stock"
    return "unknown"


def selection_flags(role: str, usage: str, material_type: str) -> tuple[bool, bool, bool, str]:
    usage_l = lower(usage)
    accent_only = role in {"decor", "insert", "border", "panel", "mosaic"}
    blocked = role in {"step", "riser", "plinth"}

    floor_by_usage = "пол" in usage_l
    wall_by_usage = "стен" in usage_l
    tile_main = material_type in {"porcelain_tile", "ceramic_tile"}

    floor_ok = not blocked and not accent_only and tile_main and floor_by_usage
    wall_ok = not blocked and tile_main and wall_by_usage
    if role == "mosaic":
        wall_ok = wall_by_usage
    if accent_only:
        wall_ok = False

    reason = ""
    if blocked:
        reason = f"not_surface_covering:{role}"
    elif accent_only:
        reason = f"accent_only:{role}"
    elif not floor_ok and not wall_ok:
        reason = "usage_not_explicit_for_floor_or_wall"
    return floor_ok, wall_ok, accent_only, reason


def build_record(item: dict[str, Any]) -> dict[str, Any]:
    raw_props = item.get("properties") or {}
    props = clean_properties(raw_props)
    name = norm(item.get("name"))
    material_type_text = " ".join([
        name,
        prop(props, "Тип"),
        prop(props, "Тип материала"),
        prop(props, "Элементы плитки"),
    ])
    material_type = normalize_material_type(material_type_text)
    role = normalize_role(name, props)
    usage = prop(props, "Назначение")
    rooms = normalize_rooms(prop(props, "Помещение"), item.get("recommendations_text", ""))
    style_text = prop(props, "Стиль")
    styles = set(normalize_styles(style_text, item.get("recommendations_text", "")))
    pattern_source = " ".join([prop(props, "Рисунок"), name, prop(props, "Элементы плитки")])
    pattern = normalize_pattern(pattern_source)
    if pattern in {"marble", "concrete", "stone", "wood", "ornament", "damask"}:
        styles.update({"classic"} if pattern in {"marble", "ornament", "damask"} else [])
        styles.update({"loft", "minimalism"} if pattern == "concrete" else [])
        styles.update({"eco", "scandinavian"} if pattern == "wood" else [])
    color_source = " ".join([prop(props, "Цвет точно"), prop(props, "Цвет"), name])
    base_color = normalize_color(color_source)
    finish = normalize_finish(" ".join([prop(props, "Поверхность"), name]))
    edge = normalize_edge(" ".join([prop(props, "Обработка края"), name]))
    features = prop(props, "Особенности")
    fmt = parse_format(prop(props, "Формат, см"))
    floor_ok, wall_ok, accent_only, exclude_reason = selection_flags(role, usage, material_type)
    material_image = item.get("material_image") or {}
    selected_path = norm(material_image.get("selected_path"))
    local_path = norm(material_image.get("local_path"))
    image_path = selected_path or local_path

    text_facts = {
        "type": prop(props, "Тип"),
        "material_type": prop(props, "Тип материала"),
        "role": prop(props, "Элементы плитки"),
        "usage": usage,
        "rooms": prop(props, "Помещение"),
        "style": style_text,
        "pattern": prop(props, "Рисунок"),
        "color": prop(props, "Цвет"),
        "precise_color": prop(props, "Цвет точно"),
        "surface": prop(props, "Поверхность"),
        "features": features,
    }
    description_parts = [
        text_facts["precise_color"] or text_facts["color"],
        text_facts["pattern"],
        text_facts["surface"],
        text_facts["style"],
    ]
    text_description = ", ".join(p for p in description_parts if p)

    return {
        "version": "surface_material.v1",
        "source": "mosplitka",
        "url": item.get("url") or item.get("final_url") or "",
        "name": name,
        "sku": norm(item.get("sku")),
        "brand": norm(item.get("brand")),
        "collection": norm(item.get("collection")),
        "price": parse_float(item.get("price")),
        "price_currency": norm(item.get("price_currency")) or "RUB",
        "availability": normalize_availability(item.get("availability", "")),
        "normalized": {
            "material_type": material_type,
            "material_role": role,
            "is_selectable_floor": floor_ok,
            "is_selectable_wall": wall_ok,
            "is_accent_only": accent_only,
            "exclude_reason": exclude_reason,
            "usage": sorted({x for x in ["floor" if "пол" in lower(usage) else "", "wall" if "стен" in lower(usage) else ""] if x}),
            "rooms": rooms,
            "style_tags": sorted(styles),
            "visual_pattern": pattern,
            "base_color": base_color,
            "precise_color_ru": text_facts["precise_color"] or None,
            "tone": normalize_tone(color_source),
            "surface_finish": finish,
            "edge": edge,
            "anti_slip": bool_from_text(features, ["противоскольз", "anti-slip", "grip"]),
            "frost_resistant": bool_from_text(features, ["морозостой"]),
            "rectified": edge == "rectified",
            "thickness_mm": parse_float(prop(props, "Толщина, мм")),
            **fmt,
        },
        "text_facts": text_facts,
        "text_description_ru": text_description,
        "material_image": {
            "path": image_path,
            "source_path": norm(material_image.get("image_file")),
            "image_url": norm(material_image.get("image_url")),
            "product_dir": norm(material_image.get("product_dir")),
            "image_index": material_image.get("image_index"),
            "width": material_image.get("width"),
            "height": material_image.get("height"),
            "aspect": material_image.get("aspect"),
            "status": norm(material_image.get("status")),
            "reason": norm(material_image.get("reason")),
        },
        "vlm": {
            "status": "not_run",
            "model": "",
            "description_ru": "",
            "color": "",
            "pattern": "",
            "style": "",
        },
        "raw_properties": props,
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=False) + "\n")


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "name",
        "sku",
        "url",
        "brand",
        "collection",
        "price",
        "availability",
        "material_type",
        "material_role",
        "is_selectable_floor",
        "is_selectable_wall",
        "is_accent_only",
        "exclude_reason",
        "usage",
        "rooms",
        "style_tags",
        "visual_pattern",
        "base_color",
        "precise_color_ru",
        "tone",
        "surface_finish",
        "anti_slip",
        "frost_resistant",
        "tile_format",
        "thickness_mm",
        "material_image_path",
        "text_description_ru",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            n = record["normalized"]
            row = {
                "name": record["name"],
                "sku": record["sku"],
                "url": record["url"],
                "brand": record["brand"],
                "collection": record["collection"],
                "price": record["price"],
                "availability": record["availability"],
                "material_type": n["material_type"],
                "material_role": n["material_role"],
                "is_selectable_floor": n["is_selectable_floor"],
                "is_selectable_wall": n["is_selectable_wall"],
                "is_accent_only": n["is_accent_only"],
                "exclude_reason": n["exclude_reason"],
                "usage": ",".join(n["usage"]),
                "rooms": ",".join(n["rooms"]),
                "style_tags": ",".join(n["style_tags"]),
                "visual_pattern": n["visual_pattern"],
                "base_color": n["base_color"],
                "precise_color_ru": n["precise_color_ru"],
                "tone": n["tone"],
                "surface_finish": n["surface_finish"],
                "anti_slip": n["anti_slip"],
                "frost_resistant": n["frost_resistant"],
                "tile_format": n["tile_format"],
                "thickness_mm": n["thickness_mm"],
                "material_image_path": record["material_image"]["path"],
                "text_description_ru": record["text_description_ru"],
            }
            writer.writerow(row)


def write_analytics(path: Path, records: list[dict[str, Any]]) -> None:
    counters = {
        "total": len(records),
        "with_image": sum(1 for r in records if r["material_image"]["path"]),
        "selectable_floor": sum(1 for r in records if r["normalized"]["is_selectable_floor"]),
        "selectable_wall": sum(1 for r in records if r["normalized"]["is_selectable_wall"]),
        "accent_only": sum(1 for r in records if r["normalized"]["is_accent_only"]),
        "material_type": Counter(r["normalized"]["material_type"] for r in records),
        "material_role": Counter(r["normalized"]["material_role"] for r in records),
        "visual_pattern": Counter(r["normalized"]["visual_pattern"] or "unknown" for r in records),
        "base_color": Counter(r["normalized"]["base_color"] or "unknown" for r in records),
        "surface_finish": Counter(r["normalized"]["surface_finish"] or "unknown" for r in records),
        "exclude_reason": Counter(r["normalized"]["exclude_reason"] or "selectable_or_partial" for r in records),
    }
    serializable = {
        key: (dict(value.most_common()) if isinstance(value, Counter) else value)
        for key, value in counters.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build normalized Mosplitka surface-material catalog from text fields.")
    parser.add_argument("--input", default="data/floor_materials/mosplitka/mosplitka_catalog_full.jsonl")
    parser.add_argument("--out-jsonl", default="data/floor_materials/mosplitka/mosplitka_surface_materials.jsonl")
    parser.add_argument("--out-csv", default="data/floor_materials/mosplitka/mosplitka_surface_materials.csv")
    parser.add_argument("--analytics", default="data/floor_materials/mosplitka/mosplitka_surface_materials_analytics.json")
    args = parser.parse_args()

    records: list[dict[str, Any]] = []
    with Path(args.input).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(build_record(json.loads(line)))

    write_jsonl(Path(args.out_jsonl), records)
    write_csv(Path(args.out_csv), records)
    write_analytics(Path(args.analytics), records)
    print(f"records: {len(records)}")
    print(f"jsonl: {args.out_jsonl}")
    print(f"csv: {args.out_csv}")
    print(f"analytics: {args.analytics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
