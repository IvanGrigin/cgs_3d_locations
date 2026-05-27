from __future__ import annotations

from typing import Any

from .kitchen_text_features import (
    contains_any,
    detect_color_families,
    detect_finish,
    detect_pattern,
    detect_style_tags,
    detect_tone,
    extract_size_triplet_mm,
    first_present,
    normalize_text,
    parse_json_maybe,
    safe_float,
    safe_int,
)


def infer_kitchen_role(raw: dict[str, Any]) -> str:
    props = parse_json_maybe(raw.get("properties_json"), {})
    raw_properties = raw.get("raw_properties") if isinstance(raw.get("raw_properties"), dict) else {}
    text_facts = raw.get("text_facts") if isinstance(raw.get("text_facts"), dict) else {}
    normalized = raw.get("normalized") if isinstance(raw.get("normalized"), dict) else {}
    text = " ".join(
        normalize_text(x)
        for x in (
            raw.get("name"),
            raw.get("description"),
            raw.get("categories"),
            raw.get("breadcrumbs"),
            raw.get("material_type"),
            props.get("Категория"),
            props.get("Тип декора"),
            props.get("Материал"),
            raw_properties.get("Категория"),
            raw_properties.get("Материал"),
            text_facts.get("type"),
            text_facts.get("material_type"),
            text_facts.get("role"),
            normalized.get("material_type"),
        )
        if x is not None
    )
    category = normalize_text(props.get("Категория") or raw_properties.get("Категория"))
    material_type = normalize_text(raw.get("material_type") or normalized.get("material_type"))

    if "планка соединительная" in category or "соединительная планка" in text:
        return "joint_profile"
    if "планка торцевая" in category or "торцевая планка" in text:
        return "end_profile"  # pragma: no cover
    if "планка угловая" in category or "угловая планка" in text:
        return "corner_profile"  # pragma: no cover
    if "решетка" in category or "решетка" in text or "решётка" in text:
        return "ventilation_grille"
    if "плинтус" in category or "плинтус" in text:
        return "countertop_wall_plinth"  # pragma: no cover
    if "стеновая панель" in category or "стеновая панель" in text or "фартук" in text:
        return "backsplash_panel"  # pragma: no cover
    if "столешница" in category or "столешница" in text or "countertop" in material_type:
        return "premium_countertop_slab" if "perfectsense" in text else "countertop_slab"
    if "фасадные полотна" in category or "фасад" in text or material_type == "facade_panel":
        return "facade_sheet"  # pragma: no cover
    if category.startswith("мдф") or category.startswith("хдф") or material_type == "mdf_panel":
        return "board_sheet"  # pragma: no cover
    if "кромка, акцентная" in category or ("акцентная" in text and "кром" in text):
        return "accent_edge_band"  # pragma: no cover
    if "кромка" in category or "edge_banding" in material_type or "кромоч" in text:
        return "edge_band"
    return "unknown"  # pragma: no cover


def extract_dimensions_mm(raw: dict[str, Any]) -> dict[str, int | None]:
    props = parse_json_maybe(raw.get("properties_json"), {})
    raw_properties = raw.get("raw_properties") if isinstance(raw.get("raw_properties"), dict) else {}
    props_merged = {**raw_properties, **props}
    normalized = raw.get("normalized") if isinstance(raw.get("normalized"), dict) else {}
    length = safe_int(first_present(props_merged, ["Длина, мм", "Длина", "Размер длина", "L, мм"])) or safe_int(normalized.get("length_mm"))
    width = safe_int(first_present(props_merged, ["Ширина, мм", "Ширина", "Размер ширина", "W, мм"])) or safe_int(normalized.get("width_mm"))
    thickness = safe_int(first_present(props_merged, ["Толщина, мм", "Толщина", "T, мм"])) or safe_int(normalized.get("thickness_mm"))
    a, b, c = extract_size_triplet_mm(raw.get("name"), raw.get("description"), raw.get("categories"), props_merged, props_merged.get("Формат, мм"), props_merged.get("Размер, мм"))
    if a and b:
        values = sorted([v for v in (a, b, c) if v], reverse=True)
        length = length or values[0]
        width = width or values[1]
        if thickness is None and len(values) >= 3:
            thickness = values[-1]
    return {"length_mm": length, "width_mm": width, "thickness_mm": thickness}


def infer_unit(raw: dict[str, Any], role: str) -> str:
    props = parse_json_maybe(raw.get("properties_json"), {})
    unit = normalize_text(props.get("Единица измерения") or raw.get("unit"))
    if "м.п" in unit or "пог" in unit or role in {"edge_band", "accent_edge_band"}:
        return "m"
    if "лист" in unit:
        return "sheet"  # pragma: no cover
    return "piece"


def normalize_material_record(raw: dict[str, Any]) -> dict[str, Any]:
    props = parse_json_maybe(raw.get("properties_json"), {})
    raw_properties = raw.get("raw_properties") if isinstance(raw.get("raw_properties"), dict) else {}
    props_merged = {**raw_properties, **props}
    normalized = raw.get("normalized") if isinstance(raw.get("normalized"), dict) else {}
    text_facts = raw.get("text_facts") if isinstance(raw.get("text_facts"), dict) else {}
    role = infer_kitchen_role(raw)
    name = raw.get("name") or ""
    sku = raw.get("sku") or raw.get("id") or name
    brand = raw.get("brand") or props_merged.get("Производитель") or raw.get("source") or "unknown"
    description = raw.get("description") or raw.get("text_description_ru") or ""
    role_and_pattern_text = " ".join(str(x) for x in (name, sku, description, props_merged, text_facts, normalized) if x)
    color_text = " ".join(str(x) for x in (name, sku, text_facts.get("color"), normalized.get("precise_color_ru")) if x)
    colors = detect_color_families(name, sku) or detect_color_families(color_text)
    if not colors and normalized.get("base_color"):
        colors = {normalized.get("base_color")}  # pragma: no cover
    colors.discard("neutral")
    local_image = raw.get("local_image") or raw.get("local_image_path")
    material_image = raw.get("material_image") if isinstance(raw.get("material_image"), dict) else {}
    if not local_image:
        local_image = material_image.get("source_path") or material_image.get("path")

    if not local_image:
        image_paths = parse_json_maybe(raw.get("image_paths"), [])
        local_image = image_paths[0] if isinstance(image_paths, list) and image_paths else None

    if not local_image:
        local_paths = parse_json_maybe(raw.get("local_image_paths_json"), [])
        local_image = local_paths[0] if isinstance(local_paths, list) and local_paths else None

    image_url = raw.get("image_url") or material_image.get("image_url")
    if not image_url:
        images = parse_json_maybe(raw.get("images_json"), [])
        image_url = images[0] if isinstance(images, list) and images else None
    availability_raw = normalize_text(raw.get("availability") or "")
    availability = "in_stock" if contains_any(availability_raw, ["в наличии", "in_stock", "available"]) else raw.get("availability") or "unknown"
    return {
        "source": raw.get("source") or "basisrf",
        "url": raw.get("url") or raw.get("final_url"),
        "name": name,
        "sku": sku,
        "brand": brand,
        "price": safe_float(raw.get("price"), None),
        "price_currency": raw.get("price_currency") or "RUB",
        "availability": availability,
        "material_type": raw.get("material_type") or normalized.get("material_type"),
        "kitchen_role": role,
        "unit": infer_unit(raw, role),
        "usage": parse_json_maybe(raw.get("usage"), normalized.get("usage") or []),
        "dimensions": extract_dimensions_mm(raw),
        "visual": {
            "base_colors": sorted(colors or {"neutral"}),
            "tone": detect_tone(colors, color_text) if colors else (normalized.get("tone") or "neutral"),
            "pattern": normalized.get("visual_pattern") or detect_pattern(role_and_pattern_text),
            "finish": normalized.get("surface_finish") or detect_finish(role_and_pattern_text),
            "style_tags": sorted(set(normalized.get("style_tags") or []) | set(detect_style_tags(role_and_pattern_text))),
        },
        "flags": {
            "is_moisture_resistant": contains_any(role_and_pattern_text, ["влагостой", "moisture"]),
            "is_premium": contains_any(role_and_pattern_text, ["perfectsense", "premium", "gloss/matt"]),
            "is_accent_only": role == "accent_edge_band" or bool(normalized.get("is_accent_only")),
        },
        "image_url": image_url,
        "local_image": local_image,
        "raw_category": props_merged.get("Категория"),
        "raw": raw,
    }
