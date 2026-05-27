from __future__ import annotations

from copy import deepcopy
from typing import Any

from .object_priors import default_category, default_labels, default_style, normalize_subclass, subclass_priors
from .zone_templates import allowed_subclasses_for_zone, structural_subclasses_for_zone, zone_templates


PLANT_SUBCLASSES = {"plant", "potted_plant", "small_potted_plant", "hanging_planter", "plant_stand", "plant_pot"}
UNKNOWN_SUBCLASSES = {"", "object", "objects", "decor", "decoration", "accessory", "item", "thing", "misc", "unknown"}
COORDINATE_KEYS = {
    "x", "y", "z", "position", "location", "coordinates", "bbox", "aabb",
    "yaw", "yaw_deg", "rotation", "rotation_z_deg", "wall_id", "anchor",
}


def extract_theme_spec(prompt: str) -> dict[str, Any]:
    p = str(prompt or "").lower()
    avoid_plants = any(token in p for token in ("без растений", "не любит растения", "не любит цветов", "no plants", "without plants"))
    plant_heavy = any(token in p for token in ("много растений", "большое количество растений", "plant-heavy", "many plants", "lots of plants", "biophilic"))
    spec = {
        "schema": "room_theme_spec/v1",
        "theme_tags": [],
        "density": "medium",
        "must_have_objects": [],
        "avoid_objects": [],
        "max_thematic_objects": 0,
        "limits": {},
    }
    if plant_heavy and not avoid_plants:
        spec.update({
            "theme_tags": ["plants", "biophilic"],
            "density": "medium_high",
            "must_have_objects": ["potted_plant", "small_potted_plant", "hanging_planter", "plant_stand"],
            "max_thematic_objects": 8,
            "limits": {"max_floor_plants": 3, "max_tabletop_plants": 4, "max_hanging_plants": 1, "max_plant_stands": 1},
        })
    if avoid_plants:
        spec["avoid_objects"] = sorted(PLANT_SUBCLASSES)
        spec["theme_tags"] = [tag for tag in spec["theme_tags"] if tag not in {"plants", "biophilic"}]
    return spec


def _plant_like_text(text: str) -> bool:
    hay = text.lower()
    return any(token in hay for token in ("plant", "soil", "leaves", "orchid", "terracotta", "ceramic pot", "растен", "цвет", "листь", "орхиде", "кашпо", "горш"))


def _classify_from_text(item: dict[str, Any]) -> str | None:
    text = " ".join(str(item.get(k) or "") for k in ("subclass", "id_hint", "label_en", "label_ru", "name", "category", "color", "material")).lower()
    if not text.strip():
        return None  # pragma: no cover
    if _plant_like_text(text):
        if any(token in text for token in ("hanging", "подвес", "ceiling", "wall planter")):
            return "hanging_planter"
        if any(token in text for token in ("stand", "стойк", "подстав")):
            return "plant_stand"  # pragma: no cover
        if any(token in text for token in ("small", "mini", "desktop", "tabletop", "малень", "мини", "настоль")):
            return "small_potted_plant"
        if any(token in text for token in ("pot", "горш", "кашпо")):  # pragma: no cover
            return "potted_plant"  # pragma: no cover
        return "potted_plant"  # pragma: no cover
    mapping = {
        "bed": "bed", "кровать": "bed",
        "desk": "desk", "стол": "desk",
        "chair": "office_chair", "кресл": "office_chair",
        "lamp": "table_lamp", "ламп": "table_lamp",
        "book": "book", "книг": "book",
        "phone": "phone", "телефон": "phone",
        "poster": "wall_art", "постер": "wall_art",
        "rug": "rug", "ков": "rug",
        "shelf": "shelf", "стеллаж": "shelf",
        "wardrobe": "wardrobe", "шкаф": "wardrobe",
    }
    for token, subclass in mapping.items():
        if token in text:
            return subclass
    return None


def _normalize_item_subclass(item: dict[str, Any]) -> str | None:
    raw = normalize_subclass(item.get("subclass") or item.get("id_hint") or item.get("type") or "")
    if raw in UNKNOWN_SUBCLASSES:
        raw = ""
    if raw == "wall_shelf":
        return "shelf"
    if raw and raw in subclass_priors:
        if raw == "plant":
            return "potted_plant"  # pragma: no cover
        return raw
    classified = _classify_from_text(item)
    if classified:
        return normalize_subclass(classified)
    return None


def sanitize_zone_items(zone_id: str, zone_type: str, raw_items: dict[str, Any], theme_spec: dict[str, Any] | None = None, priors: dict[str, Any] | None = None) -> dict[str, Any]:
    theme_spec = dict(theme_spec or {})
    avoid = set(theme_spec.get("avoid_objects") or [])
    allowed = set(allowed_subclasses_for_zone(zone_type))
    raw_objects = raw_items.get("objects") if isinstance(raw_items.get("objects"), list) else raw_items.get("items", [])
    out: list[dict[str, Any]] = []
    warnings: list[str] = []
    schema_errors: list[str] = []

    for raw in raw_objects:
        if not isinstance(raw, dict):
            warnings.append("Dropped LLM item with non-object payload")
            schema_errors.append("item is not an object")
            continue
        item = {k: deepcopy(v) for k, v in raw.items() if k not in COORDINATE_KEYS}
        subclass = _normalize_item_subclass(item)
        if not subclass:
            warnings.append("Dropped LLM item with empty/unknown subclass")
            schema_errors.append("subclass is empty or not in allowed_subclasses")
            continue
        if subclass in avoid:
            warnings.append(f"Dropped avoided object subclass: {subclass}")  # pragma: no cover
            continue  # pragma: no cover
        if subclass not in allowed:
            warnings.append(f"Dropped LLM item not allowed in {zone_type}: {subclass}")
            schema_errors.append(f"subclass {subclass} is not in allowed_subclasses")
            continue
        label_ru, label_en = default_labels(subclass)
        style = default_style(subclass)
        item["subclass"] = subclass
        item["id_hint"] = item.get("id_hint") or subclass
        item["label_ru"] = str(item.get("label_ru") or label_ru)
        item["label_en"] = str(item.get("label_en") or label_en)
        item["category"] = default_category(subclass)
        item["role"] = str(item.get("role") or ("accessory" if subclass not in structural_subclasses_for_zone(zone_type) else "secondary"))
        item["color"] = str(item.get("color") or style.get("color") or "")
        item["material"] = str(item.get("material") or style.get("material") or "")
        item["source"] = str(raw_items.get("source") or item.get("source") or "llm")
        out.append(item)

    if "plants" in set(theme_spec.get("theme_tags") or []):
        out = _ensure_plant_theme_objects(zone_id, zone_type, out, theme_spec, warnings)
    out = _cap_zone_objects(zone_type, out, warnings)
    return {
        "schema": "zone_items/v1",
        "zone_id": zone_id,
        "objects": out,
        "source": raw_items.get("source") or "llm",
        "warnings": list(raw_items.get("warnings") or []) + warnings,
        "schema_errors": schema_errors,
        "allowed_subclasses": sorted(allowed),
    }


def _ensure_plant_theme_objects(zone_id: str, zone_type: str, objects: list[dict[str, Any]], theme_spec: dict[str, Any], warnings: list[str]) -> list[dict[str, Any]]:
    existing = [o.get("subclass") for o in objects]
    additions_by_zone = {
        "sleeping_zone": ["potted_plant", "small_potted_plant", "hanging_planter", "plant_stand"],
        "work_zone": ["small_potted_plant", "potted_plant"],
        "storage_zone": ["small_potted_plant", "potted_plant", "plant_stand"],
        "living_zone": ["potted_plant", "small_potted_plant", "hanging_planter", "plant_stand"],
    }
    allowed = set(allowed_subclasses_for_zone(zone_type))
    for subclass in additions_by_zone.get(zone_type, []):
        if subclass not in allowed or subclass in existing:
            continue
        label_ru, label_en = default_labels(subclass)
        style = default_style(subclass)
        objects.append({
            "id_hint": subclass,
            "label_ru": label_ru,
            "label_en": label_en,
            "subclass": subclass,
            "category": default_category(subclass),
            "role": "accessory",
            "quantity": 1,
            "importance": "optional",
            "source": "theme_template",
            **style,
        })
        existing.append(subclass)
    return objects


def _cap_zone_objects(zone_type: str, objects: list[dict[str, Any]], warnings: list[str]) -> list[dict[str, Any]]:
    max_by_zone = {"sleeping_zone": 10, "work_zone": 10, "storage_zone": 10}
    limit = max_by_zone.get(zone_type)
    if not limit or len(objects) <= limit:
        return objects
    structural = structural_subclasses_for_zone(zone_type)
    required = [o for o in objects if o.get("subclass") in structural or o.get("role") in {"main", "secondary"}]
    accessories = [o for o in objects if o not in required]
    keep = required + accessories[: max(0, limit - len(required))]
    if len(keep) < len(objects):
        warnings.append("Dropped excessive LLM accessories for zone object cap")
    return keep


def repair_semantic_objects(objects: list[dict[str, Any]], zones: list[dict[str, Any]], theme_spec: dict[str, Any] | None = None, max_total_objects: int = 32) -> dict[str, Any]:
    theme_spec = dict(theme_spec or {})
    warnings: list[str] = []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for obj in objects:
        subclass = normalize_subclass(obj.get("subclass") or "")
        if not subclass or subclass in UNKNOWN_SUBCLASSES or subclass not in subclass_priors:
            warnings.append("Dropped normalized object with empty/unknown subclass")
            continue
        if obj.get("id") in seen:
            warnings.append(f"Dropped duplicate object id: {obj.get('id')}")
            continue
        seen.add(str(obj.get("id")))
        label_ru, label_en = default_labels(subclass)
        obj = dict(obj)
        obj["subclass"] = subclass
        obj["label_ru"] = str(obj.get("label_ru") or label_ru)
        obj["label_en"] = str(obj.get("label_en") or label_en)
        obj["category"] = default_category(subclass)
        out.append(obj)
    out = _cap_plants(out, theme_spec, warnings)
    if len(out) > max_total_objects:
        kept = [o for o in out if o.get("role") in {"main", "secondary"}]
        optional = [o for o in out if o not in kept]
        out = kept + optional[: max(0, max_total_objects - len(kept))]
        warnings.append("Dropped optional objects over max_total_objects")
    return {"objects": out, "warnings": warnings}


def _cap_plants(objects: list[dict[str, Any]], theme_spec: dict[str, Any], warnings: list[str]) -> list[dict[str, Any]]:
    if "plants" not in set(theme_spec.get("theme_tags") or []) and not set(theme_spec.get("avoid_objects") or []):
        return objects  # pragma: no cover
    avoid = set(theme_spec.get("avoid_objects") or [])
    if avoid:
        before = len(objects)
        objects = [o for o in objects if o.get("subclass") not in avoid]
        if len(objects) < before:
            warnings.append("Dropped avoided plant objects")
    limits = {"max_floor_plants": 3, "max_tabletop_plants": 4, "max_hanging_plants": 1, "max_plant_stands": 1}
    limits.update(theme_spec.get("limits") or {})
    max_total = int(theme_spec.get("max_thematic_objects") or 8)
    buckets = {
        "max_floor_plants": {"potted_plant", "plant_pot"},
        "max_tabletop_plants": {"small_potted_plant"},
        "max_hanging_plants": {"hanging_planter"},
        "max_plant_stands": {"plant_stand"},
    }
    kept: list[dict[str, Any]] = []
    dropped = 0
    counts = {k: 0 for k in limits}
    total_plants = 0
    for obj in objects:
        subclass = str(obj.get("subclass") or "")
        if subclass not in PLANT_SUBCLASSES:
            kept.append(obj)
            continue
        bucket = next((name for name, members in buckets.items() if subclass in members), "max_floor_plants")
        if total_plants >= max_total or counts[bucket] >= int(limits[bucket]):
            dropped += 1
            continue
        counts[bucket] += 1
        total_plants += 1
        kept.append(obj)
    if dropped:
        warnings.append("Dropped excessive thematic plant objects")
    return kept
