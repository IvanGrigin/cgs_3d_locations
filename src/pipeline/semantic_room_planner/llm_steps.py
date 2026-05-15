from __future__ import annotations

from typing import Any

from .llm_client import call_json_llm
from .zone_templates import ZONE_TYPES, zone_templates
from .object_priors import default_category, default_labels, default_style
from .semantic_sanitizer import extract_theme_spec, sanitize_zone_items


NO_GEOMETRY = "Do not output exact coordinates, x/y/z placement, yaw, rotation, wall_id, asset paths, catalog ids, or bounding boxes. You are not the geometry solver."
STRICT = "Return strict JSON object only. No markdown. No comments. No prose outside JSON."


def _settings(llm_settings: dict[str, Any], step: str) -> dict[str, Any]:
    s = dict(llm_settings or {})
    s.setdefault("provider", "none")
    s["provider"] = str(s.get("provider") or "none").strip().lower()
    s.pop("use_llm_catalog_queries", None)
    s.pop("llm_catalog_max_objects", None)
    s.setdefault("step_name", step)
    return s


def _provider_is_none(llm_settings: dict[str, Any] | None) -> bool:
    return str((llm_settings or {}).get("provider") or "none").strip().lower() == "none"


def _extract_theme(prompt: str) -> str | None:
    p = str(prompt or "").lower()
    if any(token in p for token in ("мальчик", "машинк", "авто", "автомоб", "car", "racing")):
        return "cars/racing"
    return None


def _prompt_preferences(prompt: str) -> dict[str, Any]:
    p = str(prompt or "").lower()
    avoid_plants = any(token in p for token in ("без растений", "не любит растения", "не любит цветов", "no plants", "without plants"))
    single_bed = any(token in p for token in ("односпаль", "single bed", "twin bed"))
    double_bed = any(token in p for token in ("двуспаль", "double bed", "queen bed"))
    theme_spec = extract_theme_spec(prompt)
    return {
        "theme": _extract_theme(prompt),
        "theme_spec": theme_spec,
        "avoid_subclasses": list(theme_spec.get("avoid_objects") or (["plant"] if avoid_plants else [])),
        "bed_preference": "single" if single_bed else "double" if double_bed else None,
    }


def build_room_intent_prompt(input_state: dict[str, Any]) -> list[dict[str, str]]:
    return [{"role": "system", "content": f"{STRICT} {NO_GEOMETRY}"}, {"role": "user", "content": f"Build room_intent/v1 from prompt, geometry summary and assumptions. Use household semantics only. Capture explicit user preferences and avoid-list as text, but do not enumerate final objects here; final object choice happens in zone_items.\nJSON input:\n{input_state}"}]


def build_zones_prompt(input_state: dict[str, Any]) -> list[dict[str, str]]:
    return [{"role": "system", "content": f"{STRICT} {NO_GEOMETRY} Zone type enum: {sorted(ZONE_TYPES)}"}, {"role": "user", "content": f"Add room_zones/v1. Include priorities, purpose, desired_area_share, placement_preferences, but no coordinates.\nJSON input:\n{input_state}"}]


def build_zone_items_prompt(input_state: dict[str, Any], zone: dict[str, Any]) -> list[dict[str, str]]:
    from .zone_templates import allowed_subclasses_for_zone
    allowed = sorted(allowed_subclasses_for_zone(str(zone.get("type") or "")))
    theme_spec = input_state.get("theme_spec") or extract_theme_spec(input_state.get("prompt") or "")
    max_by_zone = {"sleeping_zone": 10, "work_zone": 10, "storage_zone": 10}
    max_items = max_by_zone.get(str(zone.get("type") or ""), 12)
    plant_note = ""
    if "plants" in set(theme_spec.get("theme_tags") or []):
        plant_note = "For plant-heavy prompt, plant objects must be only from potted_plant, small_potted_plant, hanging_planter, plant_stand; do not invent generic object/decor subclasses."
    return [{"role": "system", "content": f"{STRICT} {NO_GEOMETRY} Code owns required large furniture, fixtures, dimensions, coordinates, walkways, doors/windows, support and collision checks. Your job in this step is semantic enrichment: choose varied small/medium accessories, plants, textiles, wall decor, shelf/table/counter items, and theme-specific details for this one zone. Subclass is mandatory and must be exactly one value from allowed_subclasses. Never output empty subclass, generic object, decor, item, or unknown. Cap this zone to at most {max_items} objects. {plant_note} If the prompt says no plants or dislikes plants, do not output plant-related subclasses. Every accessory must have enough semantic detail for code to attach it to a support surface later. Output household object semantics only."}, {"role": "user", "content": f"Return zone_items/v1 for this one zone only. Prefer role='accessory' for LLM-added objects. Use prompt, room_intent, zone purpose, explicit avoid/preferences, and theme_spec to choose objects. Do not generate the whole room; only this zone.\nAllowed_subclasses: {allowed}\nTheme_spec: {theme_spec}\nZone: {zone}\nState: {input_state}"}]


def build_zone_relations_prompt(input_state: dict[str, Any], zone: dict[str, Any], zone_items: dict[str, Any]) -> list[dict[str, str]]:
    return [{"role": "system", "content": f"{STRICT} {NO_GEOMETRY} Relation enum: on_top_of, inside, under, near, next_to, left_of, right_of, in_front_of, behind, faces, against_wall, mounted_on_wall, centered_on, aligned_with, grouped_with, around, above, below, visible_from. Every accessory needs support/context relation; every chair needs faces relation; sofa faces tv/coffee_table/room_center; pillow/blanket on bed; table_lamp on desk/nightstand; wardrobe/dresser/shelf against wall."}, {"role": "user", "content": f"Return zone_relations/v1 for this one zone only. Resolve by subclass, not object ids.\nZone: {zone}\nItems: {zone_items}\nState: {input_state}"}]


def build_catalog_queries_prompt(obj: dict[str, Any]) -> list[dict[str, str]]:
    return [{"role": "system", "content": f"{STRICT} Generate text search queries only. No coordinates, no asset ids."}, {"role": "user", "content": f"Return catalog query object for this normalized object: {obj}"}]


def _fallback_intent(state: dict[str, Any]) -> dict[str, Any]:
    prompt = str(state.get("prompt") or "").lower()
    room_type = str(((state.get("input") or {}).get("room") or {}).get("type_hint") or "room")
    if "спаль" in prompt or "bed" in prompt:
        room_type = "bedroom"
    elif "кух" in prompt or "kitchen" in prompt:
        room_type = "kitchen"
    elif "гостин" in prompt or "living" in prompt:
        room_type = "living_room"
    elif "ванн" in prompt or "bathroom" in prompt:
        room_type = "bathroom"
    elif "туалет" in prompt or "toilet" in prompt or "wc" in prompt:
        room_type = "toilet"
    elif "столов" in prompt or "обеден" in prompt or "dining" in prompt:
        room_type = "dining_room"
    elif "кабинет" in prompt or "office" in prompt:
        room_type = "office"
    functions = ["storage", "small decor"]
    if room_type == "bedroom" or "кровать" in prompt:
        functions.insert(0, "sleeping")
    if "рабоч" in prompt or "student" in prompt or "desk" in prompt:
        functions.append("working")
    if "кух" in prompt or "kitchen" in prompt:
        functions.append("cooking")
    if "столов" in prompt or "обеден" in prompt or "dining" in prompt:
        functions.append("dining")
    if "гостин" in prompt or "living" in prompt:
        functions.append("living")
    if "ванн" in prompt or "bathroom" in prompt:
        functions.append("bathing")
    if "туалет" in prompt or "toilet" in prompt or "wc" in prompt:
        functions.append("toilet")
    target_user = "student" if "студент" in prompt or "student" in prompt else "boy" if "мальчик" in prompt else "resident"
    intent = {"schema": "room_intent/v1", "room_type": room_type, "target_user": target_user, "style": "modern cozy minimalism", "density": "medium_high" if "уют" in prompt else "medium", "palette": {"base": ["warm white", "light oak", "beige"], "accent": ["black", "muted green"]}, "required_functions": list(dict.fromkeys(functions)), "avoid": ["overcrowded layout"], "source": "fallback_template"}
    prefs = _prompt_preferences(prompt)
    if prefs["avoid_subclasses"]:
        intent["avoid"].extend([f"avoid {sc}" for sc in prefs["avoid_subclasses"]])
    if prefs["bed_preference"]:
        intent["bed_preference"] = prefs["bed_preference"]
    return intent


def _fallback_zones(state: dict[str, Any]) -> dict[str, Any]:
    funcs = set((state.get("room_intent") or {}).get("required_functions") or [])
    prompt = str(state.get("prompt") or "").lower()
    room_type = str((state.get("room_intent") or {}).get("room_type") or "").lower()
    zones = []
    whole_home = any(token in prompt for token in ("все комнаты", "всех комнат", "комнаты дома", "весь дом", "whole house", "entire home"))
    if whole_home:
        zones.extend([
            ("zone_living", "living_zone", "Гостиная", "Living room", 1, 0.18),
            ("zone_kitchen", "kitchen_zone", "Кухня", "Kitchen", 2, 0.16),
            ("zone_dining", "dining_zone", "Столовая", "Dining room", 3, 0.14),
            ("zone_sleeping", "sleeping_zone", "Спальня", "Bedroom", 4, 0.18),
            ("zone_work", "work_zone", "Кабинет", "Office", 5, 0.12),
            ("zone_bathroom", "bathroom_zone", "Ванная", "Bathroom", 6, 0.12),
            ("zone_toilet", "toilet_zone", "Туалет", "Toilet", 7, 0.1),
        ])
    elif room_type in {"kitchen"} or "cooking" in funcs:
        zones.append(("zone_kitchen", "kitchen_zone", "Кухня", "Kitchen", 1, 0.7))
        if "dining" in funcs:
            zones.append(("zone_dining", "dining_zone", "Столовая", "Dining room", 2, 0.3))
    elif room_type in {"living_room"} or "living" in funcs:
        zones.append(("zone_living", "living_zone", "Гостиная", "Living room", 1, 0.7))
    elif room_type in {"bathroom"} or "bathing" in funcs:
        zones.append(("zone_bathroom", "bathroom_zone", "Ванная", "Bathroom", 1, 0.7))
        if "toilet" in funcs:
            zones.append(("zone_toilet", "toilet_zone", "Туалет", "Toilet", 2, 0.3))
    elif room_type in {"toilet"} or "toilet" in funcs:
        zones.append(("zone_toilet", "toilet_zone", "Туалет", "Toilet", 1, 0.7))
    elif room_type in {"dining_room"} or "dining" in funcs:
        zones.append(("zone_dining", "dining_zone", "Столовая", "Dining room", 1, 0.7))
    elif room_type in {"office"}:
        zones.append(("zone_work", "work_zone", "Кабинет", "Office", 1, 0.7))
    if not whole_home and not zones and "sleeping" in funcs:
        zones.append(("zone_sleeping", "sleeping_zone", "Спальная зона", "Sleeping zone", 1, 0.45))
    if not whole_home and (not zones or room_type == "bedroom") and "working" in funcs:
        zones.append(("zone_work", "work_zone", "Рабочая зона", "Work zone", 2, 0.25))
    if not whole_home and (not zones or room_type == "bedroom") and "storage" in funcs:
        zones.append(("zone_storage", "storage_zone", "Хранение", "Storage zone", 3, 0.2))
    deduped = []
    used_ids = set()
    used_types = set()
    for spec in zones:
        if spec[0] in used_ids or spec[1] in used_types:
            continue
        used_ids.add(spec[0])
        used_types.add(spec[1])
        deduped.append(spec)
    zones = deduped
    if not zones:
        zones.append(("zone_living", "living_zone", "Гостиная зона", "Living zone", 1, 0.6))
    return {"schema": "room_zones/v1", "zones": [{"id": zid, "type": zt, "name_ru": ru, "name_en": en, "priority": pr, "purpose": en, "desired_area_share": share, "placement_preferences": {"against_wall": True, "corner_allowed": True, "avoid_door": True, "near_window_if_known": zt == "work_zone"}} for zid, zt, ru, en, pr, share in zones], "source": "fallback"}


def _object_stub(sc: str, role: str, source: str = "fallback_template", dimensions_m: dict[str, float] | None = None) -> dict[str, Any]:
    label_ru, label_en = default_labels(sc)
    style = default_style(sc)
    out = {"id_hint": sc, "label_ru": label_ru, "label_en": label_en, "subclass": sc, "category": default_category(sc), "role": role, "quantity": 1, "importance": "required" if role != "accessory" else "optional", "source": source, **style}
    if dimensions_m:
        out["dimensions_m"] = dimensions_m
    return out


def _fallback_items(input_state: dict[str, Any], zone: dict[str, Any]) -> dict[str, Any]:
    tmpl = zone_templates.get(zone.get("type"), {})
    objects = []
    if tmpl.get("template_objects"):
        prefs = _prompt_preferences(input_state.get("prompt") or "")
        avoid = set(prefs.get("avoid_subclasses") or [])
        for sc in tmpl.get("template_objects", []):
            if sc in avoid:
                continue
            role = "main" if sc in tmpl.get("required_main", []) else "secondary" if sc in tmpl.get("required_secondary", []) else "accessory"
            dims = {"width": 0.95, "depth": 2.0, "height": 0.55} if sc == "bed" and prefs.get("bed_preference") == "single" else None
            dims = {"width": 1.6, "depth": 2.0, "height": 0.55} if sc == "bed" and prefs.get("bed_preference") == "double" else dims
            if sc == "blanket" and prefs.get("bed_preference") == "single":
                dims = {"width": 0.8, "depth": 1.6, "height": 0.08}
            if sc == "pillow" and prefs.get("bed_preference") == "single":
                dims = {"width": 0.42, "depth": 0.32, "height": 0.12}
            objects.append(_object_stub(sc, role, dimensions_m=dims))
        if prefs.get("theme") == "cars/racing":
            if zone.get("type") == "sleeping_zone":
                objects.extend([_object_stub("racing_rug", "accessory"), _object_stub("car_poster", "accessory")])
            elif zone.get("type") == "work_zone":
                objects.extend([_object_stub("toy_car", "accessory"), _object_stub("toy_car", "accessory"), _object_stub("car_decor", "accessory")])
            elif zone.get("type") == "storage_zone":
                objects.extend([_object_stub("toy_storage_box", "accessory"), _object_stub("toy_car", "accessory"), _object_stub("car_model", "accessory")])
        if "plants" in set((prefs.get("theme_spec") or {}).get("theme_tags") or []):
            if zone.get("type") == "sleeping_zone":
                objects.extend([_object_stub("potted_plant", "accessory"), _object_stub("small_potted_plant", "accessory"), _object_stub("hanging_planter", "accessory"), _object_stub("plant_stand", "accessory")])
            elif zone.get("type") == "work_zone":
                objects.extend([_object_stub("small_potted_plant", "accessory"), _object_stub("potted_plant", "accessory")])
            elif zone.get("type") == "storage_zone":
                objects.extend([_object_stub("small_potted_plant", "accessory"), _object_stub("potted_plant", "accessory"), _object_stub("plant_stand", "accessory")])
        return {"schema": "zone_items/v1", "zone_id": zone["id"], "objects": objects, "source": "fallback_template"}
    for role_key, role in [("required_main", "main"), ("required_secondary", "secondary")]:
        for sc in tmpl.get(role_key, []):
            objects.append(_object_stub(sc, role))
    for sc in tmpl.get("allowed_accessories", [])[: int(tmpl.get("min_accessories", 0))]:
        objects.append(_object_stub(sc, "accessory"))
    return {"schema": "zone_items/v1", "zone_id": zone["id"], "objects": objects, "source": "fallback_template"}


def _fallback_relations(zone: dict[str, Any], items: dict[str, Any]) -> dict[str, Any]:
    tmpl = zone_templates.get(zone.get("type"), {})
    rels = [{"from_subclass": a, "relation_type": r, "relation_class": "semantic", "to_subclass": b, "constraint_level": "hard", "weight": 1.0, "params": {}} for a, r, b in tmpl.get("mandatory_relations", []) if b != "room_wall"]
    return {"schema": "zone_relations/v1", "zone_id": zone["id"], "relations": rels, "source": "fallback_template"}


def run_room_intent_step(input_state: dict[str, Any], llm_settings: dict[str, Any]) -> dict[str, Any]:
    if _provider_is_none(llm_settings):
        return _fallback_intent(input_state)
    return call_json_llm(build_room_intent_prompt(input_state), **_settings(llm_settings, "02_room_intent"))


def run_zones_step(input_state: dict[str, Any], llm_settings: dict[str, Any]) -> dict[str, Any]:
    if _provider_is_none(llm_settings):
        return _fallback_zones(input_state)
    return call_json_llm(build_zones_prompt(input_state), **_settings(llm_settings, "03_zones"))


def run_zone_items_step(input_state: dict[str, Any], zone: dict[str, Any], llm_settings: dict[str, Any]) -> dict[str, Any]:
    theme_spec = input_state.get("theme_spec") or extract_theme_spec(input_state.get("prompt") or "")
    if _provider_is_none(llm_settings):
        raw = _fallback_items(input_state, zone)
        return sanitize_zone_items(str(zone.get("id") or ""), str(zone.get("type") or ""), raw, theme_spec)
    messages = build_zone_items_prompt(input_state, zone)
    raw = call_json_llm(messages, **_settings(llm_settings, f"04_zone_items_{zone.get('id')}"))
    raw["source"] = raw.get("source") or "llm"
    cleaned = sanitize_zone_items(str(zone.get("id") or ""), str(zone.get("type") or ""), raw, theme_spec)
    if cleaned.get("schema_errors"):
        retry_settings = _settings(llm_settings, f"04_zone_items_{zone.get('id')}_retry")
        retry_settings["max_attempts"] = 1
        retry_messages = messages + [{"role": "user", "content": "Schema error: subclass is empty or not in allowed_subclasses. Retry once. Every object must have non-empty subclass from allowed_subclasses; no generic object/decor/item; no coordinates."}]
        try:
            retry_raw = call_json_llm(retry_messages, **retry_settings)
            retry_raw["source"] = retry_raw.get("source") or "llm"
            retry_cleaned = sanitize_zone_items(str(zone.get("id") or ""), str(zone.get("type") or ""), retry_raw, theme_spec)
            if len(retry_cleaned.get("schema_errors") or []) <= len(cleaned.get("schema_errors") or []):
                cleaned = retry_cleaned
        except Exception as exc:
            cleaned.setdefault("warnings", []).append(f"LLM zone_items semantic retry failed: {exc}")
    return cleaned


def run_zone_relations_step(input_state: dict[str, Any], zone: dict[str, Any], zone_items: dict[str, Any], llm_settings: dict[str, Any]) -> dict[str, Any]:
    if _provider_is_none(llm_settings):
        return _fallback_relations(zone, zone_items)
    return call_json_llm(build_zone_relations_prompt(input_state, zone, zone_items), **_settings(llm_settings, f"05_zone_relations_{zone.get('id')}"))
