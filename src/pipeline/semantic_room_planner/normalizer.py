from __future__ import annotations

from collections import defaultdict
from typing import Any

from .object_priors import default_category, default_labels, default_style, get_prior, normalize_subclass, subclass_priors
from .schemas import as_float, clamp, stable_slug
from .zone_templates import apply_zone_template_minimums, structural_subclasses_for_zone


def _dims(raw: Any, prior: dict[str, Any]) -> dict[str, float]:
    default = prior["default_dimensions_m"]
    limits = prior["dimension_limits_m"]
    src = raw if isinstance(raw, dict) else {}
    out = {}
    for key in ("width", "depth", "height"):
        val = as_float(src.get(key), default[key])
        out[key] = round(clamp(val, float(limits[key][0]), float(limits[key][1])), 4)
    return out


def expand_quantity(obj: dict[str, Any]) -> list[dict[str, Any]]:
    q = int(clamp(as_float(obj.get("quantity"), 1.0), 1, 12))
    return [dict(obj, quantity=1) for _ in range(q)]


def normalize_objects(all_zone_items: list[dict[str, Any]], zones: list[dict[str, Any]], priors: dict[str, Any] | None = None, templates: dict[str, Any] | None = None) -> dict[str, Any]:
    zone_by_id = {z.get("id"): z for z in zones}
    by_zone: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for payload in all_zone_items:
        zone_id = str(payload.get("zone_id") or "")
        payload_source = str(payload.get("source") or "").strip()
        zone_type = str(zone_by_id.get(zone_id, {}).get("type") or "")
        structural = structural_subclasses_for_zone(zone_type)
        llm_payload = payload_source not in {"fallback", "fallback_template", "template", "template_required"}
        raw_objects = payload.get("objects") if isinstance(payload.get("objects"), list) else payload.get("items", [])
        for obj in raw_objects:
            if not isinstance(obj, dict):
                continue
            if isinstance(obj, dict) and "subclass" not in obj and obj.get("type"):
                obj = dict(obj, subclass=obj.get("type"), id_hint=obj.get("type"), role=obj.get("role") or "accessory")
            subclass_probe = normalize_subclass(obj.get("subclass") or obj.get("id_hint"))
            if llm_payload and subclass_probe in structural and str(obj.get("role") or "").lower() != "accessory":
                continue
            seeded = dict(obj, zone_id=zone_id)
            if "source" not in seeded and payload_source:
                seeded["source"] = payload_source
            by_zone[zone_id].extend(expand_quantity(seeded))
    templated: list[dict[str, Any]] = []
    for zone in zones:
        templated.extend(apply_zone_template_minimums(zone, by_zone.get(str(zone.get("id")), [])))
    counts: dict[str, int] = defaultdict(int)
    out = []
    for obj in templated:
        subclass = normalize_subclass(obj.get("subclass") or obj.get("id_hint"))
        prior = get_prior(subclass)
        label_ru_default, label_en_default = default_labels(subclass)
        style_default = default_style(subclass)
        counts[subclass] += 1
        zone_id = str(obj.get("zone_id") or "")
        label_en = str(obj.get("label_en") or label_en_default)
        label_ru = str(obj.get("label_ru") or label_ru_default)
        color = str(obj.get("color") or style_default.get("color") or "").strip()
        material = str(obj.get("material") or style_default.get("material") or "").strip()
        seed = " ".join(x for x in [color, material, label_en] if x).strip() or label_en
        placement_type = str(obj.get("placement_type") or prior["placement_type"])
        if str(obj.get("role") or "") == "accessory" and prior["placement_type"] == "support":
            placement_type = "support"
        source = str(obj.get("source") or "llm")
        if source == "fallback":
            source = "fallback_template"
        elif source == "template_required":
            source = "template"
        out.append({
            "id": f"obj_{stable_slug(subclass)}_{counts[subclass]:03d}",
            "zone_id": zone_id,
            "zone_type": zone_by_id.get(zone_id, {}).get("type"),
            "label_ru": label_ru,
            "label_en": label_en,
            "category": default_category(subclass),
            "subclass": subclass,
            "role": str(obj.get("role") or ("accessory" if prior["placement_type"] == "support" else "secondary")),
            "dimensions_m": _dims(obj.get("dimensions_m"), prior),
            "color": color,
            "material": material,
            "placement_type": placement_type,
            "front_axis_local": prior["front_axis_local"],
            "source": source,
            "importance": str(obj.get("importance") or "optional"),
            "catalog_search_seed": seed,
        })
    return {"schema": "objects_normalized/v1", "objects": out}
