from __future__ import annotations

from typing import Any

try:
    from ..relationship_graph_stage import RELATION_TYPES, RELATION_CLASSES, CONSTRAINT_LEVELS
except Exception:
    RELATION_TYPES = {
        "on_top_of", "inside", "under", "near", "next_to", "left_of", "right_of", "in_front_of", "behind",
        "faces", "against_wall", "mounted_on_wall", "centered_on", "aligned_with", "grouped_with", "around",
        "above", "below", "visible_from",
    }
    RELATION_CLASSES = {"support", "containment", "proximity", "orientation", "wall", "group", "semantic"}
    CONSTRAINT_LEVELS = {"hard", "soft", "decorative"}


RELATION_CLASS_BY_TYPE = {
    "on_top_of": "support",
    "under": "support",
    "above": "support",
    "inside": "containment",
    "faces": "orientation",
    "near": "proximity",
    "next_to": "proximity",
    "in_front_of": "proximity",
    "behind": "proximity",
    "left_of": "proximity",
    "right_of": "proximity",
    "centered_on": "proximity",
    "aligned_with": "proximity",
    "against_wall": "wall",
    "mounted_on_wall": "wall",
    "around": "group",
    "grouped_with": "group",
    "visible_from": "orientation",
    "below": "support",
}

SUPPORT_ALLOWED = {
    "pillow": ["bed"],
    "blanket": ["bed"],
    "book": ["desk", "nightstand", "shelf", "bookcase", "coffee_table", "dining_table"],
    "phone": ["desk", "nightstand"],
    "table_lamp": ["desk", "nightstand", "side_table"],
    "desk_lamp": ["desk"],
    "laptop": ["desk"],
    "monitor": ["desk"],
    "keyboard": ["desk"],
    "mouse": ["desk"],
    "mug": ["desk", "nightstand", "coffee_table", "dining_table", "kitchen_counter"],
    "cup": ["dining_table", "kitchen_counter", "kitchen_table"],
    "water_bottle": ["desk"],
    "notebook": ["desk"],
    "desk_organizer": ["desk"],
    "small_potted_plant": ["desk", "nightstand", "shelf", "bookcase", "dresser", "coffee_table", "dining_table"],
    "potted_plant": ["floor", "plant_stand"],
    "hanging_planter": ["ceiling", "wall", "room_wall"],
    "plant_pot": ["floor", "plant_stand", "shelf", "bookcase"],
    "stove": ["kitchen_counter"],
    "kitchen_sink": ["kitchen_counter"],
    "kettle": ["kitchen_counter"],
    "cutting_board": ["kitchen_counter"],
    "fruit_bowl": ["kitchen_counter", "dining_table"],
    "cookbook": ["kitchen_counter", "shelf", "bookcase"],
    "pan": ["kitchen_counter", "stove"],
    "pot": ["kitchen_counter", "stove"],
    "plate": ["dining_table", "kitchen_table"],
    "bowl": ["dining_table", "kitchen_table", "kitchen_counter"],
    "vase": ["coffee_table", "dining_table", "shelf", "bookcase", "dresser"],
    "remote": ["coffee_table", "tv_stand"],
    "soap_dispenser": ["sink", "kitchen_sink", "kitchen_counter"],
    "toothbrush_cup": ["sink"],
}

FLOOR_ONLY = {
    "bed", "desk", "wardrobe", "dresser", "shelf", "bookcase",
    "office_chair", "chair", "dining_chair", "rug", "racing_rug", "road_play_mat",
    "potted_plant", "plant_stand", "floor_lamp", "sofa", "coffee_table", "side_table",
    "dining_table", "kitchen_table", "kitchen_counter", "kitchen_cabinet", "fridge",
    "toilet", "sink", "bathtub", "shower", "laundry_basket",
}
WALL_ONLY = {"wall_art", "mirror", "wall_shelf", "hanging_planter", "towel_rack", "toilet_paper_holder", "range_hood"}
CONTAINMENT_ALLOWED = {
    "book": ["shelf", "bookcase"],
    "storage_box": ["wardrobe", "shelf", "bookcase"],
    "toy_storage_box": ["wardrobe", "shelf", "bookcase"],
    "small_potted_plant": ["shelf", "bookcase"],
    "plant_pot": ["shelf", "bookcase"],
    "shampoo_bottle": ["shower", "bathtub"],
    "toy_car": ["shelf", "bookcase", "toy_storage_box"],
    "car_model": ["shelf", "bookcase"],
}
VIRTUAL_TARGETS = {"room_center", "room_wall", "free_corner", "floor", "ceiling", "wall"}
REQUIRED_BY_ZONE = {
    "sleeping_zone": [{"bed"}],
    "work_zone": [{"desk"}, {"office_chair", "chair"}],
    "storage_zone": [{"wardrobe", "dresser", "shelf", "bookcase"}],
}


def normalize_relation_class(relation_type: str, relation_class: str | None = None) -> str:
    if not relation_class or relation_class == "semantic":
        return RELATION_CLASS_BY_TYPE.get(str(relation_type), "proximity")
    return RELATION_CLASS_BY_TYPE.get(str(relation_type), relation_class)


def _edge(src: str, rel: str, dst: str, cls: str | None, level: str = "hard", weight: float = 1.0, params: dict[str, Any] | None = None, source: str = "rule") -> dict[str, Any]:
    relation_class = normalize_relation_class(rel, cls)
    return {
        "from_id": src,
        "from_object_id": src,
        "relation_type": rel,
        "relation_class": relation_class,
        "to_id": dst,
        "to_object_id": dst,
        "constraint_level": level,
        "weight": weight,
        "params": params or {},
        "source": source,
    }


def _first(objects: list[dict[str, Any]], subclasses: set[str], zone_id: str | None = None) -> dict[str, Any] | None:
    for obj in objects:
        if obj.get("subclass") in subclasses and (zone_id is None or obj.get("zone_id") == zone_id):
            return obj
    return None


def _object_by_id(objects: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(o.get("id")): o for o in objects}


def _target_subclass(edge: dict[str, Any], objects_by_id: dict[str, dict[str, Any]]) -> str:
    target_id = str(edge.get("to_id") or "")
    return target_id if target_id in VIRTUAL_TARGETS else str((objects_by_id.get(target_id) or {}).get("subclass") or "")


def _fits_support(src: dict[str, Any], dst: dict[str, Any]) -> bool:
    sd = src.get("dimensions_m") or {}
    dd = dst.get("dimensions_m") or {}
    return float(sd.get("width") or 0.0) <= float(dd.get("width") or 0.0) + 0.08 and float(sd.get("depth") or 0.0) <= float(dd.get("depth") or 0.0) + 0.08


def is_relation_allowed(edge: dict[str, Any], objects_by_id: dict[str, dict[str, Any]]) -> tuple[bool, str]:
    src = objects_by_id.get(str(edge.get("from_id") or ""))
    if not src:
        return False, "missing from_id"
    sc = str(src.get("subclass") or "")
    if not sc:
        return False, "empty source subclass"
    rel = str(edge.get("relation_type") or "")
    target_id = str(edge.get("to_id") or "")
    tc = _target_subclass(edge, objects_by_id)
    if target_id not in VIRTUAL_TARGETS and not tc:
        return False, "missing to_id"
    if rel in {"on_top_of", "above"}:
        allowed_targets = SUPPORT_ALLOWED.get(sc)
        if not allowed_targets or tc not in allowed_targets:
            return False, f"support not allowed: {sc} -> {tc}"
        if target_id not in VIRTUAL_TARGETS and tc not in {"floor", "wall", "ceiling"}:
            if not _fits_support(src, objects_by_id[target_id]):
                return False, f"object larger than support: {sc} -> {tc}"
    if rel == "inside":
        allowed_targets = CONTAINMENT_ALLOWED.get(sc)
        if not allowed_targets or tc not in allowed_targets:
            return False, f"containment not allowed: {sc} -> {tc}"
        if target_id not in VIRTUAL_TARGETS and not _fits_support(src, objects_by_id[target_id]):
            return False, f"object larger than container: {sc} -> {tc}"
    if sc in FLOOR_ONLY and rel in {"inside", "on_top_of"} and not (sc == "potted_plant" and tc == "plant_stand"):
        return False, f"floor-only object cannot use {rel}: {sc}"
    return True, ""


def resolve_relations_by_subclass(relations: list[dict[str, Any]], objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rel_payload in relations:
        zone_id = rel_payload.get("zone_id")
        payload_source = str(rel_payload.get("source") or "").strip()
        edge_source = "fallback_template" if payload_source in {"fallback", "fallback_template", "template"} else "llm_enrichment"
        for r in rel_payload.get("relations", []):
            src = _first(objects, {str(r.get("from_subclass"))}, zone_id)
            dst_sub = str(r.get("to_subclass"))
            dst = _first(objects, {dst_sub}, zone_id) if dst_sub not in VIRTUAL_TARGETS else {"id": dst_sub}
            if src and dst:
                out.append(_edge(src["id"], str(r.get("relation_type")), dst["id"], str(r.get("relation_class") or "semantic"), str(r.get("constraint_level") or "soft"), float(r.get("weight") or 1.0), dict(r.get("params") or {}), edge_source))
    return out


def augment_relations_with_rules(objects: list[dict[str, Any]], existing_edges: list[dict[str, Any]], zone_templates: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    out = list(existing_edges)
    seen = {(e.get("from_id"), e.get("relation_type"), e.get("to_id")) for e in out}

    def add(src_obj, rel, dst_obj_or_id, cls, level="hard", params=None):
        if not src_obj or not dst_obj_or_id:
            return
        dst_id = dst_obj_or_id if isinstance(dst_obj_or_id, str) else dst_obj_or_id.get("id")
        key = (src_obj["id"], rel, dst_id)
        if key not in seen:
            out.append(_edge(src_obj["id"], rel, dst_id, cls, level, params=params))
            seen.add(key)

    for obj in objects:
        z = obj.get("zone_type") or ""
        sc = obj.get("subclass")
        desk = _first(objects, {"desk"}, obj.get("zone_id"))
        bed = _first(objects, {"bed"}, obj.get("zone_id"))
        nightstand = _first(objects, {"nightstand"}, obj.get("zone_id"))
        dresser = _first(objects, {"dresser"}, obj.get("zone_id"))
        shelf = _first(objects, {"shelf", "bookcase"}, obj.get("zone_id"))
        wardrobe = _first(objects, {"wardrobe"}, obj.get("zone_id"))
        table = _first(objects, {"dining_table", "kitchen_table"}, obj.get("zone_id"))
        dining_table = _first(objects, {"dining_table"}, obj.get("zone_id"))
        kitchen_counter = _first(objects, {"kitchen_counter"}, obj.get("zone_id"))
        kitchen_table = _first(objects, {"kitchen_table"}, obj.get("zone_id"))
        sofa = _first(objects, {"sofa"}, obj.get("zone_id"))
        coffee_table = _first(objects, {"coffee_table"}, obj.get("zone_id"))
        sink = _first(objects, {"sink"}, obj.get("zone_id"))
        kitchen_sink = _first(objects, {"kitchen_sink"}, obj.get("zone_id"))
        if z == "work_zone" and sc in {"mug", "cup", "water_bottle", "laptop", "book", "notebook", "mouse", "keyboard", "monitor", "table_lamp", "desk_lamp", "desk_organizer", "small_potted_plant"}:
            placement = {
                "monitor": "back_center",
                "keyboard": "front_center",
                "laptop": "left_front",
                "mouse": "right_front",
                "mug": "right_side",
                "cup": "right_side",
                "water_bottle": "back_right",
                "table_lamp": "back_left",
                "desk_lamp": "back_left",
                "notebook": "left_side",
                "desk_organizer": "back_left",
                "small_potted_plant": "back_right",
            }.get(sc, "center")
            add(obj, "on_top_of", desk, "support", params={"surface": "top", "placement_area": placement})
        if z == "work_zone" and sc in {"office_chair", "chair"}:
            add(obj, "in_front_of", desk, "proximity", params={"distance_m": {"min": 0.15, "max": 0.65}})
            add(obj, "faces", desk, "orientation", params={"tolerance_deg": 25})
        if z in {"kitchen_zone", "dining_zone"} and sc in {"chair", "dining_chair"}:
            add(obj, "around", table, "group")
            add(obj, "faces", table, "orientation")
        if z == "kitchen_zone" and sc in {"stove", "kitchen_sink", "kettle", "cutting_board", "fruit_bowl", "cookbook", "mug", "cup", "bowl", "pan", "pot", "soap_dispenser"}:
            target = kitchen_counter or kitchen_table
            placement = {
                "stove": "back_center",
                "kitchen_sink": "left_side",
                "kettle": "back_right",
                "cutting_board": "front_center",
                "fruit_bowl": "center",
                "cookbook": "left_front",
                "mug": "right_front",
                "cup": "right_front",
                "bowl": "center",
                "pan": "back_center",
                "pot": "back_center",
                "soap_dispenser": "right_side",
            }.get(sc, "center")
            add(obj, "on_top_of", target, "support", params={"surface": "top", "placement_area": placement})
        if z == "dining_zone" and sc in {"plate", "bowl", "cup", "mug", "vase", "fruit_bowl", "plant"}:
            add(obj, "on_top_of", dining_table or table, "support", params={"surface": "top", "placement_area": "center" if sc in {"vase", "fruit_bowl", "plant"} else "front_center"})
        if z == "living_zone" and sc in {"remote", "book", "mug", "vase", "plant"}:
            add(obj, "on_top_of" if coffee_table else "near", coffee_table or sofa, "support" if coffee_table else "proximity", "soft", params={"surface": "top", "placement_area": "center"})
        if z == "living_zone" and sc == "pillow":
            add(obj, "on_top_of", sofa, "support", "soft")
        if z == "living_zone" and sc == "blanket":
            add(obj, "near", sofa, "proximity", "soft")
        if sc == "sofa":
            target = _first(objects, {"tv"}, None) or _first(objects, {"coffee_table"}, obj.get("zone_id")) or "room_center"
            add(obj, "faces", target, "orientation")
        if sc == "coffee_table":
            add(obj, "in_front_of", sofa, "proximity")
        if sc == "tv":
            stand = _first(objects, {"tv_stand"}, obj.get("zone_id"))
            add(obj, "on_top_of" if stand else "mounted_on_wall", stand or "room_wall", "support" if stand else "wall")
            if sofa:
                add(obj, "visible_from", sofa, "orientation", "soft")
        if sc == "nightstand":
            add(obj, "next_to", bed, "proximity")
        if sc == "table_lamp":
            add(obj, "on_top_of", nightstand or desk, "support", params={"surface": "top", "placement_area": "center"})
        if z == "sleeping_zone" and sc in {"book", "phone"}:
            add(obj, "on_top_of", nightstand or bed, "support", params={"surface": "top", "placement_area": "center"})
        if sc in {"pillow", "blanket"}:
            add(obj, "on_top_of", bed, "support")
        if sc in {"rug", "racing_rug", "road_play_mat"}:
            add(obj, "under", bed or table or sofa, "proximity", "soft")
        if sc == "mirror":
            add(obj, "above" if dresser or sink else "mounted_on_wall", dresser or sink or "room_wall", "wall", "soft")
        if sc in {"wardrobe", "dresser", "shelf", "bookcase", "toilet", "sink", "kitchen_counter", "kitchen_cabinet", "fridge", "oven", "shower", "bathtub"}:
            add(obj, "against_wall", "room_wall", "wall")
        if sc == "plant":
            add(obj, "near", "free_corner", "proximity", "soft")
        if sc == "potted_plant":
            stand = _first(objects, {"plant_stand"}, obj.get("zone_id"))
            add(obj, "on_top_of" if stand else "near", stand or "free_corner", "support" if stand else "proximity", "soft")
        if sc == "small_potted_plant":
            target = desk or nightstand or shelf or dresser or coffee_table or dining_table
            if target:
                add(obj, "on_top_of" if target.get("subclass") not in {"shelf", "bookcase"} else "inside", target, "support" if target.get("subclass") not in {"shelf", "bookcase"} else "containment", "soft")
        if sc == "hanging_planter":
            add(obj, "mounted_on_wall", "room_wall", "wall", "soft")
        if sc == "plant_stand":
            add(obj, "near", "free_corner", "proximity", "soft")
        if sc in {"wall_art", "car_poster", "racing_wall_art"}:
            add(obj, "above", bed or sofa or desk or "room_wall", "wall", "soft")
            add(obj, "mounted_on_wall", "room_wall", "wall", "soft")
        if z != "kitchen_zone" and sc in {"soap_dispenser", "toothbrush_cup"}:
            add(obj, "on_top_of", sink or kitchen_sink or kitchen_counter, "support")
        if z in {"bathroom_zone", "toilet_zone"} and sc in {"mirror", "towel_rack", "toilet_paper_holder"}:
            add(obj, "mounted_on_wall", "room_wall", "wall", "soft")
        if z in {"bathroom_zone", "toilet_zone"} and sc in {"hand_towel", "towel"}:
            rack = _first(objects, {"towel_rack"}, obj.get("zone_id"))
            add(obj, "near", rack or sink or "room_wall", "proximity", "soft")
        if z == "bathroom_zone" and sc in {"shampoo_bottle"}:
            add(obj, "inside" if _first(objects, {"shower", "bathtub"}, obj.get("zone_id")) else "on_top_of", _first(objects, {"shower", "bathtub"}, obj.get("zone_id")) or sink, "containment" if _first(objects, {"shower", "bathtub"}, obj.get("zone_id")) else "support", "soft")
        if z == "toilet_zone" and sc == "toilet_brush":
            add(obj, "near", _first(objects, {"toilet"}, obj.get("zone_id")), "proximity", "soft")
        if z == "bathroom_zone" and sc in {"bath_mat"}:
            add(obj, "near", _first(objects, {"shower", "bathtub", "sink"}, obj.get("zone_id")), "proximity", "soft")
        if sc in {"storage_box", "toy_storage_box", "storage_box_for_toys"}:
            add(obj, "inside" if wardrobe else "near", wardrobe or shelf or dresser, "containment" if wardrobe else "proximity")
        if z == "storage_zone" and sc == "book":
            add(obj, "inside" if shelf else "on_top_of", shelf or dresser, "containment" if shelf else "support")
        if sc in {"toy_car", "car_model", "car_decor"}:
            add(obj, "inside" if shelf or wardrobe else "on_top_of", shelf or wardrobe or desk, "containment" if shelf or wardrobe else "support")
    related_sources = {e.get("from_id") for e in out}
    for obj in objects:
        if obj.get("role") != "accessory" or obj.get("id") in related_sources:
            continue
        zone_id = obj.get("zone_id")
        allowed_targets = SUPPORT_ALLOWED.get(str(obj.get("subclass") or ""), []) + CONTAINMENT_ALLOWED.get(str(obj.get("subclass") or ""), [])
        support = next((candidate for candidate in objects if candidate.get("zone_id") == zone_id and candidate.get("id") != obj.get("id") and candidate.get("subclass") in allowed_targets), None)
        if support and support.get("id") == obj.get("id"):
            support = None
            for candidate in objects:
                if candidate.get("zone_id") == zone_id and candidate.get("id") != obj.get("id") and candidate.get("subclass") in {"desk", "nightstand", "shelf", "bookcase", "dresser", "bed", "wardrobe"}:
                    support = candidate
                    break
        if support:
            add(obj, "on_top_of" if support.get("subclass") not in {"wardrobe", "shelf", "bookcase"} else "inside", support, "support" if support.get("subclass") not in {"wardrobe", "shelf", "bookcase"} else "containment", "soft")
    objects_by_id = _object_by_id(objects)
    cleaned = []
    for e in out:
        edge = _edge(str(e.get("from_id")), str(e.get("relation_type")), str(e.get("to_id")), str(e.get("relation_class") or "semantic"), str(e.get("constraint_level") or "soft"), float(e.get("weight") or 1.0), dict(e.get("params") or {}), str(e.get("source") or "rule"))
        ok, reason = is_relation_allowed(edge, objects_by_id) if edge["relation_type"] in {"on_top_of", "inside", "above"} else (True, "")
        if ok:
            cleaned.append(edge)
        elif edge.get("constraint_level") == "hard" and edge["relation_type"] in {"on_top_of", "inside"}:
            src = objects_by_id.get(edge["from_id"])
            if src and src.get("subclass") in FLOOR_ONLY:
                cleaned.append(_edge(edge["from_id"], "near", "free_corner", "proximity", "soft", source="semantic_repair", params={"repair_reason": reason}))
    return cleaned


def validate_relation_targets_exist(edges: list[dict[str, Any]], objects: list[dict[str, Any]], zones: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    ids = {o["id"] for o in objects} | VIRTUAL_TARGETS
    object_ids = [str(o.get("id") or "") for o in objects]
    objects_by_id = _object_by_id(objects)
    errors: list[Any] = []
    if len(object_ids) != len(set(object_ids)):
        errors.append("duplicate object ids")
    for obj in objects:
        if not obj.get("subclass"):
            errors.append(f"{obj.get('id')}: empty subclass")
        if not obj.get("label_en") and not obj.get("label_ru"):
            errors.append(f"{obj.get('id')}: empty labels")
        if obj.get("subclass") in WALL_ONLY and obj.get("placement_type") == "floor":
            errors.append(f"{obj.get('id')}: wall-only object has floor placement")
    for e in edges:
        base_bad = (
            e.get("from_id") not in ids
            or e.get("to_id") not in ids
            or e.get("relation_type") not in RELATION_TYPES
            or e.get("relation_class") == "semantic"
            or e.get("relation_class") not in RELATION_CLASSES
            or e.get("constraint_level") not in CONSTRAINT_LEVELS
        )
        if base_bad:
            errors.append(e)
            continue
        if e.get("relation_type") in {"on_top_of", "inside", "above"}:
            ok, reason = is_relation_allowed(e, objects_by_id)
            if not ok:
                errors.append({**e, "error": reason})
    if zones:
        for zone in zones:
            ztype = str(zone.get("type") or "")
            zone_objects = [o for o in objects if o.get("zone_id") == zone.get("id")]
            subclasses = {o.get("subclass") for o in zone_objects}
            for group in REQUIRED_BY_ZONE.get(ztype, []):
                if not subclasses.intersection(group):
                    errors.append(f"{zone.get('id')}: missing required {'/'.join(sorted(group))}")
    return {"schema": "relationship_validation/v1", "is_valid": not errors, "errors": errors}


def build_relationship_graph(objects: list[dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, Any]:
    return {"schema": "object_relationship_graph/v1", "nodes": [{"id": o["id"], "subclass": o["subclass"], "zone_id": o.get("zone_id")} for o in objects], "edges": edges}
