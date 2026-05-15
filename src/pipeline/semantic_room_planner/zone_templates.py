from __future__ import annotations

from copy import deepcopy
from typing import Any

from .object_priors import default_category, default_labels, default_style


ZONE_TYPES = {
    "sleeping_zone", "work_zone", "storage_zone", "living_zone", "dining_zone", "kitchen_zone",
    "bathroom_zone", "toilet_zone", "reading_zone", "vanity_zone", "entry_zone", "decor_zone", "utility_zone",
}

zone_templates: dict[str, dict[str, Any]] = {
    "work_zone": {
        "zone_type": "work_zone", "required_main": ["desk"], "required_secondary": ["office_chair"], "min_accessories": 8,
        "allowed_accessories": ["laptop", "monitor", "keyboard", "mouse", "mug", "water_bottle", "notebook", "table_lamp", "desk_lamp", "desk_organizer", "book", "small_potted_plant", "potted_plant"],
        "template_objects": ["desk", "office_chair", "laptop", "monitor", "keyboard", "mouse", "mug", "water_bottle", "notebook", "table_lamp", "desk_organizer", "plant"],
        "mandatory_relations": [["office_chair", "in_front_of", "desk"], ["office_chair", "faces", "desk"]],
    },
    "sleeping_zone": {
        "zone_type": "sleeping_zone", "required_main": ["bed"], "required_secondary": ["nightstand"], "optional_secondary": ["dresser", "floor_lamp"], "min_accessories": 7,
        "allowed_accessories": ["pillow", "pillow", "blanket", "table_lamp", "book", "phone", "rug", "wall_art", "potted_plant", "small_potted_plant", "hanging_planter", "plant_stand"],
        "template_objects": ["bed", "pillow", "pillow", "blanket", "nightstand", "table_lamp", "book", "phone", "rug", "wall_art"],
        "mandatory_relations": [["pillow", "on_top_of", "bed"], ["blanket", "on_top_of", "bed"], ["nightstand", "next_to", "bed"]],
    },
    "storage_zone": {"zone_type": "storage_zone", "required_main": ["wardrobe"], "required_secondary": ["dresser", "shelf"], "optional_secondary": ["bookcase"], "min_accessories": 4, "allowed_accessories": ["mirror", "storage_box", "book", "book", "small_potted_plant", "potted_plant", "plant_stand"], "template_objects": ["wardrobe", "dresser", "shelf", "mirror", "storage_box", "book", "book", "plant"], "mandatory_relations": [["wardrobe", "against_wall", "room_wall"]]},
    "living_zone": {"zone_type": "living_zone", "required_main": ["sofa"], "required_secondary": ["coffee_table"], "optional_secondary": ["tv", "tv_stand", "floor_lamp", "plant"], "min_accessories": 7, "allowed_accessories": ["remote", "book", "mug", "pillow", "blanket", "rug", "wall_art", "plant", "vase"], "template_objects": ["sofa", "coffee_table", "tv_stand", "tv", "remote", "book", "mug", "pillow", "blanket", "rug", "wall_art", "floor_lamp", "plant"], "mandatory_relations": [["coffee_table", "in_front_of", "sofa"], ["sofa", "faces", "coffee_table"]]},
    "dining_zone": {"zone_type": "dining_zone", "required_main": ["dining_table"], "required_secondary": ["dining_chair"], "required_counts": {"dining_chair": 4}, "min_accessories": 5, "allowed_accessories": ["plate", "plate", "bowl", "cup", "vase", "plant"], "template_objects": ["dining_table", "dining_chair", "dining_chair", "dining_chair", "dining_chair", "plate", "plate", "bowl", "cup", "vase"], "mandatory_relations": [["dining_chair", "around", "dining_table"], ["dining_chair", "faces", "dining_table"]]},
    "kitchen_zone": {"zone_type": "kitchen_zone", "required_main": ["kitchen_counter", "fridge"], "required_secondary": ["kitchen_cabinet", "stove", "kitchen_sink"], "min_accessories": 7, "allowed_accessories": ["kettle", "cutting_board", "fruit_bowl", "cookbook", "mug", "cup", "bowl", "pan", "pot", "soap_dispenser"], "template_objects": ["kitchen_counter", "kitchen_cabinet", "fridge", "stove", "kitchen_sink", "kettle", "cutting_board", "fruit_bowl", "cookbook", "mug", "soap_dispenser"], "mandatory_relations": [["stove", "on_top_of", "kitchen_counter"], ["kitchen_sink", "on_top_of", "kitchen_counter"], ["fridge", "against_wall", "room_wall"]]},
    "bathroom_zone": {"zone_type": "bathroom_zone", "required_main": ["sink"], "required_secondary": ["shower"], "optional_secondary": ["bathtub"], "min_accessories": 6, "allowed_accessories": ["soap_dispenser", "toothbrush_cup", "mirror", "towel_rack", "hand_towel", "bath_mat", "shampoo_bottle", "laundry_basket"], "template_objects": ["sink", "shower", "mirror", "soap_dispenser", "toothbrush_cup", "towel_rack", "hand_towel", "bath_mat", "shampoo_bottle", "laundry_basket"], "mandatory_relations": [["sink", "against_wall", "room_wall"], ["shower", "against_wall", "room_wall"]]},
    "toilet_zone": {"zone_type": "toilet_zone", "required_main": ["toilet"], "required_secondary": ["sink"], "min_accessories": 4, "allowed_accessories": ["toilet_paper_holder", "toilet_brush", "soap_dispenser", "mirror", "hand_towel"], "template_objects": ["toilet", "sink", "toilet_paper_holder", "toilet_brush", "soap_dispenser", "mirror", "hand_towel"], "mandatory_relations": [["toilet", "against_wall", "room_wall"], ["sink", "against_wall", "room_wall"]]},
    "reading_zone": {"zone_type": "reading_zone", "required_main": ["armchair"], "required_secondary": ["side_table", "floor_lamp"], "min_accessories": 1, "allowed_accessories": ["book", "mug", "rug"], "mandatory_relations": [["side_table", "next_to", "armchair"]]},
    "entry_zone": {"zone_type": "entry_zone", "required_main": ["shelf"], "optional_secondary": ["mirror"], "min_accessories": 0, "allowed_accessories": [], "mandatory_relations": [["shelf", "against_wall", "room_wall"]]},
}


def _object_stub(subclass: str, zone_id: str, source: str, role: str = "secondary") -> dict[str, Any]:
    label_ru, label_en = default_labels(subclass)
    style = default_style(subclass)
    return {"id_hint": subclass, "label_ru": label_ru, "label_en": label_en, "subclass": subclass, "category": default_category(subclass), "role": role, "quantity": 1, "importance": "required", "source": source, "zone_id": zone_id, **style}


def structural_subclasses_for_zone(zone_type: str) -> set[str]:
    tmpl = zone_templates.get(str(zone_type), {})
    return set(tmpl.get("required_main") or []) | set(tmpl.get("required_secondary") or []) | set(tmpl.get("optional_secondary") or [])


def allowed_subclasses_for_zone(zone_type: str) -> set[str]:
    tmpl = zone_templates.get(str(zone_type), {})
    allowed = set(tmpl.get("allowed_accessories") or []) | structural_subclasses_for_zone(zone_type)
    allowed.update(tmpl.get("template_objects") or [])
    return allowed


def apply_zone_template_minimums(zone: dict[str, Any], objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = [deepcopy(o) for o in objects]
    zone_type = zone.get("type")
    tmpl = zone_templates.get(str(zone_type), {})
    zone_id = str(zone.get("id") or "")
    existing = {str(o.get("subclass") or o.get("id_hint") or "").lower() for o in out}
    counts: dict[str, int] = {}
    for obj in out:
        key = str(obj.get("subclass") or obj.get("id_hint") or "").lower()
        counts[key] = counts.get(key, 0) + 1
    if not objects and tmpl.get("template_objects"):
        for subclass in tmpl.get("template_objects", []):
            role = "main" if subclass in tmpl.get("required_main", []) else "secondary" if subclass in tmpl.get("required_secondary", []) else "accessory"
            out.append(_object_stub(subclass, zone_id, "template_required", role))
        return out
    for subclass in tmpl.get("required_main", []):
        needed = int((tmpl.get("required_counts") or {}).get(subclass, 1))
        while counts.get(subclass, 0) < needed:
            out.append(_object_stub(subclass, zone_id, "template_required", "main"))
            existing.add(subclass)
            counts[subclass] = counts.get(subclass, 0) + 1
    for subclass in tmpl.get("required_secondary", []):
        needed = int((tmpl.get("required_counts") or {}).get(subclass, 1))
        while counts.get(subclass, 0) < needed:
            out.append(_object_stub(subclass, zone_id, "template_required", "secondary"))
            existing.add(subclass)
            counts[subclass] = counts.get(subclass, 0) + 1
    accessories = [o for o in out if str(o.get("role")) == "accessory" or str(o.get("subclass")) in tmpl.get("allowed_accessories", [])]
    for subclass in tmpl.get("allowed_accessories", [])[: max(0, int(tmpl.get("min_accessories", 0)) - len(accessories))]:
        if subclass not in existing:
            out.append(_object_stub(subclass, zone_id, "template_required", "accessory"))
            existing.add(subclass)
    return out
