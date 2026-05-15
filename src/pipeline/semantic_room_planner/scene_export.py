from __future__ import annotations

from pathlib import Path
from typing import Any

from .schemas import write_json


SMALL_ACCESSORY_GROUPS = {
    "laptop",
    "monitor",
    "keyboard",
    "mouse",
    "mug",
    "cup",
    "water_bottle",
    "book",
    "notebook",
    "phone",
    "remote",
    "plate",
    "bowl",
    "soap_dispenser",
    "toothbrush_cup",
    "vase",
    "desk_organizer",
    "kettle",
    "cutting_board",
    "fruit_bowl",
    "cookbook",
    "pan",
    "pot",
    "soap_dispenser",
    "toothbrush_cup",
    "shampoo_bottle",
    "toilet_paper_holder",
    "toilet_brush",
    "wall_art",
    "car_poster",
    "racing_wall_art",
    "toy_car",
    "car_model",
    "car_decor",
    "mirror",
    "decor",
    "unknown",
}

PROXY_FALLBACK_BY_GROUP = {
    "office_chair": "chair",
    "dining_chair": "chair",
    "chair": "chair",
    "bookcase": "shelf",
    "shelf": "shelf",
    "coffee_table": "table",
    "side_table": "table",
    "dining_table": "table",
    "kitchen_table": "table",
    "tv_stand": "table",
    "kitchen_counter": "table",
    "kitchen_cabinet": "closed_cabinet",
    "fridge": "closed_cabinet",
    "oven": "closed_cabinet",
    "stove": "decor_box",
    "kitchen_sink": "decor_box",
    "range_hood": "decor_box",
    "pillow": "pillow",
    "blanket": "blanket",
    "bed": "bed",
    "desk": "desk",
    "wardrobe": "wardrobe",
    "dresser": "dresser",
    "sofa": "sofa",
    "armchair": "armchair",
    "rug": "rug",
    "racing_rug": "rug",
    "road_play_mat": "rug",
    "plant": "plant",
    "bath_mat": "rug",
    "laundry_basket": "closed_cabinet",
    "towel_rack": "decor_box",
    "mirror": "decor_box",
    "wall_art": "decor_box",
    "table_lamp": "table_lamp",
    "floor_lamp": "floor_lamp",
    "toilet": "toilet",
    "sink": "sink",
    "bathtub": "bathtub",
    "shower": "shower",
}

PROXY_SUBCLASS_BY_FALLBACK = {
    "chair": "dining_chair",
    "shelf": "open_bookshelf",
    "table": "rectangular_dining_table",
    "pillow": "sleeping_pillow",
    "blanket": "folded_blanket",
    "bed": "double_bed",
    "desk": "writing_desk",
    "wardrobe": "hinged_wardrobe",
    "dresser": "chest_of_drawers",
    "sofa": "straight_sofa",
    "rug": "rectangular_rug",
    "plant": "potted_plant",
    "mirror": "round_mirror",
    "wall_art": "framed_picture",
    "table_lamp": "desk_lamp",
    "floor_lamp": "slim_floor_lamp",
    "toilet": "floor_mounted_toilet",
    "sink": "bathroom_sink",
    "bathtub": "rectangular_bathtub",
    "shower": "shower_cabin",
    "decor_box": "small_decor_box",
    "armchair": "lounge_armchair",
    "closed_cabinet": "closed_cabinet",
}


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None or isinstance(value, bool):
        return default
    try:
        return float(str(value).replace(",", ".").strip())
    except (TypeError, ValueError):
        return default


def _extract_yaw_deg(item: dict[str, Any], yaw_deg: Any = None) -> float:
    for value in (yaw_deg, item.get("yaw_deg"), item.get("rotation_z_deg"), item.get("rotation_deg")):
        if value is not None:
            return _as_float(value)
    rotation = item.get("rotation")
    if isinstance(rotation, (list, tuple)) and rotation:
        return _as_float(rotation[2] if len(rotation) >= 3 else rotation[-1])
    if isinstance(rotation, dict):
        for key in ("z_deg", "yaw_deg", "rotation_z_deg", "z"):
            if key in rotation:
                return _as_float(rotation.get(key))
    return 0.0


def normalize_rotation_for_legacy_blender(item: dict[str, Any], yaw_deg: Any = None) -> dict[str, Any]:
    yaw = _extract_yaw_deg(item, yaw_deg)
    rotation = item.get("rotation")
    if isinstance(rotation, dict):
        meta = item.setdefault("meta", {})
        if isinstance(meta, dict):
            semantic_meta = meta.setdefault("semantic_room_planner", {})
            if isinstance(semantic_meta, dict):
                semantic_meta.setdefault("original_rotation_dict", dict(rotation))
    item["yaw_deg"] = yaw
    item["rotation_z_deg"] = yaw
    item["rotation"] = [0.0, 0.0, yaw]
    return item


def _has_local_mesh_reference(item: dict[str, Any]) -> bool:
    asset = item.get("asset") if isinstance(item.get("asset"), dict) else {}
    return bool(item.get("mesh_path") or item.get("obj_path") or asset.get("mesh_path") or asset.get("obj_path"))


def _proxy_fallback_group(semantic_group: str) -> str:
    group = str(semantic_group or "").strip().lower()
    if group in SMALL_ACCESSORY_GROUPS:
        return "decor_box"
    return PROXY_FALLBACK_BY_GROUP.get(group, "closed_cabinet")


def ensure_procedural_proxy_asset(item: dict[str, Any], semantic_group: str, dimensions_m: dict[str, Any]) -> dict[str, Any]:
    if _has_local_mesh_reference(item):
        return item
    asset = item.get("asset") if isinstance(item.get("asset"), dict) else {}
    if asset.get("kind") and asset.get("kind") != "procedural_proxy" and _has_local_mesh_reference({"asset": asset}):
        return item
    group = str(semantic_group or item.get("semantic_group") or item.get("category") or "decor_box").strip().lower()
    fallback = _proxy_fallback_group(group)
    proxy_subclass = PROXY_SUBCLASS_BY_FALLBACK.get(fallback, PROXY_SUBCLASS_BY_FALLBACK["decor_box"])

    merged_asset = dict(asset)
    merged_asset.update({
        "kind": "procedural_proxy",
        "source_kind": "semantic_room_planner",
        "asset_source": "supplier_catalog_procedural_proxy",
        "fallback_subclass": fallback,
        "mesh_fit_mode": merged_asset.get("mesh_fit_mode") or "uniform",
    })
    item["asset"] = merged_asset

    source = item.get("source") if isinstance(item.get("source"), dict) else {}
    item["source"] = {**source, "asset_source": "supplier_catalog_procedural_proxy"}

    dims = dimensions_m if isinstance(dimensions_m, dict) else {}
    meta = item.setdefault("meta", {})
    if not isinstance(meta, dict):
        meta = {}
        item["meta"] = meta
    candidate = meta.get("supplier_candidate") if isinstance(meta.get("supplier_candidate"), dict) else {}
    candidate = {
        **candidate,
        "semantic_group": group,
        "base_type": group,
        "category_norm": group,
        "title": item.get("name") or group,
        "width_cm": _as_float(dims.get("width"), 0.4) * 100.0,
        "depth_cm": _as_float(dims.get("depth"), 0.4) * 100.0,
        "height_cm": _as_float(dims.get("height"), 0.4) * 100.0,
        "fallback_subclass": fallback,
        "subclass": candidate.get("subclass") or proxy_subclass,
    }
    meta["supplier_candidate"] = candidate
    return item


def _item(obj: dict[str, Any], placement: dict[str, Any], graph: dict[str, Any]) -> dict[str, Any]:
    semantic_group = str(obj["subclass"]).strip().lower()
    item = {
        "id": obj["id"], "name": obj.get("label_en") or obj["subclass"], "category": obj.get("category"), "semantic_group": obj["subclass"],
        "zone_id": obj.get("zone_id"), "aabb": placement.get("aabb"), "yaw_deg": placement.get("rotation_z_deg", 0.0), "rotation": {"z_deg": placement.get("rotation_z_deg", 0.0)},
        "placement_type": obj.get("placement_type"), "asset": {"type": "procedural_bbox", "placeholder": True},
        "meta": {"relationship_graph": {"node_id": obj["id"]}, "semantic_room_planner": obj},
    }
    item["semantic_group"] = semantic_group
    normalize_rotation_for_legacy_blender(item, placement.get("rotation_z_deg", 0.0))
    ensure_procedural_proxy_asset(item, semantic_group, obj.get("dimensions_m") or {})
    return item


def export_scene_plan(state: dict[str, Any], out_dir: str | Path | None = None) -> dict[str, Any]:
    objects = state["objects"]
    placements = {p["object_id"]: p for p in state["placements"]["placements"]}
    graph = state["relationship_graph"]
    room = state["input"]["room"]
    items = [_item(o, placements.get(o["id"], {}), graph) for o in objects if o["id"] in placements]
    status = state.get("status") or ("success" if state["geometry_validation"].get("is_valid") else "partial_success")
    plan = {
        "schema": "room_scene_plan/v1", "status": status, "input": {"prompt": state["input"].get("prompt"), "room": room},
        "room_geometry": state["room_geometry"], "design_intent": state["room_intent"], "zones": state["zones"],
        "objects": objects, "relationship_graph": graph, "anchors": state["anchors"], "placements": state["placements"]["placements"],
        "catalog_queries": state.get("catalog_queries", {}).get("items", []), "catalog_matches": [], "asset_decisions": [], "trellis2_jobs": [],
        "validations": {"relationship_validation": state.get("relationship_validation", {}), "geometry_validation": state["geometry_validation"], "semantic_validation": {}, "asset_validation": {}},
        "warnings": list(state.get("warnings") or []) + list(state["geometry_validation"].get("soft_warnings") or []),
        "artifacts": {"scene_json_path": None, "placement_json_path": None, "blend_path": None, "preview_images": []},
    }
    scene_v1 = {"schema": "scene.v1", "room": room, "items": items, "relationship_graph": graph, "semantic_room_planner": {"schema": "semantic_room_planner/v1", "status": status, "warning": "Hard constraints are validated; ideal design is not guaranteed."}}
    placement_rows = []
    for p in state["placements"]["placements"]:
        placement_item = {"id": p["object_id"], "object_id": p["object_id"], "position": p["position"], "rotation_z_deg": p["rotation_z_deg"], "aabb": p["aabb"]}
        normalize_rotation_for_legacy_blender(placement_item, p.get("rotation_z_deg", 0.0))
        placement_rows.append(placement_item)
    placement_v1 = {"schema": "placement.v1", "room": room, "placements": placement_rows, "items": items, "relationship_graph": graph, "semantic_room_planner": scene_v1["semantic_room_planner"]}
    layout_targets = {"schema": "layout_targets/v1", "targets": [{"object_id": o["id"], "semantic_group": o["subclass"], "zone_id": o.get("zone_id"), "catalog_queries": [q for q in state.get("catalog_queries", {}).get("items", []) if q.get("object_id") == o["id"]]} for o in objects]}
    if out_dir:
        root = Path(out_dir)
        scene_path = write_json(root / "scene.semantic.v1.json", scene_v1)
        placement_path = write_json(root / "placement.semantic.v1.json", placement_v1)
        layout_path = write_json(root / "layout_targets.semantic.json", layout_targets)
        plan["artifacts"]["scene_json_path"] = str(scene_path.resolve())
        plan["artifacts"]["placement_json_path"] = str(placement_path.resolve())
        plan["artifacts"]["layout_targets_json_path"] = str(layout_path.resolve())
        write_json(root / "final_room_scene_plan.json", plan)
    return {"plan": plan, "scene_v1": scene_v1, "placement_v1": placement_v1, "layout_targets": layout_targets}
