from __future__ import annotations

import copy
from collections import Counter, defaultdict
from typing import Any, Iterable


WALL_MOUNTED_CATEGORIES = {
    "wall_art",
    "mirror",
    "wall_hooks",
    "wall_light",
    "headboard",
    "curtain",
    "tv",
    "towel_rack",
    "toilet_paper_holder",
    "hygiene_shower",
}
CEILING_MOUNTED_CATEGORIES = {"ceiling_light"}
SOFT_FLOOR_CATEGORIES = {"rug", "runner_rug", "bath_mat"}
SOFT_ON_OBJECT_CATEGORIES = {"pillow", "blanket"}
ON_TOP_CATEGORIES = {
    "decor_books",
    "decor_vase",
    "decor_box",
    "decor_tray",
    "table_lamp",
    "plant",
    "storage_basket",
    "tv_accessory",
    "soap_dispenser",
    "toothbrush_cup",
    "shampoo_bottle",
    "air_freshener",
}
SOLID_FLOOR_CATEGORIES = {
    "bed",
    "nightstand",
    "wardrobe",
    "wardrobe_module",
    "dresser",
    "desk",
    "chair",
    "bench",
    "stool",
    "sofa",
    "armchair",
    "coffee_table",
    "side_table",
    "tv_stand",
    "bookshelf",
    "console_table",
    "dining_table",
    "dining_chair",
    "ottoman",
    "cabinet",
    "shelf",
    "shoe_cabinet",
    "coat_rack",
    "umbrella_stand",
    "entry_bench",
    "plant",
    "floor_lamp",
    "storage_basket",
    "toilet",
    "sink",
    "bathtub",
    "shower",
    "vanity",
    "washing_machine",
    "laundry_basket",
    "small_bin",
}

WARDROBE_TOP_INVALID_CATEGORIES = {"decor_tray", "decor_vase", "decor_books"}
SURFACE_LIMITS_BY_PARENT_CATEGORY = {
    "nightstand": 3,
    "coffee_table": 6,
    "side_table": 3,
    "dresser": 4,
    "console_table": 4,
    "tv_stand": 5,
    "bookshelf": 10,
    "shoe_cabinet": 3,
    "entry_bench": 3,
}
CATEGORY_KEEP_PRIORITY = {
    "bed": 100,
    "sofa": 95,
    "wardrobe": 90,
    "wardrobe_module": 90,
    "dresser": 85,
    "tv_stand": 84,
    "bookshelf": 82,
    "shoe_cabinet": 82,
    "toilet": 94,
    "sink": 92,
    "bathtub": 91,
    "shower": 91,
    "washing_machine": 74,
    "laundry_basket": 30,
    "nightstand": 78,
    "coffee_table": 76,
    "side_table": 72,
    "console_table": 72,
    "entry_bench": 70,
    "bench": 70,
    "stool": 66,
    "armchair": 68,
    "chair": 65,
    "desk": 65,
    "floor_lamp": 35,
    "plant": 30,
    "umbrella_stand": 28,
    "storage_basket": 25,
}
CATEGORY_LIMITS = {
    "bedroom": {
        "very_high": {"wall_art": 4, "wall_light": 2},
        "high": {"wall_art": 3, "wall_light": 2},
        "normal": {"wall_art": 2, "wall_light": 1},
    },
    "living_room": {
        "very_high": {"wall_art": 8},
        "high": {"wall_art": 5},
        "normal": {"wall_art": 3},
    },
    "corridor": {
        "very_high": {"wall_art": 4, "wall_hooks": 2},
        "high": {"wall_art": 3, "wall_hooks": 2},
        "normal": {"wall_art": 2, "wall_hooks": 1},
    },
}
SMALL_BEDROOM_CATEGORY_LIMITS = {"wall_art": 2, "pillow": 2}
SMALL_BEDROOM_ON_TOP_LIMIT = 5
TALL_WALL_BLOCKING_CATEGORIES = {"wardrobe", "bookshelf", "cabinet"}


def _meta(item: dict[str, Any]) -> dict[str, Any]:
    meta = item.get("meta")
    if not isinstance(meta, dict):
        meta = {}
        item["meta"] = meta
    return meta


def _category(item: dict[str, Any]) -> str:
    return str(item.get("category") or "").strip().lower()


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _mount_type(item: dict[str, Any]) -> str:
    return str(item.get("mount_type") or "").strip().lower()


def _parent_id(item: dict[str, Any]) -> str:
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    constraints = item.get("constraints") if isinstance(item.get("constraints"), dict) else {}
    return str(meta.get("parent_id") or meta.get("support_id") or constraints.get("parent_id") or constraints.get("support_id") or "")


def _item_id(item: dict[str, Any]) -> str:
    return str(item.get("id") or "")


def _position(item: dict[str, Any]) -> list[float]:
    pos = item.get("position_m")
    if isinstance(pos, list) and len(pos) >= 3:
        return [_to_float(pos[0]), _to_float(pos[1]), _to_float(pos[2])]
    return [0.0, 0.0, 0.0]


def _physical_role(item: dict[str, Any]) -> str:
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    role = str(meta.get("physical_role") or "").strip().lower()
    if role:
        return role
    category = _category(item)
    mount = _mount_type(item)
    if mount == "ceiling" or category in CEILING_MOUNTED_CATEGORIES:
        return "ceiling_mounted"
    if mount == "wall" or category in WALL_MOUNTED_CATEGORIES:
        return "wall_mounted"
    if _parent_id(item) or mount == "on_top" or meta.get("support_relation") == "on_top":
        if category in SOFT_ON_OBJECT_CATEGORIES:
            return "soft_on_object"
        return "on_top"
    if category in SOFT_FLOOR_CATEGORIES:
        return "soft_floor"
    if category in SOLID_FLOOR_CATEGORIES:
        return "solid_floor"
    return "soft_floor"


def _replace_with_supplier_for_role(category: str, role: str) -> bool:
    if role == "solid_floor":
        return True
    if role == "ceiling_mounted":
        return True
    if role == "wall_mounted":
        return category in {"mirror", "wall_light", "tv"}
    if role in {"on_top", "soft_on_object", "soft_floor", "decorative_soft"}:
        return False
    return False


def _removal(item: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "id": item.get("id"),
        "category": item.get("category"),
        "name": item.get("name"),
        "reason": reason,
    }


def _aabb(item: dict[str, Any]) -> dict[str, float] | None:
    aabb = item.get("aabb")
    if not isinstance(aabb, dict):
        return None
    try:
        return {
            "x_min": float(aabb.get("x_min")),
            "x_max": float(aabb.get("x_max")),
            "y_min": float(aabb.get("y_min")),
            "y_max": float(aabb.get("y_max")),
            "z_min": float(aabb.get("z_min")),
            "z_max": float(aabb.get("z_max")),
        }
    except Exception:
        return None


def _intersects_xy(a: dict[str, float], b: dict[str, float], margin: float = 0.0) -> bool:
    return not (
        a["x_max"] + margin <= b["x_min"]
        or b["x_max"] + margin <= a["x_min"]
        or a["y_max"] + margin <= b["y_min"]
        or b["y_max"] + margin <= a["y_min"]
    )


def normalize_item_semantics(item: dict[str, Any]) -> None:
    category = _category(item)
    meta = _meta(item)

    if category in WALL_MOUNTED_CATEGORIES:
        item["mount_type"] = "wall"
    elif category in CEILING_MOUNTED_CATEGORIES:
        item["mount_type"] = "ceiling"
    elif _parent_id(item) or meta.get("support_relation") == "on_top":
        item["mount_type"] = "on_top"

    role = _physical_role(item)
    meta["physical_role"] = role
    meta["replace_with_supplier"] = _replace_with_supplier_for_role(category, role)
    if role == "wall_mounted":
        item["mount_type"] = "wall"
    elif role == "ceiling_mounted":
        item["mount_type"] = "ceiling"
    elif role in {"on_top", "soft_on_object"}:
        item["mount_type"] = "on_top"
        meta["allow_collision"] = True
        meta.setdefault("support_relation", "on_top")
    elif role in {"soft_floor"}:
        meta["allow_collision"] = True


def _target_id(target: Any, item: dict[str, Any], by_id: dict[str, dict[str, Any]], by_category: dict[str, list[dict[str, Any]]]) -> str | None:
    raw = str(target or "").strip()
    if not raw:
        return None
    if raw in {"room_center", "free_space", "door", "window"}:
        return raw
    if raw in by_id:
        return raw
    category_matches = by_category.get(raw.strip().lower(), [])
    own_id = _item_id(item)
    for candidate in category_matches:
        candidate_id = _item_id(candidate)
        if candidate_id and candidate_id != own_id:
            return candidate_id
    return raw


def _access_min_clearance(category: str, room_type: str) -> float:
    if room_type in {"bathroom", "toilet"}:
        return 0.32
    if category in {"coffee_table", "bench"}:
        return 0.15
    return 0.45


def annotate_layout_contracts(items: list[dict[str, Any]], *, room_type: str) -> None:
    """Attach explicit orientation/access contracts for downstream validators."""
    by_id = {_item_id(item): item for item in items if _item_id(item)}
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_category[_category(item)].append(item)

    for item in items:
        category = _category(item)
        meta = _meta(item)
        role = _physical_role(item)
        wall_id = str(meta.get("wall_id") or "").strip()
        front_target_raw = meta.get("front_target")
        target_id = _target_id(front_target_raw, item, by_id, by_category)
        pos = _position(item)

        meta.setdefault("front_axis_local", "+Y")
        if target_id and target_id not in {"door", "window"}:
            meta["resolved_front_target_id"] = target_id

        if role == "wall_mounted":
            item["mounting_height_m"] = round(pos[2], 3)
            item["orientation_rule"] = {
                "type": "mounted_on_wall",
                "wall_id": wall_id or None,
                "back_axis_local": "-Y",
                "front_axis_local": "+Y",
                "tolerance_deg": 10,
            }
            continue

        if role == "ceiling_mounted":
            item["mounting_height_m"] = round(pos[2], 3)
            item["orientation_rule"] = {
                "type": "ceiling_mounted",
                "front_axis_local": "+Y",
                "tolerance_deg": 180,
            }
            continue

        if category == "bed" and wall_id:
            item["orientation_rule"] = {
                "type": "headboard_against_wall",
                "wall_id": wall_id,
                "head_axis_local": "-Y",
                "front_axis_local": "+Y",
                "tolerance_deg": 10,
            }
            item["clearance_rule"] = {
                "type": "side_or_front_access",
                "min_clearance_m": 0.45,
                "not_satisfied_by_adjacent_bench": True,
            }
            continue

        if category in {"toilet", "sink", "bathtub", "shower", "vanity"}:
            if str(front_target_raw or "").strip() == "door":
                meta["access_target"] = "door"
                item["access_target"] = "door"
                meta["front_target"] = "room_center"
                target_id = "room_center"
            if wall_id:
                item["orientation_rule"] = {
                    "type": "back_to_wall_face_free_space",
                    "wall_id": wall_id,
                    "back_axis_local": "-Y",
                    "front_axis_local": "+Y",
                    "target_id": target_id or "room_center",
                    "tolerance_deg": 25,
                }
            else:
                item["orientation_rule"] = {
                    "type": "face_target",
                    "target_id": target_id or "room_center",
                    "from_axis": "+Y",
                    "tolerance_deg": 35,
                }
            item["clearance_rule"] = {
                "type": "front_access",
                "min_clearance_m": _access_min_clearance(category, room_type),
                "access_target": meta.get("access_target", "free_space"),
            }
            continue

        if wall_id and target_id and target_id not in {"room_center", "door", "window"}:
            item["orientation_rule"] = {
                "type": "back_to_wall_face_target",
                "wall_id": wall_id,
                "target_id": target_id,
                "back_axis_local": "-Y",
                "front_axis_local": "+Y",
                "tolerance_deg": 15,
            }
        elif wall_id:
            item["orientation_rule"] = {
                "type": "back_to_wall",
                "wall_id": wall_id,
                "back_axis_local": "-Y",
                "front_axis_local": "+Y",
                "tolerance_deg": 10,
            }
        elif target_id:
            item["orientation_rule"] = {
                "type": "face_target",
                "target_id": target_id,
                "from_axis": "+Y",
                "tolerance_deg": 25,
            }

        if role == "solid_floor" and not isinstance(item.get("orientation_rule"), dict):
            item["orientation_rule"] = {
                "type": "face_target",
                "target_id": "room_center",
                "from_axis": "+Y",
                "tolerance_deg": 45,
            }

        if role == "solid_floor":
            item["clearance_rule"] = {
                "type": "front_or_side_access",
                "min_clearance_m": _access_min_clearance(category, room_type),
            }

        if category == "bench" and meta.get("anchor_id"):
            item["clearance_rule"] = {
                "type": "decorative_adjacent_anchor",
                "anchor_id": meta.get("anchor_id"),
                "placement_relation": meta.get("placement_relation") or "near",
                "not_a_walkway": True,
                "min_gap_m": meta.get("clearance_to_anchor_m", 0.0),
            }


def remove_invalid_wardrobe_top_items(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_id = {_item_id(item): item for item in items if _item_id(item)}
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    for item in items:
        parent = by_id.get(_parent_id(item))
        if parent and _category(parent) in {"wardrobe", "wardrobe_module"} and _category(item) in WARDROBE_TOP_INVALID_CATEGORIES:
            removed.append(_removal(item, "invalid_wardrobe_top_item"))
            continue
        kept.append(item)
    return kept, removed


def enforce_category_limits(items: list[dict[str, Any]], *, room_type: str, size_class: str, density: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    limits = dict(CATEGORY_LIMITS.get(room_type, {}).get(density, {}))
    if size_class == "small" and room_type == "corridor":
        limits["wall_art"] = min(limits.get("wall_art", 4), 4)
        limits["wall_hooks"] = min(limits.get("wall_hooks", 2), 2)
    if not limits:
        return items, []

    seen: dict[str, int] = defaultdict(int)
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    for item in items:
        category = _category(item)
        limit = limits.get(category)
        if limit is not None:
            seen[category] += 1
            if seen[category] > limit:
                removed.append(_removal(item, "category_limit"))
                continue
        kept.append(item)
    return kept, removed


def enforce_surface_limits(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_id = {_item_id(item): item for item in items if _item_id(item)}
    children_by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        parent_id = _parent_id(item)
        if parent_id:
            children_by_parent[parent_id].append(item)

    remove_ids: set[str] = set()
    removed: list[dict[str, Any]] = []
    for parent_id, children in children_by_parent.items():
        parent = by_id.get(parent_id)
        if not parent:
            continue
        limit = SURFACE_LIMITS_BY_PARENT_CATEGORY.get(_category(parent))
        if limit is None or len(children) <= limit:
            continue
        for child in children[limit:]:
            child_id = _item_id(child)
            if child_id:
                remove_ids.add(child_id)
            removed.append(_removal(child, "surface_limit"))

    if not remove_ids:
        return items, removed
    return [item for item in items if _item_id(item) not in remove_ids], removed


def repair_wall_mounted_overlaps(items: list[dict[str, Any]], *, room: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    del room
    kept: list[dict[str, Any]] = []
    kept_wall_aabbs: list[tuple[dict[str, Any], dict[str, float]]] = []
    removed: list[dict[str, Any]] = []
    tall_blockers: list[tuple[dict[str, Any], dict[str, float]]] = []
    for item in items:
        if _physical_role(item) != "solid_floor":
            continue
        category = _category(item)
        aabb = _aabb(item)
        if aabb is None:
            continue
        if category in TALL_WALL_BLOCKING_CATEGORIES or (category == "dresser" and aabb["z_max"] - aabb["z_min"] > 1.1):
            tall_blockers.append((item, aabb))

    for item in items:
        if _physical_role(item) != "wall_mounted":
            kept.append(item)
            continue
        aabb = _aabb(item)
        if aabb is None:
            kept.append(item)
            continue
        wall_id = str((_meta(item).get("wall_id") or item.get("wall_id") or ""))
        overlaps = False
        for blocker, blocker_aabb in tall_blockers:
            blocker_wall = str((_meta(blocker).get("wall_id") or blocker.get("wall_id") or ""))
            if wall_id and blocker_wall and wall_id != blocker_wall:
                continue
            if _intersects_xy(aabb, blocker_aabb, margin=0.08):
                overlaps = True
                break
        for other, other_aabb in kept_wall_aabbs:
            other_wall = str((_meta(other).get("wall_id") or other.get("wall_id") or ""))
            if wall_id and other_wall and wall_id != other_wall:
                continue
            if _category(item) in {"headboard", "curtain"} or _category(other) in {"headboard", "curtain"}:
                continue
            if _intersects_xy(aabb, other_aabb, margin=0.03):
                overlaps = True
                break
        if overlaps:
            removed.append(_removal(item, "wall_mounted_overlap"))
            continue
        kept.append(item)
        kept_wall_aabbs.append((item, aabb))
    return kept, removed


def _axis_gap(a: dict[str, float], b: dict[str, float]) -> float:
    x_overlap = not (a["x_max"] <= b["x_min"] or b["x_max"] <= a["x_min"])
    y_overlap = not (a["y_max"] <= b["y_min"] or b["y_max"] <= a["y_min"])
    if y_overlap:
        return max(b["x_min"] - a["x_max"], a["x_min"] - b["x_max"], 0.0)
    if x_overlap:
        return max(b["y_min"] - a["y_max"], a["y_min"] - b["y_max"], 0.0)
    return max(
        min(abs(b["x_min"] - a["x_max"]), abs(a["x_min"] - b["x_max"])),
        min(abs(b["y_min"] - a["y_max"]), abs(a["y_min"] - b["y_max"])),
    )


def enforce_bedroom_functional_clearances(items: list[dict[str, Any]], *, room: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if str(room.get("room_type") or room.get("type") or room.get("type_hint") or "").strip().lower() not in {"bedroom", "bed room", "спальня"}:
        return items, []

    remove_ids: set[str] = set()
    removed: list[dict[str, Any]] = []
    solid = [(item, _aabb(item)) for item in items if _physical_role(item) == "solid_floor"]
    solid = [(item, aabb) for item, aabb in solid if aabb is not None]
    blockers = [(item, aabb) for item, aabb in solid if _category(item) in {"bed", "bench"}]

    for item, aabb in solid:
        if _category(item) != "wardrobe":
            continue
        for blocker, blocker_aabb in blockers:
            if _axis_gap(aabb, blocker_aabb) < 0.55:
                item_id = _item_id(item)
                if item_id and item_id not in remove_ids:
                    remove_ids.add(item_id)
                    removed.append(_removal(item, "wardrobe_access_gap"))
                break

    area = _to_float(room.get("area_m2"), 0.0)
    if area <= 0.0:
        aabbs = [_aabb(item) for item in items]
        xs = [v for aabb in aabbs if aabb for v in (aabb["x_min"], aabb["x_max"])]
        ys = [v for aabb in aabbs if aabb for v in (aabb["y_min"], aabb["y_max"])]
        if xs and ys:
            area = (max(xs) - min(xs)) * (max(ys) - min(ys))
    if area <= 7.0:
        counts: Counter[str] = Counter()
        on_top_count = 0
        for item in items:
            item_id = _item_id(item)
            if item_id in remove_ids:
                continue
            category = _category(item)
            role = _physical_role(item)
            if category in SMALL_BEDROOM_CATEGORY_LIMITS:
                counts[category] += 1
                if counts[category] > SMALL_BEDROOM_CATEGORY_LIMITS[category]:
                    remove_ids.add(item_id)
                    removed.append(_removal(item, "small_bedroom_density_limit"))
                    continue
            if role == "on_top":
                on_top_count += 1
                if on_top_count > SMALL_BEDROOM_ON_TOP_LIMIT:
                    remove_ids.add(item_id)
                    removed.append(_removal(item, "small_bedroom_on_top_limit"))
                    continue
            if category in {"floor_lamp", "dresser"}:
                remove_ids.add(item_id)
                removed.append(_removal(item, "small_bedroom_floor_clutter"))

    if not remove_ids:
        return items, []
    return [item for item in items if _item_id(item) not in remove_ids], removed


def repair_solid_floor_overlaps(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    kept_aabbs: list[tuple[dict[str, Any], dict[str, float]]] = []
    removed: list[dict[str, Any]] = []
    removed_ids: set[str] = set()

    def priority(item: dict[str, Any]) -> int:
        return CATEGORY_KEEP_PRIORITY.get(_category(item), 50)

    for item in items:
        item_id = _item_id(item)
        if item_id in removed_ids:
            continue
        if _physical_role(item) != "solid_floor":
            kept.append(item)
            continue
        aabb = _aabb(item)
        if aabb is None:
            kept.append(item)
            continue

        overlapping_index: int | None = None
        for idx, (_other, other_aabb) in enumerate(kept_aabbs):
            if _intersects_xy(aabb, other_aabb, margin=0.02):
                overlapping_index = idx
                break
        if overlapping_index is None:
            kept.append(item)
            kept_aabbs.append((item, aabb))
            continue

        other, _other_aabb = kept_aabbs[overlapping_index]
        if priority(item) <= priority(other):
            removed.append(_removal(item, "solid_floor_overlap"))
            if item_id:
                removed_ids.add(item_id)
            continue

        other_id = _item_id(other)
        removed.append(_removal(other, "solid_floor_overlap"))
        if other_id:
            removed_ids.add(other_id)
        kept = [x for x in kept if _item_id(x) != other_id]
        kept_aabbs.pop(overlapping_index)
        kept.append(item)
        kept_aabbs.append((item, aabb))

    return kept, removed


def apply_procedural_semantic_polish(
    scene: dict[str, Any],
    *,
    room_type: str,
    size_class: str,
    density: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return polished scene and report.

    This function is safe to call repeatedly. It is deterministic and does not
    require external dependencies.
    """
    next_scene = copy.deepcopy(scene)
    room = next_scene.get("room") if isinstance(next_scene.get("room"), dict) else {}
    placements = next_scene.get("placements")
    if not isinstance(placements, list):
        report = {
            "schema": "procedural_semantic_polish/v1",
            "skipped": True,
            "reason": "missing_placements_list",
        }
        return next_scene, report

    before_count = len(placements)
    for item in placements:
        if isinstance(item, dict):
            normalize_item_semantics(item)

    items = [item for item in placements if isinstance(item, dict)]
    removed_all: list[dict[str, Any]] = []

    items, removed = remove_invalid_wardrobe_top_items(items)
    removed_all.extend(removed)

    items, removed = enforce_category_limits(items, room_type=room_type, size_class=size_class, density=density)
    removed_all.extend(removed)

    for item in items:
        normalize_item_semantics(item)

    items, removed = enforce_surface_limits(items)
    removed_all.extend(removed)

    items, removed = repair_solid_floor_overlaps(items)
    removed_all.extend(removed)

    items, removed = enforce_bedroom_functional_clearances(items, room=room)
    removed_all.extend(removed)

    items, removed = repair_wall_mounted_overlaps(items, room=room)
    removed_all.extend(removed)

    for item in items:
        normalize_item_semantics(item)

    annotate_layout_contracts(items, room_type=room_type)

    next_scene["placements"] = items

    roles = Counter(_physical_role(item) for item in items)
    by_category = Counter(_category(item) for item in items)
    by_mount = Counter(str(item.get("mount_type") or "unknown") for item in items)
    removed_by_reason = Counter(str(item.get("reason") or "unknown") for item in removed_all)

    report = {
        "schema": "procedural_semantic_polish/v1",
        "skipped": False,
        "room_type": room_type,
        "size_class": size_class,
        "density": density,
        "before_count": before_count,
        "after_count": len(items),
        "removed_count": len(removed_all),
        "removed_by_reason": dict(sorted(removed_by_reason.items())),
        "counts_by_physical_role": dict(sorted(roles.items())),
        "counts_by_mount_type": dict(sorted(by_mount.items())),
        "counts_by_category": dict(sorted(by_category.items())),
        "removed": removed_all,
    }
    return next_scene, report


def solid_floor_items(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in items if isinstance(item, dict) and _physical_role(item) == "solid_floor"]


def wall_mounted_items(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in items if isinstance(item, dict) and _physical_role(item) == "wall_mounted"]


def on_top_items(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in items if isinstance(item, dict) and _physical_role(item) in {"on_top", "soft_on_object"}]
