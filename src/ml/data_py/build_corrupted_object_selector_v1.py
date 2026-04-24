#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[3]))

try:
    from src.ml.data_py.build_repair_corruptions_v1 import (
        aabb_intersection_metrics,
        as_str,
        clone_placement,
        corners_inside_ratio,
        floor_contact_abs_error_m,
        footprint_area,
        is_floor_candidate,
        point_in_polygon,
        round6,
        should_ignore_pair_overlap,
        super_category_lower,
        update_aabb_for_placement,
    )
    from src.ml.data_py.repair_proposal_dataset_v1 import reconstruct_corrupted_scene, room_json_from_scene, load_sample_rows, load_json
except ModuleNotFoundError:
    from build_repair_corruptions_v1 import (  # type: ignore
        aabb_intersection_metrics,
        as_str,
        clone_placement,
        corners_inside_ratio,
        floor_contact_abs_error_m,
        footprint_area,
        is_floor_candidate,
        point_in_polygon,
        round6,
        should_ignore_pair_overlap,
        super_category_lower,
        update_aabb_for_placement,
    )
    from repair_proposal_dataset_v1 import reconstruct_corrupted_scene, room_json_from_scene, load_sample_rows, load_json  # type: ignore


def save_json(path: str | Path, data: Any) -> None:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build grouped object-selector dataset from repair_sample.v1 rows")
    ap.add_argument("--samples-jsonl", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--limit-val", type=int, default=0)
    ap.add_argument("--limit-test", type=int, default=0)
    ap.add_argument("--candidate-mode", default="furniture_components_v1", choices=["all_objects", "hard_v2", "furniture_components_v1"])
    ap.add_argument("--max-candidates", type=int, default=6)
    ap.add_argument("--component-limit", type=int, default=2)
    ap.add_argument("--global-fallback-k", type=int, default=3)
    ap.add_argument("--allowed-corruption-types", default="collision_shift,out_of_room_shift,shift_and_yaw")
    ap.add_argument("--allowed-room-types", default="")
    ap.add_argument("--allowed-target-categories", default="")
    ap.add_argument("--important-targets-only", action="store_true")
    return ap.parse_args()


IRRELEVANT_OBJECT_TOKENS = {
    "plate",
    "dish",
    "bowl",
    "cup",
    "mug",
    "glass",
    "bottle",
    "book",
    "books",
    "trinket",
    "decor",
    "decoration",
    "figurine",
    "pillow",
    "blanket",
    "cushion",
    "curtain",
    "rug",
    "carpet",
    "mat",
    "toy",
    "fruit",
    "food",
    "tableware",
}

IMPORTANT_OBJECT_TOKENS = {
    "bed",
    "sofa",
    "chair",
    "stool",
    "bench",
    "table",
    "desk",
    "nightstand",
    "cabinet",
    "wardrobe",
    "dresser",
    "bookcase",
    "bookshelf",
    "shelf",
    "sideboard",
    "tv stand",
    "tvstand",
    "console",
    "armoire",
    "vanity",
    "bathtub",
    "toilet",
    "washer",
    "washing machine",
    "refrigerator",
    "fridge",
    "oven",
    "dishwasher",
    "microwave",
    "lamp",
    "mirror",
}

IMPORTANT_SUPER_TOKENS = {
    "bed",
    "chair",
    "table",
    "cabinet",
    "shelf",
    "desk",
    "sofa",
    "lighting",
}

CENTRAL_FURNITURE_TOKENS = {
    "dining table",
    "coffee table",
    "corner/side table",
    "round end table",
    "dressing table",
    "table",
    "chair",
    "stool",
    "barstool",
    "armchair",
    "lounge chair",
    "dressing chair",
}

WALL_ANCHORED_OBJECT_TOKENS = {
    "wardrobe",
    "cabinet",
    "dresser",
    "drawer chest",
    "sideboard",
    "console",
    "tv stand",
    "tvstand",
    "bookcase",
    "bookshelf",
    "shelf",
    "wine cabinet",
    "children cabinet",
    "nightstand",
    "bed",
    "single bed",
    "king-size bed",
    "kids bed",
    "bunk bed",
    "refrigerator",
    "fridge",
    "oven",
    "dishwasher",
    "washer",
    "washing machine",
    "toilet",
    "bathtub",
    "vanity",
    "sink",
}


def object_text_blob(
    category: str,
    super_category: str,
    name: str = "",
    class_name: str = "",
    mount_type: str = "",
) -> str:
    return " ".join(
        x.strip().lower()
        for x in (category, super_category, name, class_name, mount_type)
        if as_str(x).strip()
    )


def is_important_furniture_metadata(
    category: str,
    super_category: str,
    name: str = "",
    class_name: str = "",
    mount_type: str = "",
    size_m: List[float] | None = None,
) -> bool:
    blob = object_text_blob(category, super_category, name, class_name, mount_type)
    if not blob:
        return False
    if any(tok in blob for tok in IRRELEVANT_OBJECT_TOKENS):
        return False
    if any(tok in as_str(super_category).strip().lower() for tok in IMPORTANT_SUPER_TOKENS):
        return True
    if any(tok in blob for tok in IMPORTANT_OBJECT_TOKENS):
        return True
    if size_m and len(size_m) >= 3:
        area = footprint_area(size_m)
        volume = max(float(size_m[0]) * float(size_m[1]) * float(size_m[2]), 0.0)
        if area >= 0.16 and volume >= 0.10:
            return True
    return False


def is_important_furniture_candidate(p: Dict[str, Any]) -> bool:
    if not is_floor_candidate(p):
        return False
    return is_important_furniture_metadata(
        category=as_str(p.get("category")),
        super_category=as_str((p.get("meta") or {}).get("super_category")),
        name=as_str(p.get("name")),
        class_name=as_str(p.get("class_name")),
        mount_type=as_str(p.get("mount_type")),
        size_m=[float(v) for v in (p.get("size_m") or [0.0, 0.0, 0.0])],
    )


def is_wall_anchored_expected(
    category: str,
    super_category: str,
    name: str = "",
    class_name: str = "",
) -> bool:
    blob = object_text_blob(category, super_category, name, class_name, "")
    if not blob:
        return False
    return any(tok in blob for tok in WALL_ANCHORED_OBJECT_TOKENS)


def is_central_furniture_expected(
    category: str,
    super_category: str,
    name: str = "",
    class_name: str = "",
) -> bool:
    blob = object_text_blob(category, super_category, name, class_name, "")
    if not blob:
        return False
    return any(tok in blob for tok in CENTRAL_FURNITURE_TOKENS)


def significant_collision_metrics_for_target(
    target: Dict[str, Any],
    placements: List[Dict[str, Any]],
    important_only: bool = True,
) -> Tuple[int, float, float, List[str]]:
    count = 0
    total_area = 0.0
    total_volume = 0.0
    colliding_ids: List[str] = []
    own_area = max(float(footprint_area(target["size_m"])), 1e-6)
    own_volume = max(float(target["size_m"][0]) * float(target["size_m"][1]) * float(target["size_m"][2]), 1e-6)
    for other in placements:
        if as_str(other.get("id")) == as_str(target.get("id")):
            continue
        if not is_floor_candidate(other):
            continue
        if important_only and not is_important_furniture_candidate(other):
            continue
        inter_area_2d, inter_volume_3d = aabb_intersection_metrics(target, other)
        if should_ignore_pair_overlap(target, other, inter_area_2d, inter_volume_3d):
            continue
        other_area = max(float(footprint_area(other["size_m"])), 1e-6)
        other_volume = max(float(other["size_m"][0]) * float(other["size_m"][1]) * float(other["size_m"][2]), 1e-6)
        rel_area = inter_area_2d / min(own_area, other_area)
        rel_vol = inter_volume_3d / min(own_volume, other_volume)
        if inter_volume_3d <= 1e-3 and rel_area <= 0.03 and rel_vol <= 0.01:
            continue
        if inter_volume_3d <= 0.01 and rel_area <= 0.08 and rel_vol <= 0.03:
            continue
        count += 1
        total_area += inter_area_2d
        total_volume += inter_volume_3d
        colliding_ids.append(as_str(other.get("id")))
    return count, float(total_area), float(total_volume), colliding_ids


def box_intersection_volume(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    ax1 = float(a["aabb"]["x_min"])
    ax2 = float(a["aabb"]["x_max"])
    ay1 = float(a["aabb"]["y_min"])
    ay2 = float(a["aabb"]["y_max"])
    az1 = float(a["aabb"]["z_min"])
    az2 = float(a["aabb"]["z_max"])
    bx1 = float(b["aabb"]["x_min"])
    bx2 = float(b["aabb"]["x_max"])
    by1 = float(b["aabb"]["y_min"])
    by2 = float(b["aabb"]["y_max"])
    bz1 = float(b["aabb"]["z_min"])
    bz2 = float(b["aabb"]["z_max"])
    dx = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    dy = max(0.0, min(ay2, by2) - max(ay1, by1))
    dz = max(0.0, min(az2, bz2) - max(az1, bz1))
    return float(dx * dy * dz)


def box_intersection_area_2d(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    ax1 = float(a["aabb"]["x_min"])
    ax2 = float(a["aabb"]["x_max"])
    ay1 = float(a["aabb"]["y_min"])
    ay2 = float(a["aabb"]["y_max"])
    bx1 = float(b["aabb"]["x_min"])
    bx2 = float(b["aabb"]["x_max"])
    by1 = float(b["aabb"]["y_min"])
    by2 = float(b["aabb"]["y_max"])
    dx = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    dy = max(0.0, min(ay2, by2) - max(ay1, by1))
    return float(dx * dy)


def safe_room_json(sample: Dict[str, Any], scene_gt: Dict[str, Any]) -> Dict[str, Any]:
    room_ref = as_str(sample.get("room_ref")).strip()
    if room_ref:
        room_path = Path(room_ref).expanduser()
        if room_path.exists() and room_path.is_file():
            return load_json(room_path)
    return room_json_from_scene(scene_gt)


def bounds_and_room_area(room_json: Dict[str, Any]) -> Tuple[Dict[str, float], float]:
    bounds = room_json["bounds_xz"]
    room_area = max(
        (float(bounds["x_max"]) - float(bounds["x_min"])) * (float(bounds["z_max"]) - float(bounds["z_min"])),
        1e-6,
    )
    return bounds, room_area


def center_norm(position_m: List[float], bounds: Dict[str, float]) -> List[float]:
    room_w = max(float(bounds["x_max"]) - float(bounds["x_min"]), 1e-6)
    room_h = max(float(bounds["z_max"]) - float(bounds["z_min"]), 1e-6)
    x = (float(position_m[0]) - float(bounds["x_min"])) / room_w
    y = (float(position_m[1]) - float(bounds["z_min"])) / room_h
    z = float(position_m[2])
    return [round6(x), round6(y), round6(z)]


def candidate_record(
    p: Dict[str, Any],
    placements: List[Dict[str, Any]],
    room_json: Dict[str, Any],
    room_area: float,
    target_id: str,
) -> Dict[str, Any]:
    room_polygon = room_json["floor_polygon_xz"]
    bounds = room_json["bounds_xz"]
    coll_count, coll_area, coll_volume, colliding_ids = significant_collision_metrics_for_target(
        p,
        placements,
        important_only=True,
    )
    inside_count, inside_ratio = corners_inside_ratio(p, room_polygon)
    center_inside = point_in_polygon((float(p["position_m"][0]), float(p["position_m"][1])), room_polygon)
    floor_err = floor_contact_abs_error_m(p)
    own_area = max(float(footprint_area(p["size_m"])), 1e-6)
    own_volume = max(float(p["size_m"][0]) * float(p["size_m"][1]) * float(p["size_m"][2]), 1e-6)
    important_furniture = is_important_furniture_candidate(p)
    wall_anchor_expected = is_wall_anchored_expected(
        category=as_str(p.get("category")),
        super_category=as_str((p.get("meta") or {}).get("super_category")),
        name=as_str(p.get("name")),
        class_name=as_str(p.get("class_name")),
    )
    central_furniture_expected = is_central_furniture_expected(
        category=as_str(p.get("category")),
        super_category=as_str((p.get("meta") or {}).get("super_category")),
        name=as_str(p.get("name")),
        class_name=as_str(p.get("class_name")),
    )
    room_dx = max(float(bounds["x_max"]) - float(bounds["x_min"]), 1e-6)
    room_dy = max(float(bounds["z_max"]) - float(bounds["z_min"]), 1e-6)
    room_center = np.asarray(
        [
            0.5 * (float(bounds["x_min"]) + float(bounds["x_max"])),
            0.5 * (float(bounds["z_min"]) + float(bounds["z_max"])),
        ],
        dtype=np.float32,
    )
    nearest_wall_dist = min(
        abs(float(p["aabb"]["x_min"]) - float(bounds["x_min"])),
        abs(float(bounds["x_max"]) - float(p["aabb"]["x_max"])),
        abs(float(p["aabb"]["y_min"]) - float(bounds["z_min"])),
        abs(float(bounds["z_max"]) - float(p["aabb"]["y_max"])),
    )
    nearest_wall_distance_norm = round6(float(nearest_wall_dist / max(min(room_dx, room_dy), 1e-6)))

    overlap_ratio_self_sum = 0.0
    overlap_ratio_self_max = 0.0
    overlap_ratio_minvol_max = 0.0
    overlap_area_ratio_self_sum = 0.0
    overlap_area_ratio_self_max = 0.0
    smaller_than_colliders = 0.0
    larger_than_colliders = 0.0
    strongest = None
    for other in placements:
        if as_str(other.get("id")) == as_str(p.get("id")) or not is_floor_candidate(other):
            continue
        if not is_important_furniture_candidate(other):
            continue
        inter_vol = box_intersection_volume(p, other)
        inter_area = box_intersection_area_2d(p, other)
        if should_ignore_pair_overlap(p, other, inter_area, inter_vol):
            continue
        if inter_vol <= 1e-9 and inter_area <= 1e-9:
            continue
        other_area = max(float(footprint_area(other["size_m"])), 1e-6)
        other_vol = max(float(other["size_m"][0]) * float(other["size_m"][1]) * float(other["size_m"][2]), 1e-6)
        overlap_ratio_self_sum += inter_vol / own_volume
        overlap_ratio_self_max = max(overlap_ratio_self_max, inter_vol / own_volume)
        overlap_ratio_minvol_max = max(overlap_ratio_minvol_max, inter_vol / min(own_volume, other_vol))
        overlap_area_ratio_self_sum += inter_area / own_area
        overlap_area_ratio_self_max = max(overlap_area_ratio_self_max, inter_area / own_area)
        strength = (inter_vol / min(own_volume, other_vol), inter_vol, inter_area)
        if strongest is None or strength > strongest["strength"]:
            strongest = {
                "strength": strength,
                "other": other,
                "inter_vol": inter_vol,
                "inter_area": inter_area,
                "other_area": other_area,
                "other_vol": other_vol,
            }
        if own_area < other_area * 0.9:
            smaller_than_colliders += 1.0
        elif own_area > other_area * 1.1:
            larger_than_colliders += 1.0

    outside_room = (not center_inside) or (inside_count == 0)
    yaw_deg = float(p.get("yaw_deg", 0.0))
    yaw_rad = np.deg2rad(yaw_deg)
    cnorm = center_norm(p["position_m"], bounds)
    strongest_other = strongest["other"] if strongest is not None else None
    strongest_center_norm = center_norm(strongest_other["position_m"], bounds) if strongest_other is not None else [0.0, 0.0, 0.0]
    room_diag = max((room_dx ** 2 + room_dy ** 2) ** 0.5, 1e-6)
    room_center_distance_norm = round6(
        float(
            np.linalg.norm(np.asarray(p["position_m"][:2], dtype=np.float32) - room_center)
            / room_diag
        )
    )
    strongest_center_distance_norm = 0.0
    strongest_size = [0.0, 0.0, 0.0]
    strongest_area_ratio_other = 0.0
    strongest_volume_ratio_other = 0.0
    strongest_intersection_volume = 0.0
    strongest_intersection_area = 0.0
    strongest_volume_ratio_self = 0.0
    strongest_volume_ratio_minvol = 0.0
    strongest_area_ratio_self = 0.0
    if strongest_other is not None:
        strongest_size = [round6(float(v)) for v in strongest_other["size_m"]]
        strongest_center_distance_norm = round6(
            float(
                np.linalg.norm(
                    np.asarray(p["position_m"][:2], dtype=np.float32)
                    - np.asarray(strongest_other["position_m"][:2], dtype=np.float32)
                )
                / room_diag
            )
        )
        strongest_intersection_volume = round6(float(strongest["inter_vol"]))
        strongest_intersection_area = round6(float(strongest["inter_area"]))
        strongest_volume_ratio_self = round6(float(strongest["inter_vol"] / own_volume))
        strongest_volume_ratio_minvol = round6(float(strongest["inter_vol"] / min(own_volume, strongest["other_vol"])))
        strongest_area_ratio_self = round6(float(strongest["inter_area"] / own_area))
        strongest_area_ratio_other = round6(float(strongest["other_area"] / own_area))
        strongest_volume_ratio_other = round6(float(strongest["other_vol"] / own_volume))
    isolated_layout_anomaly_score = 0.0
    if important_furniture and coll_count == 0:
        size_bonus = min((own_area / max(room_area, 1e-6)) * 6.0, 3.0)
        wall_miss_bonus = 0.0
        if wall_anchor_expected:
            wall_miss_bonus = max(float(nearest_wall_distance_norm) - 0.08, 0.0) * 8.0
        center_intrusion_bonus = 0.0
        if (not central_furniture_expected) and own_area / max(room_area, 1e-6) >= 0.04:
            center_intrusion_bonus = max(0.22 - float(room_center_distance_norm), 0.0) * 8.0
        isolated_layout_anomaly_score = size_bonus + wall_miss_bonus + center_intrusion_bonus + min(floor_err / 0.04, 2.0)
    node_suspect_score = (
        (4.0 if outside_room else 0.0)
        + min(floor_err / 0.04, 3.0)
        + 1.5 * float(coll_count)
        + 4.0 * overlap_ratio_self_max
        + 2.0 * overlap_area_ratio_self_max
        + (1.5 if wall_anchor_expected and nearest_wall_distance_norm >= 0.12 else 0.0)
        + 0.6 * isolated_layout_anomaly_score
        + 0.4 * smaller_than_colliders
        - 0.25 * larger_than_colliders
        - 0.5 * min(own_area / max(room_area, 1e-6), 1.0)
    )
    if not important_furniture:
        node_suspect_score -= 10.0
    return {
        "id": as_str(p.get("id")),
        "name": as_str(p.get("name")),
        "class_name": as_str(p.get("class_name")),
        "category": as_str(p.get("category")),
        "super_category": as_str((p.get("meta") or {}).get("super_category")),
        "mount_type": as_str(p.get("mount_type")),
        "center_norm": cnorm,
        "size_m": [round6(float(v)) for v in p["size_m"]],
        "yaw_sin": round6(float(np.sin(yaw_rad))),
        "yaw_cos": round6(float(np.cos(yaw_rad))),
        "footprint_area_2d": round6(own_area),
        "bbox_volume_3d": round6(own_volume),
        "area_ratio_to_room": round6(own_area / room_area),
        "collision_pair_count": int(coll_count),
        "collision_area_sum_2d": round6(coll_area),
        "collision_volume_sum_3d": round6(coll_volume),
        "collision_area_ratio_self_sum": round6(overlap_area_ratio_self_sum),
        "collision_area_ratio_self_max": round6(overlap_area_ratio_self_max),
        "collision_volume_ratio_self_sum": round6(overlap_ratio_self_sum),
        "collision_volume_ratio_self_max": round6(overlap_ratio_self_max),
        "collision_volume_ratio_minvol_max": round6(overlap_ratio_minvol_max),
        "corners_inside_count": int(inside_count),
        "corners_inside_ratio": round6(inside_ratio),
        "center_inside_room": bool(center_inside),
        "outside_room": bool(outside_room),
        "floor_contact_abs_error_m": round6(floor_err),
        "smaller_than_colliders_count": round6(smaller_than_colliders),
        "larger_than_colliders_count": round6(larger_than_colliders),
        "is_bad_by_bbox": bool(coll_count > 0 or outside_room or floor_err > 0.08),
        "colliding_with_ids": colliding_ids,
        "important_furniture_candidate": bool(important_furniture),
        "wall_anchor_expected": bool(wall_anchor_expected),
        "central_furniture_expected": bool(central_furniture_expected),
        "nearest_wall_distance_norm": nearest_wall_distance_norm,
        "room_center_distance_norm": room_center_distance_norm,
        "isolated_layout_anomaly_score": round6(isolated_layout_anomaly_score),
        "node_suspect_score": round6(node_suspect_score),
        "strongest_collider_category": as_str((strongest_other or {}).get("category")),
        "strongest_collider_super_category": as_str((((strongest_other or {}).get("meta") or {}).get("super_category"))),
        "strongest_collider_mount_type": as_str((strongest_other or {}).get("mount_type")),
        "strongest_collider_center_norm": strongest_center_norm,
        "strongest_collider_size_m": strongest_size,
        "strongest_collider_center_distance_norm": strongest_center_distance_norm,
        "strongest_intersection_volume_3d": strongest_intersection_volume,
        "strongest_intersection_area_2d": strongest_intersection_area,
        "strongest_collision_volume_ratio_self": strongest_volume_ratio_self,
        "strongest_collision_volume_ratio_minvol": strongest_volume_ratio_minvol,
        "strongest_collision_area_ratio_self": strongest_area_ratio_self,
        "strongest_collider_area_ratio_other_to_self": strongest_area_ratio_other,
        "strongest_collider_volume_ratio_other_to_self": strongest_volume_ratio_other,
        "is_target": bool(as_str(p.get("id")) == target_id),
        "component_id": -1,
        "component_size": 0,
        "component_score": 0.0,
        "component_rank_in_scene": -1,
        "suspect_rank_in_component": -1,
        "global_anomaly_score": 0.0,
        "global_suspect_rank": -1,
    }


def selector_severity_tuple(rec: Dict[str, Any]) -> Tuple[float, ...]:
    return (
        1.0 if rec.get("important_furniture_candidate") else 0.0,
        float(rec.get("component_score", 0.0)),
        -float(rec.get("suspect_rank_in_component", 999.0)),
        float(rec.get("isolated_layout_anomaly_score", 0.0)),
        float(rec.get("node_suspect_score", 0.0)),
        float(rec["collision_pair_count"]),
        float(rec["collision_volume_ratio_self_max"]),
        float(rec["collision_volume_ratio_self_sum"]),
        1.0 if rec["outside_room"] else 0.0,
        float(rec["floor_contact_abs_error_m"]),
        float(rec["collision_volume_sum_3d"]),
        float(rec["collision_area_ratio_self_max"]),
        float(rec["collision_area_sum_2d"]),
    )


def global_anomaly_tuple(rec: Dict[str, Any]) -> Tuple[float, ...]:
    return (
        1.0 if rec.get("important_furniture_candidate") else 0.0,
        float(rec.get("isolated_layout_anomaly_score", 0.0)),
        1.0 if rec.get("outside_room") else 0.0,
        float(rec.get("collision_pair_count", 0.0)),
        float(rec.get("node_suspect_score", 0.0)),
        1.0 if rec.get("wall_anchor_expected") else 0.0,
        -float(rec.get("room_center_distance_norm", 1.0)),
        -float(rec.get("nearest_wall_distance_norm", 1.0)),
        float(rec.get("area_ratio_to_room", 0.0)),
    )


def build_furniture_components(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_id = {as_str(r["id"]): r for r in records if as_str(r["id"])}
    eligible_ids = {
        rid
        for rid, rec in by_id.items()
        if rec.get("important_furniture_candidate")
    }
    adjacency: Dict[str, set[str]] = defaultdict(set)
    active_ids: set[str] = set()
    for rid in eligible_ids:
        rec = by_id[rid]
        neighbors = {
            nid
            for nid in (rec.get("colliding_with_ids") or [])
            if as_str(nid) in eligible_ids
        }
        for nid in neighbors:
            adjacency[rid].add(as_str(nid))
            adjacency[as_str(nid)].add(rid)
        is_active = (
            bool(rec.get("outside_room"))
            or float(rec.get("floor_contact_abs_error_m", 0.0)) > 0.08
            or int(rec.get("collision_pair_count", 0)) > 0
            or float(rec.get("collision_volume_ratio_self_max", 0.0)) > 1e-6
            or float(rec.get("isolated_layout_anomaly_score", 0.0)) >= 1.5
        )
        if is_active:
            active_ids.add(rid)
            active_ids.update(neighbors)

    if not active_ids:
        ranked_ids = sorted(
            eligible_ids,
            key=lambda rid: selector_severity_tuple(by_id[rid]),
            reverse=True,
        )
        active_ids.update(ranked_ids[:2])

    seen: set[str] = set()
    components: List[Dict[str, Any]] = []
    for rid in sorted(active_ids):
        if rid in seen:
            continue
        stack = [rid]
        comp_ids: List[str] = []
        seen.add(rid)
        while stack:
            cur = stack.pop()
            comp_ids.append(cur)
            for nxt in adjacency.get(cur, set()):
                if nxt in active_ids and nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        if not comp_ids:
            continue
        comp_recs = [dict(by_id[x]) for x in comp_ids]
        comp_recs = sorted(comp_recs, key=selector_severity_tuple, reverse=True)
        comp_score = sum(max(float(r.get("node_suspect_score", 0.0)), 0.0) for r in comp_recs)
        comp_score += 0.75 * sum(float(r.get("collision_pair_count", 0.0)) for r in comp_recs)
        comp_score += 2.0 * sum(1.0 for r in comp_recs if r.get("outside_room"))
        components.append(
            {
                "ids": [as_str(r["id"]) for r in comp_recs],
                "records": comp_recs,
                "score": round6(comp_score),
            }
        )

    components.sort(key=lambda c: (float(c["score"]), len(c["ids"])), reverse=True)
    enriched: List[Dict[str, Any]] = []
    for comp_rank, comp in enumerate(components, start=1):
        comp_recs = []
        for suspect_rank, rec in enumerate(comp["records"], start=1):
            crec = dict(rec)
            crec["component_id"] = comp_rank - 1
            crec["component_size"] = len(comp["records"])
            crec["component_score"] = round6(float(comp["score"]))
            crec["component_rank_in_scene"] = comp_rank
            crec["suspect_rank_in_component"] = suspect_rank
            comp_recs.append(crec)
        enriched.append(
            {
                "ids": [as_str(r["id"]) for r in comp_recs],
                "records": comp_recs,
                "score": round6(float(comp["score"])),
            }
        )

    inactive_ids = [rid for rid in eligible_ids if rid not in {x for comp in enriched for x in comp["ids"]}]
    singleton_candidates = []
    for rid in inactive_ids:
        rec = dict(by_id[rid])
        singleton_score = float(rec.get("node_suspect_score", 0.0))
        singleton_score += 1.0 if rec.get("wall_anchor_expected") and float(rec.get("nearest_wall_distance_norm", 0.0)) >= 0.12 else 0.0
        singleton_score += min(float(rec.get("area_ratio_to_room", 0.0)) * 4.0, 1.5)
        singleton_candidates.append((singleton_score, rec))
    singleton_candidates.sort(key=lambda x: x[0], reverse=True)
    for single_rank, (score, rec) in enumerate(singleton_candidates[:4], start=len(enriched) + 1):
        crec = dict(rec)
        crec["component_id"] = single_rank - 1
        crec["component_size"] = 1
        crec["component_score"] = round6(float(score))
        crec["component_rank_in_scene"] = single_rank
        crec["suspect_rank_in_component"] = 1
        enriched.append(
            {
                "ids": [as_str(crec["id"])],
                "records": [crec],
                "score": round6(float(score)),
            }
        )
    enriched.sort(key=lambda c: (float(c["score"]), len(c["ids"])), reverse=True)
    for comp_rank, comp in enumerate(enriched, start=1):
        for suspect_rank, rec in enumerate(comp["records"], start=1):
            rec["component_id"] = comp_rank - 1
            rec["component_rank_in_scene"] = comp_rank
            rec["suspect_rank_in_component"] = suspect_rank
    return enriched


def select_candidate_records(
    records: List[Dict[str, Any]],
    target_id: str = "",
    mode: str = "furniture_components_v1",
    max_candidates: int = 6,
    component_limit: int = 2,
    global_fallback_k: int = 3,
) -> Tuple[List[Dict[str, Any]], bool]:
    if mode == "all_objects":
        return list(records), any(as_str(r["id"]) == target_id for r in records)

    if mode == "furniture_components_v1":
        components = build_furniture_components(records)
        selected_components: List[Dict[str, Any]] = []
        target_in_active = False
        global_ranked = sorted(
            [dict(r) for r in records if r.get("important_furniture_candidate")],
            key=global_anomaly_tuple,
            reverse=True,
        )
        for rank, rec in enumerate(global_ranked, start=1):
            rec["global_suspect_rank"] = rank
            rec["global_anomaly_score"] = round6(float(rec.get("isolated_layout_anomaly_score", 0.0)) + float(rec.get("node_suspect_score", 0.0)))
        if target_id:
            selected_components = [c for c in components if target_id in set(c["ids"])]
            if not selected_components:
                target_rec = next((dict(r) for r in records if as_str(r["id"]) == target_id), None)
                if target_rec is None:
                    raise RuntimeError("Target not found among selector candidates")
                target_rec["component_id"] = len(components)
                target_rec["component_size"] = 1
                target_rec["component_score"] = round6(float(target_rec.get("node_suspect_score", 0.0)))
                target_rec["component_rank_in_scene"] = len(components) + 1
                target_rec["suspect_rank_in_component"] = 1
                selected_components = [
                    {
                        "ids": [target_id],
                        "records": [target_rec],
                        "score": round6(float(target_rec.get("node_suspect_score", 0.0))),
                    }
                ]
            else:
                target_in_active = True
        else:
            selected_components = components[: max(1, int(component_limit))]
        active = [rec for comp in selected_components for rec in comp["records"]]
        selected_ids = {as_str(rec["id"]) for rec in active}
        for rec in global_ranked[: max(0, int(global_fallback_k))]:
            rid = as_str(rec["id"])
            if rid in selected_ids:
                continue
            active.append(rec)
            selected_ids.add(rid)
        active = sorted(active, key=selector_severity_tuple, reverse=True)
        if target_id and not target_in_active:
            target_in_active = any(as_str(r["id"]) == target_id for comp in components for r in comp["records"]) or any(
                as_str(r["id"]) == target_id for r in global_ranked[: max(0, int(global_fallback_k))]
            )
        if int(max_candidates) > 0 and len(active) > int(max_candidates):
            keep = active[: int(max_candidates)]
            if target_id and not any(as_str(r["id"]) == target_id for r in keep):
                target_rec = next((r for r in active if as_str(r["id"]) == target_id), None)
                if target_rec is not None:
                    keep[-1] = target_rec
            dedup = []
            seen = set()
            for rec in keep:
                rid = as_str(rec["id"])
                if not rid or rid in seen:
                    continue
                seen.add(rid)
                dedup.append(rec)
            active = dedup
        return active, target_in_active

    active_ids = set()
    for rec in records:
        if rec["is_bad_by_bbox"] or float(rec["collision_volume_ratio_self_max"]) > 1e-6:
            active_ids.add(as_str(rec["id"]))
            active_ids.update(as_str(x) for x in rec.get("colliding_with_ids") or [])

    if not active_ids:
        ranked = sorted(records, key=selector_severity_tuple, reverse=True)
        active_ids = {as_str(r["id"]) for r in ranked[: max(1, int(max_candidates))]}

    active = [r for r in records if as_str(r["id"]) in active_ids]
    active = sorted(active, key=selector_severity_tuple, reverse=True)
    target_in_active = any(as_str(r["id"]) == target_id for r in active)

    if int(max_candidates) > 0 and len(active) > int(max_candidates):
        keep = active[: int(max_candidates)]
        if target_id and not any(as_str(r["id"]) == target_id for r in keep):
            target_rec = next((r for r in active if as_str(r["id"]) == target_id), None)
            if target_rec is not None:
                keep[-1] = target_rec
        dedup = []
        seen = set()
        for rec in keep:
            rid = as_str(rec["id"])
            if not rid or rid in seen:
                continue
            seen.add(rid)
            dedup.append(rec)
        active = dedup
    return active, target_in_active


def build_group_row(
    sample: Dict[str, Any],
    candidate_mode: str,
    max_candidates: int,
    component_limit: int,
    global_fallback_k: int,
) -> Dict[str, Any]:
    scene_gt, corrupted_scene, _ = reconstruct_corrupted_scene(sample)
    placements = []
    for p in corrupted_scene:
        cp = clone_placement(p)
        cp["aabb"] = update_aabb_for_placement(cp)
        placements.append(cp)
    room_json = safe_room_json(sample, scene_gt)
    _, room_area = bounds_and_room_area(room_json)
    target_id = as_str(sample["target_object_id"])
    floor_candidates = [p for p in placements if as_str(p.get("id")).strip()]
    seen_ids = set()
    unique_candidates = []
    for p in floor_candidates:
        pid = as_str(p.get("id"))
        if not pid or pid in seen_ids:
            continue
        seen_ids.add(pid)
        unique_candidates.append(p)
    all_candidates = [candidate_record(p, placements, room_json, room_area, target_id) for p in unique_candidates]
    candidates, target_in_active = select_candidate_records(
        all_candidates,
        target_id=target_id,
        mode=candidate_mode,
        max_candidates=max_candidates,
        component_limit=component_limit,
        global_fallback_k=global_fallback_k,
    )
    target_candidate_index = next((i for i, c in enumerate(candidates) if c["is_target"]), -1)
    if target_candidate_index < 0:
        raise RuntimeError(f"Target not found among selector candidates for sample_id={sample['sample_id']}")
    return {
        "schema": "corrupted_object_selector_sample.v1",
        "sample_id": as_str(sample["sample_id"]),
        "room_id": as_str(sample.get("room_id")),
        "room_type": as_str(sample.get("room_type")),
        "split": as_str(sample.get("split")),
        "target_object_id": target_id,
        "target_candidate_index": int(target_candidate_index),
        "candidate_mode": candidate_mode,
        "num_all_candidates": len(all_candidates),
        "target_in_active_before_force": bool(target_in_active),
        "candidates": candidates,
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "samples.jsonl"

    limits = {
        "train": int(args.limit_train),
        "val": int(args.limit_val),
        "test": int(args.limit_test),
    }
    allowed_corruption_types = {x.strip() for x in as_str(args.allowed_corruption_types).split(",") if x.strip()}
    allowed_room_types = {x.strip() for x in as_str(args.allowed_room_types).split(",") if x.strip()}
    allowed_target_categories = {x.strip() for x in as_str(args.allowed_target_categories).split(",") if x.strip()}
    split_counts: Counter[str] = Counter()
    cat_counts: Counter[str] = Counter()
    target_cat_counts: Counter[str] = Counter()
    total_candidates = 0
    sample_count = 0
    skipped_missing_target = 0
    skipped_corruption_type = 0
    skipped_room_type = 0
    skipped_target_category = 0
    skipped_unimportant_target = 0
    target_in_active_count = 0
    total_all_candidates = 0

    with out_jsonl.open("w", encoding="utf-8") as out_f:
        for split in ("train", "val", "test"):
            rows = load_sample_rows(args.samples_jsonl, split=split, limit=limits[split])
            for row in rows:
                if allowed_corruption_types and as_str(row.get("corruption_type")) not in allowed_corruption_types:
                    skipped_corruption_type += 1
                    continue
                if allowed_room_types and as_str(row.get("room_type")) not in allowed_room_types:
                    skipped_room_type += 1
                    continue
                if allowed_target_categories and as_str(row.get("target_category")) not in allowed_target_categories:
                    skipped_target_category += 1
                    continue
                if args.important_targets_only and not is_important_furniture_metadata(
                    category=as_str(row.get("target_category")),
                    super_category=as_str(row.get("target_super_category")),
                    name=as_str(row.get("target_category")),
                    class_name="",
                    mount_type="floor",
                    size_m=None,
                ):
                    skipped_unimportant_target += 1
                    continue
                try:
                    group = build_group_row(
                        row,
                        candidate_mode=args.candidate_mode,
                        max_candidates=int(args.max_candidates),
                        component_limit=int(args.component_limit),
                        global_fallback_k=int(args.global_fallback_k),
                    )
                except RuntimeError as e:
                    if "Target not found among selector candidates" in str(e):
                        skipped_missing_target += 1
                        continue
                    raise
                out_f.write(json.dumps(group, ensure_ascii=False) + "\n")
                sample_count += 1
                split_counts[split] += 1
                total_candidates += len(group["candidates"])
                total_all_candidates += int(group["num_all_candidates"])
                target_in_active_count += 1 if group["target_in_active_before_force"] else 0
                for cand in group["candidates"]:
                    cat_counts[as_str(cand["category"])] += 1
                    if cand["is_target"]:
                        target_cat_counts[as_str(cand["category"])] += 1

    stats = {
        "samples_total": sample_count,
        "skipped_missing_target": skipped_missing_target,
        "skipped_corruption_type": skipped_corruption_type,
        "skipped_room_type": skipped_room_type,
        "skipped_target_category": skipped_target_category,
        "skipped_unimportant_target": skipped_unimportant_target,
        "samples_by_split": dict(split_counts),
        "mean_candidates_per_sample": round6(total_candidates / max(sample_count, 1)),
        "mean_all_candidates_per_sample": round6(total_all_candidates / max(sample_count, 1)),
        "target_in_active_before_force_rate": round6(target_in_active_count / max(sample_count, 1)),
        "candidate_mode": as_str(args.candidate_mode),
        "allowed_corruption_types": sorted(allowed_corruption_types),
        "allowed_room_types": sorted(allowed_room_types),
        "allowed_target_categories": sorted(allowed_target_categories),
        "important_targets_only": bool(args.important_targets_only),
        "candidate_categories_top50": dict(cat_counts.most_common(50)),
        "target_categories_top50": dict(target_cat_counts.most_common(50)),
    }
    save_json(out_dir / "stats.json", stats)
    print(f"[corrupted_object_selector_v1] samples={sample_count}")
    print(f"[corrupted_object_selector_v1] skipped_missing_target={skipped_missing_target}")
    print(f"[corrupted_object_selector_v1] skipped_corruption_type={skipped_corruption_type}")
    print(f"[corrupted_object_selector_v1] skipped_room_type={skipped_room_type}")
    print(f"[corrupted_object_selector_v1] skipped_target_category={skipped_target_category}")
    print(f"[corrupted_object_selector_v1] skipped_unimportant_target={skipped_unimportant_target}")
    print(f"[corrupted_object_selector_v1] mean_candidates_per_sample={stats['mean_candidates_per_sample']}")
    print(f"[corrupted_object_selector_v1] target_in_active_before_force_rate={stats['target_in_active_before_force_rate']}")
    print(f"[corrupted_object_selector_v1] wrote samples={out_jsonl}")
    print(f"[corrupted_object_selector_v1] wrote stats={out_dir / 'stats.json'}")


if __name__ == "__main__":
    main()
