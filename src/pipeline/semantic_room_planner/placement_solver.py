from __future__ import annotations

import math
from typing import Any


MAIN_ORDER = [
    "bed", "sofa", "dining_table", "kitchen_table", "kitchen_counter", "fridge", "kitchen_cabinet",
    "desk", "wardrobe", "dresser", "shelf", "bookcase", "toilet", "sink", "shower", "bathtub", "oven",
]
DEP_ORDER = ["office_chair", "chair", "dining_chair", "nightstand", "coffee_table", "side_table", "floor_lamp", "plant", "rug", "bath_mat"]
DEFAULT_MAX_CANDIDATES_PER_OBJECT = 16
DEFAULT_MAX_TOTAL_CANDIDATE_COMBINATIONS = 512


def _aabb(x: float, y: float, z: float, w: float, d: float, h: float) -> dict[str, float]:
    return {"x_min": x - w / 2, "x_max": x + w / 2, "y_min": y - d / 2, "y_max": y + d / 2, "z_min": z, "z_max": z + h}


def _intersects(a: dict[str, float], b: dict[str, float], pad: float = 0.03) -> bool:
    return not (a["x_max"] + pad <= b["x_min"] or b["x_max"] + pad <= a["x_min"] or a["y_max"] + pad <= b["y_min"] or b["y_max"] + pad <= a["y_min"] or a["z_max"] <= b["z_min"] or b["z_max"] <= a["z_min"])


def _inside(a: dict[str, float], bbox: dict[str, float]) -> bool:
    return a["x_min"] >= bbox["x_min"] and a["x_max"] <= bbox["x_max"] and a["y_min"] >= bbox["y_min"] and a["y_max"] <= bbox["y_max"]


def _is_free_floor_aabb(aabb: dict[str, float], bbox: dict[str, float], placed: dict[str, dict[str, Any]], obj_id: str) -> bool:
    if not _inside(aabb, bbox):
        return False
    return not any(_intersects(aabb, p["aabb"]) for p in placed.values() if p.get("object_id") != obj_id and p["aabb"]["z_min"] < 0.2)


def _find_free_floor_position(room_geometry: dict[str, Any], dims: dict[str, float], placed: dict[str, dict[str, Any]], obj_id: str, preferred: list[tuple[float, float]] | None = None) -> tuple[float, float] | None:
    bbox = room_geometry["bbox"]
    for tx, ty in preferred or []:
        aa = _aabb(tx, ty, 0.0, dims["width"], dims["depth"], dims["height"])
        if _is_free_floor_aabb(aa, bbox, placed, obj_id):
            return tx, ty
    step = 0.35
    grid_side = 10
    for row in range(grid_side):
        for col in range(grid_side):
            tx = bbox["x_min"] + 0.25 + col * step
            ty = bbox["y_min"] + 0.25 + row * step
            aa = _aabb(tx, ty, 0.0, dims["width"], dims["depth"], dims["height"])
            if _is_free_floor_aabb(aa, bbox, placed, obj_id):
                return tx, ty
    return None


def _yaw_to(src: dict[str, float], dst: dict[str, float]) -> float:
    return math.degrees(math.atan2(dst["y"] - src["y"], dst["x"] - src["x"])) - 90.0


def generate_main_object_candidates(room_geometry: dict[str, Any], obj: dict[str, Any], zones: list[dict[str, Any]], relations: dict[str, Any], max_candidates: int = DEFAULT_MAX_CANDIDATES_PER_OBJECT) -> list[dict[str, Any]]:
    bbox = room_geometry["bbox"]
    c = room_geometry["center"]
    w = obj["dimensions_m"]["width"]
    d = obj["dimensions_m"]["depth"]
    margin = 0.05
    x_positions = [bbox["x_min"] + w / 2 + margin, c["x"], bbox["x_max"] - w / 2 - margin]
    y_positions = [bbox["y_min"] + d / 2 + margin, c["y"], bbox["y_max"] - d / 2 - margin]
    candidates = []
    for y in y_positions:
        candidates.append((bbox["x_min"] + w / 2 + margin, y, 90, "against left wall"))
        candidates.append((bbox["x_max"] - w / 2 - margin, y, -90, "against right wall"))
    for x in x_positions:
        candidates.append((x, bbox["y_min"] + d / 2 + margin, 180, "against bottom wall"))
        candidates.append((x, bbox["y_max"] - d / 2 - margin, 0, "against top wall"))
    if obj["subclass"] in {"dining_table", "kitchen_table", "coffee_table", "rug"}:
        candidates.insert(0, (c["x"], c["y"], 0, "near room center"))
    return [{"x": x, "y": y, "yaw": yaw, "reason": reason} for x, y, yaw, reason in candidates[:max(1, max_candidates)]]


def score_candidate(room_geometry: dict[str, Any], obj: dict[str, Any], cand: dict[str, Any], placed: dict[str, dict[str, Any]]) -> float:
    dims = obj["dimensions_m"]
    aabb = _aabb(cand["x"], cand["y"], 0.0, dims["width"], dims["depth"], dims["height"])
    score = 0.0
    score += 0.35 if _inside(aabb, room_geometry["bbox"]) else -1.0
    score += 0.35 if not any(_intersects(aabb, p["aabb"]) for p in placed.values() if p["aabb"]["z_min"] < 0.2 and p["object_id"] != obj["id"]) else -1.0
    score += 0.15 if "wall" in cand.get("reason", "") else 0.05
    score += 0.15
    return score


def _find_obj(objects: list[dict[str, Any]], oid: str) -> dict[str, Any] | None:
    return next((o for o in objects if o["id"] == oid), None)


def _target_edge(edges: list[dict[str, Any]], obj_id: str, rels: set[str]) -> dict[str, Any] | None:
    priority = {"above": 0, "on_top_of": 1, "inside": 2, "under": 3, "next_to": 4, "in_front_of": 5, "around": 6, "near": 7, "mounted_on_wall": 8, "against_wall": 9}
    candidates = [e for e in edges if e.get("from_id") == obj_id and e.get("relation_type") in rels]
    return min(candidates, key=lambda e: priority.get(str(e.get("relation_type")), 50), default=None)


DESK_SLOT_ORDER: dict[str, list[str]] = {
    "monitor": ["back_center", "back_left", "back_right"],
    "keyboard": ["front_center", "left_front", "right_front"],
    "laptop": ["left_front", "center", "front_center"],
    "mouse": ["right_front", "right_side"],
    "mug": ["right_side", "right_front", "back_right"],
    "cup": ["right_side", "right_front", "back_right"],
    "water_bottle": ["back_right", "right_side"],
    "table_lamp": ["back_left", "left_side"],
    "desk_lamp": ["back_left", "left_side"],
    "notebook": ["left_side", "left_front", "front_center"],
    "desk_organizer": ["back_right", "back_left"],
    "plant": ["back_right", "back_left"],
    "stove": ["back_center", "back_left", "back_right"],
    "kitchen_sink": ["left_side", "back_left"],
    "kettle": ["back_right", "right_side"],
    "cutting_board": ["front_center", "left_front"],
    "fruit_bowl": ["center", "back_center"],
    "cookbook": ["left_front", "left_side"],
    "pan": ["back_center", "back_left"],
    "pot": ["back_center", "back_right"],
    "plate": ["front_center", "left_front", "right_front"],
    "bowl": ["center", "front_center"],
    "vase": ["center", "back_center"],
    "remote": ["center", "right_side"],
    "soap_dispenser": ["right_side", "right_front"],
    "toothbrush_cup": ["left_side", "left_front"],
}


GENERIC_SLOT_ORDER = ["center", "left_front", "right_front", "back_left", "back_right", "front_center", "back_center", "left_side", "right_side"]


def _slot_offset(slot: str, target_aabb: dict[str, float], obj_dims: dict[str, float]) -> tuple[float, float]:
    tw = target_aabb["x_max"] - target_aabb["x_min"]
    td = target_aabb["y_max"] - target_aabb["y_min"]
    ow = obj_dims["width"]
    od = obj_dims["depth"]
    max_x = max(0.0, tw / 2 - ow / 2 - 0.04)
    max_y = max(0.0, td / 2 - od / 2 - 0.04)
    table = {
        "center": (0.0, 0.0),
        "left_front": (-max_x, max_y),
        "right_front": (max_x, max_y),
        "front_center": (0.0, max_y),
        "back_center": (0.0, -max_y),
        "back_left": (-max_x, -max_y),
        "back_right": (max_x, -max_y),
        "left_side": (-max_x, 0.0),
        "right_side": (max_x, 0.0),
    }
    return table.get(slot, table["center"])


def _clamp_center_to_bbox(x: float, y: float, dims: dict[str, float], bbox: dict[str, float]) -> tuple[float, float, bool]:
    nx = min(max(x, bbox["x_min"] + dims["width"] / 2), bbox["x_max"] - dims["width"] / 2)
    ny = min(max(y, bbox["y_min"] + dims["depth"] / 2), bbox["y_max"] - dims["depth"] / 2)
    return nx, ny, abs(nx - x) > 1e-6 or abs(ny - y) > 1e-6


def _wall_position_near_target(room_geometry: dict[str, Any], dims: dict[str, float], target: dict[str, Any]) -> tuple[float, float, float]:
    bbox = room_geometry["bbox"]
    tc = target["position"]
    distances = [
        ("left", abs(tc["x"] - bbox["x_min"])),
        ("right", abs(bbox["x_max"] - tc["x"])),
        ("bottom", abs(tc["y"] - bbox["y_min"])),
        ("top", abs(bbox["y_max"] - tc["y"])),
    ]
    wall = min(distances, key=lambda x: x[1])[0]
    if wall == "left":
        return bbox["x_min"] + dims["width"] / 2 + 0.01, min(max(tc["y"], bbox["y_min"] + dims["depth"] / 2), bbox["y_max"] - dims["depth"] / 2), 90.0
    if wall == "right":
        return bbox["x_max"] - dims["width"] / 2 - 0.01, min(max(tc["y"], bbox["y_min"] + dims["depth"] / 2), bbox["y_max"] - dims["depth"] / 2), -90.0
    if wall == "bottom":
        return min(max(tc["x"], bbox["x_min"] + dims["width"] / 2), bbox["x_max"] - dims["width"] / 2), bbox["y_min"] + dims["depth"] / 2 + 0.01, 180.0
    return min(max(tc["x"], bbox["x_min"] + dims["width"] / 2), bbox["x_max"] - dims["width"] / 2), bbox["y_max"] - dims["depth"] / 2 - 0.01, 0.0


def _inside_position(obj: dict[str, Any], target: dict[str, Any], edge: dict[str, Any], occupied: dict[str, set[str]]) -> tuple[float, float, float, str]:
    ta = target["aabb"]
    tc = target["position"]
    dims = obj["dimensions_m"]
    slot = _choose_support_slot(obj, edge, occupied)
    ox, oy = _slot_offset(slot, ta, dims)
    target_height = ta["z_max"] - ta["z_min"]
    support_sc = str(target.get("subclass") or target.get("semantic_group") or "")
    used_count = max(0, len(occupied.get(str(edge.get("to_id")), set())) - 1)
    if support_sc in {"wardrobe"} or target_height > 1.4:
        levels = [0.08, 0.25, 0.5, 0.75]
        z = ta["z_min"] + levels[min(used_count, len(levels) - 1)] * target_height
    else:
        levels = [0.25, 0.5, 0.75]
        z = ta["z_min"] + levels[min(used_count, len(levels) - 1)] * target_height
    z = min(max(z, ta["z_min"] + 0.01), ta["z_max"] - dims["height"] - 0.01)
    return tc["x"] + ox, tc["y"] + oy, z, slot


def _choose_support_slot(obj: dict[str, Any], edge: dict[str, Any], occupied: dict[str, set[str]]) -> str:
    target_id = str(edge.get("to_id") or "")
    requested = str((edge.get("params") or {}).get("placement_area") or "").replace("top.", "")
    candidates = []
    if requested:
        candidates.append(requested)
    candidates.extend(DESK_SLOT_ORDER.get(str(obj.get("subclass")), []))
    candidates.extend(GENERIC_SLOT_ORDER)
    used = occupied.setdefault(target_id, set())
    for slot in candidates:
        if slot and slot not in used:
            used.add(slot)
            return slot
    slot = f"center_{len(used)}"
    used.add(slot)
    return "center"


def solve_placements(room_geometry: dict[str, Any], objects: list[dict[str, Any]], relationship_graph: dict[str, Any], anchors: dict[str, Any], options: dict[str, Any] | None = None) -> dict[str, Any]:
    options = dict(options or {})
    max_candidates = max(1, int(options.get("max_candidates_per_object", DEFAULT_MAX_CANDIDATES_PER_OBJECT)))
    max_total = max(1, int(options.get("max_total_candidate_combinations", DEFAULT_MAX_TOTAL_CANDIDATE_COMBINATIONS)))
    edges = list(relationship_graph.get("edges") or [])
    bbox = room_geometry["bbox"]
    center = room_geometry["center"]
    order = MAIN_ORDER + DEP_ORDER
    sorted_objects = sorted(objects, key=lambda o: (order.index(o["subclass"]) if o["subclass"] in order else 99, o["id"]))
    placed: dict[str, dict[str, Any]] = {}
    placed_objects: dict[str, dict[str, Any]] = {}
    occupied_support_slots: dict[str, set[str]] = {}
    candidate_evaluations = 0
    for obj in sorted_objects:
        dims = obj["dimensions_m"]
        sc = obj["subclass"]
        edge = _target_edge(edges, obj["id"], {"on_top_of", "inside", "next_to", "in_front_of", "around", "under", "above", "mounted_on_wall", "against_wall", "near"})
        x, y, z, yaw, reason = center["x"], center["y"], 0.0, 0.0, "Placed by fallback grid."
        if edge and edge.get("to_id") in placed:
            target = placed[edge["to_id"]]
            ta = target["aabb"]
            tc = target["position"]
            rel = edge["relation_type"]
            if rel == "on_top_of":
                slot = _choose_support_slot(obj, edge, occupied_support_slots)
                ox, oy = _slot_offset(slot, ta, dims)
                x, y, z = tc["x"] + ox, tc["y"] + oy, ta["z_max"] + 0.003
                reason = f"{sc} snapped on top of {edge['to_id']} at {slot}."
            elif rel == "inside":
                support_obj = placed_objects.get(edge["to_id"], {})
                target_for_inside = {**target, "subclass": support_obj.get("subclass")}
                x, y, z, slot = _inside_position(obj, target_for_inside, edge, occupied_support_slots)
                reason = f"{sc} placed inside {edge['to_id']} at {slot}."
            elif rel == "above":
                x, y, yaw = _wall_position_near_target(room_geometry, dims, target)
                z = min(max(ta["z_max"] + 0.35, 1.2), 1.8)
                reason = f"{sc} placed above {edge['to_id']}."
            elif rel == "near":
                x, y, z = min(bbox["x_max"] - dims["width"] / 2, ta["x_max"] + dims["width"] / 2 + 0.18), tc["y"], 0.0
                reason = f"{sc} placed near {edge['to_id']}."
            elif rel == "in_front_of":
                x, y, z = tc["x"], ta["y_max"] + dims["depth"] / 2 + 0.25, 0.0
                reason = f"{sc} placed in front of {edge['to_id']}."
            elif rel == "next_to":
                x, y, z = ta["x_max"] + dims["width"] / 2 + 0.08, tc["y"], 0.0
                if x + dims["width"] / 2 > bbox["x_max"]:
                    x = ta["x_min"] - dims["width"] / 2 - 0.08  # pragma: no cover
                reason = f"{sc} placed next to {edge['to_id']}."
            elif rel == "around":
                idx = len([p for p in placed.values() if p.get("around_target") == edge["to_id"]])
                offsets = [(0, ta["y_max"] - tc["y"] + dims["depth"] / 2 + 0.2), (0, -(tc["y"] - ta["y_min"] + dims["depth"] / 2 + 0.2)), (-(tc["x"] - ta["x_min"] + dims["width"] / 2 + 0.2), 0), (ta["x_max"] - tc["x"] + dims["width"] / 2 + 0.2, 0)]
                ox, oy = offsets[idx % len(offsets)]
                x, y, z = tc["x"] + ox, tc["y"] + oy, 0.0
                reason = f"{sc} distributed around {edge['to_id']}."
            elif rel == "under":
                x, y, z = tc["x"], tc["y"], 0.0
                reason = f"{sc} placed under {edge['to_id']} group."
            elif rel in {"mounted_on_wall", "against_wall"}:
                cands = generate_main_object_candidates(room_geometry, obj, [], relationship_graph, max_candidates)
                candidate_evaluations += len(cands)
                best = max(cands, key=lambda c: score_candidate(room_geometry, obj, c, placed))
                x, y, yaw, reason = best["x"], best["y"], best["yaw"], best["reason"]
        elif obj["placement_type"] in {"wall"} or sc in MAIN_ORDER:
            cands = generate_main_object_candidates(room_geometry, obj, [], relationship_graph, max_candidates)
            candidate_evaluations += len(cands)
            best = max(cands, key=lambda c: score_candidate(room_geometry, obj, c, placed))
            x, y, yaw, reason = best["x"], best["y"], best["yaw"], best["reason"]
            if obj["placement_type"] == "wall":
                z = 1.2
        else:
            step = 0.35
            grid_side = min(8, max(1, int(math.sqrt(max_total))))
            found_grid_slot = False
            for row in range(grid_side):
                for col in range(grid_side):
                    candidate_evaluations += 1
                    if candidate_evaluations > max_total:
                        break  # pragma: no cover
                    tx = bbox["x_min"] + 0.3 + col * step
                    ty = bbox["y_min"] + 0.3 + row * step
                    aa = _aabb(tx, ty, 0.0, dims["width"], dims["depth"], dims["height"])
                    if _inside(aa, bbox) and not any(_intersects(aa, p["aabb"]) for p in placed.values()):
                        x, y = tx, ty
                        found_grid_slot = True
                        break
                if found_grid_slot or candidate_evaluations > max_total:
                    break
        face = _target_edge(edges, obj["id"], {"faces", "visible_from"})
        if face and face.get("to_id") in placed:
            yaw = _yaw_to({"x": x, "y": y}, placed[face["to_id"]]["position"])
        clamped = False
        if obj.get("placement_type") != "wall":
            x, y, clamped = _clamp_center_to_bbox(x, y, dims, bbox)
        aabb = _aabb(x, y, z, dims["width"], dims["depth"], dims["height"])
        if obj.get("placement_type") == "floor" and sc != "rug" and not _is_free_floor_aabb(aabb, bbox, placed, obj["id"]):
            preferred: list[tuple[float, float]] = []
            if edge and edge.get("to_id") in placed:
                ta = placed[edge["to_id"]]["aabb"]
                tc = placed[edge["to_id"]]["position"]
                preferred = [
                    (tc["x"], ta["y_max"] + dims["depth"] / 2 + 0.25),
                    (tc["x"], ta["y_min"] - dims["depth"] / 2 - 0.25),
                    (ta["x_min"] - dims["width"] / 2 - 0.25, tc["y"]),
                    (ta["x_max"] + dims["width"] / 2 + 0.25, tc["y"]),
                ]
            free_pos = _find_free_floor_position(room_geometry, dims, placed, obj["id"], preferred)
            if free_pos is not None:
                x, y = free_pos
                reason = f"{reason} Shifted to nearest free floor slot."
                aabb = _aabb(x, y, z, dims["width"], dims["depth"], dims["height"])
        others = {k: v for k, v in placed.items() if k != obj["id"]}
        raw_score = score_candidate(room_geometry, obj, {"x": x, "y": y, "reason": reason}, others)
        placed[obj["id"]] = {
            "object_id": obj["id"],
            "position": {"x": round(x, 4), "y": round(y, 4), "z": round(z, 4)},
            "rotation_z_deg": round(yaw, 2),
            "aabb": {k: round(v, 4) for k, v in aabb.items()},
            "placement_reason": reason,
            "score": round(max(0.0, min(1.0, raw_score)), 4),
            "around_target": edge.get("to_id") if edge and edge.get("relation_type") == "around" else None,
        }
        if clamped:
            placed[obj["id"]]["warnings"] = [f"{obj['id']}: placement clamped inside room bbox"]
        placed_objects[obj["id"]] = obj
    return {
        "schema": "placements_generated/v1",
        "placements": [{k: v for k, v in p.items() if k != "around_target"} for p in placed.values()],
        "solver_limits": {
            "max_candidates_per_object": max_candidates,
            "max_total_candidate_combinations": max_total,
            "candidate_evaluations": candidate_evaluations,
            "strategy": "greedy_per_object_no_exhaustive_combinatorial_search",
        },
    }
