from __future__ import annotations

import math
from typing import Any


def _inside(a: dict[str, float], bbox: dict[str, float]) -> bool:
    return a["x_min"] >= bbox["x_min"] - 1e-6 and a["x_max"] <= bbox["x_max"] + 1e-6 and a["y_min"] >= bbox["y_min"] - 1e-6 and a["y_max"] <= bbox["y_max"] + 1e-6


def _overlap(a: dict[str, float], b: dict[str, float]) -> bool:
    return not (a["x_max"] <= b["x_min"] or b["x_max"] <= a["x_min"] or a["y_max"] <= b["y_min"] or b["y_max"] <= a["y_min"] or a["z_max"] <= b["z_min"] or b["z_max"] <= a["z_min"])


def _aabb_contains_point(a: dict[str, float], x: float, y: float, pad: float = 0.0) -> bool:
    return a["x_min"] - pad <= x <= a["x_max"] + pad and a["y_min"] - pad <= y <= a["y_max"] + pad


def _opening_points(openings: list[dict[str, Any]]) -> list[tuple[float, float]]:
    pts: list[tuple[float, float]] = []
    for raw in openings:
        if not isinstance(raw, dict):
            continue
        if "x" in raw and ("y" in raw or "z" in raw):
            pts.append((float(raw.get("x") or 0.0), float(raw.get("y", raw.get("z")) or 0.0)))
            continue
        if isinstance(raw.get("center"), dict):
            c = raw["center"]
            pts.append((float(c.get("x") or 0.0), float(c.get("y", c.get("z")) or 0.0)))
            continue
        if isinstance(raw.get("from"), dict) and isinstance(raw.get("to"), dict):
            a, b = raw["from"], raw["to"]
            pts.append(((float(a.get("x") or 0.0) + float(b.get("x") or 0.0)) / 2.0, (float(a.get("y", a.get("z")) or 0.0) + float(b.get("y", b.get("z")) or 0.0)) / 2.0))
    return pts


def validate_geometry(room_geometry: dict[str, Any], objects: list[dict[str, Any]], relations: dict[str, Any], placements_payload: dict[str, Any]) -> dict[str, Any]:
    placements = {p["object_id"]: p for p in placements_payload.get("placements", [])}
    obj_by_id = {o["id"]: o for o in objects}
    hard: list[str] = []
    soft: list[str] = []
    bbox = room_geometry["bbox"]
    floor_ids = [o["id"] for o in objects if o.get("placement_type") in {"floor", "wall"}]
    floor_blockers = [o["id"] for o in objects if o.get("placement_type") == "floor" and o.get("role") != "accessory"]
    for oid in floor_ids:
        p = placements.get(oid)
        if not p:
            hard.append(f"{oid}: missing placement")
            continue
        for warning in p.get("warnings") or []:
            soft.append(str(warning))
        if obj_by_id[oid].get("placement_type") == "floor" and not _inside(p["aabb"], bbox):
            hard.append(f"{oid}: outside room bbox")
        dims = obj_by_id[oid].get("dimensions_m") or {}
        if float(dims.get("width") or 0.0) > bbox["width_m"] or float(dims.get("depth") or 0.0) > bbox["depth_m"]:
            hard.append(f"{oid}: object dimensions do not fit room bbox")
    for i, oid in enumerate(floor_ids):
        for oid2 in floor_ids[i + 1:]:
            o1, o2 = obj_by_id[oid], obj_by_id[oid2]
            if o1.get("placement_type") != "floor" or o2.get("placement_type") != "floor":
                continue
            if o1.get("role") == "accessory" or o2.get("role") == "accessory":
                continue
            p1, p2 = placements.get(oid), placements.get(oid2)
            if p1 and p2 and _overlap(p1["aabb"], p2["aabb"]):
                hard.append(f"{oid} intersects {oid2}")
    ids_with_relation = {e.get("from_id") for e in relations.get("edges", [])}
    for o in objects:
        if o.get("role") == "accessory" and o["id"] not in ids_with_relation:
            soft.append(f"{o['id']}: accessory has no support/proximity relation")
    for e in relations.get("edges", []):
        if e.get("relation_type") == "on_top_of" and e.get("from_id") in placements and e.get("to_id") in placements:
            a, b = placements[e["from_id"]]["aabb"], placements[e["to_id"]]["aabb"]
            if abs(a["z_min"] - b["z_max"]) > 0.05:
                hard.append(f"{e['from_id']}: on_top_of z mismatch")
            if not (a["x_min"] >= b["x_min"] - 0.05 and a["x_max"] <= b["x_max"] + 0.05 and a["y_min"] >= b["y_min"] - 0.05 and a["y_max"] <= b["y_max"] + 0.05):
                hard.append(f"{e['from_id']}: outside support footprint")
        if e.get("relation_type") == "inside" and e.get("from_id") in placements and e.get("to_id") in placements:
            a, b = placements[e["from_id"]]["aabb"], placements[e["to_id"]]["aabb"]
            if a["z_min"] <= b["z_min"] + 0.005:
                soft.append(f"{e['from_id']}: inside relation placed near bottom of {e['to_id']}")
            if not (a["x_min"] >= b["x_min"] - 0.05 and a["x_max"] <= b["x_max"] + 0.05 and a["y_min"] >= b["y_min"] - 0.05 and a["y_max"] <= b["y_max"] + 0.05):
                hard.append(f"{e['from_id']}: outside containment footprint")
    density = sum(o["dimensions_m"]["width"] * o["dimensions_m"]["depth"] for o in objects if o.get("placement_type") == "floor") / max(float(room_geometry.get("area_m2") or 1.0), 1.0)
    if density > 0.55:
        soft.append("Zone density may be too high; system does not guarantee ideal design.")
    if density > 0.72:
        hard.append("Floor occupancy is too high to guarantee usable passages.")
    openings = room_geometry.get("openings") if isinstance(room_geometry.get("openings"), dict) else {}
    for idx, (x, y) in enumerate(_opening_points(list(openings.get("doors") or []))):
        for oid in floor_blockers:
            p = placements.get(oid)
            if p and _aabb_contains_point(p["aabb"], x, y, pad=0.45):
                hard.append(f"{oid}: blocks known door clearance door_{idx}")
    for idx, (x, y) in enumerate(_opening_points(list(openings.get("windows") or []))):
        for oid in floor_blockers:
            p = placements.get(oid)
            obj = obj_by_id.get(oid, {})
            if p and float((obj.get("dimensions_m") or {}).get("height") or 0.0) > 1.2 and _aabb_contains_point(p["aabb"], x, y, pad=0.25):
                soft.append(f"{oid}: may block known window access window_{idx}")
    score = max(0.0, 10.0 - len(hard) * 2.0 - len(soft) * 0.35)
    return {"schema": "geometry_validation/v1", "is_valid": not hard, "score": round(score, 2), "hard_errors": hard, "soft_warnings": soft, "relation_scores": {"support": 1.0 if not any("on_top_of" in e for e in hard) else 0.5, "orientation": 0.8, "proximity": 0.8, "wall": 0.8}, "guarantee": "Hard constraints are checked by validator; ideal design quality is not guaranteed."}
