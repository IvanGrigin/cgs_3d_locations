#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import random
from copy import deepcopy
from pathlib import Path
from typing import Any


Rect = tuple[float, float, float, float]
Point = tuple[float, float]


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _scene_room_polygon(room: dict[str, Any]) -> list[Point]:
    raw = room.get("floor_polygon") or room.get("floor_polygon_xz") or []
    out: list[Point] = []
    if isinstance(raw, list):
        for point in raw:
            if not isinstance(point, dict):
                continue
            x = _to_float(point.get("x"))
            y = _to_float(point.get("y", point.get("z")))
            if x is not None and y is not None:
                out.append((float(x), float(y)))
    return out


def _poly_bounds(poly: list[Point]) -> Rect:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), max(xs), min(ys), max(ys)


def _point_in_poly_xy(x: float, y: float, poly: list[Point]) -> bool:
    inside = False
    n = len(poly)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / max(yj - yi, 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _dist_point_segment_xy(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    denom = vx * vx + vy * vy
    if denom <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, (wx * vx + wy * vy) / denom))
    return math.hypot(px - (ax + t * vx), py - (ay + t * vy))


def _dist_to_poly_edges_xy(x: float, y: float, poly: list[Point]) -> float:
    if len(poly) < 2:
        return 0.0
    return min(
        _dist_point_segment_xy(x, y, poly[i][0], poly[i][1], poly[(i + 1) % len(poly)][0], poly[(i + 1) % len(poly)][1])
        for i in range(len(poly))
    )


def _room_sample_points(poly: list[Point], *, step: float, wall_margin: float = 0.0) -> list[Point]:
    x_min, x_max, y_min, y_max = _poly_bounds(poly)
    points: list[Point] = []
    x = x_min
    while x <= x_max + 1e-9:
        y = y_min
        while y <= y_max + 1e-9:
            if _point_in_poly_xy(x, y, poly) and _dist_to_poly_edges_xy(x, y, poly) >= wall_margin:
                points.append((x, y))
            y += step
        x += step
    if points:
        return points
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    if _point_in_poly_xy(cx, cy, poly) and _dist_to_poly_edges_xy(cx, cy, poly) >= wall_margin:
        return [(cx, cy)]
    return []


def _circle_overlap_area(radius: float, distance: float) -> float:
    r = max(0.0, float(radius))
    d = max(0.0, float(distance))
    if r <= 0.0 or d >= 2.0 * r:
        return 0.0
    if d <= 1e-9:
        return math.pi * r * r
    ratio = max(-1.0, min(1.0, d / (2.0 * r)))
    return 2.0 * r * r * math.acos(ratio) - 0.5 * d * math.sqrt(max(0.0, 4.0 * r * r - d * d))


def _chandelier_coverage_radius(centers: list[Point], coverage_points: list[Point]) -> float:
    if not centers or not coverage_points:
        return 0.0
    return max(min(math.hypot(p[0] - c[0], p[1] - c[1]) for c in centers) for p in coverage_points)


def _chandelier_overlap_area(centers: list[Point], radius: float) -> float:
    overlap = 0.0
    for i, a in enumerate(centers):
        for b in centers[i + 1 :]:
            overlap += _circle_overlap_area(radius, math.hypot(a[0] - b[0], a[1] - b[1]))
    return overlap


def _chandelier_layout_score(centers: list[Point], coverage_points: list[Point]) -> tuple[float, float]:
    radius = _chandelier_coverage_radius(centers, coverage_points)
    return radius, _chandelier_overlap_area(centers, radius)


def _select_chandelier_centers(
    *,
    count: int,
    candidate_points: list[Point],
    coverage_points: list[Point],
    centroid: Point,
) -> tuple[list[Point], float, float]:
    if count <= 0 or not candidate_points:
        return [], 0.0, 0.0
    centers = [min(candidate_points, key=lambda p: math.hypot(p[0] - centroid[0], p[1] - centroid[1]))]
    while len(centers) < count:
        unused = [p for p in candidate_points if p not in centers]
        if not unused:
            break
        centers.append(max(unused, key=lambda p: min(math.hypot(p[0] - c[0], p[1] - c[1]) for c in centers)))
    while len(centers) < count:
        centers.append(centers[len(centers) % max(1, len(centers))])

    best_score = _chandelier_layout_score(centers, coverage_points)
    improved = True
    for _ in range(6):
        if not improved:
            break
        improved = False
        for idx in range(len(centers)):
            current = centers[idx]
            for cand in candidate_points:
                if cand == current or (cand in centers and candidate_points.count(cand) == 1):
                    continue
                trial = list(centers)
                trial[idx] = cand
                score = _chandelier_layout_score(trial, coverage_points)
                if score[0] < best_score[0] - 1e-6 or (
                    abs(score[0] - best_score[0]) <= 1e-6 and score[1] < best_score[1] - 1e-6
                ):
                    centers = trial
                    best_score = score
                    improved = True
                    current = cand
    return centers, best_score[0], best_score[1]


def _is_chandelier_item(item: dict[str, Any]) -> bool:
    text = f"{item.get('name') or ''} {item.get('category') or ''} {item.get('semantic_group') or ''}".lower()
    positive = ("chandelier", "ceilinglamp", "ceiling_lamp", "ceiling light", "ceiling_light", "pendant", "люстр", "потолоч")
    negative = ("floorlamp", "floor_lamp", "tablelamp", "table_lamp", "walllamp", "wall_lamp", "торшер", "настоль", "бра")
    return any(token in text for token in positive) and not any(token in text for token in negative)


def _shift_item_xy(item: dict[str, Any], dx: float, dy: float) -> None:
    pos = item.get("position_m")
    if isinstance(pos, list) and len(pos) >= 2:
        pos[0] = float(pos[0]) + dx
        pos[1] = float(pos[1]) + dy
    aabb = item.get("aabb")
    if isinstance(aabb, dict):
        for key in ("x_min", "x_max"):
            if key in aabb:
                aabb[key] = float(aabb[key]) + dx
        for key in ("y_min", "y_max"):
            if key in aabb:
                aabb[key] = float(aabb[key]) + dy


def normalize_chandelier_positions_in_scene(
    scene: dict[str, Any],
    *,
    wall_clearance_m: float = 1.0,
    sample_step_m: float = 0.35,
) -> tuple[dict[str, Any], dict[str, Any]]:
    updated = deepcopy(scene)
    room = updated.get("room") if isinstance(updated.get("room"), dict) else {}
    poly = _scene_room_polygon(room)
    placements = updated.get("placements") if isinstance(updated.get("placements"), list) else []
    chandeliers = [item for item in placements if isinstance(item, dict) and _is_chandelier_item(item)]
    info: dict[str, Any] = {
        "enabled": True,
        "chandelier_count": len(chandeliers),
        "wall_clearance_m": wall_clearance_m,
        "moved": [],
    }
    if len(poly) < 3 or not chandeliers:
        info["skipped_reason"] = "no_room_polygon_or_no_chandeliers"
        return updated, info

    candidate_points = _room_sample_points(poly, step=sample_step_m, wall_margin=wall_clearance_m)
    if not candidate_points:
        x_min, x_max, y_min, y_max = _poly_bounds(poly)
        centroid = (0.5 * (x_min + x_max), 0.5 * (y_min + y_max))
        if not _point_in_poly_xy(centroid[0], centroid[1], poly):
            info["skipped_reason"] = "no_valid_room_points_with_wall_clearance"
            return updated, info
        coverage_points = _room_sample_points(poly, step=sample_step_m, wall_margin=0.0)
        coverage_radius = _chandelier_coverage_radius([centroid], coverage_points)
        centers = [centroid]
        overlap_area = 0.0
        info["small_room_center_fallback"] = True
        info["skipped_extra_chandelier_count"] = max(0, len(chandeliers) - 1)
        info["candidate_point_count"] = 0
        info["coverage_sample_point_count"] = len(coverage_points)
        info["coverage_overlap_area_m2"] = 0.0
    else:
        x_min, x_max, y_min, y_max = _poly_bounds(poly)
        centroid = (0.5 * (x_min + x_max), 0.5 * (y_min + y_max))
        coverage_points = _room_sample_points(poly, step=sample_step_m, wall_margin=0.0)
        centers, coverage_radius, overlap_area = _select_chandelier_centers(
            count=len(chandeliers),
            candidate_points=candidate_points,
            coverage_points=coverage_points,
            centroid=centroid,
        )
        info["candidate_point_count"] = len(candidate_points)
        info["coverage_sample_point_count"] = len(coverage_points)
        info["coverage_overlap_area_m2"] = round(overlap_area, 4)

    for item, center in zip(chandeliers, centers):
        pos = item.get("position_m")
        if not (isinstance(pos, list) and len(pos) >= 2):
            continue
        old_xy = (float(pos[0]), float(pos[1]))
        dx, dy = center[0] - old_xy[0], center[1] - old_xy[1]
        _shift_item_xy(item, dx, dy)
        meta = item.setdefault("meta", {})
        if isinstance(meta, dict):
            meta["chandelier_normalized"] = True
            meta["chandelier_coverage_radius_m"] = round(coverage_radius, 3)
            meta["chandelier_coverage_overlap_area_m2"] = round(overlap_area, 3)
        info["moved"].append(
            {
                "id": item.get("id"),
                "old_xy": [round(old_xy[0], 4), round(old_xy[1], 4)],
                "new_xy": [round(center[0], 4), round(center[1], 4)],
                "move_m": round(math.hypot(dx, dy), 4),
                "wall_clearance_m": round(_dist_to_poly_edges_xy(center[0], center[1], poly), 4),
            }
        )
    info["coverage_radius_m"] = round(coverage_radius, 4)
    return updated, info


def _item_rect_xy(item: dict[str, Any]) -> Rect | None:
    aabb = item.get("aabb")
    if isinstance(aabb, dict):
        vals = [_to_float(aabb.get(k)) for k in ("x_min", "x_max", "y_min", "y_max")]
        if all(v is not None for v in vals):
            x_min, x_max, y_min, y_max = (float(v) for v in vals)  # type: ignore[arg-type]
            if x_max > x_min and y_max > y_min:
                return x_min, x_max, y_min, y_max
    pos = item.get("position_m")
    size = item.get("size_m")
    if isinstance(pos, list) and isinstance(size, list) and len(pos) >= 2 and len(size) >= 2:
        cx, cy = float(pos[0]), float(pos[1])
        sx, sy = max(0.0, float(size[0])), max(0.0, float(size[1]))
        return cx - sx * 0.5, cx + sx * 0.5, cy - sy * 0.5, cy + sy * 0.5
    return None


def _rect_area(rect: Rect) -> float:
    return max(0.0, rect[1] - rect[0]) * max(0.0, rect[3] - rect[2])


def _rect_intersection_area(a: Rect, b: Rect) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[2], b[2]))


def _rect_shift(rect: Rect, dx: float, dy: float) -> Rect:
    return rect[0] + dx, rect[1] + dx, rect[2] + dy, rect[3] + dy


def _rect_outside_room_area(rect: Rect, poly: list[Point]) -> float:
    x_min, x_max, y_min, y_max = _poly_bounds(poly)
    inside_bounds = (max(rect[0], x_min), min(rect[1], x_max), max(rect[2], y_min), min(rect[3], y_max))
    outside = _rect_area(rect) - _rect_area(inside_bounds)
    corners = [(rect[0], rect[2]), (rect[0], rect[3]), (rect[1], rect[2]), (rect[1], rect[3])]
    if any(not _point_in_poly_xy(x, y, poly) for x, y in corners):
        outside = max(outside, _rect_area(rect) * 0.25)
    return max(0.0, outside)


def _is_movable_furniture_item(item: dict[str, Any]) -> bool:
    text = (
        f"{item.get('name') or ''} {item.get('label') or ''} {item.get('category') or ''} "
        f"{item.get('type') or ''} {item.get('semantic_group') or ''}"
    ).lower()
    blocked = (
        "rug", "lamp", "light", "plant", "window", "door", "ceiling", "wall", "floor", "decor",
        "vase", "book", "pillow", "clutter", "accessory", "ковер", "торшер", "люстр",
        "раст", "декор", "ваза", "книга", "подушка",
    )
    if any(token in text for token in blocked):
        return False
    furniture_tokens = (
        "cabinet", "shelf", "table", "desk", "sofa", "chair", "bed", "dresser", "wardrobe",
        "stand", "комод", "шкаф", "стеллаж", "стол", "диван", "кресл", "кровать", "тумб",
    )
    return any(token in text for token in furniture_tokens)


def _is_support_child_candidate(item: dict[str, Any]) -> bool:
    text = (
        f"{item.get('name') or ''} {item.get('label') or ''} {item.get('category') or ''} "
        f"{item.get('type') or ''} {item.get('semantic_group') or ''}"
    ).lower()
    blocked = ("rug", "wall", "floor", "ceiling", "window", "door", "ковер", "стена", "пол", "окно", "двер")
    return not any(token in text for token in blocked)


def _z_range(item: dict[str, Any]) -> tuple[float, float] | None:
    aabb = item.get("aabb")
    if isinstance(aabb, dict):
        z_min = _to_float(aabb.get("z_min"))
        z_max = _to_float(aabb.get("z_max"))
        if z_min is not None and z_max is not None and z_max >= z_min:
            return float(z_min), float(z_max)
    pos = item.get("position_m")
    size = item.get("size_m")
    if isinstance(pos, list) and isinstance(size, list) and len(pos) >= 3 and len(size) >= 3:
        cz = float(pos[2])
        sz = max(0.0, float(size[2]))
        return cz - sz * 0.5, cz + sz * 0.5
    return None


def _rect_contains_center(container: Rect, child: Rect, margin: float = 0.08) -> bool:
    cx = 0.5 * (child[0] + child[1])
    cy = 0.5 * (child[2] + child[3])
    return (container[0] - margin) <= cx <= (container[1] + margin) and (container[2] - margin) <= cy <= (container[3] + margin)


def _support_child_indices(*, anchor_index: int, placements: list[Any], rects: list[Rect | None]) -> set[int]:
    anchor = placements[anchor_index]
    if not isinstance(anchor, dict):
        return set()
    anchor_rect = rects[anchor_index]
    anchor_z = _z_range(anchor)
    if anchor_rect is None or anchor_z is None:
        return set()
    anchor_area = _rect_area(anchor_rect)
    children: set[int] = set()
    for idx, child in enumerate(placements):
        if idx == anchor_index or not isinstance(child, dict) or not _is_support_child_candidate(child):
            continue
        if _is_movable_furniture_item(child):
            continue
        child_rect = rects[idx]
        child_z = _z_range(child)
        if child_rect is None or child_z is None:
            continue
        child_area = _rect_area(child_rect)
        if child_area > anchor_area * 0.75:
            continue
        if not _rect_contains_center(anchor_rect, child_rect):
            continue
        overlap_ratio = _rect_intersection_area(anchor_rect, child_rect) / max(child_area, 1e-9)
        if overlap_ratio < 0.55:
            continue
        on_top = abs(child_z[0] - anchor_z[1]) <= 0.18
        inside = child_z[0] >= anchor_z[0] - 0.05 and child_z[1] <= anchor_z[1] + 0.10
        if on_top or inside:
            children.add(idx)
    return children


def _support_children_by_anchor(*, placements: list[Any], rects: list[Rect | None]) -> dict[int, set[int]]:
    raw = {
        idx: _support_child_indices(anchor_index=idx, placements=placements, rects=rects)
        for idx in range(len(placements))
        if rects[idx] is not None
    }
    candidates_by_child: dict[int, list[int]] = {}
    for anchor_idx, children in raw.items():
        for child_idx in children:
            candidates_by_child.setdefault(child_idx, []).append(anchor_idx)
    pruned: dict[int, set[int]] = {idx: set() for idx in raw}
    for child_idx, anchor_indices in candidates_by_child.items():
        child_z = _z_range(placements[child_idx]) if isinstance(placements[child_idx], dict) else None
        if child_z is None:
            continue

        def score(anchor_idx: int) -> tuple[float, float, float]:
            anchor = placements[anchor_idx]
            anchor_rect = rects[anchor_idx]
            anchor_z = _z_range(anchor) if isinstance(anchor, dict) else None
            if anchor_rect is None or anchor_z is None:
                return (9.0, 9.0, 9.0)
            top_delta = abs(child_z[0] - anchor_z[1])
            on_top_rank = 0.0 if top_delta <= 0.18 else 1.0
            return (on_top_rank, top_delta if on_top_rank == 0.0 else 0.0, _rect_area(anchor_rect))

        best_anchor = min(anchor_indices, key=score)
        pruned.setdefault(best_anchor, set()).add(child_idx)
    return pruned


def _move_group_indices(anchor_index: int, children_by_anchor: dict[int, set[int]]) -> set[int]:
    group = {anchor_index}
    pending = list(children_by_anchor.get(anchor_index, set()))
    while pending:
        idx = pending.pop()
        if idx in group:
            continue
        group.add(idx)
        pending.extend(children_by_anchor.get(idx, set()))
    return group


def _union_rect(rects_for_group: list[Rect]) -> Rect:
    return (
        min(r[0] for r in rects_for_group),
        max(r[1] for r in rects_for_group),
        min(r[2] for r in rects_for_group),
        max(r[3] for r in rects_for_group),
    )


def _rect_from_points(points: list[Point]) -> Rect:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), max(xs), min(ys), max(ys)


def _rect_contains_point_with_margin(rect: Rect, x: float, y: float, margin: float) -> bool:
    return rect[0] - margin <= x <= rect[1] + margin and rect[2] - margin <= y <= rect[3] + margin


def _room_wall_segments(room: dict[str, Any], poly: list[Point]) -> dict[str, tuple[Point, Point]]:
    out: dict[str, tuple[Point, Point]] = {}
    walls = room.get("walls")
    if isinstance(walls, list):
        for idx, wall in enumerate(walls):
            if not isinstance(wall, dict):
                continue
            wall_id = str(wall.get("id") or f"w{idx}")
            try:
                a = int(wall["from_vertex"])
                b = int(wall["to_vertex"])
            except Exception:
                continue
            if 0 <= a < len(poly) and 0 <= b < len(poly):
                out[wall_id] = (poly[a], poly[b])
    if out:
        return out
    return {f"w{i}": (poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly))}


def _opening_segment_xy(opening: dict[str, Any], wall_segments: dict[str, tuple[Point, Point]]) -> tuple[Point, Point] | None:
    segment = opening.get("segment")
    if isinstance(segment, dict):
        try:
            return ((float(segment["x1"]), float(segment["y1"])), (float(segment["x2"]), float(segment["y2"])))
        except Exception:
            pass
    wall_id = str(opening.get("wall_id") or "").strip()
    if wall_id not in wall_segments:
        return None
    p0, p1 = wall_segments[wall_id]
    try:
        s = float(opening.get("s", 0.0))
        width = float(opening.get("width", 0.9))
    except Exception:
        return None
    vx, vy = p1[0] - p0[0], p1[1] - p0[1]
    length = math.hypot(vx, vy)
    if length <= 1e-9:
        return None
    ux, uy = vx / length, vy / length
    s0 = max(0.0, min(s, length))
    s1 = max(0.0, min(s + width, length))
    return ((p0[0] + ux * s0, p0[1] + uy * s0), (p0[0] + ux * s1, p0[1] + uy * s1))


def _door_clearance_context(
    room: dict[str, Any],
    poly: list[Point],
    *,
    keepout_depth_m: float = 1.05,
    side_margin_m: float = 0.35,
    passage_width_m: float = 0.85,
    passage_step_m: float = 0.25,
) -> dict[str, Any]:
    doors = room.get("doors") if isinstance(room.get("doors"), list) else []
    if not doors:
        return {"keepouts": [], "passage_points": [], "passage_half_width_m": passage_width_m * 0.5}
    x_min, x_max, y_min, y_max = _poly_bounds(poly)
    centroid = (0.5 * (x_min + x_max), 0.5 * (y_min + y_max))
    walls = _room_wall_segments(room, poly)
    keepouts: list[Rect] = []
    passage_points: list[Point] = []
    for door in doors:
        if not isinstance(door, dict):
            continue
        seg = _opening_segment_xy(door, walls)
        if seg is None:
            continue
        a, b = seg
        mx, my = 0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])
        tx, ty = b[0] - a[0], b[1] - a[1]
        length = math.hypot(tx, ty)
        if length <= 1e-9:
            continue
        tx, ty = tx / length, ty / length
        ix, iy = centroid[0] - mx, centroid[1] - my
        ilen = math.hypot(ix, iy)
        if ilen <= 1e-9:
            continue
        ix, iy = ix / ilen, iy / ilen
        keepouts.append(
            _rect_from_points(
                [
                    (a[0] - tx * side_margin_m, a[1] - ty * side_margin_m),
                    (b[0] + tx * side_margin_m, b[1] + ty * side_margin_m),
                    (b[0] + tx * side_margin_m + ix * keepout_depth_m, b[1] + ty * side_margin_m + iy * keepout_depth_m),
                    (a[0] - tx * side_margin_m + ix * keepout_depth_m, a[1] - ty * side_margin_m + iy * keepout_depth_m),
                ]
            )
        )
        route_len = math.hypot(centroid[0] - mx, centroid[1] - my)
        steps = max(1, int(math.ceil(route_len / passage_step_m)))
        for step_idx in range(1, steps + 1):
            t = step_idx / steps
            px = mx + (centroid[0] - mx) * t
            py = my + (centroid[1] - my) * t
            if _point_in_poly_xy(px, py, poly):
                passage_points.append((px, py))
    return {
        "keepouts": keepouts,
        "passage_points": passage_points,
        "passage_half_width_m": passage_width_m * 0.5,
        "passage_step_m": passage_step_m,
    }


def _passage_penalty_for_group(group_rects: dict[int, Rect], passage_context: dict[str, Any] | None) -> float:
    if not passage_context:
        return 0.0
    rects = list(group_rects.values())
    if not rects:
        return 0.0
    penalty = 0.0
    for keepout in passage_context.get("keepouts") or []:
        penalty += _rect_intersection_area(_union_rect(rects), keepout) * 10.0
    half_width = float(passage_context.get("passage_half_width_m", 0.425) or 0.425)
    step = float(passage_context.get("passage_step_m", 0.25) or 0.25)
    blocked = 0
    for px, py in passage_context.get("passage_points") or []:
        if any(_rect_contains_point_with_margin(rect, px, py, half_width) for rect in rects):
            blocked += 1
    penalty += blocked * step * step * 10.0
    return penalty


def _collision_penalty_for_group(
    group_rects: dict[int, Rect],
    *,
    group_indices: set[int],
    rects: list[Rect | None],
    movable_indices: set[int],
    poly: list[Point],
    passage_context: dict[str, Any] | None = None,
) -> float:
    penalty = sum(_rect_outside_room_area(rect, poly) * 2.0 for rect in group_rects.values())
    group_union = _union_rect(list(group_rects.values()))
    for idx, other in enumerate(rects):
        if idx in group_indices or other is None or idx not in movable_indices:
            continue
        penalty += _rect_intersection_area(group_union, other)
    penalty += _passage_penalty_for_group(group_rects, passage_context)
    return penalty


def _best_repair_shift_for_item(
    *,
    item_index: int,
    group_indices: set[int],
    rects: list[Rect | None],
    movable_indices: set[int],
    poly: list[Point],
    passage_context: dict[str, Any] | None,
    search_step_m: float,
    max_shift_m: float,
) -> tuple[float, float, float, float] | None:
    old = rects[item_index]
    if old is None:
        return None
    old_group_rects = {idx: rects[idx] for idx in group_indices if rects[idx] is not None}
    if item_index not in old_group_rects:
        return None
    old_penalty = _collision_penalty_for_group(
        old_group_rects,
        group_indices=group_indices,
        rects=rects,
        movable_indices=movable_indices,
        poly=poly,
        passage_context=passage_context,
    )
    old_group_union = _union_rect(list(old_group_rects.values()))
    best = (old_penalty, _rect_area(old_group_union), 0.0, 0.0)
    steps = max(1, int(math.ceil(max_shift_m / search_step_m)))
    directions = [(0.0, 0.0)]
    for k in range(16):
        angle = (2.0 * math.pi * k) / 16.0
        directions.append((math.cos(angle), math.sin(angle)))
    for step_idx in range(1, steps + 1):
        radius = min(max_shift_m, step_idx * search_step_m)
        for ux, uy in directions[1:]:
            dx, dy = ux * radius, uy * radius
            cand_group_rects = {idx: _rect_shift(rect, dx, dy) for idx, rect in old_group_rects.items()}
            penalty = _collision_penalty_for_group(
                cand_group_rects,
                group_indices=group_indices,
                rects=rects,
                movable_indices=movable_indices,
                poly=poly,
                passage_context=passage_context,
            )
            cand_group_union = _union_rect(list(cand_group_rects.values()))
            old_overlap = _rect_intersection_area(old_group_union, cand_group_union)
            if penalty < best[0] - 1e-6 or (abs(penalty - best[0]) <= 1e-6 and old_overlap > best[1]):
                best = (penalty, old_overlap, dx, dy)
    if best[0] < old_penalty - 1e-6:
        return best
    return None


def repair_furniture_intersections_in_scene(
    scene: dict[str, Any],
    *,
    max_passes: int = 3,
    search_step_m: float = 0.15,
    max_shift_m: float = 1.2,
) -> tuple[dict[str, Any], dict[str, Any]]:
    updated = deepcopy(scene)
    room = updated.get("room") if isinstance(updated.get("room"), dict) else {}
    poly = _scene_room_polygon(room)
    placements = updated.get("placements") if isinstance(updated.get("placements"), list) else []
    info: dict[str, Any] = {"enabled": True, "passes": [], "moved": []}
    if len(poly) < 3 or not placements:
        info["skipped_reason"] = "no_room_polygon_or_no_placements"
        return updated, info

    movable_indices = {idx for idx, item in enumerate(placements) if isinstance(item, dict) and _is_movable_furniture_item(item)}
    rects = [_item_rect_xy(item) if isinstance(item, dict) else None for item in placements]
    children_by_anchor = _support_children_by_anchor(placements=placements, rects=rects)
    passage_context = _door_clearance_context(room, poly)
    info["passage_protection"] = {
        "door_keepout_count": len(passage_context.get("keepouts") or []),
        "passage_sample_point_count": len(passage_context.get("passage_points") or []),
        "passage_half_width_m": round(float(passage_context.get("passage_half_width_m", 0.0) or 0.0), 4),
    }
    info["support_groups"] = [
        {
            "anchor_id": placements[idx].get("id") if isinstance(placements[idx], dict) else idx,
            "child_ids": [
                placements[child_idx].get("id") if isinstance(placements[child_idx], dict) else child_idx
                for child_idx in sorted(children)
            ],
        }
        for idx, children in sorted(children_by_anchor.items())
        if children
    ]
    for pass_idx in range(max(1, max_passes)):
        trouble: set[int] = set()
        for idx in movable_indices:
            rect = rects[idx]
            if rect is not None and _rect_outside_room_area(rect, poly) > 1e-6:
                trouble.add(idx)
        movable_list = sorted(movable_indices)
        for pos_i, i in enumerate(movable_list):
            ri = rects[i]
            if ri is None:
                continue
            for j in movable_list[pos_i + 1 :]:
                rj = rects[j]
                if rj is None:
                    continue
                if _rect_intersection_area(ri, rj) > 1e-6:
                    trouble.add(i if _rect_area(ri) <= _rect_area(rj) else j)
        pass_info = {"pass": pass_idx + 1, "trouble_count": len(trouble), "accepted": []}
        if not trouble:
            info["passes"].append(pass_info)
            break
        for idx in sorted(trouble):
            group_indices = _move_group_indices(idx, children_by_anchor)
            move = _best_repair_shift_for_item(
                item_index=idx,
                group_indices=group_indices,
                rects=rects,
                movable_indices=movable_indices,
                poly=poly,
                passage_context=passage_context,
                search_step_m=search_step_m,
                max_shift_m=max_shift_m,
            )
            if move is None:
                continue
            new_penalty, old_overlap, dx, dy = move
            item = placements[idx]
            old_rect = rects[idx]
            if old_rect is None or not isinstance(item, dict):
                continue
            moved_ids = []
            for move_idx in sorted(group_indices):
                move_item = placements[move_idx]
                move_rect = rects[move_idx]
                if move_rect is None or not isinstance(move_item, dict):
                    continue
                _shift_item_xy(move_item, dx, dy)
                rects[move_idx] = _rect_shift(move_rect, dx, dy)
                meta = move_item.setdefault("meta", {})
                if isinstance(meta, dict):
                    meta["furniture_overlap_repaired"] = True
                    meta["furniture_repair_anchor_id"] = item.get("id")
                    meta["furniture_repair_group_move"] = move_idx != idx
                moved_ids.append(move_item.get("id"))
            accepted = {
                "id": item.get("id"),
                "moved_ids": moved_ids,
                "dx": round(dx, 4),
                "dy": round(dy, 4),
                "move_m": round(math.hypot(dx, dy), 4),
                "new_penalty": round(new_penalty, 6),
                "old_new_overlap_area_m2": round(old_overlap, 6),
            }
            pass_info["accepted"].append(accepted)
            info["moved"].append(accepted)
        info["passes"].append(pass_info)
        if not pass_info["accepted"]:
            break
    info["moved_count"] = len(info["moved"])
    return updated, info


def _is_primary_plain_curtain_model(item: dict[str, Any] | str | Path) -> bool:
    if isinstance(item, dict):
        path = str(item.get("asset_local_path") or item.get("mesh_path") or "")
        title = str(item.get("title") or "")
    else:
        path = str(item)
        title = ""
    p = Path(path)
    text = f"{title} {path}".lower()
    return (
        p.suffix.lower() == ".fbx"
        and p.name.lower() == "shtora.fbx"
        and not any(token in text for token in ("люверс", "grommet", "curtain 2", "француз", "french", "кружев", "lace"))
    )


def _resolve_catalog_image(row: dict[str, Any], base_dir: Path) -> str | None:
    image_paths = row.get("local_image_paths")
    if not isinstance(image_paths, list):
        return None
    for raw in image_paths:
        path = Path(str(raw)).expanduser()
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        if path.is_file():
            return str(path)
    return None


def _score_curtain(row: dict[str, Any], style_text: str, room_type: str) -> float:
    text = " ".join(
        str(row.get(key) or "")
        for key in ("name", "category", "selected_material", "description", "search_text")
    ).lower()
    style = style_text.lower()
    score = 0.0
    if room_type.lower() in {"bedroom", "спальня"} and any(token in text for token in ("блэкаут", "blackout")):
        score += 2.0
    if any(token in style for token in ("scandinavian", "сканди", "миним", "modern", "современ")):
        if any(token in text for token in ("однотон", "беж", "grey", "gray", "white", "cream", "габардин", "рогож")):
            score += 1.0
    if any(token in style for token in ("classic", "классик", "luxury", "неокласс")):
        if any(token in text for token in ("атлас", "сатен", "cream", "gold", "beige")):
            score += 1.0
    if str(row.get("image_selection_note") or "") == "fallback_only_one_gallery_image":
        score -= 0.15
    if str(row.get("category") or "").strip().lower() == "шторы":
        score += 0.35
    if "вашим дизайном" in text:
        score -= 0.5
    if row.get("price") is not None:
        score += 0.05
    return score


def select_curtain_products(
    catalog: list[dict[str, Any]],
    *,
    count: int,
    style_profile: dict[str, Any] | None = None,
    seed: int = 0,
) -> list[dict[str, Any]]:
    if count <= 0 or not catalog:
        return []
    profile = style_profile or {}
    style_text = " ".join(
        str(profile.get(key) or "")
        for key in ("style_hint", "expanded_prompt", "style_label")
    )
    room_type = str(profile.get("room_type") or "")
    ranked = sorted(
        enumerate(catalog),
        key=lambda pair: (
            -_score_curtain(pair[1], style_text, room_type),
            str(pair[1].get("sku") or ""),
            pair[0],
        ),
    )
    pool = [row for _, row in ranked[: max(count * 4, count)]]
    rng = random.Random(seed)
    if len(pool) > count:
        first = pool[0]
        rest = pool[1:]
        rng.shuffle(rest)
        return [first, *rest[: count - 1]]
    return pool[:count]


def _wall_points(room: dict[str, Any], wall: dict[str, Any]) -> tuple[Point, Point] | None:
    poly = room.get("floor_polygon")
    if not isinstance(poly, list):
        return None
    try:
        i0 = int(wall.get("from_vertex"))
        i1 = int(wall.get("to_vertex"))
        p0 = poly[i0]
        p1 = poly[i1]
        return (float(p0["x"]), float(p0["y"])), (float(p1["x"]), float(p1["y"]))
    except Exception:
        return None


def _aabb_for_oriented_panel(
    *,
    center: Point,
    wall_dir: Point,
    inward: Point,
    width: float,
    depth: float,
    z_min: float,
    z_max: float,
) -> dict[str, float]:
    hx = width * 0.5
    hy = depth * 0.5
    corners = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            x = center[0] + wall_dir[0] * hx * sx + inward[0] * hy * sy
            y = center[1] + wall_dir[1] * hx * sx + inward[1] * hy * sy
            corners.append((x, y))
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    return {
        "x_min": min(xs),
        "x_max": max(xs),
        "y_min": min(ys),
        "y_max": max(ys),
        "z_min": z_min,
        "z_max": z_max,
    }


def apply_curtains_to_scene(
    scene: dict[str, Any],
    *,
    catalog: list[dict[str, Any]],
    catalog_base_dir: str | Path,
    curtain_model_paths: list[str] | None = None,
    curtain_models: list[dict[str, Any]] | None = None,
    style_profile: dict[str, Any] | None = None,
    seed: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    out = deepcopy(scene)
    room = out.get("room")
    if not isinstance(room, dict):
        return out, {"added_count": 0, "skipped_reason": "missing_room"}
    windows = room.get("windows")
    if not isinstance(windows, list) or not windows:
        return out, {"added_count": 0, "skipped_reason": "missing_windows"}
    walls = room.get("walls")
    if not isinstance(walls, list):
        return out, {"added_count": 0, "skipped_reason": "missing_walls"}
    wall_by_id = {str(w.get("id")): w for w in walls if isinstance(w, dict)}

    base_dir = Path(catalog_base_dir).expanduser()
    selected = select_curtain_products(catalog, count=len(windows), style_profile=style_profile, seed=seed)
    if not selected:
        return out, {"added_count": 0, "skipped_reason": "empty_catalog"}

    items_key = "items" if isinstance(out.get("items"), list) else "placements"
    items = out.setdefault(items_key, [])
    if not isinstance(items, list):
        return out, {"added_count": 0, "skipped_reason": "invalid_items"}
    existing_ids = {str(item.get("id") or "") for item in items if isinstance(item, dict)}
    model_items: list[dict[str, Any]] = []
    for item in curtain_models or []:
        if not isinstance(item, dict):
            continue
        path = str(item.get("asset_local_path") or item.get("mesh_path") or "").strip()
        if path and Path(path).expanduser().is_file() and _is_primary_plain_curtain_model(item):
            model_items.append(item)
    for path in curtain_model_paths or []:
        p = Path(path).expanduser()
        if p.is_file() and _is_primary_plain_curtain_model(p):
            model_items.append(
                {
                    "source": "curtain_models_dir",
                    "title": p.stem,
                    "unique_key": f"curtain_models_dir::{p}",
                    "asset_local_path": str(p.resolve()),
                    "asset_format": p.suffix.lstrip(".").lower(),
                    "asset_status": "local_file",
                }
            )

    added: list[dict[str, Any]] = []
    ceiling_height = float(room.get("ceiling_height") or 2.8)
    floor_z = float(room.get("floor_z") or 0.0)
    for idx, window in enumerate(windows):
        if not isinstance(window, dict):
            continue
        wall = wall_by_id.get(str(window.get("wall_id")))
        if not wall:
            continue
        pts = _wall_points(room, wall)
        if pts is None:
            continue
        (x0, y0), (x1, y1) = pts
        dx = x1 - x0
        dy = y1 - y0
        length = math.hypot(dx, dy)
        if length <= 1e-6:
            continue
        ux, uy = dx / length, dy / length
        inward = (-uy, ux)
        product = selected[idx % len(selected)]
        texture_path = _resolve_catalog_image(product, base_dir)
        if not texture_path:
            continue

        win_s = float(window.get("s") or 0.0)
        win_width = float(window.get("width") or 1.4)
        win_z0 = float(window.get("z0") or 0.8)
        win_height = float(window.get("height") or 1.2)
        curtain_width = max(win_width + 0.45, win_width * 1.25)
        curtain_depth = 0.10
        z_max = min(ceiling_height - 0.06, max(win_z0 + win_height + 0.25, ceiling_height - 0.18))
        z_min = max(floor_z + 0.035, z_max - max(1.8, min(3.2, z_max - floor_z - 0.035)))
        center_s = min(max(win_s + win_width * 0.5, curtain_width * 0.5), max(length - curtain_width * 0.5, curtain_width * 0.5))
        center = (x0 + ux * center_s + inward[0] * 0.075, y0 + uy * center_s + inward[1] * 0.075)
        yaw_deg = math.degrees(math.atan2(uy, ux))
        aabb = _aabb_for_oriented_panel(
            center=center,
            wall_dir=(ux, uy),
            inward=inward,
            width=curtain_width,
            depth=curtain_depth,
            z_min=z_min,
            z_max=z_max,
        )
        item_id = f"curtain_{window.get('id') or idx}"
        suffix = 2
        while item_id in existing_ids:
            item_id = f"curtain_{window.get('id') or idx}_{suffix}"
            suffix += 1
        existing_ids.add(item_id)
        price = product.get("price")
        model_item = model_items[idx % len(model_items)] if model_items else None
        model_path = str(model_item.get("asset_local_path") or "") if model_item else None
        asset = {
            "kind": "curtain_fbx_textured" if model_path else "procedural_curtain_proxy",
            "mesh_fit_mode": "curtain_soft_width",
            "texture_path": texture_path,
            "vertical_flip": True,
            "texture_tiling": {
                "mode": "mirror_repeat",
                "tile_size_m": 0.65,
            },
        }
        if model_path:
            asset["mesh_path"] = model_path
            asset["mesh_local_path"] = model_path
            asset["asset_local_path"] = model_path
            asset["asset_format"] = Path(model_path).suffix.lstrip(".").lower()
        curtain_item = {
            "id": item_id,
            "name": "ShtorystoreCurtain",
            "category": "CurtainFactory",
            "semantic_group": "curtain",
            "position_m": [center[0], center[1], (z_min + z_max) * 0.5],
            "size_m": [curtain_width, curtain_depth, z_max - z_min],
            "rotation_deg": yaw_deg,
            "yaw_deg": yaw_deg,
            "yaw_rad": math.radians(yaw_deg),
            "aabb": aabb,
            "constraints": {"mount_type": "wall", "near_window_id": window.get("id"), "wall_id": window.get("wall_id")},
            "asset": asset,
            "texture_path": texture_path,
            "texture_scale": 1.0,
            "source": {"placement_source": "shtorystore_curtain_postprocess"},
            "meta": {
                "curtain_proxy": True,
                "curtain_source": "shtorystore",
                "curtain_model_path": model_path,
                "curtain_model": {
                    "title": model_item.get("title"),
                    "unique_key": model_item.get("unique_key"),
                    "source": model_item.get("source"),
                    "asset_status": model_item.get("asset_status"),
                } if model_item else None,
                "window_id": window.get("id"),
                "wall_id": window.get("wall_id"),
                "product": {
                    "sku": product.get("sku"),
                    "name": product.get("name"),
                    "product_url": product.get("product_url"),
                    "price": price,
                    "old_price": product.get("old_price"),
                    "price_currency": product.get("price_currency") or "RUB",
                    "selected_material": product.get("selected_material"),
                    "selected_material_properties": product.get("selected_material_properties"),
                    "description": product.get("description"),
                    "image_selection_note": product.get("image_selection_note"),
                },
            },
        }
        items.append(curtain_item)
        added.append(curtain_item)

    meta = out.setdefault("meta", {})
    if isinstance(meta, dict):
        meta["curtain_postprocess"] = {
            "source": "shtorystore",
            "catalog_count": len(catalog),
            "added_count": len(added),
            "added_ids": [item["id"] for item in added],
        }
    return out, {
        "added_count": len(added),
        "added_ids": [item["id"] for item in added],
        "catalog_count": len(catalog),
        "model_count": len(model_items),
        "windows_count": len(windows),
        "selected": [
            {
                "id": item["id"],
                "sku": item["meta"]["product"].get("sku"),
                "name": item["meta"]["product"].get("name"),
                "texture_path": item.get("texture_path"),
                "price": item["meta"]["product"].get("price"),
            }
            for item in added
        ],
    }
