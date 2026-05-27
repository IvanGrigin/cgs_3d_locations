from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

COLLISION_VOLUME_EPS_M3 = 0.01


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def save_json(path: str | Path, data: Any) -> None:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def as_str(v: Any, default: str = "") -> str:
    if v is None:
        return default
    return str(v)


def round6(v: float) -> float:
    return round(float(v), 6)


def wrap_angle_deg(angle: float) -> float:
    return float((angle + 180.0) % 360.0 - 180.0)


def angle_abs_diff_deg(a: float, b: float) -> float:
    return abs(wrap_angle_deg(float(a) - float(b)))


def point_in_polygon(point: Tuple[float, float], polygon: List[List[float]]) -> bool:
    x, y = point
    inside = False
    n = len(polygon)
    if n < 3:
        return False
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        cond = ((y1 > y) != (y2 > y))
        if cond:
            x_inter = (x2 - x1) * (y - y1) / max(y2 - y1, 1e-12) + x1
            if x < x_inter:
                inside = not inside
    return inside


def polygon_area(poly: List[Tuple[float, float]]) -> float:
    if len(poly) < 3:
        return 0.0
    s = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        s += x1 * y2 - x2 * y1
    return abs(s) * 0.5


def line_intersection(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    q1: Tuple[float, float],
    q2: Tuple[float, float],
) -> Tuple[float, float]:
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = q1
    x4, y4 = q2
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-12:
        return p2
    det1 = x1 * y2 - y1 * x2
    det2 = x3 * y4 - y3 * x4
    px = (det1 * (x3 - x4) - (x1 - x2) * det2) / den
    py = (det1 * (y3 - y4) - (y1 - y2) * det2) / den
    return (float(px), float(py))


def polygon_clip_convex(
    subject: List[Tuple[float, float]],
    clipper: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    def inside(p: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> bool:
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= -1e-9

    output = subject[:]
    for i in range(len(clipper)):
        a = clipper[i]
        b = clipper[(i + 1) % len(clipper)]
        input_list = output[:]
        output = []
        if not input_list:
            break
        s = input_list[-1]
        for e in input_list:
            if inside(e, a, b):
                if not inside(s, a, b):
                    output.append(line_intersection(s, e, a, b))
                output.append(e)
            elif inside(s, a, b):
                output.append(line_intersection(s, e, a, b))
            s = e
    return output


def rectangle_corners(cx: float, cy: float, sx: float, sy: float, yaw_deg: float) -> List[Tuple[float, float]]:
    hx = 0.5 * float(sx)
    hy = 0.5 * float(sy)
    ang = math.radians(float(yaw_deg))
    c = math.cos(ang)
    s = math.sin(ang)
    local = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
    out: List[Tuple[float, float]] = []
    for lx, ly in local:
        x = cx + c * lx - s * ly
        y = cy + s * lx + c * ly
        out.append((float(x), float(y)))
    return out


def footprint_area(size_m: List[float]) -> float:
    return max(0.0, float(size_m[0]) * float(size_m[1]))


def category_lower(p: Dict[str, Any]) -> str:
    return as_str(p.get("category")).strip().lower()


def super_category_lower(p: Dict[str, Any]) -> str:
    return as_str(((p.get("meta") or {}).get("super_category"))).strip().lower()


def footprint_iou(a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[float, float]:
    poly_a = rectangle_corners(
        cx=float(a["position_m"][0]),
        cy=float(a["position_m"][1]),
        sx=float(a["size_m"][0]),
        sy=float(a["size_m"][1]),
        yaw_deg=float(a["yaw_deg"]),
    )
    poly_b = rectangle_corners(
        cx=float(b["position_m"][0]),
        cy=float(b["position_m"][1]),
        sx=float(b["size_m"][0]),
        sy=float(b["size_m"][1]),
        yaw_deg=float(b["yaw_deg"]),
    )
    inter_poly = polygon_clip_convex(poly_a, poly_b)
    inter_area = polygon_area(inter_poly)
    union_area = polygon_area(poly_a) + polygon_area(poly_b) - inter_area
    iou = 0.0 if union_area <= 1e-12 else inter_area / union_area
    return float(iou), float(inter_area)


def update_aabb_for_placement(placement: Dict[str, Any]) -> Dict[str, float]:
    x, y, z = [float(v) for v in placement["position_m"]]
    sx, sy, sz = [float(v) for v in placement["size_m"]]
    corners = rectangle_corners(x, y, sx, sy, float(placement["yaw_deg"]))
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    return {
        "x_min": round6(min(xs)),
        "x_max": round6(max(xs)),
        "y_min": round6(min(ys)),
        "y_max": round6(max(ys)),
        "z_min": round6(z - 0.5 * sz),
        "z_max": round6(z + 0.5 * sz),
    }


def clone_placement(p: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(p))


def build_scene_index(placements: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {as_str(p["id"]): p for p in placements}


def is_floor_candidate(p: Dict[str, Any]) -> bool:
    if as_str(p.get("mount_type")).strip().lower() != "floor":
        return False
    size = p.get("size_m") or []
    if len(size) < 3:
        return False
    return min(float(size[0]), float(size[1]), float(size[2])) > 1e-6


def is_yaw_sensitive(p: Dict[str, Any]) -> bool:
    sx, sy, _ = [float(v) for v in p["size_m"]]
    aspect = max(sx, sy) / max(min(sx, sy), 1e-6)
    super_cat = as_str(((p.get("meta") or {}).get("super_category"))).lower()
    return aspect >= 1.15 or super_cat in {"cabinet/shelf/desk", "bed", "sofa"}


def aabb_intersection_metrics(a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[float, float]:
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
    area_2d = dx * dy
    volume_3d = area_2d * dz
    return float(area_2d), float(volume_3d)


def is_bedside_overlap_allowed(
    a: Dict[str, Any],
    b: Dict[str, Any],
    inter_area_2d: float,
    inter_volume_3d: float,
) -> bool:
    ca = category_lower(a)
    cb = category_lower(b)
    pair = {ca, cb}
    has_bed = any("bed" in x for x in pair)
    has_bedside = ("nightstand" in pair) or ("corner/side table" in pair)
    if not (has_bed and has_bedside):
        return False
    min_area = max(min(footprint_area(a["size_m"]), footprint_area(b["size_m"])), 1e-6)
    area_ratio = inter_area_2d / min_area
    return inter_volume_3d <= 0.03 and area_ratio <= 0.40


def is_chair_table_overlap_allowed(
    a: Dict[str, Any],
    b: Dict[str, Any],
    inter_area_2d: float,
    inter_volume_3d: float,
) -> bool:
    sa = super_category_lower(a)
    sb = super_category_lower(b)
    ca = category_lower(a)
    cb = category_lower(b)
    chair_table = (
        (sa == "chair" and (sb == "table" or cb in {"desk", "dressing table"}))
        or (sb == "chair" and (sa == "table" or ca in {"desk", "dressing table"}))
    )
    if not chair_table:
        return False
    min_area = max(min(footprint_area(a["size_m"]), footprint_area(b["size_m"])), 1e-6)
    area_ratio = inter_area_2d / min_area
    return inter_volume_3d <= 0.02 and area_ratio <= 0.25


def should_ignore_pair_overlap(
    a: Dict[str, Any],
    b: Dict[str, Any],
    inter_area_2d: float,
    inter_volume_3d: float,
) -> bool:
    return (
        is_bedside_overlap_allowed(a, b, inter_area_2d, inter_volume_3d)
        or is_chair_table_overlap_allowed(a, b, inter_area_2d, inter_volume_3d)
    )


def collision_metrics_for_target(
    target: Dict[str, Any],
    placements: List[Dict[str, Any]],
) -> Tuple[int, float, float, List[str]]:
    count = 0
    total_area = 0.0
    total_volume = 0.0
    colliding_ids: List[str] = []
    for other in placements:
        if as_str(other["id"]) == as_str(target["id"]):
            continue
        if as_str(other.get("mount_type")).strip().lower() != "floor":
            continue
        inter_area_2d, inter_volume_3d = aabb_intersection_metrics(target, other)
        if should_ignore_pair_overlap(target, other, inter_area_2d, inter_volume_3d):
            continue
        if inter_volume_3d > COLLISION_VOLUME_EPS_M3:
            count += 1
            total_area += inter_area_2d
            total_volume += inter_volume_3d
            colliding_ids.append(as_str(other["id"]))
    return count, float(total_area), float(total_volume), colliding_ids


def corners_inside_ratio(placement: Dict[str, Any], room_polygon: List[List[float]]) -> Tuple[int, float]:
    corners = rectangle_corners(
        cx=float(placement["position_m"][0]),
        cy=float(placement["position_m"][1]),
        sx=float(placement["size_m"][0]),
        sy=float(placement["size_m"][1]),
        yaw_deg=float(placement["yaw_deg"]),
    )
    inside = sum(1 for c in corners if point_in_polygon(c, room_polygon))
    return inside, inside / 4.0


def floor_contact_abs_error_m(placement: Dict[str, Any]) -> float:
    z = float(placement["position_m"][2])
    sz = float(placement["size_m"][2])
    return abs((z - 0.5 * sz) - 0.0)


def quality_score(
    pos_err_m: float,
    yaw_err_deg: float,
    target_area: float,
    collision_area: float,
    corners_inside_ratio_value: float,
    floor_err_m: float,
) -> float:
    pos_pen = min(pos_err_m / 2.0, 1.0)
    yaw_pen = min(yaw_err_deg / 90.0, 1.0)
    coll_pen = min(collision_area / max(target_area, 1e-6), 1.0)
    outside_pen = 1.0 - max(0.0, min(corners_inside_ratio_value, 1.0))
    floor_pen = min(floor_err_m / 0.15, 1.0)
    score = 1.0 - (0.20 * pos_pen + 0.15 * yaw_pen + 0.35 * coll_pen + 0.20 * outside_pen + 0.10 * floor_pen)
    return max(0.0, min(1.0, float(score)))


def compute_metrics(
    clean_target: Dict[str, Any],
    eval_target: Dict[str, Any],
    eval_scene_placements: List[Dict[str, Any]],
    room_polygon: List[List[float]],
) -> Dict[str, Any]:
    pos_clean = np.array(clean_target["position_m"], dtype=np.float32)
    pos_eval = np.array(eval_target["position_m"], dtype=np.float32)
    pos_err = float(np.linalg.norm(pos_eval - pos_clean))
    yaw_err = angle_abs_diff_deg(float(eval_target["yaw_deg"]), float(clean_target["yaw_deg"]))
    iou2d, inter_area_clean = footprint_iou(clean_target, eval_target)
    coll_count, coll_area, coll_volume, colliding_ids = collision_metrics_for_target(eval_target, eval_scene_placements)
    inside_count, inside_ratio = corners_inside_ratio(eval_target, room_polygon)
    center_inside = point_in_polygon(
        (float(eval_target["position_m"][0]), float(eval_target["position_m"][1])),
        room_polygon,
    )
    floor_err = floor_contact_abs_error_m(eval_target)
    target_area = footprint_area(eval_target["size_m"])
    outside_room = (not center_inside) or (inside_count == 0)
    valid = (coll_count == 0) and (not outside_room) and (floor_err <= 0.08)
    return {
        "position_l2_m": round6(pos_err),
        "yaw_abs_error_deg": round6(yaw_err),
        "footprint_iou_2d": round6(iou2d),
        "footprint_intersection_area_with_clean_2d": round6(inter_area_clean),
        "target_area_2d": round6(target_area),
        "collision_pair_count": int(coll_count),
        "collision_area_sum_2d": round6(coll_area),
        "collision_volume_sum_3d": round6(coll_volume),
        "colliding_with_ids": colliding_ids,
        "corners_inside_count": int(inside_count),
        "corners_inside_ratio": round6(inside_ratio),
        "center_inside_room": bool(center_inside),
        "outside_room": bool(outside_room),
        "floor_contact_abs_error_m": round6(floor_err),
        "valid": bool(valid),
        "quality_score": round6(
            quality_score(
                pos_err_m=pos_err,
                yaw_err_deg=yaw_err,
                target_area=target_area,
                collision_area=coll_area,
                corners_inside_ratio_value=inside_ratio,
                floor_err_m=floor_err,
            )
        ),
    }


def copy_scene_with_target(
    placements: List[Dict[str, Any]],
    target_id: str,
    new_target: Dict[str, Any],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    changed = 0
    for p in placements:
        if as_str(p["id"]) == target_id:
            out.append(clone_placement(new_target))
            changed += 1
        else:
            out.append(clone_placement(p))
    if changed != 1:
        raise RuntimeError(f"Expected exactly one changed target, got {changed} for target_id={target_id}")
    return out


def room_bounds(room_polygon: List[List[float]]) -> Tuple[float, float, float, float]:
    xs = [float(p[0]) for p in room_polygon]
    ys = [float(p[1]) for p in room_polygon]
    return min(xs), min(ys), max(xs), max(ys)


def gen_out_of_room_shift(
    target: Dict[str, Any],
    room_polygon: List[List[float]],
    rng: np.random.Generator,
) -> Optional[Dict[str, Any]]:
    p = clone_placement(target)
    x_min, y_min, x_max, y_max = room_bounds(room_polygon)
    sx, sy, _ = [float(v) for v in p["size_m"]]
    margin = float(rng.uniform(0.05, 0.25))
    side = int(rng.integers(0, 4))
    if side == 0:
        p["position_m"][0] = round6(x_min - 0.5 * sx - margin)
        p["position_m"][1] = round6(float(rng.uniform(y_min, y_max)))
    elif side == 1:
        p["position_m"][0] = round6(x_max + 0.5 * sx + margin)
        p["position_m"][1] = round6(float(rng.uniform(y_min, y_max)))
    elif side == 2:
        p["position_m"][1] = round6(y_min - 0.5 * sy - margin)
        p["position_m"][0] = round6(float(rng.uniform(x_min, x_max)))
    else:
        p["position_m"][1] = round6(y_max + 0.5 * sy + margin)
        p["position_m"][0] = round6(float(rng.uniform(x_min, x_max)))
    p["aabb"] = update_aabb_for_placement(p)
    return p


def gen_yaw_only(
    target: Dict[str, Any],
    rng: np.random.Generator,
) -> Optional[Dict[str, Any]]:
    if not is_yaw_sensitive(target):
        return None
    p = clone_placement(target)
    delta = float(rng.choice(np.array([25.0, 35.0, 45.0, 60.0, 90.0, 120.0, 135.0, 180.0], dtype=np.float32)))
    if int(rng.integers(0, 2)) == 0:
        delta = -delta
    p["yaw_deg"] = round6(float(p["yaw_deg"]) + delta)
    p["yaw_rad"] = round6(math.radians(float(p["yaw_deg"])))
    p["rotation_deg"] = int(round(float(p["yaw_deg"]))) % 360
    p["aabb"] = update_aabb_for_placement(p)
    return p


def gen_displaced_inside_room(
    target: Dict[str, Any],
    room_polygon: List[List[float]],
    rng: np.random.Generator,
) -> Optional[Dict[str, Any]]:
    p = clone_placement(target)
    x_min, y_min, x_max, y_max = room_bounds(room_polygon)
    orig_x, orig_y = [float(v) for v in p["position_m"][:2]]
    sx, sy, _ = [float(v) for v in p["size_m"]]

    for _ in range(64):
        dx = float(rng.uniform(-1.0, 1.0)) * max(0.35, sx)
        dy = float(rng.uniform(-1.0, 1.0)) * max(0.35, sy)
        nx = orig_x + dx
        ny = orig_y + dy
        cand = clone_placement(p)
        cand["position_m"][0] = round6(float(np.clip(nx, x_min + 0.5 * sx, x_max - 0.5 * sx)))
        cand["position_m"][1] = round6(float(np.clip(ny, y_min + 0.5 * sy, y_max - 0.5 * sy)))
        cand["aabb"] = update_aabb_for_placement(cand)
        inside_count, _ = corners_inside_ratio(cand, room_polygon)
        disp = math.dist(cand["position_m"][:2], target["position_m"][:2])
        if inside_count == 4 and disp >= 0.20:
            return cand
    return None


def gen_shift_and_yaw(
    target: Dict[str, Any],
    room_polygon: List[List[float]],
    rng: np.random.Generator,
) -> Optional[Dict[str, Any]]:
    p = gen_displaced_inside_room(target, room_polygon, rng)
    if p is None:
        return None
    delta = float(rng.choice(np.array([20.0, 30.0, 45.0, 60.0, 90.0], dtype=np.float32)))
    if int(rng.integers(0, 2)) == 0:
        delta = -delta
    p["yaw_deg"] = round6(float(p["yaw_deg"]) + delta)
    p["yaw_rad"] = round6(math.radians(float(p["yaw_deg"])))
    p["rotation_deg"] = int(round(float(p["yaw_deg"]))) % 360
    p["aabb"] = update_aabb_for_placement(p)
    return p


def gen_collision_shift(
    target: Dict[str, Any],
    floor_placements: List[Dict[str, Any]],
    rng: np.random.Generator,
) -> Optional[Dict[str, Any]]:
    others = [p for p in floor_placements if as_str(p["id"]) != as_str(target["id"])]
    if not others:
        return None
    other = others[int(rng.integers(0, len(others)))]
    p = clone_placement(target)
    tx, ty = [float(v) for v in p["position_m"][:2]]
    ox, oy = [float(v) for v in other["position_m"][:2]]
    tsx, tsy, _ = [float(v) for v in p["size_m"]]
    osx, osy, _ = [float(v) for v in other["size_m"]]
    off_x = float(rng.uniform(-0.45, 0.45)) * (0.5 * (tsx + osx))
    off_y = float(rng.uniform(-0.45, 0.45)) * (0.5 * (tsy + osy))
    if abs(off_x) + abs(off_y) < 1e-4:
        off_x = 0.15 * (tsx + osx)
    p["position_m"][0] = round6(ox + off_x)
    p["position_m"][1] = round6(oy + off_y)
    p["aabb"] = update_aabb_for_placement(p)
    if math.dist((tx, ty), (p["position_m"][0], p["position_m"][1])) < 0.05:
        return None
    return p


GENERATOR_MAP = {
    "collision_shift": gen_collision_shift,
    "out_of_room_shift": gen_out_of_room_shift,
    "yaw_only": gen_yaw_only,
    "displaced_inside_room": gen_displaced_inside_room,
    "shift_and_yaw": gen_shift_and_yaw,
}


def generate_corrupted_target(
    corruption_type: str,
    target: Dict[str, Any],
    floor_placements: List[Dict[str, Any]],
    room_polygon: List[List[float]],
    rng: np.random.Generator,
) -> Optional[Dict[str, Any]]:
    fn = GENERATOR_MAP[corruption_type]
    if corruption_type == "collision_shift":
        return fn(target, floor_placements, rng)  # type: ignore[misc]
    if corruption_type in {"out_of_room_shift", "displaced_inside_room", "shift_and_yaw"}:
        return fn(target, room_polygon, rng)  # type: ignore[misc]
    return fn(target, rng)  # type: ignore[misc]


def delta_dict(clean_target: Dict[str, Any], corrupted_target: Dict[str, Any]) -> Dict[str, Any]:
    clean_pos = np.array(clean_target["position_m"], dtype=np.float32)
    corr_pos = np.array(corrupted_target["position_m"], dtype=np.float32)
    dpos = corr_pos - clean_pos
    dyaw = wrap_angle_deg(float(corrupted_target["yaw_deg"]) - float(clean_target["yaw_deg"]))
    return {
        "dx_m": round6(float(dpos[0])),
        "dy_m": round6(float(dpos[1])),
        "dz_m": round6(float(dpos[2])),
        "translation_l2_m": round6(float(np.linalg.norm(dpos))),
        "dyaw_deg": round6(float(dyaw)),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build single-object repair corruption samples from repair_v1 scenes")
    ap.add_argument("--dataset-root", required=True, help="Path to repair_v1 dataset root")
    ap.add_argument("--out-dir", required=True, help="Output directory for corruption dataset")
    ap.add_argument("--room-types", default="", help="Optional comma-separated room types")
    ap.add_argument("--split", default="", help="Optional split filter: train,val,test")
    ap.add_argument("--corruption-types", default="collision_shift,out_of_room_shift,yaw_only,displaced_inside_room,shift_and_yaw")
    ap.add_argument("--samples-per-room", type=int, default=4)
    ap.add_argument("--limit-rooms", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--write-debug-scenes", action="store_true", help="Write explicit corrupted scene_gt JSONs for smoke/debug")
    ap.add_argument("--debug-scene-limit", type=int, default=100)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = out_dir / "debug_scenes"
    if args.write_debug_scenes:
        debug_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = dataset_root / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest.jsonl not found: {manifest_path}")

    allowed_room_types = {x.strip().lower() for x in as_str(args.room_types).split(",") if x.strip()}
    allowed_split = as_str(args.split).strip().lower() or None
    corruption_types = [x.strip() for x in as_str(args.corruption_types).split(",") if x.strip()]
    for name in corruption_types:
        if name not in GENERATOR_MAP:
            raise ValueError(f"Unknown corruption type: {name}")

    rooms: List[Dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            room_type = as_str(row.get("room_type")).strip().lower()
            split = as_str(row.get("split")).strip().lower()
            if allowed_room_types and room_type not in allowed_room_types:
                continue
            if allowed_split and split != allowed_split:
                continue
            rooms.append(row)
            if args.limit_rooms > 0 and len(rooms) >= int(args.limit_rooms):
                break

    if not rooms:
        raise RuntimeError("No rooms passed the filters")

    stats_counter = Counter()
    category_counter = Counter()
    super_counter = Counter()
    split_counter = Counter()
    room_type_counter = Counter()
    translation_values: List[float] = []
    yaw_values: List[float] = []
    clean_scores: List[float] = []
    corrupted_scores: List[float] = []
    quality_drop_values: List[float] = []
    debug_written = 0
    sample_count = 0

    samples_path = out_dir / "samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as out_f:
        for room_idx, room_row in enumerate(rooms):
            room_id = as_str(room_row["room_id"])
            room_type = as_str(room_row["room_type"]).strip().lower()
            split = as_str(room_row.get("split")).strip().lower() or "unspecified"

            room_json = load_json(room_row["room_json"])
            scene_gt = load_json(room_row["scene_gt_v1_json"])
            placements = scene_gt["placements"]
            floor_placements = [clone_placement(p) for p in placements if is_floor_candidate(p)]
            if not floor_placements:
                continue

            room_polygon = room_json["floor_polygon_xz"]
            pair_candidates = []
            for p in floor_placements:
                for corruption_type in corruption_types:
                    if corruption_type == "yaw_only" and not is_yaw_sensitive(p):
                        continue
                    pair_candidates.append((as_str(p["id"]), corruption_type))

            if not pair_candidates:
                continue

            rng = np.random.default_rng(args.seed + room_idx * 1009)
            order = np.arange(len(pair_candidates))
            rng.shuffle(order)
            picked = [pair_candidates[int(i)] for i in order[: min(len(order), int(args.samples_per_room))]]

            scene_index = build_scene_index(floor_placements)

            for pair_idx, (target_id, corruption_type) in enumerate(picked):
                clean_target = clone_placement(scene_index[target_id])
                corrupted_target = generate_corrupted_target(
                    corruption_type=corruption_type,
                    target=clean_target,
                    floor_placements=floor_placements,
                    room_polygon=room_polygon,
                    rng=np.random.default_rng(args.seed + room_idx * 1009 + pair_idx * 10007),
                )
                if corrupted_target is None:
                    continue
                corrupted_target["aabb"] = update_aabb_for_placement(corrupted_target)

                clean_scene = copy_scene_with_target(placements, target_id, clean_target)
                corrupted_scene = copy_scene_with_target(placements, target_id, corrupted_target)

                clean_metrics = compute_metrics(
                    clean_target=clean_target,
                    eval_target=clean_target,
                    eval_scene_placements=clean_scene,
                    room_polygon=room_polygon,
                )
                corrupted_metrics = compute_metrics(
                    clean_target=clean_target,
                    eval_target=corrupted_target,
                    eval_scene_placements=corrupted_scene,
                    room_polygon=room_polygon,
                )

                delta = delta_dict(clean_target, corrupted_target)
                effective = (
                    corrupted_metrics["position_l2_m"] >= 0.15
                    or corrupted_metrics["yaw_abs_error_deg"] >= 15.0
                    or corrupted_metrics["collision_pair_count"] > 0
                    or corrupted_metrics["outside_room"]
                )
                if not effective:
                    continue

                target_meta = clean_target.get("meta") or {}
                sample_id = f"{room_id}__{target_id}__{corruption_type}__{pair_idx:03d}"
                sample = {
                    "schema": "repair_sample.v1",
                    "sample_id": sample_id,
                    "room_id": room_id,
                    "room_type": room_type,
                    "split": split,
                    "target_object_id": target_id,
                    "target_category": as_str(clean_target.get("category")),
                    "target_super_category": as_str(target_meta.get("super_category")),
                    "corruption_type": corruption_type,
                    "context_unchanged": True,
                    "clean_scene_ref": room_row["scene_gt_v1_json"],
                    "room_ref": room_row["room_json"],
                    "clean_pose": {
                        "position_m": [round6(v) for v in clean_target["position_m"]],
                        "size_m": [round6(v) for v in clean_target["size_m"]],
                        "yaw_deg": round6(float(clean_target["yaw_deg"])),
                        "yaw_rad": round6(float(clean_target["yaw_rad"])),
                        "mount_type": clean_target.get("mount_type"),
                    },
                    "corrupted_pose": {
                        "position_m": [round6(v) for v in corrupted_target["position_m"]],
                        "size_m": [round6(v) for v in corrupted_target["size_m"]],
                        "yaw_deg": round6(float(corrupted_target["yaw_deg"])),
                        "yaw_rad": round6(float(corrupted_target["yaw_rad"])),
                        "mount_type": corrupted_target.get("mount_type"),
                    },
                    "delta": delta,
                    "clean_metrics": clean_metrics,
                    "corrupted_metrics": corrupted_metrics,
                }

                if args.write_debug_scenes and debug_written < int(args.debug_scene_limit):
                    debug_scene_path = debug_dir / f"{sample_id}.scene_gt.v1.json"
                    debug_scene = dict(scene_gt)
                    debug_scene["placements"] = corrupted_scene
                    save_json(debug_scene_path, debug_scene)
                    sample["debug_corrupted_scene_ref"] = str(debug_scene_path.resolve())
                    debug_written += 1

                out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")

                sample_count += 1
                stats_counter["samples_total"] += 1
                stats_counter[f"by_corruption_type::{corruption_type}"] += 1
                if clean_metrics["valid"]:
                    stats_counter["clean_valid_samples"] += 1
                if corrupted_metrics["valid"]:
                    stats_counter["corrupted_valid_samples"] += 1
                if corrupted_metrics["collision_pair_count"] > 0:
                    stats_counter["corrupted_with_collision"] += 1
                if corrupted_metrics["outside_room"]:
                    stats_counter["corrupted_out_of_room"] += 1
                if corrupted_metrics["yaw_abs_error_deg"] >= 15.0:
                    stats_counter["corrupted_with_large_yaw_error"] += 1

                category_counter[sample["target_category"]] += 1
                super_counter[sample["target_super_category"]] += 1
                split_counter[split] += 1
                room_type_counter[room_type] += 1
                translation_values.append(float(delta["translation_l2_m"]))
                yaw_values.append(abs(float(delta["dyaw_deg"])))
                clean_scores.append(float(clean_metrics["quality_score"]))
                corrupted_scores.append(float(corrupted_metrics["quality_score"]))
                quality_drop_values.append(float(clean_metrics["quality_score"]) - float(corrupted_metrics["quality_score"]))

    if sample_count <= 0:
        raise RuntimeError("No corruption samples were generated")

    def mean(xs: Iterable[float]) -> float:
        xs = list(xs)
        return 0.0 if not xs else float(sum(xs) / len(xs))

    summary = {
        "schema": "repair_corruptions_v1.stats",
        "dataset_root": str(dataset_root),
        "out_dir": str(out_dir),
        "samples_total": sample_count,
        "samples_per_room_requested": int(args.samples_per_room),
        "room_filters": {
            "room_types": sorted(allowed_room_types),
            "split": allowed_split,
            "limit_rooms": int(args.limit_rooms),
        },
        "corruption_types": corruption_types,
        "clean_valid_rate": round6(stats_counter["clean_valid_samples"] / sample_count),
        "corrupted_valid_rate": round6(stats_counter["corrupted_valid_samples"] / sample_count),
        "corrupted_collision_rate": round6(stats_counter["corrupted_with_collision"] / sample_count),
        "corrupted_out_of_room_rate": round6(stats_counter["corrupted_out_of_room"] / sample_count),
        "corrupted_large_yaw_error_rate": round6(stats_counter["corrupted_with_large_yaw_error"] / sample_count),
        "mean_translation_l2_m": round6(mean(translation_values)),
        "mean_abs_dyaw_deg": round6(mean(yaw_values)),
        "mean_clean_quality_score": round6(mean(clean_scores)),
        "mean_corrupted_quality_score": round6(mean(corrupted_scores)),
        "mean_quality_drop": round6(mean(quality_drop_values)),
        "by_corruption_type": {
            name: int(stats_counter[f"by_corruption_type::{name}"])
            for name in corruption_types
        },
        "by_split": dict(split_counter),
        "by_room_type": dict(room_type_counter),
        "top_target_categories": [{"name": k, "count": int(v)} for k, v in category_counter.most_common(30)],
        "top_target_super_categories": [{"name": k, "count": int(v)} for k, v in super_counter.most_common(15)],
    }
    save_json(out_dir / "stats.json", summary)

    print(f"[repair_corruptions_v1] samples={sample_count}")
    print(f"[repair_corruptions_v1] clean_valid_rate={summary['clean_valid_rate']} corrupted_valid_rate={summary['corrupted_valid_rate']}")
    print(f"[repair_corruptions_v1] corrupted_collision_rate={summary['corrupted_collision_rate']} corrupted_out_of_room_rate={summary['corrupted_out_of_room_rate']}")
    print(f"[repair_corruptions_v1] mean_translation_l2_m={summary['mean_translation_l2_m']} mean_abs_dyaw_deg={summary['mean_abs_dyaw_deg']}")
    print(f"[repair_corruptions_v1] wrote samples={samples_path}")
    print(f"[repair_corruptions_v1] wrote stats={out_dir / 'stats.json'}")


if __name__ == "__main__":
    main()
