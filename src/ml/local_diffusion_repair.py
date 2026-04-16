#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Локальный diffusion-style repair для готовой расстановки.

Идея:
- если объект выходит за границы комнаты или пересекается с другими,
  двигаем только этот объект;
- остальные объекты остаются неподвижными;
- целевая функция предпочитает новые позиции, у которых bbox как можно
  сильнее пересекается с исходным bbox этого же объекта;
- коллизии и выход из комнаты считаются жёстким запретом.

Это не обученный denoiser, а локальный stochastic repair с диффузионным
расписанием шумов: сначала широкие смещения, затем всё более локальные.
Такой модуль удобно ставить как postprocess после любого placer.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from src.tools.evaluate_unified_scene import (
    Placement,
    Room,
    box_intersection_volume,
    parse_placements,
    parse_room,
    rect_intersection_area_xy,
    should_ignore_collision,
)
from src.tools.normalize_scene_format import (
    build_aabb_from_center_size,
    build_scene_from_room_and_placement,
    convert_to_scene_v1,
    convert_to_placement_v1,
)


EPS = 1e-9
IgnorePairSet = Set[Tuple[int, int]]


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def save_json(path: str | Path, data: Any) -> None:
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def clamp(x: float, lo: float, hi: float) -> float:
    if lo > hi:
        return 0.5 * (lo + hi)
    return max(lo, min(hi, x))


def _normalize_ignore_pair_indices(ignore_pair_indices: Optional[Sequence[Tuple[int, int]]]) -> IgnorePairSet:
    out: IgnorePairSet = set()
    for pair in ignore_pair_indices or []:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        a = int(pair[0])
        b = int(pair[1])
        if a == b:
            continue
        if a > b:
            a, b = b, a
        out.add((a, b))
    return out


def _pair_is_ignored(a: int, b: int, ignore_pair_indices: IgnorePairSet) -> bool:
    if a == b:
        return False
    if a > b:
        a, b = b, a
    return (a, b) in ignore_pair_indices


def point_in_polygon_xy(x: float, y: float, poly: Sequence[Tuple[float, float]]) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            denom = y2 - y1
            if abs(denom) < 1e-12:
                continue
            x_cross = x1 + (y - y1) * (x2 - x1) / denom
            if x <= x_cross:
                inside = not inside
    return inside


def aabb_inside_room(room: Room, aabb: Dict[str, float], margin: float = 1e-6) -> bool:
    room_height = float(room.z_max) - float(room.z_min)
    if aabb["z_min"] < room.z_min - margin:
        return False
    if room_height > max(margin, 1e-6) and aabb["z_max"] > room.z_max + margin:
        return False

    if room.floor_polygon and len(room.floor_polygon) >= 3:
        x0 = aabb["x_min"] + margin
        x1 = aabb["x_max"] - margin
        y0 = aabb["y_min"] + margin
        y1 = aabb["y_max"] - margin
        xm = 0.5 * (x0 + x1)
        ym = 0.5 * (y0 + y1)
        samples = [
            (x0, y0), (x0, y1), (x1, y0), (x1, y1),
            (xm, y0), (xm, y1), (x0, ym), (x1, ym), (xm, ym),
        ]
        return all(point_in_polygon_xy(x, y, room.floor_polygon) for x, y in samples)

    return (
        aabb["x_min"] >= room.x_min - margin and
        aabb["x_max"] <= room.x_max + margin and
        aabb["y_min"] >= room.y_min - margin and
        aabb["y_max"] <= room.y_max + margin
    )


def build_candidate_placement(src: Placement, x: float, y: float) -> Placement:
    position_m = (float(x), float(y), float(src.position_m[2]))
    aabb = build_aabb_from_center_size(list(position_m), list(src.size_m))
    return replace(src, position_m=position_m, aabb=aabb)


def collides_with_any(
    candidate: Placement,
    placements: Sequence[Placement],
    skip_idx: int,
    ignore_pair_indices: Optional[IgnorePairSet] = None,
) -> Tuple[bool, float, List[int]]:
    ignore_pair_indices = ignore_pair_indices or set()
    total_overlap = 0.0
    hit_indices: List[int] = []
    for j, other in enumerate(placements):
        if j == skip_idx:
            continue
        if _pair_is_ignored(skip_idx, j, ignore_pair_indices):
            continue
        if should_ignore_collision(candidate, other):
            continue
        inter = box_intersection_volume(candidate, other)
        if inter > 1e-6:
            total_overlap += inter
            hit_indices.append(j)
    return bool(hit_indices), total_overlap, hit_indices


def center_xy(p: Placement) -> Tuple[float, float]:
    return float(p.position_m[0]), float(p.position_m[1])


def object_volume(p: Placement) -> float:
    sx, sy, sz = p.size_m
    return max(float(sx) * float(sy) * float(sz), EPS)


def xy_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


@dataclass
class CandidateEval:
    candidate: Placement
    overlap_volume: float
    overlap_ratio: float
    overlap_area_xy: float
    displacement_m: float
    collision_overlap_volume: float
    colliding_indices: List[int]
    inside_room: bool

    @property
    def feasible(self) -> bool:
        return self.inside_room and not self.colliding_indices

    @property
    def sort_key(self) -> Tuple[float, float, float]:
        return (
            self.overlap_ratio,
            -self.displacement_m,
            -self.collision_overlap_volume,
        )


def score_candidate_pose(
    original: Placement,
    candidate: Placement,
    room: Room,
    placements: Sequence[Placement],
    index: int,
    room_margin: float = 1e-6,
    ignore_pair_indices: Optional[IgnorePairSet] = None,
) -> Dict[str, Any]:
    """
    Явная функция оценки для локального repair.

    Правила:
    - `feasible=False`, если bbox выходит из комнаты;
    - `feasible=False`, если есть коллизии с другими объектами;
    - среди feasible-кандидатов лучше тот, у кого больше overlap
      с исходным bbox этого же объекта;
    - при одинаковом overlap предпочтительнее меньший сдвиг.
    """
    inside = aabb_inside_room(room, candidate.aabb, margin=room_margin)
    has_collision, coll_vol, hit_indices = collides_with_any(
        candidate,
        placements,
        skip_idx=index,
        ignore_pair_indices=ignore_pair_indices,
    )
    overlap_vol = box_intersection_volume(original, candidate)
    overlap_area = rect_intersection_area_xy(original, candidate)
    disp = xy_distance(center_xy(original), center_xy(candidate))
    overlap_ratio = overlap_vol / object_volume(original)
    feasible = inside and not hit_indices
    return {
        "feasible": feasible,
        "inside_room": inside,
        "colliding_indices": hit_indices,
        "collision_overlap_volume": coll_vol if has_collision else 0.0,
        "overlap_volume": overlap_vol,
        "overlap_area_xy": overlap_area,
        "overlap_ratio": overlap_ratio,
        "displacement_m": disp,
        "score": overlap_ratio if feasible else -1.0,
    }


def evaluate_candidate(
    original: Placement,
    candidate: Placement,
    room: Room,
    placements: Sequence[Placement],
    index: int,
    room_margin: float,
    ignore_pair_indices: Optional[IgnorePairSet],
) -> CandidateEval:
    scored = score_candidate_pose(
        original=original,
        candidate=candidate,
        room=room,
        placements=placements,
        index=index,
        room_margin=room_margin,
        ignore_pair_indices=ignore_pair_indices,
    )
    return CandidateEval(
        candidate=candidate,
        overlap_volume=float(scored["overlap_volume"]),
        overlap_ratio=float(scored["overlap_ratio"]),
        overlap_area_xy=float(scored["overlap_area_xy"]),
        displacement_m=float(scored["displacement_m"]),
        collision_overlap_volume=float(scored["collision_overlap_volume"]),
        colliding_indices=list(scored["colliding_indices"]),
        inside_room=bool(scored["inside_room"]),
    )


def clamp_center_to_room_bbox(room: Room, size_m: Sequence[float], xy: Tuple[float, float]) -> Tuple[float, float]:
    hx = max(0.0, float(size_m[0]) * 0.5)
    hy = max(0.0, float(size_m[1]) * 0.5)
    x = clamp(float(xy[0]), room.x_min + hx, room.x_max - hx)
    y = clamp(float(xy[1]), room.y_min + hy, room.y_max - hy)
    return x, y


def dedupe_points(points: Iterable[Tuple[float, float]], ndigits: int = 5) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    seen = set()
    for x, y in points:
        key = (round(float(x), ndigits), round(float(y), ndigits))
        if key in seen:
            continue
        seen.add(key)
        out.append((float(x), float(y)))
    return out


def sigma_schedule(room: Room, item: Placement, steps: int) -> List[float]:
    base = max(float(item.size_m[0]), float(item.size_m[1]), 0.20)
    room_scale = max(room.x_max - room.x_min, room.y_max - room.y_min, base)
    hi = min(max(base * 1.5, 0.30), room_scale * 0.35)
    lo = max(min(base * 0.06, 0.08), 0.01)
    if steps <= 1:
        return [hi]
    return [
        float(hi * ((lo / hi) ** (k / max(steps - 1, 1))))
        for k in range(steps)
    ]


def proposal_points(
    origin_xy: Tuple[float, float],
    anchor_xy: Tuple[float, float],
    sigma: float,
    rng: random.Random,
    samples_per_step: int,
) -> List[Tuple[float, float]]:
    dirs = [
        (0.0, 0.0),
        (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
        (1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0),
        (2.0, 0.0), (-2.0, 0.0), (0.0, 2.0), (0.0, -2.0),
    ]
    pts: List[Tuple[float, float]] = [origin_xy, anchor_xy]
    for base in (origin_xy, anchor_xy):
        for dx, dy in dirs:
            pts.append((base[0] + dx * sigma, base[1] + dy * sigma))
            pts.append((base[0] + dx * sigma * 0.5, base[1] + dy * sigma * 0.5))
    for _ in range(max(0, samples_per_step)):
        if rng.random() < 0.55:
            base = anchor_xy
        else:
            base = origin_xy
        pts.append((base[0] + rng.gauss(0.0, sigma), base[1] + rng.gauss(0.0, sigma)))
    return dedupe_points(pts)


def collect_bad_indices(
    room: Room,
    placements: Sequence[Placement],
    room_margin: float,
    ignore_pair_indices: Optional[IgnorePairSet] = None,
) -> List[int]:
    bad = set()
    for i, p in enumerate(placements):
        if not aabb_inside_room(room, p.aabb, margin=room_margin):
            bad.add(i)
        _, _, hits = collides_with_any(
            p,
            placements,
            skip_idx=i,
            ignore_pair_indices=ignore_pair_indices,
        )
        if hits:
            bad.add(i)
            bad.update(hits)
    return sorted(bad)


def merge_repaired_placements_into_scene(
    scene: Dict[str, Any],
    placements: Sequence[Placement],
    repair_by_index: Dict[int, RepairResult],
) -> Dict[str, Any]:
    repaired_scene = deepcopy(scene)
    src_items = repaired_scene.get("placements")
    if not isinstance(src_items, list):
        src_items = []

    out_items: List[Dict[str, Any]] = []
    for i, p in enumerate(placements):
        src = deepcopy(src_items[i]) if i < len(src_items) and isinstance(src_items[i], dict) else {}
        src["position_m"] = [float(v) for v in p.position_m]
        src["size_m"] = [float(v) for v in p.size_m]
        src["rotation_deg"] = float(p.rotation_deg)
        src["yaw_deg"] = float(p.yaw_deg)
        src["yaw_rad"] = float(p.yaw_rad)
        src["aabb"] = {k: float(v) for k, v in p.aabb.items()}

        meta = src.get("meta")
        if not isinstance(meta, dict):
            meta = {}
            src["meta"] = meta
        if i in repair_by_index:
            meta["local_diffusion_repair"] = asdict(repair_by_index[i])

        out_items.append(src)

    repaired_scene["placements"] = out_items
    return repaired_scene


@dataclass
class RepairResult:
    index: int
    changed: bool
    success: bool
    reason_before: List[str]
    best_overlap_ratio: float
    best_overlap_area_xy: float
    displacement_m: float
    candidates_tested: int
    new_position_m: List[float]
    colliding_indices_before: List[int]
    colliding_indices_after: List[int]


class LocalDiffusionRepair:
    def __init__(
        self,
        steps: int = 7,
        samples_per_step: int = 96,
        room_margin: float = 1e-6,
        seed: int = 0,
        ignore_pair_indices: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> None:
        self.steps = int(max(1, steps))
        self.samples_per_step = int(max(0, samples_per_step))
        self.room_margin = float(max(0.0, room_margin))
        self.rng = random.Random(int(seed))
        self.ignore_pair_indices = _normalize_ignore_pair_indices(ignore_pair_indices)

    def repair_one(self, room: Room, placements: List[Placement], index: int) -> RepairResult:
        original = placements[index]
        original_xy = center_xy(original)

        reasons_before: List[str] = []
        inside_before = aabb_inside_room(room, original.aabb, margin=self.room_margin)
        has_collision_before, _, hits_before = collides_with_any(
            original,
            placements,
            skip_idx=index,
            ignore_pair_indices=self.ignore_pair_indices,
        )
        if not inside_before:
            reasons_before.append("outside_room")
        if has_collision_before:
            reasons_before.append("collision")

        best_eval = evaluate_candidate(
            original=original,
            candidate=original,
            room=room,
            placements=placements,
            index=index,
            room_margin=self.room_margin,
            ignore_pair_indices=self.ignore_pair_indices,
        )
        best_feasible = best_eval if best_eval.feasible else None
        best_xy = original_xy
        tested = 1

        for sigma in sigma_schedule(room, original, steps=self.steps):
            anchor = best_xy if best_feasible is not None else clamp_center_to_room_bbox(room, original.size_m, original_xy)
            points = proposal_points(
                origin_xy=original_xy,
                anchor_xy=anchor,
                sigma=sigma,
                rng=self.rng,
                samples_per_step=self.samples_per_step,
            )
            for x, y in points:
                x, y = clamp_center_to_room_bbox(room, original.size_m, (x, y))
                candidate = build_candidate_placement(original, x=x, y=y)
                cand_eval = evaluate_candidate(
                    original=original,
                    candidate=candidate,
                    room=room,
                    placements=placements,
                    index=index,
                    room_margin=self.room_margin,
                    ignore_pair_indices=self.ignore_pair_indices,
                )
                tested += 1
                if not cand_eval.feasible:
                    continue
                if best_feasible is None or cand_eval.sort_key > best_feasible.sort_key:
                    best_feasible = cand_eval
                    best_xy = (x, y)

        if best_feasible is None:
            after_hits = hits_before[:]
            return RepairResult(
                index=index,
                changed=False,
                success=False,
                reason_before=reasons_before,
                best_overlap_ratio=best_eval.overlap_ratio,
                best_overlap_area_xy=best_eval.overlap_area_xy,
                displacement_m=0.0,
                candidates_tested=tested,
                new_position_m=[float(v) for v in original.position_m],
                colliding_indices_before=hits_before,
                colliding_indices_after=after_hits,
            )

        placements[index] = best_feasible.candidate
        _, _, hits_after = collides_with_any(
            placements[index],
            placements,
            skip_idx=index,
            ignore_pair_indices=self.ignore_pair_indices,
        )
        changed = xy_distance(center_xy(original), center_xy(placements[index])) > 1e-7

        return RepairResult(
            index=index,
            changed=changed,
            success=True,
            reason_before=reasons_before,
            best_overlap_ratio=best_feasible.overlap_ratio,
            best_overlap_area_xy=best_feasible.overlap_area_xy,
            displacement_m=best_feasible.displacement_m,
            candidates_tested=tested,
            new_position_m=[float(v) for v in placements[index].position_m],
            colliding_indices_before=hits_before,
            colliding_indices_after=hits_after,
        )

    def repair_many(self, room: Room, placements: List[Placement], indices: Sequence[int]) -> List[RepairResult]:
        results: List[RepairResult] = []
        for idx in indices:
            results.append(self.repair_one(room=room, placements=placements, index=int(idx)))
        return results


def scene_from_inputs(
    scene_path: Optional[str],
    room_path: Optional[str],
    placement_path: Optional[str],
) -> Dict[str, Any]:
    if scene_path:
        scene_raw = load_json(scene_path)
        return convert_to_scene_v1(scene_raw)
    if room_path and placement_path:
        room_raw = load_json(room_path)
        placement_raw = load_json(placement_path)
        return build_scene_from_room_and_placement(room_raw, placement_raw)
    raise ValueError("Нужно передать либо --scene, либо пару --room + --placement")


def repair_scene_dict(
    scene: Dict[str, Any],
    bad_index: Optional[int],
    max_bad: Optional[int],
    steps: int,
    samples_per_step: int,
    room_margin: float,
    seed: int,
    candidate_indices: Optional[Sequence[int]] = None,
    ignore_pair_indices: Optional[Sequence[Tuple[int, int]]] = None,
) -> Tuple[Dict[str, Any], List[RepairResult], List[int]]:
    room = parse_room(scene)
    placements = parse_placements(scene)
    normalized_ignore_pairs = _normalize_ignore_pair_indices(ignore_pair_indices)
    initial_bad_all = collect_bad_indices(
        room,
        placements,
        room_margin=room_margin,
        ignore_pair_indices=normalized_ignore_pairs,
    )

    if bad_index is not None:
        indices = [int(bad_index)]
    else:
        indices = list(initial_bad_all)
        if candidate_indices is not None:
            candidate_set = {int(idx) for idx in candidate_indices}
            indices = [idx for idx in indices if idx in candidate_set]
        if max_bad is not None:
            indices = indices[: max(0, int(max_bad))]

    repairer = LocalDiffusionRepair(
        steps=steps,
        samples_per_step=samples_per_step,
        room_margin=room_margin,
        seed=seed,
        ignore_pair_indices=normalized_ignore_pairs,
    )
    results = repairer.repair_many(room=room, placements=placements, indices=indices)

    repair_by_index = {r.index: r for r in results}
    repaired_scene = merge_repaired_placements_into_scene(scene, placements, repair_by_index)
    repaired_scene["schema"] = "scene.v1"

    meta = repaired_scene.get("meta")
    if not isinstance(meta, dict):
        meta = {}
        repaired_scene["meta"] = meta
    meta["local_diffusion_repair"] = {
        "seed": int(seed),
        "steps": int(steps),
        "samples_per_step": int(samples_per_step),
        "room_margin": float(room_margin),
        "initial_bad_indices": initial_bad_all,
        "requested_indices": [int(idx) for idx in indices],
        "candidate_indices": [int(idx) for idx in candidate_indices] if candidate_indices is not None else None,
        "ignored_pair_count": len(normalized_ignore_pairs),
        "repaired_indices": [r.index for r in results],
        "results": [asdict(r) for r in results],
    }

    return repaired_scene, results, initial_bad_all


def build_output_placement(scene: Dict[str, Any]) -> Dict[str, Any]:
    placement = convert_to_placement_v1({
        "placer": "local_diffusion_repair",
        "placements": scene.get("placements", []),
        "meta": scene.get("meta", {}),
    })
    return placement


def main() -> None:
    ap = argparse.ArgumentParser(description="Local diffusion-style repair for one bad placement at a time.")
    ap.add_argument("--scene", default=None, help="Input scene-like JSON")
    ap.add_argument("--room", default=None, help="Input room.json")
    ap.add_argument("--placement", default=None, help="Input placement-like JSON")
    ap.add_argument("--out", required=True, help="Output repaired scene.v1.json")
    ap.add_argument("--out-placement", default=None, help="Optional repaired placement.v1.json")
    ap.add_argument("--bad-index", type=int, default=None, help="Repair only this placement index")
    ap.add_argument("--max-bad", type=int, default=None, help="Repair only first K auto-detected bad objects")
    ap.add_argument("--steps", type=int, default=7, help="Number of diffusion noise levels")
    ap.add_argument("--samples-per-step", type=int, default=96, help="Random proposals per noise level")
    ap.add_argument("--room-margin", type=float, default=1e-6, help="Inside-room tolerance")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    args = ap.parse_args()

    scene = scene_from_inputs(args.scene, args.room, args.placement)
    repaired_scene, results, initial_bad = repair_scene_dict(
        scene=scene,
        bad_index=args.bad_index,
        max_bad=args.max_bad,
        steps=args.steps,
        samples_per_step=args.samples_per_step,
        room_margin=args.room_margin,
        seed=args.seed,
    )
    save_json(args.out, repaired_scene)

    if args.out_placement:
        save_json(args.out_placement, build_output_placement(repaired_scene))

    repaired_count = sum(1 for r in results if r.success and r.changed)
    failed_count = sum(1 for r in results if not r.success)
    print(
        "[local_diffusion_repair] "
        f"initial_bad={len(initial_bad)} "
        f"requested={len(results)} "
        f"moved={repaired_count} "
        f"failed={failed_count} "
        f"out={Path(args.out).expanduser().resolve()}"
    )


if __name__ == "__main__":
    main()
