#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
area_appraiser.py

Кодовая оценка сцены без обращения к LLM.

Что считает:
1. Базовую геометрию комнаты и предметов.
2. Пересечения предметов на полу.
3. Выход предметов за границы комнаты.
4. Долю свободной площади пола.
5. Метрику свободного пространства:
      largest_free_rectangle_area / (room_area - sum(item_floor_areas))
6. Достижимость предметов человеком по сетке через A*.
7. Совпадение набора предметов с prompt.
8. Соблюдение простых ограничений вида touch_wall / mount_type.

Формат сцены ожидается совместимый с scene.v1:
- room.floor_polygon
- room.area_m2 (опционально)
- placements[*].aabb / size_m / position_m / rotation_deg / yaw_deg / constraints

Модуль намеренно написан без внешних зависимостей.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Optional

EPS = 1e-9


# ------------------------------------------------------------
# Dataclasses
# ------------------------------------------------------------


@dataclass
class AccessResult:
    object_id: str
    name: str
    category: str
    reachable: bool
    reason: str
    reachable_target_count: int
    total_target_count: int


@dataclass
class ConstraintCheck:
    object_id: str
    name: str
    category: str
    constraint_type: str
    expected: Any
    satisfied: bool
    details: dict[str, Any]


@dataclass
class AreaAppraisalResult:
    score_10: float
    room_area_m2: float
    raw_sum_item_area_m2: float
    free_area_m2: float
    occupied_union_area_m2: float
    free_union_area_m2: float
    largest_free_rectangle_area_m2: float
    largest_free_rectangle_ratio: float
    floor_coverage_ratio: float
    overlap_area_m2: float
    overlap_ratio: float
    outside_room_area_m2: float
    outside_room_ratio: float
    accessible_objects: int
    accessible_objects_total: int
    accessibility_ratio: float
    prompt_match_score_10: float
    constraint_score_10: float
    geometry_score_10: float
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------
# Geometry helpers
# ------------------------------------------------------------


@dataclass
class Rect:
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    @property
    def width(self) -> float:
        return max(0.0, self.x_max - self.x_min)

    @property
    def height(self) -> float:
        return max(0.0, self.y_max - self.y_min)

    @property
    def area(self) -> float:
        return self.width * self.height

    def expand(self, d: float) -> "Rect":
        return Rect(
            x_min=self.x_min - d,
            x_max=self.x_max + d,
            y_min=self.y_min - d,
            y_max=self.y_max + d,
        )

    def intersect(self, other: "Rect") -> Optional["Rect"]:
        x_min = max(self.x_min, other.x_min)
        x_max = min(self.x_max, other.x_max)
        y_min = max(self.y_min, other.y_min)
        y_max = min(self.y_max, other.y_max)
        if x_max <= x_min or y_max <= y_min:
            return None
        return Rect(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)

    def contains_point(self, x: float, y: float) -> bool:
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max


@dataclass
class SceneObject:
    object_id: str
    name: str
    category: str
    rect: Rect
    position_m: tuple[float, float, float]
    size_m: tuple[float, float, float]
    yaw_deg: float
    mount_type: str
    constraints: dict[str, Any]
    raw: dict[str, Any]

    @property
    def floor_area(self) -> float:
        return self.rect.area

    @property
    def is_floor_relevant(self) -> bool:
        return self.mount_type not in {"ceiling", "wall", "wall-mounted"}


@dataclass
class RoomInfo:
    polygon: list[tuple[float, float]]
    bounds: Rect
    area_m2: float


# ------------------------------------------------------------
# Generic helpers
# ------------------------------------------------------------


PROMPT_ALIASES: dict[str, list[str]] = {
    "bed": ["bed", "double bed", "king bed", "king-size bed", "queen bed", "кровать"],
    "nightstand": ["nightstand", "bedside table", "bedside cabinet", "тумба", "прикроватная тумба"],
    "wardrobe": ["wardrobe", "closet", "cabinet", "шкаф"],
    "dresser": ["dresser", "chest of drawers", "комод"],
    "desk": ["desk", "office desk", "письменный стол", "стол"],
    "table": ["table", "coffee table", "dining table", "стол"],
    "chair": ["chair", "armchair", "lounge chair", "office chair", "стул", "кресло"],
    "sofa": ["sofa", "couch", "диван"],
    "lamp": ["lamp", "ceiling lamp", "floor lamp", "light", "lighting", "люстра", "лампа", "светильник"],
    "tv": ["tv", "television", "телевизор"],
    "shelf": ["shelf", "bookshelf", "полка", "стеллаж"],
    "mirror": ["mirror", "зеркало"],
    "dressing table": ["dressing table", "vanity", "туалетный столик"],
}

NUMBER_WORDS: dict[str, int] = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "один": 1,
    "одна": 1,
    "два": 2,
    "две": 2,
    "три": 3,
    "четыре": 4,
    "пять": 5,
    "шесть": 6,
    "семь": 7,
    "восемь": 8,
    "девять": 9,
    "десять": 10,
}

CLAUSE_SPLIT_RE = re.compile(r"[,;\n]+|\s+(?:and|plus|with|&|и|а\s+также)\s+", re.IGNORECASE)



def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    if abs(b) <= EPS:
        return default
    return a / b


def norm_text(s: str) -> str:
    s = s.lower().strip()
    s = s.replace("ё", "е")
    s = re.sub(r"[^a-z0-9а-я\s_-]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    return json.loads(p.read_text(encoding="utf-8"))


def polygon_area(poly: list[tuple[float, float]]) -> float:
    if len(poly) < 3:
        return 0.0
    s = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        s += x1 * y2 - x2 * y1
    return abs(s) * 0.5


def point_in_polygon(x: float, y: float, poly: list[tuple[float, float]]) -> bool:
    inside = False
    n = len(poly)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) + EPS) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def extract_room(scene: dict[str, Any]) -> RoomInfo:
    room = scene.get("room") or {}
    poly_raw = room.get("floor_polygon") or []
    poly: list[tuple[float, float]] = []
    for p in poly_raw:
        if isinstance(p, dict):
            poly.append((float(p.get("x", 0.0)), float(p.get("y", 0.0))))

    if len(poly) < 3:
        width = float(room.get("width_m", 0.0))
        depth = float(room.get("depth_m", 0.0))
        poly = [(0.0, 0.0), (width, 0.0), (width, depth), (0.0, depth)]

    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    bounds = Rect(min(xs), max(xs), min(ys), max(ys))
    area = float(room.get("area_m2") or 0.0)
    if area <= EPS:
        area = polygon_area(poly)

    return RoomInfo(polygon=poly, bounds=bounds, area_m2=area)


def _object_mount_type(obj: dict[str, Any]) -> str:
    mount_type = obj.get("mount_type")
    if isinstance(mount_type, str) and mount_type.strip():
        return mount_type.strip().lower()

    constraints = obj.get("constraints") or {}
    c_mount = constraints.get("mount_type")
    if isinstance(c_mount, str) and c_mount.strip():
        return c_mount.strip().lower()

    return "floor"


def _object_rect(obj: dict[str, Any]) -> Rect:
    aabb = obj.get("aabb") or {}
    if aabb:
        return Rect(
            x_min=float(aabb.get("x_min", 0.0)),
            x_max=float(aabb.get("x_max", 0.0)),
            y_min=float(aabb.get("y_min", 0.0)),
            y_max=float(aabb.get("y_max", 0.0)),
        )

    pos = obj.get("position_m") or [0.0, 0.0, 0.0]
    size = obj.get("size_m") or [0.0, 0.0, 0.0]
    cx = float(pos[0])
    cy = float(pos[1])
    sx = float(size[0])
    sy = float(size[1])
    return Rect(cx - sx * 0.5, cx + sx * 0.5, cy - sy * 0.5, cy + sy * 0.5)


def extract_scene_objects(scene: dict[str, Any]) -> list[SceneObject]:
    out: list[SceneObject] = []
    for i, obj in enumerate(scene.get("placements") or []):
        name = str(obj.get("name") or obj.get("category") or f"obj_{i}")
        category = str(obj.get("category") or name)
        pos_raw = obj.get("position_m") or [0.0, 0.0, 0.0]
        size_raw = obj.get("size_m") or [0.0, 0.0, 0.0]
        pos = (
            float(pos_raw[0]) if len(pos_raw) > 0 else 0.0,
            float(pos_raw[1]) if len(pos_raw) > 1 else 0.0,
            float(pos_raw[2]) if len(pos_raw) > 2 else 0.0,
        )
        size = (
            float(size_raw[0]) if len(size_raw) > 0 else 0.0,
            float(size_raw[1]) if len(size_raw) > 1 else 0.0,
            float(size_raw[2]) if len(size_raw) > 2 else 0.0,
        )
        yaw_deg = float(obj.get("yaw_deg", obj.get("rotation_deg", 0.0)))
        out.append(
            SceneObject(
                object_id=str(obj.get("id") or f"obj_{i:04d}"),
                name=name,
                category=category,
                rect=_object_rect(obj),
                position_m=pos,
                size_m=size,
                yaw_deg=yaw_deg,
                mount_type=_object_mount_type(obj),
                constraints=dict(obj.get("constraints") or {}),
                raw=obj,
            )
        )
    return out


# ------------------------------------------------------------
# Grid / occupancy
# ------------------------------------------------------------


@dataclass
class GridInfo:
    step_m: float
    x_coords: list[float]
    y_coords: list[float]
    walkable: list[list[bool]]
    inside_room: list[list[bool]]
    blocked_by_objects: list[list[bool]]

    @property
    def rows(self) -> int:
        return len(self.y_coords)

    @property
    def cols(self) -> int:
        return len(self.x_coords)


def build_grid(
    room: RoomInfo,
    objects: list[SceneObject],
    step_m: float = 0.10,
    clearance_m: float = 0.20,
) -> GridInfo:
    step_m = max(0.04, float(step_m))
    cols = max(1, int(math.ceil(room.bounds.width / step_m)))
    rows = max(1, int(math.ceil(room.bounds.height / step_m)))

    x_coords = [room.bounds.x_min + (c + 0.5) * step_m for c in range(cols)]
    y_coords = [room.bounds.y_min + (r + 0.5) * step_m for r in range(rows)]

    inside_room = [[False] * cols for _ in range(rows)]
    blocked = [[False] * cols for _ in range(rows)]

    floor_rects = [o.rect.expand(clearance_m) for o in objects if o.is_floor_relevant]

    for r in range(rows):
        y = y_coords[r]
        for c in range(cols):
            x = x_coords[c]
            in_room = point_in_polygon(x, y, room.polygon)
            inside_room[r][c] = in_room
            if not in_room:
                continue
            for rect in floor_rects:
                if rect.contains_point(x, y):
                    blocked[r][c] = True
                    break

    walkable = [[inside_room[r][c] and not blocked[r][c] for c in range(cols)] for r in range(rows)]
    return GridInfo(
        step_m=step_m,
        x_coords=x_coords,
        y_coords=y_coords,
        walkable=walkable,
        inside_room=inside_room,
        blocked_by_objects=blocked,
    )


def union_occupied_area(grid: GridInfo) -> float:
    occ = 0
    for r in range(grid.rows):
        for c in range(grid.cols):
            if grid.inside_room[r][c] and grid.blocked_by_objects[r][c]:
                occ += 1
    return occ * (grid.step_m ** 2)


def largest_free_rectangle_area(grid: GridInfo) -> float:
    cols = grid.cols
    heights = [0] * cols
    best = 0.0
    cell_area = grid.step_m ** 2

    for r in range(grid.rows):
        for c in range(cols):
            if grid.walkable[r][c]:
                heights[c] += 1
            else:
                heights[c] = 0

        stack: list[int] = []
        i = 0
        while i <= cols:
            cur_h = heights[i] if i < cols else 0
            if not stack or cur_h >= heights[stack[-1]]:
                stack.append(i)
                i += 1
            else:
                top = stack.pop()
                h = heights[top]
                left = stack[-1] + 1 if stack else 0
                width = i - left
                best = max(best, h * width * cell_area)

    return best


# ------------------------------------------------------------
# A* accessibility
# ------------------------------------------------------------


def neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def heuristic_rc(a: tuple[int, int], b: tuple[int, int]) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar_to_any_target(
    grid: GridInfo,
    start: tuple[int, int],
    targets: set[tuple[int, int]],
) -> bool:
    if not targets:
        return False
    if start in targets:
        return True

    open_heap: list[tuple[float, float, tuple[int, int]]] = []
    heapq.heappush(open_heap, (0.0, 0.0, start))
    g_score: dict[tuple[int, int], float] = {start: 0.0}
    closed: set[tuple[int, int]] = set()

    target_list = list(targets)

    while open_heap:
        _, g_cur, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        if cur in targets:
            return True
        closed.add(cur)

        for nr, nc in neighbors4(*cur):
            if nr < 0 or nr >= grid.rows or nc < 0 or nc >= grid.cols:
                continue
            if not grid.walkable[nr][nc]:
                continue
            nxt = (nr, nc)
            tentative = g_cur + 1.0
            if tentative + EPS >= g_score.get(nxt, float("inf")):
                continue
            g_score[nxt] = tentative
            h = min(heuristic_rc(nxt, tgt) for tgt in target_list)
            heapq.heappush(open_heap, (tentative + h, tentative, nxt))

    return False


def _nearest_walkable_cell(grid: GridInfo, pref: tuple[int, int]) -> Optional[tuple[int, int]]:
    pr, pc = pref
    best = None
    best_d = float("inf")
    for r in range(grid.rows):
        for c in range(grid.cols):
            if not grid.walkable[r][c]:
                continue
            d = abs(r - pr) + abs(c - pc)
            if d < best_d:
                best_d = d
                best = (r, c)
    return best


def choose_start_cell(room: RoomInfo, grid: GridInfo, scene: dict[str, Any]) -> Optional[tuple[int, int]]:
    doors = (scene.get("room") or {}).get("doors") or []
    if doors:
        for door in doors:
            if isinstance(door, dict):
                dx = float(door.get("x", (room.bounds.x_min + room.bounds.x_max) * 0.5))
                dy = float(door.get("y", room.bounds.y_min))
                rr = int((dy - room.bounds.y_min) / grid.step_m)
                cc = int((dx - room.bounds.x_min) / grid.step_m)
                cell = _nearest_walkable_cell(grid, (rr, cc))
                if cell is not None:
                    return cell

    cx = (room.bounds.x_min + room.bounds.x_max) * 0.5
    cy = (room.bounds.y_min + room.bounds.y_max) * 0.5
    rr = int((cy - room.bounds.y_min) / grid.step_m)
    cc = int((cx - room.bounds.x_min) / grid.step_m)
    cell = _nearest_walkable_cell(grid, (rr, cc))
    if cell is not None:
        return cell

    for r in range(grid.rows):
        for c in range(grid.cols):
            if grid.walkable[r][c]:
                return (r, c)
    return None


def target_cells_for_object(
    grid: GridInfo,
    room: RoomInfo,
    obj: SceneObject,
    approach_margin_m: float = 0.35,
) -> set[tuple[int, int]]:
    if not obj.is_floor_relevant:
        return set()

    expanded = obj.rect.expand(approach_margin_m)
    targets: set[tuple[int, int]] = set()

    for r in range(grid.rows):
        y = grid.y_coords[r]
        for c in range(grid.cols):
            x = grid.x_coords[c]
            if not grid.walkable[r][c]:
                continue
            if not point_in_polygon(x, y, room.polygon):
                continue
            in_expanded = expanded.contains_point(x, y)
            in_object = obj.rect.contains_point(x, y)
            if in_expanded and not in_object:
                targets.add((r, c))
    return targets


def compute_accessibility(
    scene: dict[str, Any],
    room: RoomInfo,
    objects: list[SceneObject],
    grid: GridInfo,
) -> tuple[list[AccessResult], float]:
    start = choose_start_cell(room, grid, scene)
    results: list[AccessResult] = []

    if start is None:
        for obj in objects:
            if not obj.is_floor_relevant:
                continue
            results.append(
                AccessResult(
                    object_id=obj.object_id,
                    name=obj.name,
                    category=obj.category,
                    reachable=False,
                    reason="no_walkable_start_cell",
                    reachable_target_count=0,
                    total_target_count=0,
                )
            )
        return results, 0.0

    eligible = [o for o in objects if o.is_floor_relevant]
    if not eligible:
        return [], 1.0

    reachable_count = 0
    for obj in eligible:
        targets = target_cells_for_object(grid, room, obj)
        if not targets:
            results.append(
                AccessResult(
                    object_id=obj.object_id,
                    name=obj.name,
                    category=obj.category,
                    reachable=False,
                    reason="no_free_approach_cells",
                    reachable_target_count=0,
                    total_target_count=0,
                )
            )
            continue

        ok = astar_to_any_target(grid, start, targets)
        if ok:
            reachable_count += 1
        results.append(
            AccessResult(
                object_id=obj.object_id,
                name=obj.name,
                category=obj.category,
                reachable=ok,
                reason="ok" if ok else "astar_failed",
                reachable_target_count=1 if ok else 0,
                total_target_count=len(targets),
            )
        )

    ratio = safe_div(reachable_count, len(eligible), default=0.0)
    return results, ratio


# ------------------------------------------------------------
# Prompt matching
# ------------------------------------------------------------


def canonical_category_name(s: str) -> str:
    s_n = norm_text(s)
    for canon, aliases in PROMPT_ALIASES.items():
        all_names = [canon] + aliases
        for a in all_names:
            if norm_text(a) == s_n:
                return canon
    for canon, aliases in PROMPT_ALIASES.items():
        if any(norm_text(a) in s_n or s_n in norm_text(a) for a in aliases + [canon]):
            return canon
    return s_n


def _split_prompt_into_clauses(prompt_text: str) -> list[str]:
    parts = [x.strip() for x in CLAUSE_SPLIT_RE.split(prompt_text) if x and x.strip()]
    return parts or [prompt_text.strip()]



def _count_near_alias_in_clause(clause_n: str, alias_n: str) -> int:
    idx = clause_n.find(alias_n)
    if idx < 0:
        return 0
    left_tokens = clause_n[:idx].split()
    for t in reversed(left_tokens[-3:]):
        if t.isdigit():
            return max(1, int(t))
        if t in NUMBER_WORDS:
            return max(1, NUMBER_WORDS[t])
    m = re.search(rf"{re.escape(alias_n)}\s*x\s*(\d+)", clause_n, flags=re.IGNORECASE)
    if m:
        return max(1, int(m.group(1)))
    return 1



def parse_requested_items_from_prompt(prompt_text: Optional[str]) -> dict[str, int]:
    if not prompt_text:
        return {}

    requested: dict[str, int] = {}

    required_block_re = re.compile(r"^\s*-\s*(.+?)\s*x\s*(\d+)\s*$", re.MULTILINE | re.IGNORECASE)
    matches = list(required_block_re.finditer(prompt_text))
    for m in matches:
        cat = canonical_category_name(m.group(1).strip())
        cnt = max(1, int(m.group(2)))
        requested[cat] = requested.get(cat, 0) + cnt
    if requested:
        return requested

    clauses = _split_prompt_into_clauses(prompt_text)
    alias_table: list[tuple[str, str]] = []
    for canon, aliases in PROMPT_ALIASES.items():
        names = sorted({norm_text(canon), *(norm_text(a) for a in aliases)}, key=len, reverse=True)
        for alias_n in names:
            alias_table.append((canon, alias_n))

    for clause in clauses:
        clause_n = norm_text(clause)
        matched_in_clause: set[str] = set()
        for canon, alias_n in alias_table:
            if canon in matched_in_clause:
                continue
            if alias_n in clause_n:
                cnt = _count_near_alias_in_clause(clause_n, alias_n)
                requested[canon] = requested.get(canon, 0) + max(1, cnt)
                matched_in_clause.add(canon)

    return requested


def infer_present_items(objects: list[SceneObject]) -> dict[str, int]:
    present: dict[str, int] = {}
    for obj in objects:
        canon = canonical_category_name(obj.category or obj.name)
        present[canon] = present.get(canon, 0) + 1
    return present


def count_f1_score(requested: dict[str, int], present: dict[str, int]) -> float:
    if not requested:
        return 1.0
    tp = 0
    req_total = 0
    pred_total = 0
    all_keys = set(requested) | set(present)
    for k in all_keys:
        r = requested.get(k, 0)
        p = present.get(k, 0)
        tp += min(r, p)
        req_total += r
        pred_total += p
    precision = safe_div(tp, pred_total, default=0.0)
    recall = safe_div(tp, req_total, default=0.0)
    if precision + recall <= EPS:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


# ------------------------------------------------------------
# Constraint checks
# ------------------------------------------------------------


def snapped_yaw_deg(yaw_deg: float) -> int:
    y = int(round(yaw_deg / 90.0)) % 4
    return y * 90



def _rot_ccw(vec: tuple[int, int], quarter_turns: int) -> tuple[int, int]:
    x, y = vec
    q = quarter_turns % 4
    for _ in range(q):
        x, y = -y, x
    return x, y



def _neg_vec(vec: tuple[int, int]) -> tuple[int, int]:
    return (-vec[0], -vec[1])



def _vec_to_world_side(vec: tuple[int, int]) -> str:
    mapping = {
        (1, 0): "x_max",
        (-1, 0): "x_min",
        (0, 1): "y_max",
        (0, -1): "y_min",
    }
    return mapping[vec]



def world_sides_touched(room: RoomInfo, obj: SceneObject, tol_m: float = 0.12) -> tuple[set[str], dict[str, float]]:
    rect = obj.rect
    distances = {
        "x_min": abs(rect.x_min - room.bounds.x_min),
        "x_max": abs(room.bounds.x_max - rect.x_max),
        "y_min": abs(rect.y_min - room.bounds.y_min),
        "y_max": abs(room.bounds.y_max - rect.y_max),
    }
    touched = {side for side, dist in distances.items() if dist <= tol_m}
    return touched, distances



def touched_room_sides_by_convention(room: RoomInfo, obj: SceneObject, tol_m: float = 0.12) -> tuple[dict[str, set[str]], set[str], dict[str, float]]:
    yaw_quarters = snapped_yaw_deg(obj.yaw_deg) // 90
    touched_world, distances = world_sides_touched(room, obj, tol_m=tol_m)

    conventions = [
        ("front_plus_y_ccw", (0, 1), 1),
        ("front_plus_y_cw", (0, 1), -1),
        ("front_plus_x_ccw", (1, 0), 1),
        ("front_plus_x_cw", (1, 0), -1),
    ]

    out: dict[str, set[str]] = {}
    for name, front0, yaw_sign in conventions:
        right0 = _rot_ccw(front0, -1)
        front_vec = _rot_ccw(front0, yaw_sign * yaw_quarters)
        right_vec = _rot_ccw(right0, yaw_sign * yaw_quarters)
        local_to_world = {
            "front": _vec_to_world_side(front_vec),
            "back": _vec_to_world_side(_neg_vec(front_vec)),
            "right": _vec_to_world_side(right_vec),
            "left": _vec_to_world_side(_neg_vec(right_vec)),
        }
        touched_local = {local for local, world in local_to_world.items() if world in touched_world}
        out[name] = touched_local

    return out, touched_world, distances



def compute_constraint_checks(room: RoomInfo, objects: list[SceneObject]) -> tuple[list[ConstraintCheck], float]:
    checks: list[ConstraintCheck] = []

    for obj in objects:
        constraints = obj.constraints or {}

        expected_mount = constraints.get("mount_type")
        if isinstance(expected_mount, str) and expected_mount.strip():
            ok = obj.mount_type == expected_mount.strip().lower()
            checks.append(
                ConstraintCheck(
                    object_id=obj.object_id,
                    name=obj.name,
                    category=obj.category,
                    constraint_type="mount_type",
                    expected=expected_mount,
                    satisfied=ok,
                    details={"actual_mount_type": obj.mount_type},
                )
            )

        touch_wall = constraints.get("touch_wall") or {}
        sides = touch_wall.get("sides") if isinstance(touch_wall, dict) else None
        if isinstance(sides, list) and sides:
            expected_sides = [str(x).strip().lower() for x in sides if str(x).strip()]
            mode = str((touch_wall or {}).get("mode") or "any").strip().lower()
            if mode not in {"any", "all"}:
                mode = "any"
            touched_by_convention, touched_world, wall_distances = touched_room_sides_by_convention(room, obj, tol_m=0.12)

            best_convention = None
            best_matched: list[str] = []
            best_touched: set[str] = set()
            ok = False
            for conv_name, touched_local in touched_by_convention.items():
                matched = [side for side in expected_sides if side in touched_local]
                conv_ok = all(side in touched_local for side in expected_sides) if mode == "all" else bool(matched)
                if conv_ok and not ok:
                    ok = True
                    best_convention = conv_name
                    best_matched = matched
                    best_touched = touched_local
                elif not ok and len(matched) > len(best_matched):
                    best_convention = conv_name
                    best_matched = matched
                    best_touched = touched_local

            checks.append(
                ConstraintCheck(
                    object_id=obj.object_id,
                    name=obj.name,
                    category=obj.category,
                    constraint_type="touch_wall",
                    expected={"sides": expected_sides, "mode": mode},
                    satisfied=ok,
                    details={
                        "matched_convention": best_convention,
                        "matched_expected_sides": best_matched,
                        "touched_local_sides": sorted(best_touched),
                        "touched_local_sides_by_convention": {k: sorted(v) for k, v in touched_by_convention.items()},
                        "touched_world_sides": sorted(touched_world),
                        "wall_distances_m": {k: round(v, 6) for k, v in wall_distances.items()},
                        "tolerance_m": 0.12,
                    },
                )
            )

    if not checks:
        return checks, 1.0

    ok_count = sum(1 for x in checks if x.satisfied)
    return checks, safe_div(ok_count, len(checks), default=1.0)



# ------------------------------------------------------------
# Core scoring
# ------------------------------------------------------------


def compute_overlap_area(objects: list[SceneObject]) -> float:
    floor_objects = [o for o in objects if o.is_floor_relevant]
    overlap = 0.0
    for i in range(len(floor_objects)):
        for j in range(i + 1, len(floor_objects)):
            inter = floor_objects[i].rect.intersect(floor_objects[j].rect)
            if inter is not None:
                overlap += inter.area
    return overlap


def compute_outside_area(room: RoomInfo, objects: list[SceneObject]) -> float:
    outside = 0.0
    room_rect = room.bounds
    for obj in objects:
        if not obj.is_floor_relevant:
            continue
        inter = obj.rect.intersect(room_rect)
        inside_area = 0.0 if inter is None else inter.area
        outside += max(0.0, obj.rect.area - inside_area)
    return outside


def geometry_score_from_penalties(overlap_ratio: float, outside_ratio: float) -> float:
    penalty = 0.60 * clamp(overlap_ratio, 0.0, 1.0) + 0.40 * clamp(outside_ratio, 0.0, 1.0)
    return 10.0 * (1.0 - penalty)


def free_space_score_from_ratio(ratio: float) -> float:
    ratio = clamp(ratio, 0.0, 1.0)
    return 10.0 * math.sqrt(ratio)


def appraise_scene_area(
    scene: dict[str, Any],
    prompt_text: Optional[str] = None,
    grid_step_m: float = 0.10,
    clearance_m: float = 0.20,
    approach_margin_m: float = 0.35,
) -> AreaAppraisalResult:
    room = extract_room(scene)
    objects = extract_scene_objects(scene)
    floor_objects = [o for o in objects if o.is_floor_relevant]

    raw_sum_item_area = sum(o.floor_area for o in floor_objects)
    free_area = max(0.0, room.area_m2 - raw_sum_item_area)
    floor_coverage_ratio = clamp(safe_div(raw_sum_item_area, room.area_m2, default=0.0), 0.0, 10.0)

    overlap_area = compute_overlap_area(objects)
    overlap_ratio = clamp(safe_div(overlap_area, room.area_m2, default=0.0), 0.0, 10.0)

    outside_area = compute_outside_area(room, objects)
    outside_ratio = clamp(safe_div(outside_area, room.area_m2, default=0.0), 0.0, 10.0)

    grid = build_grid(room, objects, step_m=grid_step_m, clearance_m=clearance_m)
    occupied_union = union_occupied_area(grid)
    free_union = max(0.0, room.area_m2 - occupied_union)
    max_rect = largest_free_rectangle_area(grid)
    max_rect_ratio = clamp(safe_div(max_rect, max(free_area, EPS), default=0.0), 0.0, 1.0)

    access_results, access_ratio = compute_accessibility(scene, room, objects, grid)
    for idx, obj in enumerate(floor_objects):
        _ = approach_margin_m  # совместимость интерфейса; margin используется внутри target_cells_for_object
        # Параметр оставлен для дальнейшей тонкой настройки.

    requested = parse_requested_items_from_prompt(prompt_text)
    present = infer_present_items(objects)
    prompt_f1 = count_f1_score(requested, present)
    prompt_score_10 = 10.0 * prompt_f1

    constraint_checks, constraint_ratio = compute_constraint_checks(room, objects)
    constraint_score_10 = 10.0 * constraint_ratio
    geometry_score_10 = geometry_score_from_penalties(overlap_ratio, outside_ratio)
    free_space_score_10 = free_space_score_from_ratio(max_rect_ratio)
    access_score_10 = 10.0 * access_ratio

    final_score = (
        0.28 * geometry_score_10
        + 0.24 * access_score_10
        + 0.18 * free_space_score_10
        + 0.15 * prompt_score_10
        + 0.15 * constraint_score_10
    )

    details = {
        "room_bounds": asdict(room.bounds),
        "grid": {
            "step_m": grid.step_m,
            "rows": grid.rows,
            "cols": grid.cols,
        },
        "requested_items": requested,
        "present_items": present,
        "accessibility": [asdict(x) for x in access_results],
        "constraint_checks": [asdict(x) for x in constraint_checks],
        "subscores": {
            "geometry_score_10": geometry_score_10,
            "access_score_10": access_score_10,
            "free_space_score_10": free_space_score_10,
            "prompt_match_score_10": prompt_score_10,
            "constraint_score_10": constraint_score_10,
        },
    }

    return AreaAppraisalResult(
        score_10=round(final_score, 4),
        room_area_m2=round(room.area_m2, 6),
        raw_sum_item_area_m2=round(raw_sum_item_area, 6),
        free_area_m2=round(free_area, 6),
        occupied_union_area_m2=round(occupied_union, 6),
        free_union_area_m2=round(free_union, 6),
        largest_free_rectangle_area_m2=round(max_rect, 6),
        largest_free_rectangle_ratio=round(max_rect_ratio, 6),
        floor_coverage_ratio=round(floor_coverage_ratio, 6),
        overlap_area_m2=round(overlap_area, 6),
        overlap_ratio=round(overlap_ratio, 6),
        outside_room_area_m2=round(outside_area, 6),
        outside_room_ratio=round(outside_ratio, 6),
        accessible_objects=sum(1 for x in access_results if x.reachable),
        accessible_objects_total=len([o for o in objects if o.is_floor_relevant]),
        accessibility_ratio=round(access_ratio, 6),
        prompt_match_score_10=round(prompt_score_10, 6),
        constraint_score_10=round(constraint_score_10, 6),
        geometry_score_10=round(geometry_score_10, 6),
        details=details,
    )


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Code-only appraiser for scene.v1")
    p.add_argument("--scene", required=True, help="Path to scene.v1 JSON")
    p.add_argument("--prompt", default=None, help="Original text prompt")
    p.add_argument("--prompt-file", default=None, help="Path to prompt text file")
    p.add_argument("--out", default=None, help="Where to save JSON result")
    p.add_argument("--grid-step", type=float, default=0.10, help="Grid step in meters")
    p.add_argument("--clearance", type=float, default=0.20, help="Object expansion for walking grid")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    scene = load_json(args.scene)

    prompt_text = args.prompt
    if args.prompt_file:
        prompt_text = Path(args.prompt_file).expanduser().resolve().read_text(encoding="utf-8")

    result = appraise_scene_area(
        scene=scene,
        prompt_text=prompt_text,
        grid_step_m=args.grid_step,
        clearance_m=args.clearance,
    )

    text = json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).expanduser().resolve().write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
