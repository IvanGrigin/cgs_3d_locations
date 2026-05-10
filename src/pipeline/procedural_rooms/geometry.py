from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence


EPS = 1e-9


@dataclass(frozen=True)
class Vec2:
    x: float
    y: float

    def __add__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, value: float) -> "Vec2":
        return Vec2(self.x * value, self.y * value)

    def dot(self, other: "Vec2") -> float:
        return self.x * other.x + self.y * other.y

    def cross(self, other: "Vec2") -> float:
        return self.x * other.y - self.y * other.x

    def length(self) -> float:
        return math.hypot(self.x, self.y)

    def normalized(self) -> "Vec2":
        length = self.length()
        if length < EPS:
            return Vec2(1.0, 0.0)
        return Vec2(self.x / length, self.y / length)

    def perpendicular_left(self) -> "Vec2":
        return Vec2(-self.y, self.x)

    def perpendicular_right(self) -> "Vec2":
        return Vec2(self.y, -self.x)

    def as_list(self) -> list[float]:
        return [self.x, self.y]


@dataclass(frozen=True)
class AABB:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    def to_json(self) -> dict[str, float]:
        return {
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
            "z_min": self.z_min,
            "z_max": self.z_max,
        }

    def intersects_xy(self, other: "AABB", margin: float = 0.0) -> bool:
        return not (
            self.x_max + margin <= other.x_min
            or other.x_max + margin <= self.x_min
            or self.y_max + margin <= other.y_min
            or other.y_max + margin <= self.y_min
        )

    def intersects_3d(self, other: "AABB", margin: float = 0.0) -> bool:
        if not self.intersects_xy(other, margin=margin):
            return False
        return not (
            self.z_max + margin <= other.z_min
            or other.z_max + margin <= self.z_min
        )


@dataclass(frozen=True)
class WallSegment:
    id: str
    start: Vec2
    end: Vec2
    index: int
    has_door: bool = False
    has_window: bool = False

    @property
    def vector(self) -> Vec2:
        return self.end - self.start

    @property
    def length(self) -> float:
        return self.vector.length()

    @property
    def tangent(self) -> Vec2:
        return self.vector.normalized()

    def point_at(self, distance_from_start: float) -> Vec2:
        d = clamp(distance_from_start, 0.0, self.length)
        return self.start + self.tangent * d


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def point_from_mapping(data: Any) -> Vec2:
    if isinstance(data, dict):
        return Vec2(as_float(data.get("x")), as_float(data.get("y", data.get("z", 0.0))))
    if isinstance(data, Sequence) and len(data) >= 2:
        return Vec2(as_float(data[0]), as_float(data[1]))
    return Vec2(0.0, 0.0)


def polygon_from_json(points: Iterable[Any]) -> list[Vec2]:
    result = [point_from_mapping(p) for p in points]
    if len(result) >= 2 and distance(result[0], result[-1]) < EPS:
        result = result[:-1]
    return result


def polygon_area(points: Sequence[Vec2]) -> float:
    if len(points) < 3:
        return 0.0
    acc = 0.0
    for i, p in enumerate(points):
        q = points[(i + 1) % len(points)]
        acc += p.x * q.y - q.x * p.y
    return abs(acc) * 0.5


def signed_polygon_area(points: Sequence[Vec2]) -> float:
    if len(points) < 3:
        return 0.0
    acc = 0.0
    for i, p in enumerate(points):
        q = points[(i + 1) % len(points)]
        acc += p.x * q.y - q.x * p.y
    return acc * 0.5


def polygon_centroid(points: Sequence[Vec2]) -> Vec2:
    if not points:
        return Vec2(0.0, 0.0)
    area2 = 0.0
    cx = 0.0
    cy = 0.0
    for i, p in enumerate(points):
        q = points[(i + 1) % len(points)]
        cross = p.x * q.y - q.x * p.y
        area2 += cross
        cx += (p.x + q.x) * cross
        cy += (p.y + q.y) * cross
    if abs(area2) < EPS:
        return Vec2(sum(p.x for p in points) / len(points), sum(p.y for p in points) / len(points))
    return Vec2(cx / (3.0 * area2), cy / (3.0 * area2))


def polygon_bounds(points: Sequence[Vec2]) -> tuple[float, float, float, float]:
    if not points:
        return 0.0, 0.0, 0.0, 0.0
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def distance(a: Vec2, b: Vec2) -> float:
    return (a - b).length()


def point_in_polygon(point: Vec2, polygon: Sequence[Vec2]) -> bool:
    if len(polygon) < 3:
        return False
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        pi = polygon[i]
        pj = polygon[j]
        if ((pi.y > point.y) != (pj.y > point.y)) and (
            point.x < (pj.x - pi.x) * (point.y - pi.y) / (pj.y - pi.y + EPS) + pi.x
        ):
            inside = not inside
        j = i
    return inside


def rectangle_corners(center: Vec2, width: float, depth: float, yaw_deg: float) -> list[Vec2]:
    yaw = math.radians(yaw_deg)
    ux = Vec2(math.cos(yaw), math.sin(yaw))
    uy = Vec2(-math.sin(yaw), math.cos(yaw))
    hw = width * 0.5
    hd = depth * 0.5
    return [
        center + ux * (-hw) + uy * (-hd),
        center + ux * (hw) + uy * (-hd),
        center + ux * (hw) + uy * (hd),
        center + ux * (-hw) + uy * (hd),
    ]


def aabb_from_box(center_xyz: Sequence[float], size_xyz: Sequence[float], yaw_deg: float) -> AABB:
    x = as_float(center_xyz[0])
    y = as_float(center_xyz[1])
    z = as_float(center_xyz[2])
    sx = max(as_float(size_xyz[0]), 0.001)
    sy = max(as_float(size_xyz[1]), 0.001)
    sz = max(as_float(size_xyz[2]), 0.001)
    corners = rectangle_corners(Vec2(x, y), sx, sy, yaw_deg)
    xs = [p.x for p in corners]
    ys = [p.y for p in corners]
    return AABB(
        x_min=min(xs),
        x_max=max(xs),
        y_min=min(ys),
        y_max=max(ys),
        z_min=z - sz * 0.5,
        z_max=z + sz * 0.5,
    )


def yaw_for_local_y_to_vector(vector: Vec2) -> float:
    """Return yaw where object's local +Y axis points along the given vector."""
    n = vector.normalized()
    return math.degrees(math.atan2(-n.x, n.y))


def local_axes_from_yaw(yaw_deg: float) -> tuple[Vec2, Vec2]:
    yaw = math.radians(yaw_deg)
    local_x = Vec2(math.cos(yaw), math.sin(yaw))
    local_y = Vec2(-math.sin(yaw), math.cos(yaw))
    return local_x, local_y


def normalize_angle_deg(angle: float) -> float:
    result = angle % 360.0
    if result < 0:
        result += 360.0
    return result


def wall_inside_normal(wall: WallSegment, polygon: Sequence[Vec2]) -> Vec2:
    tangent = wall.tangent
    midpoint = wall.point_at(wall.length * 0.5)
    candidates = [tangent.perpendicular_left(), tangent.perpendicular_right()]
    for n in candidates:
        probe = midpoint + n.normalized() * 0.08
        if point_in_polygon(probe, polygon):
            return n.normalized()
    centroid = polygon_centroid(polygon)
    toward_centroid = (centroid - midpoint).normalized()
    if toward_centroid.length() < EPS:
        return candidates[0].normalized()
    return toward_centroid


def build_wall_segments(
    polygon: Sequence[Vec2],
    walls_json: Sequence[dict[str, Any]] | None = None,
    doors: Sequence[dict[str, Any]] | None = None,
    windows: Sequence[dict[str, Any]] | None = None,
) -> list[WallSegment]:
    door_wall_ids = {str(d.get("wall_id")) for d in (doors or []) if d.get("wall_id") is not None}
    window_wall_ids = {str(w.get("wall_id")) for w in (windows or []) if w.get("wall_id") is not None}

    result: list[WallSegment] = []
    if walls_json:
        for idx, wall in enumerate(walls_json):
            try:
                a = int(wall.get("from_vertex", idx))
                b = int(wall.get("to_vertex", (idx + 1) % len(polygon)))
            except (TypeError, ValueError):
                a, b = idx, (idx + 1) % len(polygon)
            if not polygon or a < 0 or b < 0 or a >= len(polygon) or b >= len(polygon):
                continue
            wall_id = str(wall.get("id", f"w{idx}"))
            result.append(
                WallSegment(
                    id=wall_id,
                    start=polygon[a],
                    end=polygon[b],
                    index=idx,
                    has_door=wall_id in door_wall_ids,
                    has_window=wall_id in window_wall_ids,
                )
            )

    if not result and polygon:
        for idx, p in enumerate(polygon):
            q = polygon[(idx + 1) % len(polygon)]
            wall_id = f"w{idx}"
            result.append(
                WallSegment(
                    id=wall_id,
                    start=p,
                    end=q,
                    index=idx,
                    has_door=wall_id in door_wall_ids,
                    has_window=wall_id in window_wall_ids,
                )
            )
    return [w for w in result if w.length > 0.05]


def opening_clearance_aabb(
    wall: WallSegment,
    opening: dict[str, Any],
    polygon: Sequence[Vec2],
    clearance_depth: float,
    clearance_side: float = 0.25,
) -> AABB | None:
    width = as_float(opening.get("width"), 0.9)
    s = as_float(opening.get("s"), wall.length * 0.5)
    center_s = clamp(s + width * 0.5, 0.0, wall.length)
    tangent = wall.tangent
    normal = wall_inside_normal(wall, polygon)
    center = wall.point_at(center_s) + normal * (clearance_depth * 0.5)
    size_x = width + 2.0 * clearance_side
    size_y = clearance_depth
    yaw = yaw_for_local_y_to_vector(normal)
    return aabb_from_box([center.x, center.y, 1.0], [size_x, size_y, 2.0], yaw)


def object_footprint_inside_polygon(
    center: Vec2,
    size_xy: Sequence[float],
    yaw_deg: float,
    polygon: Sequence[Vec2],
    tolerance: float = 0.02,
) -> bool:
    corners = rectangle_corners(center, as_float(size_xy[0]), as_float(size_xy[1]), yaw_deg)
    if not polygon:
        return True
    for p in corners:
        if not point_in_polygon(p, polygon):
            # A small tolerance helps with objects intentionally touching walls.
            moved = p + (polygon_centroid(polygon) - p).normalized() * tolerance
            if not point_in_polygon(moved, polygon):
                return False
    return True


def choose_longest_wall(
    walls: Sequence[WallSegment],
    *,
    avoid_windows: bool = False,
    avoid_doors: bool = True,
    min_length: float = 0.0,
) -> WallSegment | None:
    candidates = []
    for wall in walls:
        if wall.length < min_length:
            continue
        if avoid_doors and wall.has_door:
            continue
        if avoid_windows and wall.has_window:
            continue
        candidates.append(wall)
    if not candidates:
        candidates = [w for w in walls if w.length >= min_length]
    if not candidates:
        return None
    return max(candidates, key=lambda w: w.length)


def choose_wall_most_opposite(
    walls: Sequence[WallSegment],
    source_wall: WallSegment,
    polygon: Sequence[Vec2],
    *,
    avoid_windows: bool = False,
    avoid_doors: bool = True,
) -> WallSegment | None:
    source_n = wall_inside_normal(source_wall, polygon)
    candidates: list[tuple[float, WallSegment]] = []
    for wall in walls:
        if wall.id == source_wall.id:
            continue
        if avoid_doors and wall.has_door:
            continue
        if avoid_windows and wall.has_window:
            continue
        n = wall_inside_normal(wall, polygon)
        opposite_score = -source_n.dot(n)
        candidates.append((opposite_score * 10.0 + wall.length, wall))
    if not candidates:
        return choose_longest_wall(walls, avoid_windows=avoid_windows, avoid_doors=avoid_doors)
    return max(candidates, key=lambda item: item[0])[1]


def nearest_corner_candidates(polygon: Sequence[Vec2], inset: float = 0.45) -> list[Vec2]:
    if not polygon:
        return []
    centroid = polygon_centroid(polygon)
    result = []
    for p in polygon:
        direction = (centroid - p).normalized()
        result.append(p + direction * inset)
    return result
