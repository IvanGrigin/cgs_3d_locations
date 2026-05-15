from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .geometry import (
    AABB,
    Vec2,
    as_float,
    build_wall_segments,
    opening_clearance_aabb,
    polygon_area,
    polygon_bounds,
    polygon_centroid,
    polygon_from_json,
)


@dataclass
class RoomContext:
    raw_scene: dict[str, Any]
    room: dict[str, Any]
    polygon: list[Vec2]
    walls: list[Any]
    doors: list[dict[str, Any]]
    windows: list[dict[str, Any]]
    room_id: str
    room_type: str
    area_m2: float
    ceiling_height_m: float
    bounds: tuple[float, float, float, float]
    centroid: Vec2
    door_clearance_zones: list[AABB] = field(default_factory=list)
    window_clearance_zones: list[AABB] = field(default_factory=list)

    @property
    def width_m(self) -> float:
        return max(0.0, self.bounds[2] - self.bounds[0])

    @property
    def depth_m(self) -> float:
        return max(0.0, self.bounds[3] - self.bounds[1])

    @property
    def min_side_m(self) -> float:
        return min(self.width_m, self.depth_m)

    @property
    def max_side_m(self) -> float:
        return max(self.width_m, self.depth_m)

    @property
    def size_class(self) -> str:
        area = self.area_m2
        if area < 6.0:
            return "tiny"
        if area < 10.0:
            return "small"
        if area < 18.0:
            return "medium"
        if area < 30.0:
            return "large"
        return "xlarge"

    @property
    def aspect_ratio(self) -> float:
        if self.min_side_m <= 0.001:
            return 1.0
        return self.max_side_m / self.min_side_m

    @property
    def is_long_narrow(self) -> bool:
        return self.aspect_ratio >= 2.1


def _extract_room(scene: dict[str, Any]) -> dict[str, Any]:
    room = scene.get("room")
    if isinstance(room, dict):
        return room
    nested = scene.get("data")
    if isinstance(nested, dict) and isinstance(nested.get("room"), dict):
        return nested["room"]
    return {}


def normalize_room_type(room_type: Any, prompt: str = "", area_m2: float | None = None) -> str:
    raw = str(room_type or "").strip().lower()
    prompt_l = (prompt or "").lower()

    mapping = {
        "bedroom": "bedroom",
        "bed room": "bedroom",
        "спальня": "bedroom",
        "living_room": "living_room",
        "living room": "living_room",
        "гостиная": "living_room",
        "hall": "corridor",
        "corridor": "corridor",
        "коридор": "corridor",
        "прихожая": "corridor",
        "room": "room",
        "studio": "studio",
        "kitchen": "kitchen",
        "bathroom": "bathroom",
        "ванная": "bathroom",
        "ванная комната": "bathroom",
        "bath": "bathroom",
        "joint_bathroom": "bathroom",
        "санузел": "bathroom",
        "совмещенный санузел": "bathroom",
        "toilet": "toilet",
        "туалет": "toilet",
        "уборная": "toilet",
        "wc": "toilet",
    }
    if raw in mapping:
        normalized = mapping[raw]
    else:
        normalized = raw

    if normalized in {"room", "", "none", "null"}:
        if any(token in prompt_l for token in ["спаль", "кровать", "bedroom", "bed "]):
            return "bedroom"
        if any(token in prompt_l for token in ["гостин", "диван", "living", "sofa", "tv"]):
            return "living_room"
        if area_m2 is not None and area_m2 >= 16.0:
            return "living_room"
        return "bedroom"

    if normalized == "studio":
        if any(token in prompt_l for token in ["спаль", "кровать", "bedroom"]):
            return "bedroom"
        return "living_room"

    return normalized


def build_room_context(scene: dict[str, Any], prompt: str = "") -> RoomContext:
    room = _extract_room(scene)
    polygon = polygon_from_json(room.get("floor_polygon") or room.get("floor_polygon_xz") or [])
    if not polygon:
        width = as_float(room.get("width_m"), 3.5)
        depth = as_float(room.get("depth_m"), 4.0)
        polygon = [
            Vec2(0.0, 0.0),
            Vec2(width, 0.0),
            Vec2(width, depth),
            Vec2(0.0, depth),
        ]

    openings = room.get("openings") if isinstance(room.get("openings"), dict) else {}
    doors = list(room.get("doors") or openings.get("doors") or [])
    windows = list(room.get("windows") or openings.get("windows") or [])
    walls = build_wall_segments(polygon, room.get("walls") or [], doors=doors, windows=windows)
    area = as_float(room.get("area_m2"), polygon_area(polygon))
    if area <= 0.001:
        area = polygon_area(polygon)
    bounds = polygon_bounds(polygon)
    ceiling = as_float(
        room.get("ceiling_height_m", room.get("ceiling_height", room.get("height_m"))),
        2.8,
    )
    raw_type = room.get("room_type", room.get("type", room.get("type_hint", "room")))
    normalized_type = normalize_room_type(raw_type, prompt=prompt, area_m2=area)

    ctx = RoomContext(
        raw_scene=scene,
        room=room,
        polygon=polygon,
        walls=walls,
        doors=doors,
        windows=windows,
        room_id=str(room.get("id", "room")),
        room_type=normalized_type,
        area_m2=area,
        ceiling_height_m=ceiling,
        bounds=bounds,
        centroid=polygon_centroid(polygon),
    )

    wall_by_id = {wall.id: wall for wall in walls}
    for door in doors:
        wall_id = str(door.get("wall_id", ""))
        wall = wall_by_id.get(wall_id)
        if wall is None:
            continue
        zone = opening_clearance_aabb(wall, door, polygon, clearance_depth=1.0, clearance_side=0.2)
        if zone is not None:
            ctx.door_clearance_zones.append(zone)
    for window in windows:
        wall_id = str(window.get("wall_id", ""))
        wall = wall_by_id.get(wall_id)
        if wall is None:
            continue
        zone = opening_clearance_aabb(wall, window, polygon, clearance_depth=0.45, clearance_side=0.15)
        if zone is not None:
            ctx.window_clearance_zones.append(zone)
    return ctx
