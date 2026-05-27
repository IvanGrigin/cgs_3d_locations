from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def rectangular_floor(width: float = 4.0, depth: float = 3.0) -> list[list[float]]:
    return [[0, 0], [width, 0], [width, depth], [0, depth]]


def box(x_min: float = 0.0, x_max: float = 1.0, y_min: float = 0.0, y_max: float = 1.0, z_min: float = 0.0, z_max: float = 1.0) -> dict[str, float]:
    return {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max, "z_min": z_min, "z_max": z_max}


def default_room(width: float = 4.0, depth: float = 3.0, **updates: Any) -> dict[str, Any]:
    room = {
        "id": "room_1",
        "type": "bedroom",
        "room_type": "bedroom",
        "floor_polygon": rectangular_floor(width, depth),
        "ceiling_height_m": 3.0,
    }
    room.update(updates)
    return room


@dataclass
class SceneBuilder:
    room_data: dict[str, Any] = field(default_factory=default_room)
    items_data: list[dict[str, Any]] = field(default_factory=list)
    items_key: str = "items"

    def room(self, **updates: Any) -> "SceneBuilder":
        self.room_data.update(updates)
        return self

    def item(
        self,
        item_id: str,
        category: str = "chair",
        *,
        name: str | None = None,
        group: str | None = None,
        aabb: dict[str, Any] | None = None,
        asset: dict[str, Any] | None = None,
        source: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
        position: dict[str, Any] | None = None,
        position_m: list[float] | None = None,
        size_m: list[float] | None = None,
        rotation_y: float | None = None,
        **extra: Any,
    ) -> "SceneBuilder":
        item_aabb = deepcopy(aabb) if aabb is not None else box()
        payload: dict[str, Any] = {
            "id": item_id,
            "name": name or item_id,
            "category": category,
            "group": group or category,
            "aabb": item_aabb,
            "position_m": position_m or [(item_aabb["x_min"] + item_aabb["x_max"]) / 2.0, (item_aabb["y_min"] + item_aabb["y_max"]) / 2.0, (item_aabb["z_min"] + item_aabb["z_max"]) / 2.0],
            "size_m": size_m or [item_aabb["x_max"] - item_aabb["x_min"], item_aabb["y_max"] - item_aabb["y_min"], item_aabb["z_max"] - item_aabb["z_min"]],
        }
        if asset is not None:
            payload["asset"] = deepcopy(asset)
        if source is not None:
            payload["source"] = deepcopy(source)
        if meta is not None:
            payload["meta"] = deepcopy(meta)
        if position is not None:
            payload["position"] = deepcopy(position)
        if rotation_y is not None:
            payload["rotation_y"] = rotation_y
        payload.update(deepcopy(extra))
        self.items_data.append(payload)
        return self

    def placement(self, item_id: str, category: str = "chair", **kwargs: Any) -> "SceneBuilder":
        return self.item(item_id, category, **kwargs)

    def build(self) -> dict[str, Any]:
        return {"room": deepcopy(self.room_data), self.items_key: deepcopy(self.items_data)}

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.build(), ensure_ascii=False), encoding="utf-8")
        return path


def scene_with_room(**room_updates: Any) -> SceneBuilder:
    return SceneBuilder().room(**room_updates)


def placement_scene_with_room(**room_updates: Any) -> SceneBuilder:
    return SceneBuilder(items_key="placements").room(**room_updates)
