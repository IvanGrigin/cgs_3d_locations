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


def aabb_center(aabb: dict[str, Any]) -> list[float]:
    return [
        (float(aabb["x_min"]) + float(aabb["x_max"])) * 0.5,
        (float(aabb["y_min"]) + float(aabb["y_max"])) * 0.5,
        (float(aabb["z_min"]) + float(aabb["z_max"])) * 0.5,
    ]


def aabb_size(aabb: dict[str, Any]) -> list[float]:
    return [
        float(aabb["x_max"]) - float(aabb["x_min"]),
        float(aabb["y_max"]) - float(aabb["y_min"]),
        float(aabb["z_max"]) - float(aabb["z_min"]),
    ]


def aabb_item(item_id: str, category: str, aabb: dict[str, Any], **extra: Any) -> dict[str, Any]:
    payload = {
        "id": item_id,
        "name": category,
        "category": category,
        "position_m": aabb_center(aabb),
        "size_m": aabb_size(aabb),
        "aabb": dict(aabb),
    }
    payload.update(deepcopy(extra))
    return payload


def centered_item(
    item_id: str,
    category: str,
    x: float,
    y: float,
    z: float,
    *,
    sx: float = 0.5,
    sy: float = 0.5,
    sz: float = 0.5,
    **extra: Any,
) -> dict[str, Any]:
    return aabb_item(
        item_id,
        category,
        box(x - sx / 2, x + sx / 2, y - sy / 2, y + sy / 2, z - sz / 2, z + sz / 2),
        **extra,
    )


def supplier_candidate(path: Path, *, unique_key: str = "cand", fmt: str | None = None, group: str = "chair", **updates: Any) -> dict[str, Any]:
    payload = {
        "unique_key": unique_key,
        "title": f"{group} candidate",
        "semantic_group": group,
        "category_norm": group,
        "asset_local_path": str(path),
        "asset_format": fmt or path.suffix.lstrip("."),
        "asset_status": "local_supplier_asset",
        "width_cm": 50,
        "depth_cm": 60,
        "height_cm": 90,
        "source_site": "test",
        "product_url": "https://example.test/p",
    }
    payload.update(deepcopy(updates))
    return payload


def supplier_binding(candidate: dict[str, Any], *, target_id: str = "target", group: str = "chair", **updates: Any) -> dict[str, Any]:
    payload = {
        "target_id": target_id,
        "semantic_group": group,
        "selection_status": "heuristic_top1_selected",
        "provenance": {"final_asset_source": "supplier_catalog"},
        "chosen_candidate": candidate,
        "top_candidates": [candidate, dict(candidate, unique_key="dup")],
    }
    payload.update(deepcopy(updates))
    return payload


def placement_room_scene(placements: list[dict[str, Any]], *, width: float = 6.0, depth: float = 6.0, room_type: str = "livingroom") -> dict[str, Any]:
    return {
        "schema": "scene.v1",
        "room": {
            "type": room_type,
            "floor_polygon": [{"x": 0.0, "y": 0.0}, {"x": width, "y": 0.0}, {"x": width, "y": depth}, {"x": 0.0, "y": depth}],
            "floor_z": 0.0,
            "ceiling_height": 3.0,
        },
        "placements": deepcopy(placements),
    }


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
