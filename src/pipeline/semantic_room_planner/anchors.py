from __future__ import annotations

from typing import Any


COMMON_TABLE = ["top.center", "top.left_front", "top.right_front", "top.back_left", "top.back_right", "top.front_center", "top.back_center", "front.center", "chair.front", "chair.left", "chair.right", "chair.back"]


def generate_anchors(objects: list[dict[str, Any]]) -> dict[str, Any]:
    anchors: dict[str, Any] = {"schema": "object_anchors/v1", "objects": {}}
    for obj in objects:
        sc = obj.get("subclass")
        names: list[str]
        if sc in {"desk", "dining_table", "kitchen_table", "nightstand", "coffee_table", "side_table", "tv_stand"}:
            names = COMMON_TABLE
        elif sc == "bed":
            names = ["top.pillow_left", "top.pillow_right", "top.blanket_center", "left_side.nightstand", "right_side.nightstand", "foot.rug", "front.center"]
        elif sc == "sofa":
            names = ["front.coffee_table", "left_side.side_table", "right_side.side_table", "left_side.floor_lamp", "right_side.floor_lamp", "front.center"]
        elif sc in {"dresser", "shelf", "bookcase", "wardrobe", "sink"}:
            names = ["top.center", "top.left_front", "top.right_front", "front.center"]
        else:
            names = ["center"]
        anchors["objects"][obj["id"]] = {"local_frame": "bbox_yaw", "anchors": {name: {"name": name} for name in names}}
    return anchors
