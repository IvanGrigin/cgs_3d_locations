from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from src.ChooseObject.floor_material_selector import FloorMaterialSelector


def run_flooring_selection(
    prompt: str,
    style: str | None,
    room_type: str | None,
    room_description: str | None,
    room_id: str,
    materials_path: Path,
    style_rules_path: Path,
    out_path: Path,
    top_k: int = 10,
    llm_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selector = FloorMaterialSelector(materials_path=materials_path, style_rules_path=style_rules_path)
    selection = selector.select(
        prompt=prompt,
        style=style,
        room_type=room_type,
        room_description=room_description,
        top_k=top_k,
        room_id=room_id,
        llm_settings=llm_settings,
    )
    selector.save_selection(selection, out_path)
    return selection.to_dict()


def _floor_material_scene_payload(selection: dict[str, Any]) -> dict[str, Any]:
    material = selection.get("selected_material") or {}
    texture_candidate = selection.get("texture_candidate") or {}
    local_paths = material.get("local_image_paths") or []
    image_urls = material.get("image_urls") or []
    texture_path = (
        texture_candidate.get("texture_abs_path")
        or texture_candidate.get("texture_path")
        or (local_paths[0] if local_paths else (image_urls[0] if image_urls else None))
    )
    return {
        "source": "supplier_catalog",
        "sku": material.get("sku"),
        "name": material.get("name"),
        "product_url": material.get("product_url"),
        "texture_path": texture_path,
        "texture_usable_in_blender": bool(texture_candidate.get("usable_in_blender")),
        "texture_analysis_reason": texture_candidate.get("reason"),
        "texture_color_variation": ((texture_candidate.get("analysis") or {}).get("color_variation") or {}),
        "texture_tiling": {
            "mode": "mirror_repeat",
            "axis": "floor_xy",
            "origin": "room_center",
            "tile_size_m": _infer_floor_tile_size_m(material),
        },
        "material_type": material.get("material_type"),
        "decor": material.get("decor"),
        "design": material.get("design"),
        "tone": material.get("tone"),
        "plank_length_mm": material.get("plank_length_mm"),
        "plank_width_mm": material.get("plank_width_mm"),
        "thickness_mm": material.get("thickness_mm"),
        "class": material.get("class"),
    }


def _infer_floor_tile_size_m(material: dict[str, Any]) -> float:
    length_mm = _as_float(material.get("plank_length_mm"))
    width_mm = _as_float(material.get("plank_width_mm"))
    if length_mm and length_mm > 0:
        return round(max(0.35, min(length_mm / 1000.0, 2.4)), 4)
    if width_mm and width_mm > 0:
        return round(max(0.35, min((width_mm / 1000.0) * 4.0, 1.6)), 4)
    return 1.2


def _as_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def apply_flooring_to_scene(scene: dict[str, Any], flooring_selection: dict[str, Any]) -> dict[str, Any]:
    updated = copy.deepcopy(scene)
    payload = _floor_material_scene_payload(flooring_selection)
    room_id = flooring_selection.get("room_id")

    if isinstance(updated.get("rooms"), list):
        for room in updated["rooms"]:
            if isinstance(room, dict) and room.get("id") == room_id:
                room["floor_material"] = payload
                return updated
        if updated["rooms"] and isinstance(updated["rooms"][0], dict):
            updated["rooms"][0]["floor_material"] = payload
            return updated

    room = updated.setdefault("room", {})
    if isinstance(room, dict):
        room["floor_material"] = payload
    return updated


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(data: dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
