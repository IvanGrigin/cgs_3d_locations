from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from src.ChooseObject.wall_material_selector import WallMaterialSelector


def _resolve_texture_path(raw_path: str | None, materials_path: Path) -> str | None:
    text = str(raw_path or "").strip()
    if not text:
        return None
    if text.startswith(("http://", "https://")):
        return text
    path = Path(text).expanduser()
    candidates = [path]
    base_dir = materials_path if materials_path.is_dir() else materials_path.parent
    if not path.is_absolute():
        candidates.append(base_dir / path)
        candidates.append(Path.cwd() / path)
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            resolved = candidate
        if resolved.is_file():
            return str(resolved)
    return str(path.resolve() if path.is_absolute() else (base_dir / path).resolve())


def run_wall_selection(
    prompt: str,
    style: str | None,
    room_type: str | None,
    room_description: str | None,
    room_id: str,
    materials_path: Path,
    out_path: Path,
    top_k: int = 10,
    llm_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selector = WallMaterialSelector(materials_path=materials_path)
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


def _wall_material_scene_payload(selection: dict[str, Any], materials_path: Path | None = None) -> dict[str, Any]:
    material = selection.get("selected_material") or {}
    local_paths = material.get("local_image_paths") or []
    image_urls = material.get("image_urls") or []
    texture_path = local_paths[0] if local_paths else (image_urls[0] if image_urls else None)
    if materials_path is not None:
        texture_path = _resolve_texture_path(texture_path, materials_path)
    return {
        "source": "supplier_catalog",
        "sku": material.get("sku"),
        "name": material.get("name"),
        "product_url": material.get("product_url"),
        "texture_path": texture_path,
        "material_type": material.get("material_type"),
        "base_material": material.get("base_material"),
        "color": material.get("color"),
        "tone": material.get("tone"),
        "pattern": material.get("pattern"),
        "average_rgb": material.get("average_rgb"),
        "average_hex": material.get("average_hex"),
        "dominant_colors_rgb": material.get("dominant_colors_rgb") or [],
        "dominant_colors_hex": material.get("dominant_colors_hex") or [],
        "wall_tiling": {
            "mode": "repeat",
            "axis": "wall_uv",
            "tile_size_m": 1.0,
        },
    }


def apply_wall_material_to_scene(scene: dict[str, Any], wall_selection: dict[str, Any]) -> dict[str, Any]:
    updated = copy.deepcopy(scene)
    payload = _wall_material_scene_payload(wall_selection)
    room_id = wall_selection.get("room_id")
    if isinstance(updated.get("rooms"), list):
        for room in updated["rooms"]:
            if isinstance(room, dict) and room.get("id") == room_id:
                room["wall_material"] = payload
                return updated
        if updated["rooms"] and isinstance(updated["rooms"][0], dict):
            updated["rooms"][0]["wall_material"] = payload
            return updated
    room = updated.setdefault("room", {})
    if isinstance(room, dict):
        room["wall_material"] = payload
    return updated


def apply_wall_material_to_scene_with_catalog(
    scene: dict[str, Any],
    wall_selection: dict[str, Any],
    materials_path: Path,
) -> dict[str, Any]:
    updated = copy.deepcopy(scene)
    payload = _wall_material_scene_payload(wall_selection, materials_path=materials_path)
    room_id = wall_selection.get("room_id")
    if isinstance(updated.get("rooms"), list):
        for room in updated["rooms"]:
            if isinstance(room, dict) and room.get("id") == room_id:
                room["wall_material"] = payload
                return updated
        if updated["rooms"] and isinstance(updated["rooms"][0], dict):
            updated["rooms"][0]["wall_material"] = payload
            return updated
    room = updated.setdefault("room", {})
    if isinstance(room, dict):
        room["wall_material"] = payload
    return updated


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(data: dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
