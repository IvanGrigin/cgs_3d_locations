#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import copy
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class LegoRefinerConfig:
    lego_repo: Optional[Path]
    checkpoint: Optional[Path]
    room_type: str
    mode: str


def _deepcopy_json(data: Any) -> Any:
    return copy.deepcopy(data)


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _extract_placements(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        for key in ("placements", "items", "objects", "layout"):
            value = payload.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    return []


def _inject_placements_like(original_payload: Any, placements: list[dict[str, Any]]) -> Any:
    payload = _deepcopy_json(original_payload)

    if isinstance(payload, dict):
        if isinstance(payload.get("placements"), list):
            payload["placements"] = placements
            return payload
        if isinstance(payload.get("items"), list):
            payload["items"] = placements
            return payload
        if isinstance(payload.get("objects"), list):
            payload["objects"] = placements
            return payload
        payload["placements"] = placements
        return payload

    return {"placements": placements}


def _scene_replace_placements(scene_payload: Any, placements: list[dict[str, Any]]) -> Any:
    scene = _deepcopy_json(scene_payload)
    if not isinstance(scene, dict):
        return {"placements": placements}
    scene["placements"] = placements
    return scene


def _get_first(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for k in keys:
        if k in payload:
            return payload[k]
    return None


def _normalize_vec3(value: Any) -> Optional[list[float]]:
    if isinstance(value, list) and len(value) >= 3 and all(_is_number(v) for v in value[:3]):
        return [float(value[0]), float(value[1]), float(value[2])]
    return None


def _normalize_vec2(value: Any) -> Optional[list[float]]:
    if isinstance(value, list) and len(value) >= 2 and all(_is_number(v) for v in value[:2]):
        return [float(value[0]), float(value[1])]
    return None


def _placement_to_lego_item(item: dict[str, Any], idx: int) -> dict[str, Any]:
    pos = (
        _normalize_vec3(_get_first(item, ("position", "pos", "translation")))
        or _normalize_vec3(item.get("center"))
        or [0.0, 0.0, 0.0]
    )

    size = (
        _normalize_vec3(_get_first(item, ("size", "sizes", "bbox_size", "dimensions", "extent")))
        or [1.0, 1.0, 1.0]
    )

    yaw = None
    angle = _get_first(item, ("yaw", "angle", "rotation_y"))
    if _is_number(angle):
        yaw = float(angle)

    rotation = item.get("rotation")
    if yaw is None and isinstance(rotation, list) and rotation and _is_number(rotation[0]):
        yaw = float(rotation[0])

    if yaw is None:
        yaw = 0.0

    category = (
        _get_first(item, ("category", "label", "class_name", "semantic_label", "name"))
        or f"object_{idx}"
    )

    object_id = (
        _get_first(item, ("id", "object_id", "jid", "model_jid", "uid"))
        or f"item_{idx}"
    )

    return {
        "id": str(object_id),
        "category": str(category),
        "position": pos,
        "size": size,
        "yaw": yaw,
        "raw": _deepcopy_json(item),
    }


def _placements_to_lego(placements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_placement_to_lego_item(item, idx) for idx, item in enumerate(placements)]


def _apply_lego_item_back(original_item: dict[str, Any], lego_item: dict[str, Any]) -> dict[str, Any]:
    item = _deepcopy_json(original_item)

    pos = lego_item.get("position")
    if _normalize_vec3(pos) is not None:
        if "position" in item and isinstance(item["position"], list):
            item["position"] = [float(pos[0]), float(pos[1]), float(pos[2])]
        elif "pos" in item and isinstance(item["pos"], list):
            item["pos"] = [float(pos[0]), float(pos[1]), float(pos[2])]
        elif "translation" in item and isinstance(item["translation"], list):
            item["translation"] = [float(pos[0]), float(pos[1]), float(pos[2])]
        elif "center" in item and isinstance(item["center"], list):
            item["center"] = [float(pos[0]), float(pos[1]), float(pos[2])]
        else:
            item["position"] = [float(pos[0]), float(pos[1]), float(pos[2])]

    size = lego_item.get("size")
    if _normalize_vec3(size) is not None:
        if "size" in item and isinstance(item["size"], list):
            item["size"] = [float(size[0]), float(size[1]), float(size[2])]
        elif "sizes" in item and isinstance(item["sizes"], list):
            item["sizes"] = [float(size[0]), float(size[1]), float(size[2])]
        elif "bbox_size" in item and isinstance(item["bbox_size"], list):
            item["bbox_size"] = [float(size[0]), float(size[1]), float(size[2])]
        elif "dimensions" in item and isinstance(item["dimensions"], list):
            item["dimensions"] = [float(size[0]), float(size[1]), float(size[2])]
        elif "extent" in item and isinstance(item["extent"], list):
            item["extent"] = [float(size[0]), float(size[1]), float(size[2])]
        else:
            item["size"] = [float(size[0]), float(size[1]), float(size[2])]

    yaw = lego_item.get("yaw")
    if _is_number(yaw):
        if "yaw" in item:
            item["yaw"] = float(yaw)
        elif "angle" in item:
            item["angle"] = float(yaw)
        elif "rotation_y" in item:
            item["rotation_y"] = float(yaw)
        elif "rotation" in item and isinstance(item["rotation"], list) and item["rotation"]:
            item["rotation"][0] = float(yaw)
        else:
            item["yaw"] = float(yaw)

    return item


def _apply_lego_result_to_placements(
    original_placements: list[dict[str, Any]],
    lego_items_before: list[dict[str, Any]],
    lego_items_after: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(original_placements) != len(lego_items_before) or len(lego_items_before) != len(lego_items_after):
        raise RuntimeError("Размер списка placements изменился при postprocess, это не поддержано")

    refined = []
    for src_item, refined_lego in zip(original_placements, lego_items_after):
        refined.append(_apply_lego_item_back(src_item, refined_lego))
    return refined


def _dummy_lego_refine(lego_items: list[dict[str, Any]], cfg: LegoRefinerConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # Без точной схемы конвертации во внутренний формат LEGO-Net
    # выполняем безопасный pass-through.
    meta = {
        "strategy": "passthrough",
        "reason": "bridge_without_exact_lego_scene_mapping",
        "room_type": cfg.room_type,
        "mode": cfg.mode,
        "checkpoint": str(cfg.checkpoint) if cfg.checkpoint else None,
        "lego_repo": str(cfg.lego_repo) if cfg.lego_repo else None,
    }
    return _deepcopy_json(lego_items), meta


def _maybe_use_real_lego_bridge(
    lego_items: list[dict[str, Any]],
    cfg: LegoRefinerConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any], bool]:
    # Точка расширения:
    # если позже появится реальная интеграция с LEGO-Net, меняется только этот блок.
    #
    # Сейчас deliberately safe fallback:
    refined, meta = _dummy_lego_refine(lego_items, cfg)
    return refined, meta, False


def refine_scene_with_lego(
    room_payload: dict[str, Any],
    placement_legacy_payload: dict[str, Any],
    placement_v1_payload: dict[str, Any],
    scene_v1_payload: dict[str, Any],
    cfg: LegoRefinerConfig,
) -> dict[str, Any]:
    if cfg.room_type not in {"bedroom", "livingroom"}:
        raise RuntimeError(f"Unsupported room_type for LEGO-Net: {cfg.room_type}")

    original_placements_legacy = _extract_placements(placement_legacy_payload)
    original_placements_v1 = _extract_placements(placement_v1_payload)
    original_scene_placements = _extract_placements(scene_v1_payload)

    if not original_placements_legacy:
        raise RuntimeError("В legacy placement нет placements/items")
    if not original_placements_v1:
        raise RuntimeError("В placement.v1 нет placements/items")
    if not original_scene_placements:
        raise RuntimeError("В scene.v1 нет placements")

    if len(original_placements_legacy) != len(original_placements_v1):
        raise RuntimeError("Количество объектов в legacy и v1 placement различается")

    lego_items = _placements_to_lego(original_placements_v1)
    refined_lego_items, meta, used_lego = _maybe_use_real_lego_bridge(lego_items, cfg)

    refined_v1_placements = _apply_lego_result_to_placements(
        original_placements=original_placements_v1,
        lego_items_before=lego_items,
        lego_items_after=refined_lego_items,
    )

    # legacy обновляем по тем же координатам/углам
    refined_legacy_placements = _apply_lego_result_to_placements(
        original_placements=original_placements_legacy,
        lego_items_before=lego_items,
        lego_items_after=refined_lego_items,
    )

    placement_v1_out = _inject_placements_like(placement_v1_payload, refined_v1_placements)
    placement_v1_out["_lego_meta"] = meta

    placement_legacy_out = _inject_placements_like(placement_legacy_payload, refined_legacy_placements)
    placement_legacy_out["_lego_meta"] = meta

    scene_v1_out = _scene_replace_placements(scene_v1_payload, refined_v1_placements)
    if isinstance(scene_v1_out, dict):
        scene_v1_out["_lego_meta"] = meta

    return {
        "placement_legacy": placement_legacy_out,
        "placement_v1": placement_v1_out,
        "scene_v1": scene_v1_out,
        "used_lego": used_lego,
        "meta": meta,
    }