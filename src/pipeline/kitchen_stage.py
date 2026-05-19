#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import html
import json
import math
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

try:
    from ..pipeline_config import PlacementArtifacts
    from ..suppliers.kitchen.kitchen_supplier_inventory import (
        build_kitchen_selection_index,
        compact_item as compact_kitchen_inventory_item,
        load_supplier_catalog,
    )
    from ..suppliers.kitchen.kitchen_pipeline import (
        build_kitchen_zone_from_target,
        generate_kitchen_variants,
        is_kitchen_target,
    )
    from ..suppliers.kitchen.kitchen_llm_decisions import plan_dining_with_llm
except ImportError:
    from pipeline_config import PlacementArtifacts
    from suppliers.kitchen.kitchen_supplier_inventory import (
        build_kitchen_selection_index,
        compact_item as compact_kitchen_inventory_item,
        load_supplier_catalog,
    )
    from suppliers.kitchen.kitchen_pipeline import (
        build_kitchen_zone_from_target,
        generate_kitchen_variants,
        is_kitchen_target,
    )
    from suppliers.kitchen.kitchen_llm_decisions import plan_dining_with_llm


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _room_dict(scene: dict[str, Any] | None, room_json_path: Path | None = None) -> dict[str, Any]:
    if isinstance(scene, dict) and isinstance(scene.get("room"), dict):
        return deepcopy(scene["room"])
    if room_json_path and room_json_path.is_file():
        data = _read_json(room_json_path)
        if isinstance(data, dict):
            return deepcopy(data.get("room") if isinstance(data.get("room"), dict) else data)
    return {}


def _room_is_kitchen(room: dict[str, Any], prompt_text: str) -> bool:
    text = " ".join(
        str(x or "")
        for x in (
            room.get("room_type"),
            room.get("type"),
            room.get("name"),
            room.get("id"),
            prompt_text,
        )
    ).lower()
    return any(token in text for token in ("kitchen", "кухн"))


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _item_aabb(item: dict[str, Any]) -> dict[str, float]:
    aabb = item.get("aabb") or item.get("bbox") or {}
    if isinstance(aabb, dict) and {"x_min", "x_max", "y_min", "y_max"}.issubset(aabb):
        z_min = _float(aabb.get("z_min"), 0.0)
        z_max = _float(aabb.get("z_max"), z_min + _float((item.get("size_m") or [0, 0, 2.2])[2], 2.2))
        return {
            "x_min": _float(aabb.get("x_min")),
            "x_max": _float(aabb.get("x_max"), 3.0),
            "y_min": _float(aabb.get("y_min")),
            "y_max": _float(aabb.get("y_max"), 0.6),
            "z_min": z_min,
            "z_max": z_max,
        }
    pos = item.get("position_m") or item.get("position") or [1.5, 0.3, 1.1]
    size = item.get("size_m") or [
        (item.get("dimensions") or {}).get("width_m", 3.0),
        (item.get("dimensions") or {}).get("depth_m", 0.6),
        (item.get("dimensions") or {}).get("height_m", 2.2),
    ]
    cx, cy, cz = [_float(x) for x in (list(pos) + [0, 0, 0])[:3]]
    sx, sy, sz = [_float(x) for x in (list(size) + [0, 0, 0])[:3]]
    return {
        "x_min": cx - sx / 2.0,
        "x_max": cx + sx / 2.0,
        "y_min": cy - sy / 2.0,
        "y_max": cy + sy / 2.0,
        "z_min": cz - sz / 2.0,
        "z_max": cz + sz / 2.0,
    }


def _prompt_kitchen_width_m(prompt_text: str) -> float | None:
    text = str(prompt_text or "").lower().replace(",", ".")
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:м|m|метр)", text):
        value = _float(match.group(1), 0.0)
        if 1.2 <= value <= 6.0:
            return value
    return None


def _room_polygon_xy(room: dict[str, Any]) -> list[tuple[float, float]]:
    points = room.get("floor_polygon") or []
    out: list[tuple[float, float]] = []
    if not isinstance(points, list):
        return out
    for point in points:
        if not isinstance(point, dict):
            continue
        x = _float(point.get("x"), float("nan"))
        y = _float(point.get("y", point.get("z")), float("nan"))
        if math.isfinite(x) and math.isfinite(y):
            out.append((x, y))
    return out


def _polygon_signed_area(poly: list[tuple[float, float]]) -> float:
    if len(poly) < 3:
        return 0.0
    return 0.5 * sum(
        poly[i][0] * poly[(i + 1) % len(poly)][1] - poly[(i + 1) % len(poly)][0] * poly[i][1]
        for i in range(len(poly))
    )


def _wall_candidates(room: dict[str, Any]) -> list[dict[str, Any]]:
    poly = _room_polygon_xy(room)
    walls = room.get("walls") if isinstance(room.get("walls"), list) else []
    if len(poly) < 2:
        width = _float(room.get("width_m") or room.get("width"), 3.2)
        depth = _float(room.get("depth_m") or room.get("depth"), 3.0)
        poly = [(0.0, 0.0), (width, 0.0), (width, depth), (0.0, depth)]
        walls = [{"id": f"w{i}", "from_vertex": i, "to_vertex": (i + 1) % 4} for i in range(4)]
    if not walls:
        walls = [{"id": f"w{i}", "from_vertex": i, "to_vertex": (i + 1) % len(poly)} for i in range(len(poly))]

    ccw = _polygon_signed_area(poly) > 0
    out: list[dict[str, Any]] = []
    for idx, wall in enumerate(walls):
        if not isinstance(wall, dict):
            continue
        a_idx = int(_float(wall.get("from_vertex"), idx))
        b_idx = int(_float(wall.get("to_vertex"), (idx + 1) % len(poly)))
        if a_idx < 0 or b_idx < 0 or a_idx >= len(poly) or b_idx >= len(poly):
            continue
        ax, ay = poly[a_idx]
        bx, by = poly[b_idx]
        dx, dy = bx - ax, by - ay
        length = math.hypot(dx, dy)
        if length <= 1e-6:
            continue
        ux, uy = dx / length, dy / length
        nx, ny = (-uy, ux) if ccw else (uy, -ux)
        out.append(
            {
                "id": str(wall.get("id") or f"w{idx}"),
                "a": (ax, ay),
                "b": (bx, by),
                "u": (ux, uy),
                "n": (nx, ny),
                "length": length,
                "yaw_deg": math.degrees(math.atan2(uy, ux)),
            }
        )
    return out


def _opening_interval_on_wall(opening: dict[str, Any], wall: dict[str, Any], *, margin: float = 0.12) -> tuple[float, float] | None:
    if str(opening.get("wall_id") or "") != wall["id"]:
        return None
    length = float(wall["length"])
    ux, uy = wall["u"]
    ax, ay = wall["a"]
    segment = opening.get("segment") if isinstance(opening.get("segment"), dict) else {}
    values: list[float] = []
    if segment:
        for x_key, y_key in (("x1", "y1"), ("x2", "y2")):
            x = _float(segment.get(x_key), float("nan"))
            y = _float(segment.get(y_key), float("nan"))
            if math.isfinite(x) and math.isfinite(y):
                values.append((x - ax) * ux + (y - ay) * uy)
    if not values:
        center = _float(opening.get("s"), float("nan"))
        width = _float(opening.get("width"), 0.8)
        if math.isfinite(center):
            values = [center - width * 0.5, center + width * 0.5]
    if not values:
        return None
    return max(0.0, min(values) - margin), min(length, max(values) + margin)


def _free_wall_intervals(room: dict[str, Any], wall: dict[str, Any]) -> list[tuple[float, float]]:
    blocked: list[tuple[float, float]] = []
    for key in ("doors", "windows", "openings"):
        raw = room.get(key)
        if not isinstance(raw, list):
            continue
        for opening in raw:
            if not isinstance(opening, dict):
                continue
            interval = _opening_interval_on_wall(opening, wall)
            if interval and interval[1] > interval[0]:
                blocked.append(interval)
    blocked.sort()
    free = [(0.0, float(wall["length"]))]
    for start, end in blocked:
        next_free: list[tuple[float, float]] = []
        for a, b in free:
            if end <= a or start >= b:
                next_free.append((a, b))
            else:
                if start > a:
                    next_free.append((a, start))
                if end < b:
                    next_free.append((end, b))
        free = next_free
    return [(a, b) for a, b in free if b - a >= 1.2]


def _select_kitchen_wall_target(room: dict[str, Any], prompt_text: str) -> dict[str, Any] | None:
    requested_width = _prompt_kitchen_width_m(prompt_text)
    best: tuple[float, dict[str, Any], tuple[float, float]] | None = None
    for wall in _wall_candidates(room):
        for interval in _free_wall_intervals(room, wall):
            free_len = interval[1] - interval[0]
            score = free_len
            if best is None or score > best[0]:
                best = (score, wall, interval)
    if best is None:
        return None

    _, wall, interval = best
    free_len = interval[1] - interval[0]
    kitchen_width = max(1.5, min(requested_width or free_len, free_len, 3.6))
    start = interval[0] + max(0.0, (free_len - kitchen_width) * 0.5)
    end = start + kitchen_width
    ax, ay = wall["a"]
    ux, uy = wall["u"]
    nx, ny = wall["n"]
    p1 = (ax + ux * start, ay + uy * start)
    p2 = (ax + ux * end, ay + uy * end)
    depth = 0.6
    corners = [p1, p2, (p2[0] + nx * depth, p2[1] + ny * depth), (p1[0] + nx * depth, p1[1] + ny * depth)]
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    center = ((p1[0] + p2[0]) * 0.5 + nx * depth * 0.5, (p1[1] + p2[1]) * 0.5 + ny * depth * 0.5)
    return {
        "id": "kitchen_001",
        "name": "Kitchen set",
        "category": "kitchen_set",
        "wall_id": wall["id"],
        "layout_type": "straight",
        "position_m": [center[0], center[1], 1.1],
        "size_m": [kitchen_width, depth, 2.2],
        "rotation": [0.0, 0.0, float(wall["yaw_deg"])],
        "kitchen_width_m": kitchen_width,
        "aabb": {
            "x_min": min(xs),
            "x_max": max(xs),
            "y_min": min(ys),
            "y_max": max(ys),
            "z_min": 0.0,
            "z_max": 2.2,
        },
        "constraints": {"against_wall": True, "avoid_wall_openings": True},
        "meta": {
            "generated_kitchen_target": True,
            "wall_id": wall["id"],
            "free_interval_m": [round(start, 4), round(end, 4)],
            "wall_length_m": round(float(wall["length"]), 4),
            "opening_aware": True,
        },
    }


def _default_kitchen_target(room: dict[str, Any], prompt_text: str = "") -> dict[str, Any]:
    selected = _select_kitchen_wall_target(room, prompt_text)
    if selected is not None:
        return selected

    width = _float(room.get("width_m") or room.get("width"), 3.2)
    depth = _float(room.get("depth_m") or room.get("depth"), 3.0)
    longest_wall = max(width, depth)
    requested_width = _prompt_kitchen_width_m(prompt_text)
    kitchen_width = max(1.5, min(requested_width or longest_wall, longest_wall, 3.6))
    along_y = depth > width and kitchen_width > width
    if along_y:
        position_m = [0.3, kitchen_width / 2.0, 1.1]
        size_m = [0.6, kitchen_width, 2.2]
        aabb = {
            "x_min": 0.0,
            "x_max": 0.6,
            "y_min": 0.0,
            "y_max": kitchen_width,
            "z_min": 0.0,
            "z_max": 2.2,
        }
        # Local kitchen X runs along the wall and local Y is cabinet depth.
        # For a kitchen on the left long wall, -90 deg maps depth into +room X.
        rotation = [0.0, 0.0, -90.0]
    else:
        position_m = [kitchen_width / 2.0, 0.3, 1.1]
        size_m = [kitchen_width, 0.6, 2.2]
        aabb = {
            "x_min": 0.0,
            "x_max": kitchen_width,
            "y_min": 0.0,
            "y_max": 0.6,
            "z_min": 0.0,
            "z_max": 2.2,
        }
        rotation = [0.0, 0.0, 0.0]
    return {
        "id": "kitchen_001",
        "name": "Kitchen set",
        "category": "kitchen_set",
        "position_m": position_m,
        "size_m": size_m,
        "rotation": rotation,
        "kitchen_width_m": kitchen_width,
        "aabb": aabb,
        "constraints": {"against_wall": True},
        "meta": {"generated_kitchen_target": True, "room_depth_m": depth, "along_long_wall": along_y},
    }


def _has_target_like(items: list[Any], tokens: tuple[str, ...]) -> bool:
    for item in items:
        if not isinstance(item, dict):
            continue
        text = " ".join(str(item.get(k) or "") for k in ("id", "name", "category", "semantic_group")).lower()
        if any(token in text for token in tokens):
            return True
    return False


_INFINIGEN_KITCHEN_TOKENS = (
    "kitchenspacefactory",
    "kitchenfactory",
    "kitchencabinet",
    "singlecabinetfactory",
    "cabinetfactory",
    "countertop",
    "kitchen_counter",
    "kitchen counter",
    "kitchen_set",
    "kitchen cabinet",
    "base cabinet",
    "wall cabinet",
    "ovenfactory",
    "beveragefridgefactory",
    "refrigeratorfactory",
    "fridgefactory",
    "dishwasherfactory",
    "sinkfactory",
    "stovefactory",
    "cooktopfactory",
    "rangehoodfactory",
    "hoodfactory",
    "microwavefactory",
)


def _item_text_for_matching(item: dict[str, Any]) -> str:
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    source = item.get("source") if isinstance(item.get("source"), dict) else {}
    parts = [
        item.get("id"),
        item.get("name"),
        item.get("category"),
        item.get("semantic_group"),
        item.get("type"),
        source.get("blend_object_name"),
        source.get("source_object_name"),
        meta.get("source_target_id"),
        meta.get("companion_role"),
    ]
    return " ".join(str(x or "") for x in parts).lower()


def _is_infinigen_kitchen_object(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    if bool(meta.get("kitchen_stage_generated")) or meta.get("procedural_assembly") == "kitchen":
        return True
    if is_kitchen_target(item):
        return True
    text = _item_text_for_matching(item)
    return any(token in text for token in _INFINIGEN_KITCHEN_TOKENS)


def _is_kitchen_stage_dining_object(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    role = str(meta.get("companion_role") or "").lower()
    if role in {"dining_table", "dining_chair"}:
        return True
    text = _item_text_for_matching(item)
    return any(token in text for token in ("kitchen_dining_table", "kitchen_dining_chair"))


def _remove_existing_kitchen_stage_objects(items: list[Any], *, remove_dining: bool) -> tuple[list[Any], list[dict[str, Any]]]:
    kept: list[Any] = []
    removed: list[dict[str, Any]] = []
    for item in items:
        remove = _is_infinigen_kitchen_object(item) or (remove_dining and _is_kitchen_stage_dining_object(item))
        if remove:
            if isinstance(item, dict):
                removed.append(
                    {
                        "id": item.get("id"),
                        "name": item.get("name"),
                        "category": item.get("category"),
                        "semantic_group": item.get("semantic_group"),
                    }
                )
            continue
        kept.append(item)
    return kept, removed


def _has_companion_role(items: list[Any], role: str) -> bool:
    for item in items:
        if not isinstance(item, dict):
            continue
        meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
        if str(meta.get("companion_role") or "") == role:
            return True
    return False


def _item_key(item: dict[str, Any]) -> str:
    return str(item.get("unique_key") or item.get("product_url") or item.get("title") or item.get("name") or "")


def _inventory_row_text(row: dict[str, Any]) -> str:
    parts = [
        row.get("title"),
        row.get("name"),
        row.get("category_norm"),
        row.get("category_raw"),
        row.get("description"),
        row.get("vlm_description_summary"),
        row.get("vlm_description_text"),
    ]
    return " ".join(str(x or "").lower() for x in parts)


def _has_local_asset(row: dict[str, Any]) -> bool:
    return bool(str(row.get("asset_local_path") or "").strip())


def _has_downloadable_asset(row: dict[str, Any]) -> bool:
    return bool(
        str(row.get("model_download_url") or "").strip()
        or str(row.get("model_download_landing_url") or "").strip()
        or str(row.get("model_page_url") or "").strip()
    )


def _load_kitchen_selection_index(catalog_path: Path | None) -> dict[str, Any] | None:
    if not catalog_path or not catalog_path.is_file():
        return None
    try:
        return build_kitchen_selection_index(load_supplier_catalog(catalog_path))
    except Exception:
        return None


def _pick_kitchen_inventory_item(
    inventory_index: dict[str, Any] | None,
    *,
    bucket: str,
    prefer_terms: tuple[str, ...] = (),
    avoid_terms: tuple[str, ...] = (),
    used_keys: set[str] | None = None,
) -> dict[str, Any] | None:
    if not inventory_index:
        return None
    used_keys = used_keys if used_keys is not None else set()
    buckets = inventory_index.get("kitchen_buckets") if isinstance(inventory_index.get("kitchen_buckets"), dict) else {}
    rows = buckets.get(bucket) if isinstance(buckets, dict) else None
    if not isinstance(rows, list):
        return None

    preferred = tuple(term.lower() for term in prefer_terms if term)
    avoided = tuple(term.lower() for term in avoid_terms if term)
    ranked: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        key = _item_key(row)
        if key and key in used_keys:
            continue
        text = _inventory_row_text(row)
        title = str(row.get("title") or row.get("name") or "").lower()
        prefer_hits = sum(1 for term in preferred if term in text)
        title_hits = sum(1 for term in preferred if term in title)
        avoid_hits = sum(1 for term in avoided if term in text)
        asset_rank = 0 if _has_local_asset(row) else 1 if _has_downloadable_asset(row) else 2
        rank = (
            -title_hits,
            -prefer_hits,
            avoid_hits,
            asset_rank,
            idx,
            key,
        )
        ranked.append((rank, row))
    if not ranked:
        return None
    ranked.sort(key=lambda x: x[0])
    picked = ranked[0][1]
    key = _item_key(picked)
    if key:
        used_keys.add(key)
    return picked


def _inventory_rows_for_bucket(inventory_index: dict[str, Any] | None, bucket: str) -> list[dict[str, Any]]:
    if not inventory_index:
        return []
    buckets = inventory_index.get("kitchen_buckets") if isinstance(inventory_index.get("kitchen_buckets"), dict) else {}
    rows = buckets.get(bucket) if isinstance(buckets, dict) else []
    return [row for row in rows if isinstance(row, dict)]


def _inventory_item_by_key(inventory_index: dict[str, Any] | None, key: Any) -> dict[str, Any] | None:
    wanted = str(key or "").strip()
    if not wanted or not inventory_index:
        return None
    for bucket in ("kitchenware", "food_fruit", "oil_bottles_decor", "flowers_vases"):
        for row in _inventory_rows_for_bucket(inventory_index, bucket):
            if _item_key(row) == wanted:
                return row
    return None


def _extract_llm_text(resp: dict[str, Any]) -> str:
    message = resp.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"].strip()
    response_text = resp.get("response")
    if isinstance(response_text, str):
        return response_text.strip()
    return json.dumps(resp, ensure_ascii=False)


def _parse_json_object(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw).strip()
        raw = re.sub(r"```$", "", raw).strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    match = re.search(r"\{.*\}", raw, flags=re.S)
    if match:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed
    return {}


def _candidate_payload(row: dict[str, Any]) -> dict[str, Any]:
    item = compact_kitchen_inventory_item(row)
    return {
        "unique_key": item.get("unique_key"),
        "title": item.get("title"),
        "category_norm": item.get("category_norm"),
        "source_site": item.get("source_site"),
        "has_local_asset": bool(item.get("asset_local_path")),
        "has_downloadable_asset": bool(item.get("model_download_url") or item.get("model_download_landing_url")),
        "dimensions_cm": {
            "width": item.get("width_cm"),
            "depth": item.get("depth_cm"),
            "height": item.get("height_cm"),
        },
    }


def _bucket_candidates_for_llm(inventory_index: dict[str, Any] | None, bucket: str, limit: int = 12) -> list[dict[str, Any]]:
    rows = _inventory_rows_for_bucket(inventory_index, bucket)
    ranked = sorted(
        rows,
        key=lambda row: (
            0 if _has_local_asset(row) else 1 if _has_downloadable_asset(row) else 2,
            str(row.get("title") or row.get("name") or ""),
        ),
    )
    return [_candidate_payload(row) for row in ranked[:limit]]


def _default_accessory_plan(*, dining_possible: bool) -> list[dict[str, Any]]:
    plan = [
        {
            "role": "countertop_cooking_set",
            "bucket": "kitchenware",
            "surface": "countertop",
            "prefer_terms": ["kitchen accessories", "kitchen decor", "kitchenware", "набор", "посуда", "банок", "tableware"],
            "avoid_terms": ["wellness", "spa", "bathroom", "ванн"],
        },
        {
            "role": "oil_bottles",
            "bucket": "oil_bottles_decor",
            "surface": "countertop",
            "prefer_terms": ["oil", "olive", "bottle", "decanter", "jar", "бутыл", "масл", "графин", "банка"],
            "avoid_terms": ["armchair", "chair", "кресло", "стул"],
        },
        {
            "role": "fruit_plate",
            "bucket": "food_fruit",
            "surface": "countertop",
            "prefer_terms": ["fruit", "apple", "apples", "plate", "лимон", "фрукт", "яблок", "тарел"],
            "avoid_terms": ["tree", "oak", "sakura", "bamboo", "дерево"],
        },
        {
            "role": "flower_vase",
            "bucket": "flowers_vases",
            "surface": "countertop",
            "prefer_terms": ["flower", "bouquet", "vase", "букет", "цвет", "ваза"],
            "avoid_terms": ["tree", "oak", "sakura", "bamboo", "дерево"],
        },
    ]
    if dining_possible:
        plan.append(
            {
                "role": "tableware_set",
                "bucket": "kitchenware",
                "surface": "dining_table",
                "prefer_terms": ["tableware", "plate", "тарел", "посуда", "чаш", "миска"],
                "avoid_terms": ["wellness", "spa", "bathroom", "ванн"],
            }
        )
    return plan


def _plan_kitchen_accessories_with_llm(
    *,
    inventory_index: dict[str, Any] | None,
    room: dict[str, Any],
    prompt_text: str,
    dining_possible: bool,
    llm_settings: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    settings = dict(llm_settings or {})
    if str(settings.get("provider") or "none").strip().lower() == "none":
        return _default_accessory_plan(dining_possible=dining_possible), {"status": "skipped", "reason": "provider_none"}
    if str(settings.get("provider") or "").strip().lower() != "ollama":
        return _default_accessory_plan(dining_possible=dining_possible), {"status": "skipped", "reason": "unsupported_provider"}

    chat_json = None
    import_error: Exception | None = None
    for module_name in ("src.LLMModule.ollama_client", "LLMModule.ollama_client"):
        try:
            module = __import__(module_name, fromlist=["chat_json"])
            chat_json = getattr(module, "chat_json", None)
            if callable(chat_json):
                break
        except Exception as exc:
            import_error = exc
            chat_json = None
    if not callable(chat_json):
        return _default_accessory_plan(dining_possible=dining_possible), {
            "status": "failed",
            "reason": f"ollama_import_failed:{type(import_error).__name__ if import_error else 'RuntimeError'}:{import_error or 'chat_json_not_found'}",
        }

    payload = {
        "room": {
            "width_m": room.get("width_m") or room.get("width"),
            "depth_m": room.get("depth_m") or room.get("depth"),
            "room_type": room.get("room_type") or room.get("type"),
        },
        "prompt": prompt_text,
        "available_surfaces": ["countertop"] + (["dining_table"] if dining_possible else []),
        "rules": [
            "Choose 3-6 decorative kitchen accessory items.",
            "Use countertop for cooking sets, jars, oil bottles, fruit plates, flowers.",
            "Use dining_table only for tableware, fruit plate, flowers, or serving items.",
            "Do not place large plants/trees on the countertop.",
            "Copy unique_key exactly when selecting a concrete candidate; otherwise leave it empty and provide prefer_terms.",
        ],
        "candidate_buckets": {
            bucket: _bucket_candidates_for_llm(inventory_index, bucket, limit=12)
            for bucket in ("kitchenware", "food_fruit", "oil_bottles_decor", "flowers_vases")
        },
        "required_output": {
            "items": [
                {
                    "role": "short_snake_case",
                    "bucket": "kitchenware|food_fruit|oil_bottles_decor|flowers_vases",
                    "surface": "countertop|dining_table",
                    "unique_key": "optional exact candidate unique_key",
                    "prefer_terms": ["optional text hints"],
                    "avoid_terms": ["optional text hints"],
                    "reason": "short reason",
                }
            ]
        },
    }
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "role": {"type": "string"},
                        "bucket": {"type": "string"},
                        "surface": {"type": "string"},
                        "unique_key": {"type": "string"},
                        "prefer_terms": {"type": "array", "items": {"type": "string"}},
                        "avoid_terms": {"type": "array", "items": {"type": "string"}},
                        "reason": {"type": "string"},
                    },
                    "required": ["role", "bucket", "surface"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["items"],
        "additionalProperties": False,
    }
    system_prompt = (
        "You are planning kitchen countertop and dining table styling for a 3D scene. "
        "Select varied real supplier accessories from the provided kitchen catalog buckets. "
        "Return strict JSON only."
    )
    try:
        response = chat_json(
            base_url=str(settings.get("ollama_url") or "http://127.0.0.1:11434"),
            model=str(settings.get("ollama_model") or "gpt-oss:20b"),
            system_prompt=system_prompt,
            user_prompt=json.dumps(payload, ensure_ascii=False, indent=2),
            json_schema=schema,
            timeout_sec=int(settings.get("ollama_timeout") or 180),
            temperature=float(settings.get("ollama_temperature") or 0.2),
            think=str(settings.get("ollama_think") or "low"),
            extra_options={"num_ctx": int(settings.get("ollama_num_ctx") or 8192), "num_predict": 768},
        )
        parsed = _parse_json_object(_extract_llm_text(response))
        raw_items = parsed.get("items") if isinstance(parsed.get("items"), list) else []
    except Exception as exc:
        return _default_accessory_plan(dining_possible=dining_possible), {
            "status": "failed",
            "reason": f"ollama_accessory_plan_failed:{type(exc).__name__}:{exc}",
        }

    allowed_buckets = {"kitchenware", "food_fruit", "oil_bottles_decor", "flowers_vases"}
    allowed_surfaces = {"countertop", "dining_table"} if dining_possible else {"countertop"}
    plan: list[dict[str, Any]] = []
    seen_roles: set[str] = set()
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        bucket = str(raw.get("bucket") or "").strip()
        surface = str(raw.get("surface") or "countertop").strip()
        if bucket not in allowed_buckets or surface not in allowed_surfaces:
            continue
        role = re.sub(r"[^a-zA-Z0-9_]+", "_", str(raw.get("role") or bucket)).strip("_").lower() or bucket
        if role in seen_roles:
            continue
        seen_roles.add(role)
        plan.append(
            {
                "role": role,
                "bucket": bucket,
                "surface": surface,
                "unique_key": str(raw.get("unique_key") or "").strip(),
                "prefer_terms": [str(x) for x in raw.get("prefer_terms") or []],
                "avoid_terms": [str(x) for x in raw.get("avoid_terms") or []],
                "llm_reason": raw.get("reason"),
            }
        )
        if len(plan) >= 6:
            break
    if not plan:
        return _default_accessory_plan(dining_possible=dining_possible), {"status": "fallback", "reason": "empty_llm_plan"}
    return plan, {"status": "ok", "item_count": len(plan), "items": plan}


def _room_size(room: dict[str, Any]) -> tuple[float, float]:
    width = _float(room.get("width_m") or room.get("width"), 3.2)
    depth = _float(room.get("depth_m") or room.get("depth"), 3.0)
    return max(width, 1.8), max(depth, 1.8)


def _make_aabb(cx: float, cy: float, z_min: float, sx: float, sy: float, sz: float) -> dict[str, float]:
    return {
        "x_min": round(cx - sx / 2.0, 4),
        "x_max": round(cx + sx / 2.0, 4),
        "y_min": round(cy - sy / 2.0, 4),
        "y_max": round(cy + sy / 2.0, 4),
        "z_min": round(z_min, 4),
        "z_max": round(z_min + sz, 4),
    }


def _target_item(
    *,
    item_id: str,
    name: str,
    category: str,
    center_xy: tuple[float, float],
    size: tuple[float, float, float],
    z_min: float = 0.0,
    yaw_deg: float = 0.0,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sx, sy, sz = size
    cx, cy = center_xy
    return {
        "id": item_id,
        "name": name,
        "category": category,
        "semantic_group": category,
        "position_m": [round(cx, 4), round(cy, 4), round(z_min + sz / 2.0, 4)],
        "size_m": [sx, sy, sz],
        "rotation_deg": yaw_deg,
        "yaw_deg": yaw_deg,
        "aabb": _make_aabb(cx, cy, z_min, sx, sy, sz),
        "mount_type": "floor" if z_min <= 0.05 else "surface",
        "constraints": {"mount_type": "floor" if z_min <= 0.05 else "surface"},
        "meta": dict(meta or {}),
    }


def _chair_yaw_back_away_from_table(chair_xy: tuple[float, float], table_xy: tuple[float, float]) -> float:
    """Yaw where generated chair local +Y/back side points away from the table."""
    away_x = float(chair_xy[0]) - float(table_xy[0])
    away_y = float(chair_xy[1]) - float(table_xy[1])
    if abs(away_x) < 1e-6 and abs(away_y) < 1e-6:
        return 0.0
    return math.degrees(math.atan2(-away_x, away_y)) % 360.0


def _dimension_m_from_row(row: dict[str, Any] | None, key: str, default: float) -> float:
    if not row:
        return default
    value = row.get(f"{key}_cm")
    dimensions = row.get("dimensions_cm")
    if value is None and isinstance(dimensions, dict):
        value = dimensions.get(key)
    try:
        cm = float(value)
    except Exception:
        return default
    if cm <= 0:
        return default
    return max(0.04, cm / 100.0)


def _accessory_size_from_row(row: dict[str, Any] | None, fallback: tuple[float, float, float]) -> tuple[float, float, float]:
    width = _dimension_m_from_row(row, "width", fallback[0])
    depth = _dimension_m_from_row(row, "depth", fallback[1])
    height = _dimension_m_from_row(row, "height", fallback[2])
    return (
        round(min(max(width, 0.12), 0.62), 4),
        round(min(max(depth, 0.10), 0.48), 4),
        round(min(max(height, 0.08), 0.70), 4),
    )


def _fallback_size_for_bucket(bucket: str) -> tuple[float, float, float]:
    return {
        "kitchenware": (0.38, 0.28, 0.22),
        "food_fruit": (0.28, 0.28, 0.18),
        "oil_bottles_decor": (0.34, 0.22, 0.36),
        "flowers_vases": (0.28, 0.28, 0.46),
    }.get(bucket, (0.30, 0.24, 0.24))


def _surface_slots(
    *,
    surface: str,
    room_width: float,
    room_depth: float,
    dining_possible: bool,
) -> list[dict[str, float]]:
    if surface == "dining_table" and dining_possible:
        compact = room_depth < 2.35
        table_cx = min(max(room_width * (0.72 if compact else 0.54), 1.25), room_width - (0.55 if compact else 1.05))
        table_cy = min(max(0.72 + (0.62 if compact else 0.95), room_depth * (0.66 if compact else 0.58)), room_depth - (0.45 if compact else 0.75))
        return [
            {
                "x_min": table_cx - (0.24 if compact else 0.42),
                "x_max": table_cx + (0.24 if compact else 0.42),
                "y_min": table_cy - (0.15 if compact else 0.25),
                "y_max": table_cy + (0.15 if compact else 0.25),
                "z_min": 0.76,
            }
        ]
    return [
        {
            "x_min": max(0.35, room_width * 0.18),
            "x_max": min(room_width - 0.35, room_width * 0.82),
            "y_min": 0.22,
            "y_max": 0.52,
            "z_min": 0.92,
        }
    ]


def _pack_surface_items(
    specs: list[dict[str, Any]],
    *,
    surface: str,
    room_width: float,
    room_depth: float,
    dining_possible: bool,
) -> list[dict[str, Any]]:
    if not specs:
        return []
    slots = _surface_slots(surface=surface, room_width=room_width, room_depth=room_depth, dining_possible=dining_possible)
    if not slots:
        return []
    rect = slots[0]
    available_w = max(0.2, rect["x_max"] - rect["x_min"])
    gap = 0.08 if surface == "countertop" else 0.06
    total_w = sum(float(spec["size"][0]) for spec in specs) + gap * max(0, len(specs) - 1)
    scale = min(1.0, available_w / max(total_w, 0.01))
    cursor = rect["x_min"] + max(0.0, available_w - total_w * scale) / 2.0
    packed: list[dict[str, Any]] = []
    for spec in specs:
        sx, sy, sz = [float(x) for x in spec["size"]]
        sx = max(0.10, sx * scale)
        sy = min(max(0.10, sy * scale), max(0.10, rect["y_max"] - rect["y_min"]))
        cx = cursor + sx / 2.0
        cy = (rect["y_min"] + rect["y_max"]) / 2.0
        cursor += sx + gap
        next_spec = dict(spec)
        next_spec["size"] = (round(sx, 4), round(sy, 4), sz)
        next_spec["center_xy"] = (round(cx, 4), round(cy, 4))
        next_spec["z_min"] = rect["z_min"]
        packed.append(next_spec)
    return packed


def _appliance_summary(assembly: dict[str, Any]) -> dict[str, Any]:
    bindings = assembly.get("appliance_bindings") if isinstance(assembly.get("appliance_bindings"), dict) else {}
    appliances = bindings.get("appliances") if isinstance(bindings.get("appliances"), dict) else {}
    out: dict[str, Any] = {}
    for role, entry in appliances.items():
        if not isinstance(entry, dict):
            continue
        chosen = entry.get("chosen_asset") if isinstance(entry.get("chosen_asset"), dict) else {}
        out[str(role)] = {
            "unique_key": chosen.get("unique_key"),
            "title": chosen.get("title"),
            "category_norm": chosen.get("category_norm"),
            "source_site": chosen.get("source_site"),
            "asset_local_path": chosen.get("asset_local_path"),
            "product_url": chosen.get("product_url"),
            "top_candidate_count": len(entry.get("top_candidates") or []) if isinstance(entry.get("top_candidates"), list) else 0,
        }
    unavailable = bindings.get("unavailable_assets") if isinstance(bindings.get("unavailable_assets"), dict) else {}
    for role, candidates in unavailable.items():
        if role in out or not isinstance(candidates, list):
            continue
        out[str(role)] = {
            "unique_key": None,
            "title": None,
            "category_norm": None,
            "asset_local_path": None,
            "unavailable_candidate_count": len(candidates),
        }
    return out


def _append_kitchen_companion_targets(
    items: list[Any],
    *,
    room: dict[str, Any],
    prompt_text: str,
    add_dining: bool,
    add_accessories: bool,
    inventory_index: dict[str, Any] | None = None,
    accessory_llm_settings: dict[str, Any] | None = None,
    dining_llm_settings: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    additions: list[dict[str, Any]] = []
    used_inventory_keys: set[str] = set()
    width, depth = _room_size(room)
    kitchen_depth = 0.65
    dining_possible = width >= 2.6 and depth >= 1.85

    dining_plan = plan_dining_with_llm(room=room, prompt_text=prompt_text, llm_settings=dining_llm_settings)
    if dining_plan.get("status") == "ok":
        dining_possible = dining_possible and bool(dining_plan.get("add_dining", True))

    if add_dining and dining_possible and not _has_target_like(items, ("dining_table", "dining table", "обеденный стол")):
        compact_dining = depth < 2.35
        table_cx = min(max(width * (0.72 if compact_dining else 0.54), 1.25), width - (0.55 if compact_dining else 1.05))
        table_cy = min(max(kitchen_depth + (0.58 if compact_dining else 0.95), depth * (0.64 if compact_dining else 0.58)), depth - (0.43 if compact_dining else 0.75))
        table_size = (0.68, 0.50, 0.74) if compact_dining else (1.20, 0.78, 0.75)
        table_yaw = 0.0
        chair_count_override: int | None = None
        if dining_plan.get("status") == "ok" and isinstance(dining_plan.get("table"), dict):
            table_plan = dining_plan["table"]
            plan_w = min(max(_float(table_plan.get("width_m"), table_size[0]), 0.55), 1.60)
            plan_d = min(max(_float(table_plan.get("depth_m"), table_size[1]), 0.45), 1.00)
            table_size = (plan_w, plan_d, 0.75)
            table_cx = min(max(_float(table_plan.get("x_m"), table_cx), table_size[0] / 2.0 + 0.08), width - table_size[0] / 2.0 - 0.08)
            table_cy = min(max(_float(table_plan.get("y_m"), table_cy), kitchen_depth + table_size[1] / 2.0 + 0.18), depth - table_size[1] / 2.0 - 0.08)
            table_yaw = _float(table_plan.get("yaw_deg"), 0.0)
            chair_count_override = max(1, min(6, int(_float(dining_plan.get("chair_count"), 2))))
        table = _target_item(
            item_id="kitchen_dining_table_001",
            name="Dining table",
            category="dining_table",
            center_xy=(table_cx, table_cy),
            size=table_size,
            yaw_deg=table_yaw,
            meta={"kitchen_stage_generated": True, "companion_role": "dining_table", "kitchen_dining_llm_plan": dining_plan},
        )
        items.append(table)
        additions.append({"id": table["id"], "category": table["category"], "role": "dining_table"})

        if not _has_target_like(items, ("kitchen_dining_chair", "dining chair", "стул")):
            if compact_dining:
                chair_specs = [
                    ("kitchen_dining_chair_001", table_cx, table_cy - 0.50),
                    ("kitchen_dining_chair_002", table_cx, table_cy + 0.50),
                ]
                chair_size = (0.38, 0.38, 0.84)
            else:
                chair_specs = [
                    ("kitchen_dining_chair_001", table_cx - 0.52, table_cy),
                    ("kitchen_dining_chair_002", table_cx + 0.52, table_cy),
                ]
                chair_size = (0.48, 0.48, 0.88)
            if not compact_dining and width >= 3.0 and depth >= 2.8:
                chair_specs.extend(
                    [
                        ("kitchen_dining_chair_003", table_cx, table_cy - 0.46),
                        ("kitchen_dining_chair_004", table_cx, table_cy + 0.46),
                    ]
                )
            if chair_count_override is not None:
                chair_specs = chair_specs[:chair_count_override]
            for chair_id, cx, cy in chair_specs:
                chair_half_x = chair_size[0] / 2.0
                chair_half_y = chair_size[1] / 2.0
                min_chair_cx = chair_half_x + 0.05
                max_chair_cx = width - chair_half_x - 0.05
                min_chair_cy = kitchen_depth + chair_half_y + 0.04
                max_chair_cy = depth - chair_half_y - 0.04
                clamped_xy = (min(max(cx, min_chair_cx), max_chair_cx), min(max(cy, min_chair_cy), max_chair_cy))
                yaw = _chair_yaw_back_away_from_table(clamped_xy, (table_cx, table_cy))
                chair = _target_item(
                    item_id=chair_id,
                    name="Dining chair",
                    category="chair",
                    center_xy=clamped_xy,
                    size=chair_size,
                    yaw_deg=yaw,
                    meta={
                        "kitchen_stage_generated": True,
                        "companion_role": "dining_chair",
                        "support_group": "kitchen_dining",
                        "affordance": "table_chair",
                        "target_table_id": table["id"],
                        "orientation_rule": "chair_back_farther_from_table_than_center",
                    },
                )
                items.append(chair)
                additions.append({"id": chair["id"], "category": chair["category"], "role": "dining_chair"})

    if add_accessories:
        plan, llm_plan_info = _plan_kitchen_accessories_with_llm(
            inventory_index=inventory_index,
            room=room,
            prompt_text=prompt_text,
            dining_possible=dining_possible,
            llm_settings=accessory_llm_settings,
        )
        accessory_specs_by_surface: dict[str, list[dict[str, Any]]] = {"countertop": [], "dining_table": []}
        role_counts: dict[str, int] = {}
        for raw_spec in plan:
            bucket = str(raw_spec.get("bucket") or "kitchenware")
            role = re.sub(r"[^a-zA-Z0-9_]+", "_", str(raw_spec.get("role") or bucket)).strip("_").lower() or bucket
            role_counts[role] = role_counts.get(role, 0) + 1
            role_id = role if role_counts[role] == 1 else f"{role}_{role_counts[role]}"
            surface = str(raw_spec.get("surface") or "countertop")
            if surface not in accessory_specs_by_surface:
                surface = "countertop"
            catalog_row = _inventory_item_by_key(inventory_index, raw_spec.get("unique_key"))
            if catalog_row:
                key = _item_key(catalog_row)
                if key in used_inventory_keys:
                    catalog_row = None
                elif key:
                    used_inventory_keys.add(key)
            if catalog_row is None:
                catalog_row = _pick_kitchen_inventory_item(
                    inventory_index,
                    bucket=bucket,
                    prefer_terms=tuple(raw_spec.get("prefer_terms") or ()),
                    avoid_terms=tuple(raw_spec.get("avoid_terms") or ()),
                    used_keys=used_inventory_keys,
                )
            fallback_category = {
                "kitchenware": "kitchenware",
                "food_fruit": "food_drink",
                "oil_bottles_decor": "decorative_set",
                "flowers_vases": "plant_planter_vase",
            }.get(bucket, "kitchenware")
            size = _accessory_size_from_row(catalog_row, _fallback_size_for_bucket(bucket))
            accessory_specs_by_surface[surface].append(
                {
                    "item_id": f"kitchen_{role_id}_001" if role_id.startswith(surface) else f"kitchen_{surface}_{role_id}_001",
                    "name": str((catalog_row or {}).get("title") or (catalog_row or {}).get("name") or role.replace("_", " ").title()),
                    "category": fallback_category,
                    "bucket": bucket,
                    "role": role_id,
                    "size": size,
                    "catalog_row": catalog_row,
                    "support_surface": "dining_table" if surface == "dining_table" else "kitchen_countertop",
                    "llm_reason": raw_spec.get("llm_reason"),
                    "llm_plan_info": llm_plan_info,
                }
            )

        accessory_specs: list[dict[str, Any]] = []
        for surface, surface_specs in accessory_specs_by_surface.items():
            accessory_specs.extend(
                _pack_surface_items(
                    surface_specs,
                    surface=surface,
                    room_width=width,
                    room_depth=depth,
                    dining_possible=dining_possible,
                )
            )

        for spec in accessory_specs:
            if _has_companion_role(items, str(spec["role"])):
                continue
            target_name = str(spec["name"])
            target_category = str(spec["category"])
            inventory_meta: dict[str, Any] = {}
            catalog_row = spec.get("catalog_row") if isinstance(spec.get("catalog_row"), dict) else None
            if catalog_row:
                target_name = str(catalog_row.get("title") or catalog_row.get("name") or target_name)
                catalog_category = str(catalog_row.get("category_norm") or "").strip()
                if catalog_category in {"kitchenware", "food_drink", "decorative_set", "plant_planter_vase"}:
                    target_category = catalog_category
                inventory_meta = {
                    "kitchen_inventory_bucket": spec.get("bucket"),
                    "supplier_preferred_unique_key": catalog_row.get("unique_key"),
                    "supplier_preferred_title": catalog_row.get("title") or catalog_row.get("name"),
                    "supplier_preferred_category_norm": catalog_row.get("category_norm"),
                    "supplier_preferred_source_site": catalog_row.get("source_site"),
                    "supplier_preferred_asset_local_path": catalog_row.get("asset_local_path"),
                    "supplier_preferred_product_url": catalog_row.get("product_url") or catalog_row.get("source_url"),
                    "supplier_inventory_item": compact_kitchen_inventory_item(catalog_row),
                }
            inventory_meta["kitchen_accessory_llm_plan"] = spec.get("llm_plan_info")
            if spec.get("llm_reason"):
                inventory_meta["kitchen_accessory_llm_reason"] = spec.get("llm_reason")
            accessory = _target_item(
                item_id=str(spec["item_id"]),
                name=target_name,
                category=target_category,
                center_xy=spec["center_xy"],
                size=spec["size"],
                z_min=float(spec.get("z_min", 0.92)),
                meta={
                    "kitchen_stage_generated": True,
                    "companion_role": spec["role"],
                    "support_surface": spec.get("support_surface", "kitchen_countertop"),
                    **inventory_meta,
                },
            )
            items.append(accessory)
            additions.append(
                {
                    "id": accessory["id"],
                    "category": accessory["category"],
                    "role": spec["role"],
                    "support_surface": spec.get("support_surface"),
                    "position_m": accessory.get("position_m"),
                    "size_m": accessory.get("size_m"),
                    "inventory_bucket": spec.get("bucket"),
                    "supplier_preferred_unique_key": inventory_meta.get("supplier_preferred_unique_key"),
                    "supplier_preferred_title": inventory_meta.get("supplier_preferred_title"),
                    "kitchen_accessory_llm_status": (spec.get("llm_plan_info") or {}).get("status") if isinstance(spec.get("llm_plan_info"), dict) else None,
                    "kitchen_accessory_llm_reason": spec.get("llm_reason"),
                }
            )

    return additions


def _assembly_to_scene_item(assembly: dict[str, Any], original: dict[str, Any]) -> dict[str, Any]:
    aabb = _item_aabb(original)
    dims = assembly.get("dimensions") if isinstance(assembly.get("dimensions"), dict) else {}
    width = _float(dims.get("width_m"), max(0.0, aabb["x_max"] - aabb["x_min"]))
    depth = _float(dims.get("depth_m"), max(0.0, aabb["y_max"] - aabb["y_min"]))
    height = _float(dims.get("height_m"), max(0.0, aabb["z_max"] - aabb["z_min"]))
    rotation = original.get("rotation") or original.get("rotation_deg") or [0.0, 0.0, 0.0]
    yaw = rotation[2] if isinstance(rotation, list) and len(rotation) >= 3 else rotation
    try:
        yaw_norm = int(round(float(yaw))) % 180
    except Exception:
        yaw_norm = 0
    footprint_width = depth if yaw_norm == 90 else width
    footprint_depth = width if yaw_norm == 90 else depth
    if yaw_norm == 90:
        yaw_value = rotation[2] if isinstance(rotation, list) and len(rotation) >= 3 else rotation
        try:
            yaw_float = float(yaw_value)
        except Exception:
            yaw_float = 0.0
        if yaw_float < 0:
            position = [aabb["x_min"], aabb["y_max"], max(0.0, aabb["z_min"])]
        else:
            position = [aabb["x_max"], aabb["y_min"], max(0.0, aabb["z_min"])]
    else:
        position = [aabb["x_min"], aabb["y_min"], max(0.0, aabb["z_min"])]
    assembly = deepcopy(assembly)
    assembly["position"] = position
    assembly["rotation"] = rotation
    assembly["size_m"] = [footprint_width, footprint_depth, height]
    assembly["position_m"] = [aabb["x_min"] + footprint_width / 2.0, aabb["y_min"] + footprint_depth / 2.0, aabb["z_min"] + height / 2.0]
    assembly["aabb"] = {
        "x_min": aabb["x_min"],
        "x_max": aabb["x_min"] + footprint_width,
        "y_min": aabb["y_min"],
        "y_max": aabb["y_min"] + footprint_depth,
        "z_min": aabb["z_min"],
        "z_max": aabb["z_min"] + height,
    }
    assembly["name"] = original.get("name") or "Kitchen set"
    assembly["category"] = "kitchen_set"
    assembly["mount_type"] = "floor"
    assembly["constraints"] = deepcopy(original.get("constraints") or {})
    meta = deepcopy(original.get("meta") or {})
    meta.update(
        {
            "procedural_assembly": "kitchen",
            "source_target_id": original.get("id"),
            "kitchen_mode": assembly.get("mode"),
            "kitchen_layout_type": assembly.get("layout_type"),
        }
    )
    assembly["meta"] = meta
    assembly["asset"] = {
        "kind": "procedural_kitchen",
        "assembly_type": "procedural_kitchen",
    }
    return assembly


def _replace_kitchens_in_doc(
    data: dict[str, Any],
    *,
    room: dict[str, Any],
    material_catalog: Path,
    appliance_catalog: Path | None,
    inventory_index: dict[str, Any] | None,
    prompt_text: str,
    mode: str,
    add_if_missing: bool,
    add_dining: bool,
    add_accessories: bool,
    accessory_llm_settings: dict[str, Any] | None,
    kitchen_llm_settings: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    out = deepcopy(data)
    key = "placements" if isinstance(out.get("placements"), list) else "items"
    items = out.get(key)
    if not isinstance(items, list):
        return out, [], []

    if add_if_missing:
        items, removed_items = _remove_existing_kitchen_stage_objects(items, remove_dining=add_dining)
        out[key] = items
        target = _default_kitchen_target(room, prompt_text)
        target.setdefault("meta", {})["removed_infinigen_kitchen_item_count"] = len(removed_items)
        target.setdefault("meta", {})["removed_infinigen_kitchen_items"] = removed_items[:40]
        targets = [target]
        items.append(target)
    else:
        targets = [item for item in items if isinstance(item, dict) and is_kitchen_target(item)]

    replacements: list[dict[str, Any]] = []
    for target in targets:
        aabb = _item_aabb(target)
        kitchen_zone = build_kitchen_zone_from_target(target, room=room)
        kitchen_zone["available_width_mm"] = max(
            1500,
            int(round(max(kitchen_zone.get("available_width_mm") or 0, (aabb["x_max"] - aabb["x_min"]) * 1000.0))),
        )
        variants = generate_kitchen_variants(
            material_catalog=material_catalog,
            user_prompt=prompt_text,
            room=room,
            kitchen_zone=kitchen_zone,
            appliance_catalog=appliance_catalog if appliance_catalog and appliance_catalog.is_file() else None,
            modes=[mode],
            target_id=str(target.get("id") or "kitchen_001"),
            position=[aabb["x_min"], aabb["y_min"], max(0.0, aabb["z_min"])],
            llm_settings=kitchen_llm_settings,
        )
        assembly = variants[mode]
        replacements.append(
            {
                "target_id": target.get("id"),
                "assembly_id": assembly.get("id"),
                "mode": mode,
                "layout_type": assembly.get("layout_type"),
                "dimensions": assembly.get("dimensions"),
                "price_estimate": assembly.get("price_estimate"),
                "appliances": _appliance_summary(assembly),
                "warnings": assembly.get("warnings") or [],
                "removed_infinigen_kitchen_item_count": (target.get("meta") or {}).get("removed_infinigen_kitchen_item_count", 0),
            }
        )
        replacement_item = _assembly_to_scene_item(assembly, target)
        for idx, item in enumerate(items):
            if item is target or (isinstance(item, dict) and item.get("id") == target.get("id")):
                items[idx] = replacement_item
                break

    companion_additions = _append_kitchen_companion_targets(
        items,
        room=room,
        prompt_text=prompt_text,
        add_dining=bool(replacements) and add_dining,
        add_accessories=bool(replacements) and add_accessories,
        inventory_index=inventory_index,
        accessory_llm_settings=accessory_llm_settings,
        dining_llm_settings=kitchen_llm_settings,
    )

    meta = out.setdefault("meta", {})
    if isinstance(meta, dict) and replacements:
        meta["kitchen_stage"] = {
            "enabled": True,
            "mode": mode,
            "replacement_count": len(replacements),
            "companion_target_count": len(companion_additions),
            "material_catalog": str(material_catalog),
        }
    return out, replacements, companion_additions


def write_kitchen_report(run_dir: Path, replacements: list[dict[str, Any]], companion_targets: list[dict[str, Any]], suffix: str) -> dict[str, Any]:
    safe_suffix = f".{suffix.strip('.')}" if suffix.strip(".") else ""
    summary_path = run_dir / f"kitchen_stage{safe_suffix}.summary.json"
    md_path = run_dir / f"kitchen_stage{safe_suffix}.md"
    html_path = run_dir / f"kitchen_stage{safe_suffix}.html"
    total = sum(
        float((row.get("price_estimate") or {}).get("total_estimated_price") or 0.0)
        for row in replacements
    )
    summary = {
        "schema": "kitchen_stage_report/v1",
        "replacement_count": len(replacements),
        "total_estimated_price": round(total, 2),
        "items": replacements,
        "companion_targets": companion_targets,
        "companion_target_count": len(companion_targets),
        "separated_buckets": {
            "kitchen_set": [row.get("assembly_id") for row in replacements],
            "room_surfaces": ["wall_material_stage", "flooring_stage"],
            "windows_and_curtains": ["room.windows", "curtain_stage"],
            "tables_and_chairs": ["supplier_stage:table", "supplier_stage:chair"],
        },
    }
    _write_json(summary_path, summary)

    lines = ["# Kitchen stage", "", f"- replacements: {len(replacements)}", f"- total estimate: {round(total, 2)} RUB", ""]
    if companion_targets:
        lines.append("## Companion supplier targets")
        for target in companion_targets:
            lines.append(f"- {target.get('id')}: {target.get('role')} ({target.get('category')})")
        lines.append("")
    for row in replacements:
        lines.append(f"## {row.get('assembly_id')}")
        lines.append(f"- source target: {row.get('target_id')}")
        lines.append(f"- mode: {row.get('mode')}")
        lines.append(f"- layout: {row.get('layout_type')}")
        lines.append(f"- dimensions: {row.get('dimensions')}")
        lines.append(f"- price: {(row.get('price_estimate') or {}).get('total_estimated_price')} RUB")
        appliances = row.get("appliances") if isinstance(row.get("appliances"), dict) else {}
        if appliances:
            for role in ("sink", "faucet", "cooktop", "oven", "hood", "microwave", "small_kitchen_appliance"):
                appliance = appliances.get(role)
                if isinstance(appliance, dict):
                    lines.append(f"- {role}: {appliance.get('title') or 'not selected'}")
        if row.get("warnings"):
            lines.append(f"- warnings: {', '.join(str(x) for x in row.get('warnings') or [])}")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(row.get('assembly_id')))}</td>"
        f"<td>{html.escape(str(row.get('mode')))}</td>"
        f"<td>{html.escape(str(row.get('layout_type')))}</td>"
        f"<td>{html.escape(str((row.get('price_estimate') or {}).get('total_estimated_price')))}</td>"
        "</tr>"
        for row in replacements
    )
    html_path.write_text(
        "<!doctype html><meta charset='utf-8'><title>Kitchen stage</title>"
        "<style>body{font-family:system-ui,sans-serif;margin:24px}td,th{padding:6px 10px;border-bottom:1px solid #ddd;text-align:left}</style>"
        f"<h1>Kitchen stage</h1><p>Replacements: {len(replacements)}. Total: {round(total, 2)} RUB.</p>"
        "<table><thead><tr><th>Assembly</th><th>Mode</th><th>Layout</th><th>Price RUB</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>",
        encoding="utf-8",
    )
    return {
        "summary_json": str(summary_path.resolve()),
        "markdown": str(md_path.resolve()),
        "html": str(html_path.resolve()),
    }


def apply_kitchen_stage_to_artifacts(
    *,
    artifacts: PlacementArtifacts,
    run_dir: Path,
    room_json_path: Path,
    material_catalog: Path,
    appliance_catalog: Path | None,
    prompt_text: str,
    mode: str,
    policy: str,
    suffix: str,
    dining_policy: str = "auto",
    accessories_policy: str = "auto",
    accessory_llm_settings: dict[str, Any] | None = None,
    kitchen_llm_settings: dict[str, Any] | None = None,
) -> tuple[PlacementArtifacts, dict[str, Any] | None]:
    if policy == "never":
        return artifacts, None
    if not material_catalog.is_file():
        return artifacts, {"skipped_reason": "material_catalog_missing", "material_catalog": str(material_catalog)}
    inventory_index = _load_kitchen_selection_index(appliance_catalog)

    scene_data = _read_json(artifacts.scene_v1) if artifacts.scene_v1 and artifacts.scene_v1.is_file() else None
    if scene_data is not None and not isinstance(scene_data, dict):
        scene_data = None
    room = _room_dict(scene_data, room_json_path)
    add_if_missing = policy == "always" or (policy == "auto" and _room_is_kitchen(room, prompt_text))
    is_kitchen_room = _room_is_kitchen(room, prompt_text)
    add_dining = dining_policy == "always" or (dining_policy == "auto" and is_kitchen_room)
    add_accessories = accessories_policy == "always" or (accessories_policy == "auto" and is_kitchen_room)

    placement_data = _read_json(artifacts.placement_v1)
    if not isinstance(placement_data, dict):
        return artifacts, None
    placement_out, replacements, companion_additions = _replace_kitchens_in_doc(
        placement_data,
        room=room,
        material_catalog=material_catalog,
        appliance_catalog=appliance_catalog,
        inventory_index=inventory_index,
        prompt_text=prompt_text,
        mode=mode,
        add_if_missing=add_if_missing,
        add_dining=add_dining,
        add_accessories=add_accessories,
        accessory_llm_settings=accessory_llm_settings,
        kitchen_llm_settings=kitchen_llm_settings,
    )
    if not replacements:
        return artifacts, {"replacement_count": 0, "skipped_reason": "no_kitchen_targets"}

    safe_suffix = f".{suffix.strip('.')}" if suffix.strip(".") else ""
    placement_path = run_dir / f"placement_kitchen{safe_suffix}.v1.json"
    _write_json(placement_path, placement_out)

    scene_path = artifacts.scene_v1
    if scene_data is not None and artifacts.scene_v1:
        scene_out, _, _ = _replace_kitchens_in_doc(
            scene_data,
            room=room,
            material_catalog=material_catalog,
            appliance_catalog=appliance_catalog,
            inventory_index=inventory_index,
            prompt_text=prompt_text,
            mode=mode,
            add_if_missing=add_if_missing,
            add_dining=add_dining,
            add_accessories=add_accessories,
            accessory_llm_settings=accessory_llm_settings,
            kitchen_llm_settings=kitchen_llm_settings,
        )
        scene_path = run_dir / f"scene_kitchen{safe_suffix}.v1.json"
        _write_json(scene_path, scene_out)

    reports = write_kitchen_report(run_dir, replacements, companion_additions, suffix=suffix)
    info = {
        "replacement_count": len(replacements),
        "placement_v1": str(placement_path.resolve()),
        "scene_v1": str(scene_path.resolve()) if scene_path else None,
        "mode": mode,
        "companion_targets": companion_additions,
        "companion_target_count": len(companion_additions),
        "reports": reports,
        "items": replacements,
    }
    return (
        PlacementArtifacts(
            placement_legacy=artifacts.placement_legacy,
            placement_v1=placement_path,
            scene_v1=scene_path,
            scene_legacy=artifacts.scene_legacy,
        ),
        info,
    )
