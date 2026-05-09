#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any


SANITARY_REQUIRED = ("toilet", "sink", "bath_or_shower")
SCENE_CANDIDATES = (
    "pipeline/optimal/scene_supplier.optimal.v1.flooring.v1.wall_material.v1.json",
    "pipeline/optimal/scene_supplier.optimal.v1.flooring.v1.json",
    "pipeline/optimal/scene_supplier.optimal.v1.json",
    "pipeline/optimal/scene.v1.flooring.v1.wall_material.v1.json",
    "pipeline/optimal/scene.v1.json",
)
SUPPORTED_MESH_SUFFIXES = {".fbx", ".obj", ".glb", ".gltf"}
SUPPLIER_CATALOG_PATH = Path("data/sourse/suppliers/supplier_catalog_canonical.json")
LOCAL_TABLE_ASSET_ROOT = Path("data/sourse/imodern")
_CATALOG_CACHE: dict[tuple[str, ...], list[dict[str, Any]]] = {}

ROLE_CATEGORY = {
    "toilet": "ToiletFactory",
    "sink": "StandingSinkFactory",
    "shower": "ShowerFactory",
    "bath": "BathtubFactory",
    "bed": "BedFactory",
    "table": "SimpleDeskFactory",
    "flat_ceiling_light": "CeilingLightFactory",
}
ROLE_SEMANTIC_GROUP = {
    "toilet": "toilet",
    "sink": "bathroom_sink",
    "shower": "shower",
    "bath": "bathtub",
    "bed": "bed",
    "table": "dining_table",
    "flat_ceiling_light": "lamp_ceiling",
}
ROLE_CATEGORY_NORMS = {
    "toilet": {"toilet", "toilet_bidet"},
    "sink": {"bathroom_sink", "washbasin"},
    "shower": {"shower", "shower_cabin", "shower_system"},
    "bath": {"bathtub", "bath"},
    "bed": {"bed"},
    "table": {"dining_table", "desk", "table"},
}

DISCOURAGED_SUPPLIER_KEY_TOKENS = {
    # This model contains several shower-cabin variants in one asset and looks
    # like three cabins placed together after fitting into a small bathroom.
    "shower": {"ag01090", "schwarzer_diamant", "schwarzer diamant"},
}


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, data: Any) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def norm(value: Any) -> str:
    return str(value or "").replace("ё", "е").lower()


def item_text(item: dict[str, Any]) -> str:
    source = item.get("source") if isinstance(item.get("source"), dict) else {}
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    supplier = meta.get("supplier_candidate") if isinstance(meta.get("supplier_candidate"), dict) else {}
    return norm(
        " ".join(
            str(x or "")
            for x in (
                item.get("id"),
                item.get("name"),
                item.get("category"),
                item.get("semantic_group"),
                source.get("supplier_unique_key"),
                supplier.get("title"),
                supplier.get("category_norm"),
                supplier.get("category_raw"),
            )
        )
    )


def classify_item(item: dict[str, Any]) -> set[str]:
    text = item_text(item)
    out: set[str] = set()
    if any(x in text for x in ("toilet", "унитаз", "wc", "watercloset")):
        out.add("toilet")
    if any(x in text for x in ("standing sink", "bathroom_sink", "washbasin", "basin", "sink", "раковин", "умывальник")):
        out.add("sink")
    if any(x in text for x in ("bathtub", "bath tub", "bathfactory", "ванн")):
        out.add("bath")
        out.add("bath_or_shower")
    if any(x in text for x in ("shower", "душ", "душев")):
        out.add("shower")
        out.add("bath_or_shower")
    if any(x in text for x in ("bedfactory", " bed", "кровать")):
        out.add("bed")
    is_lamp = any(x in text for x in ("lamp", "light", "люстр", "светиль", "ламп"))
    if (not is_lamp) and any(
        x in text
        for x in (
            "tablefactory",
            "simpledeskfactory",
            "deskfactory",
            "dining_table",
            "coffee_table",
            "side_table",
            "стол",
            "desk",
            "table",
        )
    ):
        out.add("table")
    return out


def room_items(scene: dict[str, Any]) -> list[dict[str, Any]]:
    items = scene.get("placements") or scene.get("items") or []
    return [x for x in items if isinstance(x, dict)]


def room_bounds(room: dict[str, Any]) -> tuple[float, float]:
    width = float(room.get("width_m") or 0.0)
    depth = float(room.get("depth_m") or 0.0)
    if width > 0 and depth > 0:
        return width, depth
    poly = room.get("floor_polygon") or []
    xs = [float(p.get("x", 0.0)) for p in poly if isinstance(p, dict)]
    ys = [float(p.get("y", p.get("z", 0.0))) for p in poly if isinstance(p, dict)]
    return (max(xs) - min(xs), max(ys) - min(ys)) if xs and ys else (3.0, 3.0)


def _supported_mesh(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_MESH_SUFFIXES


def _find_preferred_mesh(root: Path) -> Path | None:
    if not root.exists():
        return None
    files = [p for p in root.rglob("*") if _supported_mesh(p)]
    if not files:
        return None
    order = {".fbx": 0, ".obj": 1, ".glb": 2, ".gltf": 3}
    files.sort(key=lambda p: (order.get(p.suffix.lower(), 99), len(p.parts), str(p).lower()))
    return files[0]


def _normalize_catalog_candidate(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    dims = out.get("dimensions_cm") if isinstance(out.get("dimensions_cm"), dict) else {}
    for src_key, dst_key in (("width", "width_cm"), ("depth", "depth_cm"), ("height", "height_cm")):
        if out.get(dst_key) is None and dims.get(src_key) is not None:
            out[dst_key] = dims.get(src_key)
    mesh_path = str(out.get("asset_local_path") or out.get("mesh_local_path") or "").strip()
    if mesh_path and Path(mesh_path).expanduser().is_file():
        out["asset_local_path"] = str(Path(mesh_path).expanduser().resolve())
        out["asset_format"] = out.get("asset_format") or Path(mesh_path).suffix.lstrip(".").lower()
        out["asset_status"] = out.get("asset_status") or "local_supplier_asset"
    category_norm = norm(out.get("category_norm"))
    if not str(out.get("semantic_group") or "").strip():
        if category_norm in {"toilet", "toilet_bidet"}:
            out["semantic_group"] = "toilet"
        elif category_norm in {"bathroom_sink", "washbasin"}:
            out["semantic_group"] = "bathroom_sink"
        elif category_norm in {"shower", "shower_cabin", "shower_system"}:
            out["semantic_group"] = "shower"
        elif category_norm in {"bathtub", "bath"}:
            out["semantic_group"] = "bathtub"
        elif category_norm in {"dining_table", "desk", "table"}:
            out["semantic_group"] = "dining_table" if category_norm != "desk" else "desk"
        elif category_norm == "bed":
            out["semantic_group"] = "bed"
    return out


def _candidate_has_local_mesh(candidate: dict[str, Any]) -> bool:
    mesh_path = str(candidate.get("asset_local_path") or candidate.get("mesh_local_path") or "").strip()
    return bool(mesh_path and _supported_mesh(Path(mesh_path).expanduser()))


def _metadata_candidates_from_roots(search_roots: tuple[Path, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in search_roots:
        if not root.exists() or root.name == "imodern":
            continue
        for meta_path in root.rglob("*.metadata.json"):
            if "supplier_assets" not in str(meta_path):
                continue
            try:
                row = read_json(meta_path)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            mesh = _find_preferred_mesh(meta_path.parent)
            if mesh is None:
                continue
            row = _normalize_catalog_candidate(row)
            row["asset_local_path"] = str(mesh.resolve())
            row["asset_format"] = mesh.suffix.lstrip(".").lower()
            row["asset_status"] = row.get("asset_status") or "local_supplier_asset_cache"
            rows.append(row)
    return rows


def _color_from_title(text: str) -> str | None:
    tokens = []
    low = norm(text)
    for color in (
        "белый",
        "белая",
        "серый",
        "серая",
        "черный",
        "черная",
        "коричневый",
        "орех",
        "бежевый",
        "бронза",
        "светлая",
        "темная",
    ):
        if color in low:
            tokens.append(color)
    return " ".join(tokens) or None


def _local_table_candidates() -> list[dict[str, Any]]:
    root = LOCAL_TABLE_ASSET_ROOT
    if not root.exists():
        return []
    rows: list[dict[str, Any]] = []
    for folder in root.iterdir():
        if not folder.is_dir():
            continue
        text = norm(folder.name)
        if ("стол" not in text and "table" not in text) or any(x in text for x in ("лампа", "lamp")):
            continue
        mesh = _find_preferred_mesh(folder)
        if mesh is None:
            continue
        if "журн" in text or "coffee" in text:
            category_norm = "coffee_table"
            semantic_group = "coffee_table"
        elif "рабоч" in text or "письмен" in text or "desk" in text:
            category_norm = "desk"
            semantic_group = "desk"
        else:
            category_norm = "dining_table"
            semantic_group = "dining_table"
        if category_norm == "coffee_table":
            continue
        nums = [float(x) for x in re.findall(r"(?<!\d)(\d{2,3})(?!\d)", folder.name)]
        width_cm = nums[0] if nums else 140.0
        depth_cm = nums[1] if len(nums) > 1 and nums[1] <= 120.0 else 80.0
        height_cm = 76.0
        title = folder.name.replace("_", " ")
        rows.append(
            {
                "unique_key": f"local_imodern::{folder.name}",
                "source_site": "imodern_local",
                "title": title,
                "category_raw": "Столы",
                "category_norm": category_norm,
                "semantic_group": semantic_group,
                "asset_status": "local_supplier_asset",
                "asset_format": mesh.suffix.lstrip(".").lower(),
                "asset_local_path": str(mesh.resolve()),
                "style": "современный",
                "color": _color_from_title(title),
                "width_cm": width_cm,
                "depth_cm": depth_cm,
                "height_cm": height_cm,
                "description": title,
            }
        )
    return rows


def load_catalog_candidates(search_roots: tuple[Path, ...]) -> list[dict[str, Any]]:
    key = tuple(str(p.expanduser().resolve()) for p in search_roots)
    cached = _CATALOG_CACHE.get(key)
    if cached is not None:
        return cached
    rows: list[dict[str, Any]] = []
    if SUPPLIER_CATALOG_PATH.is_file():
        try:
            payload = read_json(SUPPLIER_CATALOG_PATH)
            items = payload.get("items") if isinstance(payload, dict) else []
            rows.extend(_normalize_catalog_candidate(x) for x in items if isinstance(x, dict))
        except Exception:
            pass
    rows.extend(_metadata_candidates_from_roots(search_roots))
    rows.extend(_local_table_candidates())
    _CATALOG_CACHE[key] = rows
    return rows


def _candidate_dims_m(candidate: dict[str, Any], fallback: tuple[float, float, float]) -> tuple[float, float, float]:
    try:
        width = float(candidate.get("width_cm") or 0.0) / 100.0
        depth = float(candidate.get("depth_cm") or 0.0) / 100.0
        height = float(candidate.get("height_cm") or 0.0) / 100.0
    except Exception:
        width = depth = height = 0.0
    fw, fd, fh = fallback
    return (width if width > 0 else fw, depth if depth > 0 else fd, height if height > 0 else fh)


def _text_tokens(value: Any) -> set[str]:
    return {x for x in re.split(r"[^0-9a-zа-я]+", norm(value)) if len(x) > 2}


def _scene_style_tokens(scene: dict[str, Any], prompt_room_type: str | None = None) -> set[str]:
    parts: list[str] = [prompt_room_type or ""]
    room = scene.get("room") if isinstance(scene.get("room"), dict) else {}
    meta = scene.get("meta") if isinstance(scene.get("meta"), dict) else {}
    room_meta = room.get("meta") if isinstance(room.get("meta"), dict) else {}
    for source in (room, meta, room_meta):
        for key in ("style", "color", "materials", "description", "prompt", "prompt_text", "source"):
            value = source.get(key)
            if isinstance(value, (str, int, float)):
                parts.append(str(value))
    return _text_tokens(" ".join(parts))


def _candidate_role_match(candidate: dict[str, Any], role: str) -> bool:
    category_norm = norm(candidate.get("category_norm"))
    semantic_group = norm(candidate.get("semantic_group"))
    accepted = ROLE_CATEGORY_NORMS.get(role, {role})
    if category_norm in accepted or semantic_group in accepted:
        return True
    if role == "toilet" and "toilet" in semantic_group:
        return True
    if role == "sink" and semantic_group == "bathroom_sink":
        return True
    if role == "shower" and semantic_group == "shower":
        return True
    if role == "table" and semantic_group in {"dining_table", "desk"}:
        return True
    return False


def _candidate_identity_text(candidate: dict[str, Any]) -> str:
    return norm(
        " ".join(
            str(candidate.get(key) or "")
            for key in (
                "unique_key",
                "title",
                "product_url",
                "model_page_url",
                "model_download_url",
                "asset_local_path",
                "description",
            )
        )
    )


def _candidate_discouraged_for_role(candidate: dict[str, Any], role: str) -> bool:
    tokens = DISCOURAGED_SUPPLIER_KEY_TOKENS.get(role) or set()
    if not tokens:
        return False
    text = _candidate_identity_text(candidate)
    return any(token in text for token in tokens)


def select_catalog_candidate(
    role: str,
    target_size: tuple[float, float, float],
    scene: dict[str, Any],
    prompt_room_type: str | None,
    search_roots: tuple[Path, ...],
) -> dict[str, Any] | None:
    scene_tokens = _scene_style_tokens(scene, prompt_room_type=prompt_room_type)
    best: tuple[float, dict[str, Any]] | None = None
    for candidate in load_catalog_candidates(search_roots):
        if not _candidate_role_match(candidate, role):
            continue
        if _candidate_discouraged_for_role(candidate, role):
            continue
        if not _candidate_has_local_mesh(candidate):
            continue
        category_norm = norm(candidate.get("category_norm"))
        if role == "table" and category_norm not in {"dining_table", "desk", "table"}:
            continue
        cw, cd, ch = _candidate_dims_m(candidate, target_size)
        tw, td, th = target_size
        normal = abs(math.log(max(cw, 0.02) / max(tw, 0.02))) + abs(math.log(max(cd, 0.02) / max(td, 0.02))) + 0.55 * abs(math.log(max(ch, 0.02) / max(th, 0.02)))
        swapped = abs(math.log(max(cd, 0.02) / max(tw, 0.02))) + abs(math.log(max(cw, 0.02) / max(td, 0.02))) + 0.55 * abs(math.log(max(ch, 0.02) / max(th, 0.02)))
        size_score = min(normal, swapped)
        candidate_tokens = _text_tokens(
            " ".join(
                str(candidate.get(key) or "")
                for key in ("title", "category_raw", "category_norm", "semantic_group", "style", "color", "materials", "description")
            )
        )
        style_overlap = len(scene_tokens & candidate_tokens)
        category_bonus = -0.55 if norm(candidate.get("semantic_group")) == ROLE_SEMANTIC_GROUP.get(role) else 0.0
        if role == "table" and category_norm == "dining_table":
            category_bonus -= 0.35
        if role == "shower":
            if category_norm == "shower_system":
                category_bonus -= 0.75
            elif category_norm == "shower_cabin":
                category_bonus += 0.45
        ready_bonus = -0.15 if str(candidate.get("asset_status") or "").startswith("local") else 0.0
        score = size_score + category_bonus + ready_bonus - min(style_overlap, 5) * 0.04
        candidate = dict(candidate)
        candidate["requirement_match_score"] = round(score, 6)
        candidate["requirement_size_score"] = round(size_score, 6)
        candidate["requirement_style_overlap"] = sorted(scene_tokens & candidate_tokens)
        if best is None or score < best[0]:
            best = (score, candidate)
    return best[1] if best is not None else None


def _compact_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "unique_key",
        "source_site",
        "title",
        "brand",
        "collection",
        "category_raw",
        "category_norm",
        "semantic_group",
        "product_url",
        "model_page_url",
        "model_download_url",
        "model_download_landing_url",
        "model_vendor_url",
        "asset_status",
        "asset_format",
        "asset_local_path",
        "price_value",
        "price_currency",
        "style",
        "color",
        "materials",
        "width_cm",
        "depth_cm",
        "height_cm",
        "description",
        "requirement_match_score",
        "requirement_size_score",
        "requirement_style_overlap",
    ]
    return {k: deepcopy(candidate.get(k)) for k in keys if k in candidate}


def _fit_catalog_size_to_room(role: str, candidate: dict[str, Any], target_size: tuple[float, float, float], room_size: tuple[float, float]) -> tuple[float, float, float]:
    sx, sy, sz = _candidate_dims_m(candidate, target_size)
    width, depth = room_size
    category_norm = norm(candidate.get("category_norm"))
    if role == "shower" and category_norm == "shower_system":
        return 0.32, 0.18, max(1.45, min(sz, 1.95))
    max_w = max(0.25, width - 0.24)
    max_d = max(0.25, depth - 0.24)
    if role == "shower" and min(width, depth) < 1.65:
        max_w = min(max_w, max(0.55, width * 0.58))
        max_d = min(max_d, max(0.55, depth * 0.58))
    scale = min(1.0, max_w / max(sx, 1e-6), max_d / max(sy, 1e-6))
    sx, sy, sz = sx * scale, sy * scale, sz * scale
    if role == "sink" and sz < 0.24:
        sz = max(0.12, sz)
    return max(0.12, sx), max(0.12, sy), max(0.08, sz)


def _required_item_constraints(role: str, z_min: float) -> dict[str, Any]:
    if role == "sink" and z_min > 0.05:
        return {"mount_type": "wall"}
    return {"mount_type": "floor", "touch_floor": {"side": "bottom"}}


def make_supplier_required_item(
    *,
    room_id: str,
    role: str,
    index: int,
    center_xy: tuple[float, float],
    size: tuple[float, float, float],
    yaw_deg: float,
    z_min: float,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    sx, sy, sz = size
    cx, cy = center_xy
    item_id = f"req_{role}_{index:02d}"
    mesh_path = str(candidate.get("asset_local_path") or "").strip()
    semantic_group = ROLE_SEMANTIC_GROUP.get(role, role)
    return {
        "id": item_id,
        "name": str(candidate.get("title") or f"supplier {role}"),
        "category": ROLE_CATEGORY.get(role, "SupplierObject"),
        "semantic_group": semantic_group,
        "position_m": [round(cx, 4), round(cy, 4), round(z_min + sz / 2.0, 4)],
        "size_m": [round(sx, 4), round(sy, 4), round(sz, 4)],
        "yaw_deg": round(yaw_deg, 4),
        "rotation_deg": round(yaw_deg, 4),
        "yaw_rad": round(math.radians(yaw_deg), 8),
        "aabb": {
            "x_min": round(cx - sx / 2.0, 4),
            "x_max": round(cx + sx / 2.0, 4),
            "y_min": round(cy - sy / 2.0, 4),
            "y_max": round(cy + sy / 2.0, 4),
            "z_min": round(z_min, 4),
            "z_max": round(z_min + sz, 4),
        },
        "constraints": _required_item_constraints(role, z_min),
        "asset": {"mesh_path": mesh_path, "mesh_fit_mode": "uniform"},
        "source": {
            "placement_source": "requirement_postprocess",
            "asset_source": "supplier_catalog_local_asset",
            "supplier_replaced": True,
            "supplier_target_id": item_id,
            "supplier_unique_key": candidate.get("unique_key"),
            "supplier_source_site": candidate.get("source_site"),
            "supplier_product_url": candidate.get("product_url") or candidate.get("model_page_url"),
            "supplier_model_url": candidate.get("model_download_url") or candidate.get("model_download_landing_url"),
            "placeholder_bbox": False,
            "room_id": room_id,
        },
        "meta": {
            "placeholder_bbox": False,
            "supplier_binding_applied": True,
            "supplier_requirement_added": True,
            "required_role": role,
            "room_id": room_id,
            "supplier_candidate": _compact_candidate(candidate),
            "supplier_candidate_pool": [_compact_candidate(candidate)],
        },
    }


def _record_missing_catalog_asset(scene: dict[str, Any], role: str) -> None:
    meta = scene.setdefault("meta", {})
    req = meta.setdefault("requirement_postprocess", {})
    req.setdefault("missing_catalog_asset", []).append(role)


def _clamp_center(center: tuple[float, float], size: tuple[float, float], room_size: tuple[float, float], margin: float) -> tuple[float, float]:
    sx, sy = size
    width, depth = room_size
    min_x = margin + sx / 2.0
    max_x = max(min_x, width - margin - sx / 2.0)
    min_y = margin + sy / 2.0
    max_y = max(min_y, depth - margin - sy / 2.0)
    return (
        min(max(center[0], min_x), max_x),
        min(max(center[1], min_y), max_y),
    )


def _primary_door_side(room: dict[str, Any], room_size: tuple[float, float]) -> str | None:
    width, depth = room_size
    doors = room.get("doors") if isinstance(room.get("doors"), list) else []
    if not doors:
        return None
    seg = (doors[0] or {}).get("segment") if isinstance(doors[0], dict) else {}
    if not isinstance(seg, dict):
        return None
    try:
        x1, x2 = float(seg.get("x1")), float(seg.get("x2"))
        y1, y2 = float(seg.get("y1")), float(seg.get("y2"))
    except Exception:
        return None
    if abs(x1 - x2) < abs(y1 - y2):
        return "left" if (x1 + x2) * 0.5 < width * 0.5 else "right"
    return "bottom" if (y1 + y2) * 0.5 < depth * 0.5 else "top"


def _sanitary_layout(
    role: str,
    size: tuple[float, float],
    room: dict[str, Any],
    margin: float,
) -> tuple[tuple[float, float], float]:
    sx, sy = size
    room_size = room_bounds(room)
    width, depth = room_size
    door_side = _primary_door_side(room, room_size)
    if role == "toilet":
        if door_side == "bottom":
            return _clamp_center((width * 0.5, depth - margin - sy / 2.0), size, room_size, margin), 180.0
        if door_side == "top":
            return _clamp_center((width * 0.5, margin + sy / 2.0), size, room_size, margin), 0.0
        if door_side == "right":
            return _clamp_center((margin + sx / 2.0, depth * 0.5), size, room_size, margin), 90.0
        if door_side == "left":
            return _clamp_center((width - margin - sx / 2.0, depth * 0.5), size, room_size, margin), 270.0
    if role == "sink":
        if door_side == "right":
            return _clamp_center((margin + sx / 2.0, depth * 0.48), size, room_size, margin), 90.0
        if door_side == "left":
            return _clamp_center((width - margin - sx / 2.0, depth * 0.48), size, room_size, margin), 270.0
        if door_side == "bottom":
            return _clamp_center((width * 0.5, depth - margin - sy / 2.0), size, room_size, margin), 180.0
        if door_side == "top":
            return _clamp_center((width * 0.5, margin + sy / 2.0), size, room_size, margin), 0.0
    if role == "shower":
        if door_side == "right":
            return _clamp_center((margin + sx / 2.0, depth - margin - sy / 2.0), size, room_size, margin), 180.0
        if door_side == "left":
            return _clamp_center((width - margin - sx / 2.0, depth - margin - sy / 2.0), size, room_size, margin), 180.0
        if door_side == "bottom":
            return _clamp_center((margin + sx / 2.0, depth - margin - sy / 2.0), size, room_size, margin), 180.0
        if door_side == "top":
            return _clamp_center((margin + sx / 2.0, margin + sy / 2.0), size, room_size, margin), 0.0
    defaults = {
        "toilet": (min(width - 0.33, max(0.33, width * 0.28)), margin + 0.34),
        "sink": (max(0.34, width * 0.50), min(depth - 0.21, max(0.35, depth * 0.50))),
        "shower": (max(0.53, width - 0.53), max(0.53, depth - 0.53)),
    }
    return _clamp_center(defaults.get(role, (width * 0.5, depth * 0.5)), size, room_size, margin), 0.0


def _set_item_geometry(
    item: dict[str, Any],
    *,
    center_xy: tuple[float, float],
    size: tuple[float, float, float],
    yaw_deg: float,
    z_min: float,
    role: str | None = None,
) -> None:
    sx, sy, sz = size
    cx, cy = center_xy
    item["position_m"] = [round(cx, 4), round(cy, 4), round(z_min + sz / 2.0, 4)]
    item["size_m"] = [round(sx, 4), round(sy, 4), round(sz, 4)]
    item["yaw_deg"] = round(yaw_deg, 4)
    item["rotation_deg"] = round(yaw_deg, 4)
    item["yaw_rad"] = round(math.radians(yaw_deg), 8)
    item["aabb"] = {
        "x_min": round(cx - sx / 2.0, 4),
        "x_max": round(cx + sx / 2.0, 4),
        "y_min": round(cy - sy / 2.0, 4),
        "y_max": round(cy + sy / 2.0, 4),
        "z_min": round(z_min, 4),
        "z_max": round(z_min + sz, 4),
    }
    if role:
        item["semantic_group"] = ROLE_SEMANTIC_GROUP.get(role, item.get("semantic_group") or role)
        item["constraints"] = _required_item_constraints(role, z_min)
        meta = item.setdefault("meta", {})
        meta["sanitary_layout_repaired"] = True
        meta["required_role"] = role
        source = item.setdefault("source", {})
        source["placeholder_bbox"] = False


def _remove_items(scene: dict[str, Any], predicate) -> list[dict[str, Any]]:
    placements = scene.get("placements")
    if not isinstance(placements, list):
        return []
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    for item in placements:
        if isinstance(item, dict) and predicate(item):
            removed.append(item)
        else:
            kept.append(item)
    scene["placements"] = kept
    return removed


def _is_ceiling_light_item(item: dict[str, Any]) -> bool:
    text = item_text(item)
    category = norm(item.get("category"))
    semantic = norm(item.get("semantic_group"))
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    candidate = meta.get("supplier_candidate") if isinstance(meta.get("supplier_candidate"), dict) else {}
    cand_semantic = norm(candidate.get("semantic_group"))
    cand_category = norm(candidate.get("category_norm"))
    return (
        semantic == "lamp_ceiling"
        or cand_semantic == "lamp_ceiling"
        or "ceilinglightfactory" in category
        or "ceiling light" in text
        or "chandelier" in text
        or "люстр" in text
        or ("потолоч" in text and ("светиль" in text or "light" in text or "lamp" in text))
        or cand_category in {"chandelier", "ceiling_lamp", "pendant_lamp", "recessed_spot_track_light"}
    )


def _room_ceiling_height(room: dict[str, Any]) -> float:
    try:
        return float(room.get("ceiling_height_m") or room.get("ceiling_height") or 2.8)
    except Exception:
        return 2.8


def _ceiling_coverage_centers(room: dict[str, Any], count: int) -> list[tuple[float, float]]:
    width, depth = room_bounds(room)
    count = max(1, int(count))
    margin = min(0.55, max(0.18, min(width, depth) * 0.22))
    if count == 1:
        return [(width * 0.5, depth * 0.5)]
    centers: list[tuple[float, float]] = []
    if width >= depth:
        usable = max(0.01, width - margin * 2.0)
        for idx in range(count):
            t = (idx + 1) / (count + 1)
            centers.append((margin + usable * t, depth * 0.5))
    else:
        usable = max(0.01, depth - margin * 2.0)
        for idx in range(count):
            t = (idx + 1) / (count + 1)
            centers.append((width * 0.5, margin + usable * t))
    return centers


def _set_ceiling_light_geometry(item: dict[str, Any], room: dict[str, Any], center_xy: tuple[float, float], size_xy: tuple[float, float] | None = None) -> None:
    aabb = item.get("aabb") if isinstance(item.get("aabb"), dict) else {}
    if size_xy is None:
        sx = max(0.12, float(aabb.get("x_max", 0.0)) - float(aabb.get("x_min", 0.0)) if aabb else 0.28)
        sy = max(0.12, float(aabb.get("y_max", 0.0)) - float(aabb.get("y_min", 0.0)) if aabb else 0.28)
    else:
        sx, sy = size_xy
    sz = max(0.035, min(0.16, float(aabb.get("z_max", 0.0)) - float(aabb.get("z_min", 0.0)) if aabb else 0.055))
    z_max = _room_ceiling_height(room) - 0.01
    _set_item_geometry(
        item,
        center_xy=center_xy,
        size=(sx, sy, sz),
        yaw_deg=0.0,
        z_min=z_max - sz,
        role="flat_ceiling_light",
    )
    item["category"] = "CeilingLightFactory"
    item["semantic_group"] = "lamp_ceiling"
    item.setdefault("meta", {})["ceiling_light_position_repaired"] = True


def make_flat_ceiling_light_item(room_id: str, index: int, room: dict[str, Any], center_xy: tuple[float, float]) -> dict[str, Any]:
    width, depth = room_bounds(room)
    diameter = min(0.32, max(0.20, min(width, depth) * 0.18))
    height = 0.045
    z_max = _room_ceiling_height(room) - 0.012
    z_min = z_max - height
    cx, cy = center_xy
    return {
        "id": f"req_flat_ceiling_light_{index:02d}",
        "name": "Плоский потолочный светильник",
        "category": "CeilingLightFactory",
        "semantic_group": "lamp_ceiling",
        "position_m": [round(cx, 4), round(cy, 4), round(z_min + height / 2.0, 4)],
        "size_m": [round(diameter, 4), round(diameter, 4), round(height, 4)],
        "yaw_deg": 0.0,
        "rotation_deg": 0.0,
        "yaw_rad": 0.0,
        "aabb": {
            "x_min": round(cx - diameter / 2.0, 4),
            "x_max": round(cx + diameter / 2.0, 4),
            "y_min": round(cy - diameter / 2.0, 4),
            "y_max": round(cy + diameter / 2.0, 4),
            "z_min": round(z_min, 4),
            "z_max": round(z_max, 4),
        },
        "constraints": {"mount_type": "ceiling", "under_ceiling": True},
        "asset": {"kind": "procedural_flat_ceiling_light", "mesh_fit_mode": "exact"},
        "source": {
            "placement_source": "requirement_postprocess",
            "asset_source": "procedural_lighting",
            "supplier_replaced": False,
            "supplier_target_id": f"req_flat_ceiling_light_{index:02d}",
            "placeholder_bbox": False,
            "room_id": room_id,
        },
        "meta": {
            "placeholder_bbox": False,
            "procedural_lighting": True,
            "required_role": "flat_ceiling_light",
            "room_id": room_id,
            "ceiling_light_position_repaired": True,
            "sanitary_flat_light": True,
        },
    }


def repair_ceiling_lighting_layouts(scene_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    repairs: list[dict[str, Any]] = []
    for entry in scene_entries:
        scene = entry.get("scene") if isinstance(entry.get("scene"), dict) else {}
        if not scene:
            continue
        room = scene.get("room") if isinstance(scene.get("room"), dict) else {}
        room_id = str(room.get("id") or entry.get("room_id") or "room")
        text = _room_type_text(scene, _sanitary_entry_prompt(entry))
        is_sanitary = any(token in text for token in ("bathroom", "toilet", "сануз", "ванн", "туалет"))
        ceiling_lights = [item for item in room_items(scene) if _is_ceiling_light_item(item)]
        if is_sanitary:
            removed = _remove_items(scene, lambda item: _is_ceiling_light_item(item))
            if removed or not any(_is_ceiling_light_item(item) for item in room_items(scene)):
                center = _ceiling_coverage_centers(room, 1)[0]
                item = make_flat_ceiling_light_item(room_id, len(room_items(scene)) + 1, room, center)
                scene.setdefault("placements", []).append(item)
                entry.setdefault("added", []).append(item)
                repairs.append(
                    {
                        "room_id": room_id,
                        "action": "replaced_sanitary_chandelier_with_flat_light",
                        "removed_ids": [str(x.get("id") or "") for x in removed],
                        "added_id": item["id"],
                        "center_xy": [round(center[0], 4), round(center[1], 4)],
                    }
                )
            continue

        if not ceiling_lights:
            continue
        centers = _ceiling_coverage_centers(room, len(ceiling_lights))
        for item, center in zip(ceiling_lights, centers):
            old_pos = item.get("position_m")
            _set_ceiling_light_geometry(item, room, center)
            repairs.append(
                {
                    "room_id": room_id,
                    "action": "normalized_ceiling_light_position",
                    "id": item.get("id"),
                    "old_position_m": old_pos,
                    "new_position_m": item.get("position_m"),
                }
            )
        scene.setdefault("meta", {}).setdefault("requirement_postprocess", {})["ceiling_lighting_repaired"] = True
    return repairs


def repair_sanitary_layouts(
    scene_entries: list[dict[str, Any]],
    asset_search_roots: tuple[Path, ...] = (),
) -> list[dict[str, Any]]:
    repairs: list[dict[str, Any]] = []
    for entry in scene_entries:
        scene = entry.get("scene") if isinstance(entry.get("scene"), dict) else {}
        if not scene or not _is_sanitary_scene(scene, _sanitary_entry_prompt(entry)):
            continue
        room = scene.get("room") if isinstance(scene.get("room"), dict) else {}
        room_id = str(room.get("id") or entry.get("room_id") or "room")
        text = _room_type_text(scene, _sanitary_entry_prompt(entry))
        width, depth = room_bounds(room)
        margin = 0.12

        removed = _remove_items(
            scene,
            lambda item: (
                ("toilet" in text or "туалет" in text)
                and ("toilet" in classify_item(item))
                and bool((item.get("meta") or {}).get("supplier_requirement_added"))
            )
            or (
                ("shower" in classify_item(item))
                and (
                    bool((item.get("meta") or {}).get("supplier_requirement_added"))
                    or _candidate_discouraged_for_role((item.get("meta") or {}).get("supplier_candidate") or {}, "shower")
                    or "ag01090" in item_text(item)
                )
            ),
        )
        for item in removed:
            repairs.append({"room_id": room_id, "action": "removed_bad_sanitary_item", "id": item.get("id"), "name": item.get("name")})

        is_toilet_only_room = "toilet" in text or "туалет" in text
        is_bathroom_room = ("bathroom" in text or "ванн" in text or "сануз" in text) and not is_toilet_only_room

        if is_toilet_only_room:
            target_size = (0.48, 0.72, 0.8)
            candidate = select_catalog_candidate("toilet", target_size, scene, _sanitary_entry_prompt(entry), asset_search_roots)
            if candidate is not None and "toilet" not in _sanitary_roles_present(scene):
                sx, sy, sz = _fit_catalog_size_to_room("toilet", candidate, target_size, (width, depth))
                sx, sy, sz = max(0.44, sx), max(0.68, sy), max(0.76, sz)
                center, yaw = _sanitary_layout("toilet", (sx, sy), room, margin)
                item = make_supplier_required_item(
                    room_id=room_id,
                    role="toilet",
                    index=len(room_items(scene)) + 1,
                    center_xy=center,
                    size=(sx, sy, sz),
                    yaw_deg=yaw,
                    z_min=0.0,
                    candidate=candidate,
                )
                scene.setdefault("placements", []).append(item)
                entry.setdefault("added", []).append(item)
                repairs.append({"room_id": room_id, "action": "replaced_toilet_layout", "id": item["id"], "center_xy": list(center), "yaw_deg": yaw})

        if is_bathroom_room:
            sink_items = [item for item in room_items(scene) if "sink" in classify_item(item)]
            if sink_items:
                sink = sink_items[0]
                sink_size = (0.40, min(0.72, max(0.56, depth * 0.42)), 0.22)
                center, yaw = _sanitary_layout("sink", (sink_size[0], sink_size[1]), room, margin)
                _set_item_geometry(
                    sink,
                    center_xy=center,
                    size=sink_size,
                    yaw_deg=yaw,
                    z_min=0.72,
                    role="sink",
                )
                repairs.append({"room_id": room_id, "action": "reanchored_sink_to_wall", "id": sink.get("id"), "center_xy": list(center), "yaw_deg": yaw})

            if not ({"bath", "shower", "bath_or_shower"} & _sanitary_roles_present(scene)):
                target_size = (0.32, 0.18, 1.6)
                candidate = select_catalog_candidate("shower", target_size, scene, _sanitary_entry_prompt(entry), asset_search_roots)
                if candidate is not None:
                    sx, sy, sz = _fit_catalog_size_to_room("shower", candidate, target_size, (width, depth))
                    center, yaw = _sanitary_layout("shower", (sx, sy), room, margin)
                    item = make_supplier_required_item(
                        room_id=room_id,
                        role="shower",
                        index=len(room_items(scene)) + 1,
                        center_xy=center,
                        size=(sx, sy, sz),
                        yaw_deg=yaw,
                        z_min=0.0,
                        candidate=candidate,
                    )
                    scene.setdefault("placements", []).append(item)
                    entry.setdefault("added", []).append(item)
                    repairs.append({"room_id": room_id, "action": "replaced_shower_with_compact_catalog_asset", "id": item["id"], "center_xy": list(center), "yaw_deg": yaw, "candidate": candidate.get("unique_key")})

        if repairs:
            scene.setdefault("meta", {}).setdefault("requirement_postprocess", {})["sanitary_layout_repaired"] = True
    return repairs


def _room_type_text(scene: dict[str, Any], prompt_room_type: str | None = None) -> str:
    room = scene.get("room") if isinstance(scene.get("room"), dict) else {}
    return norm(" ".join([room.get("room_type") or "", room.get("source_room_type") or "", prompt_room_type or ""]))


def _is_sanitary_scene(scene: dict[str, Any], prompt_room_type: str | None = None) -> bool:
    return any(x in _room_type_text(scene, prompt_room_type) for x in ("bathroom", "toilet", "сануз", "ванн", "туалет"))


def _sanitary_roles_present(scene: dict[str, Any]) -> set[str]:
    present: set[str] = set()
    for item in room_items(scene):
        present |= classify_item(item)
    return present


def _actual_sanitary_role(role: str) -> str:
    return "shower" if role == "bath_or_shower" else role


def add_sanitary_roles_to_room(
    scene: dict[str, Any],
    roles: list[str],
    prompt_room_type: str | None = None,
    asset_search_roots: tuple[Path, ...] = (),
) -> list[dict[str, Any]]:
    room = scene.get("room") if isinstance(scene.get("room"), dict) else {}
    room_id = str(room.get("id") or "room")
    if not roles or not _is_sanitary_scene(scene, prompt_room_type):
        return []

    width, depth = room_bounds(room)
    margin = 0.12
    specs = {
        "toilet": ((0.48, 0.72, 0.8), 0.0, 0.0),
        "sink": ((0.40, 0.62, 0.22), 180.0, 0.72),
        "shower": ((0.32, 0.18, 1.6), 0.0, 0.0),
    }
    added: list[dict[str, Any]] = []
    for role in roles:
        actual_role = _actual_sanitary_role(role)
        target_size, yaw, z_min = specs[actual_role]
        candidate = select_catalog_candidate(actual_role, target_size, scene, prompt_room_type, asset_search_roots)
        if candidate is None:
            _record_missing_catalog_asset(scene, actual_role)
            continue
        sx, sy, sz = _fit_catalog_size_to_room(actual_role, candidate, target_size, (width, depth))
        (cx, cy), layout_yaw = _sanitary_layout(actual_role, (sx, sy), room, margin)
        added.append(
            make_supplier_required_item(
                room_id=room_id,
                role=actual_role,
                index=len(room_items(scene)) + len(added) + 1,
                center_xy=(cx, cy),
                size=(sx, sy, sz),
                yaw_deg=layout_yaw if layout_yaw is not None else yaw,
                z_min=z_min,
                candidate=candidate,
            )
        )
    scene.setdefault("placements", []).extend(added)
    meta = scene.setdefault("meta", {})
    req = meta.setdefault("requirement_postprocess", {})
    req.setdefault("added_sanitary", []).extend(x["id"] for x in added)
    return added


def add_missing_sanitary(
    scene: dict[str, Any],
    prompt_room_type: str | None = None,
    asset_search_roots: tuple[Path, ...] = (),
) -> list[dict[str, Any]]:
    if not _is_sanitary_scene(scene, prompt_room_type):
        return []
    present = _sanitary_roles_present(scene)
    missing = [role for role in SANITARY_REQUIRED if role not in present]
    return add_sanitary_roles_to_room(scene, missing, prompt_room_type, asset_search_roots)


def _sanitary_entry_prompt(entry: dict[str, Any]) -> str | None:
    room_meta = entry.get("room_meta") if isinstance(entry.get("room_meta"), dict) else {}
    value = room_meta.get("prompt_room_type")
    return str(value) if value is not None else None


def _sanitary_target_score(entry: dict[str, Any], role: str) -> tuple[float, float]:
    scene = entry.get("scene") if isinstance(entry.get("scene"), dict) else {}
    room = scene.get("room") if isinstance(scene.get("room"), dict) else {}
    text = _room_type_text(scene, _sanitary_entry_prompt(entry))
    width, depth = room_bounds(room)
    area = float(room.get("area_m2") or (width * depth))
    present = _sanitary_roles_present(scene)
    actual_role = _actual_sanitary_role(role)
    has_bath_or_shower = bool({"bath", "shower", "bath_or_shower"} & present)
    if actual_role == "shower":
        preference = 0.0 if ("bathroom" in text or "ванн" in text) else 2.0
        if "toilet" in text or "туалет" in text:
            preference += 4.0
        if has_bath_or_shower:
            preference += 3.0
        return preference, -area
    if actual_role == "toilet":
        preference = 0.0 if ("toilet" in text or "туалет" in text or "сануз" in text) else 1.0
        if has_bath_or_shower:
            preference += 1.5
        return preference, area
    if actual_role == "sink":
        preference = 0.0 if "toilet" in present else 1.0
        if "sink" in present:
            preference += 4.0
        return preference, area
    return 1.0, -area


def _select_sanitary_target_entry(entries: list[dict[str, Any]], role: str) -> dict[str, Any] | None:
    candidates = [
        entry
        for entry in entries
        if isinstance(entry.get("scene"), dict) and _is_sanitary_scene(entry["scene"], _sanitary_entry_prompt(entry))
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda entry: _sanitary_target_score(entry, role))[0]


def add_missing_sanitary_apartment(
    scene_entries: list[dict[str, Any]],
    asset_search_roots: tuple[Path, ...] = (),
) -> list[dict[str, Any]]:
    sanitary_entries = [
        entry
        for entry in scene_entries
        if isinstance(entry.get("scene"), dict) and _is_sanitary_scene(entry["scene"], _sanitary_entry_prompt(entry))
    ]
    if not sanitary_entries:
        return []
    present: set[str] = set()
    for entry in sanitary_entries:
        present |= _sanitary_roles_present(entry["scene"])
    missing = [role for role in SANITARY_REQUIRED if role not in present]
    added: list[dict[str, Any]] = []
    for role in missing:
        target = _select_sanitary_target_entry(sanitary_entries, role)
        if target is None:
            continue
        item_added = add_sanitary_roles_to_room(
            target["scene"],
            [role],
            prompt_room_type=_sanitary_entry_prompt(target),
            asset_search_roots=asset_search_roots,
        )
        target.setdefault("added", []).extend(item_added)
        added.extend(item_added)
    for entry in sanitary_entries:
        scene = entry["scene"]
        meta = scene.setdefault("meta", {})
        meta.setdefault("requirement_postprocess", {})["sanitary_scope"] = "apartment"
    return added


def add_missing_kitchen_table(
    scene: dict[str, Any],
    prompt_room_type: str | None = None,
    asset_search_roots: tuple[Path, ...] = (),
) -> list[dict[str, Any]]:
    room = scene.get("room") if isinstance(scene.get("room"), dict) else {}
    room_id = str(room.get("id") or "room")
    room_type_text = norm(" ".join([room.get("room_type") or "", room.get("source_room_type") or "", prompt_room_type or ""]))
    if "kitchen" not in room_type_text and "кух" not in room_type_text:
        return []

    present: set[str] = set()
    for item in room_items(scene):
        present |= classify_item(item)
    if "table" in present:
        return []

    width, depth = room_bounds(room)
    target_size = (min(1.25, max(0.75, width * 0.36)), min(0.78, max(0.55, depth * 0.24)), 0.76)
    candidate = select_catalog_candidate("table", target_size, scene, prompt_room_type, asset_search_roots)
    if candidate is None:
        _record_missing_catalog_asset(scene, "table")
        return []
    sx, sy, sz = _fit_catalog_size_to_room("table", candidate, target_size, (width, depth))
    cx = min(max(width * 0.62, sx / 2.0 + 0.18), max(sx / 2.0 + 0.18, width - sx / 2.0 - 0.18))
    cy = min(max(depth * 0.66, sy / 2.0 + 0.18), max(sy / 2.0 + 0.18, depth - sy / 2.0 - 0.18))
    item = make_supplier_required_item(
        room_id=room_id,
        role="table",
        index=len(room_items(scene)) + 1,
        center_xy=(cx, cy),
        size=(sx, sy, sz),
        yaw_deg=0.0,
        z_min=0.0,
        candidate=candidate,
    )
    scene.setdefault("placements", []).append(item)
    meta = scene.setdefault("meta", {})
    meta.setdefault("requirement_postprocess", {})["added_kitchen_table"] = [item["id"]]
    return [item]


def add_apartment_required_objects(
    apartment_scenes: list[dict[str, Any]],
    asset_search_roots: tuple[Path, ...] = (),
) -> list[dict[str, Any]]:
    present: set[str] = set()
    for scene in apartment_scenes:
        for item in room_items(scene):
            present |= classify_item(item)
    added: list[dict[str, Any]] = []
    if "bed" not in present:
        target = max(apartment_scenes, key=lambda s: float(((s.get("room") or {}).get("area_m2") or 0.0)))
        room = target.get("room") or {}
        width, depth = room_bounds(room)
        target_size = (min(2.0, max(1.1, width - 0.4)), 1.6, 0.65)
        candidate = select_catalog_candidate("bed", target_size, target, None, asset_search_roots)
        if candidate is None:
            _record_missing_catalog_asset(target, "bed")
        else:
            sx, sy, sz = _fit_catalog_size_to_room("bed", candidate, target_size, (width, depth))
            item = make_supplier_required_item(
                room_id=str(room.get("id") or "room"),
                role="bed",
                index=len(room_items(target)) + 1,
                center_xy=(min(width - 1.0, max(1.0, width * 0.5)), min(depth - 0.75, max(0.75, depth * 0.5))),
                size=(sx, sy, sz),
                yaw_deg=0.0,
                z_min=0.0,
                candidate=candidate,
            )
            target.setdefault("placements", []).append(item)
            added.append(item)
    if "table" not in present:
        candidates = sorted(
            apartment_scenes,
            key=lambda s: 0
            if norm((s.get("room") or {}).get("room_type")) in {"kitchen", "living_room", "bedroom"}
            else 1,
        )
        target = candidates[0]
        room = target.get("room") or {}
        width, depth = room_bounds(room)
        target_size = (1.2, 0.7, 0.75)
        candidate = select_catalog_candidate("table", target_size, target, None, asset_search_roots)
        if candidate is None:
            _record_missing_catalog_asset(target, "table")
        else:
            sx, sy, sz = _fit_catalog_size_to_room("table", candidate, target_size, (width, depth))
            item = make_supplier_required_item(
                room_id=str(room.get("id") or "room"),
                role="table",
                index=len(room_items(target)) + 1,
                center_xy=(max(0.6, width * 0.5), max(0.45, depth * 0.5)),
                size=(sx, sy, sz),
                yaw_deg=0.0,
                z_min=0.0,
                candidate=candidate,
            )
            target.setdefault("placements", []).append(item)
            added.append(item)
    return added


def inverse_room_frame(point: tuple[float, float], frame: dict[str, Any]) -> tuple[float, float]:
    off = frame.get("offset_xy") or [0.0, 0.0]
    origin = frame.get("origin_xy") or [0.0, 0.0]
    angle = float(frame.get("rotation_rad") or 0.0)
    x = point[0] - float(off[0])
    y = point[1] - float(off[1])
    return (
        x * math.cos(angle) - y * math.sin(angle) + float(origin[0]),
        x * math.sin(angle) + y * math.cos(angle) + float(origin[1]),
    )


def estimate_apartment_min(apartment: dict[str, Any], room_jsons: dict[str, Path]) -> tuple[float, float]:
    door_graph = (((apartment.get("room") or {}).get("meta") or {}).get("door_graph") or {})
    graph_doors = door_graph.get("doors") or []
    estimates: list[tuple[float, float]] = []
    for door in graph_doors:
        room_id = str(door.get("to") or "")
        center = door.get("center_xy")
        if room_id not in room_jsons or not isinstance(center, list) or len(center) < 2:
            continue
        room = read_json(room_jsons[room_id]).get("room") or {}
        frame = ((room.get("meta") or {}).get("coordinate_frame") or {})
        doors = room.get("doors") or []
        if not frame or not doors:
            continue
        seg = (doors[0] or {}).get("segment") or {}
        if not {"x1", "x2", "y1", "y2"} <= set(seg):
            continue
        local_center = ((float(seg["x1"]) + float(seg["x2"])) / 2.0, (float(seg["y1"]) + float(seg["y2"])) / 2.0)
        gx, gy = inverse_room_frame(local_center, frame)
        estimates.append((gx - float(center[0]), gy - float(center[1])))
    if estimates:
        return (
            sorted(x for x, _ in estimates)[len(estimates) // 2],
            sorted(y for _, y in estimates)[len(estimates) // 2],
        )
    poly = (apartment.get("room") or {}).get("floor_polygon") or []
    return (min(float(p.get("x", 0.0)) for p in poly), min(float(p.get("y", 0.0)) for p in poly)) if poly else (0.0, 0.0)


def transform_item_to_apartment(item: dict[str, Any], frame: dict[str, Any], apt_min: tuple[float, float], room_id: str) -> dict[str, Any]:
    out = deepcopy(item)
    prefix = f"{room_id}__"
    out["id"] = prefix + str(out.get("id") or "item")
    source = out.setdefault("source", {})
    source["source_room_id"] = room_id
    meta = out.setdefault("meta", {})
    meta["source_room_id"] = room_id
    angle_deg = float(frame.get("rotation_deg") or math.degrees(float(frame.get("rotation_rad") or 0.0)))

    aabb = item.get("aabb") or {}
    corners = [
        (float(aabb.get("x_min", 0.0)), float(aabb.get("y_min", 0.0))),
        (float(aabb.get("x_min", 0.0)), float(aabb.get("y_max", 0.0))),
        (float(aabb.get("x_max", 0.0)), float(aabb.get("y_min", 0.0))),
        (float(aabb.get("x_max", 0.0)), float(aabb.get("y_max", 0.0))),
    ]
    apt_pts = []
    for pt in corners:
        gx, gy = inverse_room_frame(pt, frame)
        apt_pts.append((gx - apt_min[0], gy - apt_min[1]))
    xs = [p[0] for p in apt_pts]
    ys = [p[1] for p in apt_pts]
    out["aabb"] = {
        "x_min": round(min(xs), 4),
        "x_max": round(max(xs), 4),
        "y_min": round(min(ys), 4),
        "y_max": round(max(ys), 4),
        "z_min": float(aabb.get("z_min", 0.0)),
        "z_max": float(aabb.get("z_max", 0.0)),
    }
    pos = item.get("position_m") or [
        (float(aabb.get("x_min", 0.0)) + float(aabb.get("x_max", 0.0))) / 2.0,
        (float(aabb.get("y_min", 0.0)) + float(aabb.get("y_max", 0.0))) / 2.0,
        (float(aabb.get("z_min", 0.0)) + float(aabb.get("z_max", 0.0))) / 2.0,
    ]
    gx, gy = inverse_room_frame((float(pos[0]), float(pos[1])), frame)
    out["position_m"] = [round(gx - apt_min[0], 4), round(gy - apt_min[1], 4), float(pos[2])]
    out["size_m"] = [
        round(out["aabb"]["x_max"] - out["aabb"]["x_min"], 4),
        round(out["aabb"]["y_max"] - out["aabb"]["y_min"], 4),
        round(out["aabb"]["z_max"] - out["aabb"]["z_min"], 4),
    ]
    yaw = float(item.get("yaw_deg", item.get("rotation_deg", 0.0)) or 0.0) + angle_deg
    out["yaw_deg"] = round(yaw, 4)
    out["rotation_deg"] = round(yaw, 4)
    return out


def should_skip_apartment_item(item: dict[str, Any]) -> bool:
    name = norm(item.get("name"))
    return name.startswith("room_floor_supplieroverlay") or name.startswith("room_wallpaper_supplieroverlay")


def find_room_scene(room_dir: Path) -> Path | None:
    for rel in SCENE_CANDIDATES:
        path = room_dir / rel
        if path.is_file():
            return path
    return None


def kitchen_scene_from_assembly(room_dir: Path) -> dict[str, Any] | None:
    kitchen_dir = room_dir / "kitchen"
    jsons = sorted(kitchen_dir.glob("*.json"))
    if not jsons:
        return None
    assembly = read_json(jsons[0])
    room = read_json(room_dir / "room.json").get("room") or {}
    dims = assembly.get("dimensions") or {}
    width = float(dims.get("width_m") or room.get("width_m") or 2.4)
    depth = float(dims.get("depth_m") or 0.65)
    height = float(dims.get("height_m") or 2.2)
    item = {
        "id": str(assembly.get("id") or f"{room.get('id')}_kitchen"),
        "name": "procedural kitchen set",
        "category": "kitchen_set",
        "type": "procedural_assembly",
        "assembly_type": "procedural_kitchen",
        "position_m": [width / 2.0, depth / 2.0, height / 2.0],
        "size_m": [width, depth, height],
        "yaw_deg": 0.0,
        "rotation_deg": 0.0,
        "aabb": {"x_min": 0.0, "x_max": width, "y_min": 0.0, "y_max": depth, "z_min": 0.0, "z_max": height},
        "asset": {"kind": "procedural_kitchen", "assembly_type": "procedural_kitchen"},
        "meta": {**assembly, "procedural_assembly": "kitchen"},
        "source": {"asset_source": "procedural_kitchen", "source_room_id": room.get("id")},
    }
    return {"schema": "scene.v1", "room": room, "placements": [item], "meta": {"source": str(jsons[0])}}


def process_apartment(apt_dir: Path, mode: str) -> dict[str, Any]:
    manifest_path = apt_dir / "manifest.json"
    apartment_path = apt_dir / "apartment.json"
    if not manifest_path.is_file() or not apartment_path.is_file():
        raise FileNotFoundError(f"Missing manifest/apartment json in {apt_dir}")
    manifest = read_json(manifest_path)
    apartment = read_json(apartment_path)
    rooms_meta = manifest.get("rooms") or []
    room_jsons: dict[str, Path] = {}
    loaded_scenes: list[dict[str, Any]] = []
    scene_entries: list[dict[str, Any]] = []
    room_reports: list[dict[str, Any]] = []
    asset_search_roots = (apt_dir, LOCAL_TABLE_ASSET_ROOT)

    for room_meta in rooms_meta:
        room_id = str(room_meta.get("room_id") or "")
        room_json = Path(str(room_meta.get("room_json") or ""))
        if room_id and room_json.is_file():
            room_jsons[room_id] = room_json

    for room_meta in rooms_meta:
        room_id = str(room_meta.get("room_id") or "")
        room_dir = apt_dir / "rooms" / room_id
        scene_path = find_room_scene(room_dir)
        scene = read_json(scene_path) if scene_path else kitchen_scene_from_assembly(room_dir)
        if not isinstance(scene, dict):
            room_reports.append({"room_id": room_id, "status": "missing_scene"})
            continue
        added = add_missing_kitchen_table(
            scene,
            prompt_room_type=room_meta.get("prompt_room_type"),
            asset_search_roots=asset_search_roots,
        )
        loaded_scenes.append(scene)
        scene_entries.append(
            {
                "scene": scene,
                "room_meta": room_meta,
                "room_id": room_id,
                "room_dir": room_dir,
                "scene_path": scene_path,
                "added": added,
            }
        )

    sanitary_repairs = repair_sanitary_layouts(scene_entries, asset_search_roots=asset_search_roots)
    lighting_repairs = repair_ceiling_lighting_layouts(scene_entries)
    sanitary_added = add_missing_sanitary_apartment(scene_entries, asset_search_roots=asset_search_roots)
    apartment_added = add_apartment_required_objects(loaded_scenes, asset_search_roots=asset_search_roots)

    for entry in scene_entries:
        scene = entry["scene"]
        room_meta = entry["room_meta"]
        room_id = str(entry["room_id"])
        room_dir = Path(entry["room_dir"])
        scene_path = entry.get("scene_path")
        added = entry.get("added") if isinstance(entry.get("added"), list) else []
        patched_path = room_dir / "pipeline" / mode / "scene_requirements.v1.json"
        write_json(patched_path, scene)
        room_reports.append(
            {
                "room_id": room_id,
                "room_type": room_meta.get("room_type"),
                "prompt_room_type": room_meta.get("prompt_room_type"),
                "source_scene": str(scene_path) if scene_path else str(room_dir / "kitchen"),
                "requirements_scene": str(patched_path.resolve()),
                "added": [{"id": x["id"], "role": x["meta"]["required_role"]} for x in added],
            }
        )

    apt_min = estimate_apartment_min(apartment, room_jsons)

    placements: list[dict[str, Any]] = []
    for scene in loaded_scenes:
        room = scene.get("room") or {}
        room_id = str(room.get("id") or "")
        frame = ((room.get("meta") or {}).get("coordinate_frame") or {})
        if not room_id or not frame:
            continue
        for item in room_items(scene):
            if should_skip_apartment_item(item):
                continue
            placements.append(transform_item_to_apartment(item, frame, apt_min, room_id))

    out_scene = {
        "schema": "scene.v1",
        "room": apartment.get("room") or {},
        "placements": placements,
        "meta": {
            "source": "ensure_apartment_requirements",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "apartment_dir": str(apt_dir.resolve()),
            "mode": mode,
            "apartment_global_min_xy": [round(apt_min[0], 6), round(apt_min[1], 6)],
            "room_reports": room_reports,
            "sanitary_added": [
                {"id": x["id"], "role": x.get("meta", {}).get("required_role"), "room_id": x.get("meta", {}).get("room_id")}
                for x in sanitary_added
            ],
            "sanitary_repairs": sanitary_repairs,
            "lighting_repairs": lighting_repairs,
            "apartment_added": [
                {"id": x["id"], "role": x.get("meta", {}).get("required_role"), "room_id": x.get("meta", {}).get("room_id")}
                for x in apartment_added
            ],
            "requirements": {
                "sanitary_apartment": list(SANITARY_REQUIRED),
                "apartment": ["bed", "table"],
            },
        },
    }
    out_dir = apt_dir / "apartment_pipeline" / mode
    out_path = write_json(out_dir / "scene_apartment.requirements.v1.json", out_scene)
    report_path = write_json(
        out_dir / "requirements_report.json",
        {
            "apartment_dir": str(apt_dir.resolve()),
            "scene_json": str(out_path.resolve()),
            "room_reports": room_reports,
            "sanitary_added": out_scene["meta"]["sanitary_added"],
            "sanitary_repairs": sanitary_repairs,
            "lighting_repairs": lighting_repairs,
            "apartment_added": out_scene["meta"]["apartment_added"],
            "placement_count": len(placements),
        },
    )
    return {"apartment_dir": str(apt_dir), "scene_json": str(out_path), "report_json": str(report_path), "placement_count": len(placements)}


def iter_apartments(root: Path) -> list[Path]:
    if (root / "manifest.json").is_file() and (root / "apartment.json").is_file():
        return [root]
    return sorted(p for p in root.glob("*/*") if (p / "manifest.json").is_file() and (p / "apartment.json").is_file())


def build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Ensure apartment-level required objects and assemble room scenes into one apartment scene.")
    ap.add_argument("root", help="Apartment dir or root containing project/apartment dirs.")
    ap.add_argument("--mode", default="optimal")
    ap.add_argument("--out-summary", default=None)
    return ap


def main() -> None:
    args = build_cli().parse_args()
    root = Path(args.root).expanduser().resolve()
    results = [process_apartment(apt_dir, args.mode) for apt_dir in iter_apartments(root)]
    summary_path = Path(args.out_summary).expanduser().resolve() if args.out_summary else root / "apartment_requirements_summary.json"
    write_json(summary_path, {"root": str(root), "count": len(results), "results": results})
    print(f"processed_apartments = {len(results)}")
    print(f"summary = {summary_path}")
    for result in results:
        print(f"{result['apartment_dir']} -> {result['scene_json']} ({result['placement_count']} placements)")


if __name__ == "__main__":
    main()
