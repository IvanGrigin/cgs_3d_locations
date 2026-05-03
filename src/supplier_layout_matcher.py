#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script matches layout targets to supplier catalog items for replacement.
It combines semantic grouping, size fit, design similarity, and prompt context.
Optional user preferences can hard-filter or bias selection by price, color, or site.
An optional LLM reranker can choose the final item from a heuristic top-N shortlist.
The default path stays deterministic when no extra settings are provided.
"""

from __future__ import annotations

import argparse
import colorsys
import json
import math
import re
import sqlite3
from pathlib import Path
from typing import Any, Optional

try:
    from .layout_targets import _semantic_group
except ImportError:
    from layout_targets import _semantic_group


def read_json(path: str | Path) -> Any:
    p = Path(path).expanduser().resolve()
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(path: str | Path, data: Any) -> None:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _json_loads_or(value: Any, default: Any) -> Any:
    text = str(value or "").strip()
    if not text:
        return default
    try:
        return json.loads(text)
    except Exception:
        return default


def _is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def _dedup_keep_order(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _value_to_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        out: list[str] = []
        for item in value:
            out.extend(_value_to_text_list(item))
        return out
    text = str(value).strip()
    if not text:
        return []
    if "," in text:
        return [part.strip() for part in text.split(",") if part.strip()]
    return [text]


def _normalize_color_preference_list(value: Any) -> list[str]:
    tokens: list[str] = []
    for item in _value_to_text_list(value):
        tokens.extend(sorted(_extract_color_tokens(item)))
    return _dedup_keep_order(tokens)


def _normalize_site_list(value: Any) -> list[str]:
    return _dedup_keep_order([str(x or "").strip().lower() for x in _value_to_text_list(value)])


def _normalize_brand_list(value: Any) -> list[str]:
    return _dedup_keep_order([str(x or "").strip() for x in _value_to_text_list(value)])


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


SUPPLIER_SELECTION_STRATEGIES = {
    "balanced",
    "cheapest",
    "cheap_style",
    "style",
}


STYLE_LLM_MIN_CONFIDENCE = 0.55
STYLE_LLM_MIN_QUALITY = 6


STYLE_ALIASES = {
    "industrial": "loft_industrial",
    "loft": "loft_industrial",
    "loft_industrial": "loft_industrial",
    "mid-century": "mid_century_modern",
    "mid_century": "mid_century_modern",
    "mid_century_modern": "mid_century_modern",
    "mcm": "mid_century_modern",
    "eco": "eco_organic",
    "organic": "eco_organic",
    "eco_organic": "eco_organic",
    "hightech": "high_tech",
    "high_tech": "high_tech",
    "soft_minimal": "soft_minimalism",
    "soft_minimalism": "soft_minimalism",
    "minimal": "minimalism",
    "minimalism": "minimalism",
    "modern": "modern",
    "современный": "modern",
    "sovremennyi": "modern",
    "sovremenny": "modern",
    "contemporary": "contemporary",
    "scandinavian": "scandinavian",
    "nordic": "scandinavian",
    "japandi": "japandi",
    "wabi_sabi": "eco_organic",
    "классический": "classic",
    "classic": "classic",
}


STYLE_COMPATIBILITY = {
    "modern": {"contemporary", "minimalism", "soft_minimalism", "high_tech"},
    "contemporary": {"modern", "soft_minimalism", "minimalism"},
    "minimalism": {"soft_minimalism", "modern", "japandi"},
    "soft_minimalism": {"minimalism", "modern", "contemporary", "japandi"},
    "scandinavian": {"japandi", "eco_organic", "soft_minimalism"},
    "japandi": {"scandinavian", "minimalism", "soft_minimalism", "eco_organic"},
    "loft_industrial": {"modern", "high_tech"},
    "high_tech": {"modern", "loft_industrial"},
    "eco_organic": {"scandinavian", "japandi", "soft_minimalism"},
    "mid_century_modern": {"modern", "contemporary"},
    "classic": {"contemporary"},
}


def _normalize_style_label(value: Any) -> str | None:
    text = str(value or "").strip().lower().replace("ё", "е")
    if not text:
        return None
    text = text.replace("-", "_").replace(" ", "_").replace("/", "_")
    return STYLE_ALIASES.get(text, text if text in STYLE_COMPATIBILITY else None)


def _extract_styles_from_text(value: Any) -> set[str]:
    text = str(value or "").strip().lower().replace("ё", "е")
    if not text:
        return set()
    found: set[str] = set()
    normalized_text = text.replace("-", "_").replace(" ", "_").replace("/", "_")
    direct = _normalize_style_label(normalized_text)
    if direct:
        found.add(direct)
    for raw, normalized in STYLE_ALIASES.items():
        if raw.replace("_", " ") in text or raw in normalized_text:
            found.add(normalized)
    return found


def _normalize_preference_scope(raw: dict[str, Any] | None) -> dict[str, Any]:
    raw = raw if isinstance(raw, dict) else {}
    max_price_rub = None
    for key in ("max_price_rub", "max_price", "budget_rub", "budget", "price_limit_rub"):
        max_price_rub = _safe_float(raw.get(key))
        if max_price_rub is not None:
            break

    return {
        "max_price_rub": max_price_rub,
        "preferred_colors": _normalize_color_preference_list(
            raw.get("preferred_colors", raw.get("preferred_color", raw.get("colors", raw.get("color"))))
        ),
        "avoid_colors": _normalize_color_preference_list(
            raw.get("avoid_colors", raw.get("avoid_color", raw.get("excluded_colors")))
        ),
        "preferred_brands": _normalize_brand_list(
            raw.get("preferred_brands", raw.get("preferred_brand", raw.get("brands", raw.get("brand"))))
        ),
        "allowed_sites": _normalize_site_list(raw.get("allowed_sites", raw.get("sites"))),
        "disallowed_sites": _normalize_site_list(raw.get("disallowed_sites", raw.get("excluded_sites"))),
        "strict_color": _is_truthy(raw.get("strict_color")),
        "require_real_asset": _is_truthy(raw.get("require_real_asset")),
        "require_model_url": _is_truthy(raw.get("require_model_url")),
    }


def _merge_preference_scopes(base: dict[str, Any] | None, override: dict[str, Any] | None) -> dict[str, Any]:
    base = dict(base or {})
    override = dict(override or {})
    merged = {
        "max_price_rub": override.get("max_price_rub") if override.get("max_price_rub") is not None else base.get("max_price_rub"),
        "preferred_colors": _dedup_keep_order(list(base.get("preferred_colors") or []) + list(override.get("preferred_colors") or [])),
        "avoid_colors": _dedup_keep_order(list(base.get("avoid_colors") or []) + list(override.get("avoid_colors") or [])),
        "preferred_brands": _dedup_keep_order(list(base.get("preferred_brands") or []) + list(override.get("preferred_brands") or [])),
        "allowed_sites": _dedup_keep_order(list(base.get("allowed_sites") or []) + list(override.get("allowed_sites") or [])),
        "disallowed_sites": _dedup_keep_order(list(base.get("disallowed_sites") or []) + list(override.get("disallowed_sites") or [])),
        "strict_color": bool(base.get("strict_color")) or bool(override.get("strict_color")),
        "require_real_asset": bool(base.get("require_real_asset")) or bool(override.get("require_real_asset")),
        "require_model_url": bool(base.get("require_model_url")) or bool(override.get("require_model_url")),
    }
    return merged


def _normalize_user_preferences(raw: dict[str, Any] | None) -> dict[str, Any]:
    raw = raw if isinstance(raw, dict) else {}
    has_namespaced = any(key in raw for key in ("global", "by_target_id", "by_semantic_group"))
    global_scope = raw.get("global") if has_namespaced else raw
    by_target_id_raw = raw.get("by_target_id") if isinstance(raw.get("by_target_id"), dict) else {}
    by_group_raw = raw.get("by_semantic_group") if isinstance(raw.get("by_semantic_group"), dict) else {}
    return {
        "global": _normalize_preference_scope(global_scope if isinstance(global_scope, dict) else {}),
        "by_target_id": {
            str(key): _normalize_preference_scope(value if isinstance(value, dict) else {})
            for key, value in by_target_id_raw.items()
        },
        "by_semantic_group": {
            str(key): _normalize_preference_scope(value if isinstance(value, dict) else {})
            for key, value in by_group_raw.items()
        },
    }


def _target_user_preferences(target: dict[str, Any], context: dict[str, Any] | None) -> dict[str, Any]:
    context = context or {}
    normalized = context.get("user_preferences") or {"global": {}, "by_target_id": {}, "by_semantic_group": {}}
    prefs = _merge_preference_scopes({}, normalized.get("global"))
    target_id = str(target.get("target_id") or "").strip()
    group = str(target.get("semantic_group") or "").strip()
    prefs = _merge_preference_scopes(prefs, (normalized.get("by_semantic_group") or {}).get(group))
    prefs = _merge_preference_scopes(prefs, (normalized.get("by_target_id") or {}).get(target_id))
    return prefs


def _product_size_m(row: dict[str, Any]) -> list[float] | None:
    dims = row.get("dimensions_cm") if isinstance(row.get("dimensions_cm"), dict) else {}
    width = row.get("width_cm", dims.get("width"))
    depth = row.get("depth_cm", dims.get("depth"))
    height = row.get("height_cm", dims.get("height"))
    if width is None or depth is None or height is None:
        return None
    try:
        return [float(width) / 100.0, float(depth) / 100.0, float(height) / 100.0]
    except Exception:
        return None


def _effective_target_size_m(target: dict[str, Any]) -> list[float]:
    raw = target.get("size_m") or [0.0, 0.0, 0.0]
    try:
        tw, td, th = [max(float(x), 1e-6) for x in raw]
    except Exception:
        return [1e-6, 1e-6, 1e-6]

    if str(target.get("semantic_group") or "").strip() == "lamp_ceiling" and th < 0.15:
        # Generated ceiling lights are often almost-flat placeholders.
        # Match against a realistic chandelier envelope instead of the stub size.
        tw = max(tw, 0.60)
        td = max(td, 0.60)
        th = max(th, 0.45)
    return [tw, td, th]


def _has_text(value: Any) -> bool:
    return bool(str(value or "").strip())


def _has_full_dimensions(row: dict[str, Any]) -> bool:
    return (
        row.get("width_cm") is not None
        and row.get("depth_cm") is not None
        and row.get("height_cm") is not None
    )


def _has_category(row: dict[str, Any]) -> bool:
    return _has_text(row.get("category_raw")) or _has_text(row.get("category_norm"))


def _row_is_rich(row: dict[str, Any]) -> bool:
    return all(
        (
            _has_text(row.get("title")),
            row.get("price_value") is not None,
            _has_full_dimensions(row),
            _has_text(row.get("description")) or _has_text(row.get("vlm_description_text")),
            _has_category(row),
            _has_text(row.get("brand")),
        )
    )


SUPPORTED_LOCAL_ASSET_EXTS = {"obj", "fbx", "glb", "gltf"}
LOW_QUALITY_ASSET_STATUSES = {"proxy_generated_with_blender", "needs_blender_rebuild"}
_CAMEL_RE_1 = re.compile(r"([a-z0-9])([A-Z])")
_CAMEL_RE_2 = re.compile(r"([A-Z]+)([A-Z][a-z])")
_TOKEN_STOPWORDS = {
    "factory",
    "object",
    "spawn",
    "asset",
    "temp",
    "parent",
    "placeholder",
    "generated",
    "simple",
    "large",
    "small",
    "single",
    "double",
    "medium",
    "extra",
}
_TOKEN_ALIASES: dict[str, tuple[str, ...]] = {
    "bed": ("bed",),
    "кровать": ("bed",),
    "кровати": ("bed",),
    "dresser": ("dresser",),
    "комод": ("dresser",),
    "cabinet": ("cabinet",),
    "шкаф": ("wardrobe", "cabinet"),
    "wardrobe": ("wardrobe",),
    "closet": ("wardrobe",),
    "nightstand": ("nightstand",),
    "bedside": ("nightstand",),
    "тумба": ("nightstand", "cabinet"),
    "desk": ("desk",),
    "письменныи": ("desk",),
    "письменный": ("desk",),
    "table": ("table",),
    "стол": ("table",),
    "shelf": ("shelf", "bookcase"),
    "shelves": ("shelf", "bookcase"),
    "полка": ("shelf", "bookcase"),
    "bookcase": ("bookcase", "shelf"),
    "bookshelf": ("bookcase", "shelf"),
    "стеллаж": ("bookcase", "shelf"),
    "sofa": ("sofa",),
    "диван": ("sofa",),
    "chair": ("chair",),
    "стул": ("chair",),
    "armchair": ("armchair", "chair"),
    "кресло": ("armchair", "chair"),
    "sideboard": ("sideboard", "cabinet"),
    "tv": ("tv",),
    "lamp": ("lamp",),
    "люстра": ("chandelier", "ceiling", "lamp"),
    "chandelier": ("chandelier", "ceiling", "lamp"),
    "торшер": ("floor", "lamp"),
    "mirror": ("mirror",),
    "зеркало": ("mirror",),
}
_ACCEPTANCE_DEFAULTS = {
    "known": {
        "max_primary_axis_distance": 0.75,
        "max_secondary_axis_distance": 0.85,
        "min_query_score": 18.0,
    },
    "missing": {
        "min_query_score": 24.0,
        "min_query_overlap_count": 1,
    },
}
_ACCEPTANCE_BY_GROUP: dict[str, dict[str, dict[str, float]]] = {
    "bed": {
        "known": {
            "max_primary_axis_distance": 0.55,
            "max_secondary_axis_distance": 0.7,
            "min_query_score": 16.0,
        },
        "missing": {
            "min_query_score": 22.0,
            "min_query_overlap_count": 1,
        },
    },
    "desk": {
        "known": {
            "max_primary_axis_distance": 0.5,
            "max_secondary_axis_distance": 0.65,
            "min_query_score": 18.0,
        },
        "missing": {
            "min_query_score": 24.0,
            "min_query_overlap_count": 1,
        },
    },
    "dresser": {
        "known": {
            "max_primary_axis_distance": 0.8,
            "max_secondary_axis_distance": 0.8,
            "min_query_score": 18.0,
        },
        "missing": {
            "min_query_score": 24.0,
            "min_query_overlap_count": 1,
        },
    },
    "wardrobe": {
        "known": {
            "max_primary_axis_distance": 0.7,
            "max_secondary_axis_distance": 0.7,
            "min_query_score": 18.0,
        },
        "missing": {
            "min_query_score": 24.0,
            "min_query_overlap_count": 1,
        },
    },
    "shelf": {
        "known": {
            "max_primary_axis_distance": 0.85,
            "max_secondary_axis_distance": 0.9,
            "min_query_score": 14.0,
        },
        "missing": {
            "min_query_score": 18.0,
            "min_query_overlap_count": 1,
        },
    },
    "lamp_floor": {
        "known": {
            "max_primary_axis_distance": 0.95,
            "max_secondary_axis_distance": 1.0,
            "min_query_score": 12.0,
        },
        "missing": {
            "min_query_score": 16.0,
            "min_query_overlap_count": 1,
        },
    },
    "lamp_ceiling": {
        "known": {
            "max_primary_axis_distance": 1.1,
            "max_secondary_axis_distance": 1.1,
            "min_query_score": 12.0,
        },
        "missing": {
            "min_query_score": 16.0,
            "min_query_overlap_count": 1,
        },
    },
}


def _candidate_has_ready_real_asset(row: dict[str, Any]) -> bool:
    local_path = str(row.get("asset_local_path") or "").strip()
    asset_format = str(row.get("asset_format") or "").strip().lower()
    asset_status = str(row.get("asset_status") or "").strip().lower()
    if not local_path or not Path(local_path).expanduser().is_file():
        return False
    if asset_format not in SUPPORTED_LOCAL_ASSET_EXTS:
        return False
    return asset_status not in LOW_QUALITY_ASSET_STATUSES


def _candidate_has_downloadable_asset(row: dict[str, Any]) -> bool:
    if _candidate_has_ready_real_asset(row):
        return True
    model_url = str(row.get("model_download_url") or row.get("mesh_source_url") or "").strip()
    model_format = str(row.get("model_format") or "").strip().lower().lstrip(".")
    if model_url and model_format in {"rar", "zip", "7z", "max", "fbx", "obj", "glb", "gltf"}:
        return True
    model_link_type = str(row.get("model_link_type") or "").strip().lower()
    return bool(model_url or model_link_type in {"direct_file", "landing_page", "product_page"})


def _normalize_text_tokens(value: Any) -> set[str]:
    text = str(value or "").strip()
    text = _CAMEL_RE_2.sub(r"\1 \2", text)
    text = _CAMEL_RE_1.sub(r"\1 \2", text)
    text = text.replace("_", " ").replace("-", " ").replace("/", " ")
    text = text.lower().replace("ё", "е")
    cleaned = []
    for ch in text:
        cleaned.append(ch if ch.isalnum() else " ")
    raw_parts = [part for part in "".join(cleaned).split() if len(part) >= 3]
    tokens: set[str] = set()
    for part in raw_parts:
        if part in _TOKEN_STOPWORDS:
            continue
        alias_tokens = _TOKEN_ALIASES.get(part)
        if alias_tokens:
            for alias in alias_tokens:
                if len(alias) >= 3 and alias not in _TOKEN_STOPWORDS:
                    tokens.add(alias)
        tokens.add(part)
    return tokens


def _normalize_color_token(token: str) -> str:
    t = str(token or "").strip().lower().replace("ё", "е")
    aliases = {
        "grey": "gray",
        "silver": "gray",
        "anthracite": "gray",
        "graphite": "gray",
        "charcoal": "gray",
        "oak": "brown",
        "walnut": "brown",
        "wood": "brown",
        "natural": "beige",
        "sand": "beige",
        "cream": "beige",
        "ivory": "beige",
        "golden": "yellow",
        "navy": "blue",
        "teal": "green",
        "olive": "green",
        "burgundy": "red",
        "pink": "red",
        "white": "white",
        "black": "black",
        "gray": "gray",
        "beige": "beige",
        "brown": "brown",
        "red": "red",
        "orange": "orange",
        "yellow": "yellow",
        "green": "green",
        "blue": "blue",
        "purple": "purple",
    }
    return aliases.get(t, t)


def _rgb_to_basic_color_tokens(value: Any) -> set[str]:
    if not isinstance(value, list) or len(value) < 3:
        return set()
    try:
        r, g, b = [float(value[i]) for i in range(3)]
    except Exception:
        return set()
    if max(abs(r - 0.7), abs(g - 0.7), abs(b - 0.7)) <= 0.03:
        return set()

    h, s, v = colorsys.rgb_to_hsv(max(0.0, min(1.0, r)), max(0.0, min(1.0, g)), max(0.0, min(1.0, b)))
    tokens: set[str] = set()
    if v <= 0.18:
        tokens.add("black")
    elif s <= 0.12:
        if v >= 0.85:
            tokens.add("white")
        else:
            tokens.add("gray")
    else:
        if h < 0.04 or h >= 0.96:
            tokens.add("red")
        elif h < 0.10:
            tokens.add("orange")
        elif h < 0.16:
            tokens.add("yellow")
        elif h < 0.42:
            tokens.add("green")
        elif h < 0.72:
            tokens.add("blue")
        elif h < 0.86:
            tokens.add("purple")
        else:
            tokens.add("red")
        if v < 0.72 and "orange" not in tokens and "yellow" not in tokens:
            tokens.add("brown")
        if 0.08 <= h <= 0.16 and v > 0.75 and s < 0.35:
            tokens.add("beige")
    return {_normalize_color_token(x) for x in tokens}


def _extract_color_tokens(value: Any) -> set[str]:
    if isinstance(value, list):
        return _rgb_to_basic_color_tokens(value)
    return {_normalize_color_token(x) for x in _normalize_text_tokens(value)}


def _target_category_tokens(target: dict[str, Any]) -> set[str]:
    parts = [
        target.get("category"),
        target.get("name"),
        target.get("semantic_group"),
    ]
    tokens: set[str] = set()
    for part in parts:
        tokens |= _normalize_text_tokens(part)
    return tokens


def _target_design_tokens(target: dict[str, Any]) -> set[str]:
    constraints = target.get("constraints") or {}
    parts = [
        constraints.get("style"),
        constraints.get("material"),
        constraints.get("materials"),
        constraints.get("color"),
        constraints.get("brand"),
        constraints.get("collection"),
        target.get("name"),
        target.get("category"),
    ]
    tokens: set[str] = set()
    for part in parts:
        tokens |= _normalize_text_tokens(part)
    return tokens


def _target_color_tokens(target: dict[str, Any]) -> set[str]:
    constraints = target.get("constraints") or {}
    tokens: set[str] = set()
    tokens |= _extract_color_tokens(constraints.get("color"))
    tokens |= _extract_color_tokens(target.get("color"))
    tokens |= _extract_color_tokens(target.get("color_rgb"))
    return tokens


def _target_query_tokens(target: dict[str, Any]) -> set[str]:
    constraints = target.get("constraints") or {}
    parts = [
        target.get("name"),
        target.get("category"),
        constraints.get("style"),
        constraints.get("material"),
        constraints.get("materials"),
        constraints.get("color"),
        constraints.get("brand"),
    ]
    tokens: set[str] = set()
    for part in parts:
        tokens |= _normalize_text_tokens(part)
    return tokens


def _row_query_tokens(row: dict[str, Any]) -> set[str]:
    parts = [
        row.get("title"),
        row.get("brand"),
        row.get("collection"),
        row.get("category_raw"),
        row.get("category_norm"),
        row.get("semantic_group"),
        row.get("style"),
        row.get("color"),
        row.get("materials"),
        row.get("description"),
        row.get("vlm_description_text"),
        row.get("vlm_description_summary"),
        row.get("vlm_color"),
        row.get("vlm_materials"),
        row.get("vlm_style"),
        row.get("room"),
    ]
    tokens: set[str] = set()
    for part in parts:
        tokens |= _normalize_text_tokens(part)
    return tokens


def _row_category_tokens(row: dict[str, Any]) -> set[str]:
    parts = [
        row.get("title"),
        row.get("category_raw"),
        row.get("category_norm"),
        row.get("semantic_group"),
    ]
    tokens: set[str] = set()
    for part in parts:
        tokens |= _normalize_text_tokens(part)
    return tokens


def _row_design_tokens(row: dict[str, Any]) -> set[str]:
    parts = [
        row.get("title"),
        row.get("brand"),
        row.get("collection"),
        row.get("category_norm"),
        row.get("semantic_group"),
        row.get("style"),
        row.get("color"),
        row.get("materials"),
        row.get("description"),
        row.get("vlm_description_text"),
        row.get("vlm_description_summary"),
        row.get("vlm_color"),
        row.get("vlm_materials"),
        row.get("vlm_style"),
        row.get("room"),
    ]
    tokens: set[str] = set()
    for part in parts:
        tokens |= _normalize_text_tokens(part)
    return tokens


def _row_color_tokens(row: dict[str, Any]) -> set[str]:
    tokens: set[str] = set()
    tokens |= _extract_color_tokens(row.get("color"))
    tokens |= _extract_color_tokens(row.get("vlm_color"))
    tokens |= _extract_color_tokens(row.get("vlm_description_text"))
    tokens |= _extract_color_tokens(row.get("materials"))
    tokens |= _extract_color_tokens(row.get("title"))
    return tokens


def _same_family(group_a: str, group_b: str) -> bool:
    if group_a == group_b:
        return True
    families = [
        {"desk", "side_table"},
        {"coffee_table", "side_table"},
        {"chair", "armchair"},
        {"dresser", "nightstand"},
        {"shelf", "tv_stand"},
        {"computer", "computer_monitor", "laptop_computer_keyboard_mouse"},
    ]
    return any(group_a in fam and group_b in fam for fam in families)


def _target_requires_exact_group(target_group: str) -> bool:
    return target_group in {
        "bed",
        "desk",
        "shelf",
        "wardrobe",
        "lamp_table",
        "lamp_floor",
        "lamp_ceiling",
    }


def _fits_inside_bbox(target_size_m: list[float], candidate_size_m: list[float], tolerance_ratio: float = 0.03) -> tuple[bool, dict[str, Any]]:
    if len(target_size_m) != 3 or len(candidate_size_m) != 3:
        return False, {"fit_checked": False, "fits_bbox": False}

    tw, td, th = [max(float(x), 0.0) for x in target_size_m]
    cw, cd, ch = [max(float(x), 0.0) for x in candidate_size_m]

    allowed_w = tw * (1.0 + tolerance_ratio)
    allowed_d = td * (1.0 + tolerance_ratio)
    allowed_h = th * (1.0 + tolerance_ratio)

    direct_fit = cw <= allowed_w and cd <= allowed_d and ch <= allowed_h
    swapped_fit = cd <= allowed_w and cw <= allowed_d and ch <= allowed_h
    fits = direct_fit or swapped_fit

    return fits, {
        "fit_checked": True,
        "fits_bbox": fits,
        "fit_tolerance_ratio": tolerance_ratio,
        "target_size_m": [round(tw, 4), round(td, 4), round(th, 4)],
        "candidate_size_m": [round(cw, 4), round(cd, 4), round(ch, 4)],
        "bbox_fit_mode": "width_depth_swap_allowed_height_fixed",
        "bbox_fit_orientation": "direct" if direct_fit else "swapped_xy" if swapped_fit else None,
        "bbox_overflow_m": {
            "width": round(max(0.0, cw - allowed_w), 4),
            "depth": round(max(0.0, cd - allowed_d), 4),
            "height": round(max(0.0, ch - allowed_h), 4),
        },
    }


def _rescalable_fit_policy(group: str) -> dict[str, float] | None:
    policies: dict[str, dict[str, float]] = {
        "bed": {"width_max_ratio": 1.3, "depth_max_ratio": 1.3, "height_max_ratio": 1.2},
        "sofa": {"width_max_ratio": 1.25, "depth_max_ratio": 1.25, "height_max_ratio": 1.2},
        "armchair": {"width_max_ratio": 1.2, "depth_max_ratio": 1.2, "height_max_ratio": 1.2},
        "chair": {"width_max_ratio": 1.2, "depth_max_ratio": 1.2, "height_max_ratio": 1.2},
        "desk": {"width_max_ratio": 1.18, "depth_max_ratio": 1.18, "height_max_ratio": 1.15},
        "side_table": {"width_max_ratio": 1.16, "depth_max_ratio": 1.16, "height_max_ratio": 1.12},
        "nightstand": {"width_max_ratio": 1.16, "depth_max_ratio": 1.16, "height_max_ratio": 1.12},
        "coffee_table": {"width_max_ratio": 1.18, "depth_max_ratio": 1.18, "height_max_ratio": 1.12},
        "dresser": {"width_max_ratio": 1.5, "depth_max_ratio": 1.4, "height_max_ratio": 1.5},
        "tv_stand": {"width_max_ratio": 1.45, "depth_max_ratio": 1.35, "height_max_ratio": 1.4},
        "shelf": {"width_max_ratio": 1.55, "depth_max_ratio": 1.4, "height_max_ratio": 1.55},
        "wardrobe": {"width_max_ratio": 1.45, "depth_max_ratio": 1.35, "height_max_ratio": 1.45},
        "lamp_floor": {"width_max_ratio": 1.75, "depth_max_ratio": 1.75, "height_max_ratio": 1.5},
        "lamp_ceiling": {"width_max_ratio": 2.5, "depth_max_ratio": 2.5, "height_max_ratio": 25.0},
        "lamp_table": {"width_max_ratio": 1.4, "depth_max_ratio": 1.4, "height_max_ratio": 1.35},
    }
    return policies.get(group)


def _passes_rescalable_fit(
    target_group: str,
    target_size_m: list[float],
    candidate_size_m: list[float],
) -> tuple[bool, dict[str, Any]]:
    policy = _rescalable_fit_policy(target_group)
    if not policy or len(target_size_m) != 3 or len(candidate_size_m) != 3:
        return False, {
            "rescalable_fit_checked": False,
            "passes_rescalable_fit": False,
        }

    tw, td, th = [max(float(x), 1e-6) for x in target_size_m]
    cw, cd, ch = [max(float(x), 1e-6) for x in candidate_size_m]
    candidates = [
        {
            "fit_orientation": "direct",
            "width_ratio": cw / tw,
            "depth_ratio": cd / td,
            "height_ratio": ch / th,
        },
        {
            "fit_orientation": "swapped_xy",
            "width_ratio": cd / tw,
            "depth_ratio": cw / td,
            "height_ratio": ch / th,
        },
    ]

    def passes(item: dict[str, float]) -> bool:
        return (
            item["width_ratio"] <= policy["width_max_ratio"]
            and item["depth_ratio"] <= policy["depth_max_ratio"]
            and item["height_ratio"] <= policy["height_max_ratio"]
        )

    candidates.sort(
        key=lambda item: (
            0 if passes(item) else 1,
            abs(item["width_ratio"] - 1.0) + abs(item["depth_ratio"] - 1.0) + abs(item["height_ratio"] - 1.0),
        )
    )
    best = candidates[0]
    return passes(best), {
        "rescalable_fit_checked": True,
        "passes_rescalable_fit": passes(best),
        "rescalable_fit_orientation": best["fit_orientation"],
        "rescalable_fit_policy": {
            "width_max_ratio": policy["width_max_ratio"],
            "depth_max_ratio": policy["depth_max_ratio"],
            "height_max_ratio": policy["height_max_ratio"],
        },
        "candidate_to_target_ratio": {
            "width": round(best["width_ratio"], 6),
            "depth": round(best["depth_ratio"], 6),
            "height": round(best["height_ratio"], 6),
        },
    }


def _size_distance(size_a: list[float], size_b: list[float]) -> float:
    eps = 1e-6
    vals = []
    for a, b in zip(size_a, size_b):
        aa = max(float(a), eps)
        bb = max(float(b), eps)
        vals.append(abs(math.log(aa / bb)))
    return sum(vals) / max(len(vals), 1)


def _axis_log_distance(a: float, b: float) -> float:
    eps = 1e-6
    aa = max(float(a), eps)
    bb = max(float(b), eps)
    return abs(math.log(aa / bb))


def _dimension_priority_info(target: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    target_group = str(target.get("semantic_group") or "")
    target_size_m = _effective_target_size_m(target)
    candidate_size_m = _product_size_m(row)
    if candidate_size_m is None:
        return {
            "dimension_priority": "missing_candidate_size",
            "primary_axis_distance": float("inf"),
            "secondary_axis_distance": float("inf"),
            "overall_size_distance": float("inf"),
        }

    tw, td, th = [float(x) for x in target_size_m]
    cw, cd, ch = [float(x) for x in candidate_size_m]
    footprint_target = [max(tw, td), min(tw, td)]
    footprint_candidate = [max(cw, cd), min(cw, cd)]
    footprint_distance = (
        _axis_log_distance(footprint_target[0], footprint_candidate[0])
        + _axis_log_distance(footprint_target[1], footprint_candidate[1])
    ) / 2.0
    height_distance = _axis_log_distance(th, ch)
    overall_distance = _size_distance(target_size_m, candidate_size_m)

    if target_group in {"desk", "coffee_table", "side_table", "nightstand", "dresser", "shelf", "tv_stand"}:
        return {
            "dimension_priority": "support_surface_height_first",
            "primary_axis_name": "height",
            "secondary_axis_name": "footprint",
            "primary_axis_distance": height_distance,
            "secondary_axis_distance": footprint_distance,
            "overall_size_distance": overall_distance,
        }

    if target_group == "bed":
        return {
            "dimension_priority": "bed_footprint_first",
            "primary_axis_name": "footprint",
            "secondary_axis_name": "height",
            "primary_axis_distance": footprint_distance,
            "secondary_axis_distance": height_distance,
            "overall_size_distance": overall_distance,
        }

    return {
        "dimension_priority": "overall_size_first",
        "primary_axis_name": "overall_size",
        "secondary_axis_name": "height",
        "primary_axis_distance": overall_distance,
        "secondary_axis_distance": height_distance,
        "overall_size_distance": overall_distance,
    }


def _axis_distance_info(target: dict[str, Any], row: dict[str, Any], fit_breakdown: dict[str, Any]) -> dict[str, Any]:
    target_size_m = _effective_target_size_m(target)
    candidate_size_m = _product_size_m(row)
    if candidate_size_m is None or len(target_size_m) != 3:
        return {
            "axis_sort_order": ["width", "height", "depth"],
            "oriented_candidate_size_m": None,
            "width_distance": float("inf"),
            "height_distance": float("inf"),
            "depth_distance": float("inf"),
        }

    tw, td, th = [float(x) for x in target_size_m]
    cw, cd, ch = [float(x) for x in candidate_size_m]
    orientation = str(fit_breakdown.get("bbox_fit_orientation") or fit_breakdown.get("rescalable_fit_orientation") or "direct")
    if orientation == "swapped_xy":
        ow, od = cd, cw
    else:
        ow, od = cw, cd
    return {
        "axis_sort_order": ["width", "height", "depth"],
        "oriented_candidate_size_m": [round(ow, 4), round(od, 4), round(ch, 4)],
        "width_distance": _axis_log_distance(tw, ow),
        "height_distance": _axis_log_distance(th, ch),
        "depth_distance": _axis_log_distance(td, od),
    }


def _min_fill_policy(group: str) -> dict[str, float] | None:
    policies: dict[str, dict[str, float]] = {
        "bed": {
            "width_min_ratio": 0.68,
            "depth_min_ratio": 0.68,
            "height_min_ratio": 0.45,
        },
        "desk": {
            "width_min_ratio": 0.55,
            "depth_min_ratio": 0.55,
            "height_min_ratio": 0.75,
        },
        "side_table": {
            "width_min_ratio": 0.55,
            "depth_min_ratio": 0.55,
            "height_min_ratio": 0.75,
        },
        "coffee_table": {
            "width_min_ratio": 0.55,
            "depth_min_ratio": 0.55,
            "height_min_ratio": 0.7,
        },
        "dresser": {
            "width_min_ratio": 0.45,
            "depth_min_ratio": 0.35,
            "height_min_ratio": 0.55,
        },
        "tv_stand": {
            "width_min_ratio": 0.4,
            "depth_min_ratio": 0.3,
            "height_min_ratio": 0.45,
        },
        "shelf": {
            "width_min_ratio": 0.45,
            "depth_min_ratio": 0.28,
            "height_min_ratio": 0.55,
        },
        "wardrobe": {
            "width_min_ratio": 0.55,
            "depth_min_ratio": 0.4,
            "height_min_ratio": 0.75,
        },
        "lamp_floor": {
            "width_min_ratio": 0.25,
            "depth_min_ratio": 0.25,
            "height_min_ratio": 0.35,
        },
        "lamp_ceiling": {
            "width_min_ratio": 0.2,
            "depth_min_ratio": 0.2,
            "height_min_ratio": 0.05,
        },
        "lamp_table": {
            "width_min_ratio": 0.3,
            "depth_min_ratio": 0.3,
            "height_min_ratio": 0.35,
        },
    }
    return policies.get(group)


def _bbox_fill_info(target: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    target_group = str(target.get("semantic_group") or "")
    target_size_m = _effective_target_size_m(target)
    candidate_size_m = _product_size_m(row)
    if candidate_size_m is None or len(target_size_m) != 3:
        return {
            "fill_policy_applied": False,
            "passes_min_fill": False,
        }

    tw, td, th = [max(float(x), 1e-6) for x in target_size_m]
    cw, cd, ch = [max(float(x), 1e-6) for x in candidate_size_m]
    policy = _min_fill_policy(target_group)
    if not policy:
        return {
            "fill_policy_applied": False,
            "passes_min_fill": True,
            "width_fill_ratio": round(cw / tw, 6),
            "depth_fill_ratio": round(cd / td, 6),
            "height_fill_ratio": round(ch / th, 6),
        }

    direct = {
        "width_fill_ratio": cw / tw,
        "depth_fill_ratio": cd / td,
        "height_fill_ratio": ch / th,
        "fill_orientation": "direct",
    }
    swapped = {
        "width_fill_ratio": cd / tw,
        "depth_fill_ratio": cw / td,
        "height_fill_ratio": ch / th,
        "fill_orientation": "swapped_xy",
    }

    def passes(candidate: dict[str, float]) -> bool:
        return (
            candidate["width_fill_ratio"] >= policy["width_min_ratio"]
            and candidate["depth_fill_ratio"] >= policy["depth_min_ratio"]
            and candidate["height_fill_ratio"] >= policy["height_min_ratio"]
        )

    candidates = [direct, swapped]
    candidates.sort(
        key=lambda item: (
            0 if passes(item) else 1,
            -min(item["width_fill_ratio"], item["depth_fill_ratio"], item["height_fill_ratio"]),
            -(item["width_fill_ratio"] + item["depth_fill_ratio"] + item["height_fill_ratio"]),
        )
    )
    best = candidates[0]
    return {
        "fill_policy_applied": True,
        "passes_min_fill": passes(best),
        "fill_orientation": best["fill_orientation"],
        "width_fill_ratio": round(best["width_fill_ratio"], 6),
        "depth_fill_ratio": round(best["depth_fill_ratio"], 6),
        "height_fill_ratio": round(best["height_fill_ratio"], 6),
        "min_fill_policy": {
            "width_min_ratio": policy["width_min_ratio"],
            "depth_min_ratio": policy["depth_min_ratio"],
            "height_min_ratio": policy["height_min_ratio"],
        },
    }


def _infer_row_group(row: dict[str, Any]) -> str:
    category_norm = str(row.get("category_norm") or "").strip().lower()
    title = row.get("title")
    category_raw = row.get("category_raw")

    if category_norm:
        direct_map = {
            "bed": "bed",
            "nightstand": "nightstand",
            "wardrobe": "wardrobe",
            "dresser": "dresser",
            "cabinet": "dresser",
            "sideboard": "dresser",
            "desk": "desk",
            "tv_stand": "tv_stand",
            "computer": "computer",
            "computer_monitor": "computer",
            "laptop_computer_keyboard_mouse": "computer",
            "armchair": "armchair",
            "chair": "chair",
            "sofa": "sofa",
            "coffee_table": "coffee_table",
            "side_table": "side_table",
            "shelf": "shelf",
            "bookcase": "shelf",
            "plant": "plant",
            "mirror": "mirror",
            "table": "coffee_table",
            "floor_lamp": "lamp_floor",
            "chandelier": "lamp_ceiling",
            "lamp_table": "lamp_table",
            "lighting": "lamp_ceiling",
            "light": "lamp_ceiling",
        }
        if category_norm in direct_map:
            return direct_map[category_norm]

    return _semantic_group(title, category_raw, {})


def _category_match_info(target: dict[str, Any], row: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    target_group = str(target.get("semantic_group") or "")
    candidate_group = str(row.get("semantic_group") or "")
    target_tokens = _target_category_tokens(target)
    row_tokens = _row_category_tokens(row)
    overlap = target_tokens & row_tokens

    breakdown: dict[str, Any] = {
        "target_group": target_group,
        "candidate_group": candidate_group,
        "category_overlap_count": len(overlap),
        "category_overlap_tokens": sorted(overlap)[:20],
    }

    if target_group == candidate_group:
        breakdown["category_match"] = "exact_group"
        return 0, breakdown
    if _target_requires_exact_group(target_group):
        breakdown["category_match"] = "exact_group_required_mismatch"
        return 3, breakdown
    if target_group and candidate_group and _same_family(target_group, candidate_group):
        breakdown["category_match"] = "same_family"
        return 1, breakdown
    breakdown["category_match"] = "mismatch"
    return 3, breakdown


def _size_match_info(target: dict[str, Any], row: dict[str, Any]) -> tuple[int, float, dict[str, Any]]:
    target_group = str(target.get("semantic_group") or "")
    target_size_m = _effective_target_size_m(target)
    candidate_size_m = _product_size_m(row)
    breakdown: dict[str, Any] = {
        "target_size_m": [round(float(x), 4) for x in target_size_m],
        "candidate_size_m": None,
        "size_distance": None,
        "fits_bbox": False,
        "fit_checked": False,
    }
    if candidate_size_m is None:
        return 2, float("inf"), breakdown

    dist = _size_distance(target_size_m, candidate_size_m)
    breakdown["candidate_size_m"] = [round(float(x), 4) for x in candidate_size_m]
    breakdown["size_distance"] = round(dist, 6)
    fits_bbox, fit_breakdown = _fits_inside_bbox(target_size_m, candidate_size_m)
    breakdown.update(fit_breakdown)
    if not fits_bbox:
        soft_fit, soft_breakdown = _passes_rescalable_fit(target_group, target_size_m, candidate_size_m)
        breakdown.update(soft_breakdown)
        if not soft_fit:
            breakdown["bbox_fit_rank"] = 1
            return 1, dist, breakdown
        breakdown["fits_bbox"] = True
        breakdown["bbox_fit_rank"] = 0
        breakdown["bbox_fit_mode"] = "rescalable_anchor_fit"
        breakdown["bbox_fit_orientation"] = soft_breakdown.get("rescalable_fit_orientation")
        return 0, dist, breakdown
    breakdown["bbox_fit_rank"] = 0
    return 0, dist, breakdown


def _design_match_info(target: dict[str, Any], row: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    target_tokens = _target_design_tokens(target)
    row_tokens = _row_design_tokens(row)
    overlap = target_tokens & row_tokens
    union = target_tokens | row_tokens

    constraints = target.get("constraints") or {}
    material_tokens = _normalize_text_tokens(constraints.get("material")) | _normalize_text_tokens(constraints.get("materials"))
    color_tokens = _normalize_text_tokens(constraints.get("color"))
    style_tokens = _normalize_text_tokens(constraints.get("style"))
    brand_tokens = _normalize_text_tokens(constraints.get("brand"))

    material_match = bool(material_tokens and (material_tokens & _normalize_text_tokens(row.get("materials"))))
    color_match = bool(color_tokens and (color_tokens & _normalize_text_tokens(row.get("color"))))
    style_match = bool(style_tokens and (style_tokens & _normalize_text_tokens(row.get("style"))))
    brand_match = bool(brand_tokens and (brand_tokens & _normalize_text_tokens(row.get("brand"))))

    overlap_ratio = (len(overlap) / max(len(target_tokens), 1)) if target_tokens else 0.0
    jaccard = (len(overlap) / max(len(union), 1)) if union else 0.0
    design_score = overlap_ratio * 100.0 + jaccard * 20.0
    if material_match:
        design_score += 20.0
    if color_match:
        design_score += 10.0
    if style_match:
        design_score += 15.0
    if brand_match:
        design_score += 8.0

    breakdown = {
        "design_score": round(design_score, 6),
        "design_overlap_count": len(overlap),
        "design_overlap_tokens": sorted(overlap)[:20],
        "material_match": material_match,
        "color_match": color_match,
        "style_match": style_match,
        "brand_match": brand_match,
    }
    return design_score, breakdown


def _row_style_llm_info(row: dict[str, Any]) -> dict[str, Any]:
    style = _normalize_style_label(row.get("style_llm"))
    style_source = "style_llm"
    if style is None:
        style = _normalize_style_label(row.get("style"))
        style_source = "style"
    secondary_raw = row.get("style_llm_secondary") or []
    if isinstance(secondary_raw, str):
        secondary_raw = _json_loads_or(secondary_raw, [])
        if isinstance(secondary_raw, str):
            secondary_raw = [secondary_raw]
    secondary = [
        normalized
        for normalized in (_normalize_style_label(x) for x in (secondary_raw if isinstance(secondary_raw, list) else []))
        if normalized
    ]
    confidence = _safe_float(row.get("style_llm_confidence"))
    quality = _safe_float(row.get("style_llm_quality_score"))
    if style_source == "style" and style is not None:
        confidence = 0.72 if confidence is None else confidence
        quality = 7.0 if quality is None else quality
    usable = (
        style is not None
        and confidence is not None
        and quality is not None
        and confidence >= STYLE_LLM_MIN_CONFIDENCE
        and quality >= STYLE_LLM_MIN_QUALITY
    )
    return {
        "style": style,
        "secondary": _dedup_keep_order(secondary),
        "confidence": confidence,
        "quality": quality,
        "usable": usable,
        "flags": row.get("style_llm_quality_flags") or [],
        "source": style_source,
    }


def _target_style_labels(target: dict[str, Any], context: dict[str, Any] | None) -> set[str]:
    context = context or {}
    constraints = target.get("constraints") or {}
    labels: set[str] = set()
    for part in (
        constraints.get("style"),
        constraints.get("theme"),
        context.get("style_label"),
        context.get("room_style_hint"),
        context.get("prompt_text"),
    ):
        labels |= _extract_styles_from_text(part)
    return labels


def _style_match_info(target: dict[str, Any], row: dict[str, Any], context: dict[str, Any] | None) -> tuple[int, float, dict[str, Any]]:
    target_styles = _target_style_labels(target, context)
    row_info = _row_style_llm_info(row)
    row_styles = {row_info["style"]} if row_info.get("style") else set()
    row_styles |= {x for x in row_info.get("secondary") or [] if x}

    if not target_styles:
        rank = 2
        match = "no_target_style"
    elif not row_info.get("usable") or not row_styles:
        rank = 2
        match = "candidate_style_unknown_or_low_quality"
    elif target_styles & row_styles:
        rank = 0
        match = "exact_style"
    elif any((STYLE_COMPATIBILITY.get(target_style) or set()) & row_styles for target_style in target_styles):
        rank = 1
        match = "compatible_style"
    else:
        rank = 3
        match = "style_mismatch"

    confidence = float(row_info.get("confidence") or 0.0)
    quality = float(row_info.get("quality") or 0.0)
    strength = confidence + (quality / 10.0)
    if rank == 0:
        score = 100.0 + strength * 10.0
    elif rank == 1:
        score = 70.0 + strength * 10.0
    elif rank == 2:
        score = 20.0 + strength * 2.0
    else:
        score = strength

    return rank, score, {
        "style_selection_target_styles": sorted(target_styles),
        "style_selection_candidate_style": row_info.get("style"),
        "style_selection_candidate_secondary": row_info.get("secondary") or [],
        "style_selection_confidence": row_info.get("confidence"),
        "style_selection_quality_score": row_info.get("quality"),
        "style_selection_usable": bool(row_info.get("usable")),
        "style_selection_match": match,
        "style_selection_rank": rank,
        "style_selection_score": round(score, 6),
    }


def _price_rank_info(row: dict[str, Any]) -> dict[str, Any]:
    price = _safe_float(row.get("price_value"))
    return {
        "price_known_rank": 0 if price is not None else 1,
        "price_sort_value": round(price, 6) if price is not None else 999999999999.0,
        "price_sort_value_desc": round(-price, 6) if price is not None else 999999999999.0,
    }


def _selection_strategy(context: dict[str, Any] | None) -> str:
    strategy = str((context or {}).get("supplier_selection_strategy") or "balanced").strip().lower()
    if strategy not in SUPPLIER_SELECTION_STRATEGIES:
        return "balanced"
    return strategy


def _prompt_match_info(target: dict[str, Any], row: dict[str, Any], context: dict[str, Any] | None) -> tuple[float, dict[str, Any]]:
    context = context or {}
    prompt_tokens = set(context.get("prompt_tokens") or set())
    room_style_tokens = set(context.get("room_style_tokens") or set())
    target_tokens = _target_query_tokens(target)
    row_tokens = _row_query_tokens(row)
    row_color_tokens = _row_color_tokens(row)
    target_color_tokens = _target_color_tokens(target)

    prompt_overlap = prompt_tokens & row_tokens
    room_style_overlap = room_style_tokens & row_tokens
    color_overlap = target_color_tokens & row_color_tokens

    score = 0.0
    if prompt_tokens:
        score += 30.0 * (len(prompt_overlap) / max(len(prompt_tokens), 1))
    if room_style_tokens:
        score += 18.0 * (len(room_style_overlap) / max(len(room_style_tokens), 1))
    if color_overlap:
        score += 24.0

    breakdown = {
        "prompt_score": round(score, 6),
        "prompt_overlap_count": len(prompt_overlap),
        "prompt_overlap_tokens": sorted(prompt_overlap)[:20],
        "room_style_overlap_count": len(room_style_overlap),
        "room_style_overlap_tokens": sorted(room_style_overlap)[:20],
        "target_color_tokens": sorted(target_color_tokens),
        "row_color_tokens": sorted(row_color_tokens),
        "color_match": bool(color_overlap),
        "color_overlap_tokens": sorted(color_overlap),
    }
    return score, breakdown


def _query_match_info(target: dict[str, Any], row: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    target_tokens = _target_query_tokens(target)
    row_tokens = _row_query_tokens(row)
    overlap = target_tokens & row_tokens
    union = target_tokens | row_tokens
    overlap_ratio = (len(overlap) / max(len(target_tokens), 1)) if target_tokens else 0.0
    jaccard = (len(overlap) / max(len(union), 1)) if union else 0.0
    score = overlap_ratio * 40.0 + min(len(overlap), 3) * 10.0 + jaccard * 20.0
    breakdown = {
        "query_score": round(score, 6),
        "query_overlap_count": len(overlap),
        "query_overlap_tokens": sorted(overlap)[:20],
        "target_query_tokens": sorted(target_tokens)[:20],
        "row_query_tokens": sorted(row_tokens)[:40],
    }
    return score, breakdown


def _user_preference_match_info(target: dict[str, Any], row: dict[str, Any], context: dict[str, Any] | None) -> tuple[bool, float, dict[str, Any]]:
    preferences = _target_user_preferences(target, context)
    row_site = str(row.get("source_site") or "").strip().lower()
    row_color_tokens = _row_color_tokens(row)
    row_brand_tokens = _normalize_text_tokens(row.get("brand"))
    row_price = _safe_float(row.get("price_value"))
    max_price_rub = _safe_float(preferences.get("max_price_rub"))
    preferred_color_tokens = set(_normalize_color_token(x) for x in (preferences.get("preferred_colors") or []))
    avoid_color_tokens = set(_normalize_color_token(x) for x in (preferences.get("avoid_colors") or []))

    preferred_brand_tokens: set[str] = set()
    for brand in preferences.get("preferred_brands") or []:
        preferred_brand_tokens |= _normalize_text_tokens(brand)

    allowed_sites = {str(x).strip().lower() for x in (preferences.get("allowed_sites") or []) if str(x).strip()}
    disallowed_sites = {str(x).strip().lower() for x in (preferences.get("disallowed_sites") or []) if str(x).strip()}

    reject_reason = None
    if allowed_sites and row_site not in allowed_sites:
        reject_reason = "site_not_allowed_by_user_preferences"
    elif disallowed_sites and row_site in disallowed_sites:
        reject_reason = "site_explicitly_disallowed_by_user_preferences"
    elif max_price_rub is not None and row_price is not None and row_price > max_price_rub:
        reject_reason = "price_above_user_max_price"
    elif avoid_color_tokens and (avoid_color_tokens & row_color_tokens):
        reject_reason = "color_explicitly_avoided_by_user_preferences"
    elif preferences.get("strict_color") and preferred_color_tokens and not (preferred_color_tokens & row_color_tokens):
        reject_reason = "strict_color_requested_but_candidate_color_mismatch"
    elif preferences.get("require_real_asset"):
        has_real_asset = bool(row.get("asset_local_path")) and str(row.get("asset_format") or "").lower() in {"fbx", "glb", "gltf", "obj"}
        if not has_real_asset:
            reject_reason = "real_asset_required_by_user_preferences"
    elif preferences.get("require_model_url") and not (row.get("model_download_url") or row.get("model_page_url")):
        reject_reason = "model_url_required_by_user_preferences"

    preferred_color_overlap = preferred_color_tokens & row_color_tokens
    preferred_brand_overlap = preferred_brand_tokens & row_brand_tokens

    bonus = 0.0
    if preferred_color_overlap:
        bonus += 40.0
    if preferred_brand_overlap:
        bonus += 18.0
    if max_price_rub is not None and row_price is not None and row_price <= max_price_rub:
        bonus += 10.0 * max(0.0, 1.0 - (row_price / max(max_price_rub, 1.0)))

    breakdown = {
        "user_preferences_applied": any(
            (
                max_price_rub is not None,
                bool(preferred_color_tokens),
                bool(avoid_color_tokens),
                bool(preferred_brand_tokens),
                bool(allowed_sites),
                bool(disallowed_sites),
                bool(preferences.get("strict_color")),
                bool(preferences.get("require_real_asset")),
                bool(preferences.get("require_model_url")),
            )
        ),
        "user_preference_reject_reason": reject_reason,
        "user_preference_score": round(bonus, 6),
        "user_max_price_rub": max_price_rub,
        "user_preferred_colors": sorted(preferred_color_tokens),
        "user_avoid_colors": sorted(avoid_color_tokens),
        "user_preferred_brands": list(preferences.get("preferred_brands") or []),
        "user_allowed_sites": sorted(allowed_sites),
        "user_disallowed_sites": sorted(disallowed_sites),
        "user_strict_color": bool(preferences.get("strict_color")),
        "user_require_real_asset": bool(preferences.get("require_real_asset")),
        "user_require_model_url": bool(preferences.get("require_model_url")),
        "user_preferred_color_overlap": sorted(preferred_color_overlap),
        "user_preferred_brand_overlap": sorted(preferred_brand_overlap),
    }
    return reject_reason is None, bonus, breakdown


def _row_extra_dict(row: dict[str, Any]) -> dict[str, Any]:
    extra = row.get("extra")
    if isinstance(extra, dict):
        return extra
    parsed = _json_loads_or(row.get("extra_json"), {})
    return parsed if isinstance(parsed, dict) else {}


def _three_ddd_access_type(row: dict[str, Any]) -> str | None:
    extra = _row_extra_dict(row)
    raw_type = str(extra.get("api_type") or extra.get("typeText") or "").strip().lower()
    if raw_type:
        return raw_type

    availability = str(row.get("availability") or extra.get("status") or "").strip().lower()
    if availability == "pro":
        return "pro"
    if availability == "free":
        return "free"
    return None


def _source_policy_match_info(row: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    row_site = str(row.get("source_site") or "").strip().lower()
    reject_reason = None
    three_ddd_access_type = None

    if row_site == "3ddd":
        three_ddd_access_type = _three_ddd_access_type(row)
        if three_ddd_access_type not in {"free", "om"}:
            reject_reason = "3ddd_pro_model_disallowed"

    return reject_reason is None, {
        "source_policy_reject_reason": reject_reason,
        "three_ddd_access_type": three_ddd_access_type,
    }


def _extract_ollama_text(resp: dict[str, Any]) -> str:
    message = resp.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
    response_text = resp.get("response")
    if isinstance(response_text, str):
        return response_text.strip()
    return json.dumps(resp, ensure_ascii=False)


def _parse_json_object_from_text(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw).strip()
        raw = re.sub(r"```$", "", raw).strip()
    parsed = _json_loads_or(raw, None)
    if isinstance(parsed, dict):
        return parsed
    match = re.search(r"\{.*\}", raw, flags=re.S)
    if match:
        parsed = _json_loads_or(match.group(0), None)
        if isinstance(parsed, dict):
            return parsed
    raise RuntimeError("LLM did not return JSON object")


def _llm_candidate_payload(candidate: dict[str, Any]) -> dict[str, Any]:
    score_breakdown = candidate.get("score_breakdown") if isinstance(candidate.get("score_breakdown"), dict) else {}
    return {
        "unique_key": candidate.get("unique_key"),
        "title": candidate.get("title"),
        "brand": candidate.get("brand"),
        "collection": candidate.get("collection"),
        "source_site": candidate.get("source_site"),
        "price_value": candidate.get("price_value"),
        "price_currency": candidate.get("price_currency"),
        "color": candidate.get("color"),
        "materials": candidate.get("materials"),
        "style_llm": {
            "primary": candidate.get("style_llm"),
            "secondary": candidate.get("style_llm_secondary") or [],
            "confidence": candidate.get("style_llm_confidence"),
            "quality_score": candidate.get("style_llm_quality_score"),
            "quality_flags": candidate.get("style_llm_quality_flags") or [],
            "rationale": candidate.get("style_llm_rationale"),
        },
        "dimensions_cm": {
            "width": candidate.get("width_cm"),
            "depth": candidate.get("depth_cm"),
            "height": candidate.get("height_cm"),
        },
        "description": str(candidate.get("description") or "")[:500],
        "vlm_description": str(
            candidate.get("vlm_description_text")
            or candidate.get("vlm_description_summary")
            or ""
        )[:800],
        "vlm_visual": {
            "color": candidate.get("vlm_color"),
            "materials": candidate.get("vlm_materials"),
            "style": candidate.get("vlm_style"),
        },
        "heuristic": {
            "semantic_group": candidate.get("semantic_group"),
            "rich_card": candidate.get("rich_card"),
            "has_real_asset": bool(score_breakdown.get("has_real_asset")),
            "query_score": score_breakdown.get("query_score"),
            "prompt_score": score_breakdown.get("prompt_score"),
            "design_score": score_breakdown.get("design_score"),
            "width_distance": score_breakdown.get("width_distance"),
            "height_distance": score_breakdown.get("height_distance"),
            "depth_distance": score_breakdown.get("depth_distance"),
            "user_preference_score": score_breakdown.get("user_preference_score"),
            "style_selection_match": score_breakdown.get("style_selection_match"),
            "style_selection_rank": score_breakdown.get("style_selection_rank"),
            "style_selection_score": score_breakdown.get("style_selection_score"),
            "style_selection_target_styles": score_breakdown.get("style_selection_target_styles"),
            "price_known_rank": score_breakdown.get("price_known_rank"),
            "price_sort_value": score_breakdown.get("price_sort_value"),
        },
    }


def _llm_rerank_candidates(
    *,
    target: dict[str, Any],
    top_candidates: list[dict[str, Any]],
    context: dict[str, Any] | None,
    llm_settings: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    llm_settings = dict(llm_settings or {})
    if str(llm_settings.get("provider") or "none").strip().lower() == "none":
        return top_candidates, None
    if len(top_candidates) <= 1:
        return top_candidates, None

    provider = str(llm_settings.get("provider") or "none").strip().lower()
    if provider != "ollama":
        return top_candidates, {
            "status": "skipped",
            "reason": f"unsupported_provider:{provider}",
        }

    top_n = max(1, int(llm_settings.get("top_n") or len(top_candidates)))
    candidate_slice = top_candidates[:top_n]
    candidate_map = {str(candidate.get("unique_key") or ""): candidate for candidate in candidate_slice}
    if len(candidate_map) <= 1:
        return top_candidates, None

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
        return top_candidates, {
            "status": "failed",
            "reason": f"ollama_import_failed:{type(import_error).__name__ if import_error else 'RuntimeError'}:{import_error or 'chat_json_not_found'}",
        }

    def resolve_candidate_key(raw_key: Any) -> str | None:
        key = str(raw_key or "").strip()
        if not key or key.lower() == "string":
            return None
        if key in candidate_map:
            return key
        if key + " model" in candidate_map:
            return key + " model"
        if key.endswith(" model") and key[:-6] in candidate_map:
            return key[:-6]
        key_l = key.lower()
        matches: list[str] = []
        for candidate_key, candidate in candidate_map.items():
            candidate_key_l = candidate_key.lower()
            title_l = str(candidate.get("title") or "").strip().lower()
            if key_l and (key_l in candidate_key_l or (title_l and key_l in title_l)):
                matches.append(candidate_key)
        return matches[0] if len(matches) == 1 else None

    system_prompt = (
        "You are selecting the best supplier furniture replacement for an interior scene. "
        "Choose only from the provided candidates. "
        "The candidates were already filtered and ordered by a deterministic heuristic using category, size, "
        "bbox fit, asset availability, price, prompt context, and style_llm. "
        "When a candidate contains vlm_description/vlm_visual, treat it as the most direct evidence of the item's "
        "visible shape, color, material, and style from product photos. "
        "Pick the best final candidate from this shortlist only. "
        "The chosen_unique_key must be copied exactly from one candidate.unique_key; do not invent ids, titles, "
        "or placeholder strings. "
        "Respect selection_strategy: cheapest means keep price dominant after fit/type; cheap_style means "
        "prefer style-compatible candidates then lower price; style means prioritize style fit and ignore price. "
        "Also consider visual similarity, prompt/style match, and explicit user preferences such as budget and preferred colors. "
        "Prefer realistic assets over placeholders when all else is close. "
        "Return strict JSON only."
    )
    prompt_payload = {
        "room_prompt": (context or {}).get("prompt_text"),
        "room_style_hint": (context or {}).get("room_style_hint"),
        "style_label": (context or {}).get("style_label"),
        "selection_strategy": _selection_strategy(context),
        "target": {
            "target_id": target.get("target_id"),
            "name": target.get("name"),
            "category": target.get("category"),
            "semantic_group": target.get("semantic_group"),
            "size_m": target.get("size_m"),
            "color_rgb": target.get("color_rgb"),
            "constraints": target.get("constraints") or {},
            "user_preferences": _target_user_preferences(target, context),
        },
        "candidates": [_llm_candidate_payload(candidate) for candidate in candidate_slice],
        "required_output": {
            "chosen_unique_key": "string",
            "ordered_unique_keys": ["string"],
            "reason": "short explanation",
        },
    }
    schema = {
        "type": "object",
        "properties": {
            "chosen_unique_key": {"type": "string"},
            "ordered_unique_keys": {"type": "array", "items": {"type": "string"}},
            "reason": {"type": "string"},
        },
        "required": ["chosen_unique_key"],
        "additionalProperties": False,
    }

    try:
        response = chat_json(
            base_url=str(llm_settings.get("ollama_url") or "http://127.0.0.1:11434"),
            model=str(llm_settings.get("ollama_model") or "gpt-oss:20b"),
            system_prompt=system_prompt,
            user_prompt=json.dumps(prompt_payload, ensure_ascii=False, indent=2),
            json_schema=schema,
            timeout_sec=int(llm_settings.get("ollama_timeout") or 180),
            temperature=float(llm_settings.get("ollama_temperature") or 0.0),
            think="low",
        )
        text = _extract_ollama_text(response)
        parsed = _parse_json_object_from_text(text)
    except Exception as exc:
        return top_candidates, {
            "status": "failed",
            "reason": f"ollama_rerank_failed:{type(exc).__name__}:{exc}",
        }

    chosen_key = resolve_candidate_key(parsed.get("chosen_unique_key")) or ""
    ordered_keys = []
    for raw_key in parsed.get("ordered_unique_keys") or []:
        resolved_key = resolve_candidate_key(raw_key)
        if resolved_key and resolved_key not in ordered_keys:
            ordered_keys.append(resolved_key)
    if chosen_key not in candidate_map:
        return top_candidates, {
            "status": "failed",
            "reason": "ollama_returned_unknown_candidate",
            "raw_response": parsed,
        }

    if chosen_key not in ordered_keys:
        ordered_keys = [chosen_key] + [key for key in ordered_keys if key != chosen_key]
    ordered_keys += [key for key in candidate_map if key not in ordered_keys]

    llm_rank_map = {key: index + 1 for index, key in enumerate(ordered_keys)}
    reranked_slice = [dict(candidate_map[key], llm_rank=llm_rank_map[key]) for key in ordered_keys]
    for candidate in top_candidates[top_n:]:
        reranked_slice.append(candidate)

    return reranked_slice, {
        "status": "applied",
        "provider": provider,
        "model": str(llm_settings.get("ollama_model") or ""),
        "top_n": top_n,
        "chosen_unique_key": chosen_key,
        "ordered_unique_keys": ordered_keys,
        "reason": str(parsed.get("reason") or "").strip() or None,
    }


def _rank_candidate(target: dict[str, Any], row: dict[str, Any], context: dict[str, Any] | None = None) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
    source_policy_ok, source_policy_breakdown = _source_policy_match_info(row)
    if not source_policy_ok:
        return None

    category_rank, category_breakdown = _category_match_info(target, row)
    if category_rank >= 3:
        return None

    size_rank, size_distance, size_breakdown = _size_match_info(target, row)
    has_real_asset = _candidate_has_ready_real_asset(row)
    has_downloadable_asset = _candidate_has_downloadable_asset(row)
    relaxed_missing_size = bool(
        size_breakdown.get("candidate_size_m") is None
        and category_rank == 0
        and has_downloadable_asset
    )
    if size_rank != 0 and not relaxed_missing_size:
        return None
    fill_breakdown = _bbox_fill_info(target, row)
    if not fill_breakdown.get("passes_min_fill", False):
        if not relaxed_missing_size:
            return None
        fill_breakdown = {
            "fill_policy_applied": False,
            "passes_min_fill": True,
            "fill_orientation": None,
            "width_fill_ratio": None,
            "depth_fill_ratio": None,
            "height_fill_ratio": None,
            "min_fill_policy": None,
        }
    dimension_priority = _dimension_priority_info(target, row)
    axis_info = _axis_distance_info(target, row, size_breakdown)
    design_score, design_breakdown = _design_match_info(target, row)
    query_score, query_breakdown = _query_match_info(target, row)
    prompt_score, prompt_breakdown = _prompt_match_info(target, row, context)
    style_rank, style_score, style_breakdown = _style_match_info(target, row, context)
    price_breakdown = _price_rank_info(row)
    preferences_ok, preference_score, preference_breakdown = _user_preference_match_info(target, row, context)
    if not preferences_ok:
        return None
    rich_card = _row_is_rich(row)
    has_model_link = bool(row.get("model_download_url") or row.get("model_page_url"))
    text_overlap = len(query_breakdown.get("query_overlap_tokens") or [])
    asset_priority = 0 if has_real_asset else 1 if has_downloadable_asset else 2
    target_group = str(target.get("semantic_group") or "").strip()
    strict_group_asset_priority = asset_priority if _target_requires_exact_group(target_group) else 0
    effective_size_rank = 1 if relaxed_missing_size else size_rank
    width_distance = round(float(axis_info.get("width_distance") or 999999.0), 6)
    height_distance = round(float(axis_info.get("height_distance") or 999999.0), 6)
    depth_distance = round(float(axis_info.get("depth_distance") or 999999.0), 6)
    if relaxed_missing_size:
        width_distance = 999999.0
        height_distance = 999999.0
        depth_distance = 999999.0

    target_constraints = target.get("constraints") or {}
    target_budget = target_constraints.get("budget_rub")
    if target_budget is None:
        target_budget = target_constraints.get("price_value")
    price_distance_ratio = None
    if target_budget is not None and row.get("price_value") is not None:
        try:
            price_distance_ratio = abs(float(row["price_value"]) - float(target_budget)) / max(abs(float(target_budget)), 1.0)
        except Exception:
            price_distance_ratio = None

    structural_prefix = (
        category_rank,
        strict_group_asset_priority,
        effective_size_rank,
        asset_priority,
    )
    dimension_key = (
        width_distance,
        height_distance,
        depth_distance,
    )
    similarity_suffix_no_price = (
        -round(query_score, 6),
        -round(preference_score, 6),
        -round(prompt_score, 6),
        style_rank,
        -round(style_score, 6),
        -round(design_score, 6),
        0 if rich_card else 1,
        0 if has_model_link else 1,
        -text_overlap,
        str(row.get("unique_key") or ""),
    )
    similarity_suffix_with_budget = similarity_suffix_no_price[:-2] + (
        round(price_distance_ratio if price_distance_ratio is not None else 999999.0, 6),
    ) + similarity_suffix_no_price[-2:]
    strategy = _selection_strategy(context)
    if strategy == "cheapest":
        rank_key = structural_prefix + (
            price_breakdown["price_known_rank"],
            price_breakdown["price_sort_value"],
        ) + dimension_key + similarity_suffix_no_price
    elif strategy == "cheap_style":
        rank_key = structural_prefix + (
            style_rank,
            -round(style_score, 6),
            price_breakdown["price_known_rank"],
            price_breakdown["price_sort_value"],
        ) + dimension_key + similarity_suffix_no_price
    elif strategy == "style":
        rank_key = structural_prefix + (
            style_rank,
            -round(style_score, 6),
        ) + dimension_key + similarity_suffix_no_price
    else:
        rank_key = structural_prefix + (
            style_rank,
            -round(style_score, 6),
            -round(preference_score, 6),
            -round(prompt_score, 6),
            -round(query_score, 6),
        ) + dimension_key + (
            -round(design_score, 6),
            round(price_distance_ratio if price_distance_ratio is not None else 999999.0, 6),
            0 if rich_card else 1,
            0 if has_model_link else 1,
            -text_overlap,
            str(row.get("unique_key") or ""),
        )

    reasons: dict[str, Any] = {
        **category_breakdown,
        **size_breakdown,
        **fill_breakdown,
        **{
            "dimension_priority": dimension_priority.get("dimension_priority"),
            "primary_axis_name": dimension_priority.get("primary_axis_name"),
            "secondary_axis_name": dimension_priority.get("secondary_axis_name"),
            "primary_axis_distance": round(float(dimension_priority.get("primary_axis_distance") or 999999.0), 6),
            "secondary_axis_distance": round(float(dimension_priority.get("secondary_axis_distance") or 999999.0), 6),
            "axis_sort_order": axis_info.get("axis_sort_order"),
            "oriented_candidate_size_m": axis_info.get("oriented_candidate_size_m"),
            "width_distance": round(float(axis_info.get("width_distance") or 999999.0), 6),
            "height_distance": round(float(axis_info.get("height_distance") or 999999.0), 6),
            "depth_distance": round(float(axis_info.get("depth_distance") or 999999.0), 6),
        },
        **design_breakdown,
        **query_breakdown,
        **style_breakdown,
        **price_breakdown,
        **source_policy_breakdown,
        **preference_breakdown,
        **prompt_breakdown,
        "category_rank": category_rank,
        "bbox_fit_rank": effective_size_rank,
        "size_missing": bool(size_breakdown.get("candidate_size_m") is None),
        "relaxed_missing_size_match": relaxed_missing_size,
        "rich_card": rich_card,
        "has_real_asset": has_real_asset,
        "has_downloadable_asset": has_downloadable_asset,
        "has_model_url": has_model_link,
        "strict_group_asset_priority": strict_group_asset_priority,
        "text_overlap_count": text_overlap,
        "price_distance_ratio": round(price_distance_ratio, 6) if price_distance_ratio is not None else None,
        "supplier_selection_strategy": strategy,
        "ranking_order": [
            "category",
            "bbox_fit",
            "asset_ready",
            "style_llm",
            "user_preferences",
            "prompt_color",
            "query_match",
            "width",
            "height",
            "depth",
            "strategy_price_style",
            "design",
        ],
        "rank_key": [x for x in rank_key[:-1]],
    }
    return rank_key, reasons


def _candidate_has_viable_asset_hint(candidate: dict[str, Any]) -> tuple[bool, str]:
    if _candidate_has_ready_real_asset(candidate):
        return True, "ready_real_asset"

    asset_status = str(candidate.get("asset_status") or "").strip().lower()
    asset_format = str(candidate.get("asset_format") or "").strip().lower().lstrip(".")
    model_format = str(candidate.get("model_format") or "").strip().lower().lstrip(".")

    if asset_status in LOW_QUALITY_ASSET_STATUSES:
        return False, f"asset_status:{asset_status}"
    if asset_format == "max" or model_format == "max":
        return False, "max_only_asset"
    if asset_format in SUPPORTED_LOCAL_ASSET_EXTS:
        return True, f"local_asset:{asset_format}"
    if model_format in SUPPORTED_LOCAL_ASSET_EXTS | {"zip", "rar", "7z"}:
        return True, f"downloadable_asset:{model_format}"
    if _candidate_has_downloadable_asset(candidate):
        return True, "downloadable_asset"
    return False, "no_viable_asset"


def _candidate_acceptance_thresholds(group: str, size_missing: bool) -> dict[str, float]:
    stage = "missing" if size_missing else "known"
    thresholds = dict(_ACCEPTANCE_DEFAULTS.get(stage, {}))
    thresholds.update((_ACCEPTANCE_BY_GROUP.get(group) or {}).get(stage, {}))
    return thresholds


def _candidate_acceptability(
    target: dict[str, Any],
    candidate: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    breakdown = candidate.get("score_breakdown") if isinstance(candidate.get("score_breakdown"), dict) else {}
    target_group = str(target.get("semantic_group") or "").strip()
    size_missing = bool(breakdown.get("size_missing"))
    thresholds = _candidate_acceptance_thresholds(target_group, size_missing)
    asset_ok, asset_reason = _candidate_has_viable_asset_hint(candidate)
    query_score = float(breakdown.get("query_score") or 0.0)
    query_overlap_count = int(breakdown.get("query_overlap_count") or 0)
    primary_axis_distance = float(breakdown.get("primary_axis_distance") or 999999.0)
    secondary_axis_distance = float(breakdown.get("secondary_axis_distance") or 999999.0)
    category_match = str(breakdown.get("category_match") or "")

    accept = True
    reject_reason = None
    if not size_missing:
        if not breakdown.get("passes_min_fill", False):
            accept = False
            reject_reason = "fails_min_fill"
        elif category_match not in {"exact_group", "same_family"}:
            accept = False
            reject_reason = f"category_match:{category_match or 'unknown'}"
        elif primary_axis_distance > float(thresholds.get("max_primary_axis_distance") or 999999.0):
            accept = False
            reject_reason = "primary_axis_distance_too_large"
        elif secondary_axis_distance > float(thresholds.get("max_secondary_axis_distance") or 999999.0):
            accept = False
            reject_reason = "secondary_axis_distance_too_large"
        elif query_score < float(thresholds.get("min_query_score") or 0.0):
            accept = False
            reject_reason = "query_match_too_weak"
    else:
        if category_match != "exact_group":
            accept = False
            reject_reason = f"missing_size_requires_exact_group:{category_match or 'unknown'}"
        elif not breakdown.get("relaxed_missing_size_match", False):
            accept = False
            reject_reason = "missing_size_without_relaxed_match"
        elif query_overlap_count < int(thresholds.get("min_query_overlap_count") or 0):
            accept = False
            reject_reason = "missing_size_without_query_overlap"
        elif query_score < float(thresholds.get("min_query_score") or 0.0):
            accept = False
            reject_reason = "missing_size_query_match_too_weak"

    return accept, {
        "accepted": accept,
        "reject_reason": reject_reason,
        "asset_ok": asset_ok,
        "asset_reason": asset_reason,
        "thresholds": thresholds,
        "size_missing": size_missing,
        "query_score": round(query_score, 6),
        "query_overlap_count": query_overlap_count,
        "primary_axis_distance": round(primary_axis_distance, 6) if math.isfinite(primary_axis_distance) else None,
        "secondary_axis_distance": round(secondary_axis_distance, 6) if math.isfinite(secondary_axis_distance) else None,
    }


def _table_exists(con: sqlite3.Connection, table_name: str) -> bool:
    row = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table_name,),
    ).fetchone()
    return row is not None


def load_supplier_catalog(
    db_paths: list[Path],
    sites: set[str] | None = None,
    rich_only: bool = False,
) -> list[dict[str, Any]]:
    rows_out: list[dict[str, Any]] = []

    for db_path in db_paths:
        with sqlite3.connect(db_path) as con:
            con.row_factory = sqlite3.Row
            has_mesh_catalog = _table_exists(con, "supplier_mesh_catalog")
            has_asset = _table_exists(con, "supplier_asset")

            if has_mesh_catalog:
                sql = """
                SELECT
                    smc.unique_key,
                    smc.source_site,
                    smc.source_url,
                    smc.parsed_at,
                    smc.external_id,
                    smc.category_raw,
                    smc.category_norm,
                    smc.title,
                    smc.brand,
                    smc.collection,
                    smc.product_url,
                    smc.model_link_type,
                    smc.model_download_url,
                    smc.model_download_landing_url,
                    smc.model_vendor_url,
                    smc.model_extraction_method,
                    smc.model_download_filename,
                    smc.model_format,
                    smc.model_page_url,
                    smc.price_value,
                    smc.price_currency,
                    smc.style,
                    smc.color,
                    smc.description,
                    smc.width_cm,
                    smc.depth_cm,
                    smc.height_cm,
                    smc.materials,
                    smc.room,
                    smc.availability,
                    smc.images_json,
                    smc.extra_json,
                    smc.mesh_status AS asset_status,
                    smc.mesh_format AS asset_format,
                    smc.mesh_local_path AS asset_local_path,
                    smc.semantic_group
                FROM supplier_mesh_catalog smc
                """
                for row in con.execute(sql):
                    item = dict(row)
                    if sites and str(item.get("source_site") or "") not in sites:
                        continue
                    if not item.get("title"):
                        continue
                    if not item.get("semantic_group"):
                        item["semantic_group"] = _infer_row_group(item)
                    if rich_only and not _row_is_rich(item):
                        continue
                    rows_out.append(item)
                continue

            if has_asset:
                sql = """
                SELECT
                    sp.unique_key,
                    sp.source_site,
                    sp.source_url,
                    sp.parsed_at,
                    sp.external_id,
                    sp.category_raw,
                    sp.category_norm,
                    sp.title,
                    sp.brand,
                    sp.collection,
                    sp.product_url,
                    sp.model_link_type,
                    sp.model_download_url,
                    sp.model_download_landing_url,
                    sp.model_vendor_url,
                    sp.model_extraction_method,
                    sp.model_download_filename,
                    sp.model_format,
                    sp.model_page_url,
                    sp.price_value,
                    sp.price_currency,
                    sp.style,
                    sp.color,
                    sp.description,
                    sp.width_cm,
                    sp.depth_cm,
                    sp.height_cm,
                    sp.materials,
                    sp.room,
                    sp.availability,
                    sp.images_json,
                    sp.extra_json,
                    sa.asset_status,
                    sa.asset_format,
                    sa.asset_local_path
                FROM supplier_product sp
                LEFT JOIN supplier_asset sa ON sa.unique_key = sp.unique_key
                """
            else:
                sql = """
                SELECT
                    sp.unique_key,
                    sp.source_site,
                    sp.source_url,
                    sp.parsed_at,
                    sp.external_id,
                    sp.category_raw,
                    sp.category_norm,
                    sp.title,
                    sp.brand,
                    sp.collection,
                    sp.product_url,
                    sp.model_link_type,
                    sp.model_download_url,
                    sp.model_download_landing_url,
                    sp.model_vendor_url,
                    sp.model_extraction_method,
                    sp.model_download_filename,
                    sp.model_format,
                    sp.model_page_url,
                    sp.price_value,
                    sp.price_currency,
                    sp.style,
                    sp.color,
                    sp.description,
                    sp.width_cm,
                    sp.depth_cm,
                    sp.height_cm,
                    sp.materials,
                    sp.room,
                    sp.availability,
                    sp.images_json,
                    sp.extra_json,
                    NULL AS asset_status,
                    NULL AS asset_format,
                    NULL AS asset_local_path
                FROM supplier_product sp
                """

            for row in con.execute(sql):
                item = dict(row)
                if sites and str(item.get("source_site") or "") not in sites:
                    continue
                if not item.get("title"):
                    continue
                item["semantic_group"] = _infer_row_group(item)
                if rich_only and not _row_is_rich(item):
                    continue
                rows_out.append(item)

    return rows_out


def load_supplier_catalog_json(
    json_paths: list[Path],
    sites: set[str] | None = None,
    rich_only: bool = False,
) -> list[dict[str, Any]]:
    rows_out: list[dict[str, Any]] = []

    for json_path in json_paths:
        data = read_json(json_path)
        items = data.get("items") if isinstance(data, dict) else data
        if not isinstance(items, list):
            if isinstance(data, dict) and isinstance(data.get("bindings"), list):
                raise RuntimeError(
                    f"Ожидался supplier catalog JSON с items[], а получен bindings-файл: {json_path}"
                )
            raise RuntimeError(f"Некорректный supplier catalog JSON: {json_path}")

        for item in items:
            if not isinstance(item, dict):
                continue
            dims = item.get("dimensions_cm") or {}
            row = {
                "unique_key": item.get("unique_key"),
                "source_site": item.get("source_site"),
                "source_url": item.get("source_url"),
                "parsed_at": item.get("parsed_at"),
                "external_id": item.get("external_id"),
                "category_raw": item.get("category_raw"),
                "category_norm": item.get("category_norm"),
                "title": item.get("title"),
                "brand": item.get("brand"),
                "collection": item.get("collection"),
                "product_url": item.get("product_url"),
                "model_link_type": item.get("model_link_type"),
                "model_download_url": item.get("model_download_url"),
                "model_download_landing_url": item.get("model_download_landing_url"),
                "model_vendor_url": item.get("model_vendor_url"),
                "model_extraction_method": item.get("model_extraction_method"),
                "model_download_filename": item.get("model_download_filename"),
                "model_format": item.get("model_format"),
                "model_page_url": item.get("model_page_url"),
                "price_value": item.get("price_value"),
                "price_currency": item.get("price_currency"),
                "style": item.get("style"),
                "style_llm": item.get("style_llm"),
                "style_llm_confidence": item.get("style_llm_confidence"),
                "style_llm_secondary": item.get("style_llm_secondary"),
                "style_llm_quality_score": item.get("style_llm_quality_score"),
                "style_llm_quality_flags": item.get("style_llm_quality_flags"),
                "style_llm_evidence": item.get("style_llm_evidence"),
                "style_llm_rationale": item.get("style_llm_rationale"),
                "color": item.get("color"),
                "description": item.get("description"),
                "vlm_description_text": item.get("vlm_description_text"),
                "vlm_description_summary": item.get("vlm_description_summary"),
                "vlm_color": item.get("vlm_color"),
                "vlm_materials": item.get("vlm_materials"),
                "vlm_style": item.get("vlm_style"),
                "vlm_visual_features": item.get("vlm_visual_features"),
                "width_cm": item.get("width_cm", dims.get("width")),
                "depth_cm": item.get("depth_cm", dims.get("depth")),
                "height_cm": item.get("height_cm", dims.get("height")),
                "materials": item.get("materials"),
                "room": item.get("room"),
                "availability": item.get("availability"),
                "images_json": json.dumps(item.get("images") or [], ensure_ascii=False),
                "extra_json": json.dumps(item.get("extra") or {}, ensure_ascii=False),
                "asset_status": item.get("asset_status"),
                "asset_format": item.get("asset_format"),
                "asset_local_path": item.get("asset_local_path"),
                "mesh_source_url": item.get("mesh_source_url"),
                "mesh_available": item.get("mesh_available"),
                "mesh_ready": item.get("mesh_ready"),
            }
            if sites and str(row.get("source_site") or "") not in sites:
                continue
            if not row.get("title"):
                continue
            row["semantic_group"] = str(item.get("semantic_group") or "").strip() or _infer_row_group(row)
            if rich_only and not _row_is_rich(row):
                continue
            rows_out.append(row)

    return rows_out


def _merge_catalog_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for row in rows:
        unique_key = str(row.get("unique_key") or "").strip()
        if not unique_key:
            continue
        if unique_key not in merged:
            merged[unique_key] = dict(row)
            continue
        current = merged[unique_key]
        for key, value in row.items():
            if value is None or value == "":
                continue
            if key in {"asset_status", "asset_format", "asset_local_path"}:
                current[key] = value
                continue
            if current.get(key) in (None, ""):
                current[key] = value
    return list(merged.values())


def _load_matcher_context(
    targets_json_path: Path,
    data: dict[str, Any],
    user_preferences: dict[str, Any] | None = None,
    selection_strategy: str = "balanced",
) -> dict[str, Any]:
    run_dir = targets_json_path.parent
    prompt_text = None
    for name in ("chooser_prompt.txt", "prompt.styled.txt", "prompt.txt"):
        p = run_dir / name
        if p.is_file():
            text = p.read_text(encoding="utf-8").strip()
            if text:
                prompt_text = text
                break

    room = data.get("room") or {}
    room_style_hint = str(room.get("style_hint") or "").strip()
    style_profile = data.get("style_profile") if isinstance(data.get("style_profile"), dict) else {}
    if not style_profile:
        profile_path = run_dir / "style_profile.json"
        if profile_path.is_file():
            try:
                loaded_profile = read_json(profile_path)
                if isinstance(loaded_profile, dict):
                    style_profile = loaded_profile
            except Exception:
                style_profile = {}
    if not room_style_hint:
        room_style_hint = str(style_profile.get("style_hint") or style_profile.get("description") or "").strip()
    strategy = str(selection_strategy or "balanced").strip().lower()
    if strategy not in SUPPLIER_SELECTION_STRATEGIES:
        strategy = "balanced"
    return {
        "prompt_text": prompt_text,
        "prompt_tokens": _normalize_text_tokens(prompt_text),
        "room_style_hint": room_style_hint,
        "room_style_tokens": _normalize_text_tokens(room_style_hint),
        "style_label": _normalize_style_label(style_profile.get("style_label") or room.get("style_label")),
        "supplier_selection_strategy": strategy,
        "user_preferences": _normalize_user_preferences(user_preferences),
    }


def _target_is_large_furniture_candidate(target: dict[str, Any]) -> bool:
    category = str(target.get("category") or "").strip()
    semantic_group = str(target.get("semantic_group") or "").strip()
    size_m = target.get("size_m") or [0.0, 0.0, 0.0]
    try:
        sx, sy, sz = [max(float(v), 0.0) for v in size_m]
    except Exception:
        sx, sy, sz = 0.0, 0.0, 0.0
    largest_axis = max(sx, sy, sz)
    volume = sx * sy * sz

    excluded_categories = {
        "BookColumnFactory",
        "BookStackFactory",
        "DeskLampFactory",
        "NatureShelfTrinketsFactory",
        "PillowFactory",
        "BlanketFactory",
        "BoxComforterFactory",
        "PlantContainerFactory",
        "TowelFactory",
        "RugFactory",
    }
    if category in excluded_categories and semantic_group != "lamp_table":
        return False

    allowed_groups = {
        "bed",
        "dresser",
        "desk",
        "shelf",
        "wardrobe",
        "sofa",
        "armchair",
        "chair",
        "coffee_table",
        "side_table",
        "tv_stand",
        "computer",
        "lamp_table",
        "lamp_floor",
        "lamp_ceiling",
        "mirror",
    }
    if semantic_group not in allowed_groups:
        return False

    if semantic_group in {"lamp_table", "lamp_floor", "lamp_ceiling", "mirror"}:
        return True

    return largest_axis >= 0.55 or volume >= 0.06


def _enrich_targets_from_source_scene(data: dict[str, Any], targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source_json = str(data.get("source_json") or "").strip()
    if not source_json:
        return targets
    src_path = Path(source_json).expanduser()
    if not src_path.is_file():
        return targets

    try:
        scene = read_json(src_path)
    except Exception:
        return targets

    placements = scene.get("placements") or scene.get("items") or []
    if not isinstance(placements, list):
        return targets

    by_id = {
        str(item.get("id") or "").strip(): item
        for item in placements
        if isinstance(item, dict) and str(item.get("id") or "").strip()
    }
    out: list[dict[str, Any]] = []
    for target in targets:
        if not isinstance(target, dict):
            out.append(target)
            continue
        item = by_id.get(str(target.get("target_id") or "").strip())
        if not isinstance(item, dict):
            out.append(target)
            continue
        merged = dict(target)
        if merged.get("color_rgb") is None and isinstance(item.get("color"), list):
            merged["color_rgb"] = item.get("color")
        constraints = dict(merged.get("constraints") or {})
        src_constraints = item.get("constraints") or {}
        for key in ("style", "theme", "material", "materials", "color", "brand", "collection"):
            if constraints.get(key) in (None, "", []):
                value = src_constraints.get(key)
                if value not in (None, "", []):
                    constraints[key] = value
        merged["constraints"] = constraints
        out.append(merged)
    return out


def build_bindings_with_candidates(
    *,
    targets_json_path: Path,
    catalog_rows: list[dict[str, Any]],
    top_k: int,
    selection_strategy: str = "balanced",
    user_preferences: dict[str, Any] | None = None,
    llm_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data = read_json(targets_json_path)
    targets = data.get("targets") or []
    if not isinstance(targets, list):
        raise RuntimeError(f"Некорректный layout_targets JSON: {targets_json_path}")
    targets = _enrich_targets_from_source_scene(data, targets)
    context = _load_matcher_context(
        targets_json_path,
        data,
        user_preferences=user_preferences,
        selection_strategy=selection_strategy,
    )

    bindings: list[dict[str, Any]] = []
    matched_count = 0

    for target in targets:
        if not isinstance(target, dict):
            continue

        replacement_policy = str(target.get("replacement_policy") or "keep_generated")
        binding = {
            "target_id": str(target.get("target_id") or ""),
            "category": target.get("category"),
            "semantic_group": target.get("semantic_group"),
            "requested_size_m": target.get("size_m") or [0.0, 0.0, 0.0],
            "replacement_policy": replacement_policy,
            "replacement_reason": target.get("replacement_reason"),
            "provenance": {
                "layout_source": "infinigen_layout",
                "final_asset_source": "generated" if replacement_policy != "replace_with_supplier" else "supplier_catalog_pending",
                "allowed_asset_sources": ["generated_native", "supplier_catalog"],
            },
            "selection_status": "kept_generated_stub",
            "candidate_count": 0,
            "top_candidates": [],
            "chosen_candidate": None,
            "selection_notes": [],
            "supplier_selection_strategy": context.get("supplier_selection_strategy"),
            "user_preferences": _target_user_preferences(target, context),
            "llm_rerank": None,
        }

        if replacement_policy != "replace_with_supplier":
            binding["selection_notes"].append("kept_generated_by_policy")
            bindings.append(binding)
            continue
        if not _target_is_large_furniture_candidate(target):
            binding["selection_notes"].append("kept_generated_small_or_nonfurniture_target")
            binding["provenance"]["final_asset_source"] = "generated"
            bindings.append(binding)
            continue

        scored: list[tuple[tuple[Any, ...], dict[str, Any], dict[str, Any]]] = []
        for row in catalog_rows:
            ranked = _rank_candidate(target, row, context)
            if ranked is None:
                continue
            rank_key, reasons = ranked
            scored.append((rank_key, row, reasons))

        scored.sort(key=lambda x: x[0])
        top = scored[: max(0, int(top_k))]

        top_candidates = []
        for rank_key, row, reasons in top:
            top_candidates.append(
                {
                    "rank_key": [x for x in rank_key[:-1]],
                    "score": round(float(reasons.get("design_score") or 0.0), 6),
                    "unique_key": row.get("unique_key"),
                    "source_site": row.get("source_site"),
                    "source_url": row.get("source_url"),
                    "parsed_at": row.get("parsed_at"),
                    "external_id": row.get("external_id"),
                    "title": row.get("title"),
                    "brand": row.get("brand"),
                    "collection": row.get("collection"),
                    "category_raw": row.get("category_raw"),
                    "category_norm": row.get("category_norm"),
                    "semantic_group": row.get("semantic_group"),
                    "product_url": row.get("product_url"),
                    "model_link_type": row.get("model_link_type"),
                    "model_page_url": row.get("model_page_url"),
                    "model_download_url": row.get("model_download_url"),
                    "model_download_landing_url": row.get("model_download_landing_url"),
                    "model_vendor_url": row.get("model_vendor_url"),
                    "model_extraction_method": row.get("model_extraction_method"),
                    "model_download_filename": row.get("model_download_filename"),
                    "model_format": row.get("model_format"),
                    "asset_status": row.get("asset_status"),
                    "asset_format": row.get("asset_format"),
                    "asset_local_path": row.get("asset_local_path"),
                    "price_value": row.get("price_value"),
                    "price_currency": row.get("price_currency"),
                    "style": row.get("style"),
                    "style_llm": row.get("style_llm"),
                    "style_llm_confidence": row.get("style_llm_confidence"),
                    "style_llm_secondary": row.get("style_llm_secondary"),
                    "style_llm_quality_score": row.get("style_llm_quality_score"),
                    "style_llm_quality_flags": row.get("style_llm_quality_flags"),
                    "style_llm_evidence": row.get("style_llm_evidence"),
                    "style_llm_rationale": row.get("style_llm_rationale"),
                    "color": row.get("color"),
                    "materials": row.get("materials"),
                    "room": row.get("room"),
                    "availability": row.get("availability"),
                    "images_json": row.get("images_json"),
                    "extra_json": row.get("extra_json"),
                    "width_cm": row.get("width_cm"),
                    "depth_cm": row.get("depth_cm"),
                    "height_cm": row.get("height_cm"),
                    "description": row.get("description"),
                    "score_breakdown": reasons,
                    "rich_card": _row_is_rich(row),
                }
            )

        binding["candidate_count"] = len(top_candidates)
        llm_rerank_info = None
        if top_candidates:
            top_candidates, llm_rerank_info = _llm_rerank_candidates(
                target=target,
                top_candidates=top_candidates,
                context=context,
                llm_settings=llm_settings,
            )
            if llm_rerank_info is not None:
                binding["llm_rerank"] = llm_rerank_info
        binding["top_candidates"] = top_candidates

        acceptable_count = 0
        rejection_notes: list[str] = []
        for idx, candidate in enumerate(top_candidates):
            is_acceptable, accept_info = _candidate_acceptability(target, candidate)
            candidate["acceptability"] = accept_info
            if is_acceptable:
                acceptable_count += 1
            elif len(rejection_notes) < 3:
                reason = str(accept_info.get("reject_reason") or "rejected")
                title = str(candidate.get("title") or candidate.get("unique_key") or "candidate")
                rejection_notes.append(f"{title}:{reason}")

        if top_candidates and acceptable_count > 0:
            matched_count += 1
            binding["chosen_candidate"] = top_candidates[0]
            binding["selection_notes"].append("selected_candidate_index:1")
            if isinstance(llm_rerank_info, dict) and llm_rerank_info.get("status") == "applied":
                binding["selection_status"] = "llm_reranked_top1_selected"
                binding["selection_notes"].append("selected_after_llm_rerank")
            else:
                binding["selection_status"] = "heuristic_top1_selected"
            binding["provenance"]["final_asset_source"] = "supplier_catalog"
            binding["selection_notes"].append("top1_selected_by_similarity_order")
            if rejection_notes:
                binding["selection_notes"].append("rejected_before_selection:" + " | ".join(rejection_notes))
        else:
            if top_candidates:
                binding["selection_status"] = "no_acceptable_candidates_found"
                binding["selection_notes"].append("all_supplier_candidates_rejected_by_acceptance_gate")
                if rejection_notes:
                    binding["selection_notes"].append("candidate_rejections:" + " | ".join(rejection_notes))
            else:
                binding["selection_status"] = "no_candidates_found"
                binding["selection_notes"].append("no_supplier_candidate_inside_bbox_with_similar_size")

        bindings.append(binding)

    return {
        "schema": "supplier_bindings/v1",
        "layout_targets_json": str(targets_json_path.resolve()),
        "meta": {
            "status": "heuristic_candidates_built",
            "target_count": len(bindings),
            "matched_target_count": matched_count,
            "top_k": int(top_k),
            "supplier_selection_strategy": context.get("supplier_selection_strategy"),
            "style_llm_policy": {
                "min_confidence": STYLE_LLM_MIN_CONFIDENCE,
                "min_quality_score": STYLE_LLM_MIN_QUALITY,
                "mismatch_is_penalty_not_filter": True,
            },
            "ranking_order": ["category_family", "bbox_fit_required", "width", "height", "depth", "strategy_price_style", "query_match", "user_preferences", "prompt_color", "style_llm", "design", "real_asset"],
            "category_policy": "exact_group_or_same_family",
            "bbox_fit_policy": "strict_bbox_or_rescalable_anchor_fit_with_group_limits",
            "final_selection_policy": "llm_or_heuristic_top1_sets_candidate_order_asset_acquisition_uses_first_acceptable_with_real_mesh_else_keep_generated",
            "prompt_text": context.get("prompt_text"),
            "room_style_hint": context.get("room_style_hint"),
            "user_preferences": context.get("user_preferences"),
            "llm_settings": dict(llm_settings or {}),
            "bbox_min_fill_policy": {
                "bed": {"width_min_ratio": 0.75, "depth_min_ratio": 0.75, "height_min_ratio": 0.55},
                "wardrobe": {"width_min_ratio": 0.7, "depth_min_ratio": 0.65, "height_min_ratio": 0.8},
            },
            "dimension_priority_policy": {
                "desk": "height_first",
                "coffee_table": "height_first",
                "side_table": "height_first",
                "dresser": "height_first",
                "shelf": "height_first",
                "bed": "footprint_first",
                "default": "overall_size_first",
            },
        },
        "bindings": bindings,
    }


def build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Heuristic supplier matcher for layout_targets.json")
    ap.add_argument("--targets", required=True, help="Path to layout_targets.json")
    ap.add_argument("--supplier-db", action="append", default=[], help="SQLite DB with supplier_product and optional supplier_asset")
    ap.add_argument("--supplier-json", action="append", default=[], help="Exported supplier catalog JSON; can be repeated")
    ap.add_argument("--site", action="append", default=None, help="Optional source_site filter; can be repeated")
    ap.add_argument("--rich-only", action="store_true", help="Use only rich cards with title, price, dimensions, description, category and brand")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument(
        "--selection-strategy",
        choices=sorted(SUPPLIER_SELECTION_STRATEGIES),
        default="balanced",
        help="Supplier candidate ranking strategy: cheapest, cheap_style, style, or balanced.",
    )
    ap.add_argument("--user-preferences-json", default=None, help="Optional JSON with global/by_target_id/by_semantic_group manual constraints")
    ap.add_argument("--max-price-rub", type=float, default=None, help="Global max acceptable price in RUB")
    ap.add_argument("--preferred-color", action="append", default=[], help="Preferred color token; may be repeated")
    ap.add_argument("--avoid-color", action="append", default=[], help="Color token to avoid; may be repeated")
    ap.add_argument("--preferred-brand", action="append", default=[], help="Preferred brand; may be repeated")
    ap.add_argument("--allowed-site", action="append", default=[], help="Allowed source site; may be repeated")
    ap.add_argument("--disallowed-site", action="append", default=[], help="Disallowed source site; may be repeated")
    ap.add_argument("--strict-color", action="store_true", help="Reject candidates that do not match preferred colors")
    ap.add_argument("--require-real-asset", action="store_true", help="Reject candidates without a local real mesh asset")
    ap.add_argument("--require-model-url", action="store_true", help="Reject candidates without model page or download URL")
    ap.add_argument("--llm-provider", choices=["none", "ollama"], default="none", help="Optional final LLM reranker after heuristic top-K")
    ap.add_argument("--ollama-url", default="http://127.0.0.1:11434", help="Ollama base URL for LLM reranking")
    ap.add_argument("--ollama-model", default="gpt-oss:20b", help="Ollama model for final candidate reranking")
    ap.add_argument("--ollama-timeout", type=int, default=180, help="Ollama timeout in seconds")
    ap.add_argument("--ollama-temperature", type=float, default=0.0, help="Ollama temperature for reranking")
    ap.add_argument("--llm-top-n", type=int, default=5, help="How many top heuristic candidates to send to the LLM")
    ap.add_argument("--out", required=True, help="Output bindings json")
    return ap


def main() -> None:
    args = build_cli().parse_args()
    targets_path = Path(args.targets).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    db_paths = [Path(x).expanduser().resolve() for x in args.supplier_db]
    json_paths = [Path(x).expanduser().resolve() for x in args.supplier_json]
    sites = {str(x).strip() for x in (args.site or []) if str(x).strip()} or None
    user_preferences = {}
    if args.user_preferences_json:
        user_preferences = read_json(args.user_preferences_json)
        if not isinstance(user_preferences, dict):
            raise RuntimeError("user preferences JSON must be an object")
    cli_preference_scope = _normalize_preference_scope(
        {
            "max_price_rub": args.max_price_rub,
            "preferred_colors": args.preferred_color,
            "avoid_colors": args.avoid_color,
            "preferred_brands": args.preferred_brand,
            "allowed_sites": args.allowed_site,
            "disallowed_sites": args.disallowed_site,
            "strict_color": args.strict_color,
            "require_real_asset": args.require_real_asset,
            "require_model_url": args.require_model_url,
        }
    )
    user_preferences = _normalize_user_preferences(user_preferences)
    user_preferences["global"] = _merge_preference_scopes(user_preferences.get("global"), cli_preference_scope)
    llm_settings = {
        "provider": args.llm_provider,
        "ollama_url": args.ollama_url,
        "ollama_model": args.ollama_model,
        "ollama_timeout": int(args.ollama_timeout),
        "ollama_temperature": float(args.ollama_temperature),
        "top_n": int(args.llm_top_n),
    }

    if not db_paths and not json_paths:
        raise RuntimeError("Нужно передать хотя бы один --supplier-db или --supplier-json")

    auto_db_path = targets_path.parent / "supplier_scene_assets.db"
    if json_paths and auto_db_path.is_file() and auto_db_path not in db_paths:
        db_paths.append(auto_db_path)

    catalog_rows: list[dict[str, Any]] = []
    if db_paths:
        catalog_rows.extend(load_supplier_catalog(db_paths, sites=sites, rich_only=bool(args.rich_only)))
    if json_paths:
        catalog_rows.extend(load_supplier_catalog_json(json_paths, sites=sites, rich_only=bool(args.rich_only)))
    catalog_rows = _merge_catalog_rows(catalog_rows)

    result = build_bindings_with_candidates(
        targets_json_path=targets_path,
        catalog_rows=catalog_rows,
        top_k=int(args.top_k),
        selection_strategy=str(args.selection_strategy),
        user_preferences=user_preferences,
        llm_settings=llm_settings,
    )
    write_json(out_path, result)
    print(f"catalog_rows = {len(catalog_rows)}")
    print(f"matched_target_count = {result['meta']['matched_target_count']}")
    print(f"saved = {out_path}")


if __name__ == "__main__":
    main()
