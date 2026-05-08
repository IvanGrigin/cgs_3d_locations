#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import re
from collections import defaultdict
from statistics import quantiles
from typing import Any

try:
    from .supplier_identity_gates import STRICT_GROUPS
except ImportError:
    from supplier_identity_gates import STRICT_GROUPS

from .supplier_selection_modes import combine_scores_for_mode, normalize_selection_mode


SUPPORTED_LOCAL_ASSET_EXTS = {"obj", "fbx", "glb", "gltf"}
LOW_QUALITY_ASSET_STATUSES = {"proxy_generated_with_blender", "needs_blender_rebuild"}

STYLE_ALIASES = {
    "scandi": "scandinavian",
    "сканди": "scandinavian",
    "скандинавский": "scandinavian",
    "minimalist": "minimalism",
    "minimal": "minimalism",
    "contemporary": "modern",
    "loft": "loft_industrial",
    "industrial": "loft_industrial",
}

COLOR_ALIASES = {
    "warm_white": "white",
    "ivory": "white",
    "cream": "beige",
    "sand": "beige",
    "taupe": "beige",
    "light_oak": "brown",
    "oak": "brown",
    "wood": "brown",
    "walnut": "brown",
    "natural": "beige",
    "charcoal": "gray",
    "graphite": "gray",
    "silver": "gray",
    "grey": "gray",
    "black_metal": "black",
    "terracotta": "red",
    "burgundy": "red",
    "sage": "green",
    "olive": "green",
    "navy": "blue",
}


def _tokens(value: Any) -> set[str]:
    if isinstance(value, dict):
        parts = []
        for item in value.values():
            parts.extend(_tokens(item))
        return set(parts)
    if isinstance(value, (list, tuple, set)):
        out: set[str] = set()
        for item in value:
            out |= _tokens(item)
        return out
    return set(re.findall(r"[A-Za-zА-Яа-я0-9_\-]+", str(value or "").lower().replace("ё", "е")))


def _normalize_style_token(token: str) -> str:
    text = str(token or "").strip().lower().replace("-", "_").replace(" ", "_")
    return STYLE_ALIASES.get(text, text)


def _normalize_color_token(token: str) -> str:
    text = str(token or "").strip().lower().replace("-", "_").replace(" ", "_")
    return COLOR_ALIASES.get(text, text)


def _overlap_score(candidate: set[str], desired: set[str], *, partial: bool = True) -> tuple[float, list[str]]:
    if not desired:
        return 0.55, []
    direct = candidate & desired
    matches = set(direct)
    if partial:
        for c in candidate:
            for d in desired:
                if len(c) >= 4 and len(d) >= 4 and (c in d or d in c):
                    matches.add(c)
    return min(1.0, len(matches) / max(len(desired), 1)), sorted(matches)[:20]


def _safe_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None


def _candidate_size(row: dict[str, Any]) -> list[float] | None:
    vals = [row.get("width_cm"), row.get("depth_cm"), row.get("height_cm")]
    if any(v is None for v in vals):
        return None
    try:
        return [max(float(v) / 100.0, 1e-6) for v in vals]
    except Exception:
        return None


def _target_size(target: dict[str, Any]) -> list[float] | None:
    vals = target.get("size_m")
    if not isinstance(vals, list) or len(vals) < 3:
        return None
    try:
        return [max(float(v), 1e-6) for v in vals[:3]]
    except Exception:
        return None


def _candidate_has_ready_real_asset(row: dict[str, Any]) -> bool:
    local_path = str(row.get("asset_local_path") or "").strip()
    fmt = str(row.get("asset_format") or "").strip().lower().lstrip(".")
    status = str(row.get("asset_status") or "").strip().lower()
    return bool(local_path and fmt in SUPPORTED_LOCAL_ASSET_EXTS and status not in LOW_QUALITY_ASSET_STATUSES)


def _candidate_has_downloadable_asset(row: dict[str, Any]) -> bool:
    if row.get("model_download_url") or row.get("model_page_url") or row.get("model_download_landing_url"):
        fmt = str(row.get("model_format") or row.get("asset_format") or "").strip().lower().lstrip(".")
        return fmt not in {"max"} or bool(row.get("model_download_url"))
    return False


def build_price_stats(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    values_by_group: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        price = _safe_float(row.get("price_value"))
        if price is None or price <= 0:
            continue
        group = str(row.get("semantic_group") or row.get("category_norm") or "unknown").strip() or "unknown"
        values_by_group[group].append(price)
    stats: dict[str, dict[str, float]] = {}
    for group, values in values_by_group.items():
        values = sorted(values)
        if not values:
            continue
        if len(values) >= 10:
            p90 = quantiles(values, n=10, method="inclusive")[8]
        else:
            p90 = values[-1]
        stats[group] = {"min": values[0], "p90": max(p90, values[0] + 1.0), "count": float(len(values))}
    return stats


def compute_category_score(target: dict[str, Any], row: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    target_group = str(target.get("semantic_group") or "").strip()
    row_group = str(row.get("semantic_group") or "").strip()
    if target_group and row_group and target_group == row_group:
        score = 1.0
        match = "exact_group"
    elif target_group and row_group and ({target_group, row_group} <= {"chair", "armchair"}):
        score = 0.78
        match = "same_family"
    else:
        target_tokens = _tokens([target.get("category"), target.get("name"), target_group])
        row_tokens = _tokens([row.get("category_norm"), row.get("category_raw"), row.get("title"), row_group])
        overlap = target_tokens & row_tokens
        score = min(0.55, len(overlap) / max(len(target_tokens), 1))
        match = "token_overlap" if overlap else "weak_or_none"
    if target_group in STRICT_GROUPS and match not in {"exact_group", "same_family"}:
        score = min(score, 0.45)
    return score, {"category_score": round(score, 6), "category_match_v2": match}


def compute_size_score(target: dict[str, Any], row: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    ts = _target_size(target)
    cs = _candidate_size(row)
    if not ts or not cs:
        return 0.55, {"size_score": 0.55, "size_score_reason": "missing_dimensions"}
    direct = sum(abs(math.log(max(t, 1e-6) / max(c, 1e-6))) for t, c in zip(ts, cs))
    swapped = abs(math.log(ts[0] / cs[1])) + abs(math.log(ts[1] / cs[0])) + abs(math.log(ts[2] / cs[2]))
    dist = min(direct, swapped)
    score = math.exp(-0.72 * dist)
    return max(0.0, min(1.0, score)), {"size_score": round(score, 6), "size_log_distance": round(dist, 6)}


def compute_asset_availability_score(row: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    if _candidate_has_ready_real_asset(row):
        return 1.0, {"asset_availability_score": 1.0, "asset_availability": "ready_real_asset"}
    if _candidate_has_downloadable_asset(row):
        return 0.72, {"asset_availability_score": 0.72, "asset_availability": "downloadable_asset"}
    return 0.0, {"asset_availability_score": 0.0, "asset_availability": "missing_asset"}


def _candidate_color_tokens(row: dict[str, Any]) -> set[str]:
    parts = [row.get("color"), row.get("vlm_color"), row.get("vlm_description_text"), row.get("title"), row.get("description")]
    image_features = row.get("image_color_features") if isinstance(row.get("image_color_features"), dict) else {}
    parts.append(image_features.get("color_tokens"))
    colors = image_features.get("colors") if isinstance(image_features.get("colors"), dict) else {}
    for entry in colors.get("top5") or []:
        if isinstance(entry, dict):
            parts.append(entry.get("basic_color"))
    return {_normalize_color_token(x) for x in _tokens(parts)}


def _desired_object_req(target: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    reqs = spec.get("object_requirements") if isinstance(spec.get("object_requirements"), dict) else {}
    group = str(target.get("semantic_group") or target.get("category") or "").strip()
    return reqs.get(group) if isinstance(reqs.get(group), dict) else {}


def compute_color_score(target: dict[str, Any], row: dict[str, Any], spec: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    req = _desired_object_req(target, spec)
    palette = spec.get("color_palette") if isinstance(spec.get("color_palette"), dict) else {}
    desired = {_normalize_color_token(x) for x in _tokens([req.get("colors"), palette.get("primary"), palette.get("secondary"), palette.get("accent")])}
    forbidden = {_normalize_color_token(x) for x in _tokens([req.get("avoid"), palette.get("forbidden")])}
    candidate = _candidate_color_tokens(row)
    positive, matched = _overlap_score(candidate, desired)
    negative = len(candidate & forbidden) / max(len(forbidden), 1) if forbidden else 0.0
    score = max(0.0, min(1.0, positive - 0.5 * negative))
    return score, {"color_score": round(score, 6), "matched_colors": matched, "candidate_colors": sorted(candidate)[:20]}


def compute_material_score(target: dict[str, Any], row: dict[str, Any], spec: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    req = _desired_object_req(target, spec)
    materials = spec.get("materials") if isinstance(spec.get("materials"), dict) else {}
    desired = _tokens([req.get("materials"), materials.get("preferred"), materials.get("allowed")])
    forbidden = _tokens([req.get("avoid"), materials.get("forbidden")])
    candidate = _tokens([row.get("materials"), row.get("description"), row.get("vlm_materials"), row.get("vlm_description_text"), row.get("title")])
    positive, matched = _overlap_score(candidate, desired)
    negative = len(candidate & forbidden) / max(len(forbidden), 1) if forbidden else 0.0
    score = max(0.0, min(1.0, positive - 0.55 * negative))
    return score, {"material_score": round(score, 6), "matched_materials": matched}


def compute_style_score(target: dict[str, Any], row: dict[str, Any], spec: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    req = _desired_object_req(target, spec)
    style = spec.get("style") if isinstance(spec.get("style"), dict) else {}
    desired = {_normalize_style_token(x) for x in _tokens([style.get("primary"), style.get("secondary"), style.get("allowed"), req.get("style")])}
    forbidden = {_normalize_style_token(x) for x in _tokens([style.get("forbidden"), req.get("avoid")])}
    candidate = {_normalize_style_token(x) for x in _tokens([row.get("style"), row.get("style_llm"), row.get("style_llm_secondary"), row.get("description"), row.get("vlm_description_text"), row.get("title")])}
    positive, matched = _overlap_score(candidate, desired)
    negative = len(candidate & forbidden) / max(len(forbidden), 1) if forbidden else 0.0
    score = max(0.0, min(1.0, positive - 0.7 * negative))
    return score, {"style_score": round(score, 6), "matched_styles": matched, "candidate_style_tokens": sorted(candidate)[:20]}


def compute_epoch_score(target: dict[str, Any], row: dict[str, Any], spec: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    epoch = spec.get("epoch") if isinstance(spec.get("epoch"), dict) else {}
    desired = _tokens([epoch.get("primary"), epoch.get("allowed")])
    forbidden = _tokens(epoch.get("forbidden"))
    candidate = _tokens([row.get("style"), row.get("description"), row.get("vlm_description_text"), row.get("title")])
    positive, matched = _overlap_score(candidate, desired)
    negative = len(candidate & forbidden) / max(len(forbidden), 1) if forbidden else 0.0
    score = max(0.0, min(1.0, positive - 0.7 * negative))
    return score, {"epoch_score": round(score, 6), "matched_epoch": matched}


def compute_description_score(target: dict[str, Any], row: dict[str, Any], spec: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    req = _desired_object_req(target, spec)
    desired = _tokens([spec.get("expanded_room_description"), req])
    candidate = _tokens([row.get("title"), row.get("description"), row.get("vlm_description_text"), row.get("vlm_description_summary")])
    positive, matched = _overlap_score(candidate, desired)
    return positive, {"description_score": round(positive, 6), "description_matches": matched}


def compute_price_score(row: dict[str, Any], price_stats: dict[str, dict[str, float]], mode: str | None) -> tuple[float, dict[str, Any]]:
    price = _safe_float(row.get("price_value"))
    normalized_mode = normalize_selection_mode(mode)
    if price is None or price <= 0:
        score = 0.2 if normalized_mode == "cheapest" else 0.5
        return score, {"price_score": score, "price_known": False}
    group = str(row.get("semantic_group") or row.get("category_norm") or "unknown").strip() or "unknown"
    stats = price_stats.get(group) or {}
    min_price = float(stats.get("min") or price)
    p90 = float(stats.get("p90") or max(price, min_price + 1.0))
    norm = (price - min_price) / max(p90 - min_price, 1.0)
    score = 1.0 - max(0.0, min(1.0, norm))
    return score, {"price_score": round(score, 6), "price_known": True, "price_group_min": min_price, "price_group_p90": p90}


def compute_design_similarity_score(scores: dict[str, float]) -> tuple[float, dict[str, Any]]:
    score = (
        float(scores.get("style_score") or 0.0) * 0.35
        + float(scores.get("color_score") or 0.0) * 0.25
        + float(scores.get("material_score") or 0.0) * 0.20
        + float(scores.get("description_score") or 0.0) * 0.20
    )
    return score, {"design_similarity_score": round(score, 6)}


def rank_candidate_for_mode(
    *,
    target: dict[str, Any],
    row: dict[str, Any],
    room_design_spec: dict[str, Any],
    mode: str,
    price_stats: dict[str, dict[str, float]],
) -> tuple[float, dict[str, Any]]:
    scores: dict[str, float] = {}
    breakdown: dict[str, Any] = {}
    for value, info in [
        compute_category_score(target, row),
        compute_size_score(target, row),
        compute_asset_availability_score(row),
        compute_color_score(target, row, room_design_spec),
        compute_material_score(target, row, room_design_spec),
        compute_style_score(target, row, room_design_spec),
        compute_epoch_score(target, row, room_design_spec),
        compute_description_score(target, row, room_design_spec),
        compute_price_score(row, price_stats, mode),
    ]:
        breakdown.update(info)
        for key, item in info.items():
            if key.endswith("_score"):
                scores[key] = float(item)
    design_score, design_info = compute_design_similarity_score(scores)
    scores["design_similarity_score"] = design_score
    breakdown.update(design_info)
    final_score, mode_info = combine_scores_for_mode(scores, mode)
    breakdown.update(mode_info)
    breakdown["final_score"] = round(final_score, 6)
    breakdown["score_schema"] = "supplier_design_scores/v1"
    return final_score, breakdown
