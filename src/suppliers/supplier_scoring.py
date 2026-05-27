#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from statistics import quantiles
from typing import Any

try:
    from .supplier_identity_gates import STRICT_GROUPS
except ImportError:  # pragma: no cover
    from supplier_identity_gates import STRICT_GROUPS  # pragma: no cover

from .supplier_selection_modes import combine_scores_for_mode, normalize_selection_mode


SUPPORTED_LOCAL_ASSET_EXTS = {"obj", "fbx", "glb", "gltf"}
LOW_QUALITY_ASSET_STATUSES = {"proxy_generated_with_blender", "needs_blender_rebuild"}
TRUSTED_PRODUCT_IMAGE_PATTERNS = (
    "ikea.",
    "ikea.com",
    "ikea com",
    "ikea_de",
    "moyamebel",
    "moya-mebel",
    "moya_mebel",
    "moya mebel",
    "моя мебель",
    "mebel.ru",
    "mebel ru",
    "mebelru",
    "mebel_ru",
)
SCENE_RENDER_IMAGE_PATTERNS = ("3ddd", "zeelproject")


@dataclass
class CandidateScore:
    total: float
    acceptable: bool
    hard_reject_reason: str | None
    breakdown: dict[str, Any]
    notes: list[str]


STYLE_ALIASES = {
    "scandi": "scandinavian",
    "сканди": "scandinavian",
    "скандинавский": "scandinavian",
    "minimalist": "minimalism",
    "minimal": "minimalism",
    "contemporary": "contemporary",
    "loft": "loft",
    "industrial": "loft",
    "loft_industrial": "loft",
    "eco": "eco_organic",
    "organic": "eco_organic",
    "mid-century": "mid_century_modern",
    "mid_century": "mid_century_modern",
    "mcm": "mid_century_modern",
    "classic": "classic",
    "classical": "classic",
    "traditional": "classic",
    "soft_classic": "soft_classic",
    "soft_traditional": "soft_classic",
    "residential_classic": "soft_classic",
    "baroque": "baroque",
    "japandi": "japandi",
}

STYLE_COMPATIBILITY: dict[str, dict[str, float]] = {
    "modern": {
        "modern": 1.0,
        "contemporary": 0.9,
        "minimalism": 0.85,
        "scandinavian": 0.65,
        "loft": 0.55,
        "classic": 0.25,
        "baroque": 0.10,
    },
    "contemporary": {
        "contemporary": 1.0,
        "modern": 0.9,
        "minimalism": 0.8,
        "scandinavian": 0.6,
        "classic": 0.35,
    },
    "minimalism": {
        "minimalism": 1.0,
        "modern": 0.85,
        "contemporary": 0.8,
        "scandinavian": 0.75,
        "japandi": 0.75,
        "classic": 0.2,
    },
    "scandinavian": {
        "scandinavian": 1.0,
        "japandi": 0.9,
        "eco_organic": 0.85,
        "minimalism": 0.75,
        "modern": 0.65,
        "baroque": 0.10,
    },
    "japandi": {
        "japandi": 1.0,
        "scandinavian": 0.9,
        "eco_organic": 0.85,
        "minimalism": 0.75,
        "modern": 0.60,
        "classic": 0.20,
    },
    "loft": {
        "loft": 1.0,
        "industrial": 1.0,
        "modern": 0.55,
        "contemporary": 0.5,
        "classic": 0.15,
        "baroque": 0.10,
    },
    "eco_organic": {
        "eco_organic": 1.0,
        "japandi": 0.85,
        "scandinavian": 0.85,
        "minimalism": 0.7,
        "modern": 0.55,
    },
    "soft_classic": {
        "soft_classic": 1.0,
        "classic": 0.9,
        "contemporary": 0.78,
        "modern": 0.58,
        "scandinavian": 0.55,
        "japandi": 0.45,
        "minimalism": 0.35,
        "baroque": 0.22,
        "loft": 0.10,
    },
    "classic": {
        "classic": 1.0,
        "soft_classic": 0.9,
        "contemporary": 0.35,
        "modern": 0.25,
        "baroque": 0.45,
        "minimalism": 0.2,
    },
    "baroque": {
        "baroque": 1.0,
        "classic": 0.45,
        "modern": 0.10,
        "minimalism": 0.05,
        "scandinavian": 0.10,
    },
}

COLOR_ALIASES = {
    "warm_white": "white_warm",
    "ivory": "white_warm",
    "milk": "white_warm",
    "молочный": "white_warm",
    "cream": "beige",
    "кремовый": "beige",
    "sand": "beige",
    "taupe": "beige",
    "light_oak": "wood_light",
    "oak": "wood_light",
    "дуб": "wood_light",
    "sonoma": "wood_light",
    "сонома": "wood_light",
    "wood": "wood",
    "natural": "wood_light",
    "натуральный": "wood_light",
    "walnut": "dark_brown",
    "венге": "dark_brown",
    "wenge": "dark_brown",
    "charcoal": "dark_gray",
    "graphite": "dark_gray",
    "anthracite": "dark_gray",
    "антрацит": "dark_gray",
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


_DIMENSION_UNIT_RE = r"(?:мм|mm|см|cm|метр(?:а|ов)?|м|m)?"
_DIMENSION_LABELS: dict[str, tuple[str, ...]] = {
    "width": ("ширина", "шир.", "width", "w"),
    "depth": ("глубина", "глуб.", "depth", "d"),
    "length": ("длина", "дл.", "length", "len", "l"),
    "height": ("высота", "выс.", "height", "h"),
}


def _dimension_value_to_cm(value: Any, unit: str | None = None) -> float | None:
    number = _safe_float(str(value).replace(",", ".") if value is not None else None)
    if number is None or number <= 0:
        return None  # pragma: no cover
    unit_norm = str(unit or "").strip().lower().replace(".", "")
    if unit_norm in {"мм", "mm"}:
        return number / 10.0
    if unit_norm in {"м", "m", "метр", "метра", "метров"}:
        return number * 100.0
    if not unit_norm and number <= 6.0:
        return number * 100.0
    return number


def _infer_dimensions_cm_from_text(row: dict[str, Any]) -> dict[str, float]:
    parts = [
        row.get("title"),
        row.get("description"),
        row.get("category_raw"),
        row.get("extra_json"),
        row.get("extra"),
    ]
    text = " ".join(str(part or "") for part in parts).lower().replace("ё", "е")
    found: dict[str, float] = {}
    for axis, labels in _DIMENSION_LABELS.items():
        label_re = "|".join(re.escape(label) for label in labels)
        pattern = re.compile(
            rf"(?:\b|^)(?:{label_re})(?:\b|\.?)\s*(?:[:=-]|\s)\s*"
            rf"(\d+(?:[.,]\d+)?)\s*({_DIMENSION_UNIT_RE})",
            re.IGNORECASE,
        )
        match = pattern.search(text)
        if not match:
            continue
        value = _dimension_value_to_cm(match.group(1), match.group(2))
        if value is not None:
            found[axis] = value
    if "depth" not in found and "length" in found:
        found["depth"] = found["length"]
    if not {"width", "depth", "height"} <= set(found):
        triple = re.search(
            rf"(\d+(?:[.,]\d+)?)\s*(?:x|х|×|\*)\s*"
            rf"(\d+(?:[.,]\d+)?)\s*(?:x|х|×|\*)\s*"
            rf"(\d+(?:[.,]\d+)?)\s*({_DIMENSION_UNIT_RE})",
            text,
            flags=re.IGNORECASE,
        )
        if triple:
            unit = triple.group(4)
            values = [_dimension_value_to_cm(triple.group(i), unit) for i in (1, 2, 3)]
            if all(value is not None for value in values):
                found.setdefault("width", float(values[0]))
                found.setdefault("depth", float(values[1]))
                found.setdefault("height", float(values[2]))
    return {key: round(float(value), 4) for key, value in found.items() if value and value > 0}


def _row_dimension_cm(row: dict[str, Any], axis: str) -> float | None:
    dims = row.get("dimensions_cm") if isinstance(row.get("dimensions_cm"), dict) else {}
    value = row.get(f"{axis}_cm", dims.get(axis))
    direct = _dimension_value_to_cm(value, "cm") if value is not None else None
    if direct is not None:
        return direct
    return _infer_dimensions_cm_from_text(row).get(axis)


def _candidate_size(row: dict[str, Any]) -> list[float] | None:
    vals = [_row_dimension_cm(row, "width"), _row_dimension_cm(row, "depth"), _row_dimension_cm(row, "height")]
    if any(v is None for v in vals):
        return None
    try:
        return [max(float(v) / 100.0, 1e-6) for v in vals]
    except Exception:
        return None


def _dimension_weights(group: str) -> dict[str, float]:
    if group in {"bed", "sofa", "dining_table"}:
        return {"width": 0.40, "depth": 0.40, "height": 0.20}
    if group in {"wardrobe", "shelf", "fridge", "refrigerator"}:
        return {"width": 0.25, "depth": 0.35, "height": 0.40}
    if group in {"desk", "coffee_table", "side_table", "nightstand", "dresser", "tv_stand", "stool", "bench"}:
        return {"width": 0.30, "depth": 0.30, "height": 0.40}
    return {"width": 0.34, "depth": 0.33, "height": 0.33}


def _oriented_size_candidates(ts: list[float], cs: list[float]) -> list[dict[str, Any]]:
    out = []
    for orientation, oriented in (("direct", cs), ("swapped_xy", [cs[1], cs[0], cs[2]])):
        ratios = [max(t, 1e-6) / max(c, 1e-6) for t, c in zip(ts, oriented)]
        out.append({"orientation": orientation, "size": oriented, "scale_ratios": ratios})
    return out


def _scale_policy_info(scale_ratios: list[float]) -> dict[str, Any]:
    abs_logs = [abs(math.log(max(x, 1e-6))) for x in scale_ratios]
    deformation = max(abs_logs) - min(abs_logs)
    outside_moderate = [x for x in scale_ratios if x < 0.60 or x > 1.60]
    outside_preferred = [x for x in scale_ratios if x < 0.75 or x > 1.35]
    if outside_moderate:
        return {
            "scale_policy": "reject",
            "scale_policy_penalty": 1.0,
            "scale_reject_reason": "unreasonable_scale",
            "scale_deformation": deformation,
        }
    penalty = deformation * 0.22
    if outside_preferred:
        penalty += 0.22
    return {
        "scale_policy": "preferred" if not outside_preferred else "moderate_with_penalty",
        "scale_policy_penalty": min(0.75, penalty),
        "scale_reject_reason": None,
        "scale_deformation": deformation,
    }


def _target_size(target: dict[str, Any]) -> list[float] | None:
    vals = target.get("size_m")
    if not isinstance(vals, list) or len(vals) < 3:
        return None
    try:
        out = [max(float(v), 1e-6) for v in vals[:3]]
    except Exception:
        return None
    if str(target.get("semantic_group") or target.get("category") or "").strip().lower() == "bed":
        out[2] = max(out[2], 0.90)
    return out


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
            continue  # pragma: no cover
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
    elif target_group and row_group and ({target_group, row_group} <= {"chair", "armchair", "stool", "bench"}):
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
    group = str(target.get("semantic_group") or target.get("category") or "").strip()
    weights = _dimension_weights(group)

    orientation_candidates = _oriented_size_candidates(ts, cs)
    if group == "bed":
        orientation_candidates = orientation_candidates[:1]

    candidates = []
    for candidate in orientation_candidates:
        ratios = candidate["scale_ratios"]
        policy = _scale_policy_info(ratios)
        weighted_dist = (
            abs(math.log(ratios[0])) * weights["width"]
            + abs(math.log(ratios[1])) * weights["depth"]
            + abs(math.log(ratios[2])) * weights["height"]
        )
        score = math.exp(-1.05 * weighted_dist) * (1.0 - float(policy["scale_policy_penalty"]))
        if policy["scale_reject_reason"]:
            score *= 0.15
        candidates.append({**candidate, **policy, "weighted_size_distance": weighted_dist, "score": max(0.0, min(1.0, score))})

    best = max(candidates, key=lambda item: item["score"])
    return best["score"], {
        "size_score": round(best["score"], 6),
        "size_log_distance": round(best["weighted_size_distance"], 6),
        "size_orientation": best["orientation"],
        "dimension_weights": weights,
        "scale_ratios": [round(float(x), 6) for x in best["scale_ratios"]],
        "scale_policy": best["scale_policy"],
        "scale_policy_penalty": round(float(best["scale_policy_penalty"]), 6),
        "scale_deformation": round(float(best["scale_deformation"]), 6),
        "scale_reject_reason": best["scale_reject_reason"],
    }


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


def _candidate_image_color_tokens(row: dict[str, Any]) -> set[str]:
    image_features = row.get("image_color_features") if isinstance(row.get("image_color_features"), dict) else {}
    parts: list[Any] = [image_features.get("color_tokens")]
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
    desired = {_normalize_color_token(x) for x in _tokens([
        req.get("colors"),
        req.get("preferred_colors"),
        palette.get("primary"),
        palette.get("secondary"),
        palette.get("accent"),
        palette.get("preferred_colors"),
    ])}
    forbidden = {_normalize_color_token(x) for x in _tokens([
        req.get("avoid"),
        req.get("forbidden_colors"),
        palette.get("forbidden"),
        palette.get("forbidden_colors"),
    ])}
    candidate = _candidate_color_tokens(row)
    positive, matched = _overlap_score(candidate, desired)
    image_candidate = _candidate_image_color_tokens(row)
    forbidden_image_hits = sorted(image_candidate & forbidden)
    forbidden_hits = sorted(candidate & forbidden)
    score = positive
    if forbidden_hits:
        score *= 0.45
    if forbidden_image_hits:
        score *= 0.15
    return max(0.0, min(1.0, score)), {
        "color_score": round(max(0.0, min(1.0, score)), 6),
        "matched_colors": matched,
        "candidate_colors": sorted(candidate)[:20],
        "forbidden_color_hits": forbidden_hits[:20],
        "forbidden_image_color_hits": forbidden_image_hits[:20],
    }


def compute_image_color_score(target: dict[str, Any], row: dict[str, Any], spec: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    req = _desired_object_req(target, spec)
    palette = spec.get("color_palette") if isinstance(spec.get("color_palette"), dict) else {}
    desired = {_normalize_color_token(x) for x in _tokens([
        req.get("colors"),
        req.get("preferred_colors"),
        palette.get("primary"),
        palette.get("secondary"),
        palette.get("accent"),
        palette.get("preferred_colors"),
    ])}
    forbidden = {_normalize_color_token(x) for x in _tokens([req.get("forbidden_colors"), palette.get("forbidden"), palette.get("forbidden_colors")])}
    candidate = _candidate_image_color_tokens(row)
    if not candidate:
        return 0.50, {"image_color_score": 0.50, "image_color_available": False, "image_color_tokens": []}
    positive, matched = _overlap_score(candidate, desired)
    if candidate & forbidden:
        positive *= 0.15
    return max(0.0, min(1.0, positive)), {
        "image_color_score": round(max(0.0, min(1.0, positive)), 6),
        "image_color_available": True,
        "matched_image_colors": matched,
        "image_color_tokens": sorted(candidate)[:20],
    }


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
    candidate = {_normalize_style_token(x) for x in _tokens([row.get("style"), row.get("style_llm"), row.get("style_llm_secondary"), row.get("description"), row.get("vlm_description_text"), row.get("vlm_style"), row.get("title"), row.get("tags")])}
    if not desired:
        positive, matched = 0.55, []
    else:
        best = 0.0
        matched = []
        for d in desired:
            compat = STYLE_COMPATIBILITY.get(d, {})
            for c in candidate:
                value = compat.get(c, 1.0 if c == d else 0.0)
                if value > best:
                    best = value
                    matched = [c]
        overlap, overlap_matches = _overlap_score(candidate, desired)
        positive = max(best, overlap)
        if overlap_matches:
            matched = overlap_matches
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


def compute_source_quality_score(row: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    image_count = 0
    images_raw = row.get("images_json") or row.get("images")
    if isinstance(images_raw, str):
        try:
            images_raw = json.loads(images_raw)
        except Exception:
            images_raw = []
    if isinstance(images_raw, list):
        image_count = len([x for x in images_raw if x])
    has_dims = _candidate_size(row) is not None
    has_description = bool(str(row.get("description") or row.get("vlm_description_text") or "").strip())
    has_brand = bool(str(row.get("brand") or "").strip())
    has_image_colors = bool(_candidate_image_color_tokens(row))
    source_text = " ".join(
        str(row.get(key) or "").lower()
        for key in ("unique_key", "source_site", "product_url", "model_page_url", "brand", "title")
    )
    if isinstance(images_raw, list):
        source_text += " " + " ".join(str(x).lower() for x in images_raw[:8])
    trusted_product_catalog = any(pattern in source_text for pattern in TRUSTED_PRODUCT_IMAGE_PATTERNS)
    scene_render_marketplace = any(pattern in source_text for pattern in SCENE_RENDER_IMAGE_PATTERNS)
    score = (
        (0.35 if image_count else 0.0)
        + (0.25 if has_dims else 0.0)
        + (0.20 if has_description else 0.0)
        + (0.10 if has_brand else 0.0)
        + (0.10 if has_image_colors else 0.0)
        + (0.18 if image_count and trusted_product_catalog else 0.0)
        - (0.08 if image_count and scene_render_marketplace and not trusted_product_catalog else 0.0)
    )
    score = max(0.0, min(1.0, score))
    return score, {
        "source_quality_score": round(score, 6),
        "source_quality": {
            "image_count": image_count,
            "has_dimensions": has_dims,
            "has_description": has_description,
            "has_brand": has_brand,
            "has_image_colors": has_image_colors,
            "trusted_product_catalog": trusted_product_catalog,
            "scene_render_marketplace": scene_render_marketplace,
        },
    }


def compute_price_score(row: dict[str, Any], price_stats: dict[str, dict[str, float]], mode: str | None) -> tuple[float, dict[str, Any]]:
    price = _safe_float(row.get("price_value"))
    normalized_mode = normalize_selection_mode(mode)
    if price is None or price <= 0:
        score = 0.2 if normalized_mode in {"cheapest", "cheapest_top20"} else 0.5
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
        float(scores.get("style_score") or 0.0) * 0.32
        + float(scores.get("color_score") or 0.0) * 0.23
        + float(scores.get("image_color_score") or 0.0) * 0.15
        + float(scores.get("material_score") or 0.0) * 0.15
        + float(scores.get("description_score") or 0.0) * 0.15
    )
    return score, {"design_similarity_score": round(score, 6)}


def score_candidate_for_mode(
    *,
    target: dict[str, Any],
    row: dict[str, Any],
    room_design_spec: dict[str, Any],
    mode: str,
    price_stats: dict[str, dict[str, float]],
) -> CandidateScore:
    scores: dict[str, float] = {}
    breakdown: dict[str, Any] = {}
    for value, info in [
        compute_category_score(target, row),
        compute_size_score(target, row),
        compute_color_score(target, row, room_design_spec),
        compute_image_color_score(target, row, room_design_spec),
        compute_material_score(target, row, room_design_spec),
        compute_style_score(target, row, room_design_spec),
        compute_epoch_score(target, row, room_design_spec),
        compute_description_score(target, row, room_design_spec),
        compute_price_score(row, price_stats, mode),
        compute_asset_availability_score(row),
        compute_source_quality_score(row),
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
    breakdown["score_schema"] = "supplier_design_scores/v2"
    acceptable = bool(mode_info.get("gate_passed", True))
    hard_reject_reason = None
    notes: list[str] = []
    if breakdown.get("category_match_v2") not in {"exact_group", "same_family"}:
        acceptable = False
        hard_reject_reason = "category_mismatch"
    if breakdown.get("scale_reject_reason"):
        acceptable = False
        hard_reject_reason = str(breakdown.get("scale_reject_reason"))
    if normalize_selection_mode(mode) == "best_visual_reference":
        group = str(target.get("semantic_group") or target.get("category") or "").strip().lower()
        source_quality = breakdown.get("source_quality") if isinstance(breakdown.get("source_quality"), dict) else {}
        has_visual_reference = bool(
            int(source_quality.get("image_count") or 0) > 0
            or source_quality.get("has_image_colors")
            or str(row.get("vlm_description_text") or row.get("vlm_description_summary") or "").strip()
        )
        if group == "bed" and breakdown.get("size_score_reason") == "missing_dimensions":
            acceptable = False
            hard_reject_reason = "missing_dimensions_for_bed"
        if not has_visual_reference and not _candidate_has_ready_real_asset(row):
            acceptable = False
            hard_reject_reason = "missing_visual_reference_images"
    if breakdown.get("asset_availability") == "missing_asset":
        notes.append("asset_missing_but_allowed_for_reference_generation")
    return CandidateScore(
        total=final_score,
        acceptable=acceptable,
        hard_reject_reason=hard_reject_reason,
        breakdown=breakdown,
        notes=notes,
    )


def rank_candidate_for_mode(
    *,
    target: dict[str, Any],
    row: dict[str, Any],
    room_design_spec: dict[str, Any],
    mode: str,
    price_stats: dict[str, dict[str, float]],
) -> tuple[float, dict[str, Any]]:
    result = score_candidate_for_mode(
        target=target,
        row=row,
        room_design_spec=room_design_spec,
        mode=mode,
        price_stats=price_stats,
    )
    breakdown = dict(result.breakdown)
    breakdown["candidate_score_acceptable"] = result.acceptable
    breakdown["candidate_score_hard_reject_reason"] = result.hard_reject_reason
    breakdown["candidate_score_notes"] = result.notes
    return result.total, breakdown
