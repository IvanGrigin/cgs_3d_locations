#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any


SELECTION_MODE_WEIGHTS: dict[str, dict[str, float]] = {
    "legacy_asset_priority": {
        "category_score": 0.30,
        "size_score": 0.20,
        "asset_availability_score": 0.15,
        "style_score": 0.10,
        "color_score": 0.05,
        "price_score": 0.20,
    },
    "cheapest": {
        "category_score": 0.24,
        "size_score": 0.24,
        "style_score": 0.14,
        "color_score": 0.12,
        "material_score": 0.08,
        "description_score": 0.06,
        "asset_availability_score": 0.04,
        "price_score": 0.08,
    },
    "cheapest_top20": {
        "category_score": 0.24,
        "size_score": 0.26,
        "style_score": 0.16,
        "color_score": 0.14,
        "material_score": 0.08,
        "description_score": 0.06,
        "image_color_score": 0.04,
        "price_score": 0.02,
    },
    "optimal": {
        "category_score": 0.22,
        "size_score": 0.18,
        "style_score": 0.15,
        "color_score": 0.12,
        "material_score": 0.10,
        "description_score": 0.08,
        "price_score": 0.10,
        "asset_availability_score": 0.05,
    },
    "best_match": {
        "size_score": 0.30,
        "style_score": 0.18,
        "color_score": 0.16,
        "material_score": 0.10,
        "description_score": 0.10,
        "image_color_score": 0.08,
        "price_score": 0.04,
        "asset_availability_score": 0.02,
        "source_quality_score": 0.02,
    },
    "best_match_v1": {
        "category_score": 0.18,
        "size_score": 0.12,
        "style_score": 0.22,
        "color_score": 0.16,
        "material_score": 0.12,
        "description_score": 0.12,
        "design_similarity_score": 0.05,
        "asset_availability_score": 0.03,
    },
    "best_match_v2": {
        "size_score": 0.30,
        "style_score": 0.18,
        "color_score": 0.16,
        "material_score": 0.10,
        "description_score": 0.10,
        "image_color_score": 0.08,
        "price_score": 0.04,
        "asset_availability_score": 0.02,
        "source_quality_score": 0.02,
    },
    "best_visual_reference": {
        "size_score": 0.34,
        "style_score": 0.20,
        "color_score": 0.18,
        "material_score": 0.10,
        "description_score": 0.08,
        "image_color_score": 0.06,
        "price_score": 0.01,
        "asset_availability_score": 0.01,
        "source_quality_score": 0.02,
    },
}


SELECTION_MODE_GATES: dict[str, dict[str, float]] = {
    "legacy_asset_priority": {
        "category_score": 0.68,
        "size_score": 0.45,
        "asset_availability_score": 0.25,
        "style_score": 0.10,
    },
    "cheapest": {
        "category_score": 0.68,
        "size_score": 0.30,
    },
    "cheapest_top20": {
        "category_score": 0.68,
        "size_score": 0.30,
    },
    "optimal": {
        "category_score": 0.68,
        "size_score": 0.45,
        "asset_availability_score": 0.25,
    },
    "best_match": {
        "category_score": 0.68,
        "size_score": 0.20,
    },
    "best_match_v1": {
        "category_score": 0.68,
        "size_score": 0.40,
        "asset_availability_score": 0.20,
    },
    "best_match_v2": {
        "category_score": 0.68,
        "size_score": 0.20,
    },
    "best_visual_reference": {
        "category_score": 0.68,
        "size_score": 0.20,
    },
}


SELECTION_MODES: dict[str, dict[str, Any]] = {
    "legacy_asset_priority": {
        "category_policy": "score_and_gate",
        "asset_policy": "strong_reuse_bonus",
        "allow_missing_local_asset": False,
        "allow_missing_downloadable_asset": False,
        "prefer_image_rich_items": False,
        "weights": SELECTION_MODE_WEIGHTS["legacy_asset_priority"],
    },
    "best_match_v1": {
        "category_policy": "score_and_gate",
        "asset_policy": "reuse_bonus",
        "allow_missing_local_asset": True,
        "allow_missing_downloadable_asset": True,
        "prefer_image_rich_items": False,
        "weights": SELECTION_MODE_WEIGHTS["best_match_v1"],
    },
    "best_match_v2": {
        "category_policy": "hard_gate",
        "asset_policy": "optional_tiebreaker",
        "allow_missing_local_asset": True,
        "allow_missing_downloadable_asset": True,
        "prefer_image_rich_items": True,
        "weights": SELECTION_MODE_WEIGHTS["best_match_v2"],
    },
    "cheapest": {
        "category_policy": "hard_gate_then_lowest_price",
        "asset_policy": "optional_tiebreaker",
        "allow_missing_local_asset": True,
        "allow_missing_downloadable_asset": True,
        "prefer_image_rich_items": False,
        "weights": SELECTION_MODE_WEIGHTS["cheapest"],
    },
    "cheapest_top20": {
        "category_policy": "hard_gate_then_cheapest_of_top20",
        "asset_policy": "optional_tiebreaker",
        "allow_missing_local_asset": True,
        "allow_missing_downloadable_asset": True,
        "prefer_image_rich_items": True,
        "weights": SELECTION_MODE_WEIGHTS["cheapest_top20"],
    },
    "best_visual_reference": {
        "category_policy": "hard_gate",
        "asset_policy": "optional_tiebreaker",
        "allow_missing_local_asset": True,
        "allow_missing_downloadable_asset": True,
        "prefer_image_rich_items": True,
        "weights": SELECTION_MODE_WEIGHTS["best_visual_reference"],
    },
}


def normalize_selection_mode(mode: str | None) -> str:
    text = str(mode or "optimal").strip().lower()
    if text in {"balanced", "cheap_style", "style"}:
        return {"balanced": "optimal", "cheap_style": "optimal", "style": "best_match"}[text]
    if text in {"visual_reference", "most_suitable", "best_suitable"}:
        return "best_visual_reference"  # pragma: no cover
    if text in {"min_price", "minimal_price", "minimum_price", "lowest_price"}:
        return "cheapest"  # pragma: no cover
    if text in {"cheap_top20", "cheapest_top_20", "cheapest_of_top20", "cheapest_suitable_top20"}:
        return "cheapest_top20"  # pragma: no cover
    if text not in SELECTION_MODE_WEIGHTS:
        return "optimal"  # pragma: no cover
    return text


def combine_scores_for_mode(scores: dict[str, Any], mode: str | None) -> tuple[float, dict[str, Any]]:
    normalized_mode = normalize_selection_mode(mode)
    weights = SELECTION_MODE_WEIGHTS[normalized_mode]
    gates = SELECTION_MODE_GATES[normalized_mode]
    gate_failures: list[str] = []
    for key, minimum in gates.items():
        try:
            value = float(scores.get(key) or 0.0)
        except Exception:  # pragma: no cover
            value = 0.0  # pragma: no cover
        if value < minimum:
            gate_failures.append(f"{key}<{minimum:.2f}")

    weighted_sum = 0.0
    total_weight = 0.0
    for key, weight in weights.items():
        try:
            value = max(0.0, min(1.0, float(scores.get(key) or 0.0)))
        except Exception:  # pragma: no cover
            value = 0.0  # pragma: no cover
        weighted_sum += value * weight
        total_weight += weight
    final_score = weighted_sum / max(total_weight, 1e-6)
    if gate_failures:
        final_score *= 0.35

    return final_score, {
        "selection_mode": normalized_mode,
        "weights": weights,
        "hard_gates": gates,
        "gate_failures": gate_failures,
        "gate_passed": not gate_failures,
        "mode_policy": SELECTION_MODES.get(normalized_mode, {}),
    }
