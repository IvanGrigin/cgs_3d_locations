#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any


SELECTION_MODE_WEIGHTS: dict[str, dict[str, float]] = {
    "cheapest": {
        "category_score": 0.30,
        "size_score": 0.20,
        "asset_availability_score": 0.15,
        "style_score": 0.10,
        "color_score": 0.05,
        "price_score": 0.20,
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
        "category_score": 0.18,
        "size_score": 0.12,
        "style_score": 0.22,
        "color_score": 0.16,
        "material_score": 0.12,
        "description_score": 0.12,
        "design_similarity_score": 0.05,
        "asset_availability_score": 0.03,
    },
}


SELECTION_MODE_GATES: dict[str, dict[str, float]] = {
    "cheapest": {
        "category_score": 0.68,
        "size_score": 0.45,
        "asset_availability_score": 0.25,
        "style_score": 0.10,
    },
    "optimal": {
        "category_score": 0.68,
        "size_score": 0.45,
        "asset_availability_score": 0.25,
    },
    "best_match": {
        "category_score": 0.68,
        "size_score": 0.40,
        "asset_availability_score": 0.20,
    },
}


def normalize_selection_mode(mode: str | None) -> str:
    text = str(mode or "optimal").strip().lower()
    if text in {"balanced", "cheap_style", "style"}:
        return {"balanced": "optimal", "cheap_style": "optimal", "style": "best_match"}[text]
    if text not in SELECTION_MODE_WEIGHTS:
        return "optimal"
    return text


def combine_scores_for_mode(scores: dict[str, Any], mode: str | None) -> tuple[float, dict[str, Any]]:
    normalized_mode = normalize_selection_mode(mode)
    weights = SELECTION_MODE_WEIGHTS[normalized_mode]
    gates = SELECTION_MODE_GATES[normalized_mode]
    gate_failures: list[str] = []
    for key, minimum in gates.items():
        try:
            value = float(scores.get(key) or 0.0)
        except Exception:
            value = 0.0
        if value < minimum:
            gate_failures.append(f"{key}<{minimum:.2f}")

    weighted_sum = 0.0
    total_weight = 0.0
    for key, weight in weights.items():
        try:
            value = max(0.0, min(1.0, float(scores.get(key) or 0.0)))
        except Exception:
            value = 0.0
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
    }
