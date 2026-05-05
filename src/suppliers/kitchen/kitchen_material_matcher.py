from __future__ import annotations

import statistics
from typing import Any

from .kitchen_constants import MATERIAL_ROLE_ALIASES, MODE_WEIGHTS, ROLE_PRICE_DEFAULTS_RUB, SELECTION_MODES
from .kitchen_text_features import clamp01, normalize_color_request, normalize_text, score_keyword_overlap


def _material_text(material: dict[str, Any]) -> str:
    visual = material.get("visual") or {}
    return " ".join(
        str(x)
        for x in (
            material.get("name"),
            material.get("sku"),
            material.get("brand"),
            material.get("raw_category"),
            material.get("material_type"),
            material.get("kitchen_role"),
            " ".join(visual.get("base_colors") or []),
            visual.get("pattern"),
            visual.get("finish"),
            " ".join(visual.get("style_tags") or []),
        )
        if x
    )


def _price_stats(materials: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    by_role: dict[str, list[float]] = {}
    for material in materials:
        price = material.get("price")
        if isinstance(price, (int, float)) and price > 0:
            by_role.setdefault(material.get("kitchen_role") or "unknown", []).append(float(price))
    stats: dict[str, tuple[float, float]] = {}
    for role, values in by_role.items():
        values = sorted(values)
        p_min = values[0]
        p90 = values[min(len(values) - 1, int(round(0.9 * (len(values) - 1))))] if len(values) >= 10 else values[-1]
        stats[role] = (p_min, p90 if p90 > p_min else p_min + 1.0)
    return stats


def _price_score(material: dict[str, Any], stats: dict[str, tuple[float, float]], mode: str) -> float:
    price = material.get("price")
    role = material.get("kitchen_role") or "unknown"
    if not isinstance(price, (int, float)) or price <= 0:
        return 0.2 if mode == "cheapest" else 0.5
    p_min, p90 = stats.get(role, (ROLE_PRICE_DEFAULTS_RUB.get(role, 1000.0), ROLE_PRICE_DEFAULTS_RUB.get(role, 1000.0) * 2))
    return clamp01(1.0 - ((float(price) - p_min) / max(1.0, p90 - p_min)))


def _availability_score(material: dict[str, Any]) -> float:
    availability = normalize_text(material.get("availability"))
    if "in_stock" in availability or "налич" in availability:
        return 1.0
    if "под заказ" in availability or "order" in availability:
        return 0.55
    return 0.45


def _role_score(material: dict[str, Any], target_role: str) -> float:
    aliases = MATERIAL_ROLE_ALIASES.get(target_role, (target_role,))
    role = material.get("kitchen_role")
    if role == aliases[0]:
        return 1.0
    if role in aliases:
        return 0.72
    return 0.0


def _desired_colors_for_role(design_spec: dict[str, Any], target_role: str) -> list[str]:
    palette = design_spec.get("palette") or {}
    if target_role == "facade":
        return normalize_color_request(palette.get("facades"))
    if target_role == "countertop":
        return normalize_color_request(palette.get("countertop"))
    if target_role == "backsplash":
        return normalize_color_request(palette.get("backsplash"))
    if target_role in {"edge_band", "wall_plinth", "joint_profile", "end_profile", "corner_profile"}:
        return normalize_color_request((palette.get("facades") or []) + (palette.get("countertop") or []) + (palette.get("accent") or []))
    return normalize_color_request(palette.get("facades") or palette.get("countertop") or [])


def _color_score(material: dict[str, Any], design_spec: dict[str, Any], target_role: str) -> float:
    desired = set(_desired_colors_for_role(design_spec, target_role))
    if not desired:
        return 0.5
    material_colors = set((material.get("visual") or {}).get("base_colors") or [])
    pattern = (material.get("visual") or {}).get("pattern")
    if not material_colors:
        return score_keyword_overlap(_material_text(material), sorted(desired))
    intersection = material_colors.intersection(desired)
    score = 0.55 + 0.35 * (len(intersection) / max(1, len(desired))) if intersection else 0.05
    if "wood" in desired and material_colors.intersection({"wood", "light_wood", "dark_wood"}):
        score = max(score, 0.85)
    if "light_wood" in desired and "light_wood" in material_colors:
        score = max(score, 0.95)
    if "stone" in desired and pattern in {"stone", "marble", "concrete", "terrazzo"}:
        score = max(score, 0.88)
    if "white" in desired and "white" in material_colors:
        score = max(score, 0.75)
    if target_role in {"facade", "countertop", "backsplash"} and not intersection and not ("stone" in desired and pattern in {"stone", "marble", "concrete", "terrazzo"}):
        score = min(score, 0.18)
    return clamp01(score)


def _style_score(material: dict[str, Any], design_spec: dict[str, Any]) -> float:
    style = design_spec.get("style") or {}
    desired = [x for x in [style.get("primary"), *(style.get("secondary") or [])] if x]
    if not desired:
        return 0.5
    material_tags = set((material.get("visual") or {}).get("style_tags") or [])
    direct = len(material_tags.intersection(desired)) / max(1, len(set(desired)))
    return clamp01(max(direct, score_keyword_overlap(_material_text(material), desired)))


def _pattern_score(material: dict[str, Any], design_spec: dict[str, Any], target_role: str) -> float:
    pattern = (material.get("visual") or {}).get("pattern") or "decor"
    intent = design_spec.get("materials_intent") or {}
    desired = intent.get(target_role) or intent.get(f"{target_role}s") or []
    if target_role == "facade":
        desired = intent.get("facades", [])
    elif target_role == "countertop":
        desired = intent.get("countertop", [])
    elif target_role == "backsplash":
        desired = intent.get("backsplash", [])
    if not desired:
        return 0.5
    if pattern in {normalize_text(x) for x in desired}:
        return 1.0
    return score_keyword_overlap(" ".join([pattern, _material_text(material)]), desired)


def _finish_score(material: dict[str, Any], design_spec: dict[str, Any], target_role: str) -> float:
    del target_role
    finish = normalize_text((material.get("visual") or {}).get("finish"))
    style = (design_spec.get("style") or {}).get("primary")
    if not finish or finish == "unknown":
        return 0.5
    if style in {"scandinavian", "japandi", "minimalism"}:
        return 1.0 if "mat" in finish or "ms" in finish or "pe" in finish else 0.55
    if style == "modern":
        return 0.9 if finish in {"matte", "gloss", "textured"} or finish else 0.5
    return 0.7


def _dimension_score(material: dict[str, Any], target_role: str, layout_plan: dict[str, Any]) -> float:
    dims = material.get("dimensions") or {}
    length = dims.get("length_mm")
    width = dims.get("width_mm")
    thickness = dims.get("thickness_mm")
    if target_role == "countertop":
        countertop_width = sum(seg.get("width_mm", 0) for seg in layout_plan.get("countertop_segments") or [])
        score = 0.5
        if length and length >= min(countertop_width, 2400):
            score += 0.25
        if width and width >= 600:
            score += 0.15
        if thickness and thickness >= 38:
            score += 0.10
        return clamp01(score)
    if target_role == "backsplash":
        score = 0.45
        if length and length >= 2400:
            score += 0.25
        if width and width >= 600:
            score += 0.25
        return clamp01(score)
    if target_role in {"facade", "body"}:
        if length and width and length >= 2000 and width >= 1000:
            return 1.0
        if length and width and length >= 1200 and width >= 600:
            return 0.75
        return 0.5
    if target_role == "edge_band":
        return 1.0 if width and 18 <= width <= 45 else 0.65
    return 0.6


def _durability_score(material: dict[str, Any], target_role: str) -> float:
    flags = material.get("flags") or {}
    if target_role == "countertop":
        score = 0.55
        if flags.get("is_moisture_resistant") or "влагостой" in normalize_text(_material_text(material)):
            score += 0.35
        thickness = (material.get("dimensions") or {}).get("thickness_mm", 0)
        if thickness and thickness >= 38:
            score += 0.10
        return clamp01(score)
    if target_role == "backsplash":
        return 0.85 if material.get("kitchen_role") == "backsplash_panel" else 0.45
    if target_role in {"facade", "body"}:
        return 0.75 if material.get("kitchen_role") in {"facade_sheet", "board_sheet"} else 0.5
    return 0.65


def _compatibility_score(material: dict[str, Any], target_role: str, selected: dict[str, dict[str, Any]]) -> float:
    if not selected:
        return 0.5
    material_colors = set((material.get("visual") or {}).get("base_colors") or [])
    material_pattern = (material.get("visual") or {}).get("pattern")

    def compare(other: dict[str, Any] | None) -> float:
        if not other:
            return 0.5
        other_colors = set((other.get("visual") or {}).get("base_colors") or [])
        other_pattern = (other.get("visual") or {}).get("pattern")
        color_hit = 0.6 if material_colors.intersection(other_colors) else 0.35
        pattern_hit = 0.3 if material_pattern == other_pattern else 0.1
        if material_pattern in {"stone", "marble", "concrete", "terrazzo"} and other_pattern in {"plain", "wood"}:
            pattern_hit = max(pattern_hit, 0.25)
        if material_pattern == "plain" and other_pattern in {"stone", "marble", "wood", "concrete"}:
            pattern_hit = max(pattern_hit, 0.25)
        return clamp01(color_hit + pattern_hit)

    if target_role == "backsplash":
        return compare(selected.get("countertop"))
    if target_role == "edge_band":
        return max(compare(selected.get("facade")), compare(selected.get("countertop")))
    if target_role == "wall_plinth":
        return compare(selected.get("countertop"))
    if target_role == "countertop":
        return compare(selected.get("facade"))
    return 0.5


def score_material(
    material: dict[str, Any],
    target_role: str,
    design_spec: dict[str, Any],
    layout_plan: dict[str, Any],
    mode: str,
    price_stats: dict[str, tuple[float, float]],
    selected_context: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if mode not in SELECTION_MODES:
        raise ValueError(f"Unknown selection mode: {mode}")
    breakdown = {
        "role_score": _role_score(material, target_role),
        "color_score": _color_score(material, design_spec, target_role),
        "style_score": _style_score(material, design_spec),
        "pattern_score": _pattern_score(material, design_spec, target_role),
        "finish_score": _finish_score(material, design_spec, target_role),
        "dimension_score": _dimension_score(material, target_role, layout_plan),
        "durability_score": _durability_score(material, target_role),
        "availability_score": _availability_score(material),
        "price_score": _price_score(material, price_stats, mode),
        "compatibility_score": _compatibility_score(material, target_role, selected_context or {}),
    }
    final_score = sum(breakdown[key] * MODE_WEIGHTS[mode].get(key, 0.0) for key in breakdown)
    if breakdown["role_score"] <= 0.0:
        final_score *= 0.05
    if target_role in {"facade", "countertop", "backsplash"} and breakdown["color_score"] < 0.25:
        final_score *= 0.45 if mode != "cheapest" else 0.70
    if material.get("flags", {}).get("is_accent_only") and target_role != "edge_band":
        final_score *= 0.2
    return {"final_score": round(float(clamp01(final_score)), 6), "score_breakdown": {k: round(float(v), 4) for k, v in breakdown.items()}}


def _candidate_pool(materials: list[dict[str, Any]], target_role: str) -> list[dict[str, Any]]:
    aliases = set(MATERIAL_ROLE_ALIASES.get(target_role, (target_role,)))
    return [m for m in materials if m.get("kitchen_role") in aliases]


def _select_one(
    materials: list[dict[str, Any]],
    target_role: str,
    design_spec: dict[str, Any],
    layout_plan: dict[str, Any],
    mode: str,
    price_stats: dict[str, tuple[float, float]],
    selected_context: dict[str, dict[str, Any]],
    top_n: int,
) -> dict[str, Any] | None:
    pool = _candidate_pool(materials, target_role)
    if not pool:
        return None
    scored = [{"material": material, **score_material(material, target_role, design_spec, layout_plan, mode, price_stats, selected_context)} for material in pool]
    scored.sort(key=lambda x: (x["final_score"], -float(x["material"].get("price") or 10**12)), reverse=True)
    chosen = scored[0]
    return {"chosen_material": chosen["material"], "final_score": chosen["final_score"], "score_breakdown": chosen["score_breakdown"], "top_candidates": scored[:top_n]}


def select_kitchen_materials(
    materials: list[dict[str, Any]],
    design_spec: dict[str, Any],
    layout_plan: dict[str, Any],
    mode: str = "optimal",
    top_n: int = 5,
) -> dict[str, Any]:
    if mode not in SELECTION_MODES:
        raise ValueError(f"Unknown selection mode: {mode}")
    price_stats = _price_stats(materials)
    selected: dict[str, dict[str, Any]] = {}
    result: dict[str, Any] = {"mode": mode, "materials": {}, "warnings": []}
    for target_role in ("facade", "body", "countertop", "backsplash", "edge_band", "wall_plinth", "joint_profile", "end_profile", "corner_profile"):
        selection = _select_one(materials, target_role, design_spec, layout_plan, mode, price_stats, selected, top_n)
        if selection is None:
            result["warnings"].append(f"no_material_candidates_for_role:{target_role}")
            continue
        material = selection["chosen_material"]
        selected[target_role] = material
        result["materials"][target_role] = {
            "target_role": target_role,
            "chosen_material": material,
            "final_score": selection["final_score"],
            "score_breakdown": selection["score_breakdown"],
            "top_candidates": [{"material": item["material"], "final_score": item["final_score"], "score_breakdown": item["score_breakdown"]} for item in selection["top_candidates"]],
        }
    result["palette_consistency_score"] = estimate_palette_consistency(selected)
    return result


def estimate_palette_consistency(selected: dict[str, dict[str, Any]]) -> float:
    if not selected:
        return 0.0
    scores: list[float] = []
    for left, right in (("facade", "countertop"), ("countertop", "backsplash"), ("facade", "edge_band"), ("countertop", "wall_plinth")):
        a, b = selected.get(left), selected.get(right)
        if not a or not b:
            continue
        a_colors = set((a.get("visual") or {}).get("base_colors") or [])
        b_colors = set((b.get("visual") or {}).get("base_colors") or [])
        a_pattern = (a.get("visual") or {}).get("pattern")
        b_pattern = (b.get("visual") or {}).get("pattern")
        score = 0.45
        if a_colors.intersection(b_colors):
            score += 0.25
        if a_pattern == b_pattern:
            score += 0.20
        if {a_pattern, b_pattern}.intersection({"plain"}) and {a_pattern, b_pattern}.intersection({"wood", "stone", "marble", "concrete"}):
            score += 0.15
        scores.append(clamp01(score))
    return round(float(statistics.mean(scores)), 4) if scores else 0.5
