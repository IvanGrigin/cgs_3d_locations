from __future__ import annotations

from typing import Any

from .kitchen_text_features import detect_color_families, normalize_color_request, normalize_text


def _infer_primary_style(user_prompt: str) -> str:
    text = normalize_text(user_prompt)
    if any(x in text for x in ("сканди", "scandinavian", "nordic")):
        return "scandinavian"
    if any(x in text for x in ("джапанди", "japandi")):
        return "japandi"
    if any(x in text for x in ("лофт", "loft", "industrial", "индастри")):
        return "loft"
    if any(x in text for x in ("класс", "classic", "неокласс")):
        return "classic"
    if any(x in text for x in ("минимал", "minimal")):
        return "minimalism"
    return "modern"


def _infer_palette_from_prompt(user_prompt: str, recommended_colors: dict[str, Any] | None) -> dict[str, list[str]]:
    recommended_colors = recommended_colors or {}
    prompt_colors = sorted(detect_color_families(user_prompt))
    palette = {
        "facades": normalize_color_request(recommended_colors.get("facades")) or [],
        "countertop": normalize_color_request(recommended_colors.get("countertop")) or [],
        "backsplash": normalize_color_request(recommended_colors.get("backsplash")) or [],
        "accent": normalize_color_request(recommended_colors.get("accent") or recommended_colors.get("accents")) or [],
    }
    if prompt_colors:
        for key in ("facades", "countertop", "backsplash"):
            if not palette[key]:
                palette[key] = prompt_colors
    palette["facades"] = palette["facades"] or ["white", "beige", "light_wood"]
    palette["countertop"] = palette["countertop"] or ["stone", "white", "gray"]
    palette["backsplash"] = palette["backsplash"] or ["white", "beige", "stone"]
    palette["accent"] = palette["accent"] or ["black", "metal"]
    return palette


def _style_material_intent(style: str) -> dict[str, list[str]]:
    presets = {
        "scandinavian": {
            "facades": ["matte", "plain", "wood", "light", "warm neutral"],
            "countertop": ["moisture resistant", "light stone", "wood-compatible", "38 mm preferred"],
            "backsplash": ["light", "stone", "plain", "600 mm high"],
            "edge": ["match facade", "match countertop"],
        },
        "japandi": {
            "facades": ["matte", "wood", "warm", "natural"],
            "countertop": ["stone", "beige", "gray", "moisture resistant"],
            "backsplash": ["quiet", "matte", "stone", "600 mm high"],
            "edge": ["match facade"],
        },
        "loft": {
            "facades": ["dark", "wood", "graphite", "matte"],
            "countertop": ["concrete", "stone", "dark", "moisture resistant"],
            "backsplash": ["concrete", "gray", "metal-compatible", "600 mm high"],
            "edge": ["black", "graphite", "match countertop"],
        },
        "classic": {
            "facades": ["cream", "wood", "warm", "soft"],
            "countertop": ["stone", "marble", "beige", "moisture resistant"],
            "backsplash": ["light", "marble", "plain", "600 mm high"],
            "edge": ["match facade"],
        },
    }
    return presets.get(
        style,
        {
            "facades": ["smooth", "matte", "plain", "minimal"],
            "countertop": ["stone", "marble", "concrete", "moisture resistant", "38 mm preferred"],
            "backsplash": ["stone-compatible", "plain", "600 mm high"],
            "edge": ["match facade", "match countertop"],
        },
    )


def build_kitchen_design_spec(
    user_prompt: str,
    recommended_colors: dict[str, Any] | None = None,
    budget: dict[str, Any] | None = None,
    appliances: dict[str, Any] | None = None,
    room_meta: dict[str, Any] | None = None,
    llm_client: Any | None = None,
) -> dict[str, Any]:
    del llm_client
    appliances = appliances or {}
    style = _infer_primary_style(user_prompt)
    return {
        "source_prompt": user_prompt,
        "expanded_kitchen_description": (
            "Procedural kitchen assembled from standard cabinet modules, supplier surface materials, "
            "continuous countertop segments, 600 mm backsplash, upper wall cabinets, appliance slots and sink/cooktop cutouts."
        ),
        "style": {
            "primary": style,
            "secondary": ["minimalism", "contemporary"] if style == "modern" else ["modern"],
            "forbidden": ["baroque", "rococo", "ornate heavy classical", "random neon accents"],
        },
        "palette": _infer_palette_from_prompt(user_prompt, recommended_colors),
        "materials_intent": _style_material_intent(style),
        "functional_requirements": {
            "main_sink": bool(appliances.get("sink", True)),
            "entry_handwash": bool(appliances.get("entry_handwash", False)),
            "fridge": bool(appliances.get("fridge", False)),
            "washing_machine": bool(appliances.get("washing_machine", False)),
            "dishwasher": bool(appliances.get("dishwasher", False)),
            "cooktop": bool(appliances.get("cooktop", True)),
            "oven": bool(appliances.get("oven", True)),
            "hood": bool(appliances.get("hood", True)),
            "microwave": bool(appliances.get("microwave", False)),
            "upper_cabinets": True,
        },
        "layout_constraints": {
            "backsplash_height_mm": 600,
            "countertop_depth_mm": 600,
            "countertop_thickness_mm": 38,
            "prefer_fridge_at_edge": True,
            "prefer_sink_near_plumbing": True,
            "prefer_water_appliances_near_sink": True,
            "avoid_cooktop_adjacent_to_fridge": True,
        },
        "budget": budget or {},
        "room_meta": room_meta or {},
        "consistency_rules": [
            "Countertop and backsplash should be visually compatible.",
            "Facade edge band should match facade color or be intentionally neutral.",
            "Large surfaces should follow recommended kitchen colors.",
            "Do not use accent edge bands as primary facade surfaces.",
        ],
    }
