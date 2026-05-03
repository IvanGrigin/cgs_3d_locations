#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from src.prompt_compiler.schemas import CompiledPolicy


STYLE_PROFILES: dict[str, dict[str, Any]] = {
    "minimalism": {
        "description": "airy sparse layout with restrained decor and clean composition",
        "keywords": ["minimal", "minimalism", "minimalist", "clean", "airy", "decluttered"],
        "style_axes": {
            "density": 0.25,
            "wall_affinity": 0.50,
            "symmetry": 0.25,
            "decor_richness": 0.10,
            "surface_clutter": 0.10,
            "openness": 0.90,
            "ornament_level": 0.05,
        },
        "palette_base": ["white", "beige", "gray", "light_wood"],
        "material_family": ["wood", "fabric", "ceramic", "glass"],
        "infinigen_params": {
            "furniture_fullness_pct": 0.45,
            "obj_interior_obj_pct": 0.25,
            "obj_on_storage_pct": 0.15,
            "obj_on_nonstorage_pct": 0.08,
            "has_tv": False,
            "has_aquarium_tank": False,
            "has_birthday_balloons": False,
            "has_cocktail_tables": False,
            "has_kitchen_barstools": False,
        },
    },
    "scandinavian": {
        "description": "bright warm functional room with light woods and soft fabrics",
        "keywords": ["scandinavian", "nordic", "hygge", "light wood"],
        "style_axes": {
            "density": 0.40,
            "wall_affinity": 0.55,
            "symmetry": 0.35,
            "decor_richness": 0.25,
            "surface_clutter": 0.20,
            "openness": 0.78,
            "ornament_level": 0.15,
        },
        "palette_base": ["white", "beige", "light_wood", "gray"],
        "material_family": ["wood", "fabric", "ceramic"],
        "infinigen_params": {
            "furniture_fullness_pct": 0.55,
            "obj_interior_obj_pct": 0.35,
            "obj_on_storage_pct": 0.22,
            "obj_on_nonstorage_pct": 0.15,
            "has_tv": True,
            "has_aquarium_tank": False,
            "has_birthday_balloons": False,
            "has_cocktail_tables": False,
            "has_kitchen_barstools": False,
        },
    },
    "japandi": {
        "description": "calm low-clutter blend of Japanese restraint and Scandinavian warmth",
        "keywords": ["japandi", "japanese", "zen", "calm wood", "soft neutral", "japanese minimal", "japanese scandinavian"],
        "style_axes": {
            "density": 0.30,
            "wall_affinity": 0.50,
            "symmetry": 0.35,
            "decor_richness": 0.12,
            "surface_clutter": 0.10,
            "openness": 0.88,
            "ornament_level": 0.08,
        },
        "palette_base": ["beige", "brown", "black", "light_wood"],
        "material_family": ["wood", "fabric", "ceramic", "paper"],
        "infinigen_params": {
            "furniture_fullness_pct": 0.48,
            "obj_interior_obj_pct": 0.24,
            "obj_on_storage_pct": 0.12,
            "obj_on_nonstorage_pct": 0.08,
            "has_tv": False,
            "has_aquarium_tank": False,
            "has_birthday_balloons": False,
            "has_cocktail_tables": False,
            "has_kitchen_barstools": False,
        },
    },
    "wabi_sabi": {
        "description": "quiet imperfect natural room with strong emptiness and low ornament",
        "keywords": ["wabi", "sabi", "japanese", "organic", "imperfect", "earthy"],
        "style_axes": {
            "density": 0.28,
            "wall_affinity": 0.45,
            "symmetry": 0.15,
            "decor_richness": 0.12,
            "surface_clutter": 0.08,
            "openness": 0.92,
            "ornament_level": 0.06,
        },
        "palette_base": ["beige", "brown", "olive", "stone"],
        "material_family": ["wood", "ceramic", "fabric", "stone"],
        "infinigen_params": {
            "furniture_fullness_pct": 0.42,
            "obj_interior_obj_pct": 0.22,
            "obj_on_storage_pct": 0.10,
            "obj_on_nonstorage_pct": 0.07,
            "has_tv": False,
            "has_aquarium_tank": False,
            "has_birthday_balloons": False,
            "has_cocktail_tables": False,
            "has_kitchen_barstools": False,
        },
    },
    "modern": {
        "description": "clean contemporary room with controlled accents and moderate openness",
        "keywords": ["modern", "sleek", "clean lines", "contemporary modern"],
        "style_axes": {
            "density": 0.45,
            "wall_affinity": 0.55,
            "symmetry": 0.40,
            "decor_richness": 0.25,
            "surface_clutter": 0.18,
            "openness": 0.72,
            "ornament_level": 0.12,
        },
        "palette_base": ["white", "gray", "black", "wood"],
        "material_family": ["metal", "glass", "wood", "fabric"],
        "infinigen_params": {
            "furniture_fullness_pct": 0.58,
            "obj_interior_obj_pct": 0.36,
            "obj_on_storage_pct": 0.20,
            "obj_on_nonstorage_pct": 0.14,
            "has_tv": True,
            "has_aquarium_tank": False,
            "has_birthday_balloons": False,
            "has_cocktail_tables": False,
            "has_kitchen_barstools": True,
        },
    },
    "contemporary": {
        "description": "balanced current-day room with flexible composition and curated decor",
        "keywords": ["contemporary", "current", "updated", "curated"],
        "style_axes": {
            "density": 0.50,
            "wall_affinity": 0.52,
            "symmetry": 0.35,
            "decor_richness": 0.30,
            "surface_clutter": 0.22,
            "openness": 0.68,
            "ornament_level": 0.18,
        },
        "palette_base": ["beige", "gray", "black", "walnut"],
        "material_family": ["wood", "fabric", "glass", "metal"],
        "infinigen_params": {
            "furniture_fullness_pct": 0.60,
            "obj_interior_obj_pct": 0.38,
            "obj_on_storage_pct": 0.24,
            "obj_on_nonstorage_pct": 0.18,
            "has_tv": True,
            "has_aquarium_tank": False,
            "has_birthday_balloons": False,
            "has_cocktail_tables": False,
            "has_kitchen_barstools": True,
        },
    },
    "industrial": {
        "description": "raw loft-like room with dark materials, heavier forms, and moderate clutter",
        "keywords": ["industrial", "loft", "concrete", "black metal", "brick"],
        "style_axes": {
            "density": 0.55,
            "wall_affinity": 0.62,
            "symmetry": 0.28,
            "decor_richness": 0.24,
            "surface_clutter": 0.22,
            "openness": 0.62,
            "ornament_level": 0.14,
        },
        "palette_base": ["black", "gray", "brown", "brick"],
        "material_family": ["metal", "wood", "concrete", "leather"],
        "infinigen_params": {
            "furniture_fullness_pct": 0.63,
            "obj_interior_obj_pct": 0.42,
            "obj_on_storage_pct": 0.24,
            "obj_on_nonstorage_pct": 0.18,
            "has_tv": True,
            "has_aquarium_tank": False,
            "has_birthday_balloons": False,
            "has_cocktail_tables": True,
            "has_kitchen_barstools": True,
        },
    },
    "mid_century_modern": {
        "description": "balanced room with iconic furniture, moderate openness, and curated accents",
        "keywords": ["mid century", "mcm", "mid-century", "eames", "walnut"],
        "style_axes": {
            "density": 0.48,
            "wall_affinity": 0.48,
            "symmetry": 0.38,
            "decor_richness": 0.28,
            "surface_clutter": 0.20,
            "openness": 0.72,
            "ornament_level": 0.16,
        },
        "palette_base": ["walnut", "olive", "mustard", "cream"],
        "material_family": ["wood", "fabric", "leather", "metal"],
        "infinigen_params": {
            "furniture_fullness_pct": 0.57,
            "obj_interior_obj_pct": 0.34,
            "obj_on_storage_pct": 0.20,
            "obj_on_nonstorage_pct": 0.16,
            "has_tv": True,
            "has_aquarium_tank": False,
            "has_birthday_balloons": False,
            "has_cocktail_tables": True,
            "has_kitchen_barstools": False,
        },
    },
    "bohemian": {
        "description": "layered relaxed room with many textiles, decor, and eclectic surfaces",
        "keywords": ["bohemian", "boho", "eclectic", "layered", "colorful"],
        "style_axes": {
            "density": 0.58,
            "wall_affinity": 0.40,
            "symmetry": 0.12,
            "decor_richness": 0.70,
            "surface_clutter": 0.55,
            "openness": 0.48,
            "ornament_level": 0.50,
        },
        "palette_base": ["terracotta", "red", "mustard", "green"],
        "material_family": ["fabric", "wood", "rattan", "ceramic"],
        "infinigen_params": {
            "furniture_fullness_pct": 0.68,
            "obj_interior_obj_pct": 0.58,
            "obj_on_storage_pct": 0.42,
            "obj_on_nonstorage_pct": 0.35,
            "has_tv": False,
            "has_aquarium_tank": True,
            "has_birthday_balloons": False,
            "has_cocktail_tables": True,
            "has_kitchen_barstools": False,
        },
    },
    "classicism": {
        "description": "formal balanced room with stronger symmetry and restrained ornament",
        "keywords": ["classicism", "classical", "formal", "ordered", "balanced"],
        "style_axes": {
            "density": 0.55,
            "wall_affinity": 0.72,
            "symmetry": 0.80,
            "decor_richness": 0.42,
            "surface_clutter": 0.24,
            "openness": 0.55,
            "ornament_level": 0.35,
        },
        "palette_base": ["cream", "beige", "gold", "dark_wood"],
        "material_family": ["wood", "marble", "fabric", "brass"],
        "infinigen_params": {
            "furniture_fullness_pct": 0.64,
            "obj_interior_obj_pct": 0.44,
            "obj_on_storage_pct": 0.28,
            "obj_on_nonstorage_pct": 0.20,
            "has_tv": False,
            "has_aquarium_tank": False,
            "has_birthday_balloons": False,
            "has_cocktail_tables": True,
            "has_kitchen_barstools": False,
        },
    },
    "neoclassical": {
        "description": "refined symmetrical room with light classical detailing and composed layouts",
        "keywords": ["neoclassical", "neo classical", "elegant classic", "molding"],
        "style_axes": {
            "density": 0.52,
            "wall_affinity": 0.72,
            "symmetry": 0.82,
            "decor_richness": 0.38,
            "surface_clutter": 0.22,
            "openness": 0.58,
            "ornament_level": 0.28,
        },
        "palette_base": ["cream", "white", "gold", "gray"],
        "material_family": ["marble", "wood", "fabric", "metal"],
        "infinigen_params": {
            "furniture_fullness_pct": 0.62,
            "obj_interior_obj_pct": 0.40,
            "obj_on_storage_pct": 0.24,
            "obj_on_nonstorage_pct": 0.18,
            "has_tv": False,
            "has_aquarium_tank": False,
            "has_birthday_balloons": False,
            "has_cocktail_tables": True,
            "has_kitchen_barstools": False,
        },
    },
    "baroque": {
        "description": "ornate dense room with strong symmetry, wall composition, and rich decor",
        "keywords": ["baroque", "ornate", "opulent", "gold", "dramatic"],
        "style_axes": {
            "density": 0.72,
            "wall_affinity": 0.82,
            "symmetry": 0.88,
            "decor_richness": 0.80,
            "surface_clutter": 0.46,
            "openness": 0.35,
            "ornament_level": 0.88,
        },
        "palette_base": ["gold", "cream", "burgundy", "dark_wood"],
        "material_family": ["marble", "velvet", "wood", "brass"],
        "infinigen_params": {
            "furniture_fullness_pct": 0.78,
            "obj_interior_obj_pct": 0.60,
            "obj_on_storage_pct": 0.42,
            "obj_on_nonstorage_pct": 0.32,
            "has_tv": False,
            "has_aquarium_tank": False,
            "has_birthday_balloons": False,
            "has_cocktail_tables": True,
            "has_kitchen_barstools": False,
        },
    },
    "art_deco": {
        "description": "geometric glamorous room with rich materials and strong focal composition",
        "keywords": ["art deco", "deco", "geometric glam", "brass", "black and gold"],
        "style_axes": {
            "density": 0.58,
            "wall_affinity": 0.66,
            "symmetry": 0.70,
            "decor_richness": 0.45,
            "surface_clutter": 0.22,
            "openness": 0.52,
            "ornament_level": 0.55,
        },
        "palette_base": ["black", "gold", "cream", "emerald"],
        "material_family": ["metal", "glass", "velvet", "wood"],
        "infinigen_params": {
            "furniture_fullness_pct": 0.66,
            "obj_interior_obj_pct": 0.44,
            "obj_on_storage_pct": 0.26,
            "obj_on_nonstorage_pct": 0.20,
            "has_tv": False,
            "has_aquarium_tank": False,
            "has_birthday_balloons": False,
            "has_cocktail_tables": True,
            "has_kitchen_barstools": False,
        },
    },
    "rustic": {
        "description": "warm grounded room with natural materials and moderate density",
        "keywords": ["rustic", "cabin", "natural wood", "earthy", "country"],
        "style_axes": {
            "density": 0.56,
            "wall_affinity": 0.62,
            "symmetry": 0.28,
            "decor_richness": 0.34,
            "surface_clutter": 0.26,
            "openness": 0.56,
            "ornament_level": 0.24,
        },
        "palette_base": ["brown", "beige", "green", "stone"],
        "material_family": ["wood", "stone", "fabric", "leather"],
        "infinigen_params": {
            "furniture_fullness_pct": 0.63,
            "obj_interior_obj_pct": 0.42,
            "obj_on_storage_pct": 0.26,
            "obj_on_nonstorage_pct": 0.20,
            "has_tv": False,
            "has_aquarium_tank": False,
            "has_birthday_balloons": False,
            "has_cocktail_tables": True,
            "has_kitchen_barstools": False,
        },
    },
    "coastal": {
        "description": "light relaxed room with airy palette, openness, and soft decor",
        "keywords": ["coastal", "beach", "sea", "breezy", "light blue"],
        "style_axes": {
            "density": 0.38,
            "wall_affinity": 0.52,
            "symmetry": 0.32,
            "decor_richness": 0.25,
            "surface_clutter": 0.18,
            "openness": 0.82,
            "ornament_level": 0.18,
        },
        "palette_base": ["white", "sand", "blue", "light_wood"],
        "material_family": ["wood", "linen", "ceramic", "glass"],
        "infinigen_params": {
            "furniture_fullness_pct": 0.54,
            "obj_interior_obj_pct": 0.30,
            "obj_on_storage_pct": 0.18,
            "obj_on_nonstorage_pct": 0.12,
            "has_tv": False,
            "has_aquarium_tank": False,
            "has_birthday_balloons": False,
            "has_cocktail_tables": False,
            "has_kitchen_barstools": False,
        },
    },
}

STYLE_LABELS = list(STYLE_PROFILES.keys())


STYLE_PROFILE_ALIASES = {
    "baroque_inspired": "baroque",
}


def _normalize_room_type(value: Any) -> str:
    text = str(value or "").strip().lower()
    mapping = {
        "bedroom": "Bedroom",
        "livingroom": "LivingRoom",
        "living-room": "LivingRoom",
        "living room": "LivingRoom",
        "kitchen": "Kitchen",
        "bathroom": "Bathroom",
        "diningroom": "DiningRoom",
        "dining-room": "DiningRoom",
        "dining room": "DiningRoom",
    }
    return mapping.get(text, "Bedroom")


def infer_room_type_from_prompt(prompt_text: str, room_path: str | None = None) -> str:
    low = str(prompt_text or "").lower()
    if any(token in low for token in ("bathroom", "ван", "сануз", "toilet")):
        return "Bathroom"
    if any(token in low for token in ("kitchen", "кух")):
        return "Kitchen"
    if any(token in low for token in ("dining", "столов")):
        return "DiningRoom"
    if any(token in low for token in ("living", "гостин")):
        return "LivingRoom"
    if any(token in low for token in ("bedroom", "спаль", "kids", "детск")):
        return "Bedroom"
    name = str(Path(room_path).name if room_path else "").lower()
    if "bath" in name:
        return "Bathroom"
    if "kitchen" in name:
        return "Kitchen"
    if "dining" in name:
        return "DiningRoom"
    if "living" in name:
        return "LivingRoom"
    return "Bedroom"


def default_style_label_for_room(room_type: str) -> str:
    room_type = _normalize_room_type(room_type)
    if room_type == "LivingRoom":
        return "contemporary"
    if room_type == "Kitchen":
        return "modern"
    if room_type == "Bathroom":
        return "minimalism"
    if room_type == "DiningRoom":
        return "classicism"
    return "minimalism"


def style_label_choices() -> list[str]:
    return list(STYLE_LABELS)


def build_style_hint(profile: dict[str, Any]) -> str:
    label = str(profile.get("style_label") or "").replace("_", " ").strip()
    room_type = str(profile.get("room_type") or "").strip()
    palette = ", ".join(str(x) for x in (profile.get("palette_base") or [])[:4])
    preferred = ", ".join(str(x) for x in (profile.get("preferred_colors") or [])[:4])
    materials = ", ".join(str(x) for x in (profile.get("material_family") or [])[:4])
    surface_brief = str(profile.get("surface_design_brief") or "").strip()
    parts = [label]
    if room_type:
        parts.append(room_type.lower())
    if preferred:
        parts.append(f"preferred colors {preferred}")
    if palette:
        parts.append(f"palette {palette}")
    if materials:
        parts.append(f"materials {materials}")
    if surface_brief:
        parts.append(f"surfaces {surface_brief}")
    return "; ".join(parts)


def build_chooser_style_prompt(prompt_text: str, profile: dict[str, Any]) -> str:
    expanded_prompt = str(profile.get("expanded_prompt") or prompt_text or "").strip()
    style_label = str(profile.get("style_label") or "").replace("_", " ")
    palette_base = ", ".join(str(x) for x in (profile.get("palette_base") or [])[:4])
    preferred_colors = ", ".join(str(x) for x in (profile.get("preferred_colors") or [])[:4])
    material_family = ", ".join(str(x) for x in (profile.get("material_family") or [])[:4])
    wall_palette = ", ".join(str(x) for x in (profile.get("wall_palette") or [])[:5])
    floor_palette = ", ".join(str(x) for x in (profile.get("floor_palette") or [])[:5])
    furniture_palette = ", ".join(str(x) for x in (profile.get("furniture_palette") or [])[:5])
    surface_brief = str(profile.get("surface_design_brief") or "").strip()
    return (
        expanded_prompt
        + "\n\n"
        + "STYLE_GUIDANCE:\n"
        + f"- selected style: {style_label}\n"
        + f"- preferred colors: {preferred_colors or palette_base or 'neutral'}\n"
        + f"- palette: {palette_base or 'neutral'}\n"
        + f"- material family: {material_family or 'mixed'}\n"
        + f"- wall palette/material target: {wall_palette or preferred_colors or palette_base or 'neutral'}\n"
        + f"- floor palette/material target: {floor_palette or preferred_colors or palette_base or 'neutral'}\n"
        + f"- furniture palette/finish target: {furniture_palette or preferred_colors or palette_base or 'neutral'}\n"
        + f"- surface design brief: {surface_brief or 'coherent walls, floors, and object finishes'}\n"
        + f"- style hint: {profile.get('style_hint') or build_style_hint(profile)}\n"
    )


def build_supplier_preferences(profile: dict[str, Any]) -> dict[str, Any]:
    preferred = [str(x) for x in (profile.get("preferred_colors") or []) if str(x).strip()]
    palette = [str(x) for x in (profile.get("palette_base") or []) if str(x).strip()]
    furniture_palette = [str(x) for x in (profile.get("furniture_palette") or []) if str(x).strip()]
    avoid = [str(x) for x in (profile.get("palette_avoid") or []) if str(x).strip()]
    return {
        "global": {
            "preferred_colors": (furniture_palette or preferred or palette)[:6],
            "avoid_colors": avoid[:4],
            "style_hint": profile.get("style_hint"),
            "expanded_prompt": profile.get("expanded_prompt"),
            "require_model_url": True,
        }
    }


def compile_style_profile(analysis: dict[str, Any], *, prompt_text: str = "", room_path: str | None = None) -> dict[str, Any]:
    room_type = _normalize_room_type(analysis.get("room_type") or infer_room_type_from_prompt(prompt_text, room_path))
    style_label = str(analysis.get("style_label") or "").strip().lower().replace("-", "_")
    style_label = STYLE_PROFILE_ALIASES.get(style_label, style_label)
    if style_label not in STYLE_PROFILES:
        style_label = default_style_label_for_room(room_type)
    profile = deepcopy(STYLE_PROFILES[style_label])
    compiled = {
        "schema": "room_style_profile/v1",
        "style_label": style_label,
        "room_type": room_type,
        "confidence": float(analysis.get("confidence") or 0.0),
        "description": profile.get("description"),
        "style_axes": deepcopy(profile.get("style_axes") or {}),
        "palette_base": list(profile.get("palette_base") or []),
        "palette_avoid": [str(x) for x in (analysis.get("avoid_colors") or []) if str(x).strip()],
        "material_family": list(analysis.get("material_family") or profile.get("material_family") or []),
        "preferred_colors": [str(x) for x in (analysis.get("preferred_colors") or []) if str(x).strip()],
        "expanded_prompt": str(analysis.get("expanded_prompt") or prompt_text or "").strip(),
        "wall_palette": [str(x) for x in (analysis.get("wall_palette") or []) if str(x).strip()],
        "floor_palette": [str(x) for x in (analysis.get("floor_palette") or []) if str(x).strip()],
        "furniture_palette": [str(x) for x in (analysis.get("furniture_palette") or []) if str(x).strip()],
        "surface_design_brief": str(analysis.get("surface_design_brief") or "").strip(),
        "keywords": list(profile.get("keywords") or []),
        "notes": str(analysis.get("notes") or "").strip(),
    }
    if not compiled["preferred_colors"]:
        compiled["preferred_colors"] = compiled["palette_base"][:4]
    if not compiled["wall_palette"]:
        compiled["wall_palette"] = compiled["preferred_colors"][:4]
    if not compiled["floor_palette"]:
        compiled["floor_palette"] = compiled["preferred_colors"][:4]
    if not compiled["furniture_palette"]:
        compiled["furniture_palette"] = compiled["preferred_colors"][:4]
    compiled["style_hint"] = build_style_hint(compiled)
    compiled["chooser_prompt"] = build_chooser_style_prompt(prompt_text, compiled)
    compiled["supplier_preferences"] = build_supplier_preferences(compiled)
    compiled["infinigen"] = {
        "monkeypatch_params": deepcopy(profile.get("infinigen_params") or {}),
        "overrides": [],
    }
    return compiled


def attach_style_hint_to_room_json(room_data: dict[str, Any], style_profile: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(room_data)
    room = out.get("room")
    if not isinstance(room, dict):
        room = {}
        out["room"] = room
    room["style_hint"] = str(style_profile.get("style_hint") or "")
    room["style_label"] = str(style_profile.get("style_label") or "")
    room["style_profile"] = {
        "style_label": style_profile.get("style_label"),
        "room_type": style_profile.get("room_type"),
        "preferred_colors": style_profile.get("preferred_colors"),
        "wall_palette": style_profile.get("wall_palette"),
        "floor_palette": style_profile.get("floor_palette"),
        "furniture_palette": style_profile.get("furniture_palette"),
        "material_family": style_profile.get("material_family"),
        "surface_design_brief": style_profile.get("surface_design_brief"),
    }
    return out


def build_style_profile_from_compiled_policy(compiled_policy: "CompiledPolicy") -> dict[str, Any]:
    style_label = str(compiled_policy.style_policy.style_label or "minimalism").strip().lower()
    style_profile_key = STYLE_PROFILE_ALIASES.get(style_label, style_label)
    base = deepcopy(STYLE_PROFILES.get(style_profile_key) or STYLE_PROFILES["minimalism"])
    monkeypatch_params = deepcopy(base.get("infinigen_params") or {})
    monkeypatch_params.update(compiled_policy.infinigen_policy.monkeypatch_params)
    overrides = list(compiled_policy.infinigen_policy.gin_overrides)
    preferred_colors = list(
        dict.fromkeys(
            list(compiled_policy.style_policy.preferred_colors)
            + list(compiled_policy.style_policy.palette_hint)
        )
    )
    palette_hint = list(dict.fromkeys(compiled_policy.style_policy.palette_hint or base.get("palette_base") or []))
    material_family = list(
        dict.fromkeys(
            list(compiled_policy.style_policy.material_family)
            + list(base.get("material_family") or [])
        )
    )
    profile = {
        "schema": "room_style_profile/v2",
        "style_label": style_label,
        "room_type": compiled_policy.geometry.room_type.value,
        "description": compiled_policy.style_policy.notes or base.get("description"),
        "style_axes": deepcopy(base.get("style_axes") or {}),
        "palette_base": palette_hint[:6],
        "palette_hint": palette_hint[:6],
        "palette_avoid": list(compiled_policy.style_policy.avoid_colors),
        "material_family": material_family[:6],
        "preferred_colors": preferred_colors[:6] or palette_hint[:4],
        "style_strength": float(compiled_policy.style_policy.style_strength),
        "required_semantics": list(compiled_policy.program.required_semantics),
        "forbidden_semantics": list(compiled_policy.program.forbidden_semantics),
        "factory_whitelist": list(compiled_policy.program.factory_whitelist),
        "factory_blacklist": list(compiled_policy.program.factory_blacklist),
        "max_counts": dict(compiled_policy.program.max_counts),
        "style_hint": (
            f"{style_label.replace('_', ' ')} {compiled_policy.geometry.room_type.value.lower()}; "
            f"palette {', '.join(palette_hint[:4])}; "
            f"materials {', '.join(material_family[:4])}"
        ),
        "chooser_prompt": compiled_policy.prompt_text,
        "supplier_preferences": build_supplier_preferences(
            {
                "preferred_colors": preferred_colors[:6],
                "palette_base": palette_hint[:6],
                "palette_avoid": list(compiled_policy.style_policy.avoid_colors),
            }
        ),
        "infinigen": {
            "monkeypatch_params": monkeypatch_params,
            "overrides": overrides,
            "required_semantics": list(compiled_policy.program.required_semantics),
            "forbidden_semantics": list(compiled_policy.program.forbidden_semantics),
            "factory_whitelist": list(compiled_policy.program.factory_whitelist),
            "factory_blacklist": list(compiled_policy.program.factory_blacklist),
            "effective_factory_whitelist": list(compiled_policy.program.factory_whitelist),
            "effective_factory_blacklist": list(compiled_policy.program.factory_blacklist),
            "apply_child_restrictions": bool(compiled_policy.preflight.get("apply_child_restrictions")),
            "final_restrict_child_primary": list(compiled_policy.preflight.get("final_restrict_child_primary") or []),
            "final_restrict_child_secondary": list(compiled_policy.preflight.get("final_restrict_child_secondary") or []),
            "required_factory_coverage": dict(compiled_policy.preflight.get("required_semantic_factory_coverage") or {}),
            "stage_flags": dict(compiled_policy.preflight.get("stage_flags") or compiled_policy.infinigen_policy.stage_flags),
            "solver_steps": dict(compiled_policy.preflight.get("solver_steps") or compiled_policy.infinigen_policy.solver_steps),
            "max_counts": dict(compiled_policy.program.max_counts),
            "acceptance": compiled_policy.acceptance_policy.model_dump(mode="json"),
            "compiled_policy_path": compiled_policy.artifacts.get("compiled_policy", ""),
        },
    }
    return profile
