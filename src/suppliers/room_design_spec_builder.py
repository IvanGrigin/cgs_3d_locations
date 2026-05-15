#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


STYLE_DEFAULTS: dict[str, dict[str, Any]] = {
    "soft_classic": {
        "secondary": ["contemporary", "soft traditional", "residential classic"],
        "epoch": "contemporary",
        "colors": ["cream", "beige", "warm white", "pale green", "light olive green", "light natural wood"],
        "accent": ["pale yellow green", "soft sage", "white molding"],
        "materials": ["soft fabric", "linen", "cotton", "matte painted wood", "light natural wood", "textured plaster"],
        "forbidden": ["black", "red", "purple", "dark loft", "industrial", "brutalist", "high tech", "baroque", "rococo", "glossy black"],
    },
    "scandinavian": {
        "secondary": ["minimalist", "cozy modern"],
        "epoch": "contemporary",
        "colors": ["warm white", "light oak", "beige", "soft gray"],
        "accent": ["black metal", "muted terracotta"],
        "materials": ["light wood", "linen", "cotton", "matte painted wood", "soft fabric"],
        "forbidden": ["baroque", "rococo", "heavy classical", "glossy black", "neon colors"],
    },
    "minimalism": {
        "secondary": ["modern", "soft minimalism"],
        "epoch": "contemporary",
        "colors": ["white", "warm gray", "beige", "black"],
        "accent": ["matte black", "natural wood"],
        "materials": ["matte painted surfaces", "wood", "fabric", "metal accents"],
        "forbidden": ["baroque", "ornate", "busy patterns", "neon colors"],
    },
    "loft_industrial": {
        "secondary": ["industrial", "modern"],
        "epoch": "contemporary",
        "colors": ["concrete gray", "dark wood", "black metal", "brown"],
        "accent": ["rust", "cognac leather"],
        "materials": ["metal", "wood", "leather", "concrete", "brick"],
        "forbidden": ["rococo", "pastel romantic", "glossy white classical"],
    },
    "japandi": {
        "secondary": ["minimalist", "scandinavian"],
        "epoch": "contemporary",
        "colors": ["warm white", "natural wood", "beige", "taupe", "soft gray"],
        "accent": ["black", "sage green"],
        "materials": ["light wood", "linen", "rattan", "matte ceramic", "cotton"],
        "forbidden": ["baroque", "neon colors", "high gloss plastic", "heavy ornament"],
    },
    "modern": {
        "secondary": ["contemporary", "minimalist"],
        "epoch": "contemporary",
        "colors": ["white", "gray", "beige", "black"],
        "accent": ["wood", "metal"],
        "materials": ["wood", "fabric", "metal", "glass", "matte painted surfaces"],
        "forbidden": ["baroque", "rococo", "victorian", "heavy ornament"],
    },
}


GROUP_DEFAULTS: dict[str, dict[str, Any]] = {
    "bed": {
        "role": "main visual anchor",
        "shape": ["low-profile", "rectangular", "soft edges"],
        "materials": ["fabric", "wood"],
        "visual_priority": "high",
        "price_priority": "medium",
    },
    "sofa": {
        "role": "main seating anchor",
        "shape": ["comfortable", "simple", "soft rectangular"],
        "materials": ["fabric", "leather", "wood"],
        "visual_priority": "high",
        "price_priority": "medium",
    },
    "chair": {
        "role": "supporting seating",
        "shape": ["simple", "lightweight"],
        "materials": ["wood", "fabric", "metal"],
        "visual_priority": "medium",
        "price_priority": "medium",
    },
    "armchair": {
        "role": "accent seating",
        "shape": ["comfortable", "soft", "proportional"],
        "materials": ["fabric", "leather", "wood"],
        "visual_priority": "high",
        "price_priority": "medium",
    },
    "nightstand": {
        "role": "paired storage near bed",
        "shape": ["compact", "simple", "rectangular"],
        "materials": ["wood", "matte finish"],
        "visual_priority": "medium",
        "price_priority": "medium",
    },
    "desk": {
        "role": "work surface",
        "shape": ["clean rectangular", "functional"],
        "materials": ["wood", "metal", "matte finish"],
        "visual_priority": "medium",
        "price_priority": "medium",
    },
    "dresser": {
        "role": "storage furniture",
        "shape": ["rectangular", "clean front"],
        "materials": ["wood", "matte painted wood"],
        "visual_priority": "medium",
        "price_priority": "medium",
    },
    "shelf": {
        "role": "open storage",
        "shape": ["linear", "simple"],
        "materials": ["wood", "metal"],
        "visual_priority": "medium",
        "price_priority": "medium",
    },
    "wardrobe": {
        "role": "large storage",
        "shape": ["tall", "clean front"],
        "materials": ["wood", "matte painted surfaces"],
        "visual_priority": "high",
        "price_priority": "medium",
    },
    "lamp_floor": {
        "role": "warm atmosphere lighting",
        "shape": ["slender", "soft shade", "minimal"],
        "materials": ["fabric shade", "metal", "wood"],
        "visual_priority": "medium",
        "price_priority": "low",
    },
    "lamp_table": {
        "role": "local warm lighting",
        "shape": ["compact", "soft shade", "rounded"],
        "materials": ["fabric shade", "ceramic", "metal"],
        "visual_priority": "medium",
        "price_priority": "low",
    },
    "lamp_ceiling": {
        "role": "main ambient lighting",
        "shape": ["simple", "balanced", "not oversized"],
        "materials": ["metal", "glass", "fabric shade"],
        "visual_priority": "medium",
        "price_priority": "low",
    },
}


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _tokens(text: Any) -> set[str]:
    return set(re.findall(r"[A-Za-zА-Яа-я0-9_\-]+", str(text or "").lower().replace("ё", "е")))


def _normalize_style(raw: Any) -> str:
    text = str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "scandi": "scandinavian",
        "скандинавский": "scandinavian",
        "сканди": "scandinavian",
        "minimalist": "minimalism",
        "minimal": "minimalism",
        "loft": "loft_industrial",
        "industrial": "loft_industrial",
        "contemporary": "modern",
    }
    return aliases.get(text, text or "modern")


def _infer_style(prompt: str, room: dict[str, Any], style_profile: dict[str, Any] | None) -> str:
    prompt_l = str(prompt or "").lower().replace("ё", "е")
    if any(token in prompt_l for token in ("slightly classic", "soft classic", "classic residential", "classic, residential", "слегка класс")):
        return "soft_classic"
    for value in [
        (style_profile or {}).get("style_label"),
        room.get("style_label"),
        room.get("style_hint"),
        prompt,
    ]:
        text = " ".join(sorted(_tokens(value)))
        for style in STYLE_DEFAULTS:
            if style in text or style.replace("_", " ") in str(value or "").lower():
                return style
        normalized = _normalize_style(value)
        if normalized in STYLE_DEFAULTS:
            return normalized
    return "modern"


def _infer_room_type(prompt: str, room: dict[str, Any], style_profile: dict[str, Any] | None) -> str:
    for value in [(style_profile or {}).get("room_type"), room.get("room_type"), room.get("id"), prompt]:
        text = str(value or "").strip().lower()
        if "bed" in text or "спаль" in text:
            return "bedroom"
        if "living" in text or "гост" in text:
            return "living_room"
        if "kitchen" in text or "кух" in text:
            return "kitchen"
        if "bath" in text or "ванн" in text:
            return "bathroom"
        if "dining" in text or "столов" in text:
            return "dining_room"
    return "room"


def _split_prompt_palette_segments(prompt: str) -> tuple[str, str]:
    positive_parts: list[str] = []
    negative_parts: list[str] = []
    marker = re.compile(r"\b(?:avoid|forbidden|do not use|don't use)\b", re.IGNORECASE)
    for line in str(prompt or "").splitlines():
        match = marker.search(line)
        if not match:
            positive_parts.append(line)
            continue
        before = line[: match.start()]
        after = line[match.end() :]
        if before.strip():
            positive_parts.append(before)
        negative_parts.append(after or line)
    return " ".join(positive_parts), " ".join(negative_parts)


def _palette_from_inputs(prompt: str, room: dict[str, Any], style_profile: dict[str, Any], style_defaults: dict[str, Any]) -> dict[str, list[str]]:
    preferred = list(style_profile.get("preferred_colors") or [])
    material_hint = list(style_profile.get("material_family") or [])
    raw_prompt = str(prompt or "")
    positive_prompt, negative_prompt = _split_prompt_palette_segments(raw_prompt)
    hint_text = " ".join([positive_prompt, str(room.get("style_hint") or ""), " ".join(preferred), " ".join(material_hint)])
    toks = _tokens(hint_text)
    negative_toks = _tokens(negative_prompt)
    detected: list[str] = []
    aliases = {
        "white": {"white", "warm_white", "warm", "белый", "белая", "ivory"},
        "white_warm": {"cream", "ivory", "milk", "молочный"},
        "black": {"black", "черный", "черная", "dark", "темный"},
        "gray": {"gray", "grey", "серый", "silver"},
        "beige": {"beige", "sand", "cream", "бежевый"},
        "wood_light": {"wood", "oak", "natural", "light", "wooden", "дерево", "дуб"},
        "brown": {"brown", "walnut", "коричневый"},
        "blue": {"blue", "navy", "синий", "голубой"},
        "green": {"green", "sage", "olive", "зеленый", "pale", "yellow", "зелёный"},
        "olive_green": {"olive", "sage"},
        "red": {"red", "terracotta", "burgundy", "красный"},
        "purple": {"purple", "violet", "фиолетовый"},
    }
    for color, variants in aliases.items():
        if toks & variants and not (negative_toks & variants) and color not in detected:
            detected.append(color)
    base = detected or list(style_defaults.get("colors") or ["white", "beige", "gray"])
    forbidden_colors = []
    for color, variants in aliases.items():
        if negative_toks & variants and color not in forbidden_colors:
            forbidden_colors.append(color)
    return {
        "primary": base[:3],
        "secondary": (base[3:] + list(style_defaults.get("colors") or []))[:4],
        "accent": list(style_defaults.get("accent") or [])[:3],
        "preferred_colors": base[:6],
        "forbidden": ["neon", "bright saturated colors", *forbidden_colors, *list(style_defaults.get("forbidden") or [])[:5]],
        "forbidden_colors": forbidden_colors,
    }


def _target_summary(target: dict[str, Any]) -> dict[str, Any]:
    size = target.get("size_m") or [0.0, 0.0, 0.0]
    try:
        size = [round(float(x), 4) for x in size[:3]]
    except Exception:
        size = [0.0, 0.0, 0.0]
    return {
        "target_id": target.get("target_id"),
        "name": target.get("name"),
        "category": target.get("category"),
        "semantic_group": target.get("semantic_group"),
        "size_m": size,
        "replacement_policy": target.get("replacement_policy"),
    }


def build_room_design_spec(
    *,
    user_prompt: str,
    layout_targets: dict[str, Any],
    style_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    room = layout_targets.get("room") if isinstance(layout_targets.get("room"), dict) else {}
    targets = [x for x in (layout_targets.get("targets") or []) if isinstance(x, dict)]
    style_profile = dict(style_profile or room.get("style_profile") or {})
    primary_style = _infer_style(user_prompt, room, style_profile)
    style_defaults = STYLE_DEFAULTS.get(primary_style, STYLE_DEFAULTS["modern"])
    room_type = _infer_room_type(user_prompt, room, style_profile)
    palette = _palette_from_inputs(user_prompt, room, style_profile, style_defaults)
    material_preferred = list(style_profile.get("material_family") or []) or list(style_defaults.get("materials") or [])

    groups = Counter(str(t.get("semantic_group") or t.get("category") or "object").strip() for t in targets)
    object_requirements: dict[str, Any] = {}
    for group in sorted(groups):
        if not group or group in object_requirements:
            continue
        defaults = GROUP_DEFAULTS.get(group, {})
        object_requirements[group] = {
            "role": defaults.get("role") or "scene object",
            "style": [primary_style, *list(style_defaults.get("secondary") or [])[:2]],
            "colors": [*palette.get("primary", [])[:3], *palette.get("secondary", [])[:2]],
            "materials": list(defaults.get("materials") or material_preferred[:3]),
            "shape": list(defaults.get("shape") or ["simple", "proportional"]),
            "avoid": list(style_defaults.get("forbidden") or [])[:4],
            "visual_priority": defaults.get("visual_priority") or "medium",
            "price_priority": defaults.get("price_priority") or "medium",
            "target_count": groups[group],
        }

    expanded = (
        f"The room is a {room_type.replace('_', ' ')} in a {primary_style.replace('_', ' ')} direction. "
        f"The design should feel coherent, functional and visually calm. "
        f"Use a palette led by {', '.join(palette.get('primary') or [])}, with secondary tones "
        f"{', '.join(palette.get('secondary') or [])}. Preferred materials are "
        f"{', '.join(material_preferred[:6])}. Avoid {', '.join(style_defaults.get('forbidden') or [])}. "
        "Supplier objects should match the generated layout dimensions while following the room-level style, "
        "color and material intent."
    )

    return {
        "schema": "room_design_spec/v1",
        "source_prompt": user_prompt,
        "expanded_room_description": expanded,
        "room_type": room_type,
        "room": {
            "id": room.get("id"),
            "width_m": room.get("width_m"),
            "depth_m": room.get("depth_m"),
            "area_m2": room.get("area_m2"),
            "style_hint": room.get("style_hint"),
        },
        "design_intent": {
            "function": room_type.replace("_", " "),
            "mood": ["coherent", "comfortable", "intentional"],
            "target_user_impression": "the supplier assets should look selected for one interior concept",
        },
        "style": {
            "primary": primary_style,
            "secondary": list(style_defaults.get("secondary") or []),
            "allowed": [primary_style, *list(style_defaults.get("secondary") or []), "compatible modern"],
            "forbidden": list(style_defaults.get("forbidden") or []),
        },
        "epoch": {
            "primary": style_defaults.get("epoch") or "contemporary",
            "allowed": [style_defaults.get("epoch") or "contemporary", "modern", "contemporary"],
            "forbidden": ["victorian", "baroque", "rococo"],
        },
        "color_palette": palette,
        "materials": {
            "preferred": material_preferred,
            "allowed": [*material_preferred, "metal accents", "glass", "ceramic"],
            "forbidden": ["high gloss plastic", "heavy ornament", "neon colored surfaces"],
        },
        "object_requirements": object_requirements,
        "placement_objects": [_target_summary(t) for t in targets],
        "consistency_rules": [
            "Repeated objects of the same semantic group and similar dimensions should usually use the same supplier model.",
            "Large furniture should follow the primary palette.",
            "Accent colors should be limited to small objects.",
            "Avoid mixing more than two strong style families in one room.",
        ],
        "generation": {
            "mode": "deterministic_fallback",
            "llm_used": False,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a structured room design specification for supplier matching.")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--prompt-file", default=None)
    ap.add_argument("--layout-targets", required=True)
    ap.add_argument("--style-profile", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    prompt = str(args.prompt or "")
    if args.prompt_file:
        prompt = Path(args.prompt_file).expanduser().read_text(encoding="utf-8").strip()
    layout_targets = read_json(args.layout_targets)
    style_profile = read_json(args.style_profile) if args.style_profile else None
    spec = build_room_design_spec(user_prompt=prompt, layout_targets=layout_targets, style_profile=style_profile)
    write_json(args.out, spec)
    print(f"saved = {Path(args.out).expanduser().resolve()}")
    print(json.dumps({"room_type": spec.get("room_type"), "style": spec.get("style", {}).get("primary"), "object_groups": len(spec.get("object_requirements") or {})}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
