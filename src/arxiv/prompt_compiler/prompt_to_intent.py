from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .inventory_mapping import normalize_prompt_object
from .llm_client import BaseLLMClient
from .schemas import (
    DecorRichness,
    DensityLevel,
    ObjectsIntent,
    OpeningsIntent,
    PreferencesIntent,
    PromptIntent,
    RoomType,
    StyleIntent,
    StyleLabel,
)


STYLE_ALIASES = {
    "minimalist japanese": StyleLabel.JAPANDI,
    "minimal japanese": StyleLabel.JAPANDI,
    "japanese minimal": StyleLabel.JAPANDI,
    "japanese minimalist": StyleLabel.JAPANDI,
    "japanese": StyleLabel.JAPANDI,
    "japandi": StyleLabel.JAPANDI,
    "scandinavian": StyleLabel.SCANDINAVIAN,
    "nordic": StyleLabel.SCANDINAVIAN,
    "contemporary": StyleLabel.CONTEMPORARY,
    "modern contemporary": StyleLabel.CONTEMPORARY,
    "baroque": StyleLabel.BAROQUE_INSPIRED,
    "baroque inspired": StyleLabel.BAROQUE_INSPIRED,
}

ROOM_ALIASES = {
    "bedroom": RoomType.BEDROOM,
    "living room": RoomType.LIVING_ROOM,
    "livingroom": RoomType.LIVING_ROOM,
    "kitchen": RoomType.KITCHEN,
    "bathroom": RoomType.BATHROOM,
    "dining room": RoomType.DINING_ROOM,
    "diningroom": RoomType.DINING_ROOM,
}


def _intent_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "room_type": {"type": "string", "enum": [x.value for x in RoomType]},
            "style_label": {"type": "string", "enum": [x.value for x in StyleLabel]},
            "style_raw": {"type": "string"},
            "target_area_sqm": {"type": "number"},
            "width_m": {"type": "number"},
            "depth_m": {"type": "number"},
            "height_m": {"type": "number"},
            "density": {"type": "string", "enum": [x.value for x in DensityLevel]},
            "decor_richness": {"type": "string", "enum": [x.value for x in DecorRichness]},
            "required_objects": {"type": "array", "items": {"type": "string"}},
            "desired_objects": {"type": "array", "items": {"type": "string"}},
            "forbidden_objects": {"type": "array", "items": {"type": "string"}},
            "favorite_colors": {"type": "array", "items": {"type": "string"}},
            "avoid_colors": {"type": "array", "items": {"type": "string"}},
            "material_family": {"type": "array", "items": {"type": "string"}},
            "palette_hint": {"type": "array", "items": {"type": "string"}},
            "wants_door": {"type": "boolean"},
            "wants_window": {"type": "boolean"},
            "notes": {"type": "string"},
        },
        "required": ["room_type", "style_label"],
        "additionalProperties": False,
    }


def _system_prompt() -> str:
    return (
        "Extract only normalized room intent from the user prompt. "
        "Do not invent solver parameters or gin overrides. "
        "Return conservative structured room intent suitable for deterministic compilation."
    )


def _parse_area_from_text(prompt: str) -> float | None:
    low = prompt.lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:sqm|m2|m\^2|square meters?|square metres?)", low)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*[xх]\s*(\d+(?:\.\d+)?)", low)
    if m:
        return round(float(m.group(1)) * float(m.group(2)), 3)
    return None


def _parse_dimensions_from_text(prompt: str) -> tuple[float | None, float | None]:
    low = prompt.lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*[xх]\s*(\d+(?:\.\d+)?)", low)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None, None


def _heuristic_extract(prompt: str) -> dict[str, Any]:
    low = prompt.lower()
    room_type = RoomType.BEDROOM.value
    for alias, value in ROOM_ALIASES.items():
        if alias in low:
            room_type = value.value
            break
    style_label = StyleLabel.JAPANDI.value
    for alias, value in STYLE_ALIASES.items():
        if alias in low:
            style_label = value.value
            break
    width_m, depth_m = _parse_dimensions_from_text(prompt)
    target_area = _parse_area_from_text(prompt)
    required: list[str] = []
    desired: list[str] = []
    for raw in [
        "bed",
        "nightstand",
        "wardrobe",
        "cabinet",
        "dresser",
        "lamp",
        "floor lamp",
        "ceiling light",
        "chair",
        "desk",
        "rug",
        "mirror",
        "plant",
    ]:
        if raw in low:
            normalized = normalize_prompt_object(raw)
            if normalized and normalized not in required:
                required.append(normalized)
    if "minimal" in low or "calm" in low:
        density = DensityLevel.LOW.value
        decor = DecorRichness.VERY_LOW.value
    elif "ornate" in low or "rich" in low or "decorative" in low:
        density = DensityLevel.MEDIUM.value
        decor = DecorRichness.MEDIUM.value
    else:
        density = DensityLevel.MEDIUM.value
        decor = DecorRichness.LOW.value
    colors = [token for token in ["white", "beige", "gray", "grey", "black", "brown", "gold", "green", "blue"] if token in low]
    materials = [token for token in ["wood", "metal", "marble", "velvet", "fabric", "ceramic", "paper"] if token in low]
    return {
        "room_type": room_type,
        "style_label": style_label,
        "style_raw": style_label.replace("_", " "),
        "target_area_sqm": target_area,
        "width_m": width_m,
        "depth_m": depth_m,
        "density": density,
        "decor_richness": decor,
        "required_objects": required,
        "desired_objects": desired,
        "forbidden_objects": [],
        "favorite_colors": colors,
        "avoid_colors": [],
        "material_family": materials,
        "palette_hint": colors,
        "wants_door": "no door" not in low,
        "wants_window": "no window" not in low,
        "notes": "",
    }


def _normalize_style_label(value: str | None) -> StyleLabel | None:
    if not value:
        return None
    low = str(value).strip().lower().replace("-", " ")
    low = " ".join(low.split())
    if low in STYLE_ALIASES:
        return STYLE_ALIASES[low]
    compact = low.replace(" ", "_")
    for style in StyleLabel:
        if compact == style.value:
            return style
    return None


def _normalize_room_type(value: str | None) -> RoomType:
    low = str(value or "").strip().lower()
    for alias, room_type in ROOM_ALIASES.items():
        if alias == low:
            return room_type
    compact = low.replace("-", "").replace(" ", "")
    for room_type in RoomType:
        if room_type.value.lower() == compact:
            return room_type
    return RoomType.BEDROOM


def _normalize_objects(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        normalized = normalize_prompt_object(item) or item
        if normalized not in out:
            out.append(normalized)
    return out


def extract_intent(prompt: str, llm_client: BaseLLMClient) -> PromptIntent:
    raw_payload: dict[str, Any]
    heuristic_payload = _heuristic_extract(prompt)
    try:
        raw_payload = llm_client.complete_json(_system_prompt(), prompt, _intent_schema())
    except Exception:
        raw_payload = dict(heuristic_payload)
    if not raw_payload or not raw_payload.get("room_type") or not raw_payload.get("style_label"):
        merged = dict(heuristic_payload)
        merged.update(raw_payload or {})
        raw_payload = merged
    raw_payload.setdefault("prompt_text", prompt)
    raw_payload.setdefault("required_objects", [])
    raw_payload.setdefault("desired_objects", [])
    raw_payload.setdefault("forbidden_objects", [])
    raw_payload.setdefault("favorite_colors", [])
    raw_payload.setdefault("avoid_colors", [])
    raw_payload.setdefault("material_family", [])
    raw_payload.setdefault("palette_hint", [])
    raw_payload.setdefault("wants_door", True)
    raw_payload.setdefault("wants_window", True)
    raw_payload.setdefault("notes", "")
    intent = PromptIntent(
        prompt_text=prompt,
        room_type=_normalize_room_type(raw_payload.get("room_type")),
        geometry={
            "target_area_sqm": raw_payload.get("target_area_sqm"),
            "width_m": raw_payload.get("width_m"),
            "depth_m": raw_payload.get("depth_m"),
            "height_m": raw_payload.get("height_m") or 2.7,
        },
        style={
            "style_label": _normalize_style_label(raw_payload.get("style_label")),
            "style_raw": raw_payload.get("style_raw") or raw_payload.get("style_label"),
            "density": raw_payload.get("density"),
            "decor_richness": raw_payload.get("decor_richness"),
            "palette_hint": raw_payload.get("palette_hint") or [],
            "material_family": raw_payload.get("material_family") or [],
        },
        openings={
            "wants_door": bool(raw_payload.get("wants_door", True)),
            "wants_window": bool(raw_payload.get("wants_window", True)),
        },
        objects={
            "required": raw_payload.get("required_objects") or [],
            "desired": raw_payload.get("desired_objects") or [],
            "forbidden": raw_payload.get("forbidden_objects") or [],
        },
        preferences={
            "favorite_colors": raw_payload.get("favorite_colors") or [],
            "avoid_colors": raw_payload.get("avoid_colors") or [],
            "notes": str(raw_payload.get("notes") or ""),
        },
    )
    return normalize_intent(intent)


def normalize_intent(intent: PromptIntent) -> PromptIntent:
    width_m = intent.geometry.width_m
    depth_m = intent.geometry.depth_m
    area = intent.geometry.target_area_sqm
    if width_m and depth_m and not area:
        area = round(width_m * depth_m, 3)
    if area and not width_m and not depth_m:
        width_m, depth_m = _parse_dimensions_from_text(intent.prompt_text)
    style_label = intent.style.style_label or _normalize_style_label(intent.style.style_raw)
    if style_label is None:
        style_label = StyleLabel.JAPANDI if "japan" in intent.prompt_text.lower() else StyleLabel.SCANDINAVIAN
    return PromptIntent(
        prompt_text=intent.prompt_text,
        room_type=_normalize_room_type(intent.room_type.value if isinstance(intent.room_type, RoomType) else str(intent.room_type)),
        geometry={
            "target_area_sqm": area,
            "width_m": width_m,
            "depth_m": depth_m,
            "height_m": intent.geometry.height_m or 2.7,
        },
        style=StyleIntent(
            style_label=style_label,
            style_raw=intent.style.style_raw or style_label.value,
            density=intent.style.density or DensityLevel.LOW,
            decor_richness=intent.style.decor_richness or DecorRichness.LOW,
            palette_hint=list(dict.fromkeys(intent.style.palette_hint + intent.preferences.favorite_colors)),
            material_family=list(dict.fromkeys(intent.style.material_family)),
        ),
        openings=OpeningsIntent.model_validate(intent.openings.model_dump(mode="json")),
        objects=ObjectsIntent(
            required=_normalize_objects(intent.objects.required),
            desired=_normalize_objects(intent.objects.desired),
            forbidden=_normalize_objects(intent.objects.forbidden),
        ),
        preferences=PreferencesIntent(
            favorite_colors=list(dict.fromkeys(intent.preferences.favorite_colors)),
            avoid_colors=list(dict.fromkeys(intent.preferences.avoid_colors)),
            notes=intent.preferences.notes,
        ),
    )


def save_intent_trace(intent: PromptIntent, out_dir: str | Path) -> None:
    out_path = Path(out_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "prompt.txt").write_text(intent.prompt_text, encoding="utf-8")
    (out_path / "intent.raw.json").write_text(
        json.dumps(intent.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    normalized = normalize_intent(intent)
    (out_path / "intent.normalized.json").write_text(
        normalized.model_dump_json_pretty(),
        encoding="utf-8",
    )
