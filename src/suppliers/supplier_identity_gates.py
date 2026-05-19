#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import re
from typing import Any


STRICT_GROUPS = {
    "bed",
    "sofa",
    "armchair",
    "chair",
    "desk",
    "dresser",
    "nightstand",
    "wardrobe",
    "shelf",
    "stool",
    "tv_stand",
    "tv",
    "tv_projector_screen",
    "computer",
    "lamp_table",
    "lamp_floor",
    "lamp_ceiling",
    "bathroom_sink",
}


RULES: dict[str, dict[str, list[str]]] = {
    "bed": {
        "required": ["bed", "beds", "bedroom bed", "headboard", "bedframe", "mattress", "кровать", "кровати", "спальная", "изголовье"],
        "forbidden": ["table", "desk", "coffee table", "console", "dresser", "chest", "cabinet", "shelf", "countertop", "bench", "sofa", "стол", "письменный", "журнальный", "консоль", "комод", "шкаф", "стеллаж", "столешница", "скамья", "диван"],
    },
    "sofa": {
        "required": ["sofa", "couch", "диван"],
        "forbidden": ["bed", "chair", "armchair", "table", "кровать", "стул", "кресло", "стол"],
    },
    "desk": {
        "required": ["desk", "writing desk", "computer desk", "письменный стол", "рабочий стол"],
        "forbidden": ["bed", "dining table", "coffee table", "console", "кровать", "обеденный", "журнальный", "консоль"],
    },
    "chair": {
        "required": ["chair", "dining chair", "стул"],
        "forbidden": ["armchair", "lounge chair", "sofa", "stool", "table", "bed", "кресло", "диван", "табурет", "стол", "кровать"],
    },
    "armchair": {
        "required": ["armchair", "lounge chair", "кресло"],
        "forbidden": ["sofa", "table", "bed", "диван", "стол", "кровать"],
    },
    "wardrobe": {
        "required": ["wardrobe", "closet", "шкаф", "гардероб"],
        "forbidden": ["dresser", "chest", "nightstand", "shelf", "table", "bathroom", "sink", "basin", "washbasin", "раковина", "умывальник", "ванная", "комод", "тумба", "стеллаж", "стол"],
    },
    "dresser": {
        "required": ["dresser", "chest", "chest of drawers", "комод", "тумба"],
        "forbidden": ["bed", "table", "desk", "shelf", "wardrobe", "bathroom", "sink", "basin", "washbasin", "раковина", "умывальник", "ванная", "кровать", "стол", "стеллаж", "шкаф"],
    },
    "nightstand": {
        "required": ["nightstand", "bedside table", "bedside cabinet", "bedside", "ночная тумба", "прикроватная", "тумбочка", "тумба"],
        "forbidden": ["bathroom", "sink", "basin", "washbasin", "vanity", "under sink", "shower", "toilet", "раковина", "раковину", "умывальник", "под раковину", "ванная", "санузел", "душ", "унитаз"],
    },
    "stool": {
        "required": ["stool", "ottoman", "pouf", "пуф", "табурет", "банкетка"],
        "forbidden": ["chair", "armchair", "table", "bed", "desk", "стул", "кресло", "стол", "кровать"],
    },
    "shelf": {
        "required": ["shelf", "shelves", "bookcase", "bookshelf", "стеллаж", "полка", "книжный шкаф"],
        "forbidden": ["bed", "table", "desk", "wardrobe", "sofa", "кровать", "стол", "шкаф", "диван"],
    },
    "lamp_floor": {
        "required": ["floor lamp", "lamp", "торшер", "светильник"],
        "forbidden": ["chandelier", "table lamp", "люстра", "настольная"],
    },
    "lamp_ceiling": {
        "required": ["chandelier", "pendant", "ceiling light", "люстра", "потолочный", "подвесной светильник"],
        "forbidden": ["floor lamp", "table lamp", "торшер", "настольная"],
    },
    "lamp_table": {
        "required": ["table lamp", "desk lamp", "lamp", "настольная лампа", "настольный светильник"],
        "forbidden": ["floor lamp", "chandelier", "торшер", "люстра"],
    },
    "bathroom_sink": {
        "required": ["sink", "basin", "washbasin", "раковина", "раковины", "умывальник"],
        "forbidden": ["sofa", "coffee table", "bookcase", "bookshelf", "plant", "planter", "vase", "decorative", "диван", "журнальный стол", "стеллаж", "кашпо", "ваза", "растение", "декоративный"],
    },
    "tv": {
        "required": ["tv", "television", "smart tv", "wall tv", "телевизор"],
        "forbidden": ["monitor", "computer monitor", "display", "laptop", "notebook", "imac", "keyboard", "монитор", "дисплей", "ноутбук", "клавиатура"],
    },
    "computer": {
        "required": ["monitor", "computer", "pc", "imac", "laptop", "notebook", "монитор", "компьютер", "ноутбук"],
        "forbidden": ["tv", "television", "телевизор"],
    },
}


GROUP_ALIASES = {
    "tv_projector_screen": "tv",
    "tv_stand": "tv",
    "computer_monitor": "computer",
    "laptop_computer_keyboard_mouse": "computer",
    "floor_lamp": "lamp_floor",
    "table_lamp": "lamp_table",
    "ceiling_lamp": "lamp_ceiling",
    "bath_sink": "bathroom_sink",
    "sink": "bathroom_sink",
    "washbasin": "bathroom_sink",
    "side_table": "nightstand",
    "bedside_table": "nightstand",
    "bedside_cabinet": "nightstand",
}


def normalize_text(value: Any) -> str:
    text = str(value or "").replace("ё", "е").replace("Ё", "Е").lower()
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"[_/\\\\-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zа-я0-9]+", normalize_text(text))


def _haystack(row: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in (
        "title",
        "name",
        "category_raw",
        "category_norm",
        "semantic_group",
        "description",
        "vlm_description_text",
        "vlm_description_summary",
        "vlm_style",
        "vlm_materials",
        "product_url",
        "unique_key",
    ):
        value = row.get(key)
        if value is not None:
            parts.append(str(value))
    return normalize_text(" ".join(parts))


def _hit_phrases(haystack: str, phrases: list[str]) -> list[str]:
    hits: list[str] = []
    padded = f" {haystack} "
    for phrase in phrases:
        p = normalize_text(phrase)
        if not p:
            continue
        if " " in p:
            if p in haystack:
                hits.append(phrase)
        elif re.search(rf"(?<![a-zа-я0-9]){re.escape(p)}(?![a-zа-я0-9])", padded):
            hits.append(phrase)
    return hits


def _target_group(target: dict[str, Any]) -> str:
    raw = normalize_text(target.get("semantic_group") or target.get("category_norm") or target.get("category") or target.get("name"))
    for alias, canonical in GROUP_ALIASES.items():
        if raw == alias:
            return canonical
    if "bed" in raw or "кровать" in raw:
        return "bed"
    if "sofa" in raw or "диван" in raw:
        return "sofa"
    if "wardrobe" in raw or "closet" in raw or "шкаф" in raw:
        return "wardrobe"
    if "dresser" in raw or "комод" in raw:
        return "dresser"
    if "nightstand" in raw or "bedside" in raw or "прикроват" in raw or "тумбоч" in raw:
        return "nightstand"
    if "shelf" in raw or "bookcase" in raw or "стеллаж" in raw:
        return "shelf"
    if "desk" in raw:
        return "desk"
    if "armchair" in raw or "кресло" in raw:
        return "armchair"
    if "chair" in raw or "стул" in raw:
        return "chair"
    if "stool" in raw or "ottoman" in raw or "pouf" in raw or "табурет" in raw or "пуф" in raw:
        return "stool"
    if "floor" in raw and "lamp" in raw:
        return "lamp_floor"
    if "ceiling" in raw and "lamp" in raw:
        return "lamp_ceiling"
    if "lamp" in raw:
        return "lamp_table"
    if "sink" in raw or "basin" in raw or "washbasin" in raw or "раковин" in raw or "умывальник" in raw:
        return "bathroom_sink"
    if "tv" in raw or "television" in raw or "телевиз" in raw:
        return "tv"
    if "computer" in raw or "monitor" in raw:
        return "computer"
    return raw


def candidate_identity_gate(target: dict[str, Any], row: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    group = _target_group(target)
    canonical = GROUP_ALIASES.get(group, group)
    strict = canonical in STRICT_GROUPS
    haystack = _haystack(row)
    token_list = sorted(set(_tokens(haystack)))[:80]
    rule = RULES.get(canonical)
    if not rule:
        return True, {
            "identity_gate_checked": False,
            "identity_gate_passed": True,
            "identity_target_group": canonical,
            "identity_required_hits": [],
            "identity_forbidden_hits": [],
            "identity_reject_reason": None,
            "identity_candidate_tokens": token_list,
        }

    required_hits = _hit_phrases(haystack, rule["required"])
    forbidden_hits = _hit_phrases(haystack, rule["forbidden"])
    passed = True
    reject_reason = None
    if strict and forbidden_hits:
        passed = False
        reject_reason = "identity_forbidden_terms:" + ",".join(forbidden_hits[:5])
    elif strict and not required_hits:
        passed = False
        reject_reason = "identity_required_terms_missing"

    return passed, {
        "identity_gate_checked": True,
        "identity_gate_passed": bool(passed),
        "identity_target_group": canonical,
        "identity_required_hits": required_hits,
        "identity_forbidden_hits": forbidden_hits,
        "identity_reject_reason": reject_reason,
        "identity_candidate_tokens": token_list,
    }
